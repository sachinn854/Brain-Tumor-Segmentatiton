"""
Training-time data augmentation for BraTS.

Implements the exact pipeline WAS-Mamba's BraTS setup defers to (via
UNETR++, which in turn defers to nnFormer -- see src/configs/wasmamba_config.py
for the full deferral chain): "rotation, scaling, gaussian noise, gaussian
blur, brightness and contrast adjust, simulation of low resolution, gamma
augmentation and mirroring... applied in the given order" (nnFormer,
arXiv:2109.03201v6, read 2026-09-09).

Per-transform PROBABILITY of being applied is not stated in what was
extracted from that paper -- the values below are nnU-Net's well-documented
public defaults (nnFormer builds directly on nnU-Net's training pipeline),
not independently confirmed against nnFormer's own code. Flagging this the
same way the batch_size/scheduler numbers were flagged: reasonable, standard,
but not a verified exact match.

Only applied to the 'train' split -- BratsDataset does not call this for
val/test, which use a fixed center-crop instead (see brats_dataset.py).
"""

import numpy as np
from scipy.ndimage import rotate, zoom, gaussian_filter


class BraTSAugmentor:
    """
    Args:
        p_rotation, p_scaling, ...: per-transform probability of applying
            that augmentation to a given sample. Defaults are nnU-Net's
            commonly-documented values.
        rotation_range_deg: max rotation angle per axis, degrees.
        scaling_range: (min_scale, max_scale) multiplicative zoom factor.
    """

    def __init__(
        self,
        p_rotation=0.2,
        p_scaling=0.2,
        p_gaussian_noise=0.15,
        p_gaussian_blur=0.2,
        p_brightness=0.15,
        p_contrast=0.15,
        p_low_res=0.125,
        p_gamma=0.15,
        p_mirror=0.5,
        rotation_range_deg=15.0,
        scaling_range=(0.85, 1.25),
        noise_std_range=(0.0, 0.1),
        blur_sigma_range=(0.5, 1.5),
        brightness_range=(0.7, 1.3),
        contrast_range=(0.65, 1.5),
        low_res_zoom_range=(0.5, 1.0),
        gamma_range=(0.7, 1.5),
    ):
        self.p_rotation = p_rotation
        self.p_scaling = p_scaling
        self.p_gaussian_noise = p_gaussian_noise
        self.p_gaussian_blur = p_gaussian_blur
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_low_res = p_low_res
        self.p_gamma = p_gamma
        self.p_mirror = p_mirror

        self.rotation_range_deg = rotation_range_deg
        self.scaling_range = scaling_range
        self.noise_std_range = noise_std_range
        self.blur_sigma_range = blur_sigma_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.low_res_zoom_range = low_res_zoom_range
        self.gamma_range = gamma_range

    def __call__(self, image: np.ndarray, label: np.ndarray):
        """
        image: (C, H, W, D) float32
        label: (H, W, D) int64
        Returns augmented (image, label), same shapes.
        """
        image, label = self._rotation(image, label)
        image, label = self._scaling(image, label)
        image = self._gaussian_noise(image)
        image = self._gaussian_blur(image)
        image = self._brightness(image)
        image = self._contrast(image)
        image = self._low_res_simulation(image)
        image = self._gamma(image)
        image, label = self._mirror(image, label)
        return image, label

    # --- order matches the paper's stated sequence ---

    def _rotation(self, image, label):
        if np.random.rand() >= self.p_rotation:
            return image, label
        # np.random.choice can't pick from a list of tuples directly (it
        # tries to build a 2D array and rejects non-1D input) -- index in.
        axis_pairs = [(1, 2), (1, 3), (2, 3)]
        axis_pair = axis_pairs[np.random.randint(len(axis_pairs))]
        angle = np.random.uniform(-self.rotation_range_deg, self.rotation_range_deg)
        C = image.shape[0]
        rotated_image = np.stack([
            rotate(image[c], angle, axes=(axis_pair[0] - 1, axis_pair[1] - 1), reshape=False, order=1, mode='nearest')
            for c in range(C)
        ], axis=0)
        rotated_label = rotate(label, angle, axes=(axis_pair[0] - 1, axis_pair[1] - 1), reshape=False, order=0, mode='nearest')
        return rotated_image.astype(image.dtype), rotated_label.astype(label.dtype)

    def _scaling(self, image, label):
        if np.random.rand() >= self.p_scaling:
            return image, label
        factor = np.random.uniform(*self.scaling_range)
        C = image.shape[0]
        orig_shape = image.shape[1:]

        zoomed_image = np.stack([zoom(image[c], factor, order=1) for c in range(C)], axis=0)
        zoomed_label = zoom(label, factor, order=0)

        zoomed_image = _center_crop_or_pad_chw(zoomed_image, orig_shape)
        zoomed_label = _center_crop_or_pad_hwd(zoomed_label, orig_shape)
        return zoomed_image.astype(image.dtype), zoomed_label.astype(label.dtype)

    def _gaussian_noise(self, image):
        if np.random.rand() >= self.p_gaussian_noise:
            return image
        std = np.random.uniform(*self.noise_std_range)
        noise = np.random.normal(0, std, size=image.shape).astype(image.dtype)
        return image + noise

    def _gaussian_blur(self, image):
        if np.random.rand() >= self.p_gaussian_blur:
            return image
        sigma = np.random.uniform(*self.blur_sigma_range)
        return np.stack([gaussian_filter(image[c], sigma) for c in range(image.shape[0])], axis=0)

    def _brightness(self, image):
        if np.random.rand() >= self.p_brightness:
            return image
        factor = np.random.uniform(*self.brightness_range)
        return image * factor

    def _contrast(self, image):
        if np.random.rand() >= self.p_contrast:
            return image
        factor = np.random.uniform(*self.contrast_range)
        mean = image.mean(axis=(1, 2, 3), keepdims=True)
        return (image - mean) * factor + mean

    def _low_res_simulation(self, image):
        """Downsample then upsample to simulate a lower-resolution acquisition."""
        if np.random.rand() >= self.p_low_res:
            return image
        factor = np.random.uniform(*self.low_res_zoom_range)
        C = image.shape[0]
        orig_shape = image.shape[1:]
        down = np.stack([zoom(image[c], factor, order=1) for c in range(C)], axis=0)
        up = np.stack([zoom(down[c], np.array(orig_shape) / np.array(down.shape[1:]), order=1) for c in range(C)], axis=0)
        return _center_crop_or_pad_chw(up, orig_shape).astype(image.dtype)

    def _gamma(self, image):
        if np.random.rand() >= self.p_gamma:
            return image
        gamma = np.random.uniform(*self.gamma_range)
        img_min = image.min(axis=(1, 2, 3), keepdims=True)
        img_range = image.max(axis=(1, 2, 3), keepdims=True) - img_min + 1e-8
        normalized = (image - img_min) / img_range
        gammaed = np.power(normalized, gamma)
        return gammaed * img_range + img_min

    def _mirror(self, image, label):
        for axis in (1, 2, 3):  # each spatial axis independently, matching nnU-Net's per-axis mirroring
            if np.random.rand() < self.p_mirror:
                image = np.flip(image, axis=axis)
                label = np.flip(label, axis=axis - 1)
        return np.ascontiguousarray(image), np.ascontiguousarray(label)


def _center_crop_or_pad_chw(volume, target_shape):
    """volume: (C, H, W, D) -> crop/pad spatial dims to target_shape."""
    pad_width = [(0, 0)]
    starts = []
    for dim_size, target in zip(volume.shape[1:], target_shape):
        if dim_size < target:
            pad_before = (target - dim_size) // 2
            pad_after = target - dim_size - pad_before
            pad_width.append((pad_before, pad_after))
            starts.append(0)
        else:
            pad_width.append((0, 0))
            starts.append((dim_size - target) // 2)
    volume = np.pad(volume, pad_width, mode='constant')
    return volume[:, starts[0]:starts[0] + target_shape[0], starts[1]:starts[1] + target_shape[1], starts[2]:starts[2] + target_shape[2]]


def _center_crop_or_pad_hwd(volume, target_shape):
    """volume: (H, W, D) -> crop/pad to target_shape."""
    pad_width = []
    starts = []
    for dim_size, target in zip(volume.shape, target_shape):
        if dim_size < target:
            pad_before = (target - dim_size) // 2
            pad_after = target - dim_size - pad_before
            pad_width.append((pad_before, pad_after))
            starts.append(0)
        else:
            pad_width.append((0, 0))
            starts.append((dim_size - target) // 2)
    volume = np.pad(volume, pad_width, mode='constant')
    return volume[starts[0]:starts[0] + target_shape[0], starts[1]:starts[1] + target_shape[1], starts[2]:starts[2] + target_shape[2]]
