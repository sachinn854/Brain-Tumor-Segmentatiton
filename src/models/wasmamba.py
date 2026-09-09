"""
WAS-Mamba architecture (Windowed Attention State-Space Model).

Source: WASMamba.py, https://github.com/1605066114/WAS-Mamba
Paper: Zhang et al., "WAS-Mamba: 3D Medical Image Segmentation via Windowed
Attention State Space Model," IEEE Transactions on Image Processing, vol. 35, 2026.

This file is copied faithfully from the authors' repo (only reformatted into
this project's structure, no logic changed). It is the ONLY part of the
official release that is complete and directly usable — the repo's
training/config/dataset code is Synapse-specific and incomplete (see
src/configs/wasmamba_config.py for what is missing).

External deps this file needs: torch, einops, timm, mamba_ssm
(`pip install einops timm mamba-ssm causal-conv1d`). mamba_ssm's selective_scan
op typically needs a CUDA build — plan to install/test this on whatever GPU
you end up using (Colab / college lab), not on CPU-only.
"""

import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_, to_3tuple

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    import numpy as np

    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex
    flops = 0
    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


class windowsort(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.GiantParameter1 = torch.nn.Unfold(kernel_size=(1, 1, n), stride=(1, 1, n))

    def forward(self, x):
        x = self.GiantParameter1(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=(2, 2, 2), in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape
        SHAPE_FIX = [-1, -1, -1]
        if (D % 2 != 0) or (H % 2 != 0) or (W % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = D // 2
            SHAPE_FIX[1] = H // 2
            SHAPE_FIX[2] = W // 2

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
            x4 = x4[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
            x5 = x5[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
            x6 = x6[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
            x7 = x7[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale ** 3 * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(
            x,
            'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c',
            p1=self.dim_scale, p2=self.dim_scale, p3=self.dim_scale,
            c=C // (self.dim_scale ** 3)
        )
        x = self.norm(x)
        return x


class Final_PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale ** 3 * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(
            x,
            'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c',
            p1=self.dim_scale, p2=self.dim_scale, p3=self.dim_scale,
            c=C // (self.dim_scale ** 3)
        )
        x = self.norm(x)
        return x


class WSSM(nn.Module):
    """Windowed State-Space Module — the paper's core attention replacement."""

    def __init__(
            self,
            d_model=96,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(d_state / 2)
        self.d_conv = to_3tuple(d_conv)
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 32) if dt_rank == "auto" else dt_rank

        self.resproj = nn.Linear(self.d_inner, self.d_inner, bias=bias)
        self.catconv = nn.Sequential(
            nn.Conv3d(self.d_inner, self.d_inner, kernel_size=self.d_conv, padding=tuple(p // 2 for p in self.d_conv), groups=self.d_inner),
            nn.BatchNorm3d(self.d_inner),
            nn.SiLU(inplace=True),
        )
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            padding=tuple(p // 2 for p in self.d_conv),
            bias=conv_bias,
            groups=self.d_inner,
        )
        self.fftconv = nn.Sequential(
            nn.Conv3d(
                in_channels=2 * self.d_inner,
                out_channels=self.d_inner,
                kernel_size=self.d_conv,
                padding=tuple(p // 2 for p in self.d_conv),
                bias=conv_bias,
                groups=self.d_inner,
            ),
            nn.BatchNorm3d(self.d_inner),
            nn.SiLU(inplace=True)
        )
        self.finconv = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            padding=tuple(p // 2 for p in self.d_conv),
            bias=conv_bias,
            groups=self.d_inner,
        )

        self.y1conv = nn.Sequential(nn.Conv2d(
            in_channels=2 * self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
            groups=self.d_inner,
        ), nn.BatchNorm2d(self.d_inner),
            nn.SiLU(inplace=True))
        self.y2conv = nn.Sequential(nn.Conv2d(
            in_channels=2 * self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
            groups=self.d_inner,
        ), nn.BatchNorm2d(self.d_inner),
            nn.SiLU(inplace=True))
        self.y3conv = nn.Sequential(nn.Conv2d(
            in_channels=2 * self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
            groups=self.d_inner,
        ), nn.BatchNorm2d(self.d_inner),
            nn.SiLU(inplace=True))
        self.y4conv = nn.Sequential(nn.Conv2d(
            in_channels=2 * self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
            groups=self.d_inner,
        ), nn.BatchNorm2d(self.d_inner),
            nn.SiLU(inplace=True))

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(int(self.d_inner), (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(int(self.d_inner), (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(int(self.d_inner), (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(int(self.d_inner), (self.dt_rank + self.d_state * 2), bias=False),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, int(self.d_inner), dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, int(self.d_inner), dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, int(self.d_inner), dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, int(self.d_inner), dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, int(self.d_inner), copies=4, merge=True)
        self.A2 = self.A_log_init(self.d_state, int(self.d_inner), copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0
        self.out_norm = nn.LayerNorm(int(self.d_inner))
        self.out_proj = nn.Linear(int(self.d_inner), self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, D, H, W = x.shape
        n = 8
        self.patchconv = torch.nn.Unfold(kernel_size=(1, 1, int(W / 8)), stride=(1, 1, int(W / 8))).to(x.device)
        self.unpatchconv = torch.nn.Fold(output_size=(D, H, W), kernel_size=(1, 1, int(W / 8)), stride=(1, 1, int(W / 8))).to(x.device)
        L = D * H * W
        K = 4

        if W % 8 == 0:
            xtf = self.patchconv(x.flatten(2))
            swinx = torch.roll(x, shifts=(0, 0, int(W / 8 / 2)), dims=(2, 3, 4))
            xtfswinx = self.patchconv(swinx.flatten(2))
            xt = x.transpose(3, 4).flip(dims=(3, 4))
            xtb = self.patchconv(xt.flatten(2))
            swiny = torch.roll(xt, shifts=(0, 0, int(W / 8 / 2)), dims=(2, 3, 4))
            xtbswiny = self.patchconv(swiny.flatten(2))
            xs = torch.stack((xtf, xtfswinx, xtb, xtbswiny), dim=1)
        else:
            xtf = x
            xtff = xtf.transpose(3, 4).flip(dims=(3, 4))
            xtb = torch.transpose(x, dim0=2, dim1=3)
            xtbf = xtb.transpose(3, 4).flip(dims=(3, 4))
            xs = torch.stack((xtf, xtff, xtb, xtbf), dim=1)

        xs = xs.reshape(B, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False,
        ).view(B, K, -1, n * n * n).contiguous()

        if W % 8 == 0:
            out_y = out_y.view(B, K, -1, n * n * n).contiguous()
            y1 = self.unpatchconv(out_y[:, 0, :, :]).view(B, -1, D, H, W)
            y2 = torch.roll(self.unpatchconv(out_y[:, 1, :, :]).view(B, -1, D, H, W), shifts=(0, 0, -int(W / 8 / 2)), dims=(2, 3, 4))
            y3 = self.unpatchconv(out_y[:, 2, :, :]).view(B, -1, D, H, W).transpose(3, 4).flip(dims=(3, 4))
            y4 = torch.roll(self.unpatchconv(out_y[:, 3, :, :]).view(B, -1, D, H, W), shifts=(0, 0, -int(W / 8 / 2)), dims=(2, 3, 4)).transpose(3, 4).flip(dims=(3, 4))
        else:
            out_y = out_y.view(B, K * self.d_inner, D, H, W).contiguous()
            y1, y2, y3, y4 = out_y.chunk(4, dim=1)
            y2 = y2.transpose(3, 4).flip(dims=(3, 4))
            y3 = torch.transpose(y3, dim0=2, dim1=3)
            y4 = torch.transpose(y4.transpose(3, 4).flip(dims=(3, 4)), dim0=2, dim1=3)

        return y1, y2, y3, y4

    def forward(self, x: torch.Tensor):
        B, D, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        freq = torch.fft.fftn(x, dim=(2, 3, 4)).to(torch.float32)
        freq = self.finconv(freq)
        freq = torch.fft.ifftn(freq, dim=(2, 3, 4)).to(torch.float32)
        ffreq = torch.concat([x, freq], dim=1)
        f = self.fftconv(ffreq)
        out1 = f

        x = self.act(self.conv3d(x))
        y1, y2, y3, y4 = self.forward_core(x)

        y1_in = torch.concat([out1, y1], dim=1)
        B_y, C_y, D_y, H_y, W_y = y1_in.shape
        y1 = self.y1conv(y1_in.reshape(B_y * D_y, C_y, H_y, W_y)).reshape(B_y, -1, D_y, H_y, W_y)

        y2_in = torch.concat([out1, y2], dim=1)
        B_y, C_y, D_y, H_y, W_y = y2_in.shape
        y2 = self.y2conv(y2_in.reshape(B_y * D_y, C_y, H_y, W_y)).reshape(B_y, -1, D_y, H_y, W_y)

        y3_in = torch.concat([out1, y3], dim=1)
        B_y, C_y, D_y, H_y, W_y = y3_in.shape
        y3 = self.y3conv(y3_in.reshape(B_y * D_y, C_y, H_y, W_y)).reshape(B_y, -1, D_y, H_y, W_y)

        y4_in = torch.concat([out1, y4], dim=1)
        B_y, C_y, D_y, H_y, W_y = y4_in.shape
        y4 = self.y4conv(y4_in.reshape(B_y * D_y, C_y, H_y, W_y)).reshape(B_y, -1, D_y, H_y, W_y)

        out2 = y1 + y2 + y3 + y4

        y = self.catconv(out2)
        z = self.resproj(z).permute(0, 4, 1, 2, 3)
        y = (y * F.silu(z)).permute(0, 2, 3, 4, 1)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class WASBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = WSSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class WASLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            d_state=16,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            WASBlock(
                hidden_dim=dim,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.apply(_init_weights)

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class WASLayer_up(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            WASBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.apply(_init_weights)

        self.upsample = upsample(dim=dim, norm_layer=norm_layer) if upsample is not None else None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VSSM(nn.Module):
    def __init__(self, patch_size=(2, 2, 2), in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = WASLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = WASLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand(dim=dims_decoder[-1], dim_scale=2, norm_layer=norm_layer)
        self.final_conv = nn.Sequential(
            nn.Conv3d(dims_decoder[-1] // 2, dims_decoder[-1] // 2, 3, padding=1),
            nn.BatchNorm3d(dims_decoder[-1] // 2),
            nn.SiLU(inplace=True)
        )
        self.final_conv2 = nn.Sequential(
            nn.Conv3d(dims_decoder[-1] // 2, dims_decoder[-1] // 2, 3, padding=1),
            nn.BatchNorm3d(dims_decoder[-1] // 2),
            nn.SiLU(inplace=True)
        )
        self.final_conv3 = nn.Sequential(
            nn.Conv3d(dims_decoder[-1] // 2, num_classes, 3, padding=1),
            nn.BatchNorm3d(num_classes),
            nn.SiLU(inplace=True)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.final_conv(x)
        x = self.final_conv2(x)
        x = self.final_conv3(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        return x


class WASMamba(nn.Module):
    def __init__(
            self,
            input_channels=3,
            num_classes=1,
            depths=[2, 2, 2, 2],
            depths_decoder=[2, 2, 2, 2],
            drop_path_rate=0.2,
            load_ckpt_path=None,
    ):
        super().__init__()
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.wasmamba = VSSM(
            patch_size=(2, 2, 2),
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)
        logits = self.wasmamba(x)
        return logits
