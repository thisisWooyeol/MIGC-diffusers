# Copied from original MIGC implementation / slight type checking modification
# https://github.com/limuloo/MIGC/blob/main/migc/migc_layers.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attn: bool = False,
        need_softmax: bool = True,
        guidance_mask: torch.Tensor | None = None,
        forward_layout_guidance: bool = False,
    ):
        h = self.heads
        b = x.shape[0]

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        if forward_layout_guidance:
            # sim: (B * phase_num * h, HW, 77), b = B * phase_num
            # guidance_mask: (B, phase_num, 64, 64)
            HW = sim.shape[1]
            H = W = int(math.sqrt(HW))
            guidance_mask = F.interpolate(guidance_mask, size=(H, W), mode="nearest")  # (B, phase_num, H, W)
            sim = sim.view(b, h, HW, 77)
            guidance_mask = guidance_mask.view(b, 1, HW, 1)
            guidance_mask[guidance_mask == 1] = 5.0
            guidance_mask[guidance_mask == 0] = 0.1
            sim[:, :, :, 1:] = sim[:, :, :, 1:] * guidance_mask
            sim = sim.view(b * h, HW, 77)

        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if need_softmax:
            attn = sim.softmax(dim=-1)
        else:
            attn = sim

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        if return_attn:
            attn = attn.view(b, h, attn.shape[-2], attn.shape[-1])
            return self.to_out(out), attn
        else:
            return self.to_out(out)


class LayoutAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_lora: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.use_lora = use_lora
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        guidance_mask: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attn: bool = False,
        need_softmax: bool = True,
    ):
        h = self.heads
        b = x.shape[0]

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        _, phase_num, H, W = guidance_mask.shape
        HW = H * W
        guidance_mask_o = guidance_mask.view(b * phase_num, HW, 1)
        guidance_mask_t = guidance_mask.view(b * phase_num, 1, HW)
        guidance_mask_sim = torch.bmm(guidance_mask_o, guidance_mask_t)  # (B * phase_num, HW, HW)
        guidance_mask_sim = guidance_mask_sim.view(b, phase_num, HW, HW).sum(dim=1)
        guidance_mask_sim[guidance_mask_sim > 1] = 1  # (B, HW, HW)
        guidance_mask_sim = guidance_mask_sim.view(b, 1, HW, HW)
        guidance_mask_sim = guidance_mask_sim.repeat(1, self.heads, 1, 1)
        guidance_mask_sim = guidance_mask_sim.view(b * self.heads, HW, HW)  # (B * head, HW, HW)

        sim[:, :, :HW][guidance_mask_sim == 0] = -torch.finfo(sim.dtype).max

        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of

        if need_softmax:
            attn = sim.softmax(dim=-1)
        else:
            attn = sim

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        if return_attn:
            attn = attn.view(b, h, attn.shape[-2], attn.shape[-1])
            return self.to_out(out), attn
        else:
            return self.to_out(out)


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        relu: bool = True,
        bn: bool = False,
        bias: bool = False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels: int, reduction_ratio: int = 16, pool_types: list[str] = ["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            else:
                raise ValueError(f"Unknown pool type: {pool_type}")

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor: torch.Tensor) -> torch.Tensor:
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
        pool_types: list[str] = ["avg", "max"],
        no_spatial: bool = False,
    ):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
