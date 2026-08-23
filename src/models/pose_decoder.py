"""SONIC-style dilated temporal CNN adapted for pose regression."""

from __future__ import annotations

import torch
from torch import nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad, None


def _default_dilations(n_layers: int) -> tuple[int, ...]:
    return tuple(2**i for i in range(n_layers))


class PoseDecoder(nn.Module):
    """Dilated Conv1d stack + residual pose head + session adversary."""

    def __init__(
        self,
        *,
        in_channels: int = 32,
        hidden_channels: int = 64,
        kernel_size: int = 10,
        stride: int = 1,
        n_layers: int = 2,
        n_keypoints: int = 15,
        coords: int = 2,
        dilations: tuple[int, ...] | None = None,
        dropout: float = 0.0,
        n_sessions: int = 0,
        n_classes: int = 0,
    ) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        dils = dilations if dilations is not None else _default_dilations(n_layers)
        if len(dils) != n_layers:
            raise ValueError(
                f"dilations length {len(dils)} must match n_layers {n_layers}"
            )
        if any(d < 1 for d in dils):
            raise ValueError(f"dilations must be >= 1, got {dils}")
        layers: list[nn.Module] = []
        ch_in = in_channels
        for dil in dils:
            pad = dil * (kernel_size // 2)
            layers.append(
                nn.Conv1d(
                    ch_in,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    dilation=dil,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            ch_in = hidden_channels
        self.encoder = nn.Sequential(*layers)
        self.n_classes = int(n_classes)
        out = n_classes if self.n_classes else n_keypoints * coords
        self.head = nn.Linear(hidden_channels, out)
        self.adv_head = None
        if n_sessions >= 2:
            adv: list[nn.Module] = [
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                adv.append(nn.Dropout(dropout))
            adv.append(nn.Linear(hidden_channels, n_sessions))
            self.adv_head = nn.Sequential(*adv)
        self.register_buffer("pose_mean", torch.zeros(n_keypoints, coords))
        self.n_keypoints = n_keypoints
        self.coords = coords
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_layers = n_layers
        self.dilations = dils
        self.time_stride = stride**n_layers

    def encode(self, neural: torch.Tensor) -> torch.Tensor:
        # neural: [B, T, C] -> [B, C, T]
        x = neural.transpose(1, 2)
        h = self.encoder(x)
        # even kernels and dilated padding can grow the time axis by a few samples
        if self.time_stride == 1:
            t = neural.shape[1]
            if h.shape[-1] > t:
                h = h[..., :t]
        return h.transpose(1, 2)  # [B, T', H]

    def forward(
        self,
        neural: torch.Tensor,
        pose_bin_idx: torch.Tensor | None = None,
        pose_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Class logits [B, C] or mean plus residual pose [B, P, K, 2]."""
        encoded = self.encode(neural)  # [B, T', H]
        if self.n_classes:
            return self.head(encoded.mean(dim=1))
        if pose_bin_idx is None:
            raise ValueError("pose_bin_idx is required for the pose head")
        b, p = pose_bin_idx.shape
        t = encoded.shape[1]
        # map original 20 ms bin indices onto the strided encoder timeline
        idx = (pose_bin_idx // self.time_stride).clamp(0, t - 1)
        batch_idx = torch.arange(b, device=neural.device)[:, None].expand(b, p)
        sampled = encoded[batch_idx, idx]  # [B, P, H]
        residual = self.head(sampled).view(b, p, self.n_keypoints, self.coords)
        pred = residual + self.pose_mean
        if pose_valid is not None:
            pred = pred * pose_valid[..., None, None].to(pred.dtype)
        return pred

    def session_logits(self, neural: torch.Tensor, lambd: float) -> torch.Tensor | None:
        if self.adv_head is None:
            return None
        pooled = self.encode(neural).mean(dim=1)
        return self.adv_head(_GradReverse.apply(pooled, lambd))


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    pose_mask: torch.Tensor,
    pose_valid: torch.Tensor,
) -> torch.Tensor:
    """pose_mask [B,P,K], pose_valid [B,P]."""
    mask = pose_mask * pose_valid[..., None].to(pose_mask.dtype)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    err = nn.functional.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    return (err * mask).sum() / mask.sum()


def masked_pixel_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    pose_mask: torch.Tensor,
    pose_valid: torch.Tensor,
    *,
    frame_width: float,
    frame_height: float,
) -> dict[str, float]:
    mask = pose_mask * pose_valid[..., None].to(pose_mask.dtype)
    scale = pred.new_tensor([frame_width, frame_height])
    pred_px = pred * scale
    target_px = target * scale
    dist = torch.linalg.norm(pred_px - target_px, dim=-1)
    if mask.sum() == 0:
        return {
            "rmse_px": float("nan"),
            "rmse_norm": float("nan"),
            "corr": float("nan"),
        }
    rmse_px = torch.sqrt(((dist * mask) ** 2).sum() / mask.sum()).item()
    # normalized by frame diagonal
    diag = float((frame_width**2 + frame_height**2) ** 0.5)
    rmse_norm = rmse_px / diag
    flat_mask = mask.bool()
    pred_flat = pred[flat_mask.unsqueeze(-1).expand_as(pred)].view(-1)
    target_flat = target[flat_mask.unsqueeze(-1).expand_as(target)].view(-1)
    if pred_flat.numel() < 2:
        corr = float("nan")
    else:
        pred_c = pred_flat - pred_flat.mean()
        target_c = target_flat - target_flat.mean()
        denom = pred_c.norm() * target_c.norm()
        corr = float((pred_c * target_c).sum() / denom) if denom > 0 else float("nan")
    return {"rmse_px": rmse_px, "rmse_norm": rmse_norm, "corr": corr}


def per_keypoint_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    pose_mask: torch.Tensor,
    pose_valid: torch.Tensor,
    *,
    frame_width: float,
    frame_height: float,
) -> list[float]:
    mask = pose_mask * pose_valid[..., None].to(pose_mask.dtype)
    scale = pred.new_tensor([frame_width, frame_height])
    dist = torch.linalg.norm(pred * scale - target * scale, dim=-1)
    out = []
    for k in range(dist.shape[-1]):
        m = mask[..., k]
        if m.sum() == 0:
            out.append(float("nan"))
        else:
            out.append(torch.sqrt(((dist[..., k] * m) ** 2).sum() / m.sum()).item())
    return out
