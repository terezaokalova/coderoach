"""SONIC-style two-layer temporal CNN adapted for pose regression."""

from __future__ import annotations

import torch
from torch import nn


class PoseDecoder(nn.Module):
    """Temporal Conv1d stack + linear head to keypoints."""

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
    ) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(n_layers):
            layers.append(
                nn.Conv1d(
                    ch_in,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            ch_in = hidden_channels
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_channels, n_keypoints * coords)
        self.n_keypoints = n_keypoints
        self.coords = coords
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_layers = n_layers
        self.time_stride = stride**n_layers

    def encode(self, neural: torch.Tensor) -> torch.Tensor:
        # neural: [B, T, C] -> [B, C, T]
        x = neural.transpose(1, 2)
        h = self.encoder(x)
        # same-length padding with even kernels can add one sample at stride 1
        if self.time_stride == 1:
            t = neural.shape[1]
            if h.shape[-1] != t:
                h = h[..., :t]
        return h.transpose(1, 2)  # [B, T', H]

    def forward(
        self,
        neural: torch.Tensor,
        pose_bin_idx: torch.Tensor,
        pose_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return pose predictions [B, P, K, 2] sampled at pose_bin_idx."""
        encoded = self.encode(neural)  # [B, T', H]
        b, p = pose_bin_idx.shape
        t = encoded.shape[1]
        # map original 20 ms bin indices onto the strided encoder timeline
        idx = (pose_bin_idx // self.time_stride).clamp(0, t - 1)
        batch_idx = torch.arange(b, device=neural.device)[:, None].expand(b, p)
        sampled = encoded[batch_idx, idx]  # [B, P, H]
        pred = self.head(sampled).view(b, p, self.n_keypoints, self.coords)
        if pose_valid is not None:
            pred = pred * pose_valid[..., None, None].to(pred.dtype)
        return pred


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
