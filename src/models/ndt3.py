"""NDT3 encoder plus a new pose head, loadable from joel99/ndt3 checkpoints."""

from __future__ import annotations

import math
import pickle
import urllib.request
from pathlib import Path

import torch
from torch import nn

from models.config import ModelConfig, ROOT
from models.pose_decoder import PoseDecoder


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, 2 * hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(nn.functional.silu(left) * gate)


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = _SwiGLU(dim, ff)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        return x + self.mlp(self.norm2(x))


def _rope(x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, D] with even D
    t, dim = x.shape[1], x.shape[-1]
    half = dim // 2
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=x.device, dtype=x.dtype) / half
    )
    ang = torch.arange(t, device=x.device, dtype=x.dtype)[:, None] * freq
    cos = ang.cos()[None, :, None, :]
    sin = ang.sin()[None, :, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rot.flatten(-2)


class NDT3PoseDecoder(nn.Module):
    """32-channel count tokens (NDT3 45M recipe) plus a linear keypoint head."""

    def __init__(
        self,
        *,
        in_channels: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        hidden_size: int = 1024,
        n_keypoints: int = 15,
        coords: int = 2,
        max_count: int = 21,
        freeze_backbone: bool = False,
        feedforward_factor: int = 1,
    ) -> None:
        super().__init__()
        if hidden_size % in_channels != 0:
            raise ValueError("hidden_size must be divisible by in_channels")
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.n_keypoints = n_keypoints
        self.coords = coords
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.max_count = max_count
        embed_dim = hidden_size // in_channels
        self.spike_embed = nn.Embedding(max_count, embed_dim)
        ff = hidden_size * feedforward_factor
        self.layers = nn.ModuleList(
            [_Block(hidden_size, n_heads, ff) for _ in range(n_layers)]
        )
        self.head = nn.Linear(hidden_size, n_keypoints * coords)
        self.register_buffer("pose_mean", torch.zeros(n_keypoints, coords))
        if freeze_backbone:
            for param in (*self.spike_embed.parameters(), *self.layers.parameters()):
                param.requires_grad = False

    def encode(self, neural: torch.Tensor) -> torch.Tensor:
        # neural: [B, T, C] raw spike counts
        counts = neural.round().long().clamp(0, self.max_count - 1)
        token = self.spike_embed(counts).flatten(-2)  # [B, T, H]
        b, t, _ = token.shape
        head_dim = self.hidden_size // self.n_heads
        qk = token.view(b, t, self.n_heads, head_dim)
        token = _rope(qk).reshape(b, t, self.hidden_size)
        mask = torch.triu(
            torch.ones(t, t, device=neural.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            token = layer(token, mask)
        return token

    def forward(
        self,
        neural: torch.Tensor,
        pose_bin_idx: torch.Tensor,
        pose_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encode(neural)
        b, p = pose_bin_idx.shape
        t = encoded.shape[1]
        idx = pose_bin_idx.clamp(0, t - 1)
        batch_idx = torch.arange(b, device=neural.device)[:, None].expand(b, p)
        sampled = encoded[batch_idx, idx]
        residual = self.head(sampled).view(b, p, self.n_keypoints, self.coords)
        pred = residual + self.pose_mean
        if pose_valid is not None:
            pred = pred * pose_valid[..., None, None].to(pred.dtype)
        return pred


def download_ndt3_checkpoint(
    repo: str,
    file: str,
    cache_dir: Path | None = None,
) -> Path:
    cache_dir = cache_dir or (ROOT / ".cache" / "ndt3")
    dest = cache_dir / file.replace("/", "__")
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo, filename=file, cache_dir=str(cache_dir))
        return Path(path)
    except ImportError:
        url = f"https://huggingface.co/{repo}/resolve/main/{file}"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
        return dest


class _IgnoreUnpickled:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        return


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (_IgnoreUnpickled,), {"__module__": module})


class _CheckpointPickle:
    Unpickler = _CheckpointUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _CheckpointUnpickler(file).load()


def _load_lightning_ckpt(ckpt_path: Path) -> object:
    return torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_CheckpointPickle,
    )


def _state_dict_from_ckpt(blob: object) -> dict[str, torch.Tensor]:
    if not isinstance(blob, dict):
        raise TypeError("NDT3 checkpoint is not a mapping")
    if "state_dict" in blob and isinstance(blob["state_dict"], dict):
        return blob["state_dict"]
    return {k: v for k, v in blob.items() if torch.is_tensor(v)}


def load_ndt3_weights(model: NDT3PoseDecoder, ckpt_path: Path) -> dict[str, int]:
    blob = _load_lightning_ckpt(ckpt_path)
    src = _state_dict_from_ckpt(blob)
    dest = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for key, value in src.items():
        name = key.removeprefix("model.")
        if name.endswith("readin.weight") and "spike" in name:
            target = "spike_embed.weight"
        elif ".mixer.Wqkv.weight" in name:
            parts = name.split(".")
            layer = parts[parts.index("layers") + 1]
            target = f"layers.{layer}.attn.in_proj_weight"
        elif ".mixer.out_proj.weight" in name:
            parts = name.split(".")
            layer = parts[parts.index("layers") + 1]
            target = f"layers.{layer}.attn.out_proj.weight"
        elif ".mlp.fc1.weight" in name:
            parts = name.split(".")
            layer = parts[parts.index("layers") + 1]
            target = f"layers.{layer}.mlp.fc1.weight"
        elif ".mlp.fc2.weight" in name:
            parts = name.split(".")
            layer = parts[parts.index("layers") + 1]
            target = f"layers.{layer}.mlp.fc2.weight"
        else:
            continue
        if target in dest and dest[target].shape == value.shape:
            mapped[target] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    loaded = len(mapped)
    print(
        f"NDT3 loaded {loaded} tensors from {ckpt_path.name} "
        f"(skipped {len(missing)} unmatched)"
    )
    return {
        "loaded": loaded,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def build_pose_model(
    cfg: ModelConfig, *, n_sessions: int = 0
) -> PoseDecoder | NDT3PoseDecoder:
    if cfg.backbone == "cnn":
        return PoseDecoder(
            in_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            n_layers=cfg.n_layers,
            n_keypoints=cfg.n_keypoints,
            coords=cfg.coords,
            dilations=cfg.dilations,
            dropout=cfg.dropout,
            n_sessions=n_sessions,
        )
    if cfg.backbone != "ndt3":
        raise ValueError(f"unknown backbone {cfg.backbone!r}")
    model = NDT3PoseDecoder(
        in_channels=cfg.in_channels,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        hidden_size=cfg.hidden_size,
        n_keypoints=cfg.n_keypoints,
        coords=cfg.coords,
        freeze_backbone=cfg.freeze_backbone,
        feedforward_factor=cfg.feedforward_factor,
    )
    if cfg.ndt3_file:
        ckpt = download_ndt3_checkpoint(cfg.ndt3_repo, cfg.ndt3_file)
        load_ndt3_weights(model, ckpt)
    return model
