"""Load and validate pose-decoder YAML configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data_info.yaml").exists()
)


@dataclass(frozen=True)
class DataConfig:
    info_yaml: Path
    sessions: tuple[str, ...]
    camera: str
    frame_width: int
    frame_height: int
    likelihood_threshold: float
    bin_ms: float
    window_s: float
    cache_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    protocol: str
    train_frac: float
    val_frac: float
    test_frac: float
    guard_s: float
    train_sessions: tuple[str, ...]
    val_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int
    hidden_channels: int
    kernel_size: int
    stride: int
    n_keypoints: int
    coords: int
    backbone: str
    ndt3_repo: str
    ndt3_file: str
    freeze_backbone: bool
    n_layers: int
    n_heads: int
    hidden_size: int
    feedforward_factor: int
    dilations: tuple[int, ...]
    dropout: float
    head: str
    grid_x: int
    grid_y: int


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stop_patience: int
    num_workers: int
    device: str
    checkpoint_dir: Path
    time_shift_bins: int
    adv_weight: float


@dataclass(frozen=True)
class PoseDecoderConfig:
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    train: TrainConfig


_DATA_KEYS = {
    "info_yaml",
    "sessions",
    "camera",
    "frame_width",
    "frame_height",
    "likelihood_threshold",
    "bin_ms",
    "window_s",
    "cache_dir",
}
_SPLIT_KEYS = {
    "protocol",
    "train_frac",
    "val_frac",
    "test_frac",
    "guard_s",
    "train_sessions",
    "val_sessions",
    "test_sessions",
}
_MODEL_KEYS = {
    "in_channels",
    "hidden_channels",
    "kernel_size",
    "stride",
    "n_layers",
    "n_keypoints",
    "coords",
}
_MODEL_OPTIONAL = {
    "backbone",
    "ndt3_repo",
    "ndt3_file",
    "freeze_backbone",
    "n_heads",
    "hidden_size",
    "feedforward_factor",
    "dilations",
    "dropout",
    "head",
    "grid_x",
    "grid_y",
}
_TRAIN_OPTIONAL = {"adv_weight"}
_BACKBONES = {"cnn", "ndt3"}
_HEADS = {"pose", "class"}
_TRAIN_KEYS = {
    "seed",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "max_epochs",
    "early_stop_patience",
    "num_workers",
    "device",
    "checkpoint_dir",
    "time_shift_bins",
}
_TOP_KEYS = {"data", "split", "model", "train"}


def _require_section(
    raw: dict[str, Any],
    name: str,
    expected: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if name not in raw:
        raise KeyError(f"config missing section: {name}")
    section = raw[name]
    if not isinstance(section, dict):
        raise TypeError(f"config section {name} must be a mapping")
    keys = set(section)
    missing = expected - keys
    unknown = keys - expected - (optional or set())
    if missing:
        raise KeyError(f"config.{name} missing keys: {sorted(missing)}")
    if unknown:
        raise KeyError(f"config.{name} unknown keys: {sorted(unknown)}")
    return section


def _as_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def load_pose_decoder_config(path: str | Path) -> PoseDecoderConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("config root must be a mapping")
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise KeyError(f"config unknown top-level keys: {sorted(unknown)}")
    missing = _TOP_KEYS - set(raw)
    if missing:
        raise KeyError(f"config missing top-level keys: {sorted(missing)}")

    data_raw = _require_section(raw, "data", _DATA_KEYS)
    split_raw = _require_section(raw, "split", _SPLIT_KEYS)
    model_raw = _require_section(raw, "model", _MODEL_KEYS, optional=_MODEL_OPTIONAL)
    train_raw = _require_section(raw, "train", _TRAIN_KEYS, optional=_TRAIN_OPTIONAL)

    protocol = str(split_raw["protocol"])
    if protocol not in {"temporal", "session"}:
        raise ValueError(
            f"split.protocol must be temporal or session, got {protocol!r}"
        )
    backbone = str(model_raw.get("backbone", "cnn"))
    if backbone not in _BACKBONES:
        raise ValueError(f"model.backbone must be cnn or ndt3, got {backbone!r}")
    head = str(model_raw.get("head", "pose"))
    if head not in _HEADS:
        raise ValueError(f"model.head must be pose or class, got {head!r}")
    grid_x = int(model_raw.get("grid_x", 4))
    grid_y = int(model_raw.get("grid_y", 4))
    if grid_x < 1 or grid_y < 1:
        raise ValueError(f"model.grid_x/grid_y must be >= 1, got {grid_x}x{grid_y}")
    stride = int(model_raw["stride"])
    if stride < 1:
        raise ValueError(f"model.stride must be >= 1, got {stride}")
    n_layers = int(model_raw["n_layers"])
    if n_layers < 1:
        raise ValueError(f"model.n_layers must be >= 1, got {n_layers}")
    if "dilations" in model_raw:
        dilations = tuple(int(d) for d in model_raw["dilations"])
        if len(dilations) != n_layers:
            raise ValueError(
                f"model.dilations length {len(dilations)} must match "
                f"n_layers {n_layers}"
            )
        if any(d < 1 for d in dilations):
            raise ValueError(f"model.dilations must be >= 1, got {dilations}")
    else:
        dilations = tuple(2**i for i in range(n_layers))
    dropout = float(model_raw.get("dropout", 0.0))
    if dropout < 0 or dropout >= 1:
        raise ValueError(f"model.dropout must be in [0, 1), got {dropout}")
    adv_weight = float(train_raw.get("adv_weight", 0.0))
    if adv_weight < 0:
        raise ValueError(f"train.adv_weight must be >= 0, got {adv_weight}")

    train_frac = float(split_raw["train_frac"])
    val_frac = float(split_raw["val_frac"])
    test_frac = float(split_raw["test_frac"])
    if protocol == "session":
        # Test sessions: first half is train/val, second half is held-out test.
        # Optional train_sessions are extra pool recordings.
        if abs(train_frac + val_frac - 1.0) > 1e-6:
            raise ValueError(
                "session protocol requires train_frac + val_frac == 1 "
                f"(got {train_frac + val_frac})"
            )
        if test_frac != 0.0:
            raise ValueError(
                "session protocol requires test_frac: 0 "
                "(test is the second half of test_sessions, not a time fraction)"
            )
    else:
        frac_sum = train_frac + val_frac + test_frac
        if abs(frac_sum - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1, got {frac_sum}")

    base = config_path.parent.parent
    return PoseDecoderConfig(
        data=DataConfig(
            info_yaml=_as_path(data_raw["info_yaml"], base),
            sessions=tuple(data_raw["sessions"]),
            camera=str(data_raw["camera"]),
            frame_width=int(data_raw["frame_width"]),
            frame_height=int(data_raw["frame_height"]),
            likelihood_threshold=float(data_raw["likelihood_threshold"]),
            bin_ms=float(data_raw["bin_ms"]),
            window_s=float(data_raw["window_s"]),
            cache_dir=_as_path(data_raw["cache_dir"], base),
        ),
        split=SplitConfig(
            protocol=protocol,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            guard_s=float(split_raw["guard_s"]),
            train_sessions=tuple(split_raw["train_sessions"]),
            val_sessions=tuple(split_raw["val_sessions"]),
            test_sessions=tuple(split_raw["test_sessions"]),
        ),
        model=ModelConfig(
            in_channels=int(model_raw["in_channels"]),
            hidden_channels=int(model_raw["hidden_channels"]),
            kernel_size=int(model_raw["kernel_size"]),
            stride=stride,
            n_keypoints=int(model_raw["n_keypoints"]),
            coords=int(model_raw["coords"]),
            backbone=backbone,
            ndt3_repo=str(model_raw.get("ndt3_repo", "joel99/ndt3")),
            ndt3_file=str(
                model_raw.get(
                    "ndt3_file",
                    "753jmg4u/checkpoints/val-epoch=397-val_loss=0.4987.ckpt",
                )
            ),
            freeze_backbone=bool(model_raw.get("freeze_backbone", False)),
            n_layers=n_layers,
            n_heads=int(model_raw.get("n_heads", 8)),
            hidden_size=int(model_raw.get("hidden_size", 1024)),
            feedforward_factor=int(model_raw.get("feedforward_factor", 1)),
            dilations=dilations,
            dropout=dropout,
            head=head,
            grid_x=grid_x,
            grid_y=grid_y,
        ),
        train=TrainConfig(
            seed=int(train_raw["seed"]),
            batch_size=int(train_raw["batch_size"]),
            learning_rate=float(train_raw["learning_rate"]),
            weight_decay=float(train_raw["weight_decay"]),
            max_epochs=int(train_raw["max_epochs"]),
            early_stop_patience=int(train_raw["early_stop_patience"]),
            num_workers=int(train_raw["num_workers"]),
            device=str(train_raw["device"]),
            checkpoint_dir=_as_path(train_raw["checkpoint_dir"], base),
            time_shift_bins=int(train_raw["time_shift_bins"]),
            adv_weight=adv_weight,
        ),
    )
