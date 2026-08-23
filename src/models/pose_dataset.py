"""Aligned neural spike-count / pose sequence dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.config import PoseDecoderConfig, SplitConfig
from models.preprocess import HTTP_HEADERS, build_or_load_spike_cache, load_data_info

BODYPARTS = (
    "nose",
    "eye",
    "ear_base",
    "neck_base",
    "back_base",
    "back_middle",
    "back_end",
    "tail_base",
    "tail_end",
    "forelimb_proximal",
    "forelimb_middle",
    "forepaw",
    "hindlimb_proximal",
    "hindlimb_middle",
    "hindpaw",
)


@dataclass(frozen=True)
class TimeRange:
    start_s: float
    end_s: float


@dataclass
class SessionArrays:
    session: str
    counts: np.ndarray  # [T, C]
    bin_centers_s: np.ndarray
    fs: float
    pose_xy: np.ndarray  # [P, K, 2]
    pose_mask: np.ndarray  # [P, K]
    pose_t_s: np.ndarray  # [P]
    pose_neural_sample: np.ndarray  # [P]
    pose_bin_idx: np.ndarray  # [P]


def load_pose_table(
    base_url: str,
    session: str,
    *,
    camera: str,
    frame_width: int,
    frame_height: int,
    likelihood_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    url = f"{base_url.rstrip('/')}/{session}/derived/pose.parquet"
    pose = pd.read_parquet(url, storage_options=HTTP_HEADERS)
    rows = pose[(pose["camera"] == camera) & (pose["neural_sample"] >= 0)].copy()
    rows = rows.sort_values("neural_sample").reset_index(drop=True)
    xy = np.stack(
        [
            np.stack(
                [
                    rows[f"{part}_x"].to_numpy(dtype=np.float32) / frame_width,
                    rows[f"{part}_y"].to_numpy(dtype=np.float32) / frame_height,
                ],
                axis=-1,
            )
            for part in BODYPARTS
        ],
        axis=1,
    )
    likelihood = np.stack(
        [rows[f"{part}_likelihood"].to_numpy(dtype=np.float32) for part in BODYPARTS],
        axis=1,
    )
    finite = np.isfinite(xy).all(axis=-1)
    mask = (likelihood >= likelihood_threshold) & finite
    t_s = rows["t_s"].to_numpy(dtype=np.float32)
    neural_sample = rows["neural_sample"].to_numpy(dtype=np.int64)
    return xy, mask, t_s, neural_sample


def load_session_arrays(cfg: PoseDecoderConfig, session: str) -> SessionArrays:
    info = load_data_info(cfg.data.info_yaml)
    base_url = info["base_url"]
    cache = build_or_load_spike_cache(
        base_url=base_url,
        session=session,
        bin_ms=cfg.data.bin_ms,
        cache_dir=cfg.data.cache_dir,
    )
    pose_xy, pose_mask, pose_t_s, pose_neural = load_pose_table(
        base_url,
        session,
        camera=cfg.data.camera,
        frame_width=cfg.data.frame_width,
        frame_height=cfg.data.frame_height,
        likelihood_threshold=cfg.data.likelihood_threshold,
    )
    fs = float(cache["fs"])
    bin_ms = cfg.data.bin_ms
    pose_bin_idx = np.clip(
        np.floor(pose_neural / fs / (bin_ms / 1000.0)).astype(np.int64),
        0,
        cache["counts"].shape[0] - 1,
    )
    return SessionArrays(
        session=session,
        counts=cache["counts"],
        bin_centers_s=cache["bin_centers_s"],
        fs=fs,
        pose_xy=pose_xy,
        pose_mask=pose_mask,
        pose_t_s=pose_t_s,
        pose_neural_sample=pose_neural,
        pose_bin_idx=pose_bin_idx,
    )


def temporal_ranges(
    duration_s: float,
    split: SplitConfig,
) -> dict[str, TimeRange]:
    """Chronological blocks with guard gaps.

    If test_frac is 0, returns only train and val.
    """
    guard = split.guard_s
    if split.test_frac <= 0:
        usable = duration_s - 3 * guard
        if usable <= 0:
            raise ValueError(f"duration {duration_s} too short for guard {guard}")
        train_end = guard + split.train_frac * usable
        val_start = train_end + guard
        return {
            "train": TimeRange(guard, train_end),
            "val": TimeRange(val_start, duration_s - guard),
        }
    usable = duration_s - 2 * guard
    if usable <= 0:
        raise ValueError(f"duration {duration_s} too short for guard {guard}")
    train_end = guard + split.train_frac * usable
    val_end = train_end + guard + split.val_frac * usable
    test_start = val_end + guard
    return {
        "train": TimeRange(guard, train_end),
        "val": TimeRange(train_end + guard, val_end),
        "test": TimeRange(test_start, duration_s - guard),
    }


def session_ranges(
    duration_s: float,
    split: SplitConfig,
    *,
    is_test_session: bool,
) -> dict[str, TimeRange]:
    """Train-pool sessions are chronological train/val.

    Test sessions use the first half for train/val and the second half for test.
    """
    if not is_test_session:
        return temporal_ranges(duration_s, split)
    half = 0.5 * duration_s
    ranges = temporal_ranges(half, split)
    ranges["test"] = TimeRange(half + split.guard_s, duration_s - split.guard_s)
    return ranges


def window_starts_in_range(
    *,
    n_bins: int,
    window_bins: int,
    bin_ms: float,
    time_range: TimeRange | None,
) -> np.ndarray:
    if window_bins > n_bins:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(0, n_bins - window_bins + 1, dtype=np.int64)
    if time_range is None:
        return starts
    dt = bin_ms / 1000.0
    start_s = starts * dt
    end_s = (starts + window_bins) * dt
    keep = (start_s >= time_range.start_s) & (end_s <= time_range.end_s)
    return starts[keep]


class PoseSequenceDataset(Dataset):
    """Windows of spike counts paired with pose rows whose bins fall inside."""

    def __init__(
        self,
        sessions: list[SessionArrays],
        *,
        window_s: float,
        bin_ms: float,
        split_name: str,
        split_cfg: SplitConfig,
        neural_mean: np.ndarray | None = None,
        neural_std: np.ndarray | None = None,
        time_shift_bins: int = 0,
        normalize_neural: bool = True,
    ) -> None:
        self.window_bins = int(round(window_s / (bin_ms / 1000.0)))
        self.bin_ms = bin_ms
        self.split_name = split_name
        self.time_shift_bins = int(time_shift_bins)
        self.items: list[tuple[int, int]] = []
        self.sessions = sessions
        for si, sess in enumerate(sessions):
            duration_s = float(sess.counts.shape[0] * bin_ms / 1000.0)
            if split_cfg.protocol == "temporal":
                ranges = temporal_ranges(duration_s, split_cfg)
            elif split_cfg.protocol == "session":
                ranges = session_ranges(
                    duration_s,
                    split_cfg,
                    is_test_session=sess.session in split_cfg.test_sessions,
                )
            else:
                raise ValueError(f"unknown protocol {split_cfg.protocol!r}")
            tr = ranges.get(split_name)
            if tr is None:
                continue
            starts = window_starts_in_range(
                n_bins=sess.counts.shape[0],
                window_bins=self.window_bins,
                bin_ms=bin_ms,
                time_range=tr,
            )
            for start in starts:
                end = start + self.window_bins
                if np.any((sess.pose_bin_idx >= start) & (sess.pose_bin_idx < end)):
                    self.items.append((si, int(start)))
        if neural_mean is None:
            if normalize_neural:
                neural_mean, neural_std = self._fit_neural_stats()
            else:
                n_ch = sessions[0].counts.shape[1]
                neural_mean = np.zeros(n_ch, dtype=np.float32)
                neural_std = np.ones(n_ch, dtype=np.float32)
        self.neural_mean = neural_mean.astype(np.float32)
        self.neural_std = np.maximum(neural_std.astype(np.float32), 1e-6)

    def _fit_neural_stats(self) -> tuple[np.ndarray, np.ndarray]:
        chunks = []
        for si, start in self.items:
            chunks.append(self.sessions[si].counts[start : start + self.window_bins])
        if not chunks:
            n_ch = self.sessions[0].counts.shape[1]
            return np.zeros(n_ch, dtype=np.float32), np.ones(n_ch, dtype=np.float32)
        stacked = np.concatenate(chunks, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        si, start = self.items[index]
        sess = self.sessions[si]
        end = start + self.window_bins
        neural = sess.counts[start:end].astype(np.float32)
        if self.time_shift_bins:
            shift = self.time_shift_bins
            src = np.roll(sess.counts, shift, axis=0)[start:end].astype(np.float32)
            neural = src
        neural = (neural - self.neural_mean) / self.neural_std
        pose_sel = (sess.pose_bin_idx >= start) & (sess.pose_bin_idx < end)
        pose_xy = sess.pose_xy[pose_sel]
        pose_mask = sess.pose_mask[pose_sel]
        pose_bins = sess.pose_bin_idx[pose_sel] - start
        pose_t = sess.pose_t_s[pose_sel]
        pose_ns = sess.pose_neural_sample[pose_sel]
        return {
            "neural": torch.from_numpy(neural),  # [T, C]
            "pose": torch.from_numpy(pose_xy),  # [P, K, 2]
            "pose_mask": torch.from_numpy(pose_mask.astype(np.float32)),
            "pose_bin_idx": torch.from_numpy(pose_bins.astype(np.int64)),
            "pose_t_s": torch.from_numpy(pose_t),
            "neural_sample": torch.from_numpy(pose_ns),
            "session": sess.session,
            "window_start_bin": start,
        }


def collate_pose_batch(batch: list[dict]) -> dict[str, torch.Tensor | list]:
    max_p = max(item["pose"].shape[0] for item in batch)
    n = len(batch)
    t, c = batch[0]["neural"].shape
    k = batch[0]["pose"].shape[1]
    neural = torch.stack([item["neural"] for item in batch])
    pose = torch.zeros(n, max_p, k, 2, dtype=torch.float32)
    pose_mask = torch.zeros(n, max_p, k, dtype=torch.float32)
    pose_bin_idx = torch.zeros(n, max_p, dtype=torch.int64)
    pose_valid = torch.zeros(n, max_p, dtype=torch.bool)
    pose_t_s = torch.zeros(n, max_p, dtype=torch.float32)
    neural_sample = torch.full((n, max_p), -1, dtype=torch.int64)
    for i, item in enumerate(batch):
        p = item["pose"].shape[0]
        pose[i, :p] = item["pose"]
        pose_mask[i, :p] = item["pose_mask"]
        pose_bin_idx[i, :p] = item["pose_bin_idx"]
        pose_valid[i, :p] = True
        pose_t_s[i, :p] = item["pose_t_s"]
        neural_sample[i, :p] = item["neural_sample"]
    return {
        "neural": neural,
        "pose": pose,
        "pose_mask": pose_mask,
        "pose_bin_idx": pose_bin_idx,
        "pose_valid": pose_valid,
        "pose_t_s": pose_t_s,
        "neural_sample": neural_sample,
        "session": [item["session"] for item in batch],
        "window_start_bin": torch.tensor(
            [item["window_start_bin"] for item in batch], dtype=torch.int64
        ),
    }


def build_split_datasets(
    cfg: PoseDecoderConfig,
    *,
    protocol: str | None = None,
    normalize_neural: bool | None = None,
) -> dict[str, PoseSequenceDataset]:
    protocol = protocol or cfg.split.protocol
    if normalize_neural is None:
        normalize_neural = cfg.model.backbone != "ndt3"
    split_cfg = SplitConfig(
        protocol=protocol,
        train_frac=cfg.split.train_frac,
        val_frac=cfg.split.val_frac,
        test_frac=cfg.split.test_frac,
        guard_s=cfg.split.guard_s,
        train_sessions=cfg.split.train_sessions,
        val_sessions=cfg.split.val_sessions,
        test_sessions=cfg.split.test_sessions,
    )
    if protocol == "temporal":
        # Prefer the held-out final session id when listed; else last catalog entry.
        session_ids = (
            list(split_cfg.test_sessions)
            if split_cfg.test_sessions
            else [cfg.data.sessions[-1]]
        )
        arrays = [load_session_arrays(cfg, session_ids[0])]
        train = PoseSequenceDataset(
            arrays,
            window_s=cfg.data.window_s,
            bin_ms=cfg.data.bin_ms,
            split_name="train",
            split_cfg=split_cfg,
            normalize_neural=normalize_neural,
        )
        val = PoseSequenceDataset(
            arrays,
            window_s=cfg.data.window_s,
            bin_ms=cfg.data.bin_ms,
            split_name="val",
            split_cfg=split_cfg,
            neural_mean=train.neural_mean,
            neural_std=train.neural_std,
        )
        test = PoseSequenceDataset(
            arrays,
            window_s=cfg.data.window_s,
            bin_ms=cfg.data.bin_ms,
            split_name="test",
            split_cfg=split_cfg,
            neural_mean=train.neural_mean,
            neural_std=train.neural_std,
        )
        return {"train": train, "val": val, "test": test}

    if protocol != "session":
        raise ValueError(f"unknown protocol {protocol!r}")

    def load_many(ids: tuple[str, ...]) -> list[SessionArrays]:
        return [load_session_arrays(cfg, sid) for sid in ids]

    # Optional extra pool sessions, plus test-session first half for train/val.
    # Empty train_sessions means same-session only: first half train/val, second half test.
    pool_ids = tuple(
        dict.fromkeys([*split_cfg.train_sessions, *split_cfg.val_sessions])
    )
    if not split_cfg.test_sessions:
        raise ValueError(
            "session protocol needs test_sessions for the half-session holdout"
        )
    overlap = set(pool_ids) & set(split_cfg.test_sessions)
    if overlap:
        raise ValueError(
            f"test sessions must be disjoint from train/val pool: {overlap}"
        )

    pool_arrays = load_many(pool_ids)
    test_arrays = load_many(split_cfg.test_sessions)
    train_arrays = [*pool_arrays, *test_arrays]
    train = PoseSequenceDataset(
        train_arrays,
        window_s=cfg.data.window_s,
        bin_ms=cfg.data.bin_ms,
        split_name="train",
        split_cfg=split_cfg,
        normalize_neural=normalize_neural,
    )
    val = PoseSequenceDataset(
        train_arrays,
        window_s=cfg.data.window_s,
        bin_ms=cfg.data.bin_ms,
        split_name="val",
        split_cfg=split_cfg,
        neural_mean=train.neural_mean,
        neural_std=train.neural_std,
    )
    test = PoseSequenceDataset(
        test_arrays,
        window_s=cfg.data.window_s,
        bin_ms=cfg.data.bin_ms,
        split_name="test",
        split_cfg=split_cfg,
        neural_mean=train.neural_mean,
        neural_std=train.neural_std,
    )
    return {"train": train, "val": val, "test": test}
