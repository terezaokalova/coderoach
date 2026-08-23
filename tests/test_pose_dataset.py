"""Synthetic tests for pose dataset, splits, and decoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from models.config import SplitConfig, load_pose_decoder_config
from models.ndt3 import NDT3PoseDecoder, load_ndt3_weights
from models.pose_dataset import (
    PoseSequenceDataset,
    SessionArrays,
    collate_pose_batch,
    session_ranges,
    temporal_ranges,
    window_starts_in_range,
)
from models.pose_decoder import PoseDecoder, masked_smooth_l1
from models.preprocess import cache_path
from models.train_pose_decoder import evaluate, train_mean_pose


def _synthetic_session(
    *,
    session: str = "synth.zarr",
    n_bins: int = 500,
    n_ch: int = 32,
    n_pose: int = 200,
    bin_ms: float = 20.0,
) -> SessionArrays:
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.5, size=(n_bins, n_ch)).astype(np.float32)
    centers = (np.arange(n_bins) + 0.5) * (bin_ms / 1000.0)
    pose_bin_idx = np.linspace(0, n_bins - 1, n_pose).astype(np.int64)
    pose_xy = rng.random((n_pose, 15, 2), dtype=np.float32)
    pose_mask = rng.random((n_pose, 15)) > 0.2
    pose_t_s = pose_bin_idx * (bin_ms / 1000.0)
    pose_ns = (pose_t_s * 30000).astype(np.int64)
    return SessionArrays(
        session=session,
        counts=counts,
        bin_centers_s=centers.astype(np.float32),
        fs=30000.0,
        pose_xy=pose_xy,
        pose_mask=pose_mask,
        pose_t_s=pose_t_s.astype(np.float32),
        pose_neural_sample=pose_ns,
        pose_bin_idx=pose_bin_idx,
    )


def test_load_pose_decoder_config(tmp_path: Path):
    src = Path("configs/pose_decoder.yaml")
    cfg = load_pose_decoder_config(src)
    assert cfg.model.hidden_channels == 64
    assert cfg.model.kernel_size == 10
    assert cfg.model.stride == 1
    assert cfg.model.n_layers == 3
    assert cfg.model.dilations == (1, 2, 4)
    assert cfg.model.dropout == 0.2
    assert cfg.train.adv_weight == 0.1
    assert cfg.split.protocol == "session"
    assert cfg.split.test_frac == 0.0
    assert cfg.split.test_sessions == ("SD11_rec_20260820_182553.zarr",)
    assert cfg.model.backbone == "cnn"


def test_load_ndt3_config():
    cfg = load_pose_decoder_config(Path("configs/pose_decoder_ndt3.yaml"))
    assert cfg.model.backbone == "ndt3"
    assert cfg.model.n_layers == 6
    assert cfg.model.stride == 1
    assert cfg.model.hidden_size == 1024
    assert cfg.model.ndt3_repo == "joel99/ndt3"
    assert "753jmg4u" in cfg.model.ndt3_file


def test_config_rejects_unknown_key(tmp_path: Path):
    raw = yaml.safe_load(Path("configs/pose_decoder.yaml").read_text())
    raw["data"]["extra"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    try:
        load_pose_decoder_config(path)
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "unknown keys" in str(exc)


def test_temporal_ranges_train_val_only():
    split = SplitConfig(
        protocol="session",
        train_frac=0.85,
        val_frac=0.15,
        test_frac=0.0,
        guard_s=2.0,
        train_sessions=(),
        val_sessions=(),
        test_sessions=(),
    )
    ranges = temporal_ranges(100.0, split)
    assert set(ranges) == {"train", "val"}
    assert "test" not in ranges
    assert ranges["train"].start_s == 2.0
    assert ranges["val"].start_s >= ranges["train"].end_s + 2.0 - 1e-9
    assert ranges["val"].end_s == 98.0


def test_session_protocol_val_is_temporal_not_recording():
    sess = _synthetic_session(n_bins=2000)
    split = SplitConfig(
        protocol="session",
        train_frac=0.85,
        val_frac=0.15,
        test_frac=0.0,
        guard_s=2.0,
        train_sessions=(sess.session,),
        val_sessions=(),
        test_sessions=("other.zarr",),
    )
    train_ds = PoseSequenceDataset(
        [sess],
        window_s=2.0,
        bin_ms=20.0,
        split_name="train",
        split_cfg=split,
    )
    val_ds = PoseSequenceDataset(
        [sess],
        window_s=2.0,
        bin_ms=20.0,
        split_name="val",
        split_cfg=split,
        neural_mean=train_ds.neural_mean,
        neural_std=train_ds.neural_std,
    )
    assert len(train_ds) > 0 and len(val_ds) > 0
    train_starts = {start for _, start in train_ds.items}
    val_starts = {start for _, start in val_ds.items}
    assert train_starts.isdisjoint(val_starts)
    assert max(train_starts) < min(val_starts)


def test_session_protocol_splits_test_session_in_half():
    train_sess = _synthetic_session(session="train.zarr", n_bins=4000, n_pose=400)
    test_sess = _synthetic_session(session="test.zarr", n_bins=4000, n_pose=400)
    train_sess.pose_xy[:] = 0.1
    train_sess.pose_mask[:] = True
    test_sess.pose_xy[test_sess.pose_t_s >= 40.0] = 0.9
    test_sess.pose_xy[test_sess.pose_t_s < 40.0] = 0.1
    test_sess.pose_mask[:] = True
    split = SplitConfig(
        protocol="session",
        train_frac=0.85,
        val_frac=0.15,
        test_frac=0.0,
        guard_s=2.0,
        train_sessions=(train_sess.session,),
        val_sessions=(),
        test_sessions=(test_sess.session,),
    )
    ranges = session_ranges(80.0, split, is_test_session=True)
    assert ranges["train"].end_s <= 40.0
    assert ranges["val"].end_s <= 40.0
    assert ranges["test"].start_s >= 40.0
    train_ds = PoseSequenceDataset(
        [train_sess, test_sess],
        window_s=2.0,
        bin_ms=20.0,
        split_name="train",
        split_cfg=split,
    )
    val_ds = PoseSequenceDataset(
        [train_sess, test_sess],
        window_s=2.0,
        bin_ms=20.0,
        split_name="val",
        split_cfg=split,
        neural_mean=train_ds.neural_mean,
        neural_std=train_ds.neural_std,
    )
    test_ds = PoseSequenceDataset(
        [test_sess],
        window_s=2.0,
        bin_ms=20.0,
        split_name="test",
        split_cfg=split,
        neural_mean=train_ds.neural_mean,
        neural_std=train_ds.neural_std,
    )
    dt = 0.02
    window_s = 2.0

    def _window_times(ds: PoseSequenceDataset, session: str) -> list[tuple[float, float]]:
        out = []
        for si, start in ds.items:
            if ds.sessions[si].session != session:
                continue
            out.append((start * dt, start * dt + window_s))
        return out

    train_times = _window_times(train_ds, "test.zarr")
    val_times = _window_times(val_ds, "test.zarr")
    test_times = _window_times(test_ds, "test.zarr")
    assert train_times and val_times and test_times
    assert max(end for _, end in train_times + val_times) <= 40.0
    assert min(start for start, _ in test_times) >= 40.0
    mean = train_mean_pose(train_ds)
    assert float(mean.mean()) < 0.3


def test_temporal_ranges_have_guards():
    split = SplitConfig(
        protocol="temporal",
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        guard_s=2.0,
        train_sessions=(),
        val_sessions=(),
        test_sessions=(),
    )
    ranges = temporal_ranges(100.0, split)
    assert ranges["train"].start_s == 2.0
    assert ranges["val"].start_s >= ranges["train"].end_s + 2.0 - 1e-9
    assert ranges["test"].start_s >= ranges["val"].end_s + 2.0 - 1e-9
    assert ranges["test"].end_s == 98.0


def test_windows_do_not_cross_split_boundaries():
    split = SplitConfig(
        protocol="temporal",
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        guard_s=2.0,
        train_sessions=(),
        val_sessions=(),
        test_sessions=(),
    )
    bin_ms = 20.0
    window_bins = 100
    n_bins = 5000
    duration_s = n_bins * bin_ms / 1000.0
    ranges = temporal_ranges(duration_s, split)
    for name, tr in ranges.items():
        starts = window_starts_in_range(
            n_bins=n_bins,
            window_bins=window_bins,
            bin_ms=bin_ms,
            time_range=tr,
        )
        dt = bin_ms / 1000.0
        assert starts.size > 0, name
        assert np.all(starts * dt >= tr.start_s)
        assert np.all((starts + window_bins) * dt <= tr.end_s)


def test_dataset_returns_aligned_tensors():
    sess = _synthetic_session()
    split = SplitConfig(
        protocol="session",
        train_frac=0.85,
        val_frac=0.15,
        test_frac=0.0,
        guard_s=2.0,
        train_sessions=("synth.zarr",),
        val_sessions=(),
        test_sessions=("other.zarr",),
    )
    ds = PoseSequenceDataset(
        [sess],
        window_s=2.0,
        bin_ms=20.0,
        split_name="train",
        split_cfg=split,
    )
    assert len(ds) > 0
    item = ds[0]
    assert item["neural"].shape == (100, 32)
    assert item["pose"].ndim == 3
    assert item["pose"].shape[1:] == (15, 2)
    assert item["pose_bin_idx"].min() >= 0
    assert item["pose_bin_idx"].max() < 100
    batch = collate_pose_batch([ds[0], ds[1]])
    assert batch["neural"].shape[0] == 2
    assert batch["pose_valid"].dtype == torch.bool


def test_spike_cache_roundtrip(tmp_path: Path):
    counts = np.arange(12, dtype=np.float32).reshape(4, 3)
    centers = np.array([0.01, 0.03, 0.05, 0.07], dtype=np.float32)
    path = cache_path(tmp_path, "demo.zarr", 20.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        counts=counts,
        bin_centers_s=centers,
        fs=np.float64(30000.0),
        channel_names=np.asarray(["a", "b", "c"]),
    )
    loaded = np.load(path)
    assert loaded["counts"].shape == (4, 3)
    assert float(loaded["fs"]) == 30000.0


def test_decoder_shapes_loss_and_checkpoint(tmp_path: Path):
    model = PoseDecoder(
        in_channels=32,
        hidden_channels=64,
        kernel_size=10,
        stride=1,
        n_layers=2,
        n_keypoints=15,
        coords=2,
    )
    neural = torch.randn(4, 100, 32)
    pose_bin_idx = torch.randint(0, 100, (4, 8))
    pose_valid = torch.ones(4, 8, dtype=torch.bool)
    pose = torch.randn(4, 8, 15, 2)
    pose_mask = torch.ones(4, 8, 15)
    pred = model(neural, pose_bin_idx, pose_valid)
    assert pred.shape == (4, 8, 15, 2)
    loss = masked_smooth_l1(pred, pose, pose_mask, pose_valid)
    loss.backward()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()
    with torch.no_grad():
        pred_after = model(neural, pose_bin_idx, pose_valid)
    ckpt = tmp_path / "m.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    restored = PoseDecoder(
        in_channels=32,
        hidden_channels=64,
        kernel_size=10,
        stride=1,
        n_layers=2,
        n_keypoints=15,
        coords=2,
    )
    restored.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    )
    with torch.no_grad():
        pred2 = restored(neural, pose_bin_idx, pose_valid)
    assert torch.allclose(pred_after, pred2, atol=1e-5)

    strided = PoseDecoder(
        in_channels=32,
        hidden_channels=64,
        kernel_size=10,
        stride=2,
        n_layers=3,
        n_keypoints=15,
        coords=2,
    )
    pred_s = strided(neural, pose_bin_idx, pose_valid)
    assert pred_s.shape == (4, 8, 15, 2)
    assert strided.time_stride == 8
    assert strided.dilations == (1, 2, 4)


def test_residual_mean_and_session_adversary():
    model = PoseDecoder(
        in_channels=32,
        hidden_channels=16,
        kernel_size=5,
        stride=1,
        n_layers=3,
        dilations=(1, 2, 4),
        n_keypoints=15,
        coords=2,
        n_sessions=2,
    )
    neural = torch.randn(3, 40, 32)
    pose_bin_idx = torch.randint(0, 40, (3, 5))
    pose_valid = torch.ones(3, 5, dtype=torch.bool)
    with torch.no_grad():
        base = model(neural, pose_bin_idx, pose_valid)
        model.pose_mean.fill_(0.25)
        shifted = model(neural, pose_bin_idx, pose_valid)
    assert torch.allclose(shifted, base + 0.25, atol=1e-5)
    logits = model.session_logits(neural, 1.0)
    assert logits is not None and logits.shape == (3, 2)
    labels = torch.tensor([0, 1, 0])
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    conv_grad = next(p.grad for p in model.encoder.parameters() if p.grad is not None)
    assert conv_grad.abs().sum() > 0


def test_evaluate_concat_variable_pose_counts():
    model = PoseDecoder(
        in_channels=32,
        hidden_channels=8,
        kernel_size=3,
        stride=1,
        n_layers=1,
        n_keypoints=15,
        coords=2,
    )
    batches = [
        {
            "neural": torch.randn(2, 20, 32),
            "pose": torch.rand(2, 20, 15, 2),
            "pose_mask": torch.ones(2, 20, 15),
            "pose_valid": torch.ones(2, 20, dtype=torch.bool),
            "pose_bin_idx": torch.randint(0, 20, (2, 20)),
        },
        {
            "neural": torch.randn(1, 20, 32),
            "pose": torch.rand(1, 19, 15, 2),
            "pose_mask": torch.ones(1, 19, 15),
            "pose_valid": torch.ones(1, 19, dtype=torch.bool),
            "pose_bin_idx": torch.randint(0, 20, (1, 19)),
        },
    ]
    metrics = evaluate(
        model,
        batches,
        torch.device("cpu"),
        frame_width=1280.0,
        frame_height=1024.0,
    )
    assert np.isfinite(metrics["rmse_px"])


def test_ndt3_decoder_shapes_and_weight_map(tmp_path: Path):
    model = NDT3PoseDecoder(
        in_channels=32,
        n_layers=1,
        n_heads=4,
        hidden_size=32,
        n_keypoints=15,
        coords=2,
    )
    neural = torch.randint(0, 5, (2, 8, 32)).float()
    pose_bin_idx = torch.randint(0, 8, (2, 3))
    pose_valid = torch.ones(2, 3, dtype=torch.bool)
    pred = model(neural, pose_bin_idx, pose_valid)
    assert pred.shape == (2, 3, 15, 2)
    loss = masked_smooth_l1(
        pred,
        torch.randn(2, 3, 15, 2),
        torch.ones(2, 3, 15),
        pose_valid,
    )
    loss.backward()
    class _Upstream:
        pass

    _Upstream.__module__ = "context_general_bci.config"
    fake = {
        "hyper_parameters": _Upstream(),
        "state_dict": {
            "task_pipelines.spike_infill.readin.weight": torch.randn_like(
                model.spike_embed.weight
            ),
            "backbone.layers.0.mixer.Wqkv.weight": torch.randn_like(
                model.layers[0].attn.in_proj_weight
            ),
            "backbone.layers.0.mixer.out_proj.weight": torch.randn_like(
                model.layers[0].attn.out_proj.weight
            ),
            "backbone.layers.0.mlp.fc1.weight": torch.randn_like(
                model.layers[0].mlp.fc1.weight
            ),
            "backbone.layers.0.mlp.fc2.weight": torch.randn_like(
                model.layers[0].mlp.fc2.weight
            ),
        }
    }
    path = tmp_path / "ndt3.ckpt"
    torch.save(fake, path)
    stats = load_ndt3_weights(model, path)
    assert stats["loaded"] == 5
