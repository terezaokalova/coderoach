"""Train a neural-to-pose decoder from YAML config."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.config import SplitConfig, load_pose_decoder_config
from models.pose_dataset import (
    BODYPARTS,
    PoseSequenceDataset,
    build_split_datasets,
    collate_pose_batch,
)
from models.ndt3 import build_pose_model
from models.pose_decoder import (
    PoseDecoder,
    masked_pixel_rmse,
    masked_smooth_l1,
    per_keypoint_rmse,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(
    model: PoseDecoder,
    loader: DataLoader,
    device: torch.device,
    *,
    frame_width: float,
    frame_height: float,
) -> dict:
    model.eval()
    losses = []
    preds = []
    targets = []
    masks = []
    valids = []
    for batch in loader:
        neural = batch["neural"].to(device)
        pose = batch["pose"].to(device)
        pose_mask = batch["pose_mask"].to(device)
        pose_valid = batch["pose_valid"].to(device)
        pose_bin_idx = batch["pose_bin_idx"].to(device)
        pred = model(neural, pose_bin_idx, pose_valid)
        loss = masked_smooth_l1(pred, pose, pose_mask, pose_valid)
        losses.append(float(loss.item()))
        preds.append(pred.cpu())
        targets.append(pose.cpu())
        masks.append(pose_mask.cpu())
        valids.append(pose_valid.cpu())
    if not preds:
        return {"loss": float("nan"), "rmse_px": float("nan")}
    pred_all = torch.cat(preds)
    target_all = torch.cat(targets)
    mask_all = torch.cat(masks)
    valid_all = torch.cat(valids)
    metrics = masked_pixel_rmse(
        pred_all,
        target_all,
        mask_all,
        valid_all,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    metrics["loss"] = float(np.mean(losses))
    metrics["per_keypoint_rmse_px"] = {
        name: value
        for name, value in zip(
            BODYPARTS,
            per_keypoint_rmse(
                pred_all,
                target_all,
                mask_all,
                valid_all,
                frame_width=frame_width,
                frame_height=frame_height,
            ),
        )
    }
    return metrics


@torch.no_grad()
def mean_pose_baseline(
    train_ds: PoseSequenceDataset,
    loader: DataLoader,
    *,
    frame_width: float,
    frame_height: float,
) -> dict:
    xy = []
    mask = []
    for sess in train_ds.sessions:
        xy.append(sess.pose_xy)
        mask.append(sess.pose_mask)
    xy = np.concatenate(xy, axis=0)
    mask = np.concatenate(mask, axis=0)
    mean = np.zeros((xy.shape[1], 2), dtype=np.float32)
    for k in range(xy.shape[1]):
        m = mask[:, k]
        if m.any():
            mean[k] = xy[m, k].mean(axis=0)
    preds = []
    targets = []
    masks = []
    valids = []
    for batch in loader:
        pose = batch["pose"]
        pose_mask = batch["pose_mask"]
        pose_valid = batch["pose_valid"]
        pred = torch.from_numpy(mean)[None, None].expand_as(pose).clone()
        preds.append(pred)
        targets.append(pose)
        masks.append(pose_mask)
        valids.append(pose_valid)
    return masked_pixel_rmse(
        torch.cat(preds),
        torch.cat(targets),
        torch.cat(masks),
        torch.cat(valids),
        frame_width=frame_width,
        frame_height=frame_height,
    )


def train_one_epoch(
    model: PoseDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        neural = batch["neural"].to(device)
        pose = batch["pose"].to(device)
        pose_mask = batch["pose_mask"].to(device)
        pose_valid = batch["pose_valid"].to(device)
        pose_bin_idx = batch["pose_bin_idx"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(neural, pose_bin_idx, pose_valid)
        loss = masked_smooth_l1(pred, pose, pose_mask, pose_valid)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pose_decoder.yaml"),
    )
    parser.add_argument(
        "--protocol",
        choices=["temporal", "session"],
        default=None,
        help="Override split.protocol from the YAML config",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Few batches only")
    parser.add_argument(
        "--backbone",
        choices=["cnn", "ndt3"],
        default=None,
        help="Override model.backbone from the YAML config",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip downloading / loading the NDT3 checkpoint",
    )
    args = parser.parse_args(argv)

    cfg = load_pose_decoder_config(args.config)
    if args.backbone is not None or args.no_pretrained:
        model_kw: dict = {
            "backbone": args.backbone or cfg.model.backbone,
            "ndt3_file": "" if args.no_pretrained else cfg.model.ndt3_file,
        }
        if args.backbone == "ndt3" and cfg.model.backbone != "ndt3":
            model_kw.update(
                n_layers=6,
                n_heads=8,
                hidden_size=1024,
                feedforward_factor=1,
            )
        cfg = replace(cfg, model=replace(cfg.model, **model_kw))
    set_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    protocol = args.protocol or cfg.split.protocol
    splits = build_split_datasets(cfg, protocol=protocol)
    train_ds = splits["train"]
    val_ds = splits["val"]
    test_ds = splits["test"]
    if len(train_ds) == 0:
        raise SystemExit("train split is empty")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_pose_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_pose_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_pose_batch,
    )

    model = build_pose_model(cfg.model).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("no trainable parameters")
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    ckpt_dir = cfg.train.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"best_{protocol}.pt"
    best_val = float("inf")
    patience = 0
    max_epochs = args.max_epochs or cfg.train.max_epochs
    if args.smoke:
        max_epochs = 1

    history = []
    for epoch in range(1, max_epochs + 1):
        if args.smoke:
            # one optimizer step path via truncated loader
            batch = next(iter(train_loader))
            neural = batch["neural"].to(device)
            pose = batch["pose"].to(device)
            pose_mask = batch["pose_mask"].to(device)
            pose_valid = batch["pose_valid"].to(device)
            pose_bin_idx = batch["pose_bin_idx"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(neural, pose_bin_idx, pose_valid)
            loss = masked_smooth_l1(pred, pose, pose_mask, pose_valid)
            loss.backward()
            optimizer.step()
            train_loss = float(loss.item())
        else:
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            frame_width=cfg.data.frame_width,
            frame_height=cfg.data.frame_height,
        )
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(
            f"epoch {epoch} train_loss={train_loss:.4f} "
            f"val_rmse_px={val_metrics['rmse_px']:.2f} "
            f"val_corr={val_metrics['corr']:.3f}"
        )
        if val_metrics["rmse_px"] < best_val:
            best_val = val_metrics["rmse_px"]
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg_path": str(args.config),
                    "backbone": cfg.model.backbone,
                    "protocol": protocol,
                    "neural_mean": train_ds.neural_mean,
                    "neural_std": train_ds.neural_std,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience and not args.smoke:
                print(f"early stop at epoch {epoch}")
                break
        if args.smoke:
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        frame_width=cfg.data.frame_width,
        frame_height=cfg.data.frame_height,
    )
    baseline = mean_pose_baseline(
        train_ds,
        test_loader,
        frame_width=cfg.data.frame_width,
        frame_height=cfg.data.frame_height,
    )
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
    shifted = PoseSequenceDataset(
        test_ds.sessions,
        window_s=cfg.data.window_s,
        bin_ms=cfg.data.bin_ms,
        split_name="test",
        split_cfg=split_cfg,
        neural_mean=train_ds.neural_mean,
        neural_std=train_ds.neural_std,
        time_shift_bins=cfg.train.time_shift_bins,
    )
    shift_loader = DataLoader(
        shifted,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_pose_batch,
    )
    shift_metrics = evaluate(
        model,
        shift_loader,
        device,
        frame_width=cfg.data.frame_width,
        frame_height=cfg.data.frame_height,
    )
    summary = {
        "protocol": protocol,
        "train_sessions": (
            list(dict.fromkeys([*cfg.split.train_sessions, *cfg.split.val_sessions]))
            if protocol == "session"
            else list(cfg.split.test_sessions or (cfg.data.sessions[-1],))
        ),
        "val_sessions": (
            list(dict.fromkeys([*cfg.split.train_sessions, *cfg.split.val_sessions]))
            if protocol == "session"
            else list(cfg.split.test_sessions or (cfg.data.sessions[-1],))
        ),
        "test_sessions": (
            list(cfg.split.test_sessions)
            if protocol == "session"
            else list(cfg.split.test_sessions or (cfg.data.sessions[-1],))
        ),
        "val_is_temporal_holdout": protocol == "session",
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "test": test_metrics,
        "mean_pose_baseline": baseline,
        "time_shifted_control": shift_metrics,
        "checkpoint": str(best_path),
        "history": history,
    }
    out_path = ckpt_dir / f"summary_{protocol}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n=== held-out test ===")
    print(f"test_sessions: {summary['test_sessions']}")
    print(
        f"rmse_px={test_metrics['rmse_px']:.2f}  "
        f"rmse_norm={test_metrics['rmse_norm']:.4f}  "
        f"corr={test_metrics['corr']:.3f}  "
        f"loss={test_metrics['loss']:.4f}"
    )
    print(
        f"mean_pose_baseline rmse_px={baseline['rmse_px']:.2f}  "
        f"time_shift_control rmse_px={shift_metrics['rmse_px']:.2f}"
    )
    print(f"wrote {out_path}")
    print(json.dumps({k: summary[k] for k in summary if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
