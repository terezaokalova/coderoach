# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import zarr
from scipy.signal import butter, sosfiltfilt

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [here, *here.parents] if (p / "data_info.yaml").exists())
info = yaml.safe_load((ROOT / "data_info.yaml").read_text())
DATA = info["base_url"]
T0, WIN = 10.0, 15  # seconds; WIN = None is the rest of the session from T0
BIN_MS = 20.0  # None = raster; else spike counts per bin

sessions = [
    (subject, next(s["id"] for s in meta["sessions"] if s.get("use") and s.get("id")))
    for subject, meta in info["subjects"].items()
]


def raster_window(root, t0, win):
    traces = root["traces"]
    fs = float(root.attrs["sample_rate_hz"])
    gain = float(traces.attrs.get("gain_uv_per_count", 0.195))
    names = list(traces.attrs["channel_names"])
    dur = float(root.attrs.get("duration_s", traces.shape[1] / fs))
    if win is None:
        win = max(0.0, dur - t0)
    t0 = min(t0, max(0.0, dur - win))
    i0, i1 = int(t0 * fs), int((t0 + win) * fs)
    step = int(fs)
    raw = np.concatenate(
        [np.asarray(traces[:, i : min(i + step, i1)]) for i in range(i0, i1, step)],
        axis=1,
    )
    rail = np.abs(raw) >= 30000
    uv = raw.astype(np.float64)
    idx = np.arange(uv.shape[1])
    for ch in range(uv.shape[0]):
        good = ~rail[ch]
        if good.sum() > 10:
            uv[ch, rail[ch]] = np.interp(idx[rail[ch]], idx[good], uv[ch, good])
    bp = sosfiltfilt(
        butter(3, (300, 7500), btype="bandpass", fs=fs, output="sos"),
        uv * gain,
        axis=1,
    )
    refrac = int(0.001 * fs)
    guard = int(0.002 * fs)
    spikes = []
    for ch, y in enumerate(bp):
        thr = -3.5 * max(float(np.sqrt(np.mean(y * y))), 1e-6)
        crossings = np.flatnonzero((y[1:] < thr) & (y[:-1] >= thr)) + 1
        keep = []
        last = -refrac
        for s in crossings:
            if s - last < refrac:
                continue
            lo, hi = max(0, s - guard), min(y.size, s + guard)
            if rail[ch, lo:hi].any():
                continue
            keep.append(s)
            last = s
        spikes.append(np.asarray(keep, dtype=float) / fs + t0)
    return names, spikes, t0, win


# %% raster per subject (threshold crossings; derived/spikes.npz is unpublished)
for subject, rec in sessions:
    root = zarr.open_group(
        zarr.storage.FsspecStore.from_url(f"{DATA}/{rec}", read_only=True),
        mode="r",
    )
    names, spikes, t0, win = raster_window(root, T0, WIN)
    print(
        subject,
        rec,
        "n spikes",
        [int(s.size) for s in spikes],
        "total",
        int(sum(s.size for s in spikes)),
    )
    if BIN_MS is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.eventplot(spikes, colors="k", linelengths=0.7, linewidths=0.6)
        kind = "raster"
    else:
        dt = BIN_MS / 1000.0
        edges = t0 + np.arange(int(np.round(win / dt)) + 1) * dt
        counts = np.stack([np.histogram(s, bins=edges)[0] for s in spikes])
        print(
            "counts",
            counts.shape,
            "bin_ms",
            BIN_MS,
            "max / bin",
            int(counts.max()),
            "sum",
            int(counts.sum()),
        )
        fig, (ax, axp) = plt.subplots(
            2, 1, figsize=(12, 7), height_ratios=[4, 1], sharex=True
        )
        im = ax.pcolormesh(
            edges,
            np.arange(len(names) + 1) - 0.5,
            counts,
            shading="flat",
            vmin=0,
            vmax=max(1, int(counts.max())),
        )
        cb = fig.colorbar(im, ax=ax, label="spikes / bin")
        if int(counts.max()) <= 12:
            cb.set_ticks(range(int(counts.max()) + 1))
        axp.stairs(counts.sum(0), edges, fill=True, color="0.2")
        axp.set_ylabel("sum / bin")
        kind = f"{BIN_MS:g} ms bins"
        if counts.shape[1] <= 40:
            ax.set_xticks(edges)
            axp.set_xticks(edges)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("time (s)" if BIN_MS is None else "")
    ax.set_ylabel("channel")
    ax.set_title(
        f"{subject}  {rec}  {t0:.0f}-{t0 + win:.0f} s  {kind}  BP 300-7500 Hz, -3.5 RMS"
    )
    ax.set_xlim(t0, t0 + win)
    if BIN_MS is not None:
        axp.set_xlabel("time (s)")
        axp.set_xlim(t0, t0 + win)
    fig.tight_layout()
    plt.show()
    plt.close()

# %% pose-label confidence over time
rec = info["default_session"]
pose = pd.read_parquet(
    f"{DATA}/{rec}/derived/pose.parquet",
    storage_options={"User-Agent": "Mozilla/5.0"},
)
likelihood_cols = [column for column in pose if column.endswith("_likelihood")]
labels = [column.removesuffix("_likelihood") for column in likelihood_cols]
cameras = pose["camera"].unique()
fig, axes = plt.subplots(
    len(cameras), 1, figsize=(14, 4 * len(cameras)), sharex=True, squeeze=False
)
for ax, camera in zip(axes[:, 0], cameras):
    rows = pose[(pose["camera"] == camera) & (pose["neural_sample"] >= 0)]
    image = ax.imshow(
        rows[likelihood_cols].to_numpy().T,
        aspect="auto",
        origin="lower",
        extent=(rows["t_s"].iloc[0], rows["t_s"].iloc[-1], -0.5, len(labels) - 0.5),
        vmin=0,
        vmax=1,
    )
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylabel(camera)
axes[-1, 0].set_xlabel("time (s)")
fig.colorbar(image, ax=axes[:, 0], label="DLC likelihood")
fig.suptitle(f"{rec} pose labels")
plt.show()
# %%
