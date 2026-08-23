"""Chunked threshold-crossing spike counts for pose decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
import zarr
from scipy.signal import butter, sosfiltfilt

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}


def load_data_info(info_yaml: Path) -> dict:
    return yaml.safe_load(Path(info_yaml).read_text())


def open_recording(base_url: str, session: str):
    url = f"{base_url.rstrip('/')}/{session}"
    return zarr.open_group(
        zarr.storage.FsspecStore.from_url(url, read_only=True),
        mode="r",
    )


def _detect_spikes(bp: np.ndarray, rail: np.ndarray, fs: float) -> list[np.ndarray]:
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
        spikes.append(np.asarray(keep, dtype=np.float64))
    return spikes


def spike_counts_for_session(
    root,
    *,
    bin_ms: float,
    chunk_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    """Return counts [n_bins, n_channels], bin_centers_s, fs, channel_names."""
    traces = root["traces"]
    fs = float(root.attrs["sample_rate_hz"])
    gain = float(traces.attrs.get("gain_uv_per_count", 0.195))
    names = list(traces.attrs["channel_names"])
    n_samples = int(traces.shape[1])
    dur = float(root.attrs.get("duration_s", n_samples / fs))
    dt = bin_ms / 1000.0
    n_bins = int(np.floor(dur / dt))
    edges = np.arange(n_bins + 1, dtype=np.float64) * dt
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros((n_bins, len(names)), dtype=np.float32)
    sos = butter(3, (300, 7500), btype="bandpass", fs=fs, output="sos")
    chunk = max(1, int(chunk_s * fs))
    for i0 in range(0, n_samples, chunk):
        i1 = min(i0 + chunk, n_samples)
        raw = np.asarray(traces[:, i0:i1])
        rail = np.abs(raw) >= 30000
        uv = raw.astype(np.float64)
        idx = np.arange(uv.shape[1])
        for ch in range(uv.shape[0]):
            good = ~rail[ch]
            if good.sum() > 10:
                uv[ch, rail[ch]] = np.interp(idx[rail[ch]], idx[good], uv[ch, good])
        bp = sosfiltfilt(sos, uv * gain, axis=1)
        spikes = _detect_spikes(bp, rail, fs)
        t0 = i0 / fs
        for ch, samples in enumerate(spikes):
            if samples.size == 0:
                continue
            times = samples / fs + t0
            hist, _ = np.histogram(times, bins=edges)
            counts[:, ch] += hist.astype(np.float32)
    return counts, centers.astype(np.float32), fs, names


def cache_path(cache_dir: Path, session: str, bin_ms: float) -> Path:
    safe = session.replace("/", "_")
    return Path(cache_dir) / f"{safe}.bin{bin_ms:g}ms.npz"


def build_or_load_spike_cache(
    *,
    base_url: str,
    session: str,
    bin_ms: float,
    cache_dir: Path,
    force: bool = False,
) -> dict:
    path = cache_path(cache_dir, session, bin_ms)
    if path.exists() and not force:
        data = np.load(path, allow_pickle=False)
        return {
            "counts": data["counts"],
            "bin_centers_s": data["bin_centers_s"],
            "fs": float(data["fs"]),
            "channel_names": [str(x) for x in data["channel_names"]],
            "session": session,
            "bin_ms": bin_ms,
            "path": path,
        }
    root = open_recording(base_url, session)
    counts, centers, fs, names = spike_counts_for_session(root, bin_ms=bin_ms)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        counts=counts,
        bin_centers_s=centers,
        fs=np.float64(fs),
        channel_names=np.asarray(names),
    )
    return {
        "counts": counts,
        "bin_centers_s": centers,
        "fs": fs,
        "channel_names": names,
        "session": session,
        "bin_ms": bin_ms,
        "path": path,
    }
