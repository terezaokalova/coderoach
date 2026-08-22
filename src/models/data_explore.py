# %%
import zarr
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

DATA = "https://pub-8b7715bb28444713b1c917d7c4661390.r2.dev/recordings"
rec = "SD11_rec_20260820_182553.zarr"
root = zarr.open_group(
    zarr.storage.FsspecStore.from_url(f"{DATA}/{rec}", read_only=True),
    mode="r",
)


# %% schema — HTTP listing is empty; open known paths
def _arr(node):
    return f"{list(node.shape)} {node.dtype} chunks={node.chunks}"


print("root")
for k, v in dict(root.attrs).items():
    if k == "aravis_cameras":
        print(
            f"  {k}: {v['n_cameras']} cams @ {v['cameras']['cam0']['fps']} fps, {v['dir']}"
        )
        continue
    if k == "sync_validation":
        continue
    print(f"  {k}: {v}")

print("traces", _arr(root["traces"]), dict(root["traces"].attrs))
print(
    "digital_in",
    _arr(root["digital_in"]),
    {k: root["digital_in"].attrs[k] for k in ("dims", "note")},
)
print("trigger_edges", _arr(root["trigger_edges"]), dict(root["trigger_edges"].attrs))
print("overview", dict(root["overview"].attrs))
for lvl in ("l0", "l1", "l2"):
    print(f"  {lvl}/min", _arr(root[f"overview/{lvl}/min"]))
    print(f"  {lvl}/max", _arr(root[f"overview/{lvl}/max"]))
print("derived/ is not published on this prefix (404)")

# %% 10 s raster — derived/spikes.npz is 404, so detect on traces
fs = float(root.attrs["sample_rate_hz"])
gain = float(root["traces"].attrs.get("gain_uv_per_count", 0.195))
names = list(root["traces"].attrs["channel_names"])
t0, dur = 10.0, 10.0
i0, i1 = int(t0 * fs), int((t0 + dur) * fs)
step = int(fs)
raw = np.concatenate(
    [np.asarray(root["traces"][:, i : min(i + step, i1)]) for i in range(i0, i1, step)],
    axis=1,
)
rail = np.abs(raw) >= 30000
uv = raw.astype(np.float64)
idx = np.arange(uv.shape[1])
for ch in range(uv.shape[0]):
    good = ~rail[ch]
    if good.sum() > 10:
        uv[ch, rail[ch]] = np.interp(idx[rail[ch]], idx[good], uv[ch, good])
uv *= gain
hp = sosfiltfilt(butter(3, 300, btype="highpass", fs=fs, output="sos"), uv, axis=1)

refrac = int(0.001 * fs)
guard = int(0.002 * fs)
spikes = []
for ch, y in enumerate(hp):
    sig = 1.4826 * np.median(np.abs(y))
    thr = -5.0 * max(sig, 1e-6)
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

print(
    "n spikes", [int(s.size) for s in spikes], "total", int(sum(s.size for s in spikes))
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.eventplot(spikes, colors="k", linelengths=0.7, linewidths=0.6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("time (s)")
ax.set_ylabel("channel")
ax.set_title(f"{rec}  {t0:.0f}-{t0 + dur:.0f} s  HP 300 Hz, -5 MAD")
ax.set_xlim(t0, t0 + dur)
fig.tight_layout()
plt.show()

# %%
