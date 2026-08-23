# stim/gate.py

## Plan

- [x] Read `interface/roboroach.py`, report `configure()` and `StimulationSettings`
- [x] Resolve the two spec conflicts: `configure()` returns `None`, and it is
      async while construction is not
- [x] `StimGate` with required roach, `t_refrac_s`, settings, run dir
- [x] Refractory check, Bluetooth write, and `n` increment under one lock
- [x] JSONL line per request, accepted or rejected
- [x] `sources.json` merged, not overwritten
- [x] Test suite with an injected stub, no hardware
- [x] `ruff format . ; ruff check --fix . ; pytest -q`

## Review

`StimGate` is the only object that calls `RoboRoach.turn`. It holds the
refractory period and the trial counter `n`, so the stimulation budget is
enforced once instead of once per caller.

Two things the spec did not name but that the code needs:

- `configure()` returns `None`, so the board settings come from a
  `read_settings()` readback. The readback is what will actually fire, so the
  refractory assertion is checked against it as well as against the request.
- A failed Bluetooth write may already have been delivered. It holds the
  refractory window shut, is logged as `write_failed`, does not increment `n`,
  and re-raises. `interface/AGENTS.md` requires visible failure and no retry.

`interface/` is excluded from ruff in `pyproject.toml`. Running `ruff format .`
without that would rewrite `roboroach.py`, including `turn()`.

Open: `T_refrac` still has to be set from the habituation budget, and the
shared consumer that the voice module emits into does not exist in the
repository yet.

# web/

## Plan

- [x] Read `stim/gate.py`, `traj/track.py`, `traj/control.py`, `voice/` and pin
      down the exact API each one exposes
- [x] `web/hub.py`: one tracker pump, latest frame fanned out to video, state,
      and the control loop
- [x] `web/runs.py`: per-run `path.json`, plus replay reads of the tracker and
      gate JSONL
- [x] `web/loop.py`: single control-loop task, pure pursuit into the gate
- [x] `web/app.py`: the nine endpoints, capability gating, stim journal
- [x] `web/static/index.html`: one file, pointer events, arena-space canvas
- [x] `web/__main__.py`: CLI, 0.0.0.0 bind, localhost and LAN URLs printed
- [x] Degrade to replay mode when the arena calibration is missing
- [x] QR of the LAN URL in the banner, ASCII, verified against a real decoder
- [x] `ruff format . ; ruff check --fix .`

## Review

The drawing canvas and its server did not exist. `traj/control.py:179` already
documented a browser that resamples a stroke before POSTing it, and FastAPI and
uvicorn had been in `environment.yml` since 28ee96d, but nothing imported
either. This builds that missing layer and nothing else.

Three decisions worth keeping in mind:

- **The camera is drained in one place.** `TrajectoryTracker.process_once`
  grabs a frame *and* appends a JSONL line, so calling it from the video
  stream, the state stream, and the control loop would have tripled the
  tracker log and left three readers fighting over a one-frame buffer.
  `web/hub.py` pumps it once and hands the newest frame to everyone. Readers
  skip frames rather than replay stale ones, which is what a tablet on a
  crowded network needs.

- **The canvas is in arena centimetres, not image pixels.** Mapping a stroke to
  waypoints is then a scale and a y-flip instead of a projective transform, and
  the homography never leaves the server, so there is only one copy of the
  calibration. `resample()` in the page mirrors `resample_path`; the server
  re-spaces whatever arrives anyway, because the browser's copy is the one that
  can be stale.

- **Capabilities are checked one at a time.** Camera, backpack, and microphone
  degrade independently and each endpoint refuses with its own reason. A single
  live/replay switch would not tell an operator which cable to go and plug in.
  A missing arena calibration serves the recorded runs instead of refusing to
  start, and `web/runs.py` imports nothing beyond the standard library so the
  replay deployment needs neither OpenCV nor bleak.

The QR code in the banner is drawn from half-block characters, so its
polarity depends on the terminal behind it: on a dark background the block
character is the *light* module, on a light background it is the ink. Rendering
it the wrong way round produces a photographic negative. `tests/test_web_banner.py`
rebuilds the image a camera would see and runs `cv2.QRCodeDetector` over it,
asserting that the default scans on a dark terminal, that `--qr-light` scans on
a light one, and that the mismatched pair does **not** decode. That last
assertion is why the flag exists rather than a hardcoded choice.

Two things this changed that were not asked for. `sounddevice` and
`faster-whisper` were missing from `environment.yml` even though `voice/`
imports both at module scope, so a fresh env could not have run `/api/voice`;
both are added. `T_refrac` is now required on the command line unless
`--no-roach`, following `voice/probe.py`.

Open:

- `ruff format .` rewrites `src/recording/`, which belongs to another author.
  It was reverted by hand here. `extend-exclude` in `pyproject.toml` already
  carries `interface` and `rl_control` for exactly this reason and wants a
  third entry, but that is the file owner's call.
- Binding 0.0.0.0 puts an unauthenticated control surface for a live animal on
  the local network. `interface.guard_turn` and the gate's refractory period
  still apply, but neither asks who is calling. `--host 127.0.0.1` opts out.
- The page was verified by unit test, by a hardware-free simulation, and by
  checking the resampler against its Python twin. It has not been opened on an
  actual iPad. Palm rejection in particular is calibrated on a 1.5 s grace
  window that only a real Pencil can confirm.


# traj/track.py -- arena mask

## Plan

- [x] Keep the calibration's `image_points` on `ArenaHomography`
- [x] `region_mask()`: 255 inside the arena polygon, 0 outside
- [x] `detect(..., region=None)`, applied after the morphology and before
      `findContours`
- [x] Build it once in `TrajectoryTracker.__init__`, pass it every frame
- [x] Tests that the cardboard wins unmasked and cannot win masked
- [x] `ruff format . ; ruff check --fix .`

## Review

The tracker takes the largest contour in the frame, so anything red outside the
arena competes with the backpack and wins as soon as it is bigger. No HSV
window fixes that, because the difference between cardboard and a backpack is
where they are and not what colour they are. `ArenaHomography.region_mask()`
zeroes everything outside the calibration polygon and `detect` intersects it
with the threshold mask before `findContours`.

Three details worth keeping:

- **The mask is applied after the morphology, not before.** `threshold_mask`
  closes gaps, and closing runs first so a blob reaching over the arena edge is
  cut at the edge rather than being closed across it and surviving.
- **`detect(region=...)` defaults to None.** `traj/calibrate.py` calls `detect`
  while calibrating HSV, which happens before there is an arena calibration to
  mask with, so that call is unchanged.
- **`image_points` is now a field on `ArenaHomography`.** The matrix alone
  cannot say where the arena's edge is. Existing `arena.json` files already
  carry the points -- `traj/calibrate.py` has always written them -- so nothing
  needs recalibrating. Only `fit()` constructs the dataclass, so nothing else
  had to change.

A blob straddling the arena edge is clipped and its centroid pulled inwards.
That is the intended trade: the edge is where the homography stops being valid
anyway. If the roach is being tracked right at the boundary in practice, the
fix is a larger arena calibration, not a looser mask.
