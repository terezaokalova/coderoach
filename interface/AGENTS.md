# Instructions for the RoboRoach interface

These instructions apply to every file under `interface/`.

## Purpose and scope

Maintain the hardware interface for the Backyard Brains RoboRoach: the BLE
backpack and the iPhone or webcam pose stream. Keep stimulation and camera
capture here. Put RL policy, teaching loops, notebooks, and datasets in
rl_control or elsewhere.

The public API is RoboRoach, StimulationSettings, Pose, PoseTracker,
SimulatedCamera, and KeyboardCamera from interface. PhonePoseTracker lives
in interface/track.py and must not be imported by interface/__init__.py.
Preserve the command-line entry points in interface/roboroach.py and
interface/track.py unless a replacement and migration instructions are
added in the same change.

## Protocol invariants

- Service UUID is `B2B0`, expanded with the Bluetooth base UUID.
- `B2B1`, `B2B2`, `B2B3`, `B2B4`, and `B2B7` are one-byte settings for
  frequency, pulse width, duration in 5 ms units, random mode, and gain.
- Writing `0x01` to `B2B5` requests a left turn by stimulating the right
  antenna. Writing `0x01` to `B2B6` requests a right turn by stimulating the
  left antenna.
- Do not change UUIDs, units, byte widths, or direction mapping without
  verifying the official client and firmware and documenting the evidence.
- Keep write-with-response behavior for commands and settings unless tested
  hardware demonstrates a required change.

## Safety constraints

- Importing the package, connecting, scanning, reading information, and
  starting a session must never stimulate an antenna or open a camera.
- Tests must use mocks or fakes. Never make a hardware stimulation command part
  of an automated test, startup hook, retry, heartbeat, or cleanup path.
- A user action may produce at most one turn pulse unless the user explicitly
  requests a bounded sequence. Never implement an unbounded stimulation loop.
- Keep validation on frequency, pulse width/duty cycle, duration, and gain.
  This is a living animal under online control, not an organoid culture:
  frequency 1 to 10 Hz, pulse width 1 ms, duration 200 to 300 ms, gain at
  most 10%, at least 2 s between trains (behavioral response is 1-2 s), at
  most 30 trains or 9 s of train time in a rolling 60 s window. Expired
  events drop off; there is no washout lockout. turn() must call this guard
  before the GATT write. Do not add a switch that disables the envelope.
- Keepalive traffic must rewrite an unchanged non-stimulation setting. Never
  use `B2B5` or `B2B6` as a heartbeat.
- Do not silently increase stimulation settings to compensate for an unreliable
  behavioral response. Consider habituation, battery state, and electrode or
  ground contact first.

## Connection behavior

- Use one `asyncio` event loop for scanning, connecting, GATT operations, and
  shutdown.
- Prefer a discovered `BLEDevice` over a raw address. macOS exposes a UUID-like
  identifier instead of a stable Bluetooth MAC address.
- Own one `BleakClient` per backpack and disconnect it cleanly through the async
  context manager.
- Keep terminal input off the event loop with `asyncio.to_thread`.
- A persistent session may keep the board awake with the existing two-minute,
  non-stimulating heartbeat. Connection loss must fail visibly; do not retry a
  turn command automatically because delivery may already have occurred.

## Verification

For every code change:

1. Run python -m py_compile on interface/roboroach.py, interface/camera.py,
   and interface/__init__.py. Also compile interface/track.py and
   interface/plot.py when those files change.
2. Run git diff --check.
3. Test GATT reads and writes with a fake client, including exact UUID, byte
   value, units, and response=True behavior.
4. If hardware verification is explicitly authorized, progress through
   scan, then info, then at most one requested direction command. Report
   separately whether BLE accepted the write and whether an animal visibly
   responded.

## Documentation

Update `interface/README.md` whenever commands, public imports, UUIDs, units,
validation, keepalive behavior, or hardware setup change. Keep the repository
root README concise and link here for interface details.
