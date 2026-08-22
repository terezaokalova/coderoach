# RoboRoach Bluetooth interface

This folder contains the Python interface for a Backyard Brains RoboRoach
backpack. It uses Bluetooth Low Energy (BLE) through
[`bleak`](https://bleak.readthedocs.io/) and mirrors the service and
characteristics in Backyard Brains' [official RoboRoach
repository](https://github.com/BackyardBrains/RoboRoach).

## Files

- `roboroach.py` implements scanning, connection management, settings, turn
  commands, and the persistent terminal session.
- `__init__.py` exposes `RoboRoach` and `StimulationSettings` for imports.
- `AGENTS.md` defines the protocol, safety, testing, and maintenance rules for
  changes in this folder.

## Before connecting

1. Use an adult, fully recovered cockroach whose three-electrode connector was
   installed according to the current Backyard Brains procedure.
2. Insert the CR1632 battery according to the backpack's polarity marking.
3. With the roach stationary, align and gently plug the backpack into the
   electrode connector. Do not force or reverse it.
4. Press the small black wake button. The BLE device should advertise as
   `RoboRoach` with service `B2B0`.
5. Disconnect the official phone app while Python owns the BLE connection.

Test scanning and the LEDs with the backpack off the animal whenever possible.
Stop if the connector is loose, the board becomes hot, or the animal has not
recovered normally.

## Install

From the repository root:

```bash
conda env create -f environment.yml
conda activate axohack
```

On macOS, allow the terminal application to use Bluetooth when prompted.

## Scan and inspect

These commands do not stimulate either antenna:

```bash
python interface/roboroach.py scan
python interface/roboroach.py info
```

If the backpack is not found, disconnect the official app, check the battery,
press the wake button, and retry near the computer.

## Send one turn command

These commands use the backpack's current stimulation settings:

```bash
python interface/roboroach.py left
python interface/roboroach.py right
```

Each invocation connects, sends one command, and disconnects.

## Persistent session

To avoid waking and reconnecting before every command:

```bash
python interface/roboroach.py session
```

The session accepts `left`, `right`, `info`, and `quit`. It sends a
non-stimulating heartbeat every two minutes by reading the current frequency
and writing the same value back. This resets the newer firmware's inactivity
timer without changing settings or stimulating an antenna. The connection can
still drop if the process or Bluetooth stops, the computer sleeps, the battery
is removed, or the backpack moves out of range.

## Python API

Run this from the repository root or install the repository on Python's import
path:

```python
import asyncio

from interface import RoboRoach


async def main():
    async with RoboRoach() as roach:
        print(await roach.read_settings())
        await roach.configure(
            frequency_hz=10,
            pulse_width_ms=1,
            duration_ms=250,
            gain_percent=10,
        )
        await roach.turn_left()


asyncio.run(main())
```

Use a single `asyncio.run()` call and keep the `async with` block open for the
whole control session.

## BLE protocol

All RoboRoach values below are single bytes under service `B2B0`:

| Characteristic | Meaning | Encoding |
| --- | --- | --- |
| `B2B1` | Frequency | Hz |
| `B2B2` | Pulse width | milliseconds |
| `B2B3` | Duration | 5 ms units |
| `B2B4` | Random mode | `0` off, `1` on |
| `B2B5` | Turn left | write `0x01`; stimulates right antenna |
| `B2B6` | Turn right | write `0x01`; stimulates left antenna |
| `B2B7` | Gain | percent |

The standard battery-level characteristic is `2A19`. Short UUIDs are expanded
using the Bluetooth base UUID `0000xxxx-0000-1000-8000-00805f9b34fb`.

## Biological envelope

Waveform caps stay weak (the backpack steps pulse width in milliseconds and
drives a 3 V pot). Timing is for a living cockroach under online control:
the animal turns in about 1-2 seconds, so the next train waits for that
response. There is no culture washout.

- Frequency 1 to 10 Hz, pulse width 1 ms, duration 200 to 300 ms, gain 0 to
  10%. Duty cycle stays at or below 50%.
- At least 2 seconds between trains.
- At most 30 trains, or 9 seconds of combined train time, in a rolling 60
  second window. Older events age out and control continues. The log is
  shared across processes.

A refused pulse raises an error and does not write the turn characteristic.
These limits are a floor, not a license to ignore the animal. Stop if the
board is hot, the connector is loose, or the roach is not recovering.

## Responsible operation

RoboRoach behavior is variable and repeated stimulation causes habituation.
Do not use unbounded turn loops or assume an exact angle from a pulse. Inspect
electrode and ground connections with the backpack powered off rather than
compensating for weak responses by increasing gain or pulse width.
