"""Async Python interface for the Backyard Brains RoboRoach BLE backpack.

The UUIDs and one-byte values mirror Backyard Brains' Android client and
firmware. Waveform caps stay conservative (10 Hz, 1 ms, 200-300 ms, 10%
gain). Timing is for a living animal under online control: wait ~2 s for
the turn, then allow the next train. There is no culture washout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice


def bluetooth_uuid(short_uuid: int) -> str:
    return f"0000{short_uuid:04x}-0000-1000-8000-00805f9b34fb"


SERVICE_UUID = bluetooth_uuid(0xB2B0)
FREQUENCY_UUID = bluetooth_uuid(0xB2B1)
PULSE_WIDTH_UUID = bluetooth_uuid(0xB2B2)
DURATION_UUID = bluetooth_uuid(0xB2B3)
RANDOM_MODE_UUID = bluetooth_uuid(0xB2B4)
TURN_LEFT_UUID = bluetooth_uuid(0xB2B5)
TURN_RIGHT_UUID = bluetooth_uuid(0xB2B6)
GAIN_UUID = bluetooth_uuid(0xB2B7)
BATTERY_UUID = bluetooth_uuid(0x2A19)

# Living-animal online control. A train produces a turn in about 1-2 s, so
# the next pulse waits for that response. A rolling 60 s window caps count
# and train time so charge cannot stack; expired events drop and control
# continues. No organoid washout. Waveform stays at the weakest hardware
# settings that still match a 10 Hz / 200-300 ms train.
MIN_STIM_INTERVAL_S = 2.0
MAX_PULSES_PER_WINDOW = 30
MAX_STIM_MS_PER_WINDOW = 9000
SAFETY_WINDOW_S = 60.0
MAX_FREQUENCY_HZ = 10
MAX_PULSE_WIDTH_MS = 1
MIN_DURATION_MS = 200
MAX_DURATION_MS = 300
MAX_GAIN_PERCENT = 10
_SAFETY_PATH = Path(tempfile.gettempdir()) / "roboroach-safety.json"


@dataclass(frozen=True)
class StimulationSettings:
    frequency_hz: int
    pulse_width_ms: int
    duration_ms: int
    gain_percent: int
    random_mode: bool


def _load_stim_log() -> list[dict]:
    try:
        raw = json.loads(_SAFETY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [event for event in raw if isinstance(event, dict)]


def _save_stim_log(events: list[dict]) -> None:
    _SAFETY_PATH.write_text(json.dumps(events))


def _recent_stims(now: float) -> list[dict]:
    return [
        event
        for event in _load_stim_log()
        if now - float(event.get("t", 0)) <= SAFETY_WINDOW_S
    ]


def _window_wait_s(events: list[dict], now: float) -> float:
    oldest = float(events[0].get("t", 0))
    return max(1.0, SAFETY_WINDOW_S - (now - oldest))


def guard_turn(duration_ms: int) -> None:
    """Refuse a pulse that is too soon or over the rolling charge budget."""
    now = time.time()
    events = _recent_stims(now)
    if events:
        wait = MIN_STIM_INTERVAL_S - (now - float(events[-1].get("t", 0)))
        if wait > 0:
            raise RuntimeError(
                f"Wait {wait:.1f}s for the animal to finish the last turn."
            )
    if len(events) >= MAX_PULSES_PER_WINDOW:
        raise RuntimeError(
            f"At most {MAX_PULSES_PER_WINDOW} trains in "
            f"{int(SAFETY_WINDOW_S)}s. Wait {_window_wait_s(events, now):.0f}s "
            "for the oldest pulse to age out."
        )
    used = sum(int(event.get("duration_ms", 0)) for event in events)
    if used + duration_ms > MAX_STIM_MS_PER_WINDOW:
        raise RuntimeError(
            f"Train-time budget for this {int(SAFETY_WINDOW_S)}s window is "
            f"used. Wait {_window_wait_s(events, now):.0f}s before more pulses."
        )


def guard_envelope(settings: StimulationSettings) -> None:
    """Refuse a pulse the board is not configured to deliver safely.

    configure() validates what this client writes, but nothing validated what
    was already on the board. A backpack left on the phone app's settings
    (55 Hz, 9 ms, 500 ms, 50% gain) would have stimulated at those values on the
    first turn(), because turn() checked only the charge budget and metered that
    against its own belief about duration.

    Every cap is reported at once rather than the first one hit: an operator who
    has to reconnect and retry to discover the next violation learns the board is
    out of range one parameter at a time. This raises and changes nothing --
    bringing the board back into range is configure()'s job, and doing it here
    would silently restimulate at settings the caller never asked for.
    """
    over = []
    if settings.frequency_hz > MAX_FREQUENCY_HZ:
        over.append(
            f"frequency {settings.frequency_hz} Hz over the {MAX_FREQUENCY_HZ} Hz cap"
        )
    if settings.pulse_width_ms > MAX_PULSE_WIDTH_MS:
        over.append(
            f"pulse width {settings.pulse_width_ms} ms over the "
            f"{MAX_PULSE_WIDTH_MS} ms cap"
        )
    if settings.duration_ms > MAX_DURATION_MS:
        over.append(
            f"duration {settings.duration_ms} ms over the {MAX_DURATION_MS} ms cap"
        )
    if settings.gain_percent > MAX_GAIN_PERCENT:
        over.append(
            f"gain {settings.gain_percent}% over the {MAX_GAIN_PERCENT}% cap"
        )
    if over:
        raise RuntimeError(
            "The board is outside the living-animal envelope: "
            + "; ".join(over)
            + ". Call configure() to bring it into range."
        )


def record_turn(duration_ms: int) -> None:
    now = time.time()
    events = _recent_stims(now)
    events.append({"t": now, "duration_ms": int(duration_ms)})
    _save_stim_log(events)


class RoboRoach:
    """Connect to and control one RoboRoach backpack over Bluetooth LE."""

    def __init__(
        self,
        device: BLEDevice | None = None,
        *,
        name: str = "RoboRoach",
        scan_timeout: float = 10.0,
    ) -> None:
        self.device = device
        self.name = name
        self.scan_timeout = scan_timeout
        self.client: BleakClient | None = None
        # What the board reported at connect(), refreshed by configure().
        # turn() checks the envelope and meters charge from this, so it stays
        # None until a real read has happened rather than starting life as an
        # assumption that turn() would trust.
        self._settings: StimulationSettings | None = None

    async def connect(self) -> None:
        if self.device is None:
            wanted = self.name.casefold()
            self.device = await BleakScanner.find_device_by_filter(
                lambda device, advertisement: (
                    advertisement.local_name or device.name or ""
                ).casefold()
                == wanted,
                timeout=self.scan_timeout,
            )
        if self.device is None:
            raise RuntimeError(
                f"No {self.name!r} advertisement found. Insert the CR1632 "
                "battery, press the backpack's black wake button, and retry."
            )

        self.client = BleakClient(self.device)
        await self.client.connect()
        service_uuids = {service.uuid.casefold() for service in self.client.services}
        if SERVICE_UUID not in service_uuids:
            await self.disconnect()
            raise RuntimeError(
                f"Connected to {self.device.name!r}, but it does not expose "
                f"the RoboRoach service {SERVICE_UUID}."
            )
        self._settings = await self.read_settings()

    async def disconnect(self) -> None:
        if self.client is not None:
            await self.client.disconnect()
            self.client = None

    async def __aenter__(self) -> "RoboRoach":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    def _connected_client(self) -> BleakClient:
        if self.client is None or not self.client.is_connected:
            raise RuntimeError("The RoboRoach backpack is not connected")
        return self.client

    async def read_settings(self) -> StimulationSettings:
        client = self._connected_client()

        async def read_u8(uuid: str) -> int:
            value = await client.read_gatt_char(uuid)
            if not value:
                raise RuntimeError(f"Empty value returned by {uuid}")
            return value[0]

        return StimulationSettings(
            frequency_hz=await read_u8(FREQUENCY_UUID),
            pulse_width_ms=await read_u8(PULSE_WIDTH_UUID),
            duration_ms=5 * await read_u8(DURATION_UUID),
            gain_percent=await read_u8(GAIN_UUID),
            random_mode=bool(await read_u8(RANDOM_MODE_UUID)),
        )

    def _require_settings(self) -> StimulationSettings:
        """What the board is holding, or a refusal to guess.

        turn() will not stimulate against an unverified waveform, so an object
        that has never completed a connect() has nothing to check and must not
        fall back to a default.
        """
        if self._settings is None:
            raise RuntimeError(
                "The board's stimulation settings are unknown. connect() reads "
                "them; turn() will not stimulate without them."
            )
        return self._settings

    async def read_battery_percent(self) -> int | None:
        try:
            value = await self._connected_client().read_gatt_char(BATTERY_UUID)
        except Exception:
            return None
        return value[0] if value else None

    async def configure(
        self,
        *,
        frequency_hz: int = 10,
        pulse_width_ms: int = 1,
        duration_ms: int = 250,
        gain_percent: int = 10,
        random_mode: bool = False,
    ) -> None:
        """Set values inside the living-animal waveform envelope."""
        if not 1 <= frequency_hz <= MAX_FREQUENCY_HZ:
            raise ValueError(f"frequency_hz must be from 1 to {MAX_FREQUENCY_HZ}")
        if not 1 <= pulse_width_ms <= MAX_PULSE_WIDTH_MS:
            raise ValueError(f"pulse_width_ms must be from 1 to {MAX_PULSE_WIDTH_MS}")
        if pulse_width_ms * frequency_hz > 500:
            raise ValueError("pulse width must keep duty cycle at or below 50%")
        if not MIN_DURATION_MS <= duration_ms <= MAX_DURATION_MS or duration_ms % 5:
            raise ValueError(
                f"duration_ms must be {MIN_DURATION_MS}..{MAX_DURATION_MS} "
                "in 5 ms increments"
            )
        if not 0 <= gain_percent <= MAX_GAIN_PERCENT:
            raise ValueError(f"gain_percent must be from 0 to {MAX_GAIN_PERCENT}")

        client = self._connected_client()
        values = (
            (FREQUENCY_UUID, frequency_hz),
            (PULSE_WIDTH_UUID, pulse_width_ms),
            (DURATION_UUID, duration_ms // 5),
            (GAIN_UUID, gain_percent),
            (RANDOM_MODE_UUID, int(random_mode)),
        )
        for uuid, value in values:
            await client.write_gatt_char(uuid, bytes([value]), response=True)
        self._settings = StimulationSettings(
            frequency_hz=frequency_hz,
            pulse_width_ms=pulse_width_ms,
            duration_ms=duration_ms,
            gain_percent=gain_percent,
            random_mode=random_mode,
        )

    async def turn(self, direction: Literal["left", "right"]) -> None:
        """Request one firmware-timed left or right turn stimulus."""
        settings = self._require_settings()
        guard_envelope(settings)
        guard_turn(settings.duration_ms)
        uuid = TURN_LEFT_UUID if direction == "left" else TURN_RIGHT_UUID
        await self._connected_client().write_gatt_char(uuid, b"\x01", response=True)
        record_turn(settings.duration_ms)

    async def turn_left(self) -> None:
        await self.turn("left")

    async def turn_right(self) -> None:
        await self.turn("right")

    async def keep_alive(self, interval_seconds: float = 120.0) -> None:
        """Prevent firmware hibernation without triggering stimulation.

        Cypress firmware resets its active sleep timer on a GATT write. Reading
        the current frequency and writing the same byte back leaves the
        stimulation settings unchanged while resetting that timer.
        """
        if not 30 <= interval_seconds <= 300:
            raise ValueError("interval_seconds must be from 30 to 300")

        while True:
            await asyncio.sleep(interval_seconds)
            client = self._connected_client()
            frequency = await client.read_gatt_char(FREQUENCY_UUID)
            if not frequency:
                raise RuntimeError("Empty frequency returned during keepalive")
            await client.write_gatt_char(
                FREQUENCY_UUID, bytes(frequency[:1]), response=True
            )


async def scan(timeout: float) -> None:
    found = await BleakScanner.discover(timeout=timeout, return_adv=True)
    matches = []
    for device, advertisement in found.values():
        name = advertisement.local_name or device.name or ""
        if name.casefold() == "roboroach":
            matches.append((device, advertisement.rssi))

    if not matches:
        print("No RoboRoach advertisement found.")
        return
    for device, rssi in sorted(matches, key=lambda match: match[1], reverse=True):
        print(f"{device.name or 'RoboRoach'}  {device.address}  RSSI {rssi} dBm")


async def run_command(args: argparse.Namespace) -> None:
    if args.command == "scan":
        await scan(args.timeout)
        return

    async with RoboRoach(scan_timeout=args.timeout) as roach:
        if args.command == "info":
            settings = await roach.read_settings()
            battery = await roach.read_battery_percent()
            print(f"Device:   {roach.device.name} ({roach.device.address})")
            print(f"Settings: {settings}")
            print(f"Battery:  {battery if battery is not None else 'unavailable'}")
        elif args.command == "session":
            await interactive_session(roach)
        else:
            try:
                await roach.turn(args.command)
            except RuntimeError as exc:
                raise SystemExit(str(exc))
            print(f"Sent one {args.command} command")


async def interactive_session(roach: RoboRoach) -> None:
    """Keep one connection open and accept commands from the terminal."""
    print(f"Connected to {roach.device.name} ({roach.device.address})")
    print("Commands: left, right, info, quit")
    keepalive_task = asyncio.create_task(roach.keep_alive())
    try:
        while True:
            command = (await asyncio.to_thread(input, "roach> ")).strip().casefold()
            if command in {"quit", "exit", "q"}:
                return
            if command in {"left", "right"}:
                try:
                    await roach.turn(command)
                except RuntimeError as exc:
                    print(exc)
                    continue
                print(f"Sent one {command} command")
                continue
            if command == "info":
                print(await roach.read_settings())
                continue
            if command:
                print("Use: left, right, info, or quit")
    finally:
        keepalive_task.cancel()
        with suppress(asyncio.CancelledError):
            await keepalive_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("scan", "info", "left", "right", "session"))
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    asyncio.run(run_command(args))


if __name__ == "__main__":
    main()
