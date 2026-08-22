"""Async Python interface for the Backyard Brains RoboRoach BLE backpack.

The UUIDs and one-byte values mirror Backyard Brains' Android client and
firmware. This module only talks to the backpack; it does not bypass any of
the board's firmware limits.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from dataclasses import dataclass
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


@dataclass(frozen=True)
class StimulationSettings:
    frequency_hz: int
    pulse_width_ms: int
    duration_ms: int
    gain_percent: int
    random_mode: bool


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

    async def read_battery_percent(self) -> int | None:
        try:
            value = await self._connected_client().read_gatt_char(BATTERY_UUID)
        except Exception:
            return None
        return value[0] if value else None

    async def configure(
        self,
        *,
        frequency_hz: int = 55,
        pulse_width_ms: int = 5,
        duration_ms: int = 500,
        gain_percent: int = 50,
        random_mode: bool = False,
    ) -> None:
        """Set conservative values within the published backpack ranges."""
        if not 1 <= frequency_hz <= 150:
            raise ValueError("frequency_hz must be from 1 to 150")
        if not 1 <= pulse_width_ms <= 255:
            raise ValueError("pulse_width_ms must be from 1 to 255")
        if pulse_width_ms * frequency_hz > 500:
            raise ValueError("pulse width must keep duty cycle at or below 50%")
        if not 10 <= duration_ms <= 1000 or duration_ms % 5:
            raise ValueError("duration_ms must be 10..1000 in 5 ms increments")
        if not 0 <= gain_percent <= 100:
            raise ValueError("gain_percent must be from 0 to 100")

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

    async def turn(self, direction: Literal["left", "right"]) -> None:
        """Request one firmware-timed left or right turn stimulus."""
        uuid = TURN_LEFT_UUID if direction == "left" else TURN_RIGHT_UUID
        await self._connected_client().write_gatt_char(
            uuid, b"\x01", response=True
        )

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
            await roach.turn(args.command)
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
                await roach.turn(command)
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
    parser.add_argument(
        "command", choices=("scan", "info", "left", "right", "session")
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    asyncio.run(run_command(args))


if __name__ == "__main__":
    main()
