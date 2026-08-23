"""GATT-level tests for the backpack, against a fake client.

interface/AGENTS.md forbids putting a hardware stimulation command in an
automated test, so every write here lands on FakeClient and is asserted on by
UUID, byte value, and response flag rather than by watching an animal.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from interface import RoboRoach
from interface.roboroach import (
    DURATION_UUID,
    SERVICE_UUID,
    TURN_LEFT_UUID,
    _recent_stims,
)


class FakeClient:
    """Records every GATT operation. Constructed by patching BleakClient."""

    def __init__(self, device, duration_units: int = 100) -> None:
        self.device = device
        self.is_connected = False
        self.reads: list[str] = []
        self.writes: list[tuple[str, bytes, bool]] = []
        self.values = {DURATION_UUID: bytes([duration_units])}

    async def connect(self) -> None:
        self.is_connected = True

    async def disconnect(self) -> None:
        self.is_connected = False

    @property
    def services(self):
        return [SimpleNamespace(uuid=SERVICE_UUID)]

    async def read_gatt_char(self, uuid: str) -> bytes:
        self.reads.append(uuid)
        return self.values.get(uuid, b"\x00")

    async def write_gatt_char(self, uuid: str, data: bytes, response: bool = False):
        self.writes.append((uuid, data, response))


@pytest.fixture
def isolated_budget(tmp_path, monkeypatch):
    """Keep the rolling charge log out of the shared temp file."""
    monkeypatch.setattr(
        "interface.roboroach._SAFETY_PATH", tmp_path / "roboroach-safety.json"
    )


@pytest.fixture
def client(monkeypatch):
    made: list[FakeClient] = []

    def build(device):
        made.append(FakeClient(device))
        return made[-1]

    monkeypatch.setattr("interface.roboroach.BleakClient", build)
    return made


def test_connect_takes_duration_from_the_board(client, isolated_budget):
    """A board holding 500 ms must not be metered as the assumed 250 ms."""

    async def run():
        async with RoboRoach(device=SimpleNamespace(name="RoboRoach")) as roach:
            return roach._duration_ms, client[0].reads

    duration_ms, reads = asyncio.run(run())
    assert DURATION_UUID in reads, "connect() must read the duration characteristic"
    assert duration_ms == 500, "5 ms units: 100 units is 500 ms, not 100 or 250"


def test_turn_meters_the_boards_duration(client, isolated_budget):
    """The charge budget records what the firmware delivers, not the guess."""

    async def run():
        async with RoboRoach(device=SimpleNamespace(name="RoboRoach")) as roach:
            await roach.turn("left")
            return client[0].writes

    writes = asyncio.run(run())
    uuid, payload, response = writes[-1]
    assert uuid == TURN_LEFT_UUID
    assert payload == b"\x01"
    assert response is True, "commands are write-with-response"

    events = _recent_stims(__import__("time").time())
    assert [event["duration_ms"] for event in events] == [500]


def test_configure_keeps_duration_in_step(client, isolated_budget):
    """configure() still owns the value after connect() has seeded it."""

    async def run():
        async with RoboRoach(device=SimpleNamespace(name="RoboRoach")) as roach:
            seeded = roach._duration_ms
            await roach.configure(duration_ms=250)
            return seeded, roach._duration_ms, client[0].writes

    seeded, after, writes = asyncio.run(run())
    assert seeded == 500
    assert after == 250
    assert (DURATION_UUID, bytes([50]), True) in writes, "250 ms is 50 five-ms units"
