"""GATT-level tests for the backpack, against a fake client.

interface/AGENTS.md forbids putting a hardware stimulation command in an
automated test, so every write here lands on FakeClient and is asserted on by
UUID, byte value, and response flag rather than by watching an animal.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from interface import RoboRoach, StimulationSettings
from interface.roboroach import (
    DURATION_UUID,
    FREQUENCY_UUID,
    GAIN_UUID,
    PULSE_WIDTH_UUID,
    RANDOM_MODE_UUID,
    SERVICE_UUID,
    TURN_LEFT_UUID,
    _recent_stims,
    guard_envelope,
)

# What the Backyard Brains phone app leaves on the board. Every value is over
# the living-animal cap, which is the case the envelope guard exists for.
PHONE_APP = StimulationSettings(
    frequency_hz=55,
    pulse_width_ms=9,
    duration_ms=500,
    gain_percent=50,
    random_mode=False,
)
IN_ENVELOPE = StimulationSettings(
    frequency_hz=10,
    pulse_width_ms=1,
    duration_ms=250,
    gain_percent=10,
    random_mode=False,
)


class FakeClient:
    """Records every GATT operation. Constructed by patching BleakClient."""

    board = PHONE_APP

    def __init__(self, device) -> None:
        self.device = device
        self.is_connected = False
        self.reads: list[str] = []
        self.writes: list[tuple[str, bytes, bool]] = []
        self.values = {
            FREQUENCY_UUID: bytes([self.board.frequency_hz]),
            PULSE_WIDTH_UUID: bytes([self.board.pulse_width_ms]),
            DURATION_UUID: bytes([self.board.duration_ms // 5]),
            GAIN_UUID: bytes([self.board.gain_percent]),
            RANDOM_MODE_UUID: bytes([int(self.board.random_mode)]),
        }

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


@pytest.fixture(autouse=True)
def isolated_budget(tmp_path, monkeypatch):
    """Keep the rolling charge log out of the shared temp file."""
    monkeypatch.setattr(
        "interface.roboroach._SAFETY_PATH", tmp_path / "roboroach-safety.json"
    )


@pytest.fixture
def board(monkeypatch):
    """Patch in a fake board; set `board.settings` before connecting."""
    made: list[FakeClient] = []

    def build(device):
        made.append(FakeClient(device))
        return made[-1]

    monkeypatch.setattr("interface.roboroach.BleakClient", build)
    monkeypatch.setattr(FakeClient, "board", PHONE_APP)
    return SimpleNamespace(
        clients=made,
        set=lambda s: monkeypatch.setattr(FakeClient, "board", s),
    )


def run(coro_fn):
    async def main():
        async with RoboRoach(device=SimpleNamespace(name="RoboRoach")) as roach:
            return await coro_fn(roach)

    return asyncio.run(main())


# --------------------------------------------------------------------------
# the envelope guard on its own
# --------------------------------------------------------------------------


def test_guard_envelope_names_every_parameter_over_cap():
    with pytest.raises(RuntimeError) as excinfo:
        guard_envelope(PHONE_APP)
    message = str(excinfo.value)
    assert "frequency 55 Hz over the 10 Hz cap" in message
    assert "pulse width 9 ms over the 1 ms cap" in message
    assert "duration 500 ms over the 300 ms cap" in message
    assert "gain 50% over the 10% cap" in message


def test_guard_envelope_passes_inside_the_caps():
    guard_envelope(IN_ENVELOPE)


def test_guard_envelope_names_only_the_offender():
    only_gain = StimulationSettings(10, 1, 250, 50, False)
    with pytest.raises(RuntimeError) as excinfo:
        guard_envelope(only_gain)
    message = str(excinfo.value)
    assert "gain 50%" in message
    assert "frequency" not in message and "duration" not in message


# --------------------------------------------------------------------------
# turn() against a real-ish board
# --------------------------------------------------------------------------


def test_turn_refuses_a_board_left_on_phone_app_settings(board):
    """The whole point: no GATT write reaches the turn characteristic."""
    with pytest.raises(RuntimeError, match="outside the living-animal envelope"):
        run(lambda roach: roach.turn("left"))
    turn_writes = [w for w in board.clients[0].writes if w[0] == TURN_LEFT_UUID]
    assert turn_writes == [], "refusal must happen before the GATT write"
    assert _recent_stims(time.time()) == [], "a refused pulse must not be metered"


def test_turn_writes_and_meters_when_inside_the_envelope(board):
    board.set(IN_ENVELOPE)
    writes = run(lambda roach: _turn_then_writes(roach))
    uuid, payload, response = writes[-1]
    assert uuid == TURN_LEFT_UUID
    assert payload == b"\x01"
    assert response is True, "commands are write-with-response"
    assert [e["duration_ms"] for e in _recent_stims(time.time())] == [250]


async def _turn_then_writes(roach):
    await roach.turn("left")
    return roach.client.writes


def test_connect_reads_settings_from_the_board(board):
    settings = run(lambda roach: _settings_of(roach))
    assert settings == PHONE_APP, "connect() must not assume; it reads"
    assert DURATION_UUID in board.clients[0].reads


async def _settings_of(roach):
    return roach._settings


def test_turn_without_connect_refuses_rather_than_guessing():
    roach = RoboRoach(device=SimpleNamespace(name="RoboRoach"))
    with pytest.raises(RuntimeError, match="settings are unknown"):
        asyncio.run(roach.turn("left"))


def test_configure_keeps_the_cache_in_step(board):
    before, after, writes = run(lambda roach: _configure(roach))
    assert before == PHONE_APP
    assert after == IN_ENVELOPE
    assert (DURATION_UUID, bytes([50]), True) in writes, "250 ms is 50 five-ms units"


async def _configure(roach):
    before = roach._settings
    await roach.configure(
        frequency_hz=10, pulse_width_ms=1, duration_ms=250, gain_percent=10
    )
    return before, roach._settings, roach.client.writes


def test_configure_then_turn_is_allowed(board):
    """configure() is the documented way out of a refusal."""
    writes = run(lambda roach: _configure_then_turn(roach))
    assert (TURN_LEFT_UUID, b"\x01", True) in writes


async def _configure_then_turn(roach):
    await roach.configure(
        frequency_hz=10, pulse_width_ms=1, duration_ms=250, gain_percent=10
    )
    await roach.turn("left")
    return roach.client.writes
