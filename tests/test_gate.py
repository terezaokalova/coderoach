"""StimGate behaviour, with no Bluetooth and no backpack present."""

from __future__ import annotations

import asyncio
import json

import pytest
from bleak.exc import BleakError
from interface import StimulationSettings
from interface.roboroach import MIN_STIM_INTERVAL_S

from stim import StimGate, settings_id

# Inside the envelope interface/roboroach.py enforces: 1-10 Hz, 1 ms pulse,
# 200-300 ms duration, gain at most 10 percent.
BASE = StimulationSettings(
    frequency_hz=10,
    pulse_width_ms=1,
    duration_ms=250,
    gain_percent=10,
    random_mode=False,
)
ALT = StimulationSettings(
    frequency_hz=5,
    pulse_width_ms=1,
    duration_ms=300,
    gain_percent=10,
    random_mode=False,
)
# Longer than any T_refrac these tests use, for the overlap check only.
OVERLONG = StimulationSettings(
    frequency_hz=10,
    pulse_width_ms=1,
    duration_ms=2500,
    gain_percent=10,
    random_mode=False,
)

T_REFRAC = MIN_STIM_INTERVAL_S


class FakeClock:
    """Deterministic monotonic clock.

    T_refrac cannot go below MIN_STIM_INTERVAL_S, so a real-time test of the
    refractory window would have to sleep for seconds per assertion.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class StubRoach:
    """Records what the gate asked for and returns immediately."""

    def __init__(self, settings: StimulationSettings = BASE) -> None:
        self._settings = settings
        self.configure_calls: list[dict] = []
        self.turns: list[str] = []

    async def configure(self, **kwargs) -> None:
        self.configure_calls.append(kwargs)
        self._settings = StimulationSettings(**kwargs)

    async def read_settings(self) -> StimulationSettings:
        return self._settings

    async def turn(self, direction: str) -> None:
        self.turns.append(direction)


class ClampingRoach(StubRoach):
    """Board that accepts configure but keeps reporting its own settings."""

    async def configure(self, **kwargs) -> None:
        self.configure_calls.append(kwargs)


class GuardedRoach(StubRoach):
    """turn() refused by the hardware envelope guard, before any write."""

    async def turn(self, direction: str) -> None:
        self.turns.append(direction)
        raise RuntimeError("Wait 1.4s for the animal to finish the last turn.")


class DroppedRoach(StubRoach):
    """turn() failed inside bleak, after the write may have gone out."""

    async def turn(self, direction: str) -> None:
        self.turns.append(direction)
        raise BleakError("connection dropped mid-write")


async def build(
    roach,
    run_dir,
    clock,
    t_refrac_s: float = T_REFRAC,
    settings: StimulationSettings = BASE,
) -> StimGate:
    return await StimGate.create(
        roach=roach,
        t_refrac_s=t_refrac_s,
        settings=settings,
        run_dir=run_dir,
        clock=clock,
    )


def read_log(gate: StimGate) -> list[dict]:
    return [json.loads(line) for line in gate.log_path.read_text().splitlines()]


def test_two_requests_outside_refractory_both_fire(tmp_path):
    roach = StubRoach()
    clock = FakeClock()

    async def scenario():
        gate = await build(roach, tmp_path, clock)
        first = await gate.request("left", "traj", "req-1")
        clock.advance(T_REFRAC)
        second = await gate.request("right", "traj", "req-2")
        return gate, first, second

    gate, first, second = asyncio.run(scenario())

    assert first.accepted and first.n == 1
    assert second.accepted and second.n == 2
    assert gate.n == 2
    assert roach.turns == ["left", "right"]


def test_second_request_inside_refractory_is_rejected(tmp_path):
    roach = StubRoach()
    clock = FakeClock()

    async def scenario():
        gate = await build(roach, tmp_path, clock)
        first = await gate.request("left", "traj", "req-1")
        clock.advance(T_REFRAC - 0.01)
        second = await gate.request("left", "traj", "req-2")
        return gate, first, second

    gate, first, second = asyncio.run(scenario())

    assert first.accepted and first.n == 1
    assert not second.accepted
    assert second.reject_reason == "refractory"
    assert second.t_write_complete is None
    assert second.n == 1
    assert gate.n == 1
    assert roach.turns == ["left"]


def test_sources_share_one_refractory_period(tmp_path):
    roach = StubRoach()
    clock = FakeClock()

    async def scenario():
        gate = await build(roach, tmp_path, clock)
        traj = await gate.request("left", "traj", "req-traj")
        voice = await gate.request("right", "voice", "req-voice")
        return gate, traj, voice

    gate, traj, voice = asyncio.run(scenario())

    assert traj.accepted
    assert not voice.accepted
    assert voice.reject_reason == "refractory"
    assert gate.n == 1
    assert roach.turns == ["left"]


def test_n_is_continuous_across_sources(tmp_path):
    roach = StubRoach()
    clock = FakeClock()

    async def scenario():
        gate = await build(roach, tmp_path, clock)
        results = []
        for index, source in enumerate(("traj", "voice", "text", "rl")):
            results.append(await gate.request("left", source, f"req-{index}"))
            clock.advance(T_REFRAC)
        return gate, results

    gate, results = asyncio.run(scenario())

    assert [result.n for result in results] == [1, 2, 3, 4]
    assert all(result.accepted for result in results)
    assert gate.n == 4


def test_construction_raises_when_refractory_shorter_than_duration(tmp_path):
    with pytest.raises(ValueError, match="shorter than"):
        StimGate(
            roach=StubRoach(OVERLONG),
            t_refrac_s=T_REFRAC,
            settings=OVERLONG,
            run_dir=tmp_path,
        )


def test_construction_raises_when_refractory_below_hardware_floor(tmp_path):
    # 0.5 s clears the 250 ms stimulus but not the interface's 2 s floor.
    with pytest.raises(ValueError, match="MIN_STIM_INTERVAL_S"):
        StimGate(
            roach=StubRoach(),
            t_refrac_s=0.5,
            settings=BASE,
            run_dir=tmp_path,
        )


def test_refractory_equal_to_the_hardware_floor_is_allowed(tmp_path):
    gate = StimGate(
        roach=StubRoach(),
        t_refrac_s=MIN_STIM_INTERVAL_S,
        settings=BASE,
        run_dir=tmp_path,
    )
    assert gate.t_refrac_s == MIN_STIM_INTERVAL_S


def test_create_raises_when_board_reports_a_longer_duration(tmp_path):
    # The board is the authority on what will fire, so a readback that exceeds
    # the requested duration has to fail the same check.
    roach = ClampingRoach(OVERLONG)

    with pytest.raises(ValueError, match="readback"):
        asyncio.run(build(roach, tmp_path, FakeClock(), settings=BASE))


def test_safety_guard_does_not_count_and_leaves_the_window_open(tmp_path):
    roach = GuardedRoach()
    clock = FakeClock()

    async def scenario():
        gate = await build(roach, tmp_path, clock)
        with pytest.raises(RuntimeError, match="Wait"):
            await gate.request("left", "rl", "req-1")
        # No clock advance: the guard rejected before any write, so the gate
        # must not be holding a refractory window of its own.
        roach.turn = StubRoach.turn.__get__(roach)
        retry = await gate.request("left", "rl", "req-2")
        return gate, retry

    gate, retry = asyncio.run(scenario())

    lines = read_log(gate)
    assert lines[0]["reject_reason"] == "safety_guard"
    assert lines[0]["accepted"] is False
    assert lines[0]["n"] == 0
    assert retry.accepted and retry.n == 1
    assert gate.n == 1


def test_bleak_failure_is_not_counted_but_holds_the_window(tmp_path):
    roach = DroppedRoach()
    clock = FakeClock()

    async def scenario():
        gate = await build(roach, tmp_path, clock)
        with pytest.raises(BleakError):
            await gate.request("left", "traj", "req-1")
        roach.turn = StubRoach.turn.__get__(roach)
        blocked = await gate.request("left", "traj", "req-2")
        clock.advance(T_REFRAC)
        allowed = await gate.request("left", "traj", "req-3")
        return gate, blocked, allowed

    gate, blocked, allowed = asyncio.run(scenario())

    lines = read_log(gate)
    assert lines[0]["reject_reason"] == "write_failed"
    assert lines[0]["n"] == 0
    # The write may have been delivered, so the window stays shut.
    assert not blocked.accepted and blocked.reject_reason == "refractory"
    assert allowed.accepted and allowed.n == 1
    assert gate.n == 1


def test_settings_override_logs_its_own_settings_id(tmp_path):
    roach = StubRoach()
    clock = FakeClock()

    async def scenario():
        gate = await build(roach, tmp_path, clock)
        first = await gate.request("left", "rl", "req-1", settings=BASE)
        clock.advance(T_REFRAC)
        second = await gate.request("right", "rl", "req-2", settings=ALT)
        return gate, first, second

    gate, first, second = asyncio.run(scenario())

    assert first.settings_id == settings_id(BASE)
    assert second.settings_id == settings_id(ALT)
    assert first.settings_id != second.settings_id

    lines = read_log(gate)
    assert [line["settings_id"] for line in lines] == [
        settings_id(BASE),
        settings_id(ALT),
    ]
    assert roach.configure_calls[-1]["duration_ms"] == 300

    document = json.loads((tmp_path / "sources.json").read_text())
    seen = document["stim_gate"]["settings_seen"]
    assert set(seen) == {settings_id(BASE), settings_id(ALT)}
    assert seen[settings_id(ALT)]["frequency_hz"] == 5


def test_request_without_override_does_not_reconfigure(tmp_path):
    roach = StubRoach()
    clock = FakeClock()

    async def scenario():
        gate = await build(roach, tmp_path, clock)
        return await gate.request("left", "traj", "req-1")

    result = asyncio.run(scenario())

    # One configure at construction, none for the request itself.
    assert len(roach.configure_calls) == 1
    assert result.settings_id == settings_id(BASE)


def test_every_request_is_logged_with_required_fields(tmp_path):
    roach = StubRoach()
    clock = FakeClock()

    async def scenario():
        gate = await build(roach, tmp_path, clock)
        await gate.request("left", "traj", "req-1")
        await gate.request("right", "voice", "req-2")
        return gate

    gate = asyncio.run(scenario())
    lines = read_log(gate)

    assert len(lines) == 2
    expected = {
        "request_id",
        "source",
        "direction",
        "t_request",
        "t_write_complete",
        "n",
        "accepted",
        "reject_reason",
        "settings_id",
    }
    assert all(set(line) == expected for line in lines)
    assert [line["accepted"] for line in lines] == [True, False]
    assert [line["reject_reason"] for line in lines] == [None, "refractory"]


def test_sources_json_records_settings_and_keeps_other_modules(tmp_path):
    (tmp_path / "sources.json").write_text(json.dumps({"tracker": {"fps": 30}}))
    roach = StubRoach()

    gate = asyncio.run(build(roach, tmp_path, FakeClock()))
    document = json.loads((tmp_path / "sources.json").read_text())

    assert document["tracker"] == {"fps": 30}
    assert document["stim_gate"]["settings_id_initial"] == gate.settings_id
    assert document["stim_gate"]["t_refrac_s"] == T_REFRAC
    assert document["stim_gate"]["min_stim_interval_s"] == MIN_STIM_INTERVAL_S
    assert (
        document["stim_gate"]["settings_seen"][gate.settings_id]["duration_ms"] == 250
    )
    assert roach.configure_calls == [
        {
            "frequency_hz": 10,
            "pulse_width_ms": 1,
            "duration_ms": 250,
            "gain_percent": 10,
            "random_mode": False,
        }
    ]


def test_unknown_direction_or_source_raises_before_any_write(tmp_path):
    roach = StubRoach()

    async def scenario():
        gate = await build(roach, tmp_path, FakeClock())
        with pytest.raises(ValueError, match="direction"):
            await gate.request("forward", "traj", "req-1")
        with pytest.raises(ValueError, match="source"):
            await gate.request("left", "keyboard", "req-2")
        return gate

    gate = asyncio.run(scenario())

    assert roach.turns == []
    assert gate.n == 0


class SlowRoach(StubRoach):
    """Holds the write open long enough for a racing caller to interleave."""

    async def turn(self, direction: str) -> None:
        await asyncio.sleep(0.01)
        self.turns.append(direction)


def test_concurrent_requests_cannot_both_pass_the_refractory_check(tmp_path):
    roach = SlowRoach()

    async def scenario():
        gate = await build(roach, tmp_path, FakeClock())
        results = await asyncio.gather(
            *(gate.request("left", "traj", f"req-{index}") for index in range(10))
        )
        return gate, results

    gate, results = asyncio.run(scenario())

    accepted = [result for result in results if result.accepted]
    assert len(accepted) == 1
    assert accepted[0].n == 1
    assert gate.n == 1
    assert roach.turns == ["left"]
