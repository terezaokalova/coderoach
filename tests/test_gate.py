"""StimGate behaviour, with no Bluetooth and no backpack present."""

from __future__ import annotations

import asyncio
import json

import pytest

from interface import StimulationSettings
from stim import StimGate

# duration_ms is the floor on T_refrac, so the short-stimulus settings let the
# timing tests use a refractory period they can sleep past quickly.
SHORT = StimulationSettings(
    frequency_hz=55,
    pulse_width_ms=5,
    duration_ms=10,
    gain_percent=50,
    random_mode=False,
)
LONG = StimulationSettings(
    frequency_hz=55,
    pulse_width_ms=5,
    duration_ms=500,
    gain_percent=50,
    random_mode=False,
)


class StubRoach:
    """Records what the gate asked for and returns immediately."""

    def __init__(self, settings: StimulationSettings = SHORT) -> None:
        self._settings = settings
        self.configure_calls: list[dict] = []
        self.turns: list[str] = []

    async def configure(self, **kwargs) -> None:
        self.configure_calls.append(kwargs)

    async def read_settings(self) -> StimulationSettings:
        return self._settings

    async def turn(self, direction: str) -> None:
        self.turns.append(direction)


class FailingRoach(StubRoach):
    async def turn(self, direction: str) -> None:
        self.turns.append(direction)
        raise RuntimeError("backpack went away mid-write")


async def build(roach, run_dir, t_refrac_s, settings=SHORT) -> StimGate:
    return await StimGate.create(
        roach=roach,
        t_refrac_s=t_refrac_s,
        settings=settings,
        run_dir=run_dir,
    )


def read_log(gate: StimGate) -> list[dict]:
    lines = gate.log_path.read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_two_requests_outside_refractory_both_fire(tmp_path):
    roach = StubRoach()

    async def scenario():
        gate = await build(roach, tmp_path, t_refrac_s=0.02)
        first = await gate.request("left", "traj", "req-1")
        await asyncio.sleep(0.05)
        second = await gate.request("right", "traj", "req-2")
        return gate, first, second

    gate, first, second = asyncio.run(scenario())

    assert first.accepted and first.n == 1
    assert second.accepted and second.n == 2
    assert gate.n == 2
    assert roach.turns == ["left", "right"]
    assert first.t_write_complete is not None
    assert second.t_write_complete >= first.t_write_complete


def test_second_request_inside_refractory_is_rejected(tmp_path):
    roach = StubRoach()

    async def scenario():
        gate = await build(roach, tmp_path, t_refrac_s=5.0)
        first = await gate.request("left", "traj", "req-1")
        second = await gate.request("left", "traj", "req-2")
        return gate, first, second

    gate, first, second = asyncio.run(scenario())

    assert first.accepted and first.n == 1
    assert not second.accepted
    assert second.reject_reason == "refractory"
    assert second.t_write_complete is None
    assert second.n == 1
    assert gate.n == 1
    # No Bluetooth write reached the backpack for the rejected request.
    assert roach.turns == ["left"]


def test_sources_share_one_refractory_period(tmp_path):
    roach = StubRoach()

    async def scenario():
        gate = await build(roach, tmp_path, t_refrac_s=5.0)
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

    async def scenario():
        gate = await build(roach, tmp_path, t_refrac_s=0.02)
        results = []
        for index, source in enumerate(("traj", "voice", "text", "traj")):
            results.append(await gate.request("left", source, f"req-{index}"))
            await asyncio.sleep(0.05)
        return gate, results

    gate, results = asyncio.run(scenario())

    assert [result.n for result in results] == [1, 2, 3, 4]
    assert all(result.accepted for result in results)
    assert gate.n == 4
    assert len(roach.turns) == 4


def test_construction_raises_when_refractory_shorter_than_duration(tmp_path):
    with pytest.raises(ValueError, match="shorter than"):
        StimGate(
            roach=StubRoach(LONG),
            t_refrac_s=0.4,
            settings=LONG,
            run_dir=tmp_path,
        )


def test_create_raises_when_board_reports_a_longer_duration(tmp_path):
    # The board is the authority on what will fire, so a readback that exceeds
    # the requested duration has to fail the same check.
    roach = StubRoach(LONG)

    with pytest.raises(ValueError, match="readback"):
        asyncio.run(build(roach, tmp_path, t_refrac_s=0.02, settings=SHORT))


def test_every_request_is_logged_with_required_fields(tmp_path):
    roach = StubRoach()

    async def scenario():
        gate = await build(roach, tmp_path, t_refrac_s=5.0)
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
    assert {line["settings_id"] for line in lines} == {gate.settings_id}


def test_sources_json_records_settings_and_keeps_other_modules(tmp_path):
    (tmp_path / "sources.json").write_text(json.dumps({"tracker": {"fps": 30}}))
    roach = StubRoach()

    gate = asyncio.run(build(roach, tmp_path, t_refrac_s=5.0))
    document = json.loads((tmp_path / "sources.json").read_text())

    assert document["tracker"] == {"fps": 30}
    assert document["stim_gate"]["settings_id"] == gate.settings_id
    assert document["stim_gate"]["t_refrac_s"] == 5.0
    assert document["stim_gate"]["settings_readback"]["duration_ms"] == 10
    assert roach.configure_calls == [
        {
            "frequency_hz": 55,
            "pulse_width_ms": 5,
            "duration_ms": 10,
            "gain_percent": 50,
            "random_mode": False,
        }
    ]


def test_failed_write_is_not_counted_and_is_not_retried(tmp_path):
    roach = FailingRoach()

    async def scenario():
        gate = await build(roach, tmp_path, t_refrac_s=0.02)
        with pytest.raises(RuntimeError):
            await gate.request("left", "traj", "req-1")
        return gate

    gate = asyncio.run(scenario())

    assert gate.n == 0
    assert roach.turns == ["left"]
    assert read_log(gate)[0]["reject_reason"] == "write_failed"


def test_unknown_direction_or_source_raises_before_any_write(tmp_path):
    roach = StubRoach()

    async def scenario():
        gate = await build(roach, tmp_path, t_refrac_s=0.02)
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
        gate = await build(roach, tmp_path, t_refrac_s=5.0)
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
    assert all(
        result.reject_reason == "refractory"
        for result in results
        if not result.accepted
    )
