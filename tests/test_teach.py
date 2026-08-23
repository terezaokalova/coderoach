"""Teach spec: 180 left then 180 right, without false success from camera jitter."""

from __future__ import annotations

import asyncio
import math

import pytest

import interface.roboroach as roboroach
from interface.camera import Pose
from interface.roboroach import (
    DEFAULT_DURATION_MS,
    DEFAULT_FREQUENCY_HZ,
    DEFAULT_PULSE_WIDTH_MS,
    MAX_DURATION_MS,
    MAX_FREQUENCY_HZ,
    MAX_PULSE_WIDTH_MS,
    MIN_DURATION_MS,
    MIN_STIM_INTERVAL_S,
)

from rl_control.env import MovementState, StimAction
from rl_control.policy import NoveltyBandit, make_teach_policy
from rl_control.teach import (
    MAX_STEP_TURN_RAD,
    MIN_TURN_SPEED,
    TURN_TARGET_RAD,
    WARMUP_STEPS,
    AntiHabituationEnv,
    SilentStim,
    credited_turn,
    teach,
)


class JitterTracker:
    """Tiny box wobble that used to look like 80-160 deg turns."""

    def __init__(self) -> None:
        self.n = 0

    async def read(self) -> Pose:
        self.n += 1
        ang = self.n * math.pi / 2
        return Pose(
            0.5 + 0.02 * math.cos(ang), 0.5 + 0.02 * math.sin(ang), float(self.n)
        )


class FrozenTracker:
    async def read(self) -> Pose:
        return Pose(0.5, 0.5, 0.0)


class OscillatingEnv:
    def __init__(self) -> None:
        self.direction = "left"
        self._still = 0
        self.n = 0

    async def reset(self) -> MovementState:
        return MovementState(0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0)

    def bind_action(self, frequency_hz: int, duration_ms: int) -> StimAction:
        return StimAction(self.direction, frequency_hz, 1, duration_ms)

    async def step(self, action: StimAction) -> tuple[MovementState, float, bool]:
        self.n += 1
        heading = math.radians(30) if self.n % 2 else 0.0
        state = MovementState(
            0,
            0,
            0,
            0,
            0.1,
            heading,
            heading,
            0,
            action.frequency_hz,
            action.duration_ms,
        )
        return state, 0.0, False


class FakeGattClient:
    is_connected = True

    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes, bool]] = []
        self.active = 0
        self.max_active = 0

    async def write_gatt_char(self, uuid: str, value: bytes, response: bool) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.writes.append((uuid, bytes(value), response))
        self.active -= 1


def test_defaults_match_backpack_and_safe_epoch() -> None:
    assert DEFAULT_FREQUENCY_HZ == 10
    assert DEFAULT_DURATION_MS == 250
    assert DEFAULT_PULSE_WIDTH_MS == 1
    assert MIN_STIM_INTERVAL_S == 2.0


def test_backpack_guard_rejects_a_train_inside_two_seconds(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(roboroach, "_SAFETY_PATH", tmp_path / "safety.json")
    monkeypatch.setattr(roboroach.time, "time", lambda: 100.0)
    roboroach.record_turn(DEFAULT_DURATION_MS)
    monkeypatch.setattr(roboroach.time, "time", lambda: 101.9)
    with pytest.raises(RuntimeError, match="Wait"):
        roboroach.guard_turn(DEFAULT_DURATION_MS)
    monkeypatch.setattr(roboroach.time, "time", lambda: 102.0)
    roboroach.guard_turn(DEFAULT_DURATION_MS)


def test_roboroach_serializes_gatt_and_requires_safe_configuration(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(roboroach, "_SAFETY_PATH", tmp_path / "safety.json")

    async def exercise() -> FakeGattClient:
        client = FakeGattClient()
        device = roboroach.RoboRoach()
        device.client = client
        with pytest.raises(RuntimeError, match="settings are unknown"):
            await device.turn("right")
        await asyncio.gather(
            device.configure(frequency_hz=4, duration_ms=225),
            device.configure(frequency_hz=6, duration_ms=275),
        )
        await device.turn("right")
        return client

    client = asyncio.run(exercise())
    assert client.max_active == 1
    assert client.writes[-1] == (roboroach.TURN_RIGHT_UUID, b"\x01", True)


def test_bandit_stays_inside_hardware_envelope() -> None:
    policy = NoveltyBandit()
    for action in policy.actions:
        assert 1 <= action.frequency_hz <= MAX_FREQUENCY_HZ
        assert 1 <= action.pulse_width_ms <= MAX_PULSE_WIDTH_MS
        assert MIN_DURATION_MS <= action.duration_ms <= MAX_DURATION_MS


def test_bayesian_history_tracks_failures_separately_by_direction() -> None:
    policy = NoveltyBandit()
    action = next(
        action
        for action in policy.actions
        if action.frequency_hz == 4 and action.duration_ms == 225
    )
    state = MovementState(0, 0, 0, 0, 0.1, 0, 0, 0, 4, 225)
    baseline = policy.success_probability(action, "left")

    policy.update(state, action, 0.3 * state.speed, state)
    left_probability = policy.success_probability(action, "left")
    assert left_probability < baseline
    assert policy.history[-1].turned is False

    right_action = type(action)(
        "right",
        action.frequency_hz,
        action.pulse_width_ms,
        action.duration_ms,
    )
    policy.update(state, right_action, 0.3, state)
    right_probability = policy.success_probability(action, "right")
    assert right_probability > baseline
    assert right_probability > left_probability
    assert [trial.direction for trial in policy.history] == ["left", "right"]
    assert policy.q["left"] != policy.q["right"]


def test_credited_turn_rejects_jitter_and_accepts_a_real_step() -> None:
    jitter = MovementState(0, 0, 0, 0, 0.02, 0, 0, 0, 10, 250)
    assert credited_turn("left", math.radians(84), jitter) == 0.0
    still = MovementState(0, 0, 0, 0, 0.08, 0, 0, 1, 10, 250)
    assert credited_turn("left", math.radians(20), still) == 0.0
    walking = MovementState(0, 0, 0, 0, 0.08, 0, 0, 0, 10, 250)
    got = credited_turn("left", math.radians(20), walking)
    assert math.isclose(got, math.radians(20), rel_tol=1e-6)
    assert credited_turn("right", math.radians(20), walking) == -math.radians(20)
    assert credited_turn("right", -math.radians(20), walking) == math.radians(20)
    assert credited_turn("left", math.radians(90), walking) == 0.0
    assert MAX_STEP_TURN_RAD == math.radians(35)
    assert MIN_TURN_SPEED == 0.03


def test_teach_reaches_left_then_right_on_a_walking_animal() -> None:
    logs, success = asyncio.run(
        teach(AntiHabituationEnv.simulated(), NoveltyBandit(seed=1), max_steps=120)
    )
    assert success
    assert logs[-1].left_rad >= TURN_TARGET_RAD
    assert logs[-1].right_rad >= TURN_TARGET_RAD
    left = [log for log in logs if log.action.direction == "left"]
    right = [log for log in logs if log.action.direction == "right"]
    assert left and right
    assert all(log.action.direction == "left" for log in left)
    assert all(log.action.direction == "right" for log in right)
    assert max(i for i, log in enumerate(logs) if log.action.direction == "left") < min(
        i for i, log in enumerate(logs) if log.action.direction == "right"
    )


def test_warmup_has_zero_reward_but_counts_physical_turns() -> None:
    policy = NoveltyBandit(seed=2)
    q_before = {side: list(values) for side, values in policy.q.items()}
    logs, _ = asyncio.run(teach(AntiHabituationEnv.simulated(), policy, max_steps=120))
    left_warm = [log for log in logs if log.action.direction == "left"][:WARMUP_STEPS]
    right_warm = [log for log in logs if log.action.direction == "right"][:WARMUP_STEPS]
    assert 0 < len(left_warm) <= WARMUP_STEPS
    assert 0 < len(right_warm) <= WARMUP_STEPS
    for log in left_warm + right_warm:
        assert log.warmup
        assert log.reward == 0.0
        assert log.action.frequency_hz == DEFAULT_FREQUENCY_HZ
        assert log.action.duration_ms == DEFAULT_DURATION_MS
    assert left_warm[-1].left_rad > 0.0
    assert right_warm[-1].right_rad > 0.0
    adaptive_steps = sum(not log.warmup for log in logs)
    assert len(policy.history) == adaptive_steps
    if adaptive_steps:
        assert policy.q != q_before


def test_camera_jitter_does_not_count_as_180() -> None:
    env = AntiHabituationEnv.wired(SilentStim(), tracker=JitterTracker())
    env.max_still = 0
    logs, success = asyncio.run(teach(env, NoveltyBandit(seed=3), max_steps=20))
    assert success is False
    assert logs[-1].left_rad < TURN_TARGET_RAD
    assert logs[-1].right_rad == 0.0
    assert all(abs(log.reward) < 1.0 or log.warmup for log in logs)


def test_oscillation_does_not_accumulate_into_turn_success() -> None:
    logs, success = asyncio.run(
        teach(OscillatingEnv(), NoveltyBandit(seed=3), max_steps=40)
    )
    assert success is False
    assert logs[-1].left_rad < TURN_TARGET_RAD
    assert logs[-1].right_rad == 0.0


def test_still_animal_does_not_abort_when_max_still_is_off() -> None:
    env = AntiHabituationEnv.wired(SilentStim(), tracker=FrozenTracker())
    env.max_still = 0
    logs, success = asyncio.run(teach(env, NoveltyBandit(seed=4), max_steps=15))
    assert success is False
    assert len(logs) == 15
    assert logs[-1].left_rad == 0.0


def test_live_cli_defaults_are_bounded_bandit() -> None:
    import sys
    from unittest.mock import patch

    from rl_control.run import main

    captured: dict = {}

    async def fake_live(args):
        captured["args"] = args

    old = sys.argv
    sys.argv = [
        "rl_control",
        "teach",
        "--live",
        "--source",
        "phone",
        "--run-dir",
        "runs/demo",
    ]
    try:
        with patch("rl_control.live.run_live_teach", fake_live):
            main()
    finally:
        sys.argv = old
    args = captured["args"]
    assert args.policy == "bandit"
    assert args.max_steps == 150
    assert args.cooldown == 2.0
    assert args.source == "phone"


def test_make_teach_policy_default_is_bandit() -> None:
    policy = make_teach_policy("bandit")
    assert isinstance(policy, NoveltyBandit)
