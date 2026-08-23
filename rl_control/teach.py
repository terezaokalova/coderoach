"""Teaching loop: camera pose in, stim parameters out, reward for staying responsive."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from interface.camera import Pose, PoseTracker, SimulatedCamera
from interface.roboroach import DEFAULT_DURATION_MS, DEFAULT_FREQUENCY_HZ

from .env import (
    MovementState,
    StimAction,
    clamp_duty,
    movement_from_poses,
    snap_duration_ms,
    wrap_pi,
)
from .policy import PathPolicy


class Stimulator(Protocol):
    async def pulse(self, action: StimAction) -> None: ...


class SilentStim:
    async def pulse(self, action: StimAction) -> None:
        return None


class HabituatingAnimal:
    """Simulated roach whose turn and walk fade when stim parameters repeat."""

    def __init__(
        self,
        start: tuple[float, float, float] = (0.0, 0.0, 0.0),
        memory: int = 6,
    ) -> None:
        self.x, self.y, self.heading_rad = start
        self.t = 0.0
        self.fatigue = 0.0
        self.traces: list[tuple[int, int]] = []
        self.memory = memory
        self._last_dir = ""

    def pose(self) -> Pose:
        return Pose(self.x, self.y, self.t)

    def apply(self, action: StimAction) -> None:
        if self._last_dir and action.direction != self._last_dir:
            self.fatigue *= 0.35
            self.traces.clear()
        self._last_dir = action.direction
        match = self._match(action.frequency_hz, action.duration_ms)
        self.fatigue = min(
            1.0,
            max(
                0.0,
                self.fatigue * 0.9 + 0.07 * match - 0.10 * (1.0 - match),
            ),
        )
        self.traces.append((action.frequency_hz, action.duration_ms))
        self.traces = self.traces[-self.memory :]
        response = math.exp(-2.2 * self.fatigue)
        strength = (action.duration_ms / 250) * (action.frequency_hz / 10)
        turn = 0.55 * max(0.15, strength) * response
        if action.direction == "left":
            self.heading_rad += turn
        else:
            self.heading_rad -= turn
        self._walk(response)

    def coast(self) -> None:
        self._walk(math.exp(-2.2 * self.fatigue))

    def _walk(self, response: float) -> None:
        speed = 0.09 * (0.15 + 0.85 * response)
        self.x += speed * math.cos(self.heading_rad)
        self.y += speed * math.sin(self.heading_rad)
        self.t += 1.0

    def _match(self, frequency_hz: int, duration_ms: int) -> float:
        if not self.traces:
            return 0.0
        weighted = 0.0
        weight_sum = 0.0
        for i, (freq, duration) in enumerate(self.traces):
            weight = 0.55 ** (len(self.traces) - 1 - i)
            freq_n = abs(freq - frequency_hz) / 90.0
            dur_n = abs(duration - duration_ms) / 550.0
            weighted += weight * max(0.0, 1.0 - freq_n - dur_n)
            weight_sum += weight
        return weighted / weight_sum


def _still_speed(
    still_speed: float | None, tracker: PoseTracker | None
) -> dict[str, float]:
    """Resolve the stillness threshold, preserving the previous defaults.

    Passing nothing reproduces exactly what was hardcoded before: 0.01 once a
    camera is driving pose, 0.02 for the simulated animal. Both were tuned
    against Pose.t as a step index; a tracker whose t is in seconds needs its
    own value, which is why this is reachable from the CLI at all.
    """
    if still_speed is not None:
        return {"still_speed": still_speed}
    return {"still_speed": 0.01 if tracker is not None else 0.02}


class AntiHabituationEnv:
    """Sustained turning without freezing. Action is frequency and duration.

    Teach loops pin direction so the inner policy only picks the pulse.
    PathPolicy may change direction or wait on each step.
    """

    def __init__(
        self,
        tracker: PoseTracker,
        stimulator: Stimulator | None = None,
        animal: HabituatingAnimal | None = None,
        direction: str = "left",
        still_speed: float = 0.02,
        max_still: int = 4,
        pulse_width_ms: int = 1,
    ) -> None:
        self.tracker = tracker
        self.stimulator = stimulator or SilentStim()
        self.animal = animal
        self.direction = direction
        self.still_speed = still_speed
        self.max_still = max_still
        self.pulse_width_ms = pulse_width_ms
        self._prev: Pose | None = None
        self._heading = 0.0
        self._still = 0

    @classmethod
    def simulated(
        cls,
        direction: str = "left",
        tracker: PoseTracker | None = None,
        still_speed: float | None = None,
    ) -> "AntiHabituationEnv":
        """Simulated stim. Pass a tracker to read pose from a real camera.

        The simulated animal is still constructed and stepped when a tracker is
        supplied, because the reward depends on it having moved; only the pose
        that the env observes comes from the camera instead.
        """
        animal = HabituatingAnimal()
        return cls(
            tracker=SimulatedCamera(animal) if tracker is None else tracker,
            animal=animal,
            direction=direction,
            **_still_speed(still_speed, tracker),
        )

    @classmethod
    def wired(
        cls,
        stimulator: Stimulator,
        direction: str = "left",
        tracker: PoseTracker | None = None,
        still_speed: float | None = None,
    ) -> "AntiHabituationEnv":
        """Real stim. Pass a camera tracker to drop the simulated animal."""
        if tracker is not None:
            return cls(
                tracker=tracker,
                stimulator=stimulator,
                direction=direction,
                max_still=12,
                **_still_speed(still_speed, tracker),
            )
        animal = HabituatingAnimal()
        return cls(
            tracker=SimulatedCamera(animal),
            stimulator=stimulator,
            animal=animal,
            direction=direction,
            **_still_speed(still_speed, tracker),
        )

    async def reset(self) -> MovementState:
        self._prev = await self.tracker.read()
        self._heading = 0.0
        self._still = 0
        return MovementState(
            x=self._prev.x,
            y=self._prev.y,
            vx=0.0,
            vy=0.0,
            speed=0.0,
            heading_rad=0.0,
            turn_rate_rad=0.0,
            still_steps=0,
            last_frequency_hz=0,
            last_duration_ms=0,
        )

    def bind_action(self, frequency_hz: int, duration_ms: int) -> StimAction:
        frequency_hz, pulse_width_ms = clamp_duty(frequency_hz, self.pulse_width_ms)
        return StimAction(
            direction=self.direction,
            frequency_hz=frequency_hz,
            pulse_width_ms=pulse_width_ms,
            duration_ms=snap_duration_ms(duration_ms),
        )

    async def step(self, action: StimAction) -> tuple[MovementState, float, bool]:
        if action.direction == "wait":
            if self.animal is not None:
                self.animal.coast()
            frequency_hz = 0
            duration_ms = 0
        else:
            self.direction = action.direction
            action = self.bind_action(action.frequency_hz, action.duration_ms)
            frequency_hz = action.frequency_hz
            duration_ms = action.duration_ms
            if self.animal is not None:
                self.animal.apply(action)
            await self.stimulator.pulse(action)
        pose = await self.tracker.read()
        if self._prev is None:
            self._prev = pose
        state = movement_from_poses(
            self._prev,
            pose,
            self._heading,
            self._still,
            frequency_hz,
            duration_ms,
            self.still_speed,
        )
        self._prev = pose
        self._heading = state.heading_rad
        self._still = state.still_steps
        if action.direction == "wait":
            reward = 0.3 * state.speed
        else:
            sign = 1.0 if self.direction == "left" else -1.0
            reward = sign * state.turn_rate_rad + 0.3 * state.speed
        if state.still_steps > 0:
            reward -= 0.6
        done = self.max_still > 0 and state.still_steps >= self.max_still
        return state, reward, done


@dataclass(frozen=True)
class StepLog:
    step: int
    state: MovementState
    action: StimAction
    reward: float
    progress_rad: float = 0.0
    left_rad: float = 0.0
    right_rad: float = 0.0
    warmup: bool = False


class TeachPolicy(Protocol):
    def act(self, state: MovementState) -> StimAction: ...

    def update(
        self,
        state: MovementState,
        action: StimAction,
        reward: float,
        next_state: MovementState,
    ) -> None: ...


TURN_TARGET_RAD = math.pi
REVERSAL_PHASES = ("left", "right")
WARMUP_STEPS = 10
MAX_STEP_TURN_RAD = math.radians(35)
MIN_TURN_SPEED = 0.03


def credited_turn(phase: str, delta: float, state: MovementState) -> float:
    """Return plausible signed progress toward the requested turn."""
    signed = delta if phase == "left" else -delta
    if (
        state.still_steps > 0
        or state.speed < MIN_TURN_SPEED
        or abs(delta) > MAX_STEP_TURN_RAD
    ):
        return 0.0
    return signed


def _with_direction(action: StimAction, direction: str) -> StimAction:
    return StimAction(
        direction=direction,
        frequency_hz=action.frequency_hz,
        pulse_width_ms=action.pulse_width_ms,
        duration_ms=action.duration_ms,
    )


async def follow_path(
    env: AntiHabituationEnv,
    policy: PathPolicy,
    max_steps: int,
    on_step: Callable[[StepLog], None] | None = None,
) -> tuple[list[StepLog], bool]:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    state = await env.reset()
    logs: list[StepLog] = []
    for step in range(1, max_steps + 1):
        action = policy.act(state)
        next_state, reward, done = await env.step(action)
        policy.update(state, action, reward, next_state)
        log = StepLog(step, next_state, action, reward)
        logs.append(log)
        if on_step is not None:
            on_step(log)
        state = next_state
        if policy.finished:
            return logs, True
        if done:
            return logs, False
    return logs, False


def format_path_step(log: StepLog, policy: PathPolicy) -> str:
    n = len(policy.waypoints)
    wp = min(policy.index + 1, n)
    dist = 0.0 if policy.finished else policy.observe(log.state).distance
    return (
        f"step {log.step:02d}  wp {wp}/{n}  dist {dist:.3f}  "
        f"{log.action.direction:5s}  "
        f"-> {log.action.frequency_hz} Hz  {log.action.duration_ms} ms  "
        f"r={log.reward:+.2f}"
    )


def summarize_path(logs: list[StepLog], success: bool) -> str:
    status = "SUCCESS" if success else "FAIL"
    return f"{status}  {summarize(logs)}"


async def teach(
    env: AntiHabituationEnv,
    policy: TeachPolicy,
    max_steps: int,
    on_step: Callable[[StepLog], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    target_rad: float = TURN_TARGET_RAD,
) -> tuple[list[StepLog], bool]:
    """Turn 180 degrees each way, starting with the environment direction."""
    if max_steps < 0:
        raise ValueError("max_steps must be 0 (until success) or at least 1")
    if target_rad <= 0:
        raise ValueError("target_rad must be positive")
    state = await env.reset()
    logs: list[StepLog] = []
    turns = {phase: 0.0 for phase in REVERSAL_PHASES}
    phases = (
        REVERSAL_PHASES if env.direction == "left" else tuple(reversed(REVERSAL_PHASES))
    )
    heading = state.heading_rad
    step = 0
    for phase_i, phase in enumerate(phases):
        env.direction = phase
        reheat = getattr(policy, "reheat", None)
        if callable(reheat):
            reheat(phase)
        if phase_i:
            env._still = 0
        phase_step = 0
        while turns[phase] < target_rad:
            step += 1
            phase_step += 1
            if max_steps and step > max_steps:
                return logs, False
            if should_stop is not None and should_stop():
                return logs, False
            warming = phase_step <= WARMUP_STEPS
            if warming:
                action = _with_direction(
                    env.bind_action(DEFAULT_FREQUENCY_HZ, DEFAULT_DURATION_MS),
                    phase,
                )
            else:
                action = _with_direction(policy.act(state), phase)
            next_state, reward, done = await env.step(action)
            delta = wrap_pi(next_state.heading_rad - heading)
            heading = next_state.heading_rad
            credited = credited_turn(phase, delta, next_state)
            turns[phase] += credited
            if warming:
                reward = 0.0
            else:
                reward = credited + 0.3 * next_state.speed
                if next_state.still_steps > 0:
                    reward -= 0.6
                policy.update(state, action, reward, next_state)
            log = StepLog(
                step,
                next_state,
                action,
                reward,
                turns[phase],
                turns["left"],
                turns["right"],
                warming,
            )
            logs.append(log)
            if on_step is not None:
                on_step(log)
            state = next_state
            if done:
                return logs, False
    return logs, True


async def reversal(
    env: AntiHabituationEnv,
    policy: TeachPolicy,
    max_steps: int,
    on_step: Callable[[StepLog, dict[str, float]], None] | None = None,
) -> tuple[list[StepLog], dict[str, float], bool]:
    """Compatibility wrapper around the shared left-then-right teach loop."""
    progress = {phase: 0.0 for phase in REVERSAL_PHASES}

    def report(log: StepLog) -> None:
        progress["left"] = log.left_rad
        progress["right"] = log.right_rad
        if on_step is not None:
            on_step(log, progress)

    logs, success = await teach(env, policy, max_steps, on_step=report)
    return logs, progress, success


def summarize(logs: list[StepLog], success: bool | None = None) -> str:
    if not logs:
        return "empty episode"
    n = len(logs)
    reward = sum(item.reward for item in logs)
    speed = sum(item.state.speed for item in logs) / n
    turn = sum(abs(item.state.turn_rate_rad) for item in logs) / n
    still = sum(1 for item in logs if item.state.still_steps > 0)
    left = math.degrees(logs[-1].left_rad)
    right = math.degrees(logs[-1].right_rad)
    extra = ""
    if success is not None or logs[-1].left_rad or logs[-1].right_rad:
        extra = f"left {left:.0f}/180  right {right:.0f}/180  "
    body = (
        f"steps {n}  {extra}reward {reward:.2f}  "
        f"mean speed {speed:.3f}  mean |turn| {turn:.3f}  "
        f"moving-still {still}/{n}"
    )
    if success is None:
        return body
    status = "SUCCESS" if success else "FAIL"
    return f"{status}  {body}"


def format_teach_step(log: StepLog) -> str:
    phase = "warm" if log.warmup else log.action.direction
    return (
        f"step {log.step:02d}  {phase:5s}  "
        f"spd {log.state.speed:.3f}  "
        f"L {math.degrees(log.left_rad):5.1f}/180  "
        f"R {math.degrees(log.right_rad):5.1f}/180  "
        f"still {log.state.still_steps}  "
        f"-> {log.action.frequency_hz} Hz  {log.action.duration_ms} ms  "
        f"r={log.reward:+.2f}"
    )


def format_reversal_step(log: StepLog, progress: dict[str, float]) -> str:
    left = math.degrees(progress["left"])
    right = math.degrees(progress["right"])
    return (
        f"step {log.step:02d}  {log.action.direction:5s}  "
        f"head {math.degrees(log.state.heading_rad):+6.1f}  "
        f"L {left:5.1f}/180  R {right:5.1f}/180  "
        f"-> {log.action.frequency_hz} Hz  {log.action.duration_ms} ms  "
        f"r={log.reward:+.2f}"
    )


def summarize_reversal(
    logs: list[StepLog], progress: dict[str, float], success: bool
) -> str:
    left = math.degrees(progress["left"])
    right = math.degrees(progress["right"])
    status = "SUCCESS" if success else "FAIL"
    return (
        f"{status}  left {left:.0f}/180  right {right:.0f}/180  " f"steps {len(logs)}"
    )
