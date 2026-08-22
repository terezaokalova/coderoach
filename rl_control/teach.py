"""Teaching loop: camera pose in, stim parameters out, reward for staying responsive."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .camera import (
    MovementState,
    Pose,
    PoseTracker,
    SimulatedCamera,
    movement_from_poses,
)
from .env import StimAction, clamp_duty, snap_duration_ms, wrap_pi


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


class AntiHabituationEnv:
    """Sustained turning without freezing. Action is frequency and duration.

    Direction is fixed so the agent spends its capacity on an irregular pulse
    pattern instead of choosing left vs right.
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
    def simulated(cls, direction: str = "left") -> "AntiHabituationEnv":
        animal = HabituatingAnimal()
        return cls(
            tracker=SimulatedCamera(animal),
            animal=animal,
            direction=direction,
        )

    @classmethod
    def wired(
        cls,
        stimulator: Stimulator,
        direction: str = "left",
    ) -> "AntiHabituationEnv":
        """Real stim, simulated pose. Swap tracker later for a camera."""
        animal = HabituatingAnimal()
        return cls(
            tracker=SimulatedCamera(animal),
            stimulator=stimulator,
            animal=animal,
            direction=direction,
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
        action = self.bind_action(action.frequency_hz, action.duration_ms)
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
            action.frequency_hz,
            action.duration_ms,
            self.still_speed,
        )
        self._prev = pose
        self._heading = state.heading_rad
        self._still = state.still_steps
        sign = 1.0 if self.direction == "left" else -1.0
        reward = sign * state.turn_rate_rad + 0.3 * state.speed
        if state.speed <= self.still_speed:
            reward -= 0.6
        done = state.still_steps >= self.max_still
        return state, reward, done


@dataclass(frozen=True)
class StepLog:
    step: int
    state: MovementState
    action: StimAction
    reward: float


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


def _with_direction(action: StimAction, direction: str) -> StimAction:
    return StimAction(
        direction=direction,
        frequency_hz=action.frequency_hz,
        pulse_width_ms=action.pulse_width_ms,
        duration_ms=action.duration_ms,
    )


async def teach(
    env: AntiHabituationEnv,
    policy: TeachPolicy,
    max_steps: int,
    on_step: Callable[[StepLog], None] | None = None,
) -> list[StepLog]:
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
        if done:
            break
    return logs


async def reversal(
    env: AntiHabituationEnv,
    policy: TeachPolicy,
    max_steps: int,
    on_step: Callable[[StepLog, dict[str, float]], None] | None = None,
) -> tuple[list[StepLog], dict[str, float], bool]:
    """Left 180, then right 180. Success only if both sides finish."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    state = await env.reset()
    logs: list[StepLog] = []
    progress = {phase: 0.0 for phase in REVERSAL_PHASES}
    heading = state.heading_rad
    step = 0
    for phase in REVERSAL_PHASES:
        env.direction = phase
        while progress[phase] < TURN_TARGET_RAD:
            step += 1
            if step > max_steps:
                return logs, progress, False
            action = _with_direction(policy.act(state), phase)
            next_state, reward, done = await env.step(action)
            delta = wrap_pi(next_state.heading_rad - heading)
            heading = next_state.heading_rad
            signed = delta if phase == "left" else -delta
            progress[phase] += max(0.0, signed)
            policy.update(state, action, reward, next_state)
            log = StepLog(step, next_state, action, reward)
            logs.append(log)
            if on_step is not None:
                on_step(log, progress)
            state = next_state
            if done:
                return logs, progress, False
    return logs, progress, True


def summarize(logs: list[StepLog]) -> str:
    if not logs:
        return "empty episode"
    n = len(logs)
    reward = sum(item.reward for item in logs)
    speed = sum(item.state.speed for item in logs) / n
    turn = sum(abs(item.state.turn_rate_rad) for item in logs) / n
    still = sum(1 for item in logs if item.state.still_steps > 0)
    return (
        f"steps {n}  reward {reward:.2f}  "
        f"mean speed {speed:.3f}  mean |turn| {turn:.3f}  "
        f"moving-still {still}/{n}"
    )


def format_teach_step(log: StepLog) -> str:
    return (
        f"step {log.step:02d}  spd {log.state.speed:.3f}  "
        f"turn {log.state.turn_rate_rad:+.3f}  still {log.state.still_steps}  "
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
