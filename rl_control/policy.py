"""Swap this module to try a different controller."""

from __future__ import annotations

import math
import random
from typing import Protocol

from .env import (
    MovementState,
    Observation,
    StimAction,
    clamp_duty,
    observation_from_pose,
    snap_duration_ms,
)


class Policy(Protocol):
    def act(self, observation: Observation) -> StimAction: ...


def _lerp(lo: float, hi: float, frac: float) -> float:
    frac = min(1.0, max(0.0, frac))
    return lo + (hi - lo) * frac


class HeadingPolicy:
    """Turn toward the goal; wait when aligned so the animal walks itself.

    Stim strength scales with heading error and stays inside conservative
    backpack limits. Swap this class for a learned policy later.
    """

    def __init__(
        self,
        arrive_radius: float = 0.12,
        deadband_rad: float = 0.22,
        freq_range: tuple[int, int] = (2, 10),
        pulse_range: tuple[int, int] = (1, 1),
        duration_range: tuple[int, int] = (200, 300),
    ) -> None:
        self.arrive_radius = arrive_radius
        self.deadband_rad = deadband_rad
        self.freq_range = freq_range
        self.pulse_range = pulse_range
        self.duration_range = duration_range

    def act(self, observation: Observation) -> StimAction:
        if observation.distance <= self.arrive_radius:
            return StimAction("wait", frequency_hz=0, pulse_width_ms=0, duration_ms=0)
        error = observation.heading_error_rad
        if abs(error) <= self.deadband_rad:
            return StimAction("wait", frequency_hz=0, pulse_width_ms=0, duration_ms=0)

        frac = min(1.0, abs(error) / (math.pi / 2))
        frequency_hz, pulse_width_ms = clamp_duty(
            int(round(_lerp(*self.freq_range, frac))),
            int(round(_lerp(*self.pulse_range, frac))),
        )
        return StimAction(
            direction="left" if error > 0 else "right",
            frequency_hz=frequency_hz,
            pulse_width_ms=pulse_width_ms,
            duration_ms=snap_duration_ms(int(round(_lerp(*self.duration_range, frac)))),
        )


IRREGULAR_PAIRS = (
    (2, 200),
    (5, 250),
    (8, 300),
    (10, 200),
    (3, 300),
    (10, 250),
)


def _pulse(frequency_hz: int, duration_ms: int, direction: str = "left") -> StimAction:
    frequency_hz, pulse_width_ms = clamp_duty(frequency_hz, 1)
    return StimAction(
        direction=direction,
        frequency_hz=frequency_hz,
        pulse_width_ms=pulse_width_ms,
        duration_ms=snap_duration_ms(duration_ms),
    )


class StaticPulsePolicy:
    """Baseline: the same pulse every step. Habituates quickly."""

    def __init__(
        self, frequency_hz: int = 10, duration_ms: int = 250, direction: str = "left"
    ) -> None:
        self.action = _pulse(frequency_hz, duration_ms, direction)

    def act(self, state: MovementState) -> StimAction:
        return self.action

    def update(
        self,
        state: MovementState,
        action: StimAction,
        reward: float,
        next_state: MovementState,
    ) -> None:
        return None


class IrregularPulsePolicy:
    """Hand-built anti-habituation pattern: hop across distant freq/duration pairs."""

    def __init__(
        self,
        pairs: tuple[tuple[int, int], ...] = IRREGULAR_PAIRS,
        direction: str = "left",
    ) -> None:
        self.pairs = pairs
        self.direction = direction
        self._i = 0

    def act(self, state: MovementState) -> StimAction:
        if state.speed < 0.03 and state.last_frequency_hz:
            self._i = (self._i + 2) % len(self.pairs)
        freq, duration = self.pairs[self._i]
        self._i = (self._i + 1) % len(self.pairs)
        return _pulse(freq, duration, self.direction)

    def update(
        self,
        state: MovementState,
        action: StimAction,
        reward: float,
        next_state: MovementState,
    ) -> None:
        return None


class NoveltyBandit:
    """Tiny learner: epsilon-greedy over freq/duration, with a repeat penalty.

    Repeating a pulse tanks later reward in the habituating sim, so the bandit
    is pushed toward an irregular schedule without that schedule being hardcoded.
    """

    def __init__(
        self,
        freqs: tuple[int, ...] = (2, 5, 8, 10),
        durations: tuple[int, ...] = (200, 250, 300),
        direction: str = "left",
        epsilon: float = 0.35,
        alpha: float = 0.35,
        repeat_penalty: float = 0.45,
        seed: int = 0,
    ) -> None:
        self.actions = [_pulse(f, d, direction) for f in freqs for d in durations]
        self.q = [0.0] * len(self.actions)
        self.epsilon = epsilon
        self.alpha = alpha
        self.repeat_penalty = repeat_penalty
        self._last = -1
        self._rng = random.Random(seed)

    def _similarity(self, i: int, j: int) -> float:
        a = self.actions[i]
        b = self.actions[j]
        freq_n = abs(a.frequency_hz - b.frequency_hz) / 90.0
        dur_n = abs(a.duration_ms - b.duration_ms) / 550.0
        return max(0.0, 1.0 - freq_n - dur_n)

    def act(self, state: MovementState) -> StimAction:
        if self._rng.random() < self.epsilon:
            index = self._rng.randrange(len(self.actions))
        else:
            scored = []
            for i, value in enumerate(self.q):
                penalty = 0.0
                if self._last >= 0:
                    penalty = self.repeat_penalty * self._similarity(i, self._last)
                scored.append(value - penalty)
            index = max(range(len(scored)), key=scored.__getitem__)
        self._last = index
        return self.actions[index]

    def update(
        self,
        state: MovementState,
        action: StimAction,
        reward: float,
        next_state: MovementState,
    ) -> None:
        index = self._last
        if index < 0:
            return
        self.q[index] += self.alpha * (reward - self.q[index])


class PathPolicy:
    """High-level path; low-level stim. Inner policy never chooses direction.

    Waypoints pick left, right, or wait. The wrapped teach policy only picks
    frequency and duration so the same anti-habituation controller can drive
    any route.
    """

    def __init__(
        self,
        waypoints: tuple[tuple[float, float], ...],
        inner,
        arrive_radius: float = 0.12,
        deadband_rad: float = 0.22,
    ) -> None:
        if not waypoints:
            raise ValueError("path needs at least one waypoint")
        self.waypoints = waypoints
        self.inner = inner
        self.arrive_radius = arrive_radius
        self.deadband_rad = deadband_rad
        self.index = 0

    @property
    def finished(self) -> bool:
        return self.index >= len(self.waypoints)

    def observe(self, state: MovementState) -> Observation:
        if self.finished:
            return Observation(heading_error_rad=0.0, distance=0.0)
        return observation_from_pose(
            state.x, state.y, state.heading_rad, self.waypoints[self.index]
        )

    def act(self, state: MovementState) -> StimAction:
        self._advance(state)
        if self.finished:
            return StimAction("wait", frequency_hz=0, pulse_width_ms=0, duration_ms=0)
        error = self.observe(state).heading_error_rad
        if abs(error) <= self.deadband_rad:
            return StimAction("wait", frequency_hz=0, pulse_width_ms=0, duration_ms=0)
        pulse = self.inner.act(state)
        return StimAction(
            direction="left" if error > 0 else "right",
            frequency_hz=pulse.frequency_hz,
            pulse_width_ms=pulse.pulse_width_ms,
            duration_ms=pulse.duration_ms,
        )

    def update(
        self,
        state: MovementState,
        action: StimAction,
        reward: float,
        next_state: MovementState,
    ) -> None:
        self._advance(next_state)
        if action.direction == "wait":
            return
        self.inner.update(state, action, reward, next_state)

    def _advance(self, state: MovementState) -> None:
        while not self.finished and self.observe(state).distance <= self.arrive_radius:
            self.index += 1


def make_teach_policy(name: str, direction: str = "left"):
    if name == "static":
        return StaticPulsePolicy(direction=direction)
    if name == "irregular":
        return IrregularPulsePolicy(direction=direction)
    if name == "bandit":
        return NoveltyBandit(direction=direction)
    raise ValueError("policy must be static, irregular, or bandit")


def status_text(policy) -> str:
    inner = policy.inner if isinstance(policy, PathPolicy) else policy
    if isinstance(inner, NoveltyBandit):
        rows = [
            f"bandit  eps {inner.epsilon:.2f}  a {inner.alpha:.2f}",
            f"repeat pen {inner.repeat_penalty:.2f}",
        ]
        for i, (action, q) in enumerate(zip(inner.actions, inner.q)):
            mark = "*" if i == inner._last else " "
            rows.append(
                f"{mark}{action.frequency_hz:2d} Hz {action.duration_ms:3d} ms  "
                f"Q {q:+.2f}"
            )
        return "\n".join(rows)
    if isinstance(inner, IrregularPulsePolicy):
        nxt = inner.pairs[inner._i % len(inner.pairs)]
        return (
            f"irregular  idx {inner._i % len(inner.pairs)}/{len(inner.pairs)}\n"
            f"next {nxt[0]} Hz  {nxt[1]} ms"
        )
    if isinstance(inner, StaticPulsePolicy):
        action = inner.action
        return (
            f"static  {action.frequency_hz} Hz  {action.duration_ms} ms  "
            f"{action.direction}"
        )
    if policy is None:
        return ""
    return type(inner).__name__
