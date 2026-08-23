"""Swap this module to try a different controller."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from interface.roboroach import DEFAULT_DURATION_MS, DEFAULT_FREQUENCY_HZ

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


@dataclass(frozen=True)
class BanditTrial:
    direction: str
    frequency_hz: int
    duration_ms: int
    reward: float
    turned: bool


class NoveltyBandit:
    """Q-informed Bayesian bandit with separate evidence for left and right.

    Q tracks continuous reward for each pair. Beta posteriors track whether each
    frequency and duration produced a credited turn. Their mean log-odds tempers
    the naive independence assumption because both factors share each trial.
    """

    def __init__(
        self,
        freqs: tuple[int, ...] = (1, 2, 4, 6, 8, 10),
        durations: tuple[int, ...] = (200, 225, 250, 275, 300),
        direction: str = "left",
        temperature: float = 1.8,
        temp_min: float = 0.25,
        temp_decay: float = 0.97,
        stuck_temperature: float = 1.2,
        alpha: float = 0.55,
        bayes_weight: float = 0.45,
        repeat_penalty: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.actions = [_pulse(f, d, direction) for f in freqs for d in durations]
        self.direction = direction
        self.q = {side: [0.0] * len(self.actions) for side in ("left", "right")}
        factors = tuple(("frequency", f) for f in freqs) + tuple(
            ("duration", d) for d in durations
        )
        self.factor_trials = {
            side: {factor: 0 for factor in factors} for side in ("left", "right")
        }
        self.factor_successes = {
            side: {factor: 0 for factor in factors} for side in ("left", "right")
        }
        self.history: list[BanditTrial] = []
        self.temperature = temperature
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        self.stuck_temperature = stuck_temperature
        self.alpha = alpha
        self.bayes_weight = bayes_weight
        self.repeat_penalty = repeat_penalty
        self._indices = {
            (action.frequency_hz, action.duration_ms): i
            for i, action in enumerate(self.actions)
        }
        self._start = next(
            (
                i
                for i, action in enumerate(self.actions)
                if action.frequency_hz == DEFAULT_FREQUENCY_HZ
                and action.duration_ms == DEFAULT_DURATION_MS
            ),
            0,
        )
        for values in self.q.values():
            values[self._start] = 0.5
        self._last = -1
        self._n = 0
        self._temp = temperature
        self._rng = random.Random(seed)

    def reheat(self, direction: str | None = None) -> None:
        if direction is not None:
            self.direction = direction
        self._n = 0
        self._temp = self.temperature
        self._last = -1

    def _similarity(self, i: int, j: int) -> float:
        a = self.actions[i]
        b = self.actions[j]
        freq_n = abs(a.frequency_hz - b.frequency_hz) / 9.0
        dur_n = abs(a.duration_ms - b.duration_ms) / 100.0
        return max(0.0, 1.0 - freq_n - dur_n)

    def success_probability(
        self, action: StimAction, direction: str | None = None
    ) -> float:
        side = direction or self.direction
        keys = (
            ("frequency", action.frequency_hz),
            ("duration", action.duration_ms),
        )
        probabilities = [
            (self.factor_successes[side][key] + 1) / (self.factor_trials[side][key] + 2)
            for key in keys
        ]
        log_odds = sum(
            math.log(probability / (1.0 - probability)) for probability in probabilities
        ) / len(probabilities)
        return 1.0 / (1.0 + math.exp(-log_odds))

    def _scores(self) -> list[float]:
        scored = []
        for i, value in enumerate(self.q[self.direction]):
            penalty = 0.0
            if self._last >= 0:
                penalty = self.repeat_penalty * self._similarity(i, self._last)
            bayes = self.bayes_weight * (
                2.0 * self.success_probability(self.actions[i]) - 1.0
            )
            scored.append(value + bayes - penalty)
        return scored

    def _softmax_sample(self, scores: list[float], temperature: float) -> int:
        peak = max(scores)
        weights = [
            math.exp((score - peak) / max(temperature, 1e-3)) for score in scores
        ]
        pick = self._rng.random() * sum(weights)
        acc = 0.0
        last = 0
        for i, weight in enumerate(weights):
            acc += weight
            last = i
            if pick <= acc:
                return i
        return last

    def act(self, state: MovementState) -> StimAction:
        self._n += 1
        temperature = max(
            self.temp_min, self.temperature * (self.temp_decay ** (self._n - 1))
        )
        if state.still_steps > 0 or (state.speed < 0.03 and state.last_frequency_hz):
            temperature = max(temperature, self.stuck_temperature)
        self._temp = temperature
        index = self._softmax_sample(self._scores(), temperature)
        self._last = index
        return self.actions[index]

    def update(
        self,
        state: MovementState,
        action: StimAction,
        reward: float,
        next_state: MovementState,
    ) -> None:
        side = action.direction
        if side not in self.q:
            return
        index = self._indices[(action.frequency_hz, action.duration_ms)]
        turned = reward > 0.3 * next_state.speed + 1e-9
        self.q[side][index] += self.alpha * (reward - self.q[side][index])
        for factor in (
            ("frequency", action.frequency_hz),
            ("duration", action.duration_ms),
        ):
            self.factor_trials[side][factor] += 1
            self.factor_successes[side][factor] += int(turned)
        self.history.append(
            BanditTrial(
                direction=side,
                frequency_hz=action.frequency_hz,
                duration_ms=action.duration_ms,
                reward=reward,
                turned=turned,
            )
        )


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
        direction = "left" if error > 0 else "right"
        if isinstance(self.inner, NoveltyBandit):
            self.inner.direction = direction
        pulse = self.inner.act(state)
        return StimAction(
            direction=direction,
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
        q = inner.q[inner.direction]
        rows = [
            f"bandit {inner.direction}  T {inner._temp:.2f}  a {inner.alpha:.2f}  "
            f"{len(inner.actions)} arms",
            f"history {len(inner.history)}  repeat pen {inner.repeat_penalty:.2f}",
        ]
        ranked = sorted(range(len(inner.actions)), key=lambda i: q[i], reverse=True)
        show = ranked[:8]
        if inner._last >= 0 and inner._last not in show:
            show.append(inner._last)
        for i in show:
            action = inner.actions[i]
            mark = "*" if i == inner._last else " "
            rows.append(
                f"{mark}{action.frequency_hz:2d} Hz {action.duration_ms:3d} ms  "
                f"Q {q[i]:+.2f}  P {inner.success_probability(action):.2f}"
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
