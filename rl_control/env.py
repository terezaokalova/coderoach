"""Goal-seeking observation and action types plus a simulated roach.

The backpack only stims left or right. Pose has to come from a simulator or a
human observer; the board itself does not report location.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Direction = Literal["left", "right", "wait"]


@dataclass(frozen=True)
class StimAction:
    direction: Direction
    frequency_hz: int = 10
    pulse_width_ms: int = 1
    duration_ms: int = 250


@dataclass(frozen=True)
class Observation:
    heading_error_rad: float
    distance: float


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def format_step(step: int, obs: Observation, action: StimAction) -> str:
    err_deg = math.degrees(obs.heading_error_rad)
    return (
        f"step {step:02d}  dist {obs.distance:.3f}  "
        f"err {err_deg:+6.1f} deg  -> {action.direction}  "
        f"{action.frequency_hz} Hz  {action.pulse_width_ms} ms  "
        f"{action.duration_ms} ms"
    )


def snap_duration_ms(duration_ms: int) -> int:
    snapped = 5 * max(2, min(200, round(duration_ms / 5)))
    return snapped


def clamp_duty(frequency_hz: int, pulse_width_ms: int) -> tuple[int, int]:
    frequency_hz = max(1, min(150, frequency_hz))
    pulse_width_ms = max(1, min(255, pulse_width_ms))
    if pulse_width_ms * frequency_hz > 500:
        pulse_width_ms = max(1, 500 // frequency_hz)
    return frequency_hz, pulse_width_ms


def observation_from_pose(
    x: float, y: float, heading_rad: float, goal: tuple[float, float]
) -> Observation:
    gx, gy = goal
    return Observation(
        heading_error_rad=wrap_pi(math.atan2(gy - y, gx - x) - heading_rad),
        distance=math.hypot(gx - x, gy - y),
    )


class SimWorld:
    """Unicycle kinematics: wait lets it walk; left/right rotate then walk."""

    def __init__(
        self,
        start: tuple[float, float, float] = (0.0, 0.0, 0.0),
        goal: tuple[float, float] = (1.0, 0.4),
        speed: float = 0.08,
        turn_scale: float = 0.4,
    ) -> None:
        self.x, self.y, self.heading_rad = start
        self.goal = goal
        self.speed = speed
        self.turn_scale = turn_scale

    def observe(self) -> Observation:
        return observation_from_pose(self.x, self.y, self.heading_rad, self.goal)

    def step(self, action: StimAction) -> Observation:
        if action.direction != "wait":
            strength = (
                (action.duration_ms / 500)
                * (action.frequency_hz / 55)
                * (action.pulse_width_ms / 5)
            )
            turn = self.turn_scale * max(0.15, strength)
            if action.direction == "left":
                self.heading_rad += turn
            else:
                self.heading_rad -= turn
        self.x += self.speed * math.cos(self.heading_rad)
        self.y += self.speed * math.sin(self.heading_rad)
        return self.observe()
