"""Pose stream the teaching loop reads. A future overhead camera implements this.

read() returns a continuous x, y and a timestamp. Heading is derived from the
movement vector between two poses, not from the backpack.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    t: float


@dataclass(frozen=True)
class MovementState:
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    heading_rad: float
    turn_rate_rad: float
    still_steps: int
    last_frequency_hz: int
    last_duration_ms: int

    @property
    def vector(self) -> tuple[float, float]:
        return (self.vx, self.vy)

    def as_array(self) -> list[float]:
        return [
            self.vx,
            self.vy,
            self.speed,
            self.turn_rate_rad,
            float(self.still_steps),
            self.last_frequency_hz / 150.0,
            self.last_duration_ms / 1000.0,
        ]


class PoseTracker(Protocol):
    async def read(self) -> Pose: ...


class PoseSource(Protocol):
    def pose(self) -> Pose: ...


def movement_from_poses(
    prev: Pose,
    cur: Pose,
    prev_heading_rad: float,
    still_steps: int,
    last_frequency_hz: int,
    last_duration_ms: int,
    still_speed: float,
) -> MovementState:
    dt = max(cur.t - prev.t, 1e-3)
    vx = (cur.x - prev.x) / dt
    vy = (cur.y - prev.y) / dt
    speed = math.hypot(vx, vy)
    heading = math.atan2(vy, vx) if speed > still_speed else prev_heading_rad
    turn = (heading - prev_heading_rad + math.pi) % (2 * math.pi) - math.pi
    if speed <= still_speed:
        still_steps += 1
    else:
        still_steps = 0
    return MovementState(
        x=cur.x,
        y=cur.y,
        vx=vx,
        vy=vy,
        speed=speed,
        heading_rad=heading,
        turn_rate_rad=turn / dt,
        still_steps=still_steps,
        last_frequency_hz=last_frequency_hz,
        last_duration_ms=last_duration_ms,
    )


class SimulatedCamera:
    """Reads x, y from any object with pose(). Replace with a real PoseTracker later."""

    def __init__(self, source: PoseSource) -> None:
        self.source = source

    async def read(self) -> Pose:
        return self.source.pose()


class KeyboardCamera:
    """Stand-in observer: type x y after each stim until the real camera exists."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()

    async def read(self) -> Pose:
        raw = (
            (await asyncio.to_thread(input, "camera x y  (or abort): "))
            .strip()
            .casefold()
        )
        if raw in {"abort", "quit", "q"}:
            raise KeyboardInterrupt
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError("Type two numbers: x y")
        return Pose(float(parts[0]), float(parts[1]), time.monotonic() - self._t0)
