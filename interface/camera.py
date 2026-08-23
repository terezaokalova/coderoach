"""Pose stream from a camera, a keyboard, or a simulated animal.

read() returns x, y and a timestamp. Heading is derived later from the
movement between two poses, not from the backpack. PhonePoseTracker in
interface/track.py is the live iPhone and webcam implementation.
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


class PoseTracker(Protocol):
    async def read(self) -> Pose: ...


class PoseSource(Protocol):
    def pose(self) -> Pose: ...


class PoseSmoother:
    """Time-based low-pass filter for noisy camera positions."""

    def __init__(self, time_constant_s: float = 0.12) -> None:
        if time_constant_s < 0:
            raise ValueError("time_constant_s must be non-negative")
        self.time_constant_s = time_constant_s
        self._pose: Pose | None = None

    def reset(self) -> None:
        self._pose = None

    def update(self, pose: Pose) -> Pose:
        previous = self._pose
        if previous is None or pose.t <= previous.t or self.time_constant_s == 0:
            self._pose = pose
            return pose

        alpha = -math.expm1(-(pose.t - previous.t) / self.time_constant_s)
        self._pose = Pose(
            previous.x + alpha * (pose.x - previous.x),
            previous.y + alpha * (pose.y - previous.y),
            pose.t,
        )
        return self._pose


class SimulatedCamera:
    """Reads x, y from any object with pose()."""

    def __init__(self, source: PoseSource) -> None:
        self.source = source

    async def read(self) -> Pose:
        return self.source.pose()


class KeyboardCamera:
    """Stand-in observer: type x y after each stim."""

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
