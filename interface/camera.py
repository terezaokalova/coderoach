"""Pose stream from a camera, a keyboard, or a simulated animal.

read() returns x, y and a timestamp. Heading is derived later from the
movement between two poses, not from the backpack. PhonePoseTracker in
interface/track.py is the live iPhone and webcam implementation.
"""

from __future__ import annotations

import asyncio
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
