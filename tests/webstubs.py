"""Stand-ins for the camera and the backpack, shared by the web tests.

Both are deliberately dumb. The point of every test that uses them is what the
code under test does with a frame or with the gate's answer, not the fidelity
of the frame or the answer itself.
"""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

from web.hub import PathOverlay


def frame(x: float, y: float = 0.0, theta: float | None = math.pi / 2, valid=True):
    """A stand-in FrameResult. The control loop reads four fields, no more."""
    return SimpleNamespace(
        t_frame=x, px_hat=x, py_hat=y, theta=theta, heading_valid=valid
    )


class FakeHub:
    """Replays a fixed list of frames, then parks until cancelled."""

    def __init__(self, frames):
        self.overlay = PathOverlay()
        self.seq = 0
        self.frames = list(frames)
        self.index = 0
        self.exhausted = asyncio.Event()

    async def aclose(self):
        # The runtime closes the hub on shutdown, so the stub has to offer the
        # same handle even though there is no camera behind it.
        self.exhausted.set()

    def reload(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.exhausted = asyncio.Event()

    async def next_frame(self, seen_seq):
        if self.index >= len(self.frames):
            self.exhausted.set()
            await asyncio.Event().wait()
        result = self.frames[self.index]
        self.index += 1
        self.seq += 1
        await asyncio.sleep(0)
        return self.seq, result


class StubGate:
    """Records every request and answers with a scripted outcome.

    ``outcomes`` is consumed in order and the last entry repeats, so a single
    entry means "always this".
    """

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.n = 0

    async def request(self, direction, source, request_id):
        self.requests.append((direction, source, request_id))
        index = min(len(self.requests) - 1, len(self.outcomes) - 1)
        outcome = self.outcomes[index]
        if outcome == "guard":
            # What interface.guard_turn raises: refused before any GATT write.
            raise RuntimeError("Wait 1.4s for the animal to finish the last turn.")
        accepted = outcome == "accept"
        if accepted:
            self.n += 1
        return SimpleNamespace(
            accepted=accepted,
            request_id=request_id,
            source=source,
            direction=direction,
            reject_reason=None if accepted else "refractory",
            n=self.n,
        )
