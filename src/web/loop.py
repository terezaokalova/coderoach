"""The single control task: drawn path in, at most one turn request per frame.

There is one of these at a time. Drawing a second path cancels the first before
the second starts, and the cancellation is awaited rather than fired and
forgotten, so two loops can never both hold a request in flight at the gate.

The division of labour is the one :mod:`traj.control` describes: pure pursuit
decides *whether a turn is wanted*, the gate decides *whether one may fire*,
and a rejection is dropped. Nothing here retries, backs off, or keeps its own
refractory period -- a second limiter would interact with the gate's in ways
neither could be reasoned about alone.
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from traj.control import PurePursuit, PursuitGains
from web import runs

log = logging.getLogger(__name__)

# A walked path is drawn, not analysed -- the tracker's JSONL is the record.
# Dropping sub-millimetre steps keeps a ten-minute trace from accumulating
# twenty thousand points that no display can resolve.
WALK_MIN_STEP_CM = 0.25

STOP_REQUESTED = "requested"
STOP_REPLACED = "replaced"
STOP_SHUTDOWN = "shutdown"
STOP_FAILED = "failed"


class ControlLoop:
    """Owns the trace task, the pursuit controller, and the walked path."""

    def __init__(self, *, hub, gate, journal, run_dir) -> None:
        self._hub = hub
        self._gate = gate
        self._journal = journal
        self._run_dir = run_dir
        self._task: asyncio.Task | None = None
        self._pursuit: PurePursuit | None = None
        self._trace_id = 0
        self._walked: list[list[float]] = []
        self._t_start: float | None = None
        self._last_decision = None
        self._failure: str | None = None

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def trace_id(self) -> int:
        """Bumped on every start, so a reconnecting page can tell traces apart.

        The page accumulates the walked path from per-frame heads rather than
        being sent the whole list thirty times a second. Without this it would
        have no way to know that the points it is appending belong to a
        different path than the ones already on its canvas.
        """
        return self._trace_id

    @property
    def reference_cm(self) -> list[list[float]]:
        if self._pursuit is None:
            return []
        return [[float(x), float(y)] for x, y in self._pursuit.path]

    @property
    def walked_cm(self) -> list[list[float]]:
        return self._walked

    @property
    def waypoints(self) -> int:
        return 0 if self._pursuit is None else len(self._pursuit.path)

    @property
    def length_cm(self) -> float:
        return 0.0 if self._pursuit is None else self._pursuit.length_cm

    @property
    def gains(self) -> PursuitGains | None:
        return None if self._pursuit is None else self._pursuit.gains

    @property
    def last_decision(self):
        return self._last_decision

    @property
    def failure(self) -> str | None:
        return self._failure

    def set_gains(self, lookahead_cm: float, alpha_dead_rad: float) -> PursuitGains:
        """Retune mid-trace. Raises ValueError on a value pure pursuit refuses.

        ``PursuitGains`` validates in ``__post_init__`` and ``carrot_index``
        reads ``self.gains`` on every frame, so assigning a new one takes
        effect on the next frame without disturbing the path or the walked
        track.
        """
        gains = PursuitGains(
            lookahead_cm=float(lookahead_cm),
            alpha_dead_rad=float(alpha_dead_rad),
        )
        if self._pursuit is not None:
            self._pursuit.gains = gains
        return gains

    async def start(self, reference_cm, gains: PursuitGains) -> None:
        """Follow a new path, cancelling and persisting whatever was running."""
        await self.stop(STOP_REPLACED)

        self._pursuit = PurePursuit(reference_cm, gains)
        self._trace_id += 1
        self._walked = []
        self._t_start = time.monotonic()
        self._last_decision = None
        self._failure = None

        overlay = self._hub.overlay
        overlay.reference_cm = [list(point) for point in self._pursuit.path]
        overlay.walked_cm = self._walked
        overlay.carrot_cm = None

        self._task = asyncio.create_task(self._run())

    async def stop(self, reason: str) -> dict | None:
        """Halt the loop and write the trace out. Safe to call when idle."""
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                # _run records why in self._failure before it propagates.
                log.debug("control loop ended with an error", exc_info=True)

        if self._pursuit is None:
            return None

        trace = self._persist(reason)
        self._pursuit = None
        self._hub.overlay.carrot_cm = None
        return trace

    def _persist(self, reason: str) -> dict:
        gains = self._pursuit.gains
        trace = {
            "t_start": self._t_start,
            "t_end": time.monotonic(),
            "reference_cm": [[float(x), float(y)] for x, y in self._pursuit.path],
            "walked_cm": self._walked,
            "lookahead_cm": gains.lookahead_cm,
            "alpha_dead_rad": gains.alpha_dead_rad,
            "path_length_cm": self._pursuit.length_cm,
            "stop_reason": reason if self._failure is None else STOP_FAILED,
            "failure": self._failure,
        }
        index = runs.append_trace(self._run_dir, trace)
        return {**trace, "index": index}

    async def _run(self) -> None:
        seq = self._hub.seq
        try:
            while True:
                seq, result = await self._hub.next_frame(seq)
                await self._step(result)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            self._journal.record_note("control loop stopped", self._failure)
            raise

    async def _step(self, result) -> None:
        if result.px_hat is None or result.py_hat is None:
            return

        self._record_walked(result.px_hat, result.py_hat)

        decision = self._pursuit.decide(
            result.px_hat, result.py_hat, result.theta, result.heading_valid
        )
        self._last_decision = decision
        self._hub.overlay.carrot_cm = decision.carrot_cm

        if not decision.wants_request:
            return

        request_id = f"traj-{uuid4().hex[:8]}"
        try:
            stim = await self._gate.request(decision.direction, "traj", request_id)
        except RuntimeError as exc:
            # turn() refused before writing anything: its own hardware envelope
            # is narrower than the gate's window right now. The gate has
            # already logged the rejection. Nothing was delivered and the loop
            # asks again next frame, which is what a rejection means here.
            self._journal.record_rejection(
                request_id, "traj", decision.direction, "safety_guard", str(exc)
            )
            return

        self._journal.record(stim, decision)

    def _record_walked(self, x: float, y: float) -> None:
        if self._walked:
            last_x, last_y = self._walked[-1]
            if (
                abs(x - last_x) < WALK_MIN_STEP_CM
                and abs(y - last_y) < WALK_MIN_STEP_CM
            ):
                return
        self._walked.append([float(x), float(y)])
