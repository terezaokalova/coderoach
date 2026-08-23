"""One camera, many readers.

``TrajectoryTracker.process_once`` grabs a frame, filters it, and appends one
JSONL line per call. Calling it from the video stream, the state stream, and
the control loop would write that line three times and leave the three fighting
over a capture buffer that holds exactly one frame. So it is called in one
place, here, and everyone else reads the result.

Readers always get the newest frame and never a backlog. An iPad that falls
behind on hotel wifi should skip frames; replaying stale ones several seconds
late is worse than dropping them, because the operator is steering a live
animal by what the video shows.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from itertools import pairwise

log = logging.getLogger(__name__)

JPEG_QUALITY = 70

# Drawn over the preview that traj.track.render already produced. Distinct from
# its palette so the reference and the walked track cannot be mistaken for the
# tracker's own overlays.
COLOUR_REFERENCE = (255, 128, 0)
COLOUR_WALKED = (128, 255, 128)
COLOUR_CARROT = (0, 255, 255)


@dataclass
class PathOverlay:
    """What the control loop wants drawn on the video, in arena centimetres.

    Mutated by the loop and read by the encoder. Both run on the event loop, so
    the only interleaving point is an await, and neither holds a half-written
    list across one.
    """

    reference_cm: list[list[float]] = field(default_factory=list)
    walked_cm: list[list[float]] = field(default_factory=list)
    carrot_cm: tuple[float, float] | None = None

    def clear(self) -> None:
        self.reference_cm = []
        self.walked_cm = []
        self.carrot_cm = None


class FrameHub:
    """Drains the tracker in one task and publishes the newest frame."""

    def __init__(self, tracker, *, v_min_cm_s: float, video_width: int) -> None:
        self._tracker = tracker
        self._v_min_cm_s = v_min_cm_s
        self._video_width = video_width
        self._latest = None
        self._seq = 0
        self._fresh = asyncio.Event()
        self._failure: BaseException | None = None
        self._task: asyncio.Task | None = None
        self._jpeg_seq = -1
        self._jpeg: bytes | None = None
        self._encode_lock = asyncio.Lock()
        self.overlay = PathOverlay()

    @property
    def homography(self):
        return self._tracker.homography

    @property
    def log_path(self):
        return self._tracker.log_path

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def latest(self):
        return self._latest

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._pump())

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Already recorded in self._failure and already raised to any
                # waiting reader. Surfacing it again here would mask the
                # caller's own shutdown path.
                log.debug("frame pump ended with an error", exc_info=True)
            self._task = None
        self._tracker.close()

    async def _pump(self) -> None:
        try:
            while True:
                result = await asyncio.to_thread(self._tracker.process_once)
                if result is None:
                    raise RuntimeError("Camera stopped delivering frames")
                self._publish(result)
        except BaseException as exc:  # surfaced to readers, never swallowed
            self._failure = exc
            self._wake()
            raise

    def _publish(self, result) -> None:
        self._latest = result
        self._seq += 1
        self._wake()

    def _wake(self) -> None:
        # The event is replaced rather than cleared. Clearing it afterwards can
        # drop a set() that lands in between and park every reader forever.
        self._fresh.set()
        self._fresh = asyncio.Event()

    async def next_frame(self, seen_seq: int):
        """The newest frame later than ``seen_seq``, waiting if there is none.

        Returns ``(seq, result)``. A reader that was away for a while gets the
        current frame, not the one it would have seen had it kept up.
        """
        while True:
            if self._failure is not None:
                raise self._failure
            if self._seq > seen_seq and self._latest is not None:
                return self._seq, self._latest
            # Captured before awaiting: _wake() swaps in a fresh event, and
            # reading the attribute at await time could pick up the new one
            # and miss the set() that just happened.
            waiter = self._fresh
            await waiter.wait()

    async def jpeg(self, seq: int, result) -> bytes:
        """Encode one frame, once, however many video clients are watching."""
        async with self._encode_lock:
            if self._jpeg_seq == seq and self._jpeg is not None:
                return self._jpeg
            data = await asyncio.to_thread(self._encode, result)
            self._jpeg_seq = seq
            self._jpeg = data
            return data

    def _encode(self, result) -> bytes:
        import cv2

        from traj.track import render

        canvas = render(result, self.homography, self._v_min_cm_s)
        self._draw_overlay(canvas, cv2)

        height, width = canvas.shape[:2]
        if width > self._video_width:
            scale = self._video_width / width
            canvas = cv2.resize(
                canvas,
                (self._video_width, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, buffer = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed on a preview frame")
        return buffer.tobytes()

    def _draw_overlay(self, canvas, cv2) -> None:
        overlay = self.overlay
        self._polyline(canvas, cv2, overlay.reference_cm, COLOUR_REFERENCE, 2)
        self._polyline(canvas, cv2, overlay.walked_cm, COLOUR_WALKED, 2)
        if overlay.carrot_cm is not None:
            points = self._to_px(canvas, [list(overlay.carrot_cm)])
            if points:
                cv2.circle(canvas, points[0], 6, COLOUR_CARROT, 2)

    def _polyline(self, canvas, cv2, points_cm, colour, thickness: int) -> None:
        if len(points_cm) < 2:
            return
        points = self._to_px(canvas, points_cm)
        for start, end in pairwise(points):
            cv2.line(canvas, start, end, colour, thickness)

    def _to_px(self, canvas, points_cm) -> list[tuple[int, int]]:
        height, width = canvas.shape[:2]
        # A drawn path can leave the calibrated quad, and a point behind the
        # homography's horizon comes back enormous. cv2's drawing calls raise
        # on those, so they are clamped to a wide box around the frame rather
        # than dropped: a clamped line still shows the operator where the path
        # left the arena.
        limit_x = width * 4
        limit_y = height * 4
        out = []
        for point in self.homography.to_px(points_cm):
            x, y = float(point[0]), float(point[1])
            if math.isnan(x) or math.isnan(y):  # degenerate projection
                continue
            out.append(
                (
                    int(max(-limit_x, min(limit_x, x))),
                    int(max(-limit_y, min(limit_y, y))),
                )
            )
        return out
