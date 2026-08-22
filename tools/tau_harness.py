"""Measure the loop delay from frame grab to Bluetooth write completion.

tau is the interval the trajectory controller has to predict the pose forward
by, so it is measured rather than assumed. One rep grabs a frame, runs the
detect path for its real per-frame cost, and asks the gate for one
stimulation. tau is the gate's write-completion stamp minus the grab stamp.

Run with the backpack powered and off the animal. The default gain of 0
percent asks the board for zero stimulation amplitude, so a rep should cost
nothing behaviourally while producing Bluetooth traffic identical to a real
one. --backpack-off-animal has to be passed explicitly: 100 reps is a bounded
sequence, but it is still 100 turn commands.

This tool reports the distribution. It does not choose which percentile the
controller should use.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from interface import StimulationSettings
from interface.roboroach import FREQUENCY_UUID

from stim import StimGate

LOG_NAME = "tau_harness.jsonl"
SUMMARY_NAME = "tau_summary.json"

OBSERVED = "observed"
PREDICTED = "predicted"


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    area_px: float


@dataclass(frozen=True)
class Rep:
    index: int
    request_id: str
    t_frame_grab: float
    t_detect_done: float
    t_request: float
    t_write_complete: float | None
    tau_s: float | None
    detected: bool
    centroid_x: float | None
    centroid_y: float | None
    area_px: float | None
    accepted: bool
    n: int
    reject_reason: str | None
    keepalive_affected: bool


def detect_centroid(
    frame: np.ndarray,
    lower: Sequence[int],
    upper: Sequence[int],
    min_area_px: float,
) -> Detection | None:
    """Bare HSV threshold plus centroid.

    traj/track.py does not exist yet. This stands in so the measured tau
    includes real per-frame compute instead of an empty loop. When the real
    tracker lands, swap the detect callable at the call site in main; nothing
    here should choose between them.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    moments = cv2.moments(mask, binaryImage=True)
    area = float(moments["m00"])
    if area < min_area_px:
        return None
    return Detection(moments["m10"] / area, moments["m01"] / area, area)


class Camera:
    """Frame source that stamps the grab return, not the retrieve return."""

    def __init__(self, index: int) -> None:
        self._capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index}. On macOS the terminal "
                "needs camera permission, and the Continuity Camera has to be "
                "awake and in range."
            )
        # One-frame buffer, so grab() returns the newest frame instead of
        # replaying a queue. A replayed queue would understate tau.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def describe(self) -> str:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        return f"{width}x{height} at {fps:.1f} fps nominal"

    def _grab_blocking(self) -> tuple[bool, float, np.ndarray | None]:
        grabbed = self._capture.grab()
        t_frame_grab = time.monotonic()
        if not grabbed:
            return False, t_frame_grab, None
        retrieved, frame = self._capture.retrieve()
        return bool(retrieved), t_frame_grab, frame

    async def grab(self) -> tuple[bool, float, np.ndarray | None]:
        return await asyncio.to_thread(self._grab_blocking)

    def release(self) -> None:
        self._capture.release()


def _is_frequency_characteristic(specifier: Any) -> bool:
    return str(getattr(specifier, "uuid", specifier)).casefold() == FREQUENCY_UUID


class KeepaliveWatch:
    """Record when keep_alive() touches the shared connection.

    keep_alive() exposes no firing times, so the only way to observe them
    without editing interface/roboroach.py is to watch the traffic it produces
    on the client the RoboRoach already owns. These wrappers time the call and
    forward it unchanged; they only record operations on the frequency
    characteristic, which is what the heartbeat rewrites.
    """

    def __init__(self) -> None:
        self.windows: list[tuple[float, float]] = []
        self._client: Any | None = None
        self._originals: dict[str, Callable[..., Any]] = {}

    def attach(self, roach: Any) -> bool:
        client = getattr(roach, "client", None)
        if client is None:
            return False
        for name in ("read_gatt_char", "write_gatt_char"):
            original = getattr(client, name, None)
            if original is None:
                self.detach()
                return False
            self._originals[name] = original
            setattr(client, name, self._wrap(original))
        self._client = client
        return True

    def _wrap(self, original: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(specifier: Any, *args: Any, **kwargs: Any) -> Any:
            if not _is_frequency_characteristic(specifier):
                return await original(specifier, *args, **kwargs)
            t_begin = time.monotonic()
            try:
                return await original(specifier, *args, **kwargs)
            finally:
                self.windows.append((t_begin, time.monotonic()))

        return wrapper

    def detach(self) -> None:
        if self._client is not None:
            for name, original in self._originals.items():
                setattr(self._client, name, original)
        self._client = None
        self._originals.clear()


def predicted_windows(
    t_start: float,
    interval_s: float,
    t_end: float,
    guard_s: float,
) -> list[tuple[float, float]]:
    """Where keepalive ticks should land if its firing times cannot be seen."""
    windows = []
    tick = t_start + interval_s
    while tick <= t_end:
        windows.append((tick - guard_s, tick + guard_s))
        tick += interval_s
    return windows


def overlaps_any(
    start: float, end: float, windows: Sequence[tuple[float, float]]
) -> bool:
    return any(
        window_start <= end and start <= window_end
        for window_start, window_end in windows
    )


def cluster_windows(
    windows: Sequence[tuple[float, float]],
    gap_s: float = 1.0,
) -> list[tuple[float, float]]:
    """Group the read and the write of one heartbeat into a single tick."""
    ordered = sorted(windows)
    ticks: list[tuple[float, float]] = []
    for start, end in ordered:
        if ticks and start - ticks[-1][1] <= gap_s:
            ticks[-1] = (ticks[-1][0], max(ticks[-1][1], end))
        else:
            ticks.append((start, end))
    return ticks


def summarise(taus: Sequence[float]) -> dict[str, float | int]:
    """Order statistics only. A mean would be dominated by the tail."""
    array = np.asarray(taus, dtype=float)
    q1, median, q3, p95 = (
        float(value) for value in np.percentile(array, [25, 50, 75, 95])
    )
    return {
        "n": int(array.size),
        "median_s": median,
        "q1_s": q1,
        "q3_s": q3,
        "iqr_s": q3 - q1,
        "p95_s": p95,
        "min_s": float(array.min()),
        "max_s": float(array.max()),
    }


def write_jsonl(path: Path, reps: Sequence[Rep]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        for rep in reps:
            handle.write(json.dumps(asdict(rep)) + "\n")
    temporary.replace(path)


async def run_reps(
    *,
    gate: StimGate,
    grab: Callable[[], Awaitable[tuple[bool, float, np.ndarray | None]]],
    detect: Callable[[np.ndarray], Detection | None],
    reps: int,
    t_refrac_s: float,
    settle_s: float,
    log_path: Path,
    windows: Callable[[], Sequence[tuple[float, float]]],
) -> list[Rep]:
    """One rep: grab, stamp, detect, stimulate, take the gate's stamp."""
    collected: list[Rep] = []
    last_write: float | None = None

    for index in range(reps):
        if last_write is not None:
            remaining = t_refrac_s + settle_s - (time.monotonic() - last_write)
            if remaining > 0:
                await asyncio.sleep(remaining)

        ok, t_frame_grab, frame = await grab()
        if not ok or frame is None:
            raise RuntimeError(f"Camera returned no frame on rep {index}")

        detection = detect(frame)
        t_detect_done = time.monotonic()

        direction = "left" if index % 2 == 0 else "right"
        request_id = f"tau-{index:03d}"
        t_request = time.monotonic()
        result = await gate.request(direction, "traj", request_id)

        if result.accepted and result.t_write_complete is not None:
            last_write = result.t_write_complete
            tau_s: float | None = result.t_write_complete - t_frame_grab
        else:
            tau_s = None

        rep_end = result.t_write_complete or time.monotonic()
        collected.append(
            Rep(
                index=index,
                request_id=request_id,
                t_frame_grab=t_frame_grab,
                t_detect_done=t_detect_done,
                t_request=t_request,
                t_write_complete=result.t_write_complete,
                tau_s=tau_s,
                detected=detection is not None,
                centroid_x=None if detection is None else detection.x,
                centroid_y=None if detection is None else detection.y,
                area_px=None if detection is None else detection.area_px,
                accepted=result.accepted,
                n=result.n,
                reject_reason=result.reject_reason,
                keepalive_affected=overlaps_any(t_frame_grab, rep_end, windows()),
            )
        )
        # Written every rep so a dropped connection does not cost the run.
        write_jsonl(log_path, collected)

    return remark_keepalive(collected, windows())


def remark_keepalive(
    reps: Sequence[Rep], windows: Sequence[tuple[float, float]]
) -> list[Rep]:
    """Re-mark against the full window list once every tick is known."""
    remarked = []
    for rep in reps:
        rep_end = rep.t_write_complete or rep.t_detect_done
        affected = overlaps_any(rep.t_frame_grab, rep_end, windows)
        remarked.append(
            rep if affected == rep.keepalive_affected else _replace_mark(rep, affected)
        )
    return remarked


def _replace_mark(rep: Rep, affected: bool) -> Rep:
    fields = asdict(rep)
    fields["keepalive_affected"] = affected
    return Rep(**fields)


def format_report(
    reps: Sequence[Rep],
    summary: dict[str, float | int] | None,
    keepalive_mode: str,
    ticks: Sequence[tuple[float, float]],
    requested_reps: int,
    keepalive_interval_s: float | None,
    run_span_s: float,
) -> str:
    lines = ["", "tau: frame grab to Bluetooth write completion", ""]
    measured = [rep for rep in reps if rep.tau_s is not None]
    lines.append(f"  reps requested        {requested_reps}")
    lines.append(f"  reps run              {len(reps)}")
    lines.append(f"  taus measured         {len(measured)}")
    lines.append(f"  reps with no tau      {len(reps) - len(measured)}")

    if summary is None:
        lines.append("")
        lines.append("  No accepted stimulation produced a tau. Nothing to summarise.")
        return "\n".join(lines) + "\n"

    lines.append("")
    lines.append(f"  median                {summary['median_s'] * 1000:7.1f} ms")
    lines.append(
        f"  IQR                   {summary['q1_s'] * 1000:7.1f} .. "
        f"{summary['q3_s'] * 1000:.1f} ms  (width {summary['iqr_s'] * 1000:.1f} ms)"
    )
    lines.append(f"  p95                   {summary['p95_s'] * 1000:7.1f} ms")
    lines.append(
        f"  min / max             {summary['min_s'] * 1000:7.1f} / "
        f"{summary['max_s'] * 1000:.1f} ms"
    )

    lines.append("")
    lines.append("  phase split, median")
    grab_to_detect = np.median(
        [(rep.t_detect_done - rep.t_frame_grab) * 1000 for rep in measured]
    )
    detect_to_request = np.median(
        [(rep.t_request - rep.t_detect_done) * 1000 for rep in measured]
    )
    request_to_write = np.median(
        [(rep.t_write_complete - rep.t_request) * 1000 for rep in measured]
    )
    lines.append(f"    grab return to detect done   {grab_to_detect:7.1f} ms")
    lines.append(f"    detect done to request       {detect_to_request:7.1f} ms")
    lines.append(f"    request to write complete    {request_to_write:7.1f} ms")

    lines.append("")
    lines.append("  keepalive")
    lines.append(f"    mode                       {keepalive_mode}")
    lines.append(f"    interval                   {keepalive_interval_s} s")
    lines.append(f"    run span                   {run_span_s:.1f} s")
    lines.append(f"    ticks in the run           {len(ticks)}")
    lines.append(
        f"    reps affected              {sum(rep.keepalive_affected for rep in reps)}"
    )
    if keepalive_interval_s is not None and not ticks:
        lines.append("")
        lines.append(
            f"    WARNING: the run spanned {run_span_s:.1f} s and no keepalive tick "
            f"fired in it. The affected count of 0 means not sampled, not "
            f"unaffected. Raise --reps or --t-refrac until the span exceeds "
            f"{keepalive_interval_s} s."
        )

    lines.append("")
    lines.append("  Percentile choice is left to the caller. This tool does not pick.")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--t-refrac",
        required=True,
        type=float,
        help="seconds between accepted stimulations; short, there is no animal",
    )
    parser.add_argument("--camera-index", required=True, type=int)
    parser.add_argument(
        "--backpack-off-animal",
        required=True,
        action="store_true",
        help="explicit confirmation that this run cannot stimulate an animal",
    )
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--settle", type=float, default=0.05)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--scan-timeout", type=float, default=10.0)
    parser.add_argument("--keepalive-interval", type=float, default=120.0)
    parser.add_argument("--no-keepalive", action="store_true")
    parser.add_argument("--keepalive-guard", type=float, default=0.5)
    parser.add_argument("--frequency-hz", type=int, default=55)
    parser.add_argument("--pulse-width-ms", type=int, default=5)
    parser.add_argument("--duration-ms", type=int, default=10)
    parser.add_argument("--gain-percent", type=int, default=0)
    parser.add_argument("--hsv-lower", type=int, nargs=3, default=(35, 80, 80))
    parser.add_argument("--hsv-upper", type=int, nargs=3, default=(85, 255, 255))
    parser.add_argument("--min-area", type=float, default=20.0)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    from interface import RoboRoach

    settings = StimulationSettings(
        frequency_hz=args.frequency_hz,
        pulse_width_ms=args.pulse_width_ms,
        duration_ms=args.duration_ms,
        gain_percent=args.gain_percent,
        random_mode=False,
    )
    expected_span = args.reps * (args.t_refrac + args.settle)
    if not args.no_keepalive and expected_span < args.keepalive_interval:
        print(
            f"note: {args.reps} reps at --t-refrac {args.t_refrac} span about "
            f"{expected_span:.0f} s, shorter than the {args.keepalive_interval} s "
            "keepalive interval, so no tick will be sampled."
        )

    camera = Camera(args.camera_index)
    print(f"camera index {args.camera_index}: {camera.describe()}")
    watch = KeepaliveWatch()
    keepalive_task: asyncio.Task[None] | None = None
    t_start = time.monotonic()

    try:
        async with RoboRoach(scan_timeout=args.scan_timeout) as roach:
            gate = await StimGate.create(
                roach=roach,
                t_refrac_s=args.t_refrac,
                settings=settings,
                run_dir=args.run_dir,
            )
            print(f"board settings {gate.settings} (settings_id {gate.settings_id})")

            observed = watch.attach(roach)
            keepalive_mode = OBSERVED if observed else PREDICTED
            if not args.no_keepalive:
                keepalive_task = asyncio.create_task(
                    roach.keep_alive(interval_seconds=args.keepalive_interval)
                )
                t_start = time.monotonic()

            if args.no_keepalive:

                def windows() -> Sequence[tuple[float, float]]:
                    return []
            elif observed:

                def windows() -> Sequence[tuple[float, float]]:
                    return watch.windows
            else:

                def windows() -> Sequence[tuple[float, float]]:
                    return predicted_windows(
                        t_start,
                        args.keepalive_interval,
                        time.monotonic(),
                        args.keepalive_guard,
                    )

            for _ in range(args.warmup_frames):
                await camera.grab()

            t_first = time.monotonic()
            reps = await run_reps(
                gate=gate,
                grab=camera.grab,
                detect=lambda frame: detect_centroid(
                    frame, args.hsv_lower, args.hsv_upper, args.min_area
                ),
                reps=args.reps,
                t_refrac_s=args.t_refrac,
                settle_s=args.settle,
                log_path=args.run_dir / LOG_NAME,
                windows=windows,
            )
            run_span = time.monotonic() - t_first
    finally:
        if keepalive_task is not None:
            keepalive_task.cancel()
        watch.detach()
        camera.release()

    write_jsonl(args.run_dir / LOG_NAME, reps)
    taus = [rep.tau_s for rep in reps if rep.tau_s is not None]
    summary = summarise(taus) if taus else None
    ticks = cluster_windows(windows()) if not args.no_keepalive else []

    report = format_report(
        reps,
        summary,
        keepalive_mode,
        ticks,
        args.reps,
        None if args.no_keepalive else args.keepalive_interval,
        run_span,
    )
    print(report)

    (args.run_dir / SUMMARY_NAME).write_text(
        json.dumps(
            {
                "tau": summary,
                "keepalive_mode": keepalive_mode,
                "keepalive_interval_s": None
                if args.no_keepalive
                else args.keepalive_interval,
                "keepalive_ticks": len(ticks),
                "reps_affected_by_keepalive": sum(
                    rep.keepalive_affected for rep in reps
                ),
                "reps_requested": args.reps,
                "taus_measured": len(taus),
                "run_span_s": run_span,
                "t_refrac_s": args.t_refrac,
                "settings": asdict(settings),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0 if summary is not None else 1


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
