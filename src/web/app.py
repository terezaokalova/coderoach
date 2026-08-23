"""HTTP and websocket surface for the drawing canvas.

Capabilities are checked one at a time rather than as a single live/replay
switch. A missing backpack still leaves video and tracking; a missing camera
still leaves the recorded runs. Each endpoint refuses with the specific reason
its own capability is absent, because "replay mode" on its own does not tell an
operator which cable to go and plug in.

Every request that ends in current -- a button, a spoken phrase, the control
loop -- goes through :class:`stim.StimGate`. The refractory period, the trial
counter, and the JSONL record all live there.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from web import runs

log = logging.getLogger(__name__)

PAGE = Path(__file__).resolve().parent / "static" / "index.html"
CONFIG_TOKEN = "__WEB_CONFIG__"

DIRECTIONS = frozenset(("left", "right"))

CAMERA = "camera"
ROACH = "roach"
VOICE = "voice"

# Padding around the calibrated arena so a stroke drawn slightly outside the
# ruler points is still expressible. The operator aims at the animal, not at
# the calibration quad.
ARENA_MARGIN_CM = 2.0

JOURNAL_LIMIT = 500


@dataclass(frozen=True)
class WebConfig:
    """Everything the server was started with.

    The tracker's own parameters carry defaults here, which :mod:`traj.track`
    deliberately refuses to do. That refusal is right for a measurement tool
    and wrong for a demo surface that has to come up on one command line, so
    the compromise is that :mod:`web.__main__` prints every one of them at
    startup and says which came from the command line. A guess that is printed
    is not a guess that looks like a measurement.
    """

    run_dir: Path
    t_refrac_s: float
    camera_index: int | None = None
    hsv: Path | None = None
    arena: Path | None = None
    min_contour_area: float = 150.0
    sigma_p_cm: float = 0.5
    sigma_a_cm_s2: float = 20.0
    v_min_cm_s: float = 1.5
    spacing_cm: float = 2.0
    lookahead_cm: float = 6.0
    alpha_dead_rad: float = 0.26
    video_width: int = 720
    video_fps: float = 15.0
    frequency_hz: int = 10
    pulse_width_ms: int = 1
    duration_ms: int = 250
    gain_percent: int = 0
    scan_timeout: float = 10.0
    voice_seconds: float = 3.0
    voice_model: str = "base.en"
    voice_device: int | str | None = None
    with_roach: bool = True
    with_voice: bool = True

    @property
    def runs_root(self) -> Path:
        """Where sibling runs live. ``--run-dir runs/today`` makes it ``runs``."""
        return self.run_dir.parent


@dataclass
class Capability:
    available: bool = False
    detail: str = "not started"


class StimJournal:
    """Every stimulation request the server has made, and who is watching.

    ``n`` and the refractory window are the gate's, not this object's. What is
    tracked here is the wall-clock the page displays, derived from the results
    the gate handed back -- every request in this process goes through one of
    them, so the two cannot drift apart.
    """

    def __init__(self, limit: int = JOURNAL_LIMIT) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=limit)
        self._subscribers: set[asyncio.Queue] = set()
        self._t_last_accepted: float | None = None

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def since_last_accepted_s(self) -> float | None:
        if self._t_last_accepted is None:
            return None
        return time.monotonic() - self._t_last_accepted

    def in_refractory(self, t_refrac_s: float) -> bool:
        since = self.since_last_accepted_s()
        return since is not None and since < t_refrac_s

    def record(self, stim, decision=None) -> dict[str, Any]:
        if stim.accepted:
            self._t_last_accepted = time.monotonic()
        event = {
            "type": "stim",
            "t": time.monotonic(),
            "request_id": stim.request_id,
            "source": stim.source,
            "direction": stim.direction,
            "accepted": stim.accepted,
            "reject_reason": stim.reject_reason,
            "n": stim.n,
            "detail": None,
        }
        if decision is not None:
            event["alpha"] = decision.alpha
            event["cross_track_cm"] = decision.cross_track_cm
        return self._emit(event)

    def record_rejection(
        self,
        request_id: str,
        source: str,
        direction: str | None,
        reason: str,
        detail: str | None = None,
    ) -> dict[str, Any]:
        """A refusal that never reached the gate, or that the gate raised on.

        The gate logs its own rejections to JSONL, but the ones it raises on
        come back as an exception rather than a result, and a phrase that did
        not match never gets that far. Both still belong in the operator's log
        or the page would show a button press that produced nothing at all.
        """
        return self._emit(
            {
                "type": "stim",
                "t": time.monotonic(),
                "request_id": request_id,
                "source": source,
                "direction": direction,
                "accepted": False,
                "reject_reason": reason,
                "n": None,
                "detail": detail,
            }
        )

    def record_note(self, note: str, detail: str | None = None) -> dict[str, Any]:
        return self._emit(
            {"type": "note", "t": time.monotonic(), "note": note, "detail": detail}
        )

    def _emit(self, event: dict[str, Any]) -> dict[str, Any]:
        self._events.append(event)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A client too slow to drain its own log loses lines from it.
                # Blocking here would stall the control loop on a stalled iPad.
                log.warning("state client fell behind; dropping a journal event")
        return event


class Runtime:
    """Holds the camera, the gate, and the control loop for one server."""

    def __init__(self, config: WebConfig) -> None:
        self.config = config
        self.hub = None
        self.gate = None
        self.loop = None
        self.journal = StimJournal()
        self.status: dict[str, Capability] = {
            CAMERA: Capability(),
            ROACH: Capability(),
            VOICE: Capability(),
        }
        self.notes: list[str] = []
        self.arena_bounds: dict[str, float] | None = None
        self._stack = AsyncExitStack()
        self._asr_lock = asyncio.Lock()
        self._voice_busy = asyncio.Lock()
        self._transcriber = None

    # -- lifecycle ------------------------------------------------------

    async def startup(self) -> None:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        self.arena_bounds = _arena_bounds(self.config.arena)
        await self._start_camera()
        await self._start_roach()
        self._check_voice()

        if self.hub is not None and self.gate is not None:
            from web.loop import ControlLoop

            self.loop = ControlLoop(
                hub=self.hub,
                gate=self.gate,
                journal=self.journal,
                run_dir=self.config.run_dir,
            )

    async def shutdown(self) -> None:
        if self.loop is not None:
            from web.loop import STOP_SHUTDOWN

            await self.loop.stop(STOP_SHUTDOWN)
        if self.hub is not None:
            await self.hub.aclose()
        await self._stack.aclose()

    async def _start_camera(self) -> None:
        config = self.config
        missing = [
            name
            for name, path in (("--arena", config.arena), ("--hsv", config.hsv))
            if path is None or not Path(path).exists()
        ]
        if config.camera_index is None:
            self._degrade(CAMERA, "started with --no-camera")
            return
        if missing:
            # The calibration is what turns pixels into centimetres. Without it
            # there is no arena frame, so there is nothing a drawn path could
            # mean. Serving the recorded runs is the useful thing left to do.
            self._degrade(
                CAMERA, f"calibration missing: {', '.join(missing)} -- replay only"
            )
            return

        try:
            from traj.track import HsvBounds, TrackerConfig, TrajectoryTracker
            from web.hub import FrameHub

            tracker = TrajectoryTracker(
                TrackerConfig(
                    camera_index=config.camera_index,
                    hsv_bounds=HsvBounds.from_json(config.hsv),
                    min_contour_area=config.min_contour_area,
                    sigma_p_cm=config.sigma_p_cm,
                    sigma_a_cm_s2=config.sigma_a_cm_s2,
                    v_min_cm_s=config.v_min_cm_s,
                    arena_calibration=config.arena,
                    run_dir=config.run_dir,
                )
            )
        except Exception as exc:  # noqa: BLE001
            # Deliberately blind. A missing calibration, a camera another
            # process holds, a resolution the homography was not fitted at --
            # every one of them must leave the recorded runs servable rather
            # than take the whole server down with it.
            self._degrade(CAMERA, f"{type(exc).__name__}: {exc}")
            return

        self.hub = FrameHub(
            tracker,
            v_min_cm_s=config.v_min_cm_s,
            video_width=config.video_width,
        )
        await self.hub.start()
        self.status[CAMERA] = Capability(True, f"camera {config.camera_index}")

    async def _start_roach(self) -> None:
        config = self.config
        if not config.with_roach:
            self._degrade(ROACH, "started with --no-roach")
            return
        try:
            from interface import RoboRoach, StimulationSettings

            from stim import StimGate

            roach = await self._stack.enter_async_context(
                RoboRoach(scan_timeout=config.scan_timeout)
            )
            self.gate = await StimGate.create(
                roach=roach,
                t_refrac_s=config.t_refrac_s,
                settings=StimulationSettings(
                    frequency_hz=config.frequency_hz,
                    pulse_width_ms=config.pulse_width_ms,
                    duration_ms=config.duration_ms,
                    gain_percent=config.gain_percent,
                    random_mode=False,
                ),
                run_dir=config.run_dir,
            )
        except Exception as exc:  # noqa: BLE001
            # Deliberately blind, and the same reasoning: bleak raises its own
            # exception types, a backpack with a flat battery never advertises,
            # and neither is a reason to refuse to serve the video.
            self._degrade(ROACH, f"{type(exc).__name__}: {exc}")
            return
        self.status[ROACH] = Capability(
            True, f"gain {config.gain_percent}%, T_refrac {config.t_refrac_s} s"
        )

    def _check_voice(self) -> None:
        if not self.config.with_voice:
            self._degrade(VOICE, "started with --no-voice")
            return
        try:
            import faster_whisper  # noqa: F401
            import sounddevice  # noqa: F401
        except (ImportError, OSError) as exc:
            # sounddevice raises OSError when PortAudio is absent rather than
            # ImportError, so both are the same failure wearing two names.
            self._degrade(VOICE, f"{type(exc).__name__}: {exc}")
            return
        self.status[VOICE] = Capability(True, f"model {self.config.voice_model}")

    def _degrade(self, name: str, detail: str) -> None:
        self.status[name] = Capability(False, detail)
        self.notes.append(f"{name}: {detail}")
        log.warning("%s unavailable -- %s", name, detail)

    # -- guards ---------------------------------------------------------

    def require(self, *names: str) -> None:
        for name in names:
            capability = self.status[name]
            if not capability.available:
                raise HTTPException(
                    status_code=409,
                    detail=f"{name} unavailable: {capability.detail}",
                )

    # -- messages -------------------------------------------------------

    def page_config(self) -> dict[str, Any]:
        config = self.config
        return {
            "live": self.status[CAMERA].available,
            "capabilities": {
                name: {"available": c.available, "detail": c.detail}
                for name, c in self.status.items()
            },
            "arena": self.arena_bounds,
            "spacing_cm": config.spacing_cm,
            "lookahead_cm": config.lookahead_cm,
            "alpha_dead_rad": config.alpha_dead_rad,
            "t_refrac_s": config.t_refrac_s,
            "run_id": config.run_dir.name,
            "notes": list(self.notes),
        }

    def snapshot(self) -> dict[str, Any]:
        loop = self.loop
        return {
            "type": "snapshot",
            "t": time.monotonic(),
            "tracing": bool(loop is not None and loop.active),
            "trace_id": None if loop is None else loop.trace_id,
            "reference_cm": [] if loop is None else loop.reference_cm,
            "walked_cm": [] if loop is None else list(loop.walked_cm),
            "lookahead_cm": self._gain_value("lookahead_cm"),
            "alpha_dead_rad": self._gain_value("alpha_dead_rad"),
            "n": None if self.gate is None else self.gate.n,
            "events": self.journal.events,
        }

    def state_message(self, result) -> dict[str, Any]:
        loop = self.loop
        decision = None if loop is None else loop.last_decision
        walked = [] if loop is None else loop.walked_cm
        return {
            "type": "state",
            "t": result.t_frame,
            "x": result.px_hat,
            "y": result.py_hat,
            "theta": result.theta,
            "speed": result.speed,
            "heading_valid": result.heading_valid,
            "detected": result.detection is not None,
            "n": None if self.gate is None else self.gate.n,
            "since_stim_s": self.journal.since_last_accepted_s(),
            "refractory": self.journal.in_refractory(self.config.t_refrac_s),
            "tracing": bool(loop is not None and loop.active),
            "trace_id": None if loop is None else loop.trace_id,
            "walked_head": walked[-1] if walked else None,
            "walked_count": len(walked),
            "carrot_cm": None if decision is None else decision.carrot_cm,
            "cross_track_cm": None if decision is None else decision.cross_track_cm,
            "alpha": None if decision is None else decision.alpha,
            "at_end": bool(decision is not None and decision.at_end),
            "reason": None if decision is None else decision.reason,
        }

    def _gain_value(self, field_name: str) -> float:
        gains = None if self.loop is None else self.loop.gains
        if gains is None:
            return getattr(self.config, field_name)
        return getattr(gains, field_name)

    # -- voice ----------------------------------------------------------

    async def transcriber(self):
        """Load the ASR model once, off the event loop.

        Loading dominates the wall clock and blocks for seconds. It happens on
        the first Speak press rather than at startup so that a rig with no
        microphone still comes up immediately.
        """
        async with self._asr_lock:
            if self._transcriber is None:
                from voice import Transcriber

                t_start = time.monotonic()
                self._transcriber = await asyncio.to_thread(
                    Transcriber, self.config.voice_model
                )
                self.journal.record_note(
                    "ASR model loaded",
                    f"{self.config.voice_model} in {time.monotonic() - t_start:.1f} s",
                )
            return self._transcriber

    @property
    def voice_busy(self) -> asyncio.Lock:
        return self._voice_busy


def _arena_bounds(arena: Path | str | None) -> dict[str, float] | None:
    """The arena's extent in centimetres, read without OpenCV.

    Only the ruler measurements are needed to size the drawing canvas, and they
    are plain numbers in the calibration file. Reading them directly keeps a
    replay deployment from having to import cv2 to draw a rectangle.
    """
    if arena is None:
        return None
    path = Path(arena)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text())
        points = document["arena_points_cm"]
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
        log.warning("could not read arena_points_cm from %s", path)
        return None
    if not xs or not ys:
        return None
    return {
        "x_min": min(xs) - ARENA_MARGIN_CM,
        "x_max": max(xs) + ARENA_MARGIN_CM,
        "y_min": min(ys) - ARENA_MARGIN_CM,
        "y_max": max(ys) + ARENA_MARGIN_CM,
    }


def _run_directory(root: Path, run_id: str) -> Path:
    """Resolve a run id under the runs root, refusing anything that escapes."""
    candidate = (root / run_id).resolve()
    if candidate.parent != root.resolve() or not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
    return candidate


def create_app(config: WebConfig) -> FastAPI:
    runtime = Runtime(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runtime.startup()
        try:
            yield
        finally:
            await runtime.shutdown()

    app = FastAPI(title="CodeRoach canvas", lifespan=lifespan)
    app.state.runtime = runtime

    # -- page -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        # Read per request rather than cached: editing the page during a
        # session then reloading the iPad is the whole debug loop, and a
        # restart would drop the Bluetooth connection to do it.
        html = PAGE.read_text()
        payload = json.dumps(runtime.page_config()).replace("<", "\\u003c")
        return HTMLResponse(html.replace(CONFIG_TOKEN, payload))

    # -- video ----------------------------------------------------------

    @app.websocket("/ws/video")
    async def ws_video(socket: WebSocket) -> None:
        await socket.accept()
        if runtime.hub is None:
            await socket.send_json(
                {"type": "unavailable", "detail": runtime.status[CAMERA].detail}
            )
            await socket.close()
            return

        interval = 1.0 / max(config.video_fps, 1.0)
        seq = 0
        next_at = 0.0
        try:
            while True:
                now = time.monotonic()
                if now < next_at:
                    await asyncio.sleep(next_at - now)
                # The newest frame is fetched after the sleep, so a capped
                # stream drops the frames it slept through instead of falling
                # progressively further behind the animal.
                seq, result = await runtime.hub.next_frame(seq)
                next_at = time.monotonic() + interval
                await socket.send_bytes(await runtime.hub.jpeg(seq, result))
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001
            # One client's stream dying must not propagate into the server.
            log.warning("video stream ended: %s", exc)
            await _close_quietly(socket)

    # -- state ----------------------------------------------------------

    @app.websocket("/ws/state")
    async def ws_state(socket: WebSocket) -> None:
        await socket.accept()
        queue = runtime.journal.subscribe()
        receiver = asyncio.create_task(_receive_state(socket, runtime))
        try:
            await socket.send_json(runtime.snapshot())
            seq = 0
            while True:
                if runtime.hub is not None:
                    seq, result = await runtime.hub.next_frame(seq)
                    await socket.send_json(runtime.state_message(result))
                else:
                    await asyncio.sleep(0.25)
                while not queue.empty():
                    await socket.send_json(queue.get_nowait())
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001
            # As above: an iPad that slept is not a server fault.
            log.warning("state stream ended: %s", exc)
            await _close_quietly(socket)
        finally:
            runtime.journal.unsubscribe(queue)
            receiver.cancel()

    # -- control --------------------------------------------------------

    @app.post("/api/turn")
    async def api_turn(request: Request) -> JSONResponse:
        runtime.require(ROACH)
        body = await _json_body(request)
        direction = body.get("direction")
        if direction not in DIRECTIONS:
            raise HTTPException(
                status_code=400, detail="direction must be 'left' or 'right'"
            )
        return JSONResponse(await _request_stim(runtime, direction, "text"))

    @app.post("/api/voice")
    async def api_voice() -> JSONResponse:
        runtime.require(VOICE, ROACH)
        if runtime.voice_busy.locked():
            raise HTTPException(status_code=409, detail="a capture is already running")

        async with runtime.voice_busy:
            from voice import interpret, record_window

            transcriber = await runtime.transcriber()
            recording = await asyncio.to_thread(
                record_window, config.voice_seconds, config.voice_device
            )
            command = await asyncio.to_thread(interpret, recording, transcriber)

            heard = {
                "heard": command.heard,
                "raw_text": command.raw_text,
                "peak": command.peak,
                "accepted": command.accepted,
                "reject_reason": command.reject_reason,
            }
            if not command.accepted:
                request_id = f"voice-{uuid4().hex[:8]}"
                runtime.journal.record_rejection(
                    request_id,
                    "voice",
                    None,
                    command.reject_reason,
                    command.heard or f"peak {command.peak:.3f}",
                )
                return JSONResponse({**heard, "stim": None})

            stim = await _request_stim(
                runtime, command.direction, "voice", detail=command.heard
            )
            return JSONResponse({**heard, "stim": stim})

    @app.post("/api/path")
    async def api_path(request: Request) -> JSONResponse:
        runtime.require(CAMERA, ROACH)
        body = await _json_body(request)
        points = body.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise HTTPException(
                status_code=400, detail="points must be a list of at least two [x, y]"
            )

        from traj.control import PursuitGains, resample_path

        try:
            # Re-spaced server-side even though the browser already did it.
            # The browser's resampler is the one that can be stale, and pure
            # pursuit reads spacing as distance.
            waypoints = resample_path(points, config.spacing_cm)
            gains = PursuitGains(
                lookahead_cm=float(body.get("lookahead_cm", config.lookahead_cm)),
                alpha_dead_rad=float(body.get("alpha_dead_rad", config.alpha_dead_rad)),
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if len(waypoints) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"the stroke is shorter than one {config.spacing_cm} cm step",
            )

        await runtime.loop.start(waypoints, gains)
        runtime.journal.record_note(
            "tracing", f"{len(waypoints)} waypoints, {runtime.loop.length_cm:.1f} cm"
        )
        return JSONResponse(
            {
                "trace_id": runtime.loop.trace_id,
                "waypoints_cm": waypoints,
                "length_cm": runtime.loop.length_cm,
                "lookahead_cm": gains.lookahead_cm,
                "alpha_dead_rad": gains.alpha_dead_rad,
            }
        )

    @app.post("/api/stop")
    async def api_stop() -> JSONResponse:
        if runtime.loop is None:
            return JSONResponse({"stopped": False, "detail": "no control loop"})
        from web.loop import STOP_REQUESTED

        trace = await runtime.loop.stop(STOP_REQUESTED)
        runtime.journal.record_note("stopped", None if trace is None else "trace saved")
        return JSONResponse(
            {
                "stopped": trace is not None,
                "trace_index": None if trace is None else trace["index"],
                "walked_points": 0 if trace is None else len(trace["walked_cm"]),
            }
        )

    # -- runs -----------------------------------------------------------

    @app.get("/api/runs")
    async def api_runs() -> JSONResponse:
        return JSONResponse(
            {
                "root": str(config.runs_root),
                "current": config.run_dir.name,
                "runs": runs.list_runs(config.runs_root),
            }
        )

    @app.get("/api/runs/{run_id}")
    async def api_run(run_id: str) -> JSONResponse:
        return JSONResponse(runs.read_run(_run_directory(config.runs_root, run_id)))

    return app


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="body must be JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


async def _request_stim(
    runtime: Runtime, direction: str, source: str, detail: str | None = None
) -> dict[str, Any]:
    """One request through the gate, with its refusals reported as outcomes.

    A rejection is not an HTTP error: it is what the gate is for, and the page
    shows it in the log next to the accepted ones. A write that failed partway
    is different -- that one propagates, because interface/AGENTS.md requires
    it to be visible and not retried.
    """
    request_id = f"{source}-{uuid4().hex[:8]}"
    try:
        stim = await runtime.gate.request(direction, source, request_id)
    except RuntimeError as exc:
        event = runtime.journal.record_rejection(
            request_id, source, direction, "safety_guard", str(exc)
        )
        return event
    event = runtime.journal.record(stim)
    if detail is not None:
        event["detail"] = detail
    return event


async def _receive_state(socket: WebSocket, runtime: Runtime) -> None:
    """Inbound messages on the state socket. Only this task ever receives.

    Nothing here sends. Two tasks writing to one websocket interleave frames,
    so a reply goes into the journal and leaves over the sending loop instead.
    """
    while True:
        try:
            message = await socket.receive_json()
        except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
            return
        if not isinstance(message, dict) or message.get("type") != "gains":
            continue
        if runtime.loop is None:
            continue
        try:
            gains = runtime.loop.set_gains(
                message["lookahead_cm"], message["alpha_dead_rad"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            runtime.journal.record_note("gains rejected", str(exc))
            continue
        runtime.journal.record_note(
            "gains",
            f"L_d {gains.lookahead_cm:.1f} cm, "
            f"alpha_dead {gains.alpha_dead_rad:.3f} rad",
        )


async def _close_quietly(socket: WebSocket) -> None:
    try:
        await socket.close()
    except RuntimeError:
        # Already closed from the other end; there is nothing left to say.
        pass
