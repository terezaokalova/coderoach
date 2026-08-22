"""The single point through which every module requests a stimulation.

Text, voice, and trajectory control all call :meth:`StimGate.request`. The gate
owns the refractory period and the trial counter ``n``, so the stimulation
budget is enforced in one place instead of once per caller, and ``n`` stays a
valid index into the habituation curve across every source.

The gate does not reimplement the Bluetooth interface. It holds an already
connected ``interface.RoboRoach`` and calls its public ``configure``,
``read_settings``, and ``turn``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from interface import StimulationSettings

Direction = Literal["left", "right"]
Source = Literal["text", "voice", "traj"]

DIRECTIONS: frozenset[str] = frozenset(("left", "right"))
REQUEST_SOURCES: frozenset[str] = frozenset(("text", "voice", "traj"))

LOG_NAME = "stim_gate.jsonl"
SOURCES_JSON_NAME = "sources.json"

REFRACTORY = "refractory"
WRITE_FAILED = "write_failed"


class RoachLike(Protocol):
    """The part of ``interface.RoboRoach`` that the gate consumes.

    Declared so the test suite can inject a stub with no Bluetooth present.
    It carries no behaviour of its own.
    """

    async def configure(
        self,
        *,
        frequency_hz: int = ...,
        pulse_width_ms: int = ...,
        duration_ms: int = ...,
        gain_percent: int = ...,
        random_mode: bool = ...,
    ) -> None: ...

    async def read_settings(self) -> StimulationSettings: ...

    async def turn(self, direction: Direction) -> None: ...


@dataclass(frozen=True)
class StimResult:
    """The outcome of one request, accepted or rejected."""

    accepted: bool
    request_id: str
    source: str
    direction: str
    t_request: float
    t_write_complete: float | None
    n: int
    reject_reason: str | None


def settings_id(settings: StimulationSettings) -> str:
    """A stable short id for one set of board settings.

    Every log line carries this, so a trial can be traced back to the
    stimulation parameters that were live on the board when it fired.
    """
    payload = json.dumps(asdict(settings), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _git_provenance() -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                arguments,
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                check=True,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "commit": commit or None,
        "dirty": None if status is None else bool(status),
    }


class StimGate:
    """Serialise every stimulation request behind one refractory period.

    Build it with :meth:`create`, which performs the Bluetooth configuration
    that a synchronous constructor cannot await.
    """

    def __init__(
        self,
        *,
        roach: RoachLike,
        t_refrac_s: float,
        settings: StimulationSettings,
        run_dir: Path | str,
    ) -> None:
        t_refrac_s = float(t_refrac_s)
        _check_refractory_covers_stimulus(t_refrac_s, settings.duration_ms, "requested")

        self._roach = roach
        self._t_refrac_s = t_refrac_s
        self._requested_settings = settings
        self._run_dir = Path(run_dir)
        self._lock = asyncio.Lock()
        self._n = 0
        self._t_last_accepted: float | None = None
        self._settings: StimulationSettings | None = None

    @classmethod
    async def create(
        cls,
        *,
        roach: RoachLike,
        t_refrac_s: float,
        settings: StimulationSettings,
        run_dir: Path | str,
    ) -> StimGate:
        """Configure the board once, read back what it holds, and record it."""
        gate = cls(
            roach=roach,
            t_refrac_s=t_refrac_s,
            settings=settings,
            run_dir=run_dir,
        )
        await gate._apply_settings()
        return gate

    @property
    def n(self) -> int:
        """Accepted stimulations so far, across every source."""
        return self._n

    @property
    def t_refrac_s(self) -> float:
        return self._t_refrac_s

    @property
    def settings(self) -> StimulationSettings:
        """What the board reported after configuration."""
        return self._require_configured()

    @property
    def settings_id(self) -> str:
        """Short id for the settings the board reported."""
        return settings_id(self._require_configured())

    @property
    def log_path(self) -> Path:
        return self._run_dir / LOG_NAME

    async def request(
        self,
        direction: Direction,
        source: Source,
        request_id: str,
    ) -> StimResult:
        """Ask for one stimulation. Returns whether it fired.

        The refractory check, the Bluetooth write, and the counter increment
        happen under one lock, so two callers racing at the same instant
        cannot both pass the check and stimulate twice.
        """
        t_request = time.monotonic()
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}")
        if source not in REQUEST_SOURCES:
            raise ValueError(f"source must be one of {sorted(REQUEST_SOURCES)}")
        self._require_configured()

        async with self._lock:
            if self._in_refractory():
                return self._log(
                    StimResult(
                        accepted=False,
                        request_id=request_id,
                        source=source,
                        direction=direction,
                        t_request=t_request,
                        t_write_complete=None,
                        n=self._n,
                        reject_reason=REFRACTORY,
                    )
                )

            try:
                await self._roach.turn(direction)
            except Exception:
                # The write may already have reached the backpack, so hold the
                # refractory window shut. The stimulation is not confirmed, so
                # it does not enter the habituation count. Fail visibly and do
                # not retry: interface/AGENTS.md requires both.
                self._t_last_accepted = time.monotonic()
                self._log(
                    StimResult(
                        accepted=False,
                        request_id=request_id,
                        source=source,
                        direction=direction,
                        t_request=t_request,
                        t_write_complete=None,
                        n=self._n,
                        reject_reason=WRITE_FAILED,
                    )
                )
                raise

            t_write_complete = time.monotonic()
            self._t_last_accepted = t_write_complete
            self._n += 1
            return self._log(
                StimResult(
                    accepted=True,
                    request_id=request_id,
                    source=source,
                    direction=direction,
                    t_request=t_request,
                    t_write_complete=t_write_complete,
                    n=self._n,
                    reject_reason=None,
                )
            )

    def _in_refractory(self) -> bool:
        if self._t_last_accepted is None:
            return False
        return (time.monotonic() - self._t_last_accepted) < self._t_refrac_s

    def _require_configured(self) -> StimulationSettings:
        if self._settings is None:
            raise RuntimeError(
                "StimGate was constructed directly. Build it with "
                "await StimGate.create(...) so the board is configured first."
            )
        return self._settings

    async def _apply_settings(self) -> None:
        requested = self._requested_settings
        await self._roach.configure(
            frequency_hz=requested.frequency_hz,
            pulse_width_ms=requested.pulse_width_ms,
            duration_ms=requested.duration_ms,
            gain_percent=requested.gain_percent,
            random_mode=requested.random_mode,
        )
        # configure() returns None, so the settings the board holds have to be
        # read back. The readback is what will actually fire, so it is what the
        # refractory period is checked against and what the run records.
        board = await self._roach.read_settings()
        _check_refractory_covers_stimulus(
            self._t_refrac_s, board.duration_ms, "readback"
        )

        self._settings = board
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._write_sources_json()

    def _write_sources_json(self) -> None:
        path = self._run_dir / SOURCES_JSON_NAME
        document: dict[str, object] = {}
        if path.exists():
            text = path.read_text()
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path} exists but is not valid JSON. Refusing to "
                    "overwrite another module's provenance."
                ) from error
            if not isinstance(loaded, dict):
                raise RuntimeError(f"{path} exists but is not a JSON object.")
            document = loaded

        document["stim_gate"] = {
            "settings_id": self.settings_id,
            "t_refrac_s": self._t_refrac_s,
            "settings_requested": asdict(self._requested_settings),
            "settings_readback": asdict(self._require_configured()),
            **_git_provenance(),
        }

        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    def _log(self, result: StimResult) -> StimResult:
        line = {
            "request_id": result.request_id,
            "source": result.source,
            "direction": result.direction,
            "t_request": result.t_request,
            "t_write_complete": result.t_write_complete,
            "n": result.n,
            "accepted": result.accepted,
            "reject_reason": result.reject_reason,
            "settings_id": self.settings_id,
        }
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(line) + "\n")
        return result


def _check_refractory_covers_stimulus(
    t_refrac_s: float,
    duration_ms: int,
    which: str,
) -> None:
    duration_s = duration_ms / 1000
    if t_refrac_s < duration_s:
        raise ValueError(
            f"T_refrac of {t_refrac_s} s is shorter than the {which} stimulus "
            f"duration of {duration_s} s. Stimulations would overlap. Raise "
            "T_refrac or shorten duration_ms deliberately; neither is adjusted "
            "here."
        )
