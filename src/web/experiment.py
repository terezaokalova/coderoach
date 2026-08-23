"""One RL experiment at a time, exclusive with the drawn-path loop.

Pose comes from the hub that already owns the camera, not a second
AsyncPoseTracker. Two pumps would each call process_once and fight over a
one-frame capture buffer. Sim experiments do not touch the hub.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from interface.camera import Pose

log = logging.getLogger(__name__)

STEPS_NAME = "rl_steps.jsonl"

COMMANDS = frozenset(("teach", "reversal", "path"))
POLICIES = frozenset(("static", "irregular", "bandit"))
DIRECTIONS = frozenset(("left", "right"))

DEFAULT_MAX_STEPS = {"teach": 150, "reversal": 30, "path": 40}
DEFAULT_PATH = ((0.4, 0.15), (0.8, 0.4), (1.2, 0.1))

STOP_REQUESTED = "requested"
STOP_REPLACED = "replaced"
STOP_SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class ExperimentSpec:
    command: str
    policy: str = "bandit"
    direction: str = "left"
    max_steps: int = 150
    waypoints: tuple[tuple[float, float], ...] | None = None


class HubPoseTracker:
    """PoseTracker that reads the newest detected pose from a FrameHub."""

    def __init__(self, hub) -> None:
        self._hub = hub
        self._seq = 0
        self._returned_t: float | None = None

    async def read(self) -> Pose:
        while True:
            self._seq, result = await self._hub.next_frame(self._seq)
            if result.px_hat is None or result.py_hat is None:
                continue
            if result.t_frame == self._returned_t:
                continue
            self._returned_t = result.t_frame
            return Pose(float(result.px_hat), float(result.py_hat), float(result.t_frame))


class _GateStim:
    """Policy pulse through the shared gate, then wait out the refractory."""

    def __init__(self, gate, cooldown: float, gain_percent: int, journal, settings_cls) -> None:
        self.gate = gate
        self.cooldown = cooldown
        self.gain_percent = gain_percent
        self.journal = journal
        self._settings_cls = settings_cls
        self._step = 0

    async def pulse(self, action) -> None:
        self._step += 1
        result = await self.gate.request(
            action.direction,
            "rl",
            f"rl-{self._step:03d}",
            settings=self._settings_cls(
                frequency_hz=action.frequency_hz,
                pulse_width_ms=action.pulse_width_ms,
                duration_ms=action.duration_ms,
                gain_percent=self.gain_percent,
                random_mode=False,
            ),
        )
        self.journal.record(result)
        await asyncio.sleep(self.cooldown)


class _PauseStim:
    def __init__(self, cooldown: float) -> None:
        self.cooldown = cooldown

    async def pulse(self, action) -> None:
        await asyncio.sleep(self.cooldown)


class ExperimentRunner:
    """Owns the teach/reversal/path task and the step series for one page."""

    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._task: asyncio.Task | None = None
        self._stop = False
        self._spec: ExperimentSpec | None = None
        self._policy = None
        self._episode = 0
        self._success: bool | None = None
        self._steps: list[dict] = []
        self._left_rad = 0.0
        self._right_rad = 0.0
        self._failure: str | None = None
        self._status_text = None

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def spec(self) -> ExperimentSpec | None:
        return self._spec

    @property
    def steps(self) -> list[dict]:
        return self._steps

    def status(self) -> dict:
        spec = self._spec
        return {
            "running": self.active,
            "command": None if spec is None else spec.command,
            "policy": None if spec is None else spec.policy,
            "direction": None if spec is None else spec.direction,
            "episode": self._episode,
            "success": self._success,
            "left_rad": self._left_rad,
            "right_rad": self._right_rad,
            "step": None if not self._steps else self._steps[-1]["step"],
            "policy_text": self._policy_text(),
            "failure": self._failure,
        }

    def snapshot(self) -> dict:
        return {**self.status(), "steps": list(self._steps)}

    def _policy_text(self) -> str:
        if self._policy is None or self._status_text is None:
            return ""
        return self._status_text(self._policy)

    def _should_stop(self) -> bool:
        return self._stop

    async def start(self, spec: ExperimentSpec) -> None:
        await self.stop(STOP_REPLACED)
        self._stop = False
        self._spec = spec
        self._episode += 1
        self._success = None
        self._steps = []
        self._left_rad = 0.0
        self._right_rad = 0.0
        self._failure = None
        self._policy = None
        hub = self._runtime.hub
        if hub is not None:
            hub.overlay.clear()
        self._runtime.journal.record_note(
            "experiment",
            f"{spec.command} {spec.policy} episode {self._episode}",
        )
        self._runtime.journal.emit({**self.status(), "type": "rl_status"})
        self._task = asyncio.create_task(self._run())

    async def stop(self, reason: str = STOP_REQUESTED) -> None:
        self._stop = True
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.debug("experiment ended with an error", exc_info=True)
        if self._spec is not None:
            self._runtime.journal.emit({**self.status(), "type": "rl_status"})

    async def restart(self) -> None:
        if self._spec is None:
            raise RuntimeError("no experiment to restart")
        spec = self._spec
        await self.stop(STOP_REPLACED)
        await self.start(spec)

    async def _run(self) -> None:
        # teach.py imports interface.roboroach, which imports bleak. A replay
        # process must not load that until someone actually starts an experiment.
        from rl_control.policy import PathPolicy, make_teach_policy, status_text
        from rl_control.teach import (
            AntiHabituationEnv,
            SilentStim,
            follow_path,
            reversal,
            teach,
        )

        self._status_text = status_text
        spec = self._spec
        assert spec is not None
        stim = self._stim(SilentStim)
        hub = self._runtime.hub
        if hub is not None:
            env = AntiHabituationEnv.wired(
                stim, tracker=HubPoseTracker(hub), direction=spec.direction
            )
        elif self._runtime.gate is not None:
            env = AntiHabituationEnv.wired(stim, direction=spec.direction)
        else:
            env = AntiHabituationEnv.simulated(direction=spec.direction)
        if spec.command == "teach":
            env.max_still = 0
        elif spec.command == "reversal":
            env.max_still = 8

        inner = make_teach_policy(spec.policy, direction=spec.direction)
        self._policy = inner
        try:
            if spec.command == "path":
                waypoints = spec.waypoints or DEFAULT_PATH
                policy = PathPolicy(waypoints, inner)
                self._policy = policy
                _, success = await follow_path(
                    env,
                    policy,
                    spec.max_steps,
                    on_step=self._on_step,
                    should_stop=self._should_stop,
                )
            elif spec.command == "reversal":
                _, _, success = await reversal(
                    env,
                    inner,
                    spec.max_steps,
                    on_step=lambda log, progress: self._on_step(log),
                    should_stop=self._should_stop,
                )
            else:
                _, success = await teach(
                    env,
                    inner,
                    spec.max_steps,
                    on_step=self._on_step,
                    should_stop=self._should_stop,
                )
            if not self._stop:
                self._success = success
                self._runtime.journal.record_note(
                    "experiment finished",
                    "success" if success else "incomplete",
                )
                self._runtime.journal.emit(
                    {**self.status(), "type": "rl_status", "running": False}
                )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            self._runtime.journal.record_note("experiment stopped", self._failure)
            self._runtime.journal.emit({**self.status(), "type": "rl_status"})
            raise

    def _stim(self, silent):
        config = self._runtime.config
        if self._runtime.gate is not None:
            from interface.roboroach import StimulationSettings

            return _GateStim(
                self._runtime.gate,
                config.t_refrac_s,
                config.gain_percent,
                self._runtime.journal,
                StimulationSettings,
            )
        if self._runtime.hub is not None:
            return _PauseStim(config.t_refrac_s)
        return silent()

    def _on_step(self, log) -> None:
        row = _step_row(log, self._episode, self._policy_text())
        self._steps.append(row)
        self._left_rad = log.left_rad
        self._right_rad = log.right_rad
        self._runtime.journal.emit(row)
        _append_step(self._runtime.config.run_dir, row)
        hub = self._runtime.hub
        if hub is not None:
            hub.overlay.walked_cm.append([row["x"], row["y"]])


def _step_row(log, episode: int, policy_text: str) -> dict:
    state = log.state
    action = log.action
    return {
        "type": "rl_step",
        "episode": episode,
        "step": log.step,
        "x": state.x,
        "y": state.y,
        "vx": state.vx,
        "vy": state.vy,
        "speed": state.speed,
        "heading_rad": state.heading_rad,
        "turn_rate_rad": state.turn_rate_rad,
        "still_steps": state.still_steps,
        "direction": action.direction,
        "frequency_hz": action.frequency_hz,
        "duration_ms": action.duration_ms,
        "pulse_width_ms": action.pulse_width_ms,
        "reward": log.reward,
        "progress_rad": log.progress_rad,
        "left_rad": log.left_rad,
        "right_rad": log.right_rad,
        "warmup": log.warmup,
        "policy_text": policy_text,
    }


def _append_step(run_dir: Path, row: dict) -> None:
    path = Path(run_dir) / STEPS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
