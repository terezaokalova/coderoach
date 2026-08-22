"""Real-time dashboard of movement state and the pulse just chosen."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from .teach import StepLog


class LiveRunPlot:
    def __init__(self, title: str = "teach") -> None:
        plt.ion()
        self.fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
        (
            self.ax_xy,
            self.ax_motion,
            self.ax_vec,
            self.ax_stim,
            self.ax_reward,
            self.ax_now,
        ) = axes.ravel()
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("RoboRoach teach")
        self._lines = {}
        self._init_axes()
        self.reset(title)
        self.fig.tight_layout()
        self.fig.show()

    def _init_axes(self) -> None:
        self.ax_xy.set_title("path")
        self.ax_xy.set_xlabel("x")
        self.ax_xy.set_ylabel("y")
        self.ax_xy.set_aspect("equal", adjustable="datalim")
        (self._lines["path"],) = self.ax_xy.plot([], [], "-", color="0.2")
        (self._lines["here"],) = self.ax_xy.plot([], [], "o", color="C3", ms=8)

        self.ax_motion.set_title("motion")
        (self._lines["speed"],) = self.ax_motion.plot([], [], label="speed")
        (self._lines["turn"],) = self.ax_motion.plot([], [], label="|turn|")
        (self._lines["still"],) = self.ax_motion.plot([], [], label="still")
        self.ax_motion.legend(loc="upper right", fontsize=8)
        self.ax_motion.set_xlabel("step")

        self.ax_vec.set_title("vector / heading")
        (self._lines["vx"],) = self.ax_vec.plot([], [], label="vx")
        (self._lines["vy"],) = self.ax_vec.plot([], [], label="vy")
        (self._lines["heading"],) = self.ax_vec.plot([], [], label="heading deg")
        self.ax_vec.legend(loc="upper right", fontsize=8)
        self.ax_vec.set_xlabel("step")

        self.ax_stim.set_title("stimulation")
        (self._lines["freq"],) = self.ax_stim.plot(
            [], [], drawstyle="steps-post", label="Hz"
        )
        (self._lines["dur"],) = self.ax_stim.plot(
            [], [], drawstyle="steps-post", label="duration ms"
        )
        (self._lines["pw"],) = self.ax_stim.plot(
            [], [], drawstyle="steps-post", label="pulse ms"
        )
        self.ax_stim.legend(loc="upper right", fontsize=8)
        self.ax_stim.set_xlabel("step")

        self.ax_reward.set_title("reward")
        (self._lines["reward"],) = self.ax_reward.plot([], [], color="C2")
        self.ax_reward.axhline(0.0, color="0.7", lw=0.8)
        self.ax_reward.set_xlabel("step")

        self.ax_now.set_axis_off()
        self.ax_now.set_title("chosen pulse")
        self._now_text = self.ax_now.text(
            0.04,
            0.96,
            "",
            va="top",
            family="monospace",
            fontsize=11,
            transform=self.ax_now.transAxes,
        )

    def reset(self, title: str) -> None:
        self.fig.suptitle(title)
        self.steps: list[int] = []
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.speed: list[float] = []
        self.turn: list[float] = []
        self.still: list[int] = []
        self.vx: list[float] = []
        self.vy: list[float] = []
        self.heading: list[float] = []
        self.freq: list[int] = []
        self.dur: list[int] = []
        self.pw: list[int] = []
        self.reward: list[float] = []
        for line in self._lines.values():
            line.set_data([], [])
        self._now_text.set_text("")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update(self, log: StepLog) -> None:
        state = log.state
        action = log.action
        self.steps.append(log.step)
        self.xs.append(state.x)
        self.ys.append(state.y)
        self.speed.append(state.speed)
        self.turn.append(abs(state.turn_rate_rad))
        self.still.append(state.still_steps)
        self.vx.append(state.vx)
        self.vy.append(state.vy)
        self.heading.append(math.degrees(state.heading_rad))
        self.freq.append(action.frequency_hz)
        self.dur.append(action.duration_ms)
        self.pw.append(action.pulse_width_ms)
        self.reward.append(log.reward)

        self._lines["path"].set_data(self.xs, self.ys)
        self._lines["here"].set_data([state.x], [state.y])
        self._lines["speed"].set_data(self.steps, self.speed)
        self._lines["turn"].set_data(self.steps, self.turn)
        self._lines["still"].set_data(self.steps, self.still)
        self._lines["vx"].set_data(self.steps, self.vx)
        self._lines["vy"].set_data(self.steps, self.vy)
        self._lines["heading"].set_data(self.steps, self.heading)
        self._lines["freq"].set_data(self.steps, self.freq)
        self._lines["dur"].set_data(self.steps, self.dur)
        self._lines["pw"].set_data(self.steps, self.pw)
        self._lines["reward"].set_data(self.steps, self.reward)

        for axis in (
            self.ax_xy,
            self.ax_motion,
            self.ax_vec,
            self.ax_stim,
            self.ax_reward,
        ):
            axis.relim()
            axis.autoscale_view()

        self._now_text.set_text(
            f"step {log.step}\n"
            f"dir   {action.direction}\n"
            f"freq  {action.frequency_hz} Hz\n"
            f"dur   {action.duration_ms} ms\n"
            f"pulse {action.pulse_width_ms} ms\n"
            f"\n"
            f"x,y   {state.x:+.3f}, {state.y:+.3f}\n"
            f"v     {state.vx:+.3f}, {state.vy:+.3f}\n"
            f"speed {state.speed:.3f}\n"
            f"turn  {state.turn_rate_rad:+.3f}\n"
            f"head  {math.degrees(state.heading_rad):+.1f} deg\n"
            f"still {state.still_steps}\n"
            f"r     {log.reward:+.2f}"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def hold(self) -> None:
        plt.ioff()
        plt.show(block=True)
