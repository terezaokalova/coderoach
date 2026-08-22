"""Real-time dashboard: camera, movement, stimulation, and policy values."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from contextlib import suppress

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


class LiveRunPlot:
    def __init__(
        self,
        title: str = "teach",
        extra: Callable[[], str] | None = None,
        on_box: (
            Callable[[tuple[float, float], tuple[float, float]], None] | None
        ) = None,
        image_xy: bool = False,
    ) -> None:
        plt.ion()
        self.extra = extra
        self._on_box = on_box
        self.image_xy = image_xy
        self.fig = plt.figure(figsize=(16.4, 8.2))
        grid = self.fig.add_gridspec(
            2, 4, width_ratios=[1.35, 1, 1, 1.05], wspace=0.32, hspace=0.35
        )
        self.ax_video = self.fig.add_subplot(grid[:, 0])
        self.ax_xy = self.fig.add_subplot(grid[0, 1])
        self.ax_motion = self.fig.add_subplot(grid[0, 2])
        self.ax_stim = self.fig.add_subplot(grid[0, 3])
        self.ax_vec = self.fig.add_subplot(grid[1, 1])
        self.ax_reward = self.fig.add_subplot(grid[1, 2])
        self.ax_now = self.fig.add_subplot(grid[1, 3])
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("RoboRoach live")
        self._lines = {}
        self._drag: tuple[float, float] | None = None
        self._rect = Rectangle((0, 0), 0, 0, fill=False, ec="C1", lw=1.4)
        self.ax_video.add_patch(self._rect)
        self._im = self.ax_video.imshow(np.zeros((360, 480, 3), dtype=np.uint8))
        self.ax_video.set_axis_off()
        self._init_axes()
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.reset(title)
        self.fig.show()

    def _init_axes(self) -> None:
        self.ax_xy.set_title("path")
        self.ax_xy.set_xlabel("x")
        self.ax_xy.set_ylabel("y")
        self.ax_xy.set_aspect("equal", adjustable="box")
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
        self.ax_now.set_title("pulse / policy")
        self._now_text = self.ax_now.text(
            0.02,
            0.98,
            "",
            va="top",
            family="monospace",
            fontsize=8.5,
            transform=self.ax_now.transAxes,
        )

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax_video or event.xdata is None or event.ydata is None:
            return
        self._drag = (event.xdata, event.ydata)

    def _on_motion(self, event) -> None:
        if (
            self._drag is None
            or event.inaxes != self.ax_video
            or event.xdata is None
            or event.ydata is None
        ):
            return
        x0, y0 = self._drag
        self._rect.set_xy((min(x0, event.xdata), min(y0, event.ydata)))
        self._rect.set_width(abs(event.xdata - x0))
        self._rect.set_height(abs(event.ydata - y0))
        self.fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._drag is None or event.xdata is None or event.ydata is None:
            self._drag = None
            return
        if self._on_box is not None:
            self._on_box(self._drag, (event.xdata, event.ydata))
        self._drag = None
        self._rect.set_width(0)
        self._rect.set_height(0)

    def set_frame(self, rgb: np.ndarray | None, status: str = "") -> None:
        if rgb is None:
            self.ax_video.set_title(status or "camera")
            return
        self._im.set_data(rgb)
        rows, cols = rgb.shape[:2]
        self._im.set_extent((-0.5, cols - 0.5, rows - 0.5, -0.5))
        self.ax_video.set_xlim(-0.5, cols - 0.5)
        self.ax_video.set_ylim(rows - 0.5, -0.5)
        self.ax_video.set_title(status or "camera")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def closed(self) -> bool:
        return not plt.fignum_exists(self.fig.number)

    async def pump_video(self, tracker) -> None:
        while not self.closed():
            self.set_frame(tracker.view_rgb(), tracker.status)
            await asyncio.sleep(0.05)

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
        if self.image_xy:
            self.ax_xy.set_xlim(0.0, 1.0)
            self.ax_xy.set_ylim(1.0, 0.0)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update(self, log) -> None:
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

        for axis in (self.ax_motion, self.ax_vec, self.ax_stim, self.ax_reward):
            axis.relim()
            axis.autoscale_view()
        if not self.image_xy:
            self.ax_xy.relim()
            self.ax_xy.autoscale_view()

        self._now_text.set_text(self._status_text(log))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def _status_text(self, log) -> str:
        state = log.state
        action = log.action
        lines = [
            f"step {log.step}",
            f"dir   {action.direction}",
            f"freq  {action.frequency_hz} Hz",
            f"dur   {action.duration_ms} ms",
            f"pulse {action.pulse_width_ms} ms",
            "",
            f"x,y   {state.x:+.3f}, {state.y:+.3f}",
            f"v     {state.vx:+.3f}, {state.vy:+.3f}",
            f"speed {state.speed:.3f}",
            f"turn  {state.turn_rate_rad:+.3f} rad/s",
            f"head  {math.degrees(state.heading_rad):+.1f} deg",
            f"still {state.still_steps}",
            f"r     {log.reward:+.2f}",
            "",
            "envelope  1-10 Hz  1 ms  200-300 ms",
            "gap 2 s   gain <= 10%",
        ]
        extra = self.extra() if self.extra is not None else ""
        if extra:
            lines.extend(["", extra])
        return "\n".join(lines)

    def hold(self) -> None:
        plt.ioff()
        plt.show(block=True)


def make_live_plot(title: str, tracker, extra=None) -> LiveRunPlot:
    return LiveRunPlot(
        title,
        extra=extra,
        on_box=None if tracker is None else tracker.set_box,
        image_xy=tracker is not None,
    )


async def run_with_dashboard(plot, tracker, work):
    pump = None
    if plot is not None and tracker is not None:
        pump = asyncio.create_task(plot.pump_video(tracker))
        print("Click the roach in the video panel. The run starts after lock.")
        await tracker.read()
    try:
        return await work()
    finally:
        if pump is not None:
            pump.cancel()
            with suppress(asyncio.CancelledError):
                await pump
        if tracker is not None:
            tracker.stop()
