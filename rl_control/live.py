"""Hardware path for the bounded goal loop. Importing this opens the BLE stack."""

from __future__ import annotations

import argparse
import asyncio
import math
from contextlib import suppress

from interface import RoboRoach

from .env import SimWorld, StimAction, format_step
from .policy import HeadingPolicy, make_teach_policy
from .teach import (
    AntiHabituationEnv,
    format_reversal_step,
    format_teach_step,
    reversal,
    summarize,
    summarize_reversal,
    teach,
)


async def run_live(args: argparse.Namespace) -> None:
    print(
        f"Live goal: stim on backpack, pose from sim (no camera). "
        f"At most {args.max_steps} pulses, {args.cooldown:.0f} s apart."
    )
    world = SimWorld(
        start=(args.start_x, args.start_y, math.radians(args.start_heading)),
        goal=(args.goal_x, args.goal_y),
    )
    policy = HeadingPolicy(arrive_radius=args.arrive)
    async with RoboRoach(scan_timeout=args.timeout) as roach:
        keepalive = asyncio.create_task(roach.keep_alive())
        try:
            obs = world.observe()
            for step in range(1, args.max_steps + 1):
                action = policy.act(obs)
                print(format_step(step, obs, action))
                if obs.distance <= args.arrive:
                    print("arrived")
                    return
                if action.direction != "wait":
                    await roach.configure(
                        frequency_hz=action.frequency_hz,
                        pulse_width_ms=action.pulse_width_ms,
                        duration_ms=action.duration_ms,
                    )
                    await roach.turn(action.direction)
                    await asyncio.sleep(args.cooldown)
                obs = world.step(action)
            print("max steps reached")
        except RuntimeError as exc:
            print(exc)
        finally:
            keepalive.cancel()
            with suppress(asyncio.CancelledError):
                await keepalive


class BackpackStim:
    def __init__(self, roach: RoboRoach, cooldown: float) -> None:
        self.roach = roach
        self.cooldown = cooldown

    async def pulse(self, action: StimAction) -> None:
        await self.roach.configure(
            frequency_hz=action.frequency_hz,
            pulse_width_ms=action.pulse_width_ms,
            duration_ms=action.duration_ms,
        )
        await self.roach.turn(action.direction)
        await asyncio.sleep(self.cooldown)


async def run_live_teach(args: argparse.Namespace) -> None:
    if args.compare:
        raise SystemExit("compare is simulation-only")
    print(
        f"Live teach: stim on backpack, pose from sim (no camera). "
        f"At most {args.max_steps} pulses, {args.cooldown:.0f} s apart."
    )
    policy = make_teach_policy(args.policy, direction=args.direction)
    async with RoboRoach(scan_timeout=args.timeout) as roach:
        keepalive = asyncio.create_task(roach.keep_alive())
        try:
            env = AntiHabituationEnv.wired(
                BackpackStim(roach, args.cooldown),
                direction=args.direction,
            )
            plot = None
            if not args.no_plot:
                from .plot import LiveRunPlot

                plot = LiveRunPlot(f"{args.policy}  state=sim  stim=backpack")
            logs = await teach(
                env,
                policy,
                max_steps=args.max_steps,
                on_step=None if plot is None else plot.update,
            )
            for log in logs:
                print(format_teach_step(log))
            print(summarize(logs))
            if plot is not None:
                plot.hold()
        except RuntimeError as exc:
            print(exc)
        except KeyboardInterrupt:
            print("aborted")
        finally:
            keepalive.cancel()
            with suppress(asyncio.CancelledError):
                await keepalive


async def run_live_reversal(args: argparse.Namespace) -> None:
    print(
        f"Live reversal: 180 left then 180 right. "
        f"Stim on backpack, pose from sim (no camera). "
        f"At most {args.max_steps} pulses, {args.cooldown:.0f} s apart."
    )
    policy = make_teach_policy(args.policy, direction="left")
    async with RoboRoach(scan_timeout=args.timeout) as roach:
        keepalive = asyncio.create_task(roach.keep_alive())
        try:
            env = AntiHabituationEnv.wired(BackpackStim(roach, args.cooldown))
            env.max_still = 8
            plot = None
            if not args.no_plot:
                from .plot import LiveRunPlot

                plot = LiveRunPlot(f"{args.policy}  reversal  stim=backpack")

            def on_step(log, progress: dict[str, float]) -> None:
                if plot is not None:
                    plot.update(log)
                print(format_reversal_step(log, progress))

            logs, progress, success = await reversal(
                env,
                policy,
                max_steps=args.max_steps,
                on_step=on_step,
            )
            print(summarize_reversal(logs, progress, success))
            if plot is not None:
                plot.hold()
        except RuntimeError as exc:
            print(exc)
        except KeyboardInterrupt:
            print("aborted")
        finally:
            keepalive.cancel()
            with suppress(asyncio.CancelledError):
                await keepalive
