"""Hardware path for the bounded goal loop. Importing this opens the BLE stack."""

from __future__ import annotations

import argparse
import asyncio
import math
from contextlib import suppress

from interface import RoboRoach, StimulationSettings

from stim import StimGate

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


def waveform(action: StimAction, gain_percent: int) -> StimulationSettings:
    """The policy's chosen pulse, as a per-request settings override.

    The gate reconfigures the board from this before the write, which is what
    the configure()-then-turn() pair used to do inline. gain_percent is not part
    of StimAction, so it comes from the CLI; its default matches the
    RoboRoach.configure() default this path relied on before.
    """
    return StimulationSettings(
        frequency_hz=action.frequency_hz,
        pulse_width_ms=action.pulse_width_ms,
        duration_ms=action.duration_ms,
        gain_percent=gain_percent,
        random_mode=False,
    )


def report(result) -> None:
    """Print a gate rejection.

    The return value used to be discarded. A refractory rejection is silent on
    the board, so without this the policy credits itself with a pulse that never
    left and learns from the reward that followed nothing.
    """
    if not result.accepted:
        print(f"  gate rejected {result.request_id}: {result.reject_reason}")


async def open_gate(roach: RoboRoach, args) -> StimGate:
    """One gate per live run, with --cooldown as the refractory period."""
    return await StimGate.create(
        roach=roach,
        t_refrac_s=args.cooldown,
        settings=waveform(StimAction(direction="left"), args.gain_percent),
        run_dir=args.run_dir,
    )


async def open_pose_tracker(args):
    """His iPhone tracker, our src/traj tracker, or None for the sim animal."""
    if getattr(args, "pose_source", "sim") == "camera":
        from .run import open_traj_tracker

        tracker = open_traj_tracker(args)
        await tracker.start()
        return tracker
    from interface.track import open_camera

    return open_camera(args)


class BackpackStim:
    """Stimulator that requests through the shared gate instead of the board.

    The gate owns the refractory period and the trial counter, so an RL step
    arriving inside the refractory window is rejected there rather than reaching
    turn(). Its own sleep stays: the gate rejects early requests, it does not
    pace them, and removing the sleep would turn every step after the first into
    a rejection.
    """

    def __init__(self, gate: StimGate, cooldown: float, gain_percent: int) -> None:
        self.gate = gate
        self.cooldown = cooldown
        self.gain_percent = gain_percent
        self._step = 0

    async def pulse(self, action: StimAction) -> None:
        self._step += 1
        result = await self.gate.request(
            action.direction,
            "rl",
            f"rl-{self._step:03d}",
            settings=waveform(action, self.gain_percent),
        )
        report(result)
        await asyncio.sleep(self.cooldown)


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
            gate = await open_gate(roach, args)
            obs = world.observe()
            for step in range(1, args.max_steps + 1):
                action = policy.act(obs)
                print(format_step(step, obs, action))
                if obs.distance <= args.arrive:
                    print("arrived")
                    return
                if action.direction != "wait":
                    report(
                        await gate.request(
                            action.direction,
                            "rl",
                            f"rl-goal-{step:03d}",
                            settings=waveform(action, args.gain_percent),
                        )
                    )
                    await asyncio.sleep(args.cooldown)
                obs = world.step(action)
            print("max steps reached")
        except RuntimeError as exc:
            print(exc)
        finally:
            keepalive.cancel()
            with suppress(asyncio.CancelledError):
                await keepalive


async def run_live_teach(args: argparse.Namespace) -> None:
    if args.compare:
        raise SystemExit("compare is simulation-only")
    tracker = await open_pose_tracker(args)
    pose = "camera" if tracker is not None else "sim"
    first = args.direction
    second = "right" if first == "left" else "left"
    print(
        f"Live teach: stim on backpack, pose from {pose}. "
        f"180 {first} then 180 {second}, or {args.max_steps} pulses, "
        f"{args.cooldown:.0f} s apart."
    )
    policy = make_teach_policy(args.policy, direction=args.direction)
    async with RoboRoach(scan_timeout=args.timeout) as roach:
        keepalive = asyncio.create_task(roach.keep_alive())
        try:
            gate = await open_gate(roach, args)
            env = AntiHabituationEnv.wired(
                BackpackStim(gate, args.cooldown, args.gain_percent),
                direction=args.direction,
                tracker=tracker,
                still_speed=args.still_speed,
            )
            env.max_still = 0

            def on_step(log) -> None:
                print(format_teach_step(log), flush=True)

            logs, success = await teach(
                env,
                policy,
                max_steps=args.max_steps,
                on_step=on_step,
            )
            print(summarize(logs, success))
        except RuntimeError as exc:
            print(exc)
        except KeyboardInterrupt:
            print("aborted")
        finally:
            keepalive.cancel()
            with suppress(asyncio.CancelledError):
                await keepalive
            if tracker is not None:
                tracker.stop()


async def run_live_reversal(args: argparse.Namespace) -> None:
    tracker = await open_pose_tracker(args)
    pose = "camera" if tracker is not None else "sim"
    print(
        f"Live reversal: 180 left then 180 right. "
        f"Stim on backpack, pose from {pose}. "
        f"At most {args.max_steps} pulses, {args.cooldown:.0f} s apart."
    )
    policy = make_teach_policy(args.policy, direction="left")
    async with RoboRoach(scan_timeout=args.timeout) as roach:
        keepalive = asyncio.create_task(roach.keep_alive())
        try:
            gate = await open_gate(roach, args)
            env = AntiHabituationEnv.wired(
                BackpackStim(gate, args.cooldown, args.gain_percent),
                tracker=tracker,
                still_speed=args.still_speed,
            )
            env.max_still = 8

            def on_step(log, progress: dict[str, float]) -> None:
                print(format_reversal_step(log, progress))

            logs, progress, success = await reversal(
                env,
                policy,
                max_steps=args.max_steps,
                on_step=on_step,
            )
            print(summarize_reversal(logs, progress, success))
        except RuntimeError as exc:
            print(exc)
        except KeyboardInterrupt:
            print("aborted")
        finally:
            keepalive.cancel()
            with suppress(asyncio.CancelledError):
                await keepalive
            if tracker is not None:
                tracker.stop()
