"""CLI for teaching tasks and the older goal-heading loop."""

from __future__ import annotations

import argparse
import asyncio
import math
from pathlib import Path

from interface.roboroach import MIN_STIM_INTERVAL_S, RoboRoach

from .env import SimWorld, StimAction, format_step
from .policy import HeadingPolicy, PathPolicy, make_teach_policy, status_text
from .teach import (
    AntiHabituationEnv,
    follow_path,
    format_path_step,
    format_reversal_step,
    format_teach_step,
    reversal,
    summarize,
    summarize_path,
    summarize_reversal,
    teach,
)

DEFAULT_PATH = ((0.4, 0.15), (0.8, 0.4), (1.2, 0.1))


class PauseStim:
    def __init__(self, cooldown: float) -> None:
        self.cooldown = cooldown

    async def pulse(self, action: StimAction) -> None:
        await asyncio.sleep(self.cooldown)


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        default="phone",
        help="phone, camera index, jpeg path, or HTTP/RTSP URL",
    )
    parser.add_argument("--tracker", choices=("blob", "csrt"), default="blob")
    parser.add_argument(
        "--sim-pose",
        action="store_true",
        help="keep the simulated animal instead of the iPhone tracker",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="use the iPhone stream without pulsing the backpack",
    )


TRAJ_ARGS = (
    ("traj_camera", "--traj-camera"),
    ("traj_hsv", "--traj-hsv"),
    ("traj_arena", "--traj-arena"),
    ("traj_min_area", "--traj-min-area"),
    ("traj_sigma_p", "--traj-sigma-p"),
    ("traj_sigma_a", "--traj-sigma-a"),
    ("traj_v_min", "--traj-v-min"),
)


def add_gate_args(parser: argparse.ArgumentParser) -> None:
    """What StimGate needs and has no default for."""
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="directory for the gate log; required with --live",
    )
    parser.add_argument(
        "--gain-percent",
        type=int,
        default=10,
        help="stimulation gain; 10 matches the RoboRoach.configure() default "
        "this path used before the gate (default: 10)",
    )


def add_pose_args(parser: argparse.ArgumentParser) -> None:
    """Choose the pose source. --tracker is already taken by blob/csrt."""
    parser.add_argument(
        "--pose-source",
        choices=("sim", "camera"),
        default="sim",
        help="sim keeps the simulated animal; camera reads pose from the "
        "src/traj overhead tracker (default: sim)",
    )
    parser.add_argument(
        "--still-speed",
        type=float,
        default=None,
        help="displacement between consecutive poses below which the animal "
        "counts as still, in whatever units the pose source reports -- "
        "simulated animal walks ~0.09 per step, the src/traj tracker reports "
        "centimetres. Despite the name it is compared against per-step "
        "displacement, not speed. Default: 0.01 with a camera, 0.02 with the "
        "simulated animal",
    )
    parser.add_argument("--traj-camera", type=int, help="camera index")
    parser.add_argument("--traj-hsv", type=Path, help="HSV bounds JSON")
    parser.add_argument("--traj-arena", type=Path, help="arena homography JSON")
    parser.add_argument("--traj-min-area", type=float, help="min contour area, px^2")
    parser.add_argument("--traj-sigma-p", type=float, help="centroid noise, cm")
    parser.add_argument("--traj-sigma-a", type=float, help="process noise, cm/s^2")
    parser.add_argument("--traj-v-min", type=float, help="heading cutoff speed, cm/s")


def open_traj_tracker(args: argparse.Namespace):
    """Build src/traj's AsyncPoseTracker from the --traj-* arguments.

    TrackerConfig gives none of these a default on purpose: each is a property
    of one physical setup, and a default would be a guess that looks like a
    measurement. So every one is required here too, reported together rather
    than one per run.
    """
    try:
        from traj.track import AsyncPoseTracker, HsvBounds, TrackerConfig
    except ImportError as exc:  # pragma: no cover - depends on sys.path
        raise SystemExit(
            f"--pose-source camera needs src/ on the path: {exc}. "
            "Run with PYTHONPATH=src."
        ) from exc

    missing = [flag for name, flag in TRAJ_ARGS if getattr(args, name) is None]
    if args.run_dir is None:
        missing.append("--run-dir")
    if missing:
        raise SystemExit(f"--pose-source camera needs {' '.join(missing)}")

    return AsyncPoseTracker(
        TrackerConfig(
            camera_index=args.traj_camera,
            hsv_bounds=HsvBounds.from_json(args.traj_hsv),
            min_contour_area=args.traj_min_area,
            sigma_p_cm=args.traj_sigma_p,
            sigma_a_cm_s2=args.traj_sigma_a,
            v_min_cm_s=args.traj_v_min,
            arena_calibration=args.traj_arena,
            run_dir=args.run_dir,
        )
    )


def parse_waypoint(text: str) -> tuple[float, float]:
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("waypoint must be x,y")
    return float(parts[0]), float(parts[1])


def run_goal_sim(args: argparse.Namespace) -> None:
    world = SimWorld(
        start=(args.start_x, args.start_y, math.radians(args.start_heading)),
        goal=(args.goal_x, args.goal_y),
    )
    policy = HeadingPolicy(arrive_radius=args.arrive)
    obs = world.observe()
    for step in range(1, args.max_steps + 1):
        action = policy.act(obs)
        print(format_step(step, obs, action))
        if obs.distance <= args.arrive:
            print("arrived")
            return
        obs = world.step(action)
    print("max steps reached")


async def run_teach_sim(args: argparse.Namespace) -> None:
    names = ("static", "irregular") if args.compare else (args.policy,)
    plot = None
    if not args.no_plot:
        from interface.plot import LiveRunPlot

        plot = LiveRunPlot(names[0])
    for name in names:
        env = AntiHabituationEnv.simulated(
            direction=args.direction, still_speed=args.still_speed
        )
        policy = make_teach_policy(name, direction=args.direction)
        if plot is not None:
            plot.extra = lambda current=policy: status_text(current)
            plot.reset(name)
        logs, success = await teach(
            env,
            policy,
            max_steps=args.max_steps,
            on_step=None if plot is None else plot.update,
        )
        print(f"policy {name}  state=sim  stim=silent")
        for log in logs:
            print(format_teach_step(log))
        print(summarize(logs, success))
        print()
    if plot is not None:
        plot.hold()


async def run_reversal_sim(args: argparse.Namespace) -> None:
    plot = None
    if not args.no_plot:
        from interface.plot import LiveRunPlot

        plot = LiveRunPlot(args.policy)
    env = AntiHabituationEnv.simulated(still_speed=args.still_speed)
    env.max_still = 8
    policy = make_teach_policy(args.policy, direction="left")
    if plot is not None:
        plot.extra = lambda: status_text(policy)

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
    print(f"policy {args.policy}  state=sim  stim=silent")
    print(summarize_reversal(logs, progress, success))
    if plot is not None:
        plot.hold()


async def run_camera_teach(args: argparse.Namespace) -> None:
    if args.compare:
        raise SystemExit("compare is simulation-only")
    from interface.plot import make_live_plot, run_with_dashboard
    from interface.track import open_camera

    traj = args.pose_source == "camera"
    if traj:
        tracker = open_traj_tracker(args)
        await tracker.start()
    else:
        tracker = open_camera(args)
    if tracker is None:
        raise SystemExit("camera teach needs the phone stream, not --sim-pose")
    policy = make_teach_policy(args.policy, direction=args.direction)
    env = AntiHabituationEnv.wired(
        PauseStim(args.cooldown),
        direction=args.direction,
        tracker=tracker,
        still_speed=args.still_speed,
    )
    env.max_still = 0
    plot = None
    if not args.no_plot and not traj:
        plot = make_live_plot(
            f"{args.policy}  state=camera  stim=silent",
            tracker,
            extra=lambda: status_text(policy),
        )
    second = "right" if args.direction == "left" else "left"
    print(
        "Camera teach: pose from iPhone, no backpack pulse. "
        f"180 {args.direction}, then 180 {second}, or {args.max_steps} observations."
    )

    def on_step(log) -> None:
        if plot is not None:
            plot.update(log)
        print(format_teach_step(log), flush=True)

    async def work():
        return await teach(
            env,
            policy,
            max_steps=args.max_steps,
            on_step=on_step,
            should_stop=None if plot is None else plot.closed,
        )

    try:
        logs, success = await run_with_dashboard(plot, tracker, work)
        print(summarize(logs, success))
    except RuntimeError as exc:
        print(exc)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if tracker is not None:
            tracker.stop()


async def run_path_sim(args: argparse.Namespace) -> None:
    waypoints = tuple(args.waypoints) if args.waypoints else DEFAULT_PATH
    env = AntiHabituationEnv.simulated()
    policy = PathPolicy(
        waypoints,
        make_teach_policy(args.policy),
        arrive_radius=args.arrive,
    )
    logs, success = await follow_path(
        env,
        policy,
        max_steps=args.max_steps,
        on_step=lambda log: print(format_path_step(log, policy)),
    )
    print(f"policy {args.policy}  waypoints {len(waypoints)}")
    print(summarize_path(logs, success))


async def run_spam(args: argparse.Namespace) -> None:
    total = 2 * args.pulses
    sent = 0
    print(
        f"Open-loop turn: {args.pulses} left pulses, then "
        f"{args.pulses} right pulses, {args.cooldown:g} s apart."
    )
    async with RoboRoach(scan_timeout=args.timeout) as roach:
        await roach.configure()
        for direction in ("left", "right"):
            for pulse in range(1, args.pulses + 1):
                await roach.turn(direction)
                sent += 1
                print(f"{direction} {pulse}/{args.pulses}", flush=True)
                if sent < total:
                    await asyncio.sleep(args.cooldown)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    teach_p = sub.add_parser(
        "teach",
        help="anti-habituation: vary pulse frequency and duration",
    )
    teach_p.add_argument(
        "--policy",
        choices=("static", "irregular", "bandit"),
        default="bandit",
    )
    teach_p.add_argument(
        "--compare",
        action="store_true",
        help="run static then irregular in simulation",
    )
    teach_p.add_argument("--direction", choices=("left", "right"), default="left")
    teach_p.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="stop after this many pulses or after completing both turns",
    )
    teach_p.add_argument("--live", action="store_true")
    teach_p.add_argument("--no-plot", action="store_true")
    teach_p.add_argument("--cooldown", type=float, default=2.0)
    teach_p.add_argument("--timeout", type=float, default=10.0)
    add_camera_args(teach_p)
    add_gate_args(teach_p)
    add_pose_args(teach_p)

    rev_p = sub.add_parser(
        "reversal",
        help="success test: 180 deg left, then 180 deg right",
    )
    rev_p.add_argument(
        "--policy",
        choices=("static", "irregular", "bandit"),
        default="bandit",
    )
    rev_p.add_argument("--max-steps", type=int, default=30)
    rev_p.add_argument("--live", action="store_true")
    rev_p.add_argument("--no-plot", action="store_true")
    rev_p.add_argument("--cooldown", type=float, default=2.0)
    rev_p.add_argument("--timeout", type=float, default=10.0)
    add_camera_args(rev_p)
    add_gate_args(rev_p)
    add_pose_args(rev_p)

    spam_p = sub.add_parser(
        "spam",
        help="send a bounded left sequence followed by a right sequence",
    )
    spam_p.add_argument("--pulses", type=int, default=6)
    spam_p.add_argument("--cooldown", type=float, default=MIN_STIM_INTERVAL_S)
    spam_p.add_argument("--timeout", type=float, default=10.0)

    goal_p = sub.add_parser("goal", help="heading correction toward a point")
    goal_p.add_argument("--live", action="store_true")
    goal_p.add_argument("--max-steps", type=int, default=20)
    goal_p.add_argument("--arrive", type=float, default=0.12)
    goal_p.add_argument("--cooldown", type=float, default=2.0)
    goal_p.add_argument("--timeout", type=float, default=10.0)
    goal_p.add_argument("--start-x", type=float, default=0.0)
    goal_p.add_argument("--start-y", type=float, default=0.0)
    goal_p.add_argument("--start-heading", type=float, default=0.0)
    goal_p.add_argument("--goal-x", type=float, default=1.0)
    goal_p.add_argument("--goal-y", type=float, default=0.4)
    add_gate_args(goal_p)

    path_p = sub.add_parser(
        "path",
        help="follow waypoints with a swappable stim policy underneath",
    )
    path_p.add_argument(
        "--policy",
        choices=("static", "irregular", "bandit"),
        default="bandit",
    )
    path_p.add_argument(
        "--waypoint",
        dest="waypoints",
        action="append",
        type=parse_waypoint,
        help="x,y pair; repeat for a path. Default is a 3-point bend",
    )
    path_p.add_argument("--max-steps", type=int, default=40)
    path_p.add_argument("--arrive", type=float, default=0.12)

    args = parser.parse_args()
    if getattr(args, "max_steps", 1) < 1:
        raise SystemExit("max-steps must be at least 1")
    if getattr(args, "pulses", 1) < 1:
        raise SystemExit("pulses must be at least 1")
    if args.command == "spam" and args.cooldown < MIN_STIM_INTERVAL_S:
        raise SystemExit(f"cooldown must be at least {MIN_STIM_INTERVAL_S:g} seconds")
    if getattr(args, "live", False) and args.cooldown < MIN_STIM_INTERVAL_S:
        raise SystemExit(
            f"cooldown must be at least {MIN_STIM_INTERVAL_S:g} seconds for live stim"
        )
    if getattr(args, "live", False) and args.run_dir is None:
        # StimGate writes its log here and has no default. Failing now beats
        # failing after the backpack has already connected.
        raise SystemExit(f"{args.command} --live needs --run-dir")
    if getattr(args, "pose_source", "sim") == "camera" and not (
        getattr(args, "live", False) or getattr(args, "camera", False)
    ):
        raise SystemExit("--pose-source camera needs --live or --camera")
    if args.command == "spam":
        asyncio.run(run_spam(args))
        return
    if args.command == "teach" and args.camera:
        asyncio.run(run_camera_teach(args))
        return
    if args.command == "teach":
        if args.live:
            from .live import run_live_teach

            asyncio.run(run_live_teach(args))
            return
        asyncio.run(run_teach_sim(args))
        return
    if args.command == "reversal":
        if args.live:
            from .live import run_live_reversal

            asyncio.run(run_live_reversal(args))
            return
        asyncio.run(run_reversal_sim(args))
        return
    if args.command == "path":
        asyncio.run(run_path_sim(args))
        return
    if args.live:
        from .live import run_live

        asyncio.run(run_live(args))
        return
    run_goal_sim(args)


if __name__ == "__main__":
    main()
