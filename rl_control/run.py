"""CLI for teaching tasks and the older goal-heading loop."""

from __future__ import annotations

import argparse
import asyncio
import math

from .env import SimWorld, format_step
from .policy import HeadingPolicy, PathPolicy, make_teach_policy
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
        from .plot import LiveRunPlot

        plot = LiveRunPlot(names[0])
    for name in names:
        env = AntiHabituationEnv.simulated(direction=args.direction)
        policy = make_teach_policy(name, direction=args.direction)
        if plot is not None:
            plot.reset(name)
        logs = await teach(
            env,
            policy,
            max_steps=args.max_steps,
            on_step=None if plot is None else plot.update,
        )
        print(f"policy {name}  state=sim  stim=silent")
        for log in logs:
            print(format_teach_step(log))
        print(summarize(logs))
        print()
    if plot is not None:
        plot.hold()


async def run_reversal_sim(args: argparse.Namespace) -> None:
    plot = None
    if not args.no_plot:
        from .plot import LiveRunPlot

        plot = LiveRunPlot(args.policy)
    env = AntiHabituationEnv.simulated()
    env.max_still = 8
    policy = make_teach_policy(args.policy, direction="left")

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
        default="irregular",
    )
    teach_p.add_argument(
        "--compare",
        action="store_true",
        help="run static then irregular in simulation",
    )
    teach_p.add_argument("--direction", choices=("left", "right"), default="left")
    teach_p.add_argument("--max-steps", type=int, default=16)
    teach_p.add_argument("--live", action="store_true")
    teach_p.add_argument("--no-plot", action="store_true")
    teach_p.add_argument("--cooldown", type=float, default=2.0)
    teach_p.add_argument("--timeout", type=float, default=10.0)

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
    if args.max_steps < 1:
        raise SystemExit("max-steps must be at least 1")
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
