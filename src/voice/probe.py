"""Drive one voice command end to end: capture, ASR, match, gate.

``--dry-run`` stops after the match and prints it. That is the mode to use with
no backpack connected, and it imports no Bluetooth stack at all. Without it the
matched direction goes to the shared :class:`StimGate`, which owns the
refractory period and the trial counter; nothing here writes to the board on
its own, so the voice path cannot outrun the stimulation budget that every
other source is held to.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from uuid import uuid4

from .asr import Transcriber
from .audio import DEFAULT_INPUT_DEVICE, record_window
from .command import REPEAT_PROMPT, TOO_QUIET, VoiceCommand, interpret


def list_devices() -> None:
    import sounddevice as sd

    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            marker = " <- module default" if index == DEFAULT_INPUT_DEVICE else ""
            print(f"{index:>3}  {device['name']}{marker}")
    print(f"\nOS default input: {sd.default.device[0]}")


def capture_command(transcriber: Transcriber, args: argparse.Namespace) -> VoiceCommand:
    """Record until something clears the peak gate, or until attempts run out.

    Only ``too_quiet`` is retried. A phrase that was heard clearly and still did
    not match is a different failure -- the speaker said something else -- and
    repeating the recording does not address it.
    """
    for attempt in range(1, args.attempts + 1):
        print(f"\n[{attempt}/{args.attempts}] say 'turn left' or 'turn right'...")
        recording = record_window(args.seconds, device=args.device)
        command = interpret(recording, transcriber)
        print(f"  peak {command.peak:.3f}")

        if command.reject_reason != TOO_QUIET:
            return command
        print(f"  {REPEAT_PROMPT}")

    return command


async def deliver(command: VoiceCommand, args: argparse.Namespace) -> int:
    """Hand one matched direction to the gate over a live connection."""
    # Imported here so --dry-run never pulls in bleak or touches the adapter.
    from interface import RoboRoach, StimulationSettings

    from stim import StimGate

    settings = StimulationSettings(
        frequency_hz=args.frequency_hz,
        pulse_width_ms=args.pulse_width_ms,
        duration_ms=args.duration_ms,
        gain_percent=args.gain_percent,
        random_mode=False,
    )
    # Unique per run rather than a per-run counter: the gate log is appended to
    # across runs, and a bare "voice-001" in it would not say which run it came
    # from.
    request_id = f"voice-{uuid4().hex[:8]}"

    async with RoboRoach(scan_timeout=args.scan_timeout) as roach:
        gate = await StimGate.create(
            roach=roach,
            t_refrac_s=args.t_refrac,
            settings=settings,
            run_dir=args.run_dir,
        )
        print(f"board settings {gate.settings} (settings_id {gate.settings_id})")
        result = await gate.request(command.direction, "voice", request_id)

    if result.accepted:
        print(f"stimulated {result.direction} (request {request_id}, n={result.n})")
        return 0
    print(f"gate rejected {request_id}: {result.reject_reason}")
    return 1


async def main_async(args: argparse.Namespace) -> int:
    # Loading dominates the wall clock and happens before the mic opens, so it
    # is timed on its own: a slow run is nearly always a slow load rather than
    # slow recognition, and reporting the two as one number hides that.
    t_load_start = time.monotonic()
    transcriber = Transcriber(args.model)
    print(f"model {args.model!r} loaded in {time.monotonic() - t_load_start:.1f} s")

    command = capture_command(transcriber, args)

    if not command.accepted:
        if command.raw_text:
            print(f"  heard {command.heard!r} (raw {command.raw_text.strip()!r})")
        print(f"\nrejected: {command.reject_reason}")
        return 1

    print(f"  heard {command.heard!r} -> {command.direction}")

    if args.dry_run:
        print(f"\ndry run: matched {command.direction}, gate not called")
        return 0

    return await deliver(command, args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--model", default="base.en")
    parser.add_argument(
        "--device",
        default=None,
        help=f"input device index or name (default {DEFAULT_INPUT_DEVICE})",
    )
    parser.add_argument(
        "--attempts", type=int, default=3, help="retries allowed on too_quiet"
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the match and stop, without connecting or stimulating",
    )
    parser.add_argument("--run-dir", type=Path, help="gate log directory (live only)")
    parser.add_argument("--t-refrac", type=float, help="refractory seconds (live only)")
    parser.add_argument("--scan-timeout", type=float, default=10.0)
    parser.add_argument("--frequency-hz", type=int, default=10)
    parser.add_argument("--pulse-width-ms", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=250)
    parser.add_argument(
        "--gain-percent",
        type=int,
        default=0,
        help="0 delivers no current; raise it deliberately for a real turn",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    if args.device is None:
        args.device = DEFAULT_INPUT_DEVICE
    elif args.device.isdigit():
        # argparse hands back a string, and sounddevice reads a string as a name
        # to match, so a bare "0" would go looking for a device called "0".
        args.device = int(args.device)

    if not args.dry_run and (args.run_dir is None or args.t_refrac is None):
        parser.error("--run-dir and --t-refrac are required unless --dry-run")

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
