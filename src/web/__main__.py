"""Serve the canvas on the LAN so an iPad can reach it.

Binds 0.0.0.0 rather than localhost, and prints the address the iPad should
actually open. A server on 127.0.0.1 is reachable only from the machine
running it, which is the one place the Pencil is not.

That bind puts an unauthenticated control surface for a live animal on
whatever network this machine is on. ``interface.guard_turn`` still enforces
the hardware envelope underneath, and the gate still enforces the refractory
period, but neither of them asks who is calling. On a shared network, pass
``--host 127.0.0.1`` and reach the rig some other way.
"""

from __future__ import annotations

import argparse
import io
import logging
import socket
import sys
from dataclasses import fields
from pathlib import Path

import uvicorn

from web.app import WebConfig, create_app

log = logging.getLogger(__name__)

# Optional tracker parameters. traj.track refuses to default any of these,
# because a default is a guess that looks like a measurement. The web server
# has to come up on one command line, so it defaults them and then prints
# which values were measured and which were guessed.
DEFAULTED = {
    "min_area": "min_contour_area",
    "sigma_p": "sigma_p_cm",
    "sigma_a": "sigma_a_cm_s2",
    "v_min": "v_min_cm_s",
}

# Addresses that only the machine running the server can reach. There is no
# point drawing a QR code for one: the iPad would scan it and time out.
LOOPBACK = frozenset(("127.0.0.1", "localhost", "::1"))
ANY_INTERFACE = frozenset(("0.0.0.0", "::"))

# The full four-module quiet zone the QR specification asks for. It costs four
# lines of terminal and it is what makes a camera lock on straight away.
QR_BORDER = 4

# U+2588 FULL BLOCK, which is what print_ascii draws a module with.
FULL_BLOCK = "\u2588"


def lan_address() -> str | None:
    """This machine's address on the network the iPad is also on.

    Found by asking the routing table which local address would be used to
    reach the outside world. The socket is UDP and never sends a packet, so
    this works with no internet connection and contacts nothing.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


def tablet_address(host: str) -> str | None:
    """The address to hand the iPad, or None if nothing off-machine can reach.

    Binding one specific interface is still reachable from the network -- it is
    only loopback that is not -- so a host that is neither loopback nor a
    wildcard is returned unchanged rather than looked up.
    """
    if host in LOOPBACK:
        return None
    if host in ANY_INTERFACE:
        return lan_address()
    return host


def qr_ascii(url: str, *, invert: bool = True) -> str | None:
    """``url`` as a block-character QR code, or None if qrcode is not installed.

    ``invert`` decides which way round the modules are drawn, and it is not
    cosmetic. With it set, the *light* modules are block characters, which is
    what a dark terminal needs. A light-background terminal wants the opposite.
    Getting it backwards prints a photographic negative, and plenty of phone
    cameras will not read one.
    """
    try:
        import qrcode
    except ImportError:
        return None

    code = qrcode.QRCode(border=QR_BORDER)
    code.add_data(url)
    code.make(fit=True)
    buffer = io.StringIO()
    code.print_ascii(out=buffer, invert=invert)

    # Indented with the light module rather than with spaces, so the banner
    # stays aligned without eating into the quiet zone on the left. Padding
    # with spaces would read as four more dark modules there.
    pad = (FULL_BLOCK if invert else " ") * 2
    return "\n".join(pad + line for line in buffer.getvalue().rstrip("\n").split("\n"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m web",
        description=(
            "Serve the drawing canvas over the LAN. Draw a path on an iPad and "
            "the control loop follows it. Starts in replay mode, rather than "
            "refusing to start, when the arena calibration is missing."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="this run's directory; its parent is what /api/runs lists",
    )
    parser.add_argument(
        "--t-refrac",
        type=float,
        help="refractory seconds, required unless --no-roach",
    )

    parser.add_argument("--camera", type=int, help="camera index; omit for replay only")
    parser.add_argument("--hsv", type=Path, help="HSV bounds JSON from traj.calibrate")
    parser.add_argument("--arena", type=Path, help="arena homography JSON")
    parser.add_argument("--min-area", type=float, help="minimum contour area, px^2")
    parser.add_argument("--sigma-p", type=float, help="centroid noise, cm")
    parser.add_argument("--sigma-a", type=float, help="process noise, cm/s^2")
    parser.add_argument("--v-min", type=float, help="heading gate, cm/s")

    parser.add_argument(
        "--spacing", type=float, default=2.0, help="waypoint spacing cm"
    )
    parser.add_argument("--lookahead", type=float, default=6.0, help="initial L_d, cm")
    parser.add_argument(
        "--alpha-dead", type=float, default=0.26, help="initial deadband, rad"
    )
    parser.add_argument("--video-width", type=int, default=720)
    parser.add_argument("--video-fps", type=float, default=15.0)

    parser.add_argument("--frequency-hz", type=int, default=10)
    parser.add_argument("--pulse-width-ms", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=250)
    parser.add_argument(
        "--gain-percent",
        type=int,
        default=0,
        help="0 delivers no current; raise it deliberately for a real turn",
    )
    parser.add_argument("--scan-timeout", type=float, default=10.0)

    parser.add_argument("--voice-seconds", type=float, default=3.0)
    parser.add_argument("--voice-model", default="base.en")
    parser.add_argument(
        "--voice-device", default=None, help="input device index or name"
    )

    parser.add_argument(
        "--no-qr",
        action="store_true",
        help="skip the QR code in the startup banner",
    )
    parser.add_argument(
        "--qr-light",
        action="store_true",
        help="draw the QR for a light-background terminal instead of a dark one",
    )

    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-roach", action="store_true")
    parser.add_argument("--no-voice", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> tuple[WebConfig, dict[str, str]]:
    """Build the config, and report where each defaulted value came from."""
    defaults = {field.name: field.default for field in fields(WebConfig)}
    resolved: dict[str, object] = {}
    provenance: dict[str, str] = {}

    for cli_name, config_name in DEFAULTED.items():
        given = getattr(args, cli_name)
        if given is None:
            resolved[config_name] = defaults[config_name]
            provenance[config_name] = "default"
        else:
            resolved[config_name] = given
            provenance[config_name] = "given"

    device = args.voice_device
    if isinstance(device, str) and device.isdigit():
        # argparse hands back a string and sounddevice reads a string as a
        # name to match, so a bare "0" would look for a device called "0".
        device = int(device)

    config = WebConfig(
        run_dir=args.run_dir,
        t_refrac_s=args.t_refrac if args.t_refrac is not None else 0.0,
        camera_index=None if args.no_camera else args.camera,
        hsv=args.hsv,
        arena=args.arena,
        spacing_cm=args.spacing,
        lookahead_cm=args.lookahead,
        alpha_dead_rad=args.alpha_dead,
        video_width=args.video_width,
        video_fps=args.video_fps,
        frequency_hz=args.frequency_hz,
        pulse_width_ms=args.pulse_width_ms,
        duration_ms=args.duration_ms,
        gain_percent=args.gain_percent,
        scan_timeout=args.scan_timeout,
        voice_seconds=args.voice_seconds,
        voice_model=args.voice_model,
        voice_device=device,
        with_roach=not args.no_roach,
        with_voice=not args.no_voice,
        **resolved,
    )
    return config, provenance


def describe(
    config: WebConfig,
    provenance: dict[str, str],
    host: str,
    port: int,
    *,
    show_qr: bool = True,
    qr_light: bool = False,
) -> str:
    lines = ["", "CodeRoach canvas", f"  run dir      {config.run_dir}"]

    if config.camera_index is None:
        lines.append("  camera       none -- replay only")
    else:
        missing = [
            name
            for name, path in (("--arena", config.arena), ("--hsv", config.hsv))
            if path is None or not Path(path).exists()
        ]
        if missing:
            verb = "is" if len(missing) == 1 else "are"
            lines.append(
                f"  camera       {config.camera_index}, but "
                f"{', '.join(missing)} {verb} missing -- replay only"
            )
        else:
            lines.append(f"  camera       {config.camera_index}")
        for name in ("min_contour_area", "sigma_p_cm", "sigma_a_cm_s2", "v_min_cm_s"):
            mark = (
                "" if provenance.get(name) == "given" else "  (default, not measured)"
            )
            lines.append(f"  {name:<16} {getattr(config, name)}{mark}")

    if config.with_roach:
        lines.append(f"  T_refrac     {config.t_refrac_s} s")
        gain = f"  gain         {config.gain_percent}%"
        if config.gain_percent == 0:
            gain += "   (no current delivered; raise it deliberately)"
        lines.append(gain)
    else:
        lines.append("  roach        disabled")

    lines.append("")
    lines.append(f"  local        http://127.0.0.1:{port}")

    if host in LOOPBACK:
        lines.append(f"  bound to     {host} only -- no iPad can reach this")
        lines.append("")
        return "\n".join(lines)

    address = tablet_address(host)
    if address is None:
        lines.append("  lan          no route found -- is wifi on?")
        lines.append("")
        return "\n".join(lines)

    url = f"http://{address}:{port}"
    lines.append(f"  lan          {url}    <- open on the iPad")

    if show_qr:
        code = qr_ascii(url, invert=not qr_light)
        lines.append("")
        if code is None:
            lines.append("  (install qrcode to get a scannable code here)")
        else:
            lines.append("  point the iPad camera at this:")
            lines.append("")
            lines.append(code)

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.no_roach and args.t_refrac is None:
        parser.error("--t-refrac is required unless --no-roach")
    if args.camera is not None and args.no_camera:
        parser.error("--camera and --no-camera contradict each other")

    config, provenance = config_from_args(args)
    # Flushed explicitly: uvicorn logs to stderr, which is unbuffered, so a
    # redirected stdout would print the address to open after the server's own
    # startup lines. The address is the reason this banner exists.
    print(
        describe(
            config,
            provenance,
            args.host,
            args.port,
            show_qr=not args.no_qr,
            qr_light=args.qr_light,
        ),
        flush=True,
    )

    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
