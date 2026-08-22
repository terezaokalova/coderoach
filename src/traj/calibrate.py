"""Interactive calibration for :mod:`traj.track`. Produces its two JSON files.

HSV bounds cannot be picked blind -- they depend on the backpack's exact red,
the arena lighting and the camera's white balance -- so this opens a live window
with trackbars and shows the mask the tracker would actually threshold.

    python -m traj.calibrate --camera 1 --out cal/hsv.json

The arena homography needs four image points and their real positions in
centimetres. ``track`` refuses to start without it and has no pixel fallback,
so producing it is part of getting the demo up:

    python -m traj.calibrate arena --camera 1 \\
        --arena-cm '0,0 100,0 100,60 0,60' --out cal/arena.json

Click the four arena points in the same order as ``--arena-cm``.

    python -m traj.calibrate devices

lists what enumerates, which is how to find the Continuity Camera's index --
it is not reliably 0, and the built-in FaceTime camera usually holds a lower
index than a phone that has just joined.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from traj.track import HsvBounds, detect, open_capture

HSV_WINDOW = "traj.calibrate  --  hsv"
MASK_WINDOW = "traj.calibrate  --  mask"
ARENA_WINDOW = "traj.calibrate  --  arena"

# (trackbar name, HsvBounds field, maximum, sensible starting point for red)
TRACKBARS = (
    ("H lo", "h_lo", 179, 170),
    ("H hi", "h_hi", 179, 10),
    ("S lo", "s_lo", 255, 120),
    ("S hi", "s_hi", 255, 255),
    ("V lo", "v_lo", 255, 70),
    ("V hi", "v_hi", 255, 255),
)
MIN_AREA_TRACKBAR = "min area /10"
MIN_AREA_START = 30

TEXT = (255, 255, 255)
GOOD = (0, 255, 0)
WARN = (0, 165, 255)


def _noop(_value: int) -> None:
    """Trackbar callback. State is polled in the loop instead."""


def list_devices(highest: int = 6) -> int:
    """Probe camera indices and report which open and at what resolution."""
    print(f"probing camera indices 0..{highest - 1}")
    found = 0
    for index in range(highest):
        try:
            capture = open_capture(index)
        except RuntimeError:
            print(f"  [{index}] not available")
            continue
        ok, frame = capture.read()
        if ok:
            height, width = frame.shape[:2]
            print(f"  [{index}] OPEN   {width}x{height}")
            found += 1
        else:
            print(f"  [{index}] opens but delivers no frames")
        capture.release()
    if not found:
        print(
            "\nNothing opened. Check System Settings > Privacy & Security > "
            "Camera for this terminal, and make sure the Continuity Camera is "
            "unlocked and near the Mac."
        )
        return 1
    return 0


def calibrate_hsv(camera_index: int, out_path: Path) -> int:
    """Live HSV trackbars over the real camera. 's' writes, 'q'/Esc quits."""
    capture = open_capture(camera_index)
    cv2.namedWindow(HSV_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)

    # Resuming from a saved file has to survive a legitimately saved 0, so the
    # lookup falls back on a missing key rather than on a falsy value.
    start: dict[str, int] = {}
    if out_path.exists():
        start = HsvBounds.from_json(out_path).to_dict()
        print(f"starting from the bounds already in {out_path}: {start}")

    for name, field, maximum, default in TRACKBARS:
        cv2.createTrackbar(name, HSV_WINDOW, maximum, maximum, _noop)
        cv2.setTrackbarPos(name, HSV_WINDOW, start.get(field, default))
    cv2.createTrackbar(MIN_AREA_TRACKBAR, HSV_WINDOW, MIN_AREA_START, 500, _noop)

    print(
        "\nHSV calibration is up.\n"
        "  Drag the trackbars until the mask shows the backpack and nothing "
        "else.\n"
        "  H lo > H hi is fine and is the normal case for red -- the window "
        "wraps\n"
        "  around H=0 and the tracker handles it the same way.\n"
        "  s = save   r = reset to the red starting point   q / Esc = quit\n"
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("camera stopped delivering frames")
                return 1

            bounds = HsvBounds(
                **{
                    field: cv2.getTrackbarPos(name, HSV_WINDOW)
                    for name, field, _, _ in TRACKBARS
                }
            )
            min_area = 10.0 * max(1, cv2.getTrackbarPos(MIN_AREA_TRACKBAR, HSV_WINDOW))
            detection, mask = detect(frame, bounds, min_area)

            preview = frame.copy()
            if detection is None:
                _label(preview, "no contour above min area", WARN, 0)
            else:
                cv2.drawContours(preview, [detection.contour], -1, GOOD, 2)
                cv2.circle(
                    preview,
                    (int(detection.cx_px), int(detection.cy_px)),
                    4,
                    (0, 0, 255),
                    -1,
                )
                _label(preview, f"area {detection.area:.0f} px^2", GOOD, 0)
            _label(
                preview,
                f"min area {min_area:.0f} px^2   "
                f"{'WRAPPED hue' if bounds.wraps else 'plain hue'}",
                TEXT,
                1,
            )
            _label(preview, "s = save    r = reset    q = quit", TEXT, 2)

            cv2.imshow(HSV_WINDOW, preview)
            cv2.imshow(MASK_WINDOW, mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("quit without saving")
                return 0
            if key == ord("r"):
                for name, _, _, default in TRACKBARS:
                    cv2.setTrackbarPos(name, HSV_WINDOW, default)
            if key == ord("s"):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                payload = bounds.to_dict()
                out_path.write_text(json.dumps(payload, indent=2) + "\n")
                print(f"wrote {out_path}: {payload}")
                print(f"min contour area that looked right here: {min_area:.0f}")
                return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _label(image: np.ndarray, text: str, colour, row: int) -> None:
    cv2.putText(
        image,
        text,
        (12, 26 + 24 * row),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        colour,
        1,
        cv2.LINE_AA,
    )


def parse_arena_cm(raw: str) -> np.ndarray:
    """``'0,0 100,0 100,60 0,60'`` to an Nx2 array, y measured UP."""
    points = []
    for token in raw.split():
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"'{token}' is not an x,y pair")
        points.append([float(parts[0]), float(parts[1])])
    if len(points) < 4:
        raise ValueError("need at least four arena points")
    return np.asarray(points, dtype=np.float64)


def calibrate_arena(camera_index: int, arena_cm: np.ndarray, out_path: Path) -> int:
    """Click the arena points that match ``arena_cm``, in the same order."""
    from traj.track import ArenaHomography

    capture = open_capture(camera_index)
    clicked: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < len(arena_cm):
            clicked.append((float(x), float(y)))
            print(
                f"  point {len(clicked)} -> pixel ({x}, {y}) "
                f"= arena {tuple(arena_cm[len(clicked) - 1])} cm"
            )

    cv2.namedWindow(ARENA_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(ARENA_WINDOW, on_mouse)
    print(
        f"\nArena calibration is up. Click {len(arena_cm)} points in this "
        f"order:\n  "
        + "\n  ".join(
            f"{i + 1}. ({p[0]:g}, {p[1]:g}) cm" for i, p in enumerate(arena_cm)
        )
        + "\n  u = undo the last click   s = save (once all are placed)   "
        "q / Esc = quit\n"
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("camera stopped delivering frames")
                return 1
            height, width = frame.shape[:2]

            preview = frame.copy()
            for index, (x, y) in enumerate(clicked):
                cv2.circle(preview, (int(x), int(y)), 6, GOOD, 2)
                cv2.putText(
                    preview,
                    f"{index + 1}",
                    (int(x) + 9, int(y) - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    GOOD,
                    2,
                    cv2.LINE_AA,
                )
            if len(clicked) > 1:
                cv2.polylines(
                    preview,
                    [np.asarray(clicked, np.int32)],
                    len(clicked) == len(arena_cm),
                    GOOD,
                    1,
                )

            remaining = len(arena_cm) - len(clicked)
            if remaining:
                nxt = arena_cm[len(clicked)]
                _label(
                    preview,
                    f"click point {len(clicked) + 1}: ({nxt[0]:g}, {nxt[1]:g}) cm",
                    WARN,
                    0,
                )
            else:
                _label(preview, "all points placed -- s to save", GOOD, 0)
            _label(preview, f"{width}x{height}   u = undo   q = quit", TEXT, 1)

            cv2.imshow(ARENA_WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("quit without saving")
                return 0
            if key == ord("u") and clicked:
                clicked.pop()
            if key == ord("s"):
                if remaining:
                    print(f"{remaining} point(s) still unplaced")
                    continue
                payload = {
                    "frame_size": [width, height],
                    "image_points": [list(p) for p in clicked],
                    "arena_points_cm": arena_cm.tolist(),
                }
                # Fit before writing: a degenerate click set should fail here,
                # not later inside the tracker.
                homography = ArenaHomography.fit(
                    np.asarray(clicked, np.float64), arena_cm, (width, height)
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2) + "\n")
                print(
                    f"wrote {out_path}\n"
                    f"  reprojection residual: mean "
                    f"{homography.residual_mean_cm:.4f} cm, max "
                    f"{homography.residual_max_cm:.4f} cm"
                )
                return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m traj.calibrate",
        description="Produce the JSON calibration files that traj.track needs.",
    )
    sub = parser.add_subparsers(dest="command")

    hsv = sub.add_parser("hsv", help="live HSV trackbars (default)")
    hsv.add_argument("--camera", type=int, required=True)
    hsv.add_argument("--out", type=Path, default=Path("cal/hsv.json"))

    arena = sub.add_parser("arena", help="click four points, fit the homography")
    arena.add_argument("--camera", type=int, required=True)
    arena.add_argument(
        "--arena-cm",
        required=True,
        help="matching arena points, y UP, e.g. '0,0 100,0 100,60 0,60'",
    )
    arena.add_argument("--out", type=Path, default=Path("cal/arena.json"))

    sub.add_parser("devices", help="probe which camera indices open")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Bare 'python -m traj.calibrate --camera 1' means HSV, which is the one
    # that has to be run first and the one run most often. -h is exempt, so
    # that bare --help still lists the subcommands rather than only hsv's.
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv.insert(0, "hsv")
    args = build_parser().parse_args(argv)

    if args.command == "devices":
        return list_devices()
    if args.command == "arena":
        return calibrate_arena(args.camera, parse_arena_cm(args.arena_cm), args.out)
    return calibrate_hsv(args.camera, args.out)


if __name__ == "__main__":
    sys.exit(main())
