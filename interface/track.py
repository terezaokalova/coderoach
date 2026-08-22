"""Livestream an iPhone or webcam and lock onto the roach every frame.

PhonePoseTracker.read() returns the latest centroid as a Pose so the teaching
loop can swap SimulatedCamera for a real camera. x and y are image-normalized
(0 to 1, origin at the top left).
"""

from __future__ import annotations

import argparse
import asyncio
import math
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal, assert_never

import cv2
import numpy as np

from .camera import Pose

TrackerName = Literal["blob", "csrt"]
Box = tuple[int, int, int, int]
LIVE_JPEG = "/tmp/roach_cam/live.jpg"
HELPER_APP = Path("/tmp/RoachCam.app")
MACOS_DIR = Path(__file__).resolve().parent / "macos"


def parse_source(text: str) -> int | str:
    return int(text) if text.isdigit() else text


class JpegPoll:
    """Re-read a jpeg that a camera helper overwrites, such as /tmp/roach_cam/live.jpg."""

    def __init__(self, path: str) -> None:
        self.path = path

    def isOpened(self) -> bool:
        return True

    def set(self, *_: object) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = cv2.imread(self.path)
        return frame is not None, frame

    def release(self) -> None:
        return None


def open_capture(source: int | str) -> cv2.VideoCapture | JpegPoll:
    if isinstance(source, str) and source.lower().endswith((".jpg", ".jpeg")):
        cap: cv2.VideoCapture | JpegPoll = JpegPoll(source)
    elif isinstance(source, int):
        backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
        cap = cv2.VideoCapture(source, backend)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened() and not isinstance(cap, JpegPoll):
        raise RuntimeError(
            f"Could not open camera {source!r}. Unlock the iPhone, keep it "
            "near the Mac, allow Camera access for Terminal, and try "
            "--source 0 or --source 1. Or paste an HTTP/RTSP URL from a "
            "phone webcam app."
        )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _jpeg_is_live(path: str, max_age_s: float = 2.5) -> bool:
    try:
        return time.time() - Path(path).stat().st_mtime < max_age_s
    except OSError:
        return False


def _build_helper(app: Path) -> None:
    binary = app / "Contents" / "MacOS" / "RoachCam"
    binary.parent.mkdir(parents=True, exist_ok=True)
    (app / "Contents" / "Resources").mkdir(parents=True, exist_ok=True)
    shutil.copy(MACOS_DIR / "Info.plist", app / "Contents" / "Info.plist")
    sdk = subprocess.check_output(
        ["xcrun", "--show-sdk-path", "--sdk", "macosx"], text=True
    ).strip()
    machine = subprocess.check_output(["uname", "-m"], text=True).strip()
    subprocess.check_call(
        [
            "xcrun",
            "swiftc",
            "-sdk",
            sdk,
            "-target",
            f"{machine}-apple-macos14.0",
            "-framework",
            "AppKit",
            "-framework",
            "AVFoundation",
            "-framework",
            "CoreImage",
            "-framework",
            "CoreMedia",
            "-framework",
            "ImageIO",
            "-framework",
            "Network",
            "-framework",
            "UniformTypeIdentifiers",
            "-o",
            str(binary),
            str(MACOS_DIR / "RoachCam.swift"),
        ]
    )
    subprocess.check_call(["codesign", "--force", "--deep", "--sign", "-", str(app)])


def ensure_phone_stream() -> str:
    """Start the Continuity Camera helper if live.jpg is not being written."""
    if _jpeg_is_live(LIVE_JPEG):
        return LIVE_JPEG
    if not (HELPER_APP / "Contents" / "MacOS" / "RoachCam").exists():
        print("Building the iPhone camera helper...")
        _build_helper(HELPER_APP)
    binary = HELPER_APP / "Contents" / "MacOS" / "RoachCam"
    subprocess.Popen(
        [str(binary)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("Waiting for iPhone frames. Allow Camera for RoachCam if macOS asks.")
    deadline = time.time() + 20
    while time.time() < deadline:
        if _jpeg_is_live(LIVE_JPEG):
            return LIVE_JPEG
        time.sleep(0.2)
    raise RuntimeError(
        "No iPhone frames yet. Unlock the phone, keep the cable in, and "
        "allow Camera access for RoachCam in Privacy settings."
    )


def list_cameras(limit: int = 5) -> None:
    backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    found = False
    for index in range(limit):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                print(f"{index}  {width}x{height}")
                found = True
            else:
                print(f"{index}  opened but produced no frame")
                found = True
        cap.release()
    if not found:
        print("No local cameras opened. Check Camera permission for Terminal.")


def _csrt_create():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("This OpenCV build has no CSRT tracker. Use --tracker blob.")


class BlobLock:
    """Follow the clicked color near the last box. Built for the red backpack."""

    def __init__(self, frame: np.ndarray, box: Box) -> None:
        self.box = box
        self.hsv = _median_hsv(frame, box)

    def update(self, frame: np.ndarray) -> Box | None:
        x, y, width, height = self.box
        pad = max(2 * max(width, height), 80)
        rows, cols = frame.shape[:2]
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(cols, x + width + pad)
        y1 = min(rows, y + height + pad)
        roi = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int16)
        hue = np.abs(hsv[:, :, 0] - int(self.hsv[0]))
        hue = np.minimum(hue, 180 - hue)
        sat = np.abs(hsv[:, :, 1] - int(self.hsv[1]))
        val = np.abs(hsv[:, :, 2] - int(self.hsv[2]))
        mask = ((hue < 18) & (sat < 80) & (val < 90)).astype(np.uint8) * 255
        if int(mask.sum()) < 12 * 255:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        picked = _nearest_contour(contours, x0, y0, x + width / 2, y + height / 2)
        if picked is None:
            return None
        self.box = picked
        return self.box


def _median_hsv(frame: np.ndarray, box: Box) -> np.ndarray:
    x, y, width, height = box
    roi = frame[y : y + height, x : x + width]
    if roi.size == 0:
        return np.zeros(3, np.float32)
    return np.median(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0)


def _nearest_contour(
    contours,
    x0: int,
    y0: int,
    last_x: float,
    last_y: float,
) -> Box | None:
    best: Box | None = None
    best_score = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 12:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        cx = x0 + bx + bw / 2
        cy = y0 + by + bh / 2
        score = math.hypot(cx - last_x, cy - last_y) - 0.12 * math.sqrt(area)
        if best_score is None or score < best_score:
            best = (x0 + bx, y0 + by, bw, bh)
            best_score = score
    return best


class CsrtLock:
    def __init__(self, frame: np.ndarray, box: Box) -> None:
        self.tracker = _csrt_create()
        self.tracker.init(frame, box)

    def update(self, frame: np.ndarray) -> Box | None:
        ok, raw = self.tracker.update(frame)
        if not ok:
            return None
        x, y, width, height = (int(v) for v in raw)
        if width < 2 or height < 2:
            return None
        return (x, y, width, height)


def start_lock(name: TrackerName, frame: np.ndarray, box: Box) -> BlobLock | CsrtLock:
    match name:
        case "blob":
            return BlobLock(frame, box)
        case "csrt":
            return CsrtLock(frame, box)
        case _:
            assert_never(name)


def box_from_drag(
    start: tuple[int, int], end: tuple[int, int], frame_size: tuple[int, int]
) -> Box:
    x0, y0 = start
    x1, y1 = end
    x, y = min(x0, x1), min(y0, y1)
    width, height = abs(x1 - x0), abs(y1 - y0)
    cols, rows = frame_size
    if width < 8 and height < 8:
        width = height = max(48, min(cols, rows) // 14)
        x -= width // 2
        y -= height // 2
    x = max(0, min(x, cols - 2))
    y = max(0, min(y, rows - 2))
    width = max(8, min(width, cols - x))
    height = max(8, min(height, rows - y))
    return (x, y, width, height)


def pose_from_box(box: Box, frame_size: tuple[int, int], t: float) -> Pose:
    x, y, width, height = box
    cols, rows = frame_size
    return Pose((x + width / 2) / cols, (y + height / 2) / rows, t)


class PhonePoseTracker:
    """Grab frames in a thread, click the roach, then read() the latest pose."""

    def __init__(
        self,
        source: int | str = LIVE_JPEG,
        tracker: TrackerName = "blob",
    ) -> None:
        self.source = source
        self.tracker_name = tracker
        self._guard = threading.Lock()
        self._pose: Pose | None = None
        self._view: np.ndarray | None = None
        self._pending: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._lock: BlobLock | CsrtLock | None = None
        self._trail: deque[tuple[int, int]] = deque(maxlen=80)
        self._t0 = time.monotonic()
        self._running = False
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | JpegPoll | None = None
        self.status = "click the roach"

    def latest(self) -> Pose | None:
        with self._guard:
            return self._pose

    def view_rgb(self) -> np.ndarray | None:
        with self._guard:
            frame = None if self._view is None else self._view.copy()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def set_box(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        with self._guard:
            self._pending = (
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
            )

    def start(self) -> None:
        if self._running:
            return
        self._cap = open_capture(self.source)
        self._t0 = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.5)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    async def read(self) -> Pose:
        while True:
            pose = self.latest()
            if pose is not None:
                return pose
            if not self._running:
                raise RuntimeError("Camera stopped before the roach was locked.")
            await asyncio.sleep(0.05)

    def _loop(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue
            view = self._track(frame)
            with self._guard:
                self._view = view
            time.sleep(0.01)

    def _track(self, frame: np.ndarray) -> np.ndarray:
        rows, cols = frame.shape[:2]
        with self._guard:
            pending = self._pending
            self._pending = None
        if pending is not None:
            self._lock = start_lock(
                self.tracker_name,
                frame,
                box_from_drag(*pending, (cols, rows)),
            )
            self._trail.clear()
            with self._guard:
                self._pose = None
        view = frame.copy()
        if self._lock is None:
            self.status = "click the roach"
            cv2.putText(
                view,
                "CLICK THE ROACH",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 180, 255),
                2,
            )
            return view
        box = self._lock.update(frame)
        if box is None:
            self.status = "lost"
            cv2.putText(
                view,
                "LOST  click the roach",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return view
        pose = pose_from_box(box, (cols, rows), time.monotonic() - self._t0)
        with self._guard:
            self._pose = pose
        self.status = "locked"
        x, y, width, height = box
        cx, cy = x + width // 2, y + height // 2
        self._trail.append((cx, cy))
        cv2.rectangle(view, (x, y), (x + width, y + height), (0, 220, 0), 2)
        cv2.circle(view, (cx, cy), 4, (0, 220, 0), -1)
        cv2.putText(
            view,
            f"x={pose.x:.3f}  y={pose.y:.3f}  t={pose.t:.1f}s",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 0),
            2,
        )
        if len(self._trail) >= 2:
            cv2.polylines(
                view,
                [np.array(self._trail, dtype=np.int32)],
                False,
                (0, 180, 255),
                2,
            )
        return view

    def stream(self) -> None:
        self.start()
        window = "roach track"
        print(
            "Click and drag a box on the roach. Click alone uses a small box. "
            "r relocks, q quits."
        )
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        drag: tuple[int, int] | None = None
        hover: tuple[int, int] | None = None

        def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
            nonlocal drag, hover
            if event == cv2.EVENT_LBUTTONDOWN:
                drag = (x, y)
                hover = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drag is not None:
                hover = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and drag is not None:
                self.set_box(drag, (x, y))
                drag = None
                hover = None

        cv2.setMouseCallback(window, on_mouse)
        try:
            while self._running:
                with self._guard:
                    view = None if self._view is None else self._view.copy()
                if view is None:
                    if cv2.waitKey(20) in (ord("q"), 27):
                        return
                    continue
                if drag is not None and hover is not None:
                    cv2.rectangle(view, drag, hover, (255, 180, 0), 1)
                cv2.imshow(window, view)
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return
                if key == ord("r"):
                    self._lock = None
                    with self._guard:
                        self._pose = None
                    self._trail.clear()
        finally:
            self.stop()
            cv2.destroyAllWindows()


def open_camera(args: argparse.Namespace) -> PhonePoseTracker | None:
    if getattr(args, "sim_pose", False):
        return None
    source = args.source
    if source in {"phone", "iphone"}:
        source = ensure_phone_stream()
    else:
        source = parse_source(source)
    tracker = PhonePoseTracker(source, getattr(args, "tracker", "blob"))
    tracker.start()
    return tracker


def run_track(args: argparse.Namespace) -> None:
    if args.list:
        list_cameras()
        return
    source = args.source
    if source in {"phone", "iphone"}:
        source = ensure_phone_stream()
    else:
        source = parse_source(source)
    PhonePoseTracker(source, args.tracker).stream()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="phone")
    parser.add_argument("--tracker", choices=("blob", "csrt"), default="blob")
    parser.add_argument("--list", action="store_true")
    run_track(parser.parse_args())


if __name__ == "__main__":
    main()
