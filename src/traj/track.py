"""Overhead-camera pose tracker for the RoboRoach, implementing PoseTracker.

The chain is: BGR frame -> HSV threshold on the red backpack -> largest contour
-> centroid in pixels -> homography to arena centimetres -> constant-velocity
Kalman filter -> heading from the filtered velocity.

Coordinates follow ``rl_control`` exactly: **x right, y up, theta CCW-positive,
so a left turn is +theta**. That matches ``AntiHabituationEnv`` and
``HabituatingAnimal``, which both integrate ``x += v*cos(h); y += v*sin(h)`` and
apply ``heading_rad += turn`` for ``direction == "left"``.

An image's y grows downward, which is the opposite handedness. That single
inversion lives in :func:`flip_image_y` and nowhere else. It is applied to the
calibration points once when the homography is fitted and to the centroid once
per frame, so the homography itself contains no flip and the arena points in the
calibration file are plain ruler measurements in the y-up arena frame.

Two notes for the call site in ``rl_control``
---------------------------------------------
``Pose`` carries only ``x``, ``y``, ``t`` -- there is no heading field, so the
theta this module computes and cross-checks cannot cross the Protocol boundary.
It is logged on this side only. ``movement_from_poses`` re-derives heading from
consecutive poses downstream, and unlike this module it *holds* the previous
heading while speed is below ``still_speed``.

``Pose.t`` here is **seconds from time.monotonic()**. In ``teach.py`` the same
field is a step index: ``HabituatingAnimal._walk`` does ``self.t += 1.0``, so
``movement_from_poses`` currently sees ``dt == 1.0`` every step. Swapping this
tracker in makes ``dt`` about 0.033, which scales every velocity by roughly 30x.
``AntiHabituationEnv``'s ``still_speed=0.02`` default is calibrated against the
step convention and will never gate against cm/s -- it needs retuning to real
units at the call site. Nothing in this module can fix that from the outside.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np
from interface.camera import Pose

log = logging.getLogger(__name__)

LOG_NAME = "traj_track.jsonl"

# Speckle removal on the threshold mask. Small enough not to erode a backpack
# that is already close to ``min_contour_area``.
MORPH_KERNEL = np.ones((5, 5), np.uint8)

# A homography fitted from four points reproduces those four points exactly,
# so any real residual means the correspondences are wrong -- most often three
# clicks that landed nearly on one line, which cv2.findHomography answers with a
# garbage matrix rather than with None. Refusing to start beats tracking in a
# frame that is quietly skewed.
MAX_RESIDUAL_CM = 2.0

# The filter starts with no velocity information. Initial velocity variance is
# what one second of unmodelled acceleration would produce, which is a wide but
# not absurd prior that collapses within a few frames.
INIT_VELOCITY_HORIZON_S = 1.0

# Preview drawing only.
ARROW_LEN_CM = 8.0
COLOUR_CONTOUR = (0, 255, 0)
COLOUR_RECT = (0, 220, 255)
COLOUR_RAW = (0, 0, 255)
COLOUR_FILTERED = (255, 255, 0)
COLOUR_HEADING = (255, 0, 255)
COLOUR_AXIS = (0, 220, 255)
COLOUR_TEXT = (255, 255, 255)
COLOUR_WARN = (0, 165, 255)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def flip_image_y(points_px, frame_height: int) -> np.ndarray:
    """Convert between the image's downward y and the arena's upward y.

    This is the only place in the package where that handedness change happens.
    It is its own involution, so the same function converts in both directions.
    """
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2).copy()
    pts[:, 1] = (frame_height - 1) - pts[:, 1]
    return pts


def wrap_pi(angle: float) -> float:
    """Fold an angle into (-pi, pi]. Same convention as ``rl_control.env``."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class ArenaHomography:
    """Maps camera pixels to arena centimetres, fitted from a JSON file.

    The JSON holds::

        {
          "frame_size": [width, height],
          "image_points": [[u, v], ...],      # raw pixels, y down, as clicked
          "arena_points_cm": [[x, y], ...],   # arena frame, y UP, ruler units
        }

    At least four correspondences are needed, in matching order. ``frame_size``
    is recorded because a homography fitted at one capture resolution is
    meaningless at another, and because :func:`flip_image_y` needs the height.
    """

    matrix: np.ndarray
    inverse: np.ndarray
    frame_size: tuple[int, int]
    residual_mean_cm: float
    residual_max_cm: float

    @classmethod
    def from_json(cls, path: Path | str) -> ArenaHomography:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Arena calibration {path} is absent. The tracker has no pixel "
                f"fallback -- positions are only ever reported in cm. Create it "
                f"with:  python -m traj.calibrate arena --camera N "
                f"--arena-cm '0,0 100,0 100,60 0,60' --out {path}"
            )
        blob = json.loads(path.read_text())
        try:
            frame_size = tuple(int(v) for v in blob["frame_size"])
            image_points = np.asarray(blob["image_points"], dtype=np.float64)
            arena_points = np.asarray(blob["arena_points_cm"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed arena calibration {path}: {exc}") from exc

        if len(frame_size) != 2:
            raise ValueError(f"{path}: frame_size must be [width, height]")
        if image_points.shape != arena_points.shape:
            raise ValueError(
                f"{path}: {len(image_points)} image points but "
                f"{len(arena_points)} arena points -- they must correspond"
            )
        if image_points.shape[0] < 4 or image_points.shape[1] != 2:
            raise ValueError(f"{path}: need at least four [u, v] correspondences")

        return cls.fit(image_points, arena_points, frame_size)

    @classmethod
    def fit(
        cls,
        image_points: np.ndarray,
        arena_points_cm: np.ndarray,
        frame_size: tuple[int, int],
    ) -> ArenaHomography:
        src = flip_image_y(image_points, frame_size[1])
        dst = np.asarray(arena_points_cm, dtype=np.float64).reshape(-1, 2)

        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise ValueError(
                "cv2.findHomography failed -- the four image points are "
                "probably collinear or coincident. Click the arena corners in "
                "order around its perimeter, not along one edge."
            )
        inverse = np.linalg.inv(matrix)

        projected = cv2.perspectiveTransform(
            src.reshape(-1, 1, 2).astype(np.float64), matrix
        ).reshape(-1, 2)
        residuals = np.linalg.norm(projected - dst, axis=1)

        homography = cls(
            matrix=matrix,
            inverse=inverse,
            frame_size=(int(frame_size[0]), int(frame_size[1])),
            residual_mean_cm=float(residuals.mean()),
            residual_max_cm=float(residuals.max()),
        )
        if homography.residual_max_cm > MAX_RESIDUAL_CM:
            raise ValueError(
                f"Arena homography reproduces its own calibration points to no "
                f"better than {homography.residual_max_cm:.2f} cm (limit "
                f"{MAX_RESIDUAL_CM:.2f} cm). The correspondences are wrong: "
                f"most likely three of the image points are nearly collinear, "
                f"or the clicked points and the arena points are not in the "
                f"same order."
            )
        # Reported at load from whichever entry point loads it, including the
        # RL call site. With exactly four correspondences the fit is exact and
        # this is near zero.
        log.info(
            "arena homography: %d points, reprojection residual mean %.3f cm, "
            "max %.3f cm",
            len(dst),
            homography.residual_mean_cm,
            homography.residual_max_cm,
        )
        return homography

    def to_cm(self, points_px) -> np.ndarray:
        """Pixels (y down) to arena centimetres (y up)."""
        src = flip_image_y(points_px, self.frame_size[1])
        out = cv2.perspectiveTransform(src.reshape(-1, 1, 2), self.matrix)
        return out.reshape(-1, 2)

    def to_px(self, points_cm) -> np.ndarray:
        """Arena centimetres back to pixels. Used only to draw the preview."""
        src = np.asarray(points_cm, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(src, self.inverse).reshape(-1, 2)
        return flip_image_y(out, self.frame_size[1])

    def check_frame_size(self, width: int, height: int) -> None:
        if (width, height) != self.frame_size:
            raise ValueError(
                f"Camera opened at {width}x{height} but the arena calibration "
                f"was fitted at {self.frame_size[0]}x{self.frame_size[1]}. A "
                f"homography does not survive a resolution change -- recalibrate "
                f"at the resolution you intend to run at."
            )


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HsvBounds:
    """Inclusive HSV window in OpenCV ranges: H 0-179, S 0-255, V 0-255.

    Red straddles the H=0 seam, so ``h_lo > h_hi`` is legal and means "wrap":
    the window is ``[h_lo, 179]`` union ``[0, h_hi]``. That is the normal case
    for the red board and :func:`threshold_mask` handles it.
    """

    h_lo: int
    h_hi: int
    s_lo: int
    s_hi: int
    v_lo: int
    v_hi: int

    @property
    def wraps(self) -> bool:
        return self.h_lo > self.h_hi

    @classmethod
    def from_json(cls, path: Path | str) -> HsvBounds:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"HSV calibration {path} is absent. Create it with:  "
                f"python -m traj.calibrate --camera N --out {path}"
            )
        blob = json.loads(path.read_text())
        try:
            return cls(**{k: int(blob[k]) for k in cls.__dataclass_fields__})
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed HSV calibration {path}: {exc}") from exc

    def to_dict(self) -> dict[str, int]:
        return {k: int(getattr(self, k)) for k in self.__dataclass_fields__}


def threshold_mask(frame_bgr: np.ndarray, bounds: HsvBounds) -> np.ndarray:
    """HSV threshold plus an open/close pass to drop speckle."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sv_lo = (bounds.s_lo, bounds.v_lo)
    sv_hi = (bounds.s_hi, bounds.v_hi)

    if bounds.wraps:
        low = cv2.inRange(
            hsv,
            np.array((0, *sv_lo), np.uint8),
            np.array((bounds.h_hi, *sv_hi), np.uint8),
        )
        high = cv2.inRange(
            hsv,
            np.array((bounds.h_lo, *sv_lo), np.uint8),
            np.array((179, *sv_hi), np.uint8),
        )
        mask = cv2.bitwise_or(low, high)
    else:
        mask = cv2.inRange(
            hsv,
            np.array((bounds.h_lo, *sv_lo), np.uint8),
            np.array((bounds.h_hi, *sv_hi), np.uint8),
        )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)


@dataclass(frozen=True, eq=False)
class Detection:
    """Largest red blob in one frame, in pixels."""

    cx_px: float
    cy_px: float
    area: float
    axis_p0_px: tuple[float, float]
    axis_p1_px: tuple[float, float]
    contour: np.ndarray
    box_px: np.ndarray


def detect(
    frame_bgr: np.ndarray, bounds: HsvBounds, min_contour_area: float
) -> tuple[Detection | None, np.ndarray]:
    """Largest contour above ``min_contour_area``, its centroid and its axis.

    Returns ``(detection_or_None, mask)``. The mask is returned so the
    calibrator can display exactly what the tracker would threshold.
    """
    mask = threshold_mask(frame_bgr, bounds)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_contour_area:
        return None, mask

    moments = cv2.moments(contour)
    if moments["m00"] <= 0.0:
        return None, mask
    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])

    # minAreaRect on the same contour, for the principal axis. The axis is taken
    # from the longest edge of the box rather than from ``rect[2]``, whose
    # sign and range have shifted between OpenCV releases.
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float64)
    edges = np.roll(box, -1, axis=0) - box
    longest = int(np.argmax(np.linalg.norm(edges, axis=1)))
    direction = edges[longest]
    half = direction / 2.0
    centre = np.asarray(rect[0], dtype=np.float64)

    return (
        Detection(
            cx_px=cx,
            cy_px=cy,
            area=area,
            axis_p0_px=tuple(centre - half),
            axis_p1_px=tuple(centre + half),
            contour=contour,
            box_px=box,
        ),
        mask,
    )


# --------------------------------------------------------------------------
# filter
# --------------------------------------------------------------------------


class ConstantVelocityFilter:
    """Kalman filter over ``[px, py, vx, vy]`` in centimetres and cm/s.

    ``dt`` is supplied per step and is always the measured wall interval between
    two frames, never a nominal 1/30. A dropped or slow frame therefore widens
    the covariance by the right amount instead of silently under-reporting it.
    """

    def __init__(self, *, sigma_p_cm: float, sigma_a_cm_s2: float) -> None:
        sigma_p_cm = float(sigma_p_cm)
        sigma_a_cm_s2 = float(sigma_a_cm_s2)
        if sigma_p_cm <= 0.0:
            raise ValueError("sigma_p must be positive (centroid noise, cm)")
        if sigma_a_cm_s2 <= 0.0:
            raise ValueError("sigma_a must be positive (process noise, cm/s^2)")

        self.sigma_p = sigma_p_cm
        self.sigma_a = sigma_a_cm_s2
        self.h_matrix = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self.r_matrix = sigma_p_cm**2 * np.eye(2)
        self.state: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

    @property
    def initialised(self) -> bool:
        return self.state is not None

    def initialise(self, measurement_cm: np.ndarray) -> None:
        self.state = np.array(
            [float(measurement_cm[0]), float(measurement_cm[1]), 0.0, 0.0]
        )
        velocity_var = (self.sigma_a * INIT_VELOCITY_HORIZON_S) ** 2
        self.covariance = np.diag(
            [self.sigma_p**2, self.sigma_p**2, velocity_var, velocity_var]
        )

    def predict(self, dt: float) -> None:
        f_matrix = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q_matrix = self.sigma_a**2 * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ]
        )
        self.state = f_matrix @ self.state
        self.covariance = f_matrix @ self.covariance @ f_matrix.T + q_matrix

    def update(self, measurement_cm: np.ndarray) -> None:
        innovation = np.asarray(measurement_cm, float) - self.h_matrix @ self.state
        s_matrix = self.h_matrix @ self.covariance @ self.h_matrix.T + self.r_matrix
        gain = self.covariance @ self.h_matrix.T @ np.linalg.inv(s_matrix)
        self.state = self.state + gain @ innovation
        # Joseph form: keeps the covariance symmetric and positive definite
        # across the long runs the demo does.
        spread = np.eye(4) - gain @ self.h_matrix
        self.covariance = (
            spread @ self.covariance @ spread.T + gain @ self.r_matrix @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)


# --------------------------------------------------------------------------
# tracker
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackerConfig:
    """Everything the tracker needs. Nothing here has a default on purpose.

    Every value is a property of one physical setup -- this camera, this
    lighting, this backpack, this arena. A default would be a guess that looks
    like a measurement, and the failure mode is a tracker that runs and is
    quietly wrong rather than one that refuses to start.
    """

    camera_index: int
    hsv_bounds: HsvBounds
    min_contour_area: float
    sigma_p_cm: float
    sigma_a_cm_s2: float
    v_min_cm_s: float
    arena_calibration: Path
    run_dir: Path


@dataclass(frozen=True, eq=False)
class FrameResult:
    """One frame's worth of tracking. :meth:`record` is the logged subset."""

    t_frame: float
    cx_px: float | None
    cy_px: float | None
    px_cm: float | None
    py_cm: float | None
    px_hat: float | None
    py_hat: float | None
    vx_hat: float | None
    vy_hat: float | None
    speed: float | None
    theta: float | None
    heading_valid: bool
    axis_disagreement_deg: float | None
    contour_area: float | None
    frame: np.ndarray
    detection: Detection | None
    axis_theta: float | None
    dt: float | None

    def record(self) -> dict[str, object]:
        return {
            "t_frame": self.t_frame,
            "cx_px": self.cx_px,
            "cy_px": self.cy_px,
            "px_cm": self.px_cm,
            "py_cm": self.py_cm,
            "px_hat": self.px_hat,
            "py_hat": self.py_hat,
            "vx_hat": self.vx_hat,
            "vy_hat": self.vy_hat,
            "speed": self.speed,
            "theta": self.theta,
            "heading_valid": self.heading_valid,
            "axis_disagreement_deg": self.axis_disagreement_deg,
            "contour_area": self.contour_area,
        }


def open_capture(camera_index: int) -> cv2.VideoCapture:
    """Open a camera, preferring AVFoundation so Continuity Camera enumerates."""
    if platform.system() == "Darwin":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    else:
        capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera index {camera_index}. Run "
            f"'python -m traj.calibrate devices' to see what enumerates, and "
            f"check that the Continuity Camera is awake and that this terminal "
            f"holds macOS camera permission."
        )
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class TrajectoryTracker:
    """Owns the camera, the homography, the filter and the JSONL log.

    Synchronous and single-threaded by design: :meth:`process_once` handles
    exactly one frame. :class:`AsyncPoseTracker` wraps it for the RL loop and
    :func:`run_demo` wraps it for the preview window.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self.homography = ArenaHomography.from_json(config.arena_calibration)
        self._filter = ConstantVelocityFilter(
            sigma_p_cm=config.sigma_p_cm, sigma_a_cm_s2=config.sigma_a_cm_s2
        )
        if config.min_contour_area <= 0.0:
            raise ValueError("min_contour_area must be positive (pixels squared)")
        if config.v_min_cm_s <= 0.0:
            raise ValueError("v_min must be positive (cm/s)")

        self._capture = open_capture(config.camera_index)
        ok, priming_frame = self._capture.read()
        if not ok:
            self._capture.release()
            raise RuntimeError(
                f"Camera index {config.camera_index} opened but delivered no frames."
            )
        height, width = priming_frame.shape[:2]
        self.homography.check_frame_size(width, height)

        self._run_dir = Path(config.run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("a")
        self._t_prev: float | None = None

    @property
    def log_path(self) -> Path:
        return self._run_dir / LOG_NAME

    def process_once(self) -> FrameResult | None:
        """Grab, detect, filter, log. ``None`` once the camera stops."""
        ok, frame = self._capture.read()
        if not ok:
            return None
        t_frame = time.monotonic()

        detection, _ = detect(
            frame, self._config.hsv_bounds, self._config.min_contour_area
        )

        measurement = None
        if detection is not None:
            measurement = self.homography.to_cm([[detection.cx_px, detection.cy_px]])[0]

        # dt is always measured. A stalled frame widens the covariance honestly
        # instead of pretending the roach moved for a nominal 1/30 s.
        dt = None if self._t_prev is None else max(t_frame - self._t_prev, 1e-6)
        self._t_prev = t_frame

        if not self._filter.initialised:
            if measurement is not None:
                self._filter.initialise(measurement)
        else:
            if dt is not None:
                self._filter.predict(dt)
            if measurement is not None:
                self._filter.update(measurement)

        px_hat = py_hat = vx_hat = vy_hat = speed = theta = None
        heading_valid = False
        if self._filter.initialised:
            px_hat, py_hat, vx_hat, vy_hat = (float(v) for v in self._filter.state)
            speed = math.hypot(vx_hat, vy_hat)
            heading_valid = speed > self._config.v_min_cm_s
            # Gated means unavailable, not stale: the previous heading is never
            # carried forward, because a held heading reads downstream as a
            # measurement rather than as an absence of one.
            theta = math.atan2(vy_hat, vx_hat) if heading_valid else None

        axis_theta, disagreement = self._axis_cross_check(detection, theta)

        result = FrameResult(
            t_frame=t_frame,
            cx_px=None if detection is None else detection.cx_px,
            cy_px=None if detection is None else detection.cy_px,
            px_cm=None if measurement is None else float(measurement[0]),
            py_cm=None if measurement is None else float(measurement[1]),
            px_hat=px_hat,
            py_hat=py_hat,
            vx_hat=vx_hat,
            vy_hat=vy_hat,
            speed=speed,
            theta=theta,
            heading_valid=heading_valid,
            axis_disagreement_deg=disagreement,
            contour_area=None if detection is None else detection.area,
            frame=frame,
            detection=detection,
            axis_theta=axis_theta,
            dt=dt,
        )
        self._log.write(json.dumps(result.record()) + "\n")
        self._log.flush()
        return result

    def _axis_cross_check(
        self, detection: Detection | None, theta: float | None
    ) -> tuple[float | None, float | None]:
        """Principal axis in cm, and how far it sits from the velocity heading.

        The rect's axis is a line, not a direction, so it is only defined up to
        180 degrees. The ambiguity is resolved by picking the sense that agrees
        with the filtered velocity -- which means the disagreement this reports
        is always in [0, 90] and is a check on the *body angle*, not a way of
        recovering which end is the head. With no valid heading there is nothing
        to resolve against, so it reports nothing rather than guessing.

        Both endpoints go through the homography, so the comparison happens in
        centimetres. Comparing a pixel-space angle against a cm-space heading
        would fold the camera's perspective into the disagreement.
        """
        if detection is None:
            return None, None
        endpoints = self.homography.to_cm([detection.axis_p0_px, detection.axis_p1_px])
        delta = endpoints[1] - endpoints[0]
        if math.hypot(float(delta[0]), float(delta[1])) <= 0.0:
            return None, None

        axis_theta = math.atan2(float(delta[1]), float(delta[0]))
        if theta is None:
            return axis_theta, None
        if math.cos(axis_theta - theta) < 0.0:
            axis_theta = wrap_pi(axis_theta + math.pi)
        return axis_theta, abs(math.degrees(wrap_pi(axis_theta - theta)))

    def close(self) -> None:
        self._capture.release()
        self._log.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# preview
# --------------------------------------------------------------------------


def _int_point(point) -> tuple[int, int]:
    """Clamp to a range cv2's drawing calls accept.

    A filtered position can sit far outside the frame while the filter is still
    converging, and an unclamped coordinate raises out of the drawing call
    rather than simply landing off-screen.
    """
    x, y = float(point[0]), float(point[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return (0, 0)
    limit = 1 << 20
    return (int(max(-limit, min(limit, x))), int(max(-limit, min(limit, y))))


def render(
    result: FrameResult, homography: ArenaHomography, v_min_cm_s: float
) -> np.ndarray:
    """Annotate one frame: contour, centroid, filtered position, heading."""
    canvas = result.frame.copy()
    detection = result.detection

    if detection is not None:
        cv2.drawContours(canvas, [detection.contour], -1, COLOUR_CONTOUR, 2)
        cv2.drawContours(
            canvas, [detection.box_px.astype(np.int32)], -1, COLOUR_RECT, 1
        )
        cv2.circle(
            canvas, _int_point((detection.cx_px, detection.cy_px)), 4, COLOUR_RAW, -1
        )

    if result.px_hat is not None:
        origin_cm = (result.px_hat, result.py_hat)
        origin_px = _int_point(homography.to_px([origin_cm])[0])
        cv2.circle(canvas, origin_px, 7, COLOUR_FILTERED, 2)

        if result.axis_theta is not None:
            axis_tip_cm = (
                result.px_hat + 0.6 * ARROW_LEN_CM * math.cos(result.axis_theta),
                result.py_hat + 0.6 * ARROW_LEN_CM * math.sin(result.axis_theta),
            )
            cv2.arrowedLine(
                canvas,
                origin_px,
                _int_point(homography.to_px([axis_tip_cm])[0]),
                COLOUR_AXIS,
                1,
                tipLength=0.25,
            )

        if result.heading_valid:
            tip_cm = (
                result.px_hat + ARROW_LEN_CM * math.cos(result.theta),
                result.py_hat + ARROW_LEN_CM * math.sin(result.theta),
            )
            cv2.arrowedLine(
                canvas,
                origin_px,
                _int_point(homography.to_px([tip_cm])[0]),
                COLOUR_HEADING,
                2,
                tipLength=0.3,
            )

    fps = None if not result.dt else 1.0 / result.dt
    lines = [
        (
            "no detection"
            if detection is None
            else f"area {result.contour_area:8.0f} px^2"
        ),
        (
            "filter: waiting for first detection"
            if result.px_hat is None
            else f"pos  ({result.px_hat:7.2f}, {result.py_hat:7.2f}) cm"
        ),
        (
            ""
            if result.speed is None
            else f"speed {result.speed:6.2f} cm/s   (v_min {v_min_cm_s:.2f})"
        ),
        (
            f"theta {math.degrees(result.theta):7.2f} deg"
            if result.heading_valid
            else "theta unavailable (speed <= v_min)"
        ),
        (
            ""
            if result.axis_disagreement_deg is None
            else f"axis disagreement {result.axis_disagreement_deg:5.1f} deg"
        ),
        "" if fps is None else f"{fps:5.1f} fps  (dt {result.dt * 1000:5.1f} ms)",
    ]
    for row, text in enumerate(line for line in lines if line):
        cv2.putText(
            canvas,
            text,
            (12, 26 + 22 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                COLOUR_WARN
                if "unavailable" in text or "no detection" in text
                else COLOUR_TEXT
            ),
            1,
            cv2.LINE_AA,
        )
    return canvas


# --------------------------------------------------------------------------
# the PoseTracker implementation
# --------------------------------------------------------------------------


class AsyncPoseTracker:
    """``rl_control.camera.PoseTracker``: ``async def read(self) -> Pose``.

    The camera is drained continuously by a background task rather than one
    frame per :meth:`read`. An RL step is far slower than a frame, and a
    VideoCapture that is not drained hands back stale buffered frames, so
    on-demand grabbing would report a pose several hundred milliseconds old
    while claiming the timestamp of the moment it was asked.

    :meth:`read` returns the newest pose that has not been returned yet,
    waiting if none has arrived. That guarantees distinct, strictly increasing
    timestamps, which matters because ``movement_from_poses`` divides by
    ``cur.t - prev.t`` -- returning the same pose twice would produce a
    near-zero dt and a velocity spike.

    The first :meth:`read` waits until the roach is detected for the first
    time, with no timeout. If the backpack is out of frame at startup that wait
    is unbounded by design; wrap it in ``asyncio.wait_for`` if the call site
    needs to give up.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self._tracker = TrajectoryTracker(config)
        self._latest: Pose | None = None
        self._fresh = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._failure: BaseException | None = None
        self._returned_t: float | None = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        try:
            while True:
                result = await asyncio.to_thread(self._tracker.process_once)
                if result is None:
                    raise RuntimeError("Camera stopped delivering frames")
                if result.px_hat is not None:
                    self._latest = Pose(result.px_hat, result.py_hat, result.t_frame)
                    self._fresh.set()
        except BaseException as exc:  # surfaced to read(), never swallowed
            self._failure = exc
            self._fresh.set()
            raise

    async def read(self) -> Pose:
        if self._task is None:
            await self.start()
        while True:
            # Cleared before the check, not after: clearing afterwards can drop
            # a set() that lands in between and park read() forever.
            self._fresh.clear()
            if self._failure is not None:
                raise self._failure
            pose = self._latest
            if pose is not None and pose.t != self._returned_t:
                self._returned_t = pose.t
                return pose
            await self._fresh.wait()

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Whatever the pump died of is already in self._failure and has
                # already been raised to any waiting read(). Surfacing it a
                # second time here would mask the caller's own exit path.
                log.debug("tracker pump ended with an error", exc_info=True)
            self._task = None
        self._tracker.close()

    def stop(self) -> None:
        """Synchronous shutdown, for ``rl_control``'s non-async cleanup paths.

        ``interface.plot.run_with_dashboard`` and ``rl_control.live`` both close
        a tracker with a bare ``tracker.stop()`` inside a ``finally``, where
        there is no loop to await on. Cancelling the pump without awaiting it is
        enough here: the task owns no resource of its own, and the capture is
        released on this line rather than by the task.
        """
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._tracker.close()


# --------------------------------------------------------------------------
# demo entry point
# --------------------------------------------------------------------------

WINDOW = "traj.track"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m traj.track",
        description=(
            "Track the RoboRoach from an overhead camera, log JSONL per frame, "
            "and show a live preview. Every parameter is required: each one is "
            "a property of the physical setup, not something with a sane "
            "default."
        ),
    )
    parser.add_argument("--camera", type=int, required=True, help="camera index")
    parser.add_argument(
        "--hsv",
        type=Path,
        required=True,
        help="HSV bounds JSON from 'python -m traj.calibrate'",
    )
    parser.add_argument(
        "--arena",
        type=Path,
        required=True,
        help="arena homography JSON from 'python -m traj.calibrate arena'",
    )
    parser.add_argument(
        "--min-area", type=float, required=True, help="minimum contour area, px^2"
    )
    parser.add_argument(
        "--sigma-p", type=float, required=True, help="centroid noise, cm"
    )
    parser.add_argument(
        "--sigma-a",
        type=float,
        required=True,
        help="acceleration process noise, cm/s^2",
    )
    parser.add_argument(
        "--v-min",
        type=float,
        required=True,
        help="speed below which heading is reported unavailable, cm/s",
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help=f"directory for {LOG_NAME}"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="run headless; still logs every frame",
    )
    return parser


def run_demo(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    config = TrackerConfig(
        camera_index=args.camera,
        hsv_bounds=HsvBounds.from_json(args.hsv),
        min_contour_area=args.min_area,
        sigma_p_cm=args.sigma_p,
        sigma_a_cm_s2=args.sigma_a,
        v_min_cm_s=args.v_min,
        arena_calibration=args.arena,
        run_dir=args.run_dir,
    )

    with TrajectoryTracker(config) as tracker:
        log.info("logging to %s", tracker.log_path)
        if not args.no_preview:
            log.info("preview up -- press q or Esc in the window to stop")
        frames = 0
        gated = 0
        disagreements: list[float] = []
        try:
            while True:
                result = tracker.process_once()
                if result is None:
                    log.info("camera stopped delivering frames")
                    break
                frames += 1
                if result.px_hat is not None and not result.heading_valid:
                    gated += 1
                if result.axis_disagreement_deg is not None:
                    disagreements.append(result.axis_disagreement_deg)

                if not args.no_preview:
                    cv2.imshow(WINDOW, render(result, tracker.homography, args.v_min))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

        log.info("%d frames, %d with heading gated below v_min", frames, gated)
        if disagreements:
            axis = np.asarray(disagreements)
            log.info(
                "axis vs velocity disagreement: median %.1f deg, 90th pct %.1f deg",
                float(np.median(axis)),
                float(np.percentile(axis, 90)),
            )
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
