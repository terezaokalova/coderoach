"""The video encode path, which runs on every frame of every live session.

Nothing here needs a camera: a synthetic calibration and a black frame are
enough to drive traj.track.render, the path overlay, and the JPEG encoder. That
is the whole of what the browser receives, and it is the part most likely to
raise on the first live frame rather than in a review.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from traj.track import ArenaHomography, FrameResult
from web.hub import FrameHub

FRAME_SIZE = (320, 240)
ARENA_CM = [[0, 0], [100, 0], [100, 60], [0, 60]]
IMAGE_PX = [[20, 220], [300, 220], [300, 20], [20, 20]]


@pytest.fixture
def homography(tmp_path):
    path = tmp_path / "arena.json"
    path.write_text(
        json.dumps(
            {
                "frame_size": list(FRAME_SIZE),
                "image_points": IMAGE_PX,
                "arena_points_cm": ARENA_CM,
            }
        )
    )
    return ArenaHomography.from_json(path)


def result_at(x_cm: float, y_cm: float) -> FrameResult:
    return FrameResult(
        t_frame=1.0,
        cx_px=None,
        cy_px=None,
        px_cm=x_cm,
        py_cm=y_cm,
        px_hat=x_cm,
        py_hat=y_cm,
        vx_hat=3.0,
        vy_hat=0.0,
        speed=3.0,
        theta=0.0,
        heading_valid=True,
        axis_disagreement_deg=None,
        contour_area=None,
        frame=np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8),
        detection=None,
        axis_theta=None,
        dt=1 / 30,
    )


class StillTracker:
    def __init__(self, homography):
        self.homography = homography

    def process_once(self):  # pragma: no cover - the hub is not started here
        return None

    def close(self):
        pass


def encode(hub, result) -> bytes:
    return asyncio.run(hub.jpeg(1, result))


@pytest.fixture
def hub(homography):
    return FrameHub(StillTracker(homography), v_min_cm_s=1.0, video_width=FRAME_SIZE[0])


def test_a_frame_encodes_to_a_jpeg(hub):
    data = encode(hub, result_at(50.0, 30.0))
    assert data[:2] == b"\xff\xd8"  # JPEG start of image
    assert len(data) > 500


def test_the_overlay_draws_both_paths(hub):
    plain = encode(hub, result_at(50.0, 30.0))

    hub.overlay.reference_cm = [[float(x), 30.0] for x in range(10, 90, 5)]
    hub.overlay.walked_cm = [[float(x), 32.0] for x in range(10, 60, 5)]
    hub.overlay.carrot_cm = (60.0, 30.0)
    drawn = asyncio.run(hub.jpeg(2, result_at(50.0, 30.0)))

    # Same frame, more ink: the overlay reached the encoder.
    assert drawn != plain
    assert len(drawn) > len(plain)


def test_a_path_drawn_outside_the_arena_does_not_raise(hub):
    """A stroke can leave the calibrated quad, and cv2 refuses huge coordinates.

    The operator draws at the animal, not at the calibration corners, so a
    point beyond the arena has to clamp rather than take the video stream down.
    """
    hub.overlay.reference_cm = [[-5000.0, -5000.0], [50.0, 30.0], [9000.0, 9000.0]]
    assert encode(hub, result_at(50.0, 30.0))[:2] == b"\xff\xd8"


def test_a_wide_frame_is_scaled_down_for_the_tablet(homography):
    hub = FrameHub(StillTracker(homography), v_min_cm_s=1.0, video_width=160)
    small = encode(hub, result_at(50.0, 30.0))

    full = FrameHub(StillTracker(homography), v_min_cm_s=1.0, video_width=FRAME_SIZE[0])
    assert len(small) < len(encode(full, result_at(50.0, 30.0)))


def test_the_encoder_caches_one_frame_per_sequence_number(hub):
    once = encode(hub, result_at(50.0, 30.0))
    # A second viewer asking for the same sequence gets the cached bytes rather
    # than a second pass through render and imencode.
    again = asyncio.run(hub.jpeg(1, result_at(10.0, 10.0)))
    assert again is once
