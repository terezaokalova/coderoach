"""Restricting the contour search to the calibrated arena.

The tracker takes the largest contour in the frame. Anything red outside the
arena -- cardboard, a cable, a sleeve -- therefore competes with the backpack
and wins the moment it is bigger, and no HSV window separates the two reliably
because the difference between them is where they are, not what colour they
are. These tests build exactly that frame and check both halves of the claim:
that the search does pick the wrong blob unmasked, and that masking to the
arena makes it impossible.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from traj.track import ArenaHomography, HsvBounds, detect

FRAME_SIZE = (320, 240)

# Clicked around the perimeter, y down, as traj.calibrate records them.
ARENA_PX = [[100, 180], [220, 180], [220, 60], [100, 60]]
ARENA_CM = [[0, 0], [60, 0], [60, 60], [0, 60]]

# Wide open on hue: the point is that no HSV window can fix this.
RED = HsvBounds(h_lo=0, h_hi=179, s_lo=100, s_hi=255, v_lo=80, v_hi=255)

MIN_AREA = 100.0


@pytest.fixture
def homography(tmp_path):
    path = tmp_path / "arena.json"
    path.write_text(
        json.dumps(
            {
                "frame_size": list(FRAME_SIZE),
                "image_points": ARENA_PX,
                "arena_points_cm": ARENA_CM,
            }
        )
    )
    return ArenaHomography.from_json(path)


def frame_with(*boxes) -> np.ndarray:
    """A black frame with a saturated red rectangle for each (x0, y0, x1, y1)."""
    frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        frame[y0:y1, x0:x1] = (0, 0, 255)
    return frame


# The backpack: modest, inside the arena.
ROACH = (145, 105, 175, 135)
# The cardboard: much larger, entirely outside the arena quad.
CARDBOARD = (10, 10, 90, 230)


def test_the_mask_covers_the_arena_and_nothing_else(homography):
    mask = homography.region_mask()
    assert mask.shape == (FRAME_SIZE[1], FRAME_SIZE[0])
    assert mask.dtype == np.uint8

    assert mask[120, 160] == 255, "the middle of the arena must be searched"
    for point in ((20, 20), (120, 20), (20, 200), (300, 200)):
        assert mask[point[1], point[0]] == 0, f"{point} is outside the arena"

    # The clicked rectangle is 120x120 px in a 320x240 frame.
    assert cv2.countNonZero(mask) == pytest.approx(120 * 120, rel=0.02)


def test_unmasked_the_cardboard_wins(homography):
    """The bug, stated as a test. Without a region the biggest blob wins."""
    detection, _ = detect(frame_with(ROACH, CARDBOARD), RED, MIN_AREA)
    assert detection is not None
    assert detection.cx_px < 100, "the cardboard outside the arena was selected"


def test_masked_the_roach_wins_however_big_the_cardboard(homography):
    detection, _ = detect(
        frame_with(ROACH, CARDBOARD), RED, MIN_AREA, region=homography.region_mask()
    )
    assert detection is not None
    # The centroid of the backpack, not of the cardboard.
    assert detection.cx_px == pytest.approx(160, abs=3)
    assert detection.cy_px == pytest.approx(120, abs=3)


def test_the_returned_mask_is_what_was_searched(homography):
    _, mask = detect(
        frame_with(ROACH, CARDBOARD), RED, MIN_AREA, region=homography.region_mask()
    )
    # The calibrator displays this, so it has to show the cardboard gone.
    assert mask[120, 160] == 255
    assert cv2.countNonZero(mask[:, :100]) == 0


def test_nothing_inside_the_arena_means_no_detection(homography):
    detection, _ = detect(
        frame_with(CARDBOARD), RED, MIN_AREA, region=homography.region_mask()
    )
    # Reporting nothing beats reporting the cardboard's centroid as the animal.
    assert detection is None


def test_a_blob_on_the_boundary_is_cut_at_the_boundary(homography):
    """Straddling the edge clips, rather than closing across it.

    The morphology runs before the region is applied, so a blob reaching over
    the arena edge cannot be closed into the excluded side and then survive.
    """
    straddle = (60, 100, 160, 140)
    detection, _ = detect(
        frame_with(straddle), RED, MIN_AREA, region=homography.region_mask()
    )
    assert detection is not None
    # Centroid of the part inside the arena (x from 100 to 160), not of the
    # whole blob, which would sit near x=110.
    assert detection.cx_px == pytest.approx(130, abs=4)


def test_region_none_leaves_the_old_behaviour_alone(homography):
    """The HSV calibrator calls detect before any arena calibration exists."""
    with_none, mask_none = detect(frame_with(ROACH), RED, MIN_AREA, region=None)
    positional, mask_positional = detect(frame_with(ROACH), RED, MIN_AREA)
    assert with_none.cx_px == positional.cx_px
    assert np.array_equal(mask_none, mask_positional)


def test_a_degenerate_outline_is_refused(tmp_path):
    """An arena that encloses nothing would mask the whole frame away."""
    path = tmp_path / "arena.json"
    path.write_text(
        json.dumps(
            {
                "frame_size": list(FRAME_SIZE),
                # Collinear clicks: no area, and a tracker that silently never
                # detects anything again.
                "image_points": [[10, 10], [20, 10], [30, 10], [40, 10]],
                "arena_points_cm": ARENA_CM,
            }
        )
    )
    # The homography fit rejects this first, which is the earlier and better
    # place for it to fail.
    with pytest.raises(ValueError):
        ArenaHomography.from_json(path)


def test_the_outline_survives_the_round_trip(homography):
    assert np.allclose(homography.image_points, np.asarray(ARENA_PX, float))
