"""Pure pursuit with a deadband, driving the shared stimulation gate.

The tracker reports where the roach is and which way it is moving. This turns
that into at most one stimulation request per frame: find the nearest point on
the drawn path, look ``L_d`` further along it for a carrot, express the carrot
in the roach's own frame, and ask for a turn only when the bearing to it
exceeds ``alpha_dead``.

The gate owns the refractory period. Nothing here duplicates it -- this decides
*whether a turn is wanted*, the gate decides *whether one may fire*, and a
rejection is simply dropped. Two independent limiters would interact in ways
neither could be reasoned about alone.

Angles follow ``rl_control``: theta is CCW-positive and a left turn is +theta,
so a carrot to the roach's left gives alpha > 0 and asks for ``"left"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NO_PATH = "no_path"
NO_HEADING = "heading_invalid"
IN_DEADBAND = "in_deadband"
AT_END = "at_end"


@dataclass(frozen=True)
class PursuitGains:
    """Both are required: they are the two knobs the operator tunes live.

    ``lookahead_cm`` (L_d) trades cornering accuracy against oscillation --
    short lookahead cuts corners tightly and hunts, long lookahead is smooth
    and wide. ``alpha_dead_rad`` is the bearing error tolerated before asking
    for a turn, and it is what keeps a roach that is already tracking well from
    burning its refractory budget on noise.
    """

    lookahead_cm: float
    alpha_dead_rad: float

    def __post_init__(self) -> None:
        if self.lookahead_cm <= 0.0:
            raise ValueError("lookahead_cm (L_d) must be positive")
        if self.alpha_dead_rad < 0.0:
            raise ValueError("alpha_dead_rad must not be negative")


@dataclass(frozen=True)
class PursuitDecision:
    """What the controller wants, and every intermediate that produced it.

    The intermediates are kept because this is the piece most likely to be
    wrong on a live animal, and a decision that cannot be read back is a
    decision that cannot be debugged mid-run.
    """

    direction: str | None
    reason: str | None
    nearest_index: int
    carrot_index: int
    carrot_cm: tuple[float, float] | None
    e_lon: float | None
    e_lat: float | None
    alpha: float | None
    cross_track_cm: float | None
    at_end: bool

    @property
    def wants_request(self) -> bool:
        return self.direction is not None


class PurePursuit:
    """Holds the reference path and turns one pose into one decision."""

    def __init__(self, waypoints_cm, gains: PursuitGains) -> None:
        path = np.asarray(waypoints_cm, dtype=np.float64).reshape(-1, 2)
        if len(path) < 2:
            raise ValueError("a reference path needs at least two waypoints")
        self.path = path
        self.gains = gains
        # Arc length to each waypoint, so the carrot advances by distance along
        # the path rather than by a fixed number of indices. Evenly spaced
        # waypoints make those nearly the same; unevenly spaced ones do not,
        # and the drawn stroke is only as even as the resampler made it.
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.arc_length = np.concatenate([[0.0], np.cumsum(steps)])

    @property
    def length_cm(self) -> float:
        return float(self.arc_length[-1])

    def nearest_index(self, x: float, y: float) -> int:
        """i*: index of the reference point nearest the roach."""
        return int(np.argmin(np.linalg.norm(self.path - (x, y), axis=1)))

    def carrot_index(self, nearest: int) -> int:
        """The point L_d further along the path from i*.

        Clamped at the final waypoint, which is what makes the roach drive at
        the end of the path once it is within a lookahead of it rather than
        losing its target.
        """
        target = self.arc_length[nearest] + self.gains.lookahead_cm
        return int(
            np.searchsorted(self.arc_length, target, side="left").clip(
                0, len(self.path) - 1
            )
        )

    def decide(
        self, x: float, y: float, theta: float | None, heading_valid: bool
    ) -> PursuitDecision:
        """One pose in, at most one turn request out.

        ``theta`` is only meaningful while ``heading_valid``. The tracker
        reports heading as unavailable below its own v_min rather than holding
        a stale value, and a stale heading here would rotate the whole roach
        frame and turn the animal the wrong way, so an invalid heading means no
        request at all.
        """
        nearest = self.nearest_index(x, y)
        carrot = self.carrot_index(nearest)
        target = self.path[carrot]
        at_end = carrot == len(self.path) - 1
        cross_track = float(np.linalg.norm(self.path[nearest] - (x, y)))

        if not heading_valid or theta is None:
            return PursuitDecision(
                direction=None,
                reason=NO_HEADING,
                nearest_index=nearest,
                carrot_index=carrot,
                carrot_cm=(float(target[0]), float(target[1])),
                e_lon=None,
                e_lat=None,
                alpha=None,
                cross_track_cm=cross_track,
                at_end=at_end,
            )

        # The carrot in the roach's own frame: e_lon ahead, e_lat to its left.
        dx = float(target[0]) - x
        dy = float(target[1]) - y
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        e_lon = dx * cos_t + dy * sin_t
        e_lat = -dx * sin_t + dy * cos_t
        alpha = math.atan2(e_lat, e_lon)

        if abs(alpha) < self.gains.alpha_dead_rad:
            direction, reason = None, IN_DEADBAND
        else:
            # alpha > 0 puts the carrot to the roach's left, and +theta is a
            # left turn, so the sign maps straight onto the gate's direction.
            direction, reason = ("left" if alpha > 0.0 else "right"), None

        return PursuitDecision(
            direction=direction,
            reason=reason,
            nearest_index=nearest,
            carrot_index=carrot,
            carrot_cm=(float(target[0]), float(target[1])),
            e_lon=e_lon,
            e_lat=e_lat,
            alpha=alpha,
            cross_track_cm=cross_track,
            at_end=at_end,
        )


def resample_path(points_cm, spacing_cm: float) -> list[list[float]]:
    """Evenly spaced waypoints along a polyline, by arc length.

    The browser resamples the drawn stroke before POSTing it; this is the same
    operation server-side, used to re-space a path that arrives uneven. A
    mouse stroke bunches points where the hand slowed down, and pure pursuit
    reads spacing as distance, so unresampled input makes the carrot advance
    at the speed the operator happened to draw.
    """
    path = np.asarray(points_cm, dtype=np.float64).reshape(-1, 2)
    if spacing_cm <= 0.0:
        raise ValueError("spacing_cm must be positive")
    if len(path) < 2:
        return [[float(p[0]), float(p[1])] for p in path]

    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(arc[-1])
    if total <= 0.0:
        return [[float(path[0][0]), float(path[0][1])]]

    wanted = np.arange(0.0, total, spacing_cm)
    wanted = np.append(wanted, total)
    return [
        [float(np.interp(d, arc, path[:, 0])), float(np.interp(d, arc, path[:, 1]))]
        for d in wanted
    ]
