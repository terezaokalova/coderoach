"""Overhead-camera trajectory tracking: pixels in, filtered pose in cm out.

:mod:`traj.track` implements ``rl_control.camera.PoseTracker`` so the RL loop
can read from a real camera instead of a simulator. :mod:`traj.calibrate`
produces the two JSON files that :mod:`traj.track` requires to start.
"""

from __future__ import annotations

__all__ = ["ArenaHomography", "HsvBounds", "TrackerConfig", "TrajectoryTracker"]


def __getattr__(name: str):
    # Deferred so that ``import traj`` stays cheap and does not pull in cv2.
    if name in __all__:
        from traj import track

        return getattr(track, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
