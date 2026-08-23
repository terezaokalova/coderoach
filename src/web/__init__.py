"""Browser front end: draw a path on an iPad, drive the roach along it.

The page is served over the LAN so that an iPad can reach it, which is why
:mod:`web.__main__` binds 0.0.0.0 rather than localhost. Every path through
this package that ends in current ends at :class:`stim.StimGate`, which owns
the refractory period. Nothing here re-implements it, and a rejection from it
is a no-op rather than something to retry.

Three moving parts, kept apart on purpose:

* :mod:`web.hub` owns the one camera and fans its frames out.
* :mod:`web.loop` owns the one control task.
* :mod:`web.app` owns the HTTP surface and holds neither directly.

:mod:`web.runs` is pure standard library, and the OpenCV and Bluetooth imports
elsewhere are deferred, so a replay-only deployment needs nothing but FastAPI
and uvicorn installed.
"""

from __future__ import annotations

__all__ = ["WebConfig", "create_app"]


def __getattr__(name: str):
    # Deferred so that importing web.runs for a replay deployment does not
    # pull in cv2 or bleak by way of the application module.
    if name in __all__:
        from web import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
