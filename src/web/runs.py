"""Reading and writing what a run leaves behind.

A run directory holds three files written by three owners: the tracker's
per-frame JSONL, the gate's per-request JSONL, and ``path.json``, which is this
module's. Between them they are enough to redraw a run with no camera and no
backpack present, which is what the public deployment serves.

Deliberately standard library only. A replay-only deployment installs FastAPI
and uvicorn and nothing else -- no OpenCV, no bleak -- so nothing here may
import :mod:`traj` or :mod:`stim`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PATH_NAME = "path.json"

# Owned by traj.track.LOG_NAME and stim.gate.LOG_NAME. Repeated rather than
# imported because importing either module pulls in cv2 and bleak, which the
# replay deployment does not install. tests/test_web.py asserts the two spell
# the same thing, so a rename over there fails there rather than silently
# emptying replay over here.
TRACK_LOG_NAME = "traj_track.jsonl"
GATE_LOG_NAME = "stim_gate.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every well-formed object in a JSONL file, skipping a torn last line.

    A run that was killed mid-write leaves a partial final line. That is the
    common case for an interrupted session, and refusing to replay the whole
    run because of it would be the wrong trade.
    """
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _load_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"traces": []}
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{path} exists but is not valid JSON. Refusing to overwrite it."
        ) from error
    if not isinstance(loaded, dict):
        # RuntimeError rather than TypeError, matching stim.gate on the same
        # situation: the fault is a file on disk holding something other than
        # a run, not a caller passing the wrong type.
        raise RuntimeError(  # noqa: TRY004
            f"{path} exists but is not a JSON object."
        )
    loaded.setdefault("traces", [])
    return loaded


def append_trace(run_dir: Path | str, trace: dict[str, Any]) -> int:
    """Add one trace to the run's ``path.json`` and return its index.

    One run directory can hold several traces: the operator draws, watches,
    stops, and draws again without restarting the server. Overwriting on the
    second draw would throw away the first, so they accumulate.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / PATH_NAME

    document = _load_document(path)
    traces = document["traces"]
    index = len(traces)
    traces.append({"index": index, **trace})

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)
    return index


def list_runs(root: Path | str) -> list[dict[str, Any]]:
    """Every run directory under ``root``, newest first."""
    root = Path(root)
    if not root.is_dir():
        return []

    runs = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        summary = summarise_run(entry)
        if summary is not None:
            runs.append(summary)
    runs.sort(key=lambda run: run["modified"], reverse=True)
    return runs


def summarise_run(run_dir: Path) -> dict[str, Any] | None:
    """One run's headline numbers, or None if the directory holds no run."""
    path_file = run_dir / PATH_NAME
    track_file = run_dir / TRACK_LOG_NAME
    gate_file = run_dir / GATE_LOG_NAME

    present = [f for f in (path_file, track_file, gate_file) if f.exists()]
    if not present:
        return None

    traces = _load_document(path_file)["traces"] if path_file.exists() else []
    stims = read_jsonl(gate_file)
    return {
        "id": run_dir.name,
        "modified": max(f.stat().st_mtime for f in present),
        "traces": len(traces),
        "stims_accepted": sum(1 for row in stims if row.get("accepted")),
        "stims_total": len(stims),
        "has_track": track_file.exists(),
    }


def read_run(run_dir: Path | str) -> dict[str, Any]:
    """One run's reference paths, walked paths, and stimulation log."""
    run_dir = Path(run_dir)
    document = _load_document(run_dir / PATH_NAME)
    track = read_jsonl(run_dir / TRACK_LOG_NAME)
    stims = read_jsonl(run_dir / GATE_LOG_NAME)

    traces = []
    for trace in document["traces"]:
        walked = trace.get("walked_cm") or []
        if not walked:
            # The trace never got its walked path written, which is what an
            # interrupted run looks like. The tracker log was flushed per
            # frame, so the track survives even though path.json did not.
            walked = _walked_from_track(track, trace.get("t_start"), trace.get("t_end"))
        traces.append({**trace, "walked_cm": walked})

    return {
        "id": run_dir.name,
        "traces": traces,
        "stims": stims,
        "track_points": len(track),
    }


def _walked_from_track(
    track: list[dict[str, Any]],
    t_start: float | None,
    t_end: float | None,
) -> list[list[float]]:
    walked = []
    for row in track:
        t_frame = row.get("t_frame")
        x = row.get("px_hat")
        y = row.get("py_hat")
        if x is None or y is None or t_frame is None:
            continue
        if t_start is not None and t_frame < t_start:
            continue
        if t_end is not None and t_frame > t_end:
            continue
        walked.append([float(x), float(y)])
    return walked
