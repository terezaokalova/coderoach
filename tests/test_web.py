"""The web surface, with no camera, no backpack, and no microphone present.

Everything here runs in the degraded mode the server falls back to when the
arena calibration is missing, which is also the mode the public deployment
serves. That is deliberate: it is the configuration most likely to be deployed
without anyone watching it start.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from web import runs
from web.app import PAGE, WebConfig, _arena_bounds, create_app


@pytest.fixture
def run_root(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def client(run_root):
    config = WebConfig(
        run_dir=run_root / "today",
        t_refrac_s=2.0,
        camera_index=None,
        with_roach=False,
        with_voice=False,
    )
    with TestClient(create_app(config)) as client:
        yield client


# -- the duplicated file names ------------------------------------------


def test_log_names_match_their_owners():
    """web.runs repeats two names it may not import. This is the guard.

    A replay deployment installs neither cv2 nor bleak, so runs.py spells the
    tracker and gate log names out by hand. If either owner renames its log,
    replay would quietly return empty runs forever. It fails here instead.
    """
    from stim.gate import LOG_NAME as GATE_LOG_NAME
    from traj.track import LOG_NAME as TRACK_LOG_NAME
    from web.experiment import STEPS_NAME

    assert runs.TRACK_LOG_NAME == TRACK_LOG_NAME
    assert runs.GATE_LOG_NAME == GATE_LOG_NAME
    assert runs.RL_STEPS_NAME == STEPS_NAME


def test_runs_module_imports_nothing_heavy():
    """The replay deployment installs FastAPI and uvicorn and nothing else."""
    text = Path(runs.__file__).read_text()
    for forbidden in ("import cv2", "import numpy", "from traj", "from stim"):
        assert forbidden not in text


# -- run directories -----------------------------------------------------


def test_append_trace_accumulates(tmp_path):
    run_dir = tmp_path / "run"
    first = runs.append_trace(run_dir, {"reference_cm": [[0, 0], [1, 1]]})
    second = runs.append_trace(run_dir, {"reference_cm": [[2, 2], [3, 3]]})

    assert (first, second) == (0, 1)
    document = json.loads((run_dir / runs.PATH_NAME).read_text())
    assert [trace["index"] for trace in document["traces"]] == [0, 1]


def test_read_run_falls_back_to_the_tracker_log(tmp_path):
    """An interrupted trace still replays: the tracker log was flushed."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / runs.TRACK_LOG_NAME).write_text(
        "\n".join(
            json.dumps({"t_frame": t, "px_hat": float(t), "py_hat": 0.0})
            for t in (1, 2, 3, 4)
        )
        # A run killed mid-write leaves a torn final line.
        + '\n{"t_frame": 5, "px_h'
    )
    runs.append_trace(run_dir, {"t_start": 2, "t_end": 3, "walked_cm": []})

    trace = runs.read_run(run_dir)["traces"][0]
    assert trace["walked_cm"] == [[2.0, 0.0], [3.0, 0.0]]


def test_list_runs_skips_directories_that_hold_no_run(run_root):
    (run_root / "empty").mkdir()
    runs.append_trace(run_root / "real", {"reference_cm": []})

    listed = runs.list_runs(run_root)
    assert [run["id"] for run in listed] == ["real"]


# -- the arena -----------------------------------------------------------


def test_arena_bounds_pads_the_calibration(tmp_path):
    path = tmp_path / "arena.json"
    path.write_text(json.dumps({"arena_points_cm": [[0, 0], [40, 0], [40, 30]]}))

    bounds = _arena_bounds(path)
    assert bounds == {"x_min": -2.0, "x_max": 42.0, "y_min": -2.0, "y_max": 32.0}


def test_arena_bounds_survives_a_missing_or_broken_file(tmp_path):
    assert _arena_bounds(None) is None
    assert _arena_bounds(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert _arena_bounds(broken) is None


# -- the page ------------------------------------------------------------


def test_index_substitutes_the_config(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "__WEB_CONFIG__" not in response.text
    assert '"live": false' in response.text


def test_page_handles_pointers_not_mice():
    """The iPad requirement, asserted against the page itself."""
    page = PAGE.read_text()
    for event in ("pointerdown", "pointermove", "pointerup", "pointercancel"):
        assert f'canvas.addEventListener("{event}"' in page
    # Mouse events would leave the Pencil drawing nothing.
    for event in ("mousedown", "mousemove", "mouseup"):
        assert event not in page
    # Without this Safari scrolls and zooms the page instead of drawing.
    assert "touch-action: none" in page
    assert 'pointerType === "pen"' in page


# -- capability gating ---------------------------------------------------


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/turn", {"direction": "left"}),
        ("/api/path", {"points": [[0, 0], [5, 5]]}),
        ("/api/voice", None),
    ],
)
def test_control_endpoints_refuse_with_a_specific_reason(client, path, body):
    response = client.post(path, json=body)
    assert response.status_code == 409
    # Not just "unavailable": the operator has to know what to go and plug in.
    assert "unavailable" in response.json()["detail"]


def test_stop_is_safe_when_there_is_no_control_loop(client):
    response = client.post("/api/stop")
    assert response.status_code == 200
    assert response.json()["stopped"] is False


def test_runs_endpoints_read_the_sibling_directories(client, run_root):
    runs.append_trace(run_root / "earlier", {"reference_cm": [[0, 0], [1, 0]]})

    listed = client.get("/api/runs").json()
    assert listed["current"] == "today"
    assert [run["id"] for run in listed["runs"]] == ["earlier"]

    detail = client.get("/api/runs/earlier").json()
    assert detail["traces"][0]["reference_cm"] == [[0, 0], [1, 0]]


def test_run_id_cannot_escape_the_runs_root(client, run_root):
    assert client.get("/api/runs/absent").status_code == 404
    assert client.get("/api/runs/..").status_code == 404
    assert client.get("/api/runs/%2e%2e%2f%2e%2e").status_code == 404


# -- the wiring between the endpoints and the loop -----------------------


@pytest.fixture
def live(run_root):
    """A client whose camera and backpack are stubs rather than absent.

    The lifespan brings the server up degraded, exactly as it would with no
    hardware attached, and the stubs are put in afterwards. That keeps this
    fixture from depending on how startup discovers hardware -- which is the
    part that cannot be tested without any.
    """
    from webstubs import FakeHub, StubGate, frame

    from web.app import CAMERA, ROACH, Capability
    from web.loop import ControlLoop

    config = WebConfig(
        run_dir=run_root / "today",
        t_refrac_s=2.0,
        spacing_cm=2.0,
        camera_index=None,
        with_roach=False,
        with_voice=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        runtime = app.state.runtime
        hub = FakeHub([frame(i) for i in range(4)])
        gate = StubGate(["accept"])
        runtime.hub = hub
        runtime.gate = gate
        runtime.loop = ControlLoop(
            hub=hub, gate=gate, journal=runtime.journal, run_dir=config.run_dir
        )
        runtime.status[CAMERA] = Capability(True, "stub camera")
        runtime.status[ROACH] = Capability(True, "stub backpack")
        yield SimpleNamespace(client=client, runtime=runtime, gate=gate, config=config)


def test_turn_reaches_the_gate_and_lands_in_the_journal(live):
    response = live.client.post("/api/turn", json={"direction": "left"})
    assert response.status_code == 200

    event = response.json()
    assert (event["accepted"], event["direction"], event["source"]) == (
        True,
        "left",
        "text",
    )
    assert [direction for direction, _, _ in live.gate.requests] == ["left"]
    assert live.runtime.journal.events[-1]["request_id"] == event["request_id"]


def test_turn_rejects_a_direction_the_gate_would_refuse(live):
    assert live.client.post("/api/turn", json={"direction": "up"}).status_code == 400
    assert live.client.post("/api/turn", json={}).status_code == 400
    assert live.gate.requests == []


def test_path_resamples_server_side_before_following_it(live):
    # Two points 10 cm apart, at 2 cm spacing, is six waypoints including both
    # ends. The browser's own resampling is not trusted to have happened.
    response = live.client.post("/api/path", json={"points": [[0.0, 0.0], [10.0, 0.0]]})
    assert response.status_code == 200

    body = response.json()
    assert len(body["waypoints_cm"]) == 6
    assert body["waypoints_cm"][0] == [0.0, 0.0]
    assert body["waypoints_cm"][-1] == [10.0, 0.0]
    assert body["length_cm"] == pytest.approx(10.0)


def test_path_refuses_a_stroke_shorter_than_one_step(live):
    response = live.client.post("/api/path", json={"points": [[0.0, 0.0], [0.0, 0.0]]})
    assert response.status_code == 400
    assert "shorter than one" in response.json()["detail"]


def test_path_refuses_gains_pure_pursuit_would_not_accept(live):
    response = live.client.post(
        "/api/path",
        json={"points": [[0.0, 0.0], [10.0, 0.0]], "lookahead_cm": 0.0},
    )
    assert response.status_code == 400
    assert "lookahead_cm" in response.json()["detail"]


def test_stop_writes_the_trace_out(live):
    live.client.post("/api/path", json={"points": [[0.0, 0.0], [10.0, 0.0]]})
    response = live.client.post("/api/stop")
    assert response.status_code == 200
    assert response.json()["stopped"] is True

    document = json.loads((live.config.run_dir / runs.PATH_NAME).read_text())
    assert document["traces"][0]["stop_reason"] == "requested"


# -- the page's own refusal handling -------------------------------------


def test_send_is_gated_on_what_api_path_requires():
    """Send was enabled while Left/Right were greyed out.

    POST /api/path requires the camera and the backpack, so a drawn path with
    no backpack produced a 409 from a button that looked live. Both now come
    from one predicate.
    """
    page = PAGE.read_text()
    assert "function sendBlockedBy()" in page
    # The same two capabilities api_path calls require() with.
    assert '["camera", "roach"]' in page
    # Nothing sets the button by hand any more; it all goes through one place.
    assert '$("send").disabled = ' not in page
    assert page.count("refreshSend()") >= 4


def test_a_refusal_shows_its_status_code(client):
    """The log row used to read 'request failed' and nothing else."""
    page = PAGE.read_text()
    assert "error.status = response.status" in page
    assert "HTTP ${error.status}" in page

    # And the server still supplies a detail worth showing.
    body = client.post("/api/path", json={"points": [[0, 0], [5, 5]]}).json()
    assert "unavailable" in body["detail"]


def test_a_capability_refusal_is_logged_server_side(client, caplog):
    import logging as stdlib_logging

    with caplog.at_level(stdlib_logging.WARNING, logger="web.app"):
        client.post("/api/turn", json={"direction": "left"})
    assert any("roach unavailable" in r.getMessage() for r in caplog.records)
