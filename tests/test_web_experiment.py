"""The web experiment console, with no camera and no backpack."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from webstubs import FakeHub, StubGate, frame

from web.app import CAMERA, ROACH, Capability, WebConfig, create_app
from web.experiment import HubPoseTracker
from web.loop import ControlLoop
from web import runs


@pytest.fixture
def run_root(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def client(run_root):
    config = WebConfig(
        run_dir=run_root / "today",
        t_refrac_s=0.0,
        camera_index=None,
        with_roach=False,
        with_voice=False,
    )
    with TestClient(create_app(config)) as client:
        yield client


@pytest.fixture
def live(run_root):
    config = WebConfig(
        run_dir=run_root / "today",
        t_refrac_s=0.0,
        spacing_cm=2.0,
        camera_index=None,
        with_roach=False,
        with_voice=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        runtime = app.state.runtime
        hub = FakeHub([frame(i) for i in range(8)])
        gate = StubGate(["accept"])
        runtime.hub = hub
        runtime.gate = gate
        runtime.loop = ControlLoop(
            hub=hub, gate=gate, journal=runtime.journal, run_dir=config.run_dir
        )
        runtime.status[CAMERA] = Capability(True, "stub camera")
        runtime.status[ROACH] = Capability(True, "stub backpack")
        yield SimpleNamespace(client=client, runtime=runtime, gate=gate, config=config)


def _wait_steps(runtime, count=1, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        steps = [event for event in runtime.journal.events if event.get("type") == "rl_step"]
        if len(steps) >= count:
            return steps
        time.sleep(0.05)
    raise AssertionError("experiment did not emit an rl_step")


def test_hub_pose_skips_undetected_and_duplicate_timestamps():
    frames = [
        SimpleNamespace(t_frame=1.0, px_hat=None, py_hat=None),
        SimpleNamespace(t_frame=2.0, px_hat=1.0, py_hat=2.0),
        SimpleNamespace(t_frame=2.0, px_hat=1.0, py_hat=2.0),
        SimpleNamespace(t_frame=3.0, px_hat=3.0, py_hat=4.0),
    ]
    hub = FakeHub(frames)
    tracker = HubPoseTracker(hub)

    async def scenario():
        first = await tracker.read()
        second = await tracker.read()
        return first, second

    first, second = asyncio.run(scenario())
    assert (first.x, first.y, first.t) == (1.0, 2.0, 2.0)
    assert (second.x, second.y, second.t) == (3.0, 4.0, 3.0)


def test_start_sim_teach_emits_a_step_and_writes_jsonl(run_root):
    config = WebConfig(
        run_dir=run_root / "today",
        t_refrac_s=0.0,
        camera_index=None,
        with_roach=False,
        with_voice=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        runtime = app.state.runtime
        response = client.post(
            "/api/experiment/start",
            json={"command": "teach", "policy": "static", "max_steps": 3},
        )
        assert response.status_code == 200
        assert response.json()["running"] is True
        assert response.json()["episode"] == 1

        steps = _wait_steps(runtime)
        assert steps[0]["type"] == "rl_step"
        assert steps[0]["episode"] == 1
        assert "reward" in steps[0]
        assert "policy_text" in steps[0]

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and runtime.experiment.active:
            time.sleep(0.05)
        finished = [
            event
            for event in runtime.journal.events
            if event.get("type") == "rl_status" and event.get("running") is False
        ]
        assert finished
        assert runtime.experiment.active is False

        stop = client.post("/api/experiment/stop")
        assert stop.status_code == 200
        assert stop.json()["running"] is False

        restart = client.post("/api/experiment/restart")
        assert restart.status_code == 200
        assert restart.json()["episode"] == 2
        _wait_steps(runtime, count=len(steps) + 1)
        client.post("/api/experiment/stop")

    detail = runs.read_run(config.run_dir)
    assert detail["steps"]
    assert all(row.get("type") == "rl_step" for row in detail["steps"])
    listed = runs.summarise_run(config.run_dir)
    assert listed["steps"] == len(detail["steps"])


def test_restart_without_a_prior_start_is_conflict(client):
    response = client.post("/api/experiment/restart")
    assert response.status_code == 409


def test_start_refuses_a_bad_command(client):
    response = client.post("/api/experiment/start", json={"command": "spam"})
    assert response.status_code == 400


def test_starting_an_experiment_stops_a_path_trace(live):
    live.client.post("/api/path", json={"points": [[0.0, 0.0], [10.0, 0.0]]})
    assert live.runtime.loop.active

    response = live.client.post(
        "/api/experiment/start",
        json={"command": "teach", "policy": "static", "max_steps": 2},
    )
    assert response.status_code == 200
    assert live.runtime.loop.active is False
    live.client.post("/api/experiment/stop")


def test_starting_a_path_stops_an_experiment(run_root):
    config = WebConfig(
        run_dir=run_root / "today",
        t_refrac_s=0.0,
        spacing_cm=2.0,
        camera_index=None,
        with_roach=False,
        with_voice=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        runtime = app.state.runtime
        hub = FakeHub([frame(i) for i in range(8)])
        gate = StubGate(["accept"])
        runtime.hub = hub
        runtime.gate = gate
        runtime.loop = ControlLoop(
            hub=hub, gate=gate, journal=runtime.journal, run_dir=config.run_dir
        )
        runtime.status[CAMERA] = Capability(True, "stub camera")
        runtime.status[ROACH] = Capability(True, "stub backpack")

        started = client.post(
            "/api/experiment/start",
            json={"command": "teach", "policy": "static", "max_steps": 80},
        )
        assert started.status_code == 200
        _wait_steps(runtime)

        response = client.post(
            "/api/path", json={"points": [[0.0, 0.0], [10.0, 0.0]]}
        )
        assert response.status_code == 200
        assert runtime.experiment.active is False
        client.post("/api/stop")
