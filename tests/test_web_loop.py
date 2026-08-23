"""The frame hub and the control loop, with no camera and no backpack.

This is the path that ends in current delivered to an animal, so it is tested
against stubs rather than left to the first live run. What matters is not that
a turn gets requested -- it is that the loop keeps its side of the bargain with
the gate: at most one request per frame, a rejection dropped rather than
retried, and a refusal from the hardware envelope that does not end the run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading

import pytest
from webstubs import FakeHub, StubGate, frame

from traj.control import PursuitGains
from web import runs
from web.app import StimJournal
from web.hub import FrameHub
from web.loop import STOP_REPLACED, STOP_REQUESTED, ControlLoop

# -- the hub -------------------------------------------------------------


class BlockingTracker:
    """Hands over one frame per ``release()``, blocking in between.

    Blocking is what the real tracker does: ``process_once`` returns when the
    camera has produced a frame. Doing it with a semaphore makes the pump
    deterministic instead of a race against a busy loop.
    """

    def __init__(self, frames):
        self.frames = list(frames)
        self.permits = threading.Semaphore(0)
        self.index = 0
        self.closed = False
        self.homography = None

    def process_once(self):
        self.permits.acquire()
        if self.index >= len(self.frames):
            return None
        self.index += 1
        return self.frames[self.index - 1]

    def release(self, count=1):
        for _ in range(count):
            self.permits.release()

    def close(self):
        self.closed = True
        # Releasing the real VideoCapture unblocks the read() a pump thread is
        # parked in. Without the same courtesy here the thread would sit on the
        # semaphore forever and asyncio.run would hang waiting for it at exit.
        self.release(8)

    @property
    def log_path(self):
        return None


def test_hub_hands_out_the_newest_frame_not_a_backlog():
    async def scenario():
        tracker = BlockingTracker([frame(i) for i in range(4)])
        hub = FrameHub(tracker, v_min_cm_s=1.0, video_width=720)
        await hub.start()
        try:
            tracker.release()
            first = await asyncio.wait_for(hub.next_frame(0), 2)

            # Three frames pass while the reader is away. It resumes at the
            # newest, because stale video is worse than dropped video when the
            # operator is steering by it.
            tracker.release(3)
            await asyncio.sleep(0.05)
            second = await asyncio.wait_for(hub.next_frame(first[0]), 2)
            return tracker, first, second
        finally:
            await hub.aclose()

    tracker, first, second = asyncio.run(scenario())
    assert (first[0], first[1].px_hat) == (1, 0)
    assert (second[0], second[1].px_hat) == (4, 3)
    assert tracker.closed


def test_hub_surfaces_a_dead_camera_to_every_reader():
    async def scenario():
        tracker = BlockingTracker([])
        hub = FrameHub(tracker, v_min_cm_s=1.0, video_width=720)
        await hub.start()
        try:
            tracker.release()  # process_once returns None: the camera stopped
            with pytest.raises(RuntimeError, match="stopped delivering frames"):
                await asyncio.wait_for(hub.next_frame(0), 2)
        finally:
            await hub.aclose()

    asyncio.run(scenario())


# -- the loop ------------------------------------------------------------


# A straight path along +x. A roach sitting on it facing +y has its carrot hard
# to one side, so every frame wants a turn.
STRAIGHT = [[float(x), 0.0] for x in range(0, 22, 2)]
GAINS = PursuitGains(lookahead_cm=6.0, alpha_dead_rad=0.05)


def build(hub, gate, run_dir, journal=None):
    return ControlLoop(
        hub=hub, gate=gate, journal=journal or StimJournal(), run_dir=run_dir
    )


def test_loop_requests_one_turn_per_frame(tmp_path):
    async def scenario():
        hub = FakeHub([frame(i) for i in range(5)])
        gate = StubGate(["accept"])
        loop = build(hub, gate, tmp_path)
        await loop.start(STRAIGHT, GAINS)
        await asyncio.wait_for(hub.exhausted.wait(), 2)
        await loop.stop(STOP_REQUESTED)
        return gate

    gate = asyncio.run(scenario())
    assert len(gate.requests) == 5
    assert {source for _, source, _ in gate.requests} == {"traj"}
    # Carrot to the roach's right, so it asks to turn right every time.
    assert {direction for direction, _, _ in gate.requests} == {"right"}


def test_a_rejection_is_a_no_op_and_a_guard_does_not_end_the_run(tmp_path):
    async def scenario():
        hub = FakeHub([frame(i) for i in range(4)])
        # accepted, refused by the gate, refused by the hardware envelope,
        # accepted again.
        gate = StubGate(["accept", "refractory", "guard", "accept"])
        journal = StimJournal()
        loop = build(hub, gate, tmp_path, journal)
        await loop.start(STRAIGHT, GAINS)
        await asyncio.wait_for(hub.exhausted.wait(), 2)
        trace = await loop.stop(STOP_REQUESTED)
        return gate, journal, trace, loop.failure

    gate, journal, trace, failure = asyncio.run(scenario())
    assert len(gate.requests) == 4, "the loop stopped asking after a refusal"
    assert failure is None
    assert trace["stop_reason"] == STOP_REQUESTED

    reasons = [e.get("reject_reason") for e in journal.events if e["type"] == "stim"]
    assert reasons == [None, "refractory", "safety_guard", None]


def test_an_invalid_heading_asks_for_nothing(tmp_path):
    async def scenario():
        hub = FakeHub([frame(i, theta=None, valid=False) for i in range(4)])
        gate = StubGate(["accept"])
        loop = build(hub, gate, tmp_path)
        await loop.start(STRAIGHT, GAINS)
        await asyncio.wait_for(hub.exhausted.wait(), 2)
        await loop.stop(STOP_REQUESTED)
        return gate

    # A stale heading rotates the whole roach frame and turns the animal the
    # wrong way, so below v_min the loop asks for nothing at all.
    assert asyncio.run(scenario()).requests == []


def test_starting_a_second_path_cancels_and_saves_the_first(tmp_path):
    async def scenario():
        hub = FakeHub([frame(i) for i in range(3)])
        loop = build(hub, StubGate(["accept"]), tmp_path)

        await loop.start(STRAIGHT, GAINS)
        await asyncio.wait_for(hub.exhausted.wait(), 2)
        first = loop.trace_id

        hub.reload([frame(i) for i in range(3, 6)])
        await loop.start(STRAIGHT, GAINS)
        second = loop.trace_id
        await asyncio.wait_for(hub.exhausted.wait(), 2)
        await loop.stop(STOP_REQUESTED)
        return first, second

    first, second = asyncio.run(scenario())
    assert first != second

    document = json.loads((tmp_path / runs.PATH_NAME).read_text())
    assert [t["index"] for t in document["traces"]] == [0, 1]
    assert document["traces"][0]["stop_reason"] == STOP_REPLACED
    assert document["traces"][1]["stop_reason"] == STOP_REQUESTED


def test_gains_retune_mid_trace_and_a_bad_value_is_refused(tmp_path):
    async def scenario():
        hub = FakeHub([frame(i) for i in range(3)])
        loop = build(hub, StubGate(["accept"]), tmp_path)
        await loop.start(STRAIGHT, GAINS)
        await asyncio.wait_for(hub.exhausted.wait(), 2)

        walked_before = list(loop.walked_cm)
        loop.set_gains(12.0, 0.4)
        retuned = loop.gains

        with pytest.raises(ValueError, match="lookahead_cm"):
            loop.set_gains(0.0, 0.4)

        await loop.stop(STOP_REQUESTED)
        return walked_before, retuned, list(loop.walked_cm)

    walked_before, retuned, walked_after = asyncio.run(scenario())
    assert (retuned.lookahead_cm, retuned.alpha_dead_rad) == (12.0, 0.4)
    # Retuning replaces the gains only. The track already recorded stands.
    assert walked_before
    assert walked_after[: len(walked_before)] == walked_before


def test_stop_is_safe_when_nothing_is_running(tmp_path):
    async def scenario():
        loop = build(FakeHub([]), StubGate(["accept"]), tmp_path)
        return await loop.stop(STOP_REQUESTED), loop.active

    stopped, active = asyncio.run(scenario())
    assert stopped is None
    assert active is False


# -- what the terminal sees ----------------------------------------------


def test_the_repeated_reject_reason_still_matches_the_gate():
    """web.loop spells 'refractory' out rather than importing stim.

    Importing stim pulls in bleak, which the replay deployment does not
    install. If the gate renames the reason, the loop would silently start
    logging every refractory rejection at WARNING -- one line per frame.
    """
    from stim.gate import REFRACTORY as GATE_REFRACTORY
    from web.loop import REFRACTORY

    assert REFRACTORY == GATE_REFRACTORY


def test_a_guard_refusal_reaches_the_terminal(tmp_path, caplog):
    """The bug that made this invisible: guard refusals only hit the journal."""

    async def scenario():
        hub = FakeHub([frame(i) for i in range(3)])
        loop = build(hub, StubGate(["guard"]), tmp_path)
        await loop.start(STRAIGHT, GAINS)
        await asyncio.wait_for(hub.exhausted.wait(), 2)
        await loop.stop(STOP_REQUESTED)

    with caplog.at_level(logging.WARNING, logger="web.loop"):
        asyncio.run(scenario())

    guard_lines = [r for r in caplog.records if "hardware guard" in r.message]
    assert guard_lines, "a guard refusal printed nothing at WARNING"
    assert guard_lines[0].levelno == logging.WARNING


def test_refractory_rejections_do_not_flood_the_terminal(tmp_path, caplog):
    """Expected rejections go to DEBUG; the summary carries the count.

    At 30 Hz with a 2 s refractory window, roughly 59 of every 60 requests are
    refused this way. One INFO line each would bury everything else.
    """

    async def scenario():
        hub = FakeHub([frame(i) for i in range(6)])
        loop = build(hub, StubGate(["refractory"]), tmp_path)
        await loop.start(STRAIGHT, GAINS)
        await asyncio.wait_for(hub.exhausted.wait(), 2)
        return await loop.stop(STOP_REQUESTED)

    with caplog.at_level(logging.INFO, logger="web.loop"):
        asyncio.run(scenario())

    at_info = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert not [r for r in at_info if "rejected (refractory)" in r.message]
    # The stop line still reports every one of them.
    stop_line = [r for r in at_info if "stop" in r.message]
    assert stop_line and "'refractory': 6" in stop_line[0].message


def test_start_and_stop_are_logged(tmp_path, caplog):
    async def scenario():
        hub = FakeHub([frame(i) for i in range(3)])
        loop = build(hub, StubGate(["accept"]), tmp_path)
        await loop.start(STRAIGHT, GAINS)
        await asyncio.wait_for(hub.exhausted.wait(), 2)
        await loop.stop(STOP_REQUESTED)

    with caplog.at_level(logging.INFO, logger="web.loop"):
        asyncio.run(scenario())

    messages = [r.message for r in caplog.records]
    assert any("start" in m and "waypoints" in m for m in messages)
    assert any("stop (requested)" in m for m in messages)
    assert any("FIRED" in m for m in messages)
