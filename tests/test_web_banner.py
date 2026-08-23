"""The startup banner, and whether the QR code in it can actually be read.

The QR is rendered from half-block characters, so its polarity depends on the
terminal it lands in: on a dark background the block character is the *light*
module, on a light background it is the ink. Getting that backwards produces a
photographic negative. These tests rebuild the image a camera would see and
hand it to a real decoder, because a QR that renders and does not scan looks
exactly like one that works.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from web.__main__ import (
    build_parser,
    config_from_args,
    describe,
    qr_ascii,
    tablet_address,
)

URL = "http://192.168.1.42:8000"

# Every character print_ascii emits, as its (top, bottom) half-modules.
# 1 is the block character, 0 is the terminal background.
HALVES = {"█": (1, 1), "▀": (1, 0), "▄": (0, 1), "\xa0": (0, 0), " ": (0, 0)}


def as_camera_sees_it(text: str, *, dark_terminal: bool) -> np.ndarray:
    """The rendered code as a black and white image, the way a lens gets it."""
    rows: list[list[int]] = []
    for line in text.split("\n"):
        top, bottom = [], []
        for character in line:
            high, low = HALVES[character]
            if not dark_terminal:
                # Light background: the block character is ink, not light.
                high, low = 1 - high, 1 - low
            top.append(high)
            bottom.append(low)
        rows.append(top)
        rows.append(bottom)

    import cv2

    grid = np.array(rows, dtype=np.uint8) * 255
    return cv2.resize(grid, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST)


def qr_block(text: str) -> str:
    """Just the code out of a whole banner: the lines made only of blocks."""
    lines = [
        line
        for line in text.split("\n")
        if line and all(character in HALVES for character in line)
    ]
    return "\n".join(lines)


def decode(text: str, *, dark_terminal: bool) -> str:
    import cv2

    decoded, *_ = cv2.QRCodeDetector().detectAndDecode(
        as_camera_sees_it(text, dark_terminal=dark_terminal)
    )
    return decoded


@pytest.fixture(autouse=True)
def _needs_qrcode():
    pytest.importorskip("qrcode")


def test_the_default_render_scans_on_a_dark_terminal():
    assert decode(qr_ascii(URL), dark_terminal=True) == URL


def test_qr_light_scans_on_a_light_terminal():
    assert decode(qr_ascii(URL, invert=False), dark_terminal=False) == URL


def test_the_wrong_polarity_does_not_scan():
    """Why --qr-light exists.

    A dark-terminal render viewed on a light background is a negative. This
    asserts the failure so that nobody 'simplifies' the flag away: it is the
    difference between a code that scans and one that only looks like it does.
    """
    assert decode(qr_ascii(URL), dark_terminal=False) != URL


def test_the_quiet_zone_survives_the_indent():
    """The banner indent is drawn with light modules, not spaces.

    Padding with spaces would read as four more dark modules hard against the
    left edge of the code, eating the quiet zone a decoder needs.
    """
    first = qr_ascii(URL).split("\n")[0]
    assert first.startswith("██")
    assert " " not in first


def test_qr_ascii_degrades_when_the_package_is_absent(monkeypatch):
    # An env built before qrcode was added to environment.yml still has to
    # start and still has to print the URL.
    monkeypatch.setitem(sys.modules, "qrcode", None)
    assert qr_ascii(URL) is None


# -- which address the code should carry ---------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_has_no_address_to_offer(host):
    # Scanning a code for 127.0.0.1 would send the iPad to its own loopback.
    assert tablet_address(host) is None


def test_a_specific_interface_is_used_as_given(monkeypatch):
    monkeypatch.setattr("web.__main__.lan_address", lambda: "10.0.0.1")
    assert tablet_address("192.168.1.42") == "192.168.1.42"


def test_a_wildcard_bind_is_looked_up(monkeypatch):
    monkeypatch.setattr("web.__main__.lan_address", lambda: "10.0.0.1")
    assert tablet_address("0.0.0.0") == "10.0.0.1"


# -- the banner itself ---------------------------------------------------


def banner(host="0.0.0.0", **kwargs):
    args = build_parser().parse_args(
        ["--run-dir", "runs/demo", "--t-refrac", "2.0", "--no-camera"]
    )
    config, provenance = config_from_args(args)
    return describe(config, provenance, host, 8000, **kwargs)


def test_the_banner_carries_the_lan_url_and_its_code(monkeypatch):
    monkeypatch.setattr("web.__main__.lan_address", lambda: "192.168.1.42")
    text = banner()
    assert "http://192.168.1.42:8000" in text
    assert "open on the iPad" in text
    assert decode(qr_block(text), dark_terminal=True) == URL


def test_no_qr_leaves_the_url(monkeypatch):
    monkeypatch.setattr("web.__main__.lan_address", lambda: "192.168.1.42")
    text = banner(show_qr=False)
    assert "http://192.168.1.42:8000" in text
    assert "█" not in text


def test_a_loopback_bind_says_so_and_draws_nothing(monkeypatch):
    text = banner(host="127.0.0.1")
    assert "no iPad can reach this" in text
    assert "█" not in text


def test_no_route_says_so_and_draws_nothing(monkeypatch):
    monkeypatch.setattr("web.__main__.lan_address", lambda: None)
    text = banner()
    assert "no route found" in text
    assert "█" not in text
