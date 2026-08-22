"""Fixed-window microphone capture.

The window is fixed rather than voice-activity gated: a demo command is one
short word, and endpoint detection would add its own latency and its own
failure mode on top of the recogniser's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

# What Whisper consumes. Anything else has to be resampled before ASR.
SAMPLE_RATE = 16000

# The rig mic, which is not the system default: CoreAudio hands index 1 (the
# built-in MacBook mic) to anything that asks for the default, and it sits too
# far from the speaker to clear the peak gate reliably. Index is positional, so
# plugging in another input device can renumber it; pass `device=` to override,
# or `device=None` to fall back to whatever the OS calls the default.
DEFAULT_INPUT_DEVICE = 0


@dataclass(frozen=True)
class Recording:
    samples: np.ndarray
    sample_rate: int
    t_capture_start: float
    t_capture_end: float

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def peak(self) -> float:
        """Loudest sample, 0.0 to 1.0. Near zero means nothing was heard."""
        return float(np.abs(self.samples).max()) if self.samples.size else 0.0


def record_window(
    seconds: float, device: int | str | None = DEFAULT_INPUT_DEVICE
) -> Recording:
    """Block for `seconds`, then return mono float32 at SAMPLE_RATE.

    t_capture_start is stamped once the stream is actually running, and
    t_capture_end once the last frame has landed, so the pair brackets the
    real capture rather than the surrounding function call.
    """
    if seconds <= 0:
        raise ValueError("recording window must be longer than 0 s")

    frames = int(round(seconds * SAMPLE_RATE))
    buffer = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    t_capture_start = time.monotonic()
    sd.wait()
    t_capture_end = time.monotonic()

    return Recording(
        samples=buffer.reshape(-1),
        sample_rate=SAMPLE_RATE,
        t_capture_start=t_capture_start,
        t_capture_end=t_capture_end,
    )
