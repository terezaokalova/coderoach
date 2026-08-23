"""Voice command path into the shared stimulation gate."""

from .asr import Transcriber, Transcript
from .audio import DEFAULT_INPUT_DEVICE, Recording, record_window
from .command import (
    MIN_PEAK,
    NO_MATCH,
    PHRASES,
    REPEAT_PROMPT,
    TOO_QUIET,
    VoiceCommand,
    interpret,
    match,
    normalize,
)

__all__ = [
    "DEFAULT_INPUT_DEVICE",
    "MIN_PEAK",
    "NO_MATCH",
    "PHRASES",
    "REPEAT_PROMPT",
    "TOO_QUIET",
    "Recording",
    "Transcriber",
    "Transcript",
    "VoiceCommand",
    "interpret",
    "match",
    "normalize",
    "record_window",
]
