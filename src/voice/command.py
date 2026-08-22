"""Turn one captured utterance into a direction, or into a reason why not.

The decision lives here rather than in the caller so that "was that a command?"
has exactly one answer no matter who asks. Two things are deliberately absent:
fuzzy matching and nearest-phrase fallback. An utterance that lands near a
phrase is not evidence that the speaker meant it, and this path ends in current
delivered to a live animal, so anything unrecognised is rejected outright.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .asr import Transcriber
from .audio import Recording

# A silent room on the built-in mic measures about 0.01-0.02 peak; speech at a
# normal level clears 0.1. The gate sits between the two. It is high on purpose:
# Whisper handed near-silence does not return an empty string, it returns a
# confident hallucination, and the cheapest way to not act on one is to never
# produce it.
MIN_PEAK = 0.08

# The whole vocabulary. Bare "left" and "right" are excluded deliberately: both
# turn up in ordinary conversation near the rig, and the two-word form does not.
PHRASES: dict[str, str] = {"turn left": "left", "turn right": "right"}

TOO_QUIET = "too_quiet"
NO_MATCH = "no_match"

REPEAT_PROMPT = "too quiet - say it again, louder and closer to the mic."


@dataclass(frozen=True)
class VoiceCommand:
    """One utterance, resolved. `direction` is None whenever it was rejected."""

    direction: str | None
    raw_text: str
    heard: str
    peak: float
    reject_reason: str | None

    @property
    def accepted(self) -> bool:
        return self.direction is not None


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    Punctuation becomes a space rather than nothing, so a hyphenated
    "turn-left" normalises to the same string as "turn left" instead of to
    "turnleft". Whisper punctuates freely, and the trailing period on
    "Turn left." is the common case rather than the exotic one.
    """
    folded = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text.lower()
    )
    return " ".join(folded.split())


def match(normalized: str) -> str | None:
    """Exact lookup of an already-normalised utterance, or None.

    Takes normalised text rather than raw so the two stages stay separable and
    the caller keeps the normalised string it matched on for the log.
    """
    return PHRASES.get(normalized)


def interpret(recording: Recording, transcriber: Transcriber) -> VoiceCommand:
    """Peak gate, then ASR, then normalise, then exact match.

    The peak gate runs first and returns without transcribing, so a recording
    that never really heard the speaker costs no ASR and produces no text that
    could be acted on by mistake.
    """
    if recording.peak < MIN_PEAK:
        return VoiceCommand(
            direction=None,
            raw_text="",
            heard="",
            peak=recording.peak,
            reject_reason=TOO_QUIET,
        )

    transcript = transcriber.transcribe(recording.samples)
    heard = normalize(transcript.text)
    direction = match(heard)
    return VoiceCommand(
        direction=direction,
        raw_text=transcript.text,
        heard=heard,
        peak=recording.peak,
        reject_reason=None if direction is not None else NO_MATCH,
    )
