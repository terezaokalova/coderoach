"""Speech to text with faster-whisper on CPU.

The model is loaded once and reused. Loading is slow; transcribing is not,
and only transcription happens inside the demo loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class Transcript:
    text: str
    t_asr_return: float


class Transcriber:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
    ) -> None:
        self.model_name = model_name
        # Four threads is the M1 performance-core count. Letting ctranslate2
        # spread over the efficiency cores as well measured no faster.
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(self, samples: np.ndarray) -> Transcript:
        """Greedy decode of one short utterance.

        beam_size=1 and no conditioning on previous text: the vocabulary here
        is two words, and a wider beam buys nothing but latency.
        """
        segments, _ = self.model.transcribe(
            samples,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        # transcribe() returns a generator, so the work happens on consumption.
        text = "".join(segment.text for segment in segments)
        return Transcript(text=text, t_asr_return=time.monotonic())
