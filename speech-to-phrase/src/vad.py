#!/usr/bin/env python3
"""Voice-activity trimming for Speech-to-Phrase.

Home Assistant streams the whole utterance starting from the moment the wake
word fires, so the audio we receive usually opens with a wake-word chime / sound
effect plus a beat of silence and ends with trailing silence. Silero VAD is
trained on human speech, so a chime scores low and is trimmed along with the
silence, leaving just the spoken command for the recognizer -- which speeds up
decoding and improves the score (the acoustic model isn't asked to explain the
non-speech padding).

Runnable standalone for a quick check:
    python src/vad.py turn-on-the-office-lamp.wav [trimmed.wav]
"""
import logging
import threading
from typing import Optional

import numpy as np
from pysilero_vad import SileroVoiceActivityDetector

from speech_to_phrase.audio import SAMPLE_RATE

_LOGGER = logging.getLogger(__name__)

# Silero's window is fixed at 512 samples (32 ms @ 16 kHz).
_CHUNK = SileroVoiceActivityDetector.chunk_samples()

# The detector loads a small model and carries per-utterance recurrent state, so
# it is neither free to recreate nor safe to share across threads. Keep one
# instance, reset() it per call, and serialize access (transcribe() and thus
# trimming can run from the Wyoming server's executor pool).
_vad: Optional[SileroVoiceActivityDetector] = None
_vad_lock = threading.Lock()


def _detector() -> SileroVoiceActivityDetector:
    global _vad
    if _vad is None:
        _vad = SileroVoiceActivityDetector()
    return _vad


def trim_silence(
    samples: np.ndarray,
    *,
    threshold: float = 0.5,
    padding_chunks: int = 4,
) -> np.ndarray:
    """Trim leading/trailing non-speech (silence and wake-word sound effects).

    ``samples`` is mono float32 PCM at :data:`SAMPLE_RATE` (roughly [-1, 1], as
    produced by the server's PCM conversion). Returns the sub-array spanning the
    first to last VAD window whose speech probability is ``>= threshold``,
    widened by ``padding_chunks`` windows on each side so onsets/offsets aren't
    clipped. If no speech is found (or the clip is shorter than one window) the
    input is returned unchanged, leaving the recognizer's score gate to reject
    it.
    """
    n_chunks = samples.shape[0] // _CHUNK
    if n_chunks == 0:
        return samples

    first = last = -1
    with _vad_lock:
        vad = _detector()
        vad.reset()
        for i in range(n_chunks):
            chunk = samples[i * _CHUNK : (i + 1) * _CHUNK]
            if vad.process_samples(chunk.tolist()) >= threshold:
                if first < 0:
                    first = i
                last = i

    if first < 0:
        _LOGGER.debug(
            "VAD found no speech in %d samples; passing through", samples.shape[0]
        )
        return samples

    start = max(0, first - padding_chunks) * _CHUNK
    end = min(n_chunks, last + 1 + padding_chunks) * _CHUNK
    _LOGGER.debug(
        "VAD trimmed %d -> %d samples (speech in windows %d..%d of %d)",
        samples.shape[0],
        end - start,
        first,
        last,
        n_chunks,
    )
    return samples[start:end]


def _main() -> None:
    import sys

    import soundfile as sf

    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    logging.basicConfig(level=logging.DEBUG)
    src = sys.argv[1]
    data, sr = sf.read(src, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{src}: expected {SAMPLE_RATE} Hz, got {sr}")

    trimmed = trim_silence(data)
    print(
        f"{src}: {len(data) / sr:.2f}s -> {len(trimmed) / sr:.2f}s "
        f"({len(data) - len(trimmed)} samples removed)"
    )
    if len(sys.argv) > 2:
        sf.write(sys.argv[2], trimmed, SAMPLE_RATE)
        print(f"wrote {sys.argv[2]}")


if __name__ == "__main__":
    _main()
