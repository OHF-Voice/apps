"""Production audio preparation for externally bounded Wyoming utterances."""

import numpy as np


def prepare_audio(samples: np.ndarray) -> np.ndarray:
    """Keep the complete utterance at its captured level.

    Home Assistant performs endpointing before sending ``AudioStop``. Running a
    second VAD here clipped quiet or noise-suppressed words, while peak-based gain
    amplified VPE processing artifacts. Citrinet already normalizes every mel
    feature, so the lossless front-end is the correct input.
    """
    return np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
