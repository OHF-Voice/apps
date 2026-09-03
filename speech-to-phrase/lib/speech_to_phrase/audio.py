"""Audio loading helpers. The pipeline operates on 16 kHz mono float32."""

from pathlib import Path
from typing import Union

import numpy as np

SAMPLE_RATE = 16000


def resample(data: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample mono float32 audio.

    This is what ``librosa.resample`` does at its default quality: it hands the
    work to soxr and then fixes the length to ``ceil(n * target / orig)``, which
    soxr can be a sample off. Calling soxr directly skips librosa's import,
    which drags in numba and llvmlite.
    """
    if orig_sr == target_sr:
        return data

    import soxr

    resampled = soxr.resample(data, orig_sr, target_sr, quality="HQ")
    expected = int(np.ceil(data.shape[-1] * target_sr / orig_sr))
    if resampled.shape[-1] > expected:
        return resampled[..., :expected]
    if resampled.shape[-1] < expected:
        return np.pad(resampled, (0, expected - resampled.shape[-1]))
    return resampled


def load_audio(audio: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Return mono float32 samples in ``[-1, 1]`` at 16 kHz.

    Accepts an array (assumed already 16 kHz mono float) or a path to any
    soundfile-readable file (resampled / downmixed as needed).
    """
    if isinstance(audio, np.ndarray):
        return np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)

    import soundfile as sf

    data, sample_rate = sf.read(str(audio), dtype="float32", always_2d=True)
    data = data.mean(axis=1)  # downmix to mono

    data = resample(data, sample_rate, SAMPLE_RATE)
    return np.ascontiguousarray(data, dtype=np.float32)
