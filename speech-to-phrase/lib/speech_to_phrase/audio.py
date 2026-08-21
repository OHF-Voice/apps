"""Audio loading helpers. The pipeline operates on 16 kHz mono float32."""

from pathlib import Path
from typing import Union

import numpy as np

SAMPLE_RATE = 16000


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

    if sample_rate != SAMPLE_RATE:
        import librosa

        data = librosa.resample(data, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

    return np.ascontiguousarray(data, dtype=np.float32)
