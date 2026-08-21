"""Log-mel feature extraction for Citrinet, matching NeMo's
``AudioToMelSpectrogramPreprocessor`` (the mel preprocessor is not part of the
exported ONNX graph, so we reproduce it here).

Defaults match NeMo's Citrinet config: 16 kHz, 80 mel bins, 25 ms window /
10 ms stride, ``n_fft=512``, preemphasis 0.97, power spectrum, natural log with a
small guard, and ``per_feature`` normalization. NeMo builds its mel filterbank
with ``librosa.filters.mel``; we use the same call for parity.
"""

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class MelFeaturizer:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 512
    win_length: int = 400  # 0.025 s
    hop_length: int = 160  # 0.010 s
    preemph: float = 0.97
    mag_power: float = 2.0
    log_zero_guard: float = 2.0**-24
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        # Slaney-normalized HTK=False mel filterbank, as NeMo uses.
        self._mel_fb = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=0.0,
            fmax=self.sample_rate / 2.0,
            norm="slaney",
            htk=False,
        ).astype(np.float32)
        self._window = np.hanning(self.win_length + 1)[:-1].astype(np.float32)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Return log-mel features of shape ``[n_mels, T]`` (float32)."""
        x = np.asarray(audio, dtype=np.float32).reshape(-1)

        # Preemphasis (matches NeMo: keep first sample, high-pass the rest).
        if self.preemph is not None and self.preemph != 0.0:
            x = np.concatenate(([x[0]], x[1:] - self.preemph * x[:-1])).astype(
                np.float32
            )

        # Short-time Fourier transform -> power spectrum.
        stft = librosa.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window,
            center=True,
            pad_mode="reflect",
        )
        power = np.abs(stft) ** self.mag_power  # [n_fft/2+1, T]

        mel = self._mel_fb @ power  # [n_mels, T]
        feats = np.log(mel + self.log_zero_guard)

        # Per-feature normalization: zero mean / unit std over time, per mel bin.
        mean = feats.mean(axis=1, keepdims=True)
        std = feats.std(axis=1, ddof=1, keepdims=True)
        feats = (feats - mean) / (std + self.norm_eps)

        return feats.astype(np.float32)
