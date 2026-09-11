"""Optional server-side voice-command endpoint detection."""

from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16000
SAMPLES_PER_VAD_CHUNK = 512
SECONDS_PER_VAD_CHUNK = SAMPLES_PER_VAD_CHUNK / SAMPLE_RATE


class SpeechProbabilityDetector(Protocol):
    """Stateful detector that scores one 16 kHz mono sample window."""

    def reset(self) -> None:
        pass

    def process_samples(self, samples: list[float]) -> float:
        pass


class VoiceCommandSegmenter:
    """Detect the end of speech without treating brief pauses as an endpoint."""

    def __init__(
        self,
        silence_seconds: float,
        *,
        speech_seconds: float = 0.3,
        command_seconds: float = 1.0,
        before_command_speech_threshold: float = 0.2,
        in_command_speech_threshold: float = 0.5,
    ) -> None:
        if silence_seconds <= 0:
            raise ValueError("silence_seconds must be greater than zero")
        if speech_seconds <= 0:
            raise ValueError("speech_seconds must be greater than zero")
        if command_seconds < speech_seconds:
            raise ValueError("command_seconds must be at least speech_seconds")

        self.silence_seconds = silence_seconds
        self.speech_seconds = speech_seconds
        self.command_seconds = command_seconds
        self.before_command_speech_threshold = before_command_speech_threshold
        self.in_command_speech_threshold = in_command_speech_threshold
        self.in_command = False
        self._speech_seconds_left = 0.0
        self._command_seconds_left = 0.0
        self._silence_seconds_left = 0.0
        self.reset()

    def reset(self) -> None:
        """Reset state for a new audio stream."""
        self.in_command = False
        self._speech_seconds_left = self.speech_seconds
        self._command_seconds_left = self.command_seconds - self.speech_seconds
        self._silence_seconds_left = self.silence_seconds

    def process(self, chunk_seconds: float, speech_probability: float) -> bool:
        """Return False when a started voice command has ended."""
        if not self.in_command:
            if speech_probability > self.before_command_speech_threshold:
                self._speech_seconds_left -= chunk_seconds
                if self._speech_seconds_left <= 0:
                    self.in_command = True
            else:
                self._speech_seconds_left = self.speech_seconds
            return True

        self._command_seconds_left -= chunk_seconds
        if speech_probability > self.in_command_speech_threshold:
            self._silence_seconds_left = self.silence_seconds
        else:
            self._silence_seconds_left -= chunk_seconds

        if self._command_seconds_left <= 0 and self._silence_seconds_left <= 0:
            self.reset()
            return False

        return True


class SileroEndpointDetector:
    """Adapt Wyoming PCM chunks to pySilero's fixed-size input windows."""

    def __init__(
        self,
        silence_seconds: float,
        detector: Optional[SpeechProbabilityDetector] = None,
    ) -> None:
        if detector is None:
            from pysilero_vad import (  # type: ignore[import-not-found]
                SileroVoiceActivityDetector,
            )

            detector = SileroVoiceActivityDetector()

        self._vad = detector
        self._segmenter = VoiceCommandSegmenter(silence_seconds)
        self._pending = np.empty(0, dtype=np.float32)
        self._format: Optional[tuple[int, int, int]] = None
        self._resampler = None

    def reset(self, rate: int, width: int, channels: int) -> None:
        """Reset VAD and conversion state for a new Wyoming audio stream."""
        self._validate_format(rate, width, channels)
        self._vad.reset()
        self._segmenter.reset()
        self._pending = np.empty(0, dtype=np.float32)
        self._format = (rate, width, channels)
        self._resampler = None
        if rate != SAMPLE_RATE:
            import soxr  # type: ignore[import-untyped]

            self._resampler = soxr.ResampleStream(
                rate,
                SAMPLE_RATE,
                num_channels=1,
                dtype="float32",
                quality="HQ",
            )

    def process(self, audio: bytes, rate: int, width: int, channels: int) -> bool:
        """Return False when enough post-command silence has been observed."""
        audio_format = (rate, width, channels)
        if self._format != audio_format:
            raise ValueError(
                "audio format changed during stream: "
                f"expected {self._format}, got {audio_format}"
            )

        samples = _decode_pcm(audio, width, channels)
        if self._resampler is not None:
            samples = self._resampler.resample_chunk(samples)
        if not samples.size:
            return True

        self._pending = np.concatenate((self._pending, samples))
        offset = 0
        while (offset + SAMPLES_PER_VAD_CHUNK) <= self._pending.size:
            vad_chunk = self._pending[offset : offset + SAMPLES_PER_VAD_CHUNK]
            probability = float(self._vad.process_samples(vad_chunk.tolist()))
            offset += SAMPLES_PER_VAD_CHUNK
            if not self._segmenter.process(SECONDS_PER_VAD_CHUNK, probability):
                self._pending = np.empty(0, dtype=np.float32)
                return False

        if offset:
            self._pending = self._pending[offset:].copy()
        return True

    @staticmethod
    def _validate_format(rate: int, width: int, channels: int) -> None:
        if rate <= 0:
            raise ValueError("audio sample rate must be greater than zero")
        if width not in (1, 2, 4):
            raise ValueError(f"unsupported audio sample width: {width}")
        if channels <= 0:
            raise ValueError("audio channel count must be greater than zero")


def _decode_pcm(audio: bytes, width: int, channels: int) -> NDArray[np.float32]:
    """Decode integer PCM to mono float32 without changing its sample rate."""
    dtype = np.dtype({1: "int8", 2: "<i2", 4: "<i4"}[width])
    data = np.frombuffer(audio, dtype=dtype)
    if data.size % channels:
        raise ValueError("audio chunk contains a partial frame")

    samples = data.astype(np.float32)
    samples /= float(np.iinfo(dtype).max + 1)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)
