"""Finding the end of a command: a voice activity detector's probabilities,
with hysteresis.

Silero VAD (pysilero-vad) and microVAD (pymicro-vad) run on the same audio
and the louder of the two is believed, because each one is deaf to things
the other hears. Silero misses some voices: a child's commands, loud and
clear to speech-to-text, were 0.3 s of speech to it and left the command
unstarted for 15 s, where microVAD heard a third of them as speech. microVAD
in turn hears more of what is not speech.

Each chunk of audio is speech or silence. Speech begins at
``speech_threshold`` and lasts until the probability falls below
``silence_threshold``; between the two, a chunk is whatever the one before
it was. The command has begun after ``speech_seconds`` of speech -- until
then, silence ends nothing -- and ends after ``silence_seconds`` of silence.
Speech during that silence starts it over only after ``resume_seconds`` in a
row, so a cough or a click pauses the count rather than resetting it. With no
speech at all ``no_speech_seconds`` after the wake word, nothing was said.

Two things come from the wake word audio, which this app has in hand because
Home Assistant streams the whole run:

Silero hears the command louder when the wake word was quiet (``gain``). It
misses speech far below full scale, as a capture without automatic gain
carries it, and leaves the command unstarted. Only the detector's copy is
raised; the audio Home Assistant holds is untouched.

Silero also hears the wake word first (``warm``). Started cold at the split,
it scored a command said straight after the wake word ~0.4, where it had
heard speech all along ~0.8, and the command waited for a second try to
begin. What it carries over lasts ~0.15 s, so its first ``SETTLE_SECONDS``
after the split count for nothing, lest a pause after the wake word end a
command not yet said.

This is a subset of the endpointer in the experimental Voice Labs app, with
the speaker-identification parts left out. The numbers are its measurements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .protocol import BYTE_RATE, SAMPLE_WIDTH

_LOGGER = logging.getLogger(__name__)

VAD_LEVEL_DBFS = -30.0
"""Where Silero hears the wake word's loudest moment. Gain only ever raises."""

MAX_VAD_GAIN = 16.0

SETTLE_SECONDS = 0.2
"""After a warm start, the detector's chunks that count for nothing."""

SILENCE_SECONDS_OFFSET = 0.3
"""Added to the silence Home Assistant asks for.

Home Assistant's number is calibrated against its own segmenter, which reads
the Speex-enhanced audio's speech probabilities, and 0.7 s is what it calls
default. Against Silero and microVAD together the same wait is 1.0 s: their
probabilities are smoother and fall about half a second after the speech
does, so the same number cuts people off mid-sentence. The user's three
choices keep their spacing -- aggressive stays as far below default as it
was -- which is the part that has to survive, since the number is a promise
about how long someone may pause and not a tuning of this detector.
"""

MIN_SILENCE_SECONDS = 0.3
MAX_SILENCE_SECONDS = 5.0


@dataclass(frozen=True)
class EndOfSpeech:
    """How the end of a command is found. Nothing here is user-facing: the
    one number a person chose is their voice activity sensitivity, which
    arrives as ``silence_seconds``."""

    speech_threshold: float = 0.5
    """A chunk this likely to be speech, or more, is speech."""

    silence_threshold: float = 0.35
    """Below this, silence; between the two, whatever the chunk before was.
    Silero's own default (its threshold less 0.15)."""

    speech_seconds: float = 0.2
    """Speech before the command has begun."""

    silence_seconds: float = 1.0
    """Silence after it that ends it."""

    resume_seconds: float = 0.1
    """Speech in a row that starts the silence over."""

    max_seconds: float = 15.0
    """The longest a command may be, from the wake word. The same ceiling
    Home Assistant's own segmenter uses."""

    no_speech_seconds: float = 8.0
    """With no speech by then, nothing was said."""


def for_silence_seconds(silence_seconds: float) -> EndOfSpeech:
    """Settings for the silence Home Assistant asked for."""
    return EndOfSpeech(
        silence_seconds=min(
            MAX_SILENCE_SECONDS,
            max(MIN_SILENCE_SECONDS, silence_seconds + SILENCE_SECONDS_OFFSET),
        )
    )


_FRAME = 512


def vad_gain(wake: bytes) -> float:
    """The gain Silero hears the command at.

    The loudest 32 ms of ``wake`` -- the audio before the split -- brought up
    to ``VAD_LEVEL_DBFS``, at least 1 and at most ``MAX_VAD_GAIN``.
    """
    usable = len(wake) // (_FRAME * SAMPLE_WIDTH) * _FRAME * SAMPLE_WIDTH
    samples = np.frombuffer(wake[:usable], dtype="<i2")
    if not samples.size:
        return 1.0
    frames = samples.astype(np.float64).reshape(-1, _FRAME) / 32768.0
    loudest = float(np.sqrt((frames**2).mean(axis=1)).max())
    if loudest <= 0.0:
        return MAX_VAD_GAIN
    return float(np.clip(10 ** (VAD_LEVEL_DBFS / 20) / loudest, 1.0, MAX_VAD_GAIN))


def _amplify(chunk: bytes, gain: float) -> bytes:
    samples = np.frombuffer(chunk, dtype="<i2").astype(np.float64) * gain
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


class Detector:
    """A voice activity detector over fixed-size chunks: each one's
    probability of speech, or None while it has nothing to say yet."""

    chunk_bytes: int

    def __call__(self, chunk: bytes) -> float | None:
        raise NotImplementedError


class SileroDetector(Detector):
    """pysilero-vad: 512 samples (32 ms) a chunk."""

    chunk_bytes = 512 * SAMPLE_WIDTH

    def __init__(self) -> None:
        from pysilero_vad import SileroVoiceActivityDetector

        self._vad = SileroVoiceActivityDetector()

    def __call__(self, chunk: bytes) -> float | None:
        return self._vad.process_chunk(chunk)


class MicroDetector(Detector):
    """pymicro-vad (microWakeWord's VAD): 160 samples (10 ms) a chunk.

    It says nothing for its first 0.76 s, while its window fills.
    """

    chunk_bytes = 160 * SAMPLE_WIDTH

    def __init__(self) -> None:
        from pymicro_vad import MicroVad

        self._vad = MicroVad()

    def __call__(self, chunk: bytes) -> float | None:
        probability = self._vad.process_10ms(chunk)
        return None if probability < 0 else probability


class BothDetector(Detector):
    """Silero and microVAD on the same audio, in Silero's chunks: the likelier
    of the two (microVAD's likeliest 10 ms within the chunk).

    Silero hears it raised by ``gain``, microVAD as it came. Until microVAD's
    window has filled, Silero alone.
    """

    chunk_bytes = SileroDetector.chunk_bytes

    def __init__(self, gain: float = 1.0) -> None:
        self.micro = MicroDetector()
        self.silero = SileroDetector()
        self.gain = gain
        self._leftover = b""

    def __call__(self, chunk: bytes) -> float | None:
        silero = self.silero(_amplify(chunk, self.gain) if self.gain != 1.0 else chunk)
        self._leftover += chunk
        size, micro = self.micro.chunk_bytes, []
        while len(self._leftover) >= size:
            probability = self.micro(self._leftover[:size])
            self._leftover = self._leftover[size:]
            if probability is not None:
                micro.append(probability)
        if silero is not None:
            return max([silero, *micro])
        return max(micro) if micro else None


def make_detector(gain: float = 1.0) -> Detector:
    """Both detectors, or Silero alone when microVAD is not installed."""
    try:
        return BothDetector(gain)
    except ImportError:
        _LOGGER.warning("pymicro-vad is not installed; using Silero VAD alone")
        return SileroDetector()


class Endpointer:
    """Marks the start and the end of the command."""

    def __init__(
        self,
        settings: EndOfSpeech | None = None,
        gain: float = 1.0,
        warm: bytes = b"",
        detector: Detector | None = None,
    ) -> None:
        """``warm`` is the audio before the split, and ``gain`` what Silero
        hears the command raised by -- both of them the wake word's."""
        self.settings = settings or EndOfSpeech()
        self._detector = detector if detector is not None else make_detector(gain)
        self.gain = gain
        if isinstance(self._detector, BothDetector):
            # It raises Silero's copy itself; microVAD hears the audio as it came.
            self._detector.gain = gain
            self._raise = False
        else:
            self._raise = gain != 1.0 and isinstance(self._detector, SileroDetector)
        self._chunk_seconds = self._detector.chunk_bytes / BYTE_RATE
        self._leftover = b""
        self._run = 0.0
        """Speech in a row, since the last silence."""
        self._settle = 0
        self.speaking = False
        self.speech_seconds = 0.0
        self.silence_seconds = 0.0
        self.elapsed = 0.0
        """Seconds of command audio judged."""
        self.speech_started_at: float | None = None
        """When the command began, in seconds from the split."""
        self.ended_at: float | None = None
        self.reason: str | None = None
        """Why it ended: "silence", "no_speech" or "max_seconds"."""
        if warm:
            self._warm(warm)

    @property
    def ended(self) -> bool:
        return self.ended_at is not None

    def _warm(self, pcm: bytes) -> None:
        """Silero's state from the audio before the split.

        What it says of that audio, and of the first ``SETTLE_SECONDS``
        after, counts for nothing. With both detectors only Silero hears it:
        microVAD carried the wake word on for over a second, so a pause after
        it was speech that ended, and a command not yet said was over before
        it began. microVAD starts as it does alone, its own window filling.
        """
        both = self._detector if isinstance(self._detector, BothDetector) else None
        silero = both.silero if both is not None else self._detector
        if not isinstance(silero, SileroDetector):
            return
        size = silero.chunk_bytes
        pcm = pcm[len(pcm) % size :]
        for start in range(0, len(pcm), size):
            chunk = pcm[start : start + size]
            silero(_amplify(chunk, self.gain) if self.gain != 1.0 else chunk)
        self._settle = round(SETTLE_SECONDS / self._chunk_seconds)

    def process(self, pcm: bytes) -> bool:
        """Feed command audio; True once the command has ended."""
        self._leftover += pcm
        size = self._detector.chunk_bytes
        seconds = self._chunk_seconds
        settings = self.settings

        while len(self._leftover) >= size:
            chunk = self._leftover[:size]
            self._leftover = self._leftover[size:]
            if self._raise:
                chunk = _amplify(chunk, self.gain)
            probability = self._detector(chunk)
            self.elapsed += seconds

            if (self.speech_started_at is None) and (
                self.elapsed >= settings.no_speech_seconds - 1e-9
            ):
                return self._end("no_speech")
            if self.elapsed >= settings.max_seconds - 1e-9:
                return self._end("max_seconds")
            if probability is None:
                continue
            if self._settle:
                self._settle -= 1
                continue

            self.speaking = probability >= (
                settings.silence_threshold if self.speaking else settings.speech_threshold
            )
            if self.speaking:
                self.speech_seconds += seconds
                self._run += seconds
                if self.speech_started_at is None:
                    if self.speech_seconds >= settings.speech_seconds - 1e-9:
                        self.speech_started_at = self.elapsed
                        _LOGGER.debug("Speech started %.2fs in", self.elapsed)
                elif self._run >= settings.resume_seconds - 1e-9:
                    self.silence_seconds = 0.0
            else:
                self._run = 0.0
                if self.speech_started_at is not None:
                    self.silence_seconds += seconds
                    if self.silence_seconds >= settings.silence_seconds - 1e-9:
                        return self._end("silence")
        return False

    def _end(self, reason: str) -> bool:
        self.ended_at = self.elapsed
        self.reason = reason
        _LOGGER.debug(
            "Command ended after %.2fs (%s: %.2fs speech, %.2fs silence)",
            self.elapsed,
            reason,
            self.speech_seconds,
            self.silence_seconds,
        )
        return True
