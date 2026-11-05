"""The messages Home Assistant and this app exchange.

PROTOCOL.md is the agreement; this is it in code. One websocket per run
(``/run``), audio in continuously, events back:

    HA  -> text    {"type": "run-start", ...}     RunStart
    HA  -> binary  16 kHz 16-bit mono PCM         the run's audio
    HA  -> text    {"type": "audio-end"}          the satellite stopped
    app -> text    {"type": "wake-word-end", ...} WakeWordEnd
    app -> text    {"type": "stt-vad-start", ...} SttVadStart
    app -> text    {"type": "stt-vad-end", ...}   SttVadEnd

The events back are Home Assistant's own pipeline events, named and shaped
as it names and shapes them, so that nothing in between has to translate.
The wake word's end is not a new idea needing a new message: the
``wake_word_audio_ms`` in ``wake-word-end`` is already where speech-to-text
starts.

What this app can verify is asked for over HTTP instead (``GET /info``),
because Home Assistant needs it before a wake word is detected -- it is how
it decides whether to ask satellites for wake word audio at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

API_VERSION = 1
"""Bumped when a change to the messages here is not backwards compatible.

Home Assistant refuses to talk to an app whose version it does not know, so
that an app from the future fails loudly rather than by quietly rejecting
every wake word it is asked about.
"""

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
SAMPLE_CHANNELS = 1
BYTE_RATE = SAMPLE_RATE * SAMPLE_WIDTH * SAMPLE_CHANNELS
"""The format, fixed. There is no resampling here, and a format field on the
wire would only suggest otherwise."""

MAX_RUN_SECONDS = 30.0
"""Audio one run will hold before it stops taking more.

Nothing legitimate comes near it. It is here so that a satellite which
streams forever costs a bounded amount of memory, and so that the ceiling is
something Home Assistant was told about rather than something it discovers
as silence.
"""


def ms_to_bytes(ms: float) -> int:
    """Byte offset of a timestamp, rounded down to a whole sample."""
    return int(ms * BYTE_RATE / 1000) // SAMPLE_WIDTH * SAMPLE_WIDTH


def bytes_to_ms(num_bytes: int) -> int:
    return int(num_bytes * 1000 / BYTE_RATE)


@dataclass(frozen=True)
class WakeWordInfo:
    """A wake word this app can verify, as ``GET /info`` reports it."""

    id: str
    """What a detection has to call itself to be verified here.

    Matched against the pipeline's wake word after normalizing separators
    and case, so an app that knows "okay nabu" also answers for "okay_nabu".
    """

    phrase: str | None = None
    """What is said, for logs and for a person reading the app's info."""

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "phrase": self.phrase}


@dataclass(frozen=True)
class AppInfo:
    """``GET /info``: what Home Assistant needs before a wake word fires."""

    name: str
    version: str
    wake_words: list[WakeWordInfo] = field(default_factory=list)
    """Empty means "ask me about nothing". An app with no models installed
    says so here rather than blankly accepting every detection -- and is
    still sent audio, because endpointing does not need a model."""

    max_run_ms: int = int(MAX_RUN_SECONDS * 1000)
    api_version: int = API_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "name": self.name,
            "version": self.version,
            "max_run_ms": self.max_run_ms,
            "wake_words": [w.to_json() for w in self.wake_words],
        }


@dataclass(frozen=True)
class RunStart:
    """The text frame that opens a run."""

    satellite_id: str | None = None
    """The satellite entity whose audio this is. Nothing uses it yet; it is
    what will let this app answer "several satellites heard one utterance,
    which one takes it" without Home Assistant having to model that."""

    wake_word_phrase: str | None = None
    """What the pipeline calls the wake word, or None when the run did not
    start at one: a button press, or a satellite that starts at
    speech-to-text. The app then only endpoints."""

    wake_word_audio_ms: int = 0
    """How much of the audio the satellite called the wake word.

    The detection sits at the end of it; everything after is the command.
    """

    silence_seconds: float | None = None
    """The user's voice activity sensitivity, already resolved to seconds by
    Home Assistant. None means do not endpoint, which is what a run with
    voice activity detection switched off looks like."""

    @staticmethod
    def from_json(data: dict[str, Any]) -> RunStart:
        silence = data.get("silence_seconds")
        return RunStart(
            satellite_id=_optional_str(data.get("satellite_id")),
            wake_word_phrase=_optional_str(data.get("wake_word_phrase")),
            wake_word_audio_ms=int(data.get("wake_word_audio_ms") or 0),
            silence_seconds=None if silence is None else float(silence),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "run-start",
            "satellite_id": self.satellite_id,
            "wake_word_phrase": self.wake_word_phrase,
            "wake_word_audio_ms": self.wake_word_audio_ms,
            "silence_seconds": self.silence_seconds,
        }


@dataclass(frozen=True)
class WakeWordEnd:
    """The detection stands, and this is where the wake word ends."""

    wake_word_phrase: str
    wake_word_audio_ms: int

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "wake-word-end",
            "wake_word_output": {
                "wake_word_phrase": self.wake_word_phrase,
                "wake_word_audio_ms": self.wake_word_audio_ms,
            },
        }


@dataclass(frozen=True)
class WakeWordRejected:
    """That was not the wake word. The run ends here."""

    wake_word_phrase: str
    reason: str | None = None
    """The verifier's own vocabulary ("prefix", "last", "confusable"), for
    logs. Nothing in Home Assistant branches on it."""

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "wake-word-rejected",
            "wake_word_output": {
                "wake_word_phrase": self.wake_word_phrase,
                "reason": self.reason,
            },
        }


@dataclass(frozen=True)
class SttVadStart:
    """Speech began. Satellites animate on this."""

    timestamp: int
    """Milliseconds from the first audio byte of the run."""

    def to_json(self) -> dict[str, Any]:
        return {"type": "stt-vad-start", "timestamp": self.timestamp}


@dataclass(frozen=True)
class SttVadEnd:
    """Speech ended, and so does the audio speech-to-text is given.

    Why it ended -- silence, nothing said at all, or the run reaching
    ``max_run_ms`` -- is in the app's log, not on the wire. Home Assistant is
    told where the boundary is.
    """

    timestamp: int

    def to_json(self) -> dict[str, Any]:
        return {"type": "stt-vad-end", "timestamp": self.timestamp}


@dataclass(frozen=True)
class Error:
    """Something went wrong here. Home Assistant carries on without us."""

    code: str
    message: str

    def to_json(self) -> dict[str, Any]:
        return {"type": "error", "code": self.code, "message": self.message}


def normalize_wake_word(wake_word: str) -> str:
    """Treat "Okay Nabu", "okay-nabu" and "okay_nabu" as one wake word.

    Pipelines name a wake word by the phrase said and detectors by an id,
    and the difference between the two spellings has never meant anything.
    """
    return "".join(
        c
        for c in wake_word.lower().replace("-", "_").replace(" ", "_")
        if c.isalnum() or c == "_"
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
