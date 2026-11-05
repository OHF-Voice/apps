"""The messages Home Assistant and this app exchange. PROTOCOL.md is the
agreement; this is it in code.

One websocket per run (``/run``): run-start, audio continuously, audio-end,
with wake-word-end, stt-vad-start and stt-vad-end coming back. Those are
Home Assistant's own pipeline events, named and shaped as it names them, so
nothing in between translates.

``GET /info`` is separate because Home Assistant asks it before a wake word
is detected, to decide whether to ask satellites for wake word audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

API_VERSION = 1
"""Bumped on a breaking change. Home Assistant refuses a version it does not
know, so an app from the future fails loudly rather than by rejecting every
wake word it is asked about."""

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
SAMPLE_CHANNELS = 1
BYTE_RATE = SAMPLE_RATE * SAMPLE_WIDTH * SAMPLE_CHANNELS
"""The format is fixed: no resampling here, and a format field on the wire
would only suggest otherwise."""

MAX_RUN_SECONDS = 30.0
"""Audio one run will hold. Nothing legitimate comes near it; it bounds what
a satellite streaming forever costs, and /info reports it so the ceiling is
not something Home Assistant discovers as silence."""


def ms_to_bytes(ms: float) -> int:
    """Byte offset of a timestamp, rounded down to a whole sample."""
    return int(ms * BYTE_RATE / 1000) // SAMPLE_WIDTH * SAMPLE_WIDTH


def bytes_to_ms(num_bytes: int) -> int:
    return int(num_bytes * 1000 / BYTE_RATE)


@dataclass(frozen=True)
class WakeWordInfo:
    """A wake word this app can verify, as ``GET /info`` reports it."""

    id: str
    """Matched against the pipeline's wake word after normalizing separators
    and case, so "okay nabu" and "okay_nabu" are the same wake word."""

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
    """Empty means "ask me about nothing", rather than blankly accepting
    everything. Audio still arrives: endpointing needs no model."""

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
    """Unused so far: it is what will let this app answer "several
    satellites heard one utterance, which one takes it"."""

    wake_word_phrase: str | None = None
    """None when the run did not start at a wake word -- a button press, or
    a satellite that starts at speech-to-text -- and the app only
    endpoints."""

    wake_word_audio_ms: int = 0
    """The detection sits at the end of this; the rest is the command."""

    silence_seconds: float | None = None
    """The user's voice activity sensitivity, resolved to seconds. None
    means do not endpoint."""

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
    logs only."""

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
    """Speech ended, and so does the audio speech-to-text is given. Why --
    silence, nothing said, or the run's ceiling -- is in the app's log, not
    on the wire."""

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
    """Treat "Okay Nabu", "okay-nabu" and "okay_nabu" as one wake word:
    pipelines name it by the phrase, detectors by an id."""
    return "".join(
        c
        for c in wake_word.lower().replace("-", "_").replace(" ", "_")
        if c.isalnum() or c == "_"
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
