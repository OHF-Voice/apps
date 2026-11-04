"""The websocket protocol Home Assistant speaks with this app.

A satellite's own wake word detector has already fired by the time Home
Assistant calls here. What it sends is the window the detector fired on --
the audio the device buffered before the detection, which arrives at the head
of the pipeline's audio stream -- and what it wants back is one verdict:
does the command that follows get run, or was that a false wake.

One websocket per verification (``/verify``). Audio frames carry no room for
a request id, so a shared connection would need every chunk tagged; a
connection costs a loopback handshake and nothing else, and a verification
that hangs then takes nothing down with it.

    HA  -> text    {"type": "verify", ...}     VerifyRequest
    HA  -> binary  16 kHz 16-bit mono PCM      the window, in any chunking
    HA  -> text    {"type": "audio-end", ...}  AudioEnd
    app -> text    {"type": "verdict", ...}    Verdict

An app may answer ``audio-end`` with ``{"type": "more-audio", "ms": N}``
instead of a verdict, and Home Assistant sends that much further audio and
another ``audio-end``. This exists because the useful amount of audio after
the detection is not a constant: a verifier that finds the wake phrase
running off the end of its window needs more of it, and one that does not is
measurably *worse* off being handed command audio it has to explain away. So
the window grows only for the detections that turn out to need it, rather
than every detection paying the latency.

Home Assistant refuses after ``MAX_MORE_AUDIO_MS`` in all, and says so by
setting ``final`` on the last ``audio-end``; it also sets it when the
satellite stopped streaming. A verdict has to follow a final one.

What this app can verify is asked for over HTTP instead (``GET /info``), not
on the connection, because Home Assistant needs it before a wake word is
detected -- it is how it decides whether to ask at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

API_VERSION = 1
"""Bumped when a change to the messages here is not backwards compatible.

Home Assistant refuses to talk to an app whose version it does not know, so
that an app from the future fails loudly rather than by rejecting every wake
word it is asked about.
"""

MAX_MORE_AUDIO_MS = 600
"""The most further audio Home Assistant will add to one window, in ms.

This is latency the person waits through with the wake word already spoken,
so it is small on purpose. 300 ms was what the experimental app's retry used.
"""

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
SAMPLE_CHANNELS = 1
BYTE_RATE = SAMPLE_RATE * SAMPLE_WIDTH * SAMPLE_CHANNELS


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

    These are matched against the pipeline's wake word id the way the
    detectors spell it ("okay_nabu"), after normalizing separators and case,
    so an app that knows "okay nabu" also answers for "okay_nabu".
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
    """Empty means "ask me about nothing": an app with no models installed
    says so here rather than rejecting, or blankly accepting, every window."""

    tail_ms: int = 0
    """Audio from *after* the detection to add to the window, in ms.

    The pre-roll ends at the detection, so a detector that fires before the
    speaker has finished leaves the last phoneme outside the window and the
    verifier judges a truncated phrase. The cost is latency: Home Assistant
    holds the run open until this much more audio has arrived.
    """

    api_version: int = API_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "name": self.name,
            "version": self.version,
            "tail_ms": self.tail_ms,
            "wake_words": [w.to_json() for w in self.wake_words],
        }


@dataclass(frozen=True)
class VerifyRequest:
    """The text frame that opens a verification."""

    wake_word_phrase: str
    """What the pipeline calls the wake word: the phrase said.

    It is matched against the configured wake words loosely, so "okay nabu"
    and "okay_nabu" are the same wake word.
    """

    satellite_id: str | None = None
    """The satellite entity that heard it. Nothing uses it yet; it is what
    lets this app answer "several satellites heard one utterance, which one
    takes it" without Home Assistant having to model that."""

    wake_word_audio_ms: int = 0
    """How much of the window the satellite called the wake word.

    The detection sits at the end of it; anything after that is command
    audio Home Assistant was asked for.
    """

    rate: int = SAMPLE_RATE
    width: int = SAMPLE_WIDTH
    channels: int = SAMPLE_CHANNELS

    @staticmethod
    def from_json(data: dict[str, Any]) -> "VerifyRequest":
        return VerifyRequest(
            wake_word_phrase=str(data["wake_word_phrase"]),
            satellite_id=_optional_str(data.get("satellite_id")),
            wake_word_audio_ms=int(data.get("wake_word_audio_ms") or 0),
            rate=int(data.get("rate") or SAMPLE_RATE),
            width=int(data.get("width") or SAMPLE_WIDTH),
            channels=int(data.get("channels") or SAMPLE_CHANNELS),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "verify",
            "wake_word_phrase": self.wake_word_phrase,
            "satellite_id": self.satellite_id,
            "wake_word_audio_ms": self.wake_word_audio_ms,
            "rate": self.rate,
            "width": self.width,
            "channels": self.channels,
        }


@dataclass(frozen=True)
class AudioEnd:
    """"That is the window" -- or, with ``final``, "and there is no more"."""

    final: bool = False

    @staticmethod
    def from_json(data: dict[str, Any]) -> "AudioEnd":
        return AudioEnd(final=bool(data.get("final")))

    def to_json(self) -> dict[str, Any]:
        return {"type": "audio-end", "final": self.final}


@dataclass(frozen=True)
class MoreAudio:
    """The app asking for the window to be extended."""

    ms: int

    @staticmethod
    def from_json(data: dict[str, Any]) -> "MoreAudio":
        return MoreAudio(ms=int(data["ms"]))

    def to_json(self) -> dict[str, Any]:
        return {"type": "more-audio", "ms": self.ms}


@dataclass(frozen=True)
class Verdict:
    """The one message that comes back."""

    accepted: bool

    reason: str | None = None
    """Why it was refused, for logs: the verifier's own vocabulary ("prefix",
    "last", "confusable"), or "error" when the verifier itself failed.
    Nothing in Home Assistant branches on it."""

    wake_word_audio_ms: int | None = None
    """Where the wake phrase really ends, when the verifier aligned it.

    It replaces the length the satellite reported, which is where a
    classifier crossed a threshold -- early for some speakers and late for
    others. None leaves the boundary where it was.
    """

    detail: dict[str, Any] = field(default_factory=dict)
    """Scores and timings. Logged by Home Assistant at debug level only."""

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Verdict":
        split = data.get("wake_word_audio_ms")
        return Verdict(
            accepted=bool(data.get("accepted")),
            reason=_optional_str(data.get("reason")),
            wake_word_audio_ms=None if split is None else int(split),
            detail=dict(data.get("detail") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "verdict",
            "accepted": self.accepted,
            "reason": self.reason,
            "wake_word_audio_ms": self.wake_word_audio_ms,
            "detail": self.detail,
        }


def normalize_wake_word(wake_word: str) -> str:
    """Treat "Okay Nabu", "okay-nabu" and "okay_nabu" as one wake word.

    Pipelines name a wake word by the phrase said and detectors by an id,
    and the difference between the two spellings has never meant anything.
    """
    return "".join(
        c for c in wake_word.lower().replace("-", "_").replace(" ", "_") if c.isalnum() or c == "_"
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
