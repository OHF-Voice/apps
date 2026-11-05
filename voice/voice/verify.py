"""What a verifier is, and which one this app runs.

A verifier answers one question -- was that window really the wake word --
and, when it can, says where the wake phrase ended. Everything else in the
app is transport.

The question is worth asking because the two stages fail differently. The
on-device detector is a small streaming classifier run on every frame, tuned
to fire cheaply; a verifier is told which wake word to look for and gets a
bounded window to look at, so it can afford a decode that the device cannot.

The backend is chosen once at startup and never configured by hand: whichever
one finds its models reports the wake words it can verify, and an app that
finds nothing reports none, which is how Home Assistant is told not to ask.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .protocol import WakeWordInfo, normalize_wake_word

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifyRequest:
    """One detection, as the verifier is asked about it. None of this is on
    the wire: the app assembles it from the run."""

    wake_word_phrase: str
    """Matched loosely, so "okay nabu" and "okay_nabu" are the same."""

    satellite_id: str | None = None
    wake_word_audio_ms: int = 0
    """The detection sits at the end of this much of the window."""


@dataclass(frozen=True)
class Verdict:
    """What the verifier made of the window."""

    accepted: bool

    reason: str | None = None
    """The verifier's own vocabulary ("prefix", "last", "confusable")."""

    wake_word_audio_ms: int | None = None
    """Where the wake phrase really ends, when the verifier aligned it;
    None leaves the boundary where the satellite put it."""

    detail: dict[str, Any] = field(default_factory=dict)
    """Scores and timings for the log. Nothing branches on it."""


@dataclass(frozen=True)
class MoreAudio:
    """The verifier asking to hear further into the run. It never reaches
    Home Assistant: audio is already arriving, so the app waits."""

    ms: int


class Verifier(ABC):
    """A second opinion on a wake word detection."""

    @property
    @abstractmethod
    def wake_words(self) -> list[WakeWordInfo]:
        """The wake words this can verify. Empty: nothing is verified here."""

    def knows(self, wake_word: str) -> bool:
        """Whether this verifier has a model for this wake word.

        Home Assistant names it by the phrase said; the configuration names
        it by an id. Both spellings resolve to the same wake word.
        """
        wanted = normalize_wake_word(wake_word)
        return any(normalize_wake_word(w.id) == wanted for w in self.wake_words)

    async def start(self) -> None:
        """Load whatever has to be resident. Raises to refuse to run."""

    async def stop(self) -> None:
        """Release it again."""

    @abstractmethod
    async def verify(
        self, request: VerifyRequest, pcm: bytes, final: bool
    ) -> Verdict | MoreAudio:
        """Judge one window of 16 kHz 16-bit mono PCM.

        The detection sits ``request.wake_word_audio_ms`` into the window;
        anything after that is command audio Home Assistant was asked for.

        ``MoreAudio`` asks to be shown further into the run and offered the
        whole window again. ``final`` means none is coming, so a verdict
        must come back.
        """


class NoVerifier(Verifier):
    """The app with nothing installed to verify with.

    It reports no wake words rather than accepting everything, so that a
    missing model looks like a missing model instead of like a verifier that
    never refuses anything.
    """

    @property
    def wake_words(self) -> list[WakeWordInfo]:
        return []

    async def verify(
        self, request: VerifyRequest, pcm: bytes, final: bool
    ) -> Verdict | MoreAudio:
        return Verdict(accepted=True, reason="no_verifier")


SPLIT_NONE = "none"
SPLIT_KALDI = "kaldi"
SPLITS = (SPLIT_NONE, SPLIT_KALDI)
"""Which boundary a verdict carries back as ``wake_word_audio_ms``.

``none`` keeps the satellite's own number, which is what Home Assistant
cuts at today. ``kaldi`` sends where the decode aligned the end of the wake
phrase; it is a real alignment, but the experimental app measured it landing
a median 0.3-0.4 s late against a CTC placement of the same audio, late
enough to take the first word of the command with it. There will be a
better one to choose when this app recognizes commands itself.
"""


def load_verifier(
    config_path: Path | None,
    train_dir: Path | None = None,
    split: str = SPLIT_NONE,
) -> Verifier:
    """The verifier this install can run.

    Kaldi is the only backend today, and it needs a config naming its models
    and the tools to decode with. Nothing here is fatal: an install without
    those runs as an app that verifies nothing, which is a running app Home
    Assistant can discover and ignore rather than a crash loop.
    """
    if config_path is None or not config_path.exists():
        _LOGGER.warning("No wake verifier config (%s): nothing will be verified", config_path)
        return NoVerifier()

    try:
        from .wake_kaldi import KaldiVerifier
    except ImportError as err:
        _LOGGER.warning("Kaldi verifier unavailable (%s): nothing will be verified", err)
        return NoVerifier()

    try:
        return KaldiVerifier(config_path, train_dir=train_dir, split=split)
    except Exception:  # pylint: disable=broad-except
        _LOGGER.exception("Cannot load %s: nothing will be verified", config_path)
        return NoVerifier()
