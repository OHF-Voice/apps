#!/usr/bin/env python3
"""Wyoming speech-to-text server for Speech-to-Phrase.

Loads the acoustic model once and the grammar compiled by the web UI / training
pipeline (``<data>/<lang>/grammar.fst``), decodes incoming audio against it, and
returns a Transcript. The grammar is **hot-reloaded** when the file changes, so
saving in the web UI takes effect without restarting the server.

Score gating: the recognition library returns a per-token ``score`` (lower = more
confident). If it exceeds ``--max-score`` the utterance is out-of-grammar /
low-confidence, so we return an EMPTY transcript — Home Assistant then treats it
as a failed local recognition and can fall back (e.g. to cloud STT) instead of
acting on a guessed command.

Debug mode (toggled in the web UI, checked every utterance): every recognition is
logged for the UI to display — text, score, and whether the gate accepted it —
and Home Assistant is sent an empty transcript regardless. Tuning the gate means
speaking commands that *should* be rejected, and you do not want the ones that
pass to be acted on while you do it. It is in-memory only (see ``debug_log``), so
a restart always comes back with it off.

Run locally:
    python src/wyoming_server.py --uri tcp://0.0.0.0:10300 \
        --backend nemo --model <model_dir> --grammar ./data/en/grammar.fst
"""

import argparse
import asyncio
import logging
import math
import threading
import time
from asyncio import StreamReader, StreamWriter
from functools import lru_cache, partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from speech_to_phrase import Result, load_recognizer
from speech_to_phrase.audio import SAMPLE_RATE
from speech_to_phrase.audio import resample as resample_audio
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

import debug_log
import models
import settings
from audio_frontend import prepare_audio

_LOGGER = logging.getLogger("wyoming-speech-to-phrase")
NAME = "speech-to-phrase"
ADDON_ROOT = Path(__file__).resolve().parent.parent

# Ceiling on one buffered utterance. A command is a couple of seconds long, so
# this is not a limit anyone speaking to their house can reach; it exists
# because the buffer grows on every AudioChunk and only AudioStop empties it.
# A satellite that stops sending AudioStop -- crashed, wedged, or simply
# streaming an open microphone -- would otherwise grow it until the add-on is
# killed for using too much memory. Past the cap the tail is dropped and the
# head kept: the command follows the wake word, so the start is the part worth
# decoding, and an over-long capture decodes to something the score gate
# rejects anyway.
MAX_UTTERANCE_SECONDS = 30.0
# ...and an absolute byte ceiling, because rate/width/channels are whatever the
# client announced: a bogus 10 MHz sample rate would make a duration-derived
# limit meaninglessly large. 16 MiB is ~8 minutes of 16 kHz 16-bit mono.
MAX_UTTERANCE_BYTES = 16 * 1024 * 1024


def _buffer_limit(rate: int, width: int, channels: int) -> int:
    """Byte budget for the audio buffer at the announced format."""
    frame = max(1, width * channels)
    return min(int(MAX_UTTERANCE_SECONDS * max(rate, 1) * frame), MAX_UTTERANCE_BYTES)


@lru_cache(maxsize=1)
def addon_version() -> str:
    """Version to advertise over Wyoming, read from the add-on manifest rather
    than duplicated here -- a second copy only ever drifts from the one
    Supervisor actually installed."""
    try:
        import yaml

        return str(yaml.safe_load((ADDON_ROOT / "config.yaml").read_text())["version"])
    except Exception:  # noqa: BLE001 (running from a checkout, or no manifest)
        _LOGGER.debug("could not read version from config.yaml", exc_info=True)
        return "0.0.0"


class GrammarHolder:
    """Owns the Recognizer (acoustic model + grammar) and hot-reloads the
    grammar file when its mtime changes. Also tracks the per-language score gate
    (``max_score``), re-read from ``<lang>/settings.json`` so edits in the web UI
    take effect without a restart."""

    def __init__(
        self,
        backend: str,
        model_dir: Path,
        language: str,
        grammar_path: Path,
        default_max_score: float,
        beam: Optional[float] = None,
        token_bonus: float = 0.0,
    ) -> None:
        self._rec = load_recognizer(
            backend, model_dir, language=language, beam=beam, token_bonus=token_bonus
        )
        self.language = language
        self._grammar_path = grammar_path
        self._settings_path = Path(grammar_path).parent / settings.FILENAME
        self._default_max_score = default_max_score
        self.max_score = default_max_score
        self._mtime: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._rec.grammar is not None

    @property
    def debug_mode(self) -> bool:
        """Session state, not a setting: nothing is persisted, so a restart
        always comes back with it off (see debug_log)."""
        return debug_log.enabled()

    async def maybe_reload(self) -> None:
        async with self._lock:
            # Cheap: re-read the gate every utterance so UI saves apply at once.
            self.max_score = settings.read_max_score_file(
                self._settings_path, self._default_max_score
            )
            if not self._grammar_path.exists():
                return
            mtime = self._grammar_path.stat().st_mtime
            if mtime != self._mtime:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._rec.load, self._grammar_path
                )
                self._mtime = mtime
                _LOGGER.info("Loaded grammar %s", self._grammar_path)

    async def transcribe(self, samples: NDArray[np.float32]) -> Result:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._rec.transcribe, samples
        )


def _pcm_to_float(
    audio: bytes, rate: int, width: int, channels: int
) -> NDArray[np.float32]:
    dtype = np.dtype({1: "int8", 4: "int32"}.get(width, "int16"))
    data = np.frombuffer(audio, dtype=dtype).astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        data /= float(np.iinfo(dtype).max + 1)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    data = resample_audio(data, rate, SAMPLE_RATE)
    return data


class S2PEventHandler(AsyncEventHandler):
    def __init__(
        self,
        reader: StreamReader,
        writer: StreamWriter,
        *,
        holder: GrammarHolder,
        info: Info,
    ) -> None:
        super().__init__(reader, writer)
        self._holder = holder
        self._info = info
        self._buf = bytearray()
        self._rate = SAMPLE_RATE
        self._width = 2
        self._channels = 1
        self._full = False  # hit MAX_UTTERANCE_SECONDS; warned once already

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True

        if Transcribe.is_type(event.type):
            return True  # language selection could be honored here

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self._buf = bytearray()
            self._full = False
            self._rate, self._width, self._channels = (
                start.rate,
                start.width,
                start.channels,
            )
            await self._holder.maybe_reload()
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._rate, self._width, self._channels = (
                chunk.rate,
                chunk.width,
                chunk.channels,
            )
            limit = _buffer_limit(chunk.rate, chunk.width, chunk.channels)
            if len(self._buf) >= limit:
                if not self._full:
                    self._full = True
                    _LOGGER.warning(
                        "Utterance exceeded %.0fs (%d bytes); ignoring the rest. "
                        "The satellite may not be sending audio-stop.",
                        MAX_UTTERANCE_SECONDS,
                        limit,
                    )
                return True
            self._buf += chunk.audio
            return True

        if AudioStop.is_type(event.type):
            text = ""
            if self._holder.ready and self._buf:
                # Everything from here to the transcript is what Home Assistant
                # waits on: the audio has stopped, so this is dead air in the
                # conversation. Timed as one number (conversion, front-end and
                # decode) because that is the latency a user perceives, and
                # surfaced in debug mode -- a slow decode and a mis-decode look
                # the same from the outside otherwise.
                started = time.monotonic()
                samples = _pcm_to_float(
                    bytes(self._buf), self._rate, self._width, self._channels
                )
                samples = prepare_audio(samples)
                result = await self._holder.transcribe(samples)
                processing = time.monotonic() - started
                accepted = (
                    result.score <= self._holder.max_score and result.score != math.inf
                )
                if accepted:
                    text = result.text
                    _LOGGER.debug(
                        "matched (score=%.3f, %.2fs): %r",
                        result.score,
                        processing,
                        result.text,
                    )
                else:
                    _LOGGER.debug(
                        "gated (score=%.3f, %.2fs): %r",
                        result.score,
                        processing,
                        result.text,
                    )
                if self._holder.debug_mode:
                    debug_log.record(
                        language=self._holder.language,
                        text=result.text,
                        score=result.score,
                        margin=result.margin,
                        accepted=accepted,
                        max_score=self._holder.max_score,
                        duration=len(samples) / SAMPLE_RATE,
                        processing=processing,
                    )
                    # Debug mode observes; it does not act. Handing HA a
                    # transcript here would run the command being diagnosed.
                    text = ""
            await self.write_event(
                Transcript(text=text, language=self._holder.language).event()
            )
            return True

        return True


def build_info(language: str, model_name: str) -> Info:
    version = addon_version()
    return Info(
        asr=[
            AsrProgram(
                name=NAME,
                description="Constrained speech-to-text",
                installed=True,
                version=version,
                attribution=Attribution(
                    name="OHF Voice", url="https://openhomefoundation.org"
                ),
                models=[
                    AsrModel(
                        name=model_name,
                        installed=True,
                        description=model_name,
                        version=version,
                        attribution=Attribution(name="", url=""),
                        languages=[language],
                    )
                ],
                prefers_auto_gain_enabled=True,
                prefers_noise_reduction_enabled=False,
            )
        ]
    )


async def serve(
    uri: str,
    backend: str,
    model_dir: Union[str, Path],
    language: str,
    grammar_path: Union[str, Path],
    max_score: float,
    token_bonus: float = 0.0,
) -> None:
    """Run the Wyoming server against an already-resolved model directory."""
    model_dir = Path(model_dir)
    holder = GrammarHolder(
        backend,
        model_dir,
        language,
        Path(grammar_path),
        default_max_score=max_score,
        token_bonus=token_bonus,
    )
    await holder.maybe_reload()
    info = build_info(language, model_dir.name)
    server = AsyncServer.from_uri(uri)
    _LOGGER.info(
        "Wyoming server ready on %s (grammar=%s, ready=%s, max_score=%s, "
        "token_bonus=%s)",
        uri,
        grammar_path,
        holder.ready,
        holder.max_score,
        token_bonus,
    )
    await server.run(partial(S2PEventHandler, holder=holder, info=info))


def start_background(
    uri: str,
    backend: str,
    model_dir: Union[str, Path],
    language: str,
    grammar_path: Union[str, Path],
    max_score: float,
    token_bonus: float = 0.0,
) -> "threading.Thread":
    """Run serve() in a daemon thread with its own asyncio loop, so it can sit
    alongside a blocking server (e.g. Flask) in the same process."""

    def _runner() -> None:
        asyncio.run(
            serve(
                uri, backend, model_dir, language, grammar_path, max_score, token_bonus
            )
        )

    t = threading.Thread(target=_runner, name="wyoming", daemon=True)
    t.start()
    return t


async def run(cfg: argparse.Namespace) -> None:
    model_dir = models.resolve(
        cfg.model, Path(cfg.models_dir), cfg.language, cfg.backend
    )
    if model_dir is None:
        raise SystemExit("No acoustic model: pass --model or add a MODEL_NAMES entry")
    await serve(
        cfg.uri,
        cfg.backend,
        model_dir,
        cfg.language,
        cfg.grammar,
        cfg.max_score,
        cfg.token_bonus,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="tcp://0.0.0.0:10300")
    ap.add_argument("--backend", default="nemo")
    ap.add_argument(
        "--model",
        default=None,
        help="model dir (dev) or HuggingFace model name; "
        "if unset, derived from language+backend",
    )
    ap.add_argument("--models-dir", default="/data/models")
    ap.add_argument("--language", default="en")
    ap.add_argument("--grammar", required=True, help="path to grammar.fst")
    ap.add_argument(
        "--max-score",
        type=float,
        default=None,
        help="score gate; if unset, a model/backend default is used "
        "(English Parakeet 3.8, other NeMo CTC 5.0, Coqui 2.0)",
    )
    ap.add_argument(
        "--token-bonus",
        type=float,
        default=None,
        help="word-insertion reward per emitted token (0 = off); "
        "if unset, a per-backend default is used "
        "(nemo 2.0, coqui 0.0). Counters the CTC length "
        "bias that lets a short parse win over a longer, "
        "better-fitting one",
    )
    ap.add_argument("--debug", action="store_true")
    cfg = ap.parse_args()
    # Same per-language selection app.py uses, including the legacy Citrinet
    # spelling and Coqui-only languages such as Czech.
    cfg.backend = models.resolve_backend(cfg.language, cfg.backend)
    if cfg.max_score is None:
        configured_model = cfg.model or models.model_name_for(cfg.language, cfg.backend)
        cfg.max_score = models.default_max_score(cfg.backend, configured_model)
    if cfg.token_bonus is None:
        cfg.token_bonus = models.default_token_bonus(cfg.backend)
    logging.basicConfig(level=logging.DEBUG if cfg.debug else logging.INFO)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
