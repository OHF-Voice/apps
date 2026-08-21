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

Run locally:
    python src/wyoming_server.py --uri tcp://0.0.0.0:10300 \
        --backend citrinet --model <model_dir> --grammar ./data/en/grammar.fst
"""
import argparse
import asyncio
import logging
import math
from functools import lru_cache, partial
from pathlib import Path
from typing import Optional

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from speech_to_phrase import load_recognizer
from speech_to_phrase.audio import SAMPLE_RATE

import models
import settings
from vad import normalize_level, trim_silence

_LOGGER = logging.getLogger("wyoming-speech-to-phrase")
NAME = "speech-to-phrase"
ADDON_ROOT = Path(__file__).resolve().parent.parent


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

    def __init__(self, backend: str, model_dir: Path, language: str,
                 grammar_path: Path, default_max_score: float,
                 beam: Optional[float] = None, token_bonus: float = 0.0):
        self._rec = load_recognizer(backend, model_dir, language=language,
                                    beam=beam, token_bonus=token_bonus)
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

    async def transcribe(self, samples: np.ndarray):
        return await asyncio.get_event_loop().run_in_executor(
            None, self._rec.transcribe, samples
        )


def _pcm_to_float(audio: bytes, rate: int, width: int, channels: int) -> np.ndarray:
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    data = np.frombuffer(audio, dtype=dtype).astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        data /= float(np.iinfo(dtype).max + 1)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        from librosa import resample

        data = resample(data, orig_sr=rate, target_sr=SAMPLE_RATE)
    return data


class S2PEventHandler(AsyncEventHandler):
    def __init__(self, reader, writer, *, holder: GrammarHolder, info: Info):
        super().__init__(reader, writer)
        self._holder = holder
        self._info = info
        self._buf = bytearray()
        self._rate = SAMPLE_RATE
        self._width = 2
        self._channels = 1

    async def handle_event(self, event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True

        if Transcribe.is_type(event.type):
            return True  # language selection could be honored here

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self._buf = bytearray()
            self._rate, self._width, self._channels = (
                start.rate, start.width, start.channels
            )
            await self._holder.maybe_reload()
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._buf += chunk.audio
            self._rate, self._width, self._channels = (
                chunk.rate, chunk.width, chunk.channels
            )
            return True

        if AudioStop.is_type(event.type):
            text = ""
            if self._holder.ready and self._buf:
                samples = _pcm_to_float(
                    bytes(self._buf), self._rate, self._width, self._channels
                )
                # Boost very quiet mic audio to a nominal level before VAD + STT
                # (both under-perform on ~-46 dBFS input); no-op for normal
                # levels. Then drop the leading wake-word chime + silence and
                # trailing silence so the recognizer only decodes the command.
                samples = normalize_level(samples)
                samples = await asyncio.get_event_loop().run_in_executor(
                    None, trim_silence, samples
                )
                result = await self._holder.transcribe(samples)
                if result.score <= self._holder.max_score and result.score != math.inf:
                    text = result.text
                    _LOGGER.debug("matched (score=%.3f): %r", result.score, result.text)
                else:
                    _LOGGER.debug("gated (score=%.3f): %r", result.score, result.text)
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
                attribution=Attribution(name="OHF Voice", url="https://openhomefoundation.org"),
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
            )
        ]
    )


async def serve(uri: str, backend: str, model_dir, language: str,
                grammar_path, max_score: float, token_bonus: float = 0.0) -> None:
    """Run the Wyoming server against an already-resolved model directory."""
    model_dir = Path(model_dir)
    holder = GrammarHolder(backend, model_dir, language, Path(grammar_path),
                           default_max_score=max_score, token_bonus=token_bonus)
    await holder.maybe_reload()
    info = build_info(language, model_dir.name)
    server = AsyncServer.from_uri(uri)
    _LOGGER.info("Wyoming server ready on %s (grammar=%s, ready=%s, max_score=%s, "
                 "token_bonus=%s)",
                 uri, grammar_path, holder.ready, holder.max_score, token_bonus)
    await server.run(
        partial(S2PEventHandler, holder=holder, info=info)
    )


def start_background(uri: str, backend: str, model_dir, language: str,
                     grammar_path, max_score: float,
                     token_bonus: float = 0.0) -> "threading.Thread":
    """Run serve() in a daemon thread with its own asyncio loop, so it can sit
    alongside a blocking server (e.g. Flask) in the same process."""
    import threading

    def _runner():
        asyncio.run(serve(uri, backend, model_dir, language, grammar_path,
                          max_score, token_bonus))

    t = threading.Thread(target=_runner, name="wyoming", daemon=True)
    t.start()
    return t


async def run(cfg) -> None:
    model_dir = models.resolve(cfg.model, Path(cfg.models_dir), cfg.language, cfg.backend)
    if model_dir is None:
        raise SystemExit("No acoustic model: pass --model or add a MODEL_NAMES entry")
    await serve(cfg.uri, cfg.backend, model_dir, cfg.language, cfg.grammar,
                cfg.max_score, cfg.token_bonus)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="tcp://0.0.0.0:10300")
    ap.add_argument("--backend", default="citrinet")
    ap.add_argument("--model", default=None,
                    help="model dir (dev) or HuggingFace model name; "
                         "if unset, derived from language+backend")
    ap.add_argument("--models-dir", default="/data/models")
    ap.add_argument("--language", default="en")
    ap.add_argument("--grammar", required=True, help="path to grammar.fst")
    ap.add_argument("--max-score", type=float, default=None,
                    help="score gate; if unset, a per-backend default is used "
                         "(citrinet 5.0, coqui 2.0)")
    ap.add_argument("--token-bonus", type=float, default=None,
                    help="word-insertion reward per emitted token (0 = off); "
                         "if unset, a per-backend default is used "
                         "(citrinet 2.0, coqui 0.0). Counters the CTC length "
                         "bias that lets a short parse win over a longer, "
                         "better-fitting one")
    ap.add_argument("--debug", action="store_true")
    cfg = ap.parse_args()
    if cfg.backend == "auto":
        # Same per-language selection app.py uses, so a coqui-only language
        # (sl/nl/cs) picks coqui here too instead of failing to find a model.
        cfg.backend = models.resolve_backend(cfg.language, "auto")
    if cfg.max_score is None:
        cfg.max_score = models.default_max_score(cfg.backend)
    if cfg.token_bonus is None:
        cfg.token_bonus = models.default_token_bonus(cfg.backend)
    logging.basicConfig(level=logging.DEBUG if cfg.debug else logging.INFO)
    logging.getLogger("numba").setLevel(logging.INFO)  # silence librosa's JIT traces
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
