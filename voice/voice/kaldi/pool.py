"""Resident decoder: one process holding the model and every graph.

The acoustic model, i-vector extractor and the verifier graphs are loaded once
and stay resident, so a verification costs only the decodes instead of a
process spawn plus a model load per graph.

Graphs sharing a process share that process's acoustic pass -- the nnet3
forward pass is identical for all of them, since they score the same audio
against different grammars. Graphs in different processes run concurrently.
`decoder.groups` sets the split; see DEFAULT_GROUPS in config.py for what the
trade costs, measured.

Build the daemon once per machine:

    decoderd/build.sh local/tools

A missing or broken daemon is not an error: the verifier falls back to one-shot
subprocess decodes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from .config import UNCAPPED_GRAPHS, DecoderConfig
from .kaldi import FRAME_SHIFT, BestCost, CtmWord, SpeechTools, read_words_txt
from .model import PreparedModel
from .train import CTM_TAGS, GRAPH_TAGS

_LOGGER = logging.getLogger(__name__)

DEFAULT_DAEMON_BIN = Path(__file__).parent / "decoderd" / "verify_decoderd"

_START_TIMEOUT = 120.0  # model + graph load (slow disk / first run)
_DECODE_TIMEOUT = 30.0

DecodeResult = tuple[BestCost | None, list[CtmWord]]


class _DecoderProcess:
    """One process decoding one group of graphs, sharing one acoustic pass."""

    def __init__(
        self,
        daemon_bin: Path,
        model: PreparedModel,
        tools: SpeechTools,
        decoder: DecoderConfig,
        entries: tuple[tuple[str, str, Path], ...],
    ) -> None:
        """``entries`` is (name the daemon knows it by, graph tag, graph dir).

        The two names differ only when one process holds more than one wake
        word's graphs, where every tag has to be unique across them.
        """
        self.daemon_bin = Path(daemon_bin)
        self.model = model
        self.tools = tools
        self.decoder = decoder
        self.entries = entries
        self.tags = tuple(name for name, _, _ in entries)
        self.proc: asyncio.subprocess.Process | None = None
        self._id_to_word: dict[str, dict[str, str]] = {}
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self) -> None:
        args: list[str] = [
            f"--config={self.model.online_conf}",
            "--acoustic-scale=1.0",
            # One search config for every graph, except that max_active must
            # not reach the confusable graph; the daemon applies the same rule
            # the Python side does.
            *self.decoder.args(GRAPH_TAGS[0]),
            "--frame-subsampling-factor=3",
            # By the names the daemon knows them by: it matches these against
            # the tags it was given, and a confusable graph it fails to
            # recognize here is capped, which silently accepts false wakes.
            "--uncapped-graphs="
            + ",".join(
                sorted(
                    name
                    for name, tag, _ in self.entries
                    if tag in UNCAPPED_GRAPHS
                )
            ),
            str(self.model.final_mdl),
        ]
        for name, tag, root in self.entries:
            graph_dir = root / f"graph_{tag}"
            lang_dir = root / "data" / f"lang_{tag}"
            want_ctm = tag in CTM_TAGS
            args += [
                name,
                str(graph_dir / "HCLG.fst"),
                str(lang_dir / "phones" / "word_boundary.int") if want_ctm else "-",
            ]
            if want_ctm:
                self._id_to_word[name] = read_words_txt(graph_dir / "words.txt")

        proc = await asyncio.create_subprocess_exec(
            str(self.daemon_bin),
            *args,
            env=self.tools.extended_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdout is not None
        try:
            ready = await asyncio.wait_for(
                proc.stdout.readline(), timeout=_START_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise

        if ready.decode(errors="replace").strip() != "READY":
            detail = b""
            if proc.stderr is not None:
                try:
                    detail = await asyncio.wait_for(proc.stderr.read(2000), timeout=2)
                except asyncio.TimeoutError:
                    pass
            proc.kill()
            raise RuntimeError(
                f"decoder failed to start: {detail.decode(errors='replace')[-500:]}"
            )

        self.proc = proc

    async def decode(
        self, wav_path: Path, tags: tuple[str, ...] | None = None
    ) -> dict[str, DecodeResult]:
        """Decode these graphs; the acoustic model runs once for them.

        ``tags`` names the daemon's own tags, and leaving it out asks for
        every graph this process holds. Asking for a subset costs what a
        process holding only those graphs would: the ones left out are never
        started, so they take no acoustic pass and no search.
        """
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("decoder is not running")

        wanted = self.tags if tags is None else tags
        request = str(wav_path) if tags is None else ",".join(tags) + f"\t{wav_path}"

        async with self._lock:
            self.proc.stdin.write(f"{request}\n".encode())
            await self.proc.stdin.drain()
            lines = [
                await asyncio.wait_for(
                    self.proc.stdout.readline(), timeout=_DECODE_TIMEOUT
                )
                for _ in wanted
            ]

        results: dict[str, DecodeResult] = {}
        for line in lines:
            text = line.decode(errors="replace").strip()
            if not text:
                raise RuntimeError("decoder closed unexpectedly")
            parts = text.split()
            if parts[0] != "OK":
                raise RuntimeError(f"decoder: {text[:200]}")

            tag = parts[1]
            cost = (float(parts[2]), int(parts[3]))
            id_to_word = self._id_to_word.get(tag, {})
            words: list[CtmWord] = []
            for token in parts[4:]:
                word_id, beg, end = token.split(":")
                words.append(
                    (
                        id_to_word.get(word_id, word_id),
                        int(beg) * FRAME_SHIFT,
                        int(end) * FRAME_SHIFT,
                    )
                )
            results[tag] = (cost, words)

        missing = set(wanted) - set(results)
        if missing:
            raise RuntimeError(f"decoder did not answer for {sorted(missing)}")
        return results

    async def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.returncode is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.close()
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError, OSError):
                self.proc.kill()
        self.proc = None


class SharedView:
    """One wake word's four graphs inside a process that holds more."""

    def __init__(self, process: "_DecoderProcess", wake_id: str) -> None:
        self._process = process
        self._wake_id = wake_id
        self._names = tuple(f"{wake_id}_{tag}" for tag in GRAPH_TAGS)

    @property
    def running(self) -> bool:
        return self._process.running

    async def decode_all(self, wav_path: Path) -> dict[str, DecodeResult]:
        """Decode this wake word's graphs, and nobody else's."""
        results = await self._process.decode(wav_path, self._names)
        prefix = len(self._wake_id) + 1
        return {name[prefix:]: value for name, value in results.items()}

    async def stop(self) -> None:
        """Nothing: the process is shared, and its owner stops it."""


class SharedDecoder:
    """One process holding every wake word's graphs.

    The acoustic model is what costs memory, and it is identical for every
    wake word, so holding one copy instead of one per wake word is most of
    the app's resident size. A verification still decodes only its own four
    graphs, which the daemon selects per request.
    """

    def __init__(self, process: "_DecoderProcess") -> None:
        self.process = process

    @property
    def running(self) -> bool:
        return self.process.running

    def view(self, wake_id: str) -> SharedView:
        """This wake word's handle on the shared process."""
        return SharedView(self.process, wake_id)

    async def stop(self) -> None:
        await self.process.stop()


async def maybe_start_shared_decoder(
    daemon_bin: Path | None,
    model: PreparedModel,
    graph_dirs: dict[str, Path],
    tools: SpeechTools,
    decoder: DecoderConfig | None = None,
) -> SharedDecoder | None:
    """Start one decoder for every wake word, or None when unavailable."""
    daemon_bin = Path(daemon_bin) if daemon_bin else DEFAULT_DAEMON_BIN
    decoder = decoder or DecoderConfig()
    if not daemon_bin.is_file():
        _LOGGER.info(
            "Resident decoder daemon not built (%s); using subprocess decodes. "
            "Build it with decoderd/build.sh for lower latency.",
            daemon_bin,
        )
        return None

    started = time.monotonic()
    entries = tuple(
        (f"{wake_id}_{tag}", tag, graph_dir)
        for wake_id, graph_dir in graph_dirs.items()
        for tag in GRAPH_TAGS
    )
    process = _DecoderProcess(daemon_bin, model, tools, decoder, entries)
    try:
        await process.start()
    except Exception:
        _LOGGER.exception(
            "Shared resident decoder failed to start; falling back per wake word"
        )
        await process.stop()
        return None

    _LOGGER.info(
        "One resident decoder for %s (%d graphs; beam=%s max-active=%d "
        "except %s; %.1f s)",
        ", ".join(graph_dirs),
        len(entries),
        decoder.beam,
        decoder.max_active,
        "/".join(sorted(UNCAPPED_GRAPHS)),
        time.monotonic() - started,
    )
    return SharedDecoder(process)


class ResidentDecoder:
    """The resident decoders for one wake word, however they are grouped."""

    def __init__(self, processes: list[_DecoderProcess]) -> None:
        self.processes = processes

    @property
    def running(self) -> bool:
        return bool(self.processes) and all(p.running for p in self.processes)

    async def decode_all(self, wav_path: Path) -> dict[str, DecodeResult]:
        """Decode every graph, running the groups concurrently."""
        parts = await asyncio.gather(
            *(process.decode(wav_path) for process in self.processes)
        )
        results: dict[str, DecodeResult] = {}
        for part in parts:
            results.update(part)
        return results

    async def stop(self) -> None:
        await asyncio.gather(
            *(process.stop() for process in self.processes), return_exceptions=True
        )
        self.processes = []


async def maybe_start_decoder(
    daemon_bin: Path | None,
    model: PreparedModel,
    graph_dir: Path,
    tools: SpeechTools,
    decoder: DecoderConfig | None = None,
) -> ResidentDecoder | None:
    """Start the resident decoders, or return None when unavailable."""
    daemon_bin = Path(daemon_bin) if daemon_bin else DEFAULT_DAEMON_BIN
    decoder = decoder or DecoderConfig()
    if not daemon_bin.is_file():
        _LOGGER.info(
            "Resident decoder daemon not built (%s); using subprocess decodes. "
            "Build it with decoderd/build.sh for lower latency.",
            daemon_bin,
        )
        return None

    started = time.monotonic()
    processes = [
        _DecoderProcess(
            daemon_bin,
            model,
            tools,
            decoder,
            tuple((tag, tag, graph_dir) for tag in tags),
        )
        for tags in decoder.groups
    ]
    try:
        await asyncio.gather(*(process.start() for process in processes))
    except Exception:
        _LOGGER.exception(
            "Resident decoders failed to start; falling back to subprocess decodes"
        )
        await asyncio.gather(
            *(process.stop() for process in processes), return_exceptions=True
        )
        return None

    _LOGGER.info(
        "Resident decoders ready for %s (%s; beam=%s max-active=%d except %s; "
        "%.1f s)",
        graph_dir.name,
        " | ".join("+".join(tags) for tags in decoder.groups),
        decoder.beam,
        decoder.max_active,
        "/".join(sorted(UNCAPPED_GRAPHS)),
        time.monotonic() - started,
    )
    return ResidentDecoder(processes)
