"""Wake-word verification and wake/command split point.

Four grammar-constrained decodes of the shared acoustic model (see train.py)
produce a verdict plus the split point. Costs are Kaldi best-path totals;
the filler regions cancel between graphs:

  lr        = (cost_bg - cost_kw)     / wake_frames
  lr_prefix = (cost_bg - cost_prefix) / prefix_frames
  lr_last   = (cost_prefix - cost_kw) / last_word_frames
  margin    = max(0, cost_kw - cost_kwc) / wake_frames

Accept iff lr_prefix >= -theta_prefix AND lr_last >= -theta_last AND
margin <= theta_competitor. The default thresholds were calibrated on
labeled far-field recordings of "okay nabu"; re-calibrate for other wake
words or microphones.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DecoderConfig, Phrase, WakeWordConfig
from .kaldi import FRAME_SHIFT, BestCost, CtmWord, SpeechTools, decode_best_path
from .model import PreparedModel
from .train import CTM_TAGS, GRAPH_TAGS

_LOGGER = logging.getLogger(__name__)

_IGNORED_WORDS = ("<sil>", "<eps>", "<unk>", "!SIL")


@dataclass
class VerifyResult:
    """Verdict for one wake-word verification."""

    wake_word_id: str
    accepted: bool = False
    rejected_by: str | None = None
    """prefix | last | confusable | no_alignment | error"""

    phrase: str | None = None
    """The accepted form that the 1-best alignment matched."""

    lr: float | None = None
    lr_prefix: float | None = None
    lr_last: float | None = None
    confusable_margin: float | None = None
    confusable_words: str = ""
    wake_start: float | None = None
    """Seconds from the start of the window to the first wake word."""

    split: float | None = None
    """Seconds from the start of the window to the end of the wake phrase."""

    aligned_words: list[CtmWord] = field(default_factory=list)
    decode_ms: float = 0.0
    error: str | None = None

    @property
    def split_ms(self) -> int | None:
        return None if self.split is None else int(round(self.split * 1000))

    def to_dict(self) -> dict[str, Any]:
        return {
            "wake_word": self.wake_word_id,
            "accepted": self.accepted,
            "rejected_by": self.rejected_by,
            "phrase": self.phrase,
            "lr": self.lr,
            "lr_prefix": self.lr_prefix,
            "lr_last": self.lr_last,
            "confusable_margin": self.confusable_margin,
            "confusable_words": self.confusable_words,
            "wake_start": self.wake_start,
            "split": self.split,
            "split_ms": self.split_ms,
            "aligned_words": [list(w) for w in self.aligned_words],
            "decode_ms": round(self.decode_ms, 1),
            "error": self.error,
        }


class Verifier:
    """Decodes one wake word's graphs; optionally through a resident pool."""

    def __init__(
        self,
        wake_word: WakeWordConfig,
        model: PreparedModel,
        graph_dir: Path,
        tools: SpeechTools,
        pool: Any | None = None,
        decoder: DecoderConfig | None = None,
    ) -> None:
        self.wake_word = wake_word
        self.model = model
        self.graph_dir = Path(graph_dir)
        self.tools = tools
        self.pool = pool
        self.decoder = decoder or DecoderConfig()

    async def verify(self, wav_path: Path) -> VerifyResult:
        wav_path = Path(wav_path).absolute()
        result = VerifyResult(wake_word_id=self.wake_word.id)
        started = time.monotonic()

        try:
            if self.pool is not None:
                # One process, one acoustic pass, all four graphs.
                decoded = await self.pool.decode_all(wav_path)
                results = [decoded[tag] for tag in GRAPH_TAGS]
            else:
                results = await asyncio.gather(
                    *(self._decode(tag, wav_path) for tag in GRAPH_TAGS)
                )
        except (RuntimeError, OSError, KeyError, asyncio.TimeoutError) as err:
            result.error = str(err)
            result.rejected_by = "error"
            _LOGGER.error("Decode failed for %s: %s", self.wake_word.id, err)
            return result

        result.decode_ms = (time.monotonic() - started) * 1000
        (kw_cost, kw_words), (bg_cost, _), (pre_cost, _), (kwc_cost, kwc_words) = results

        return score(
            result,
            self.wake_word,
            kw_cost,
            kw_words,
            bg_cost,
            pre_cost,
            kwc_cost,
            kwc_words,
        )

    async def _decode(
        self, tag: str, wav_path: Path
    ) -> tuple[BestCost | None, list[CtmWord]]:
        """One subprocess decode; the fallback when no daemon is available."""
        return await asyncio.to_thread(
            decode_best_path,
            self.tools,
            self.model.final_mdl,
            self.model.online_conf,
            self.graph_dir / f"graph_{tag}",
            self.graph_dir / "data" / f"lang_{tag}",
            wav_path,
            tag in CTM_TAGS,
            self.decoder.args(tag),
        )


def match_phrase(
    aligned_words: list[CtmWord], phrases: tuple[Phrase, ...]
) -> tuple[Phrase, list[tuple[float, float]]] | None:
    """Find which accepted form the 1-best alignment starts with.

    The grammar may take explicit <sil> arcs between words, and the filler
    loop follows the phrase, so only the leading real words are considered.
    Longer forms win, so "okay nabu nabu" is not matched as "okay nabu".
    """
    spoken = [w for w in aligned_words if not _is_ignorable(w[0])]

    best: tuple[Phrase, list[tuple[float, float]]] | None = None
    for phrase in phrases:
        count = len(phrase.words)
        if len(spoken) < count:
            continue
        if any(spoken[i][0] != phrase.words[i] for i in range(count)):
            continue
        spans = [(spoken[i][1], spoken[i][2]) for i in range(count)]
        if best is None or count > len(best[1]):
            best = (phrase, spans)

    return best


def _is_ignorable(word: str) -> bool:
    return word in _IGNORED_WORDS or word.startswith("f_")


def score(
    result: VerifyResult,
    wake_word: WakeWordConfig,
    kw_cost: BestCost | None,
    kw_words: list[CtmWord],
    bg_cost: BestCost | None,
    pre_cost: BestCost | None,
    kwc_cost: BestCost | None,
    kwc_words: list[CtmWord],
) -> VerifyResult:
    """Turn four decodes into a verdict. Split out for testability."""
    thresholds = wake_word.thresholds
    result.aligned_words = kw_words
    accepted_words = {w for phrase in wake_word.phrases for w in phrase.words}
    result.confusable_words = " ".join(
        word
        for word, _start, _end in kwc_words
        if word not in accepted_words and not _is_ignorable(word)
    )

    matched = match_phrase(kw_words, wake_word.phrases)
    if matched is None or kw_cost is None or bg_cost is None:
        result.rejected_by = "no_alignment"
        return result

    phrase, spans = matched
    result.phrase = phrase.text
    result.wake_start = spans[0][0]
    result.split = spans[-1][1]

    wake_frames = max(1.0, (result.split - result.wake_start) / FRAME_SHIFT)
    result.lr = (bg_cost[0] - kw_cost[0]) / wake_frames

    if len(spans) > 1 and pre_cost is not None:
        prefix_end = spans[-2][1]
        prefix_frames = max(1.0, (prefix_end - result.wake_start) / FRAME_SHIFT)
        last_frames = max(1.0, (spans[-1][1] - spans[-1][0]) / FRAME_SHIFT)
        result.lr_prefix = (bg_cost[0] - pre_cost[0]) / prefix_frames
        result.lr_last = (pre_cost[0] - kw_cost[0]) / last_frames
    else:
        # single-word phrase: the whole-phrase ratio carries the decision
        result.lr_prefix = result.lr
        result.lr_last = result.lr

    if kwc_cost is not None:
        result.confusable_margin = max(0.0, (kw_cost[0] - kwc_cost[0]) / wake_frames)

    if result.lr_prefix < -thresholds.theta_prefix:
        result.rejected_by = "prefix"
    elif result.lr_last < -thresholds.theta_last:
        result.rejected_by = "last"
    elif (
        result.confusable_margin is not None
        and result.confusable_margin > thresholds.theta_competitor
    ):
        result.rejected_by = "confusable"
    else:
        result.accepted = True

    return result
