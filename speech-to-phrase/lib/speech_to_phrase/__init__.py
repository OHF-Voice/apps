"""speech_to_phrase: constrained CTC speech-to-text over sentence templates.

Build a grammar from sentence templates, then transcribe audio constrained to
that grammar with either a Citrinet (ONNX) or Coqui (TFLite) CTC backend:

    from speech_to_phrase import load_recognizer
    rec = load_recognizer("citrinet", "local/stt_en_citrinet_512", language="en")
    rec.train(open("sentences.txt").read().splitlines())
    result = rec.transcribe("clip.wav")
    print(result.text, result.score, result.margin)

``score`` is the constrained-vs-greedy cost penalty per emitted token (lower is
better); ``margin`` is the gap to the second-best parse. Both gate whether the
local result is trusted versus deferring to a cloud recognizer.
"""

import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .audio import load_audio
from .backends.base import AcousticModel
from .grammar import Grammar
from .templates import templates_to_fst

__all__ = [
    "Recognizer",
    "Result",
    "load_recognizer",
    "__version__",
]

__version__ = "0.1.0.dev0"


@dataclass
class Result:
    text: str
    score: float  # penalty per token (lower is better); inf if no parse
    margin: float  # per-token gap to second-best parse (inf if only one parse)


def _normalize_template(line: str) -> str:
    return unicodedata.normalize("NFC", line).lower().strip()


def _normalize_list_values(
    list_values: Optional[Mapping[str, Sequence[str]]]
) -> Optional[Dict[str, List[str]]]:
    """Normalize slot values the same way as templates (NFC + lowercase + strip).

    The acoustic vocabulary is lowercase, so an un-normalized value such as
    ``"Office Lamp"`` tokenizes to ``"<unk>ffice <unk>amp"`` and never matches.
    Values are de-duplicated (preserving order) after normalization.
    """
    if not list_values:
        return None if list_values is None else {}
    normalized: Dict[str, List[str]] = {}
    for name, values in list_values.items():
        seen = {_normalize_template(v): None for v in values if v and v.strip()}
        normalized[name] = list(seen)
    return normalized


class Recognizer:
    """Couples an acoustic backend with a compiled template grammar."""

    def __init__(
        self,
        model: AcousticModel,
        language: str,
        beam: float = 0.0,
        token_bonus: float = 0.0,
        token_bonus_max_unbiased_score: float = math.inf,
        blank_retry_penalty: float = 0.0,
        blank_retry_min_score: float = math.inf,
        blank_retry_max_score: float = -math.inf,
    ):
        self.model = model
        self.language = language
        # Per-frame log-prob beam for decode pruning (0 = exact). A positive
        # beam is a large speedup for many-frame / character grammars.
        self.beam = beam
        # Word-insertion reward per emitted token (0 = off). Counters the CTC
        # length bias that otherwise lets a short parse (e.g. "stop") win over a
        # longer, better-fitting one (e.g. "brighten the office") purely on
        # token count. See Grammar.decode.
        self.token_bonus = token_bonus
        # A length reward may rerank plausible parses, but must never rescue
        # audio that the unbiased grammar decode would reject as OOV.
        self.token_bonus_max_unbiased_score = token_bonus_max_unbiased_score
        # Citrinet can put almost all probability on CTC blank while a VPE noise
        # suppressor distorts one word. In a narrow confidence band, retry with a
        # selection-only blank penalty so the full entity name can compete with a
        # short generic command. Bounding the retry prevents rejected OOV audio
        # from becoming an accepted long phrase.
        self.blank_retry_penalty = blank_retry_penalty
        self.blank_retry_min_score = blank_retry_min_score
        self.blank_retry_max_score = blank_retry_max_score
        self.grammar: Optional[Grammar] = None

    def train(
        self,
        sentences: Sequence[str],
        list_values: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        templates = [
            _normalize_template(line) for line in sentences if line and line.strip()
        ]
        if not templates:
            raise ValueError("No sentence templates provided")
        fst = templates_to_fst(
            templates,
            tokenizer=self.model.tokenizer,
            locale=self.language,
            list_values=_normalize_list_values(list_values),
        )
        self.grammar = Grammar.build(
            self.model.tokenizer.num_tokens, self.model.tokenizer.blank_id, fst
        )

    def save(self, path: Union[str, Path]) -> None:
        if self.grammar is None:
            raise RuntimeError("Recognizer is not trained")
        self.grammar.save(Path(path))

    def load(self, path: Union[str, Path]) -> None:
        self.grammar = Grammar.load(Path(path))

    def _decode(self, log_probs: np.ndarray) -> tuple[Result, List[int]]:
        """Decode both acoustic-only and length-corrected candidates.

        ``token_bonus`` is deliberately a candidate generator rather than the
        final objective. It recovers long commands from CTC's blank/short-output
        bias, while selecting by the reported per-token acoustic penalty avoids
        adding unsupported optional words merely because they are longer.
        """
        assert self.grammar is not None
        greedy_cost = float(-log_probs.max(axis=-1).sum())
        def decode_one(bonus: float) -> tuple[Result, List[int]]:
            decoded = self.grammar.decode(
                log_probs,
                self.model.tokenizer.blank_id,
                beam=self.beam,
                token_bonus=bonus,
            )
            token_ids, best_cost, second_cost = decoded  # pylint: disable=E0633
            text = self.model.tokenizer.ids_to_text(token_ids)
            if best_cost is None or not token_ids:
                result = Result(text="", score=math.inf, margin=math.inf)
            else:
                n_tokens = len(token_ids)
                score = (best_cost - greedy_cost) / n_tokens
                if second_cost is None:
                    margin = math.inf
                else:
                    margin = (second_cost - best_cost) / n_tokens
                result = Result(text=text, score=score, margin=margin)
            return result, token_ids

        unbiased = decode_one(0.0)
        if (
            not self.token_bonus
            or unbiased[0].score > self.token_bonus_max_unbiased_score
        ):
            return unbiased

        corrected = decode_one(self.token_bonus)
        return min((unbiased, corrected), key=lambda candidate: candidate[0].score)

    def transcribe(self, audio: Union[str, Path, np.ndarray]) -> Result:
        if self.grammar is None:
            raise RuntimeError("Recognizer is not trained or loaded")

        samples = load_audio(audio)
        log_probs = self.model.log_probs(samples)  # [T, V]
        result, token_ids = self._decode(log_probs)

        if (
            self.blank_retry_penalty
            and self.blank_retry_min_score < result.score <= self.blank_retry_max_score
        ):
            retry_probs = log_probs.copy()
            retry_probs[:, self.model.tokenizer.blank_id] -= self.blank_retry_penalty
            retry_result, retry_ids = self._decode(retry_probs)
            if len(retry_ids) > len(token_ids) and retry_result.score < result.score:
                return retry_result

        return result


def load_recognizer(
    backend: str,
    model_dir: Union[str, Path],
    language: str,
    beam: Optional[float] = None,
    token_bonus: float = 0.0,
    token_bonus_max_unbiased_score: Optional[float] = None,
    blank_retry_penalty: Optional[float] = None,
    blank_retry_min_score: Optional[float] = None,
    blank_retry_max_score: Optional[float] = None,
    **kwargs,
) -> Recognizer:
    """Create a :class:`Recognizer` for ``"citrinet"`` or ``"coqui"``.

    Extra keyword arguments are forwarded to the backend (e.g. ``spm_model=...``
    for Citrinet; ``stt_binary=...`` for Coqui). ``beam`` sets the decode
    pruning beam; if unset it defaults per backend (Coqui benefits from pruning
    its many-frame character grammar). ``token_bonus_max_unbiased_score`` keeps
    the length reward from rescuing an OOV utterance. The blank-retry options
    control Citrinet's bounded recovery pass for low-confidence short decodes;
    leaving them unset uses the calibrated backend defaults.
    """
    backend = backend.lower()
    if backend == "citrinet":
        from .backends.citrinet import CitrinetModel

        model: AcousticModel = CitrinetModel(Path(model_dir), **kwargs)
        # Beam relative to the best grammar candidate per frame; validated to
        # not sever low-probability required subwords (e.g. rare names).
        default_beam = 15.0
        default_token_bonus_max_score = 5.0
        default_blank_retry = (1.0, 3.0, 5.0)
    elif backend == "coqui":
        from .backends.coqui import CoquiModel

        model = CoquiModel(Path(model_dir), **kwargs)
        default_beam = 10.0
        default_token_bonus_max_score = math.inf
        default_blank_retry = (0.0, math.inf, -math.inf)
    else:
        raise ValueError(f"Unknown backend: {backend!r} (expected citrinet|coqui)")

    return Recognizer(
        model,
        language,
        beam=default_beam if beam is None else beam,
        token_bonus=token_bonus,
        token_bonus_max_unbiased_score=(
            default_token_bonus_max_score
            if token_bonus_max_unbiased_score is None
            else token_bonus_max_unbiased_score
        ),
        blank_retry_penalty=(
            default_blank_retry[0]
            if blank_retry_penalty is None
            else blank_retry_penalty
        ),
        blank_retry_min_score=(
            default_blank_retry[1]
            if blank_retry_min_score is None
            else blank_retry_min_score
        ),
        blank_retry_max_score=(
            default_blank_retry[2]
            if blank_retry_max_score is None
            else blank_retry_max_score
        ),
    )
