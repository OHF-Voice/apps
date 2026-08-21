"""Python wrapper around the native :mod:`speech_to_phrase._fst` extension."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import _fst
from .templates import Fst


class Grammar:
    """A compiled token->sentence FST plus its decode entry point."""

    def __init__(self, capsule: object):
        self._capsule = capsule

    @classmethod
    def build(cls, num_tokens: int, blank_id: int, fst: Fst) -> "Grammar":
        arcs = np.asarray(fst.arc_tuples(), dtype=np.int32).reshape(-1)
        finals = np.asarray(sorted(fst.final_states), dtype=np.int32)
        if arcs.size == 0:
            raise ValueError("Template FST has no arcs")
        capsule = _fst.build_grammar(int(num_tokens), int(blank_id), arcs, finals)
        return cls(capsule)

    def save(self, path: Path) -> None:
        _fst.save_grammar(self._capsule, str(path))

    @classmethod
    def load(cls, path: Path) -> "Grammar":
        return cls(_fst.load_grammar(str(path)))

    def decode(
        self,
        log_probs: np.ndarray,
        blank_id: int,
        beam: float = 0.0,
        token_bonus: float = 0.0,
    ) -> Tuple[List[int], Optional[float], Optional[float]]:
        """Constrained CTC decode.

        ``log_probs`` is ``[T, V]`` (``V == num_tokens + 1``). Returns the best
        path's token ids and the best / second-best path costs (``None`` when no
        path / only one path exists).

        ``beam`` (log-prob units) prunes per-frame candidates to those within
        ``beam`` of the best; ``0`` disables pruning (exact).

        ``token_bonus`` is a word-insertion reward subtracted from each emitting
        arc's cost during path *selection* only, so a longer well-fitting parse
        isn't beaten purely by having more tokens than a short one; ``0``
        disables it. The returned costs are the true acoustic costs (the bonus
        is added back), so the per-token score keeps its meaning.
        """
        lp = np.ascontiguousarray(log_probs, dtype=np.float32)
        T, V = lp.shape
        args = (
            self._capsule,
            lp.reshape(-1),
            int(T),
            int(V),
            int(blank_id),
            float(beam),
        )
        # token_bonus is optional in the native extension; only pass it when set
        # so a stale 6-arg _fst build (default behaviour) keeps working.
        if token_bonus:
            return _fst.decode(*args, float(token_bonus))
        return _fst.decode(*args)
