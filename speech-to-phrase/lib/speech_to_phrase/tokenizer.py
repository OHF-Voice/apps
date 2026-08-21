"""Tokenizers map text <-> acoustic-model token ids for each backend.

- :class:`CharTokenizer` (Coqui): one id per character, from ``alphabet.txt``.
- :class:`SubwordTokenizer` (Citrinet): SentencePiece subwords from ``tokens.json``;
  uses a ``*.model`` via the ``sentencepiece`` package when available, otherwise a
  greedy longest-match fallback over the vocabulary.

Every tokenizer exposes ``num_tokens`` (count of emit-able, non-blank classes) and
``blank_id`` (the CTC blank, conventionally the last class). The acoustic model's
output dimension is ``num_tokens + 1``.
"""

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

# SentencePiece word-boundary marker (U+2581).
SPM_SPACE = "▁"

_BLANK_PIECES = ("<blk>", "<blank>", "<pad>")
_UNK_PIECES = ("<unk>", "<UNK>")


@runtime_checkable
class Tokenizer(Protocol):
    num_tokens: int
    blank_id: int

    def text_to_ids(self, text: str) -> List[int]: ...  # noqa: E704

    def ids_to_text(self, ids: Sequence[int]) -> str: ...  # noqa: E704


# ----------------------------------------------------------------------------
# Coqui: character alphabet
# ----------------------------------------------------------------------------


def _read_alphabet(path: Path) -> List[str]:
    """Parse a Coqui ``alphabet.txt`` into an ordered list of single labels.

    A line beginning with ``#`` is a comment; ``\\#`` escapes a literal ``#``
    label. The label is the line content without its trailing newline, so a line
    that is just a space yields the space character.
    """
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as alphabet_file:
        for raw in alphabet_file:
            line = raw.rstrip("\n").rstrip("\r")
            if line.startswith("#"):
                continue
            if line.startswith("\\#"):
                line = line[1:]
            labels.append(line)
    if not labels:
        raise ValueError(f"No labels found in alphabet: {path}")
    return labels


# Apostrophe-like codepoints that orthographies use interchangeably; folded to
# whichever variant a model's alphabet actually contains.
_APOSTROPHES = "'’ʼʹ‘′´`"


class CharTokenizer:
    """Character tokenizer for Coqui STT models."""

    def __init__(self, labels: Sequence[str]):
        self._labels = list(labels)
        self._char2id: Dict[str, int] = {c: i for i, c in enumerate(self._labels)}
        self.num_tokens = len(self._labels)
        self.blank_id = self.num_tokens  # blank is the last class

    @classmethod
    def from_alphabet_file(cls, path: Path) -> "CharTokenizer":
        return cls(_read_alphabet(Path(path)))

    def text_to_ids(self, text: str) -> List[int]:
        ids: List[int] = []
        for ch in text:
            token_id = self._char2id.get(ch)
            if token_id is not None:
                ids.append(token_id)
                continue
            # Apostrophe-family confusables: an orthography may use a different
            # apostrophe codepoint than the one in the model's alphabet (e.g.
            # icu spells Ukrainian "пʼять" with U+02BC, while the alphabet only
            # contains U+2019). Map any apostrophe variant to the one present.
            if ch in _APOSTROPHES:
                alt = next(
                    (self._char2id[a] for a in _APOSTROPHES if a in self._char2id),
                    None,
                )
                if alt is not None:
                    ids.append(alt)
                    continue
            # Fold to the model's alphabet: many Coqui models are trained on
            # de-accented text (e.g. é -> e), so decompose and keep the base
            # characters that the alphabet does contain. Characters already in
            # the alphabet (such as ñ) are handled above and never folded away.
            folded = [
                self._char2id[c]
                for c in unicodedata.normalize("NFD", ch)
                if c in self._char2id
            ]
            if not folded:
                raise ValueError(
                    f"Character {ch!r} is not in the alphabet (text={text!r})"
                )
            ids.extend(folded)
        return ids

    def ids_to_text(self, ids: Sequence[int]) -> str:
        return "".join(self._labels[i] for i in ids if 0 <= i < self.num_tokens).strip()


# ----------------------------------------------------------------------------
# Citrinet: SentencePiece subwords
# ----------------------------------------------------------------------------


def _read_tokens_json(path: Path) -> Dict[int, str]:
    """Parse a ``tokens.json`` (JSON array of pieces, index == token id)."""
    pieces = json.loads(Path(path).read_text(encoding="utf-8"))
    if not pieces:
        raise ValueError(f"No tokens found in: {path}")
    return dict(enumerate(pieces))


def _read_tokens_txt(path: Path) -> Dict[int, str]:
    """Parse a NeMo ``tokens.txt`` (``<piece> <id>`` per line)."""
    id2piece: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as tokens_file:
        for line in tokens_file:
            line = line.rstrip("\n")
            if not line:
                continue
            piece, _, id_str = line.rpartition(" ")
            if not piece:
                continue
            id2piece[int(id_str)] = piece
    if not id2piece:
        raise ValueError(f"No tokens found in: {path}")
    return id2piece


def _read_tokens(path: Path) -> Dict[int, str]:
    """Load a token vocabulary from ``tokens.json`` or a legacy ``tokens.txt``."""
    path = Path(path)
    if path.suffix == ".json":
        return _read_tokens_json(path)
    return _read_tokens_txt(path)


class SubwordTokenizer:
    """SentencePiece subword tokenizer for Citrinet models.

    Prefers an exact SentencePiece model (``model_path``); otherwise falls back to
    greedy longest-match over the vocabulary, which approximates the segmentation
    (recommended to supply the ``*.model`` for best accuracy).
    """

    def __init__(self, id2piece: Dict[int, str], model_path: Optional[Path] = None):
        self._id2piece = dict(id2piece)
        self._piece2id: Dict[str, int] = {p: i for i, p in self._id2piece.items()}

        # SentencePiece models mark word starts with U+2581; some NeMo models
        # (e.g. Chinese) are plain-character vocabularies that use a literal
        # space instead and have no U+2581 piece. Don't inject the marker there.
        self._has_spm_space = any(SPM_SPACE in p for p in self._piece2id)

        blank_id = next(
            (i for i, p in self._id2piece.items() if p in _BLANK_PIECES), None
        )
        if blank_id is None:
            # No explicit blank piece: assume blank is one past the max id.
            blank_id = max(self._id2piece) + 1
        self.blank_id = blank_id
        self.num_tokens = blank_id  # emit ids are 0..blank_id-1 (contiguous)

        self._unk_id = next(
            (i for i, p in self._id2piece.items() if p in _UNK_PIECES), 0
        )
        self._max_piece_len = max(len(p) for p in self._piece2id)

        self._sp = None
        if model_path is not None:
            import sentencepiece as spm  # noqa: PLC0415

            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(str(model_path))

    @classmethod
    def from_model_dir(
        cls, tokens_path: Path, model_path: Optional[Path] = None
    ) -> "SubwordTokenizer":
        return cls(_read_tokens(Path(tokens_path)), model_path)

    def text_to_ids(self, text: str) -> List[int]:
        if self._sp is not None:
            return list(self._sp.EncodeAsIds(text))
        return self._greedy_encode(text)

    def segment_lattice(self, text: str) -> Tuple[int, List[Tuple[int, int, int]]]:
        """All valid subword segmentations of ``text`` as a position lattice.

        ``text_to_ids`` commits to a single (greedy longest-match) segmentation,
        but a CTC acoustic model is free to emit any valid one -- e.g. it emits
        "percent" as the pieces for "per cent". When the grammar hard-codes only
        the canonical segmentation, the model's natural output can't match, which
        inflates the constrained-decode cost (worse scores, sometimes the wrong
        parse). Returning the whole lattice lets the grammar accept whichever
        segmentation the model produces.

        Returns ``(n_positions, edges)`` over the SPM-marked text, where each
        edge ``(i, j, token_id)`` means the vocabulary piece for
        ``spm_text[i:j]`` spans character positions ``i -> j``. Positions ``0``
        and ``n_positions`` are the start and end. Positions with no vocabulary
        match get a single ``<unk>`` step (mirroring ``_greedy_encode``).
        """
        if self._has_spm_space:
            spm_text = SPM_SPACE + text.strip().replace(" ", SPM_SPACE)
        else:
            spm_text = text.strip()
        n = len(spm_text)
        edges: List[Tuple[int, int, int]] = []
        reachable = [False] * (n + 1)
        reachable[0] = True
        for i in range(n):
            if not reachable[i]:
                continue  # not on any path from the start; skip (and its unk)
            upper = min(self._max_piece_len, n - i)
            matched = False
            for length in range(1, upper + 1):
                token_id = self._piece2id.get(spm_text[i : i + length])
                if token_id is not None:
                    edges.append((i, i + length, token_id))
                    reachable[i + length] = True
                    matched = True
            if not matched:
                edges.append((i, i + 1, self._unk_id))
                reachable[i + 1] = True
        return n, edges

    def _greedy_encode(self, text: str) -> List[int]:
        # SentencePiece marks word starts with U+2581 and adds one at the front.
        # Plain-character models (no U+2581 in the vocab) keep literal spaces.
        if self._has_spm_space:
            spm_text = SPM_SPACE + text.strip().replace(" ", SPM_SPACE)
        else:
            spm_text = text.strip()
        ids: List[int] = []
        pos = 0
        n = len(spm_text)
        while pos < n:
            matched = False
            upper = min(self._max_piece_len, n - pos)
            for length in range(upper, 0, -1):
                token_id = self._piece2id.get(spm_text[pos : pos + length])
                if token_id is not None:
                    ids.append(token_id)
                    pos += length
                    matched = True
                    break
            if not matched:
                ids.append(self._unk_id)
                pos += 1
        return ids

    def ids_to_text(self, ids: Sequence[int]) -> str:
        pieces = [
            self._id2piece[i]
            for i in ids
            if (i != self.blank_id) and (i in self._id2piece)
        ]
        return "".join(pieces).replace(SPM_SPACE, " ").strip()
