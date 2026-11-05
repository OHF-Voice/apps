"""Model lexicon and phone inventory.

The verifier never guesses a pronunciation. Every token of a wake phrase or a
confusable phrase must either be a word in the model's `lexicon.db` or carry
an explicit pronunciation from the YAML config, and every phoneme in such a
pronunciation must exist in the model's phone inventory.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class LexiconError(Exception):
    """A wake word or confusable cannot be expressed with this model."""


class ModelLexicon:
    """Read-only view of a Kaldi model's lexicon and phone inventory."""

    def __init__(self, model_data_dir: Path) -> None:
        self.model_data_dir = Path(model_data_dir).absolute()
        self.model_dir = self.model_data_dir / "model"
        self.lexicon_db_path = self.model_data_dir / "lexicon.db"
        self.phones_dir = self.model_dir / "phones"

        if not self.lexicon_db_path.is_file():
            raise LexiconError(f"Model lexicon not found: {self.lexicon_db_path}")
        if not self.phones_dir.is_dir():
            raise LexiconError(f"Model phones not found: {self.phones_dir}")

        self.silence_phones = _read_phone_set(self.phones_dir / "silence_phones.txt")
        self.nonsilence_phones = _read_phone_set(
            self.phones_dir / "nonsilence_phones.txt"
        )
        self.phones = self.silence_phones | self.nonsilence_phones

        self._conn = sqlite3.connect(
            f"file:{self.lexicon_db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._cache: dict[str, list[str]] = {}

    def close(self) -> None:
        self._conn.close()

    def filler_inventory(self) -> list[tuple[str, list[str]]]:
        """One filler word per nonsilence phone line, stress variants as prons."""
        lines = (
            (self.phones_dir / "nonsilence_phones.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        fillers: list[tuple[str, list[str]]] = []
        for line in lines:
            variants = line.split()
            if variants:
                fillers.append((f"f_{variants[0]}", variants))
        return fillers

    def pronunciations(self, word: str) -> list[str]:
        """All lexicon pronunciations for `word`, in lexicon order."""
        cached = self._cache.get(word)
        if cached is None:
            cur = self._conn.execute(
                "SELECT phonemes FROM word_phonemes WHERE word = ? "
                "ORDER BY pron_order",
                (word,),
            )
            cached = [row[0] for row in cur]
            self._cache[word] = cached
        return cached

    def validate_pronunciation(self, word: str, pron: str) -> list[str]:
        """Split a config-supplied pronunciation and check it against the model."""
        phonemes = pron.split()
        if not phonemes:
            raise LexiconError(f"Empty pronunciation for {word!r}")

        unknown = [p for p in phonemes if p not in self.phones]
        if unknown:
            raise LexiconError(
                f"Pronunciation for {word!r} uses phonemes that are not in the "
                f"{self.model_data_dir.name} inventory: {' '.join(unknown)}. "
                f"See {self.phones_dir / 'nonsilence_phones.txt'}."
            )
        return phonemes


def _read_phone_set(path: Path) -> set[str]:
    phones: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        phones.update(line.split())
    return phones
