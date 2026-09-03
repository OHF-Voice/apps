"""Tokenizer regression tests."""

from speech_to_phrase.tokenizer import SPM_SPACE, SubwordTokenizer


def test_subword_detokenization_removes_space_after_apostrophe() -> None:
    tokenizer = SubwordTokenizer(
        {
            0: "allume",
            1: f"{SPM_SPACE}l'",
            2: f"{SPM_SPACE}enceintes",
            3: "<blk>",
        }
    )

    assert tokenizer.ids_to_text([0, 1, 2]) == "allume l'enceintes"
