"""Spoken forms of numbers that inflect, per language."""

import pytest
from speech_to_phrase.templates import _NUMBER_VARIANTS, _spellout_variants
from unicode_rbnf import FormatPurpose, RbnfEngine

WELSH = {
    2: {"dau", "dwy"},
    3: {"tri", "tair"},
    4: {"pedwar", "pedair"},
    5: {"pump", "pum"},
    6: {"chwech", "chwe"},
}


@pytest.mark.parametrize("locale", ["cy", "cy_GB", "cy-GB"])
@pytest.mark.parametrize("number", sorted(WELSH))
def test_welsh_variants(locale: str, number: int) -> None:
    """Every Welsh form of 2-6 is in the grammar, whatever the locale spelling."""
    forms = {" ".join(words) for words in _spellout_variants(number, locale)}
    assert forms == WELSH[number]


@pytest.mark.parametrize("number", sorted(_NUMBER_VARIANTS["cy"]))
def test_welsh_variants_are_understood_by_home_assistant(number: int) -> None:
    """Speech-to-Phrase sends the words it heard to Home Assistant, which parses
    numbers with unicode-rbnf. A form missing there would be recognized here and
    then fail to match an intent, so only add forms Home Assistant accepts."""
    result = RbnfEngine.for_language("cy").format_number(number, FormatPurpose.CARDINAL)
    understood = set(result.text_by_ruleset.values())
    assert set(_NUMBER_VARIANTS["cy"][number]) <= understood
