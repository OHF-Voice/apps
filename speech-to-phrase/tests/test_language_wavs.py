"""End-to-end multilingual aggregate-grammar tests."""

from __future__ import annotations

from typing import Callable

import language_test_common as common
import pytest

_MANIFEST = common.load_manifests()


def _case_id(item: dict) -> str:
    return f"{item['language']}-{item['intent']}/{item['combo']}"


@pytest.fixture(scope="session")
def language_harness(
    tmp_path_factory,
) -> Callable[[str], common.LanguageHarness]:
    cache: dict[str, common.LanguageHarness] = {}

    def get(language: str) -> common.LanguageHarness:
        if language not in cache:
            grammar_path = (
                tmp_path_factory.mktemp(f"grammar-{language}") / "grammar.fst"
            )
            cache[language] = common.build_harness(language, grammar_path)
        return cache[language]

    return get


@pytest.mark.skipif(not _MANIFEST, reason="run tests/generate_language_wavs.py first")
@pytest.mark.parametrize("item", _MANIFEST, ids=[_case_id(i) for i in _MANIFEST])
def test_clip_is_recognized(item, language_harness):
    harness = language_harness(item["language"])
    wav_path = common.WAV_ROOT / item["language"] / item["wav"]
    assert wav_path.exists(), f"missing {wav_path}"

    result = common.transcribe_clip(harness.recognizer, wav_path)
    assert result.score <= harness.max_score, (
        f"{item['text']!r} -> {result.text!r} declined "
        f"({result.score:.2f} > {harness.max_score:.2f})"
    )
    matched = harness.matcher.match(result.text)
    assert matched is not None, f"{item['text']!r} -> {result.text!r} matched no intent"
    assert matched.intent.name == item["expected_intent"], (
        f"{item['text']!r} -> {result.text!r} -> {matched.intent.name}, "
        f"expected {item['expected_intent']}"
    )
    assert common.result_slots(matched) == item["expected_slots"], (
        f"{item['text']!r} -> {result.text!r} -> "
        f"{common.result_slots(matched)}, expected {item['expected_slots']}"
    )


@pytest.mark.parametrize("language", common.languages())
def test_manifest_covers_every_template(language):
    path = common.manifest_path(language)
    assert path.exists(), f"generate the {language} corpus first"
    cached = {
        (item["intent"], item["combo"])
        for item in common.load_manifests()
        if item["language"] == language
    }
    assert cached == set(common.combos(language))
