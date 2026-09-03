"""Shared corpus and recognition helpers for multilingual grammar tests."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

TESTS_DIR = Path(__file__).resolve().parent
APP_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(APP_ROOT / "src"))

from speech_to_phrase import Recognizer, load_recognizer  # noqa: E402
from speech_to_phrase.audio import SAMPLE_RATE  # noqa: E402

import audio_frontend  # noqa: E402
import gating  # noqa: E402
import intent_matcher  # noqa: E402
import models  # noqa: E402
import presets  # noqa: E402
import s2p_intents  # noqa: E402
import training  # noqa: E402

SERVER_ROOT = Path(training.__file__).resolve().parent
INTENT_TESTS_DIR = Path.home() / "opt/intent-sentences/tests"
MODELS_DIR = APP_ROOT / "local/models"
TOOLS_DIR = APP_ROOT / "local/tools"
WAV_ROOT = TESTS_DIR / "wav"
_EXAMPLE_OVERRIDES = {
    ("en", "HassUnpauseTimer", "default"): "continue the timer",
    ("nl", "HassTurnOff", "name_only"): "doe het Slaapkamerlampje uit",
}
_SYNTHESIS_OVERRIDES = {
    ("ca", "HassTurnOff", "name_only"): "apaga Interruptor A",
    ("ca", "HassTurnOn", "name_area"): "engega Interruptor A a la Cuina",
    ("en", "HassMediaUnpause", "default"): "un pause the music",
    ("fr", "HassTurnOn", "name_only"): "allume les enceintes",
}


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio while preserving channels."""
    if source_rate == target_rate:
        return audio

    rate_gcd = gcd(source_rate, target_rate)
    return resample_poly(
        audio,
        target_rate // rate_gcd,
        source_rate // rate_gcd,
        axis=0,
    ).astype("float32")


@dataclass
class FixtureData:
    entities: List[dict]
    slot_lists: Dict[str, List[str]]


@dataclass
class LanguageHarness:
    recognizer: Recognizer
    matcher: intent_matcher.IntentMatcher
    max_score: float


def languages() -> List[str]:
    """Languages with packaged Speech-to-Phrase templates."""
    return s2p_intents.languages()


def combos(language: str) -> List[tuple[str, str]]:
    """Every packaged intent/slot combination for a language."""
    return s2p_intents.combos(language)


def enabled_all(language: str) -> List[List[str]]:
    return [[intent, combo] for intent, combo in combos(language)]


def load_fixtures(language: str) -> FixtureData:
    """Merge fixtures for all Speech-to-Phrase combinations in a language."""
    entities_by_key: Dict[tuple[str, str], dict] = {}
    areas_by_name: Dict[str, dict] = {}
    floors_by_name: Dict[str, dict] = {}

    fixture_paths = sorted((INTENT_TESTS_DIR / language).glob("*/*.yaml"))
    if not fixture_paths:
        raise FileNotFoundError(
            f"No slot-combination fixtures for {language} in {INTENT_TESTS_DIR}"
        )
    for fixture_path in fixture_paths:
        fixture = yaml.safe_load(fixture_path.read_text(encoding="utf-8")) or {}
        for entity in fixture.get("entities", []):
            if entity.get("is_exposed", True):
                entities_by_key.setdefault(
                    (entity["name"], entity["domain"]), dict(entity)
                )
        for area in fixture.get("areas", []):
            areas_by_name.setdefault(area["name"], dict(area))
        for floor in fixture.get("floors", []):
            floors_by_name.setdefault(floor["name"], dict(floor))

    area_floors = {
        area_name: area.get("floor") for area_name, area in areas_by_name.items()
    }
    entity_records = []
    for entity in entities_by_key.values():
        attributes = entity.get("attributes") or {}
        area = entity.get("area")
        record = {
            "name": entity["name"],
            "domain": entity["domain"],
            "device_class": attributes.get("device_class"),
            "area": area,
            "floor": area_floors.get(area) if isinstance(area, str) else None,
        }
        entity_records.append(record)

    return FixtureData(
        entities=entity_records,
        slot_lists={
            "area": list(areas_by_name),
            "floor": list(floors_by_name),
        },
    )


def build_matcher(
    language: str, fixtures: Optional[FixtureData] = None
) -> intent_matcher.IntentMatcher:
    fixtures = fixtures or load_fixtures(language)
    matcher = intent_matcher.build_matcher(
        SERVER_ROOT,
        language,
        enabled_all(language),
        fixtures.entities,
        fixtures.slot_lists,
    )
    assert matcher is not None
    return matcher


def result_slots(result) -> Dict[str, object]:
    return intent_matcher.result_slots(result)


def _example_slots(language: str, fixtures: FixtureData) -> Dict[str, List[str]]:
    return {
        **{
            key: [value]
            for key, value in presets.load_example_values(SERVER_ROOT).items()
        },
        **s2p_intents.example_text_values(language),
        **fixtures.slot_lists,
    }


def _example_for(
    language: str,
    intent: str,
    combo: str,
    info: gating.EntityInfo,
    name_domains: Dict[str, str],
    slot_values: Dict[str, List[str]],
) -> Optional[str]:
    override = _EXAMPLE_OVERRIDES.get((language, intent, combo))
    if override is not None:
        return override

    capability = gating.required_capability(intent, combo)
    for block in s2p_intents.combo_blocks(language, intent, combo):
        sentences = block.get("sentences") or []
        domains = block.get("name_domains")
        if not gating.keep_block(
            domains,
            block.get("inferred_domain"),
            capability,
            gating.capability_domains(intent),
            info,
        ):
            continue
        if domains:
            allowed_names = set(info.names(domains, capability))
            entities = {
                name: name_domains[name]
                for name in allowed_names
                if name in name_domains
            }
            if not entities:
                continue
        else:
            entities = name_domains

        for sentence in sentences:
            resolved = s2p_intents.resolve_rules(sentence, language)
            text = presets.sample_sentence(
                resolved, domains, entities, slot_values
            ).strip()
            if text:
                return text
    return None


def enumerate_examples(language: str) -> List[dict]:
    """Render one matchable utterance per packaged intent/slot combination."""
    fixtures = load_fixtures(language)
    info = training.as_entity_info(fixtures.entities)
    name_domains = {record.name: record.domain for record in info.records}
    slot_values = _example_slots(language, fixtures)
    matcher = build_matcher(language, fixtures)

    corpus = []
    for intent, combo in combos(language):
        text = _example_for(
            language,
            intent,
            combo,
            info,
            name_domains,
            slot_values,
        )
        if not text:
            raise ValueError(f"{language}: cannot render {intent}/{combo}")
        result = matcher.match(text)
        if result is None:
            raise ValueError(
                f"{language}: rendered example does not match {intent}/{combo}: {text!r}"
            )
        corpus.append(
            {
                "language": language,
                "intent": intent,
                "combo": combo,
                "text": text,
                "expected_intent": result.intent.name,
                "expected_slots": result_slots(result),
                "wav": f"{intent}__{combo}.flac",
            }
        )
    return corpus


def synthesis_text(item: dict) -> str:
    """Text sent to TTS, including pronunciation-only punctuation hints."""
    return _SYNTHESIS_OVERRIDES.get(
        (item["language"], item["intent"], item["combo"]), item["text"]
    )


def manifest_path(language: str) -> Path:
    return WAV_ROOT / language / "manifest.json"


def load_manifests() -> List[dict]:
    items = []
    for language in languages():
        path = manifest_path(language)
        if path.exists():
            items.extend(json.loads(path.read_text(encoding="utf-8")))
    return items


def build_harness(language: str, grammar_path: Path) -> LanguageHarness:
    """Build one aggregate grammar and matcher for a language."""
    fixtures = load_fixtures(language)
    backend = models.resolve_backend(language, "auto")
    model_dir = models.resolve(None, MODELS_DIR, language, backend, tools_dir=TOOLS_DIR)
    if model_dir is None:
        raise ValueError(f"No acoustic model for {language}/{backend}")

    templates, list_values = training.assemble(
        SERVER_ROOT,
        language,
        enabled_all(language),
        [],
        fixtures.entities,
        fixtures.slot_lists,
    )
    if not templates:
        raise ValueError(f"No grammar templates for {language}")
    training.train(
        backend,
        model_dir,
        language,
        templates,
        list_values,
        grammar_path,
    )
    recognizer = load_recognizer(
        backend,
        model_dir,
        language=language,
        token_bonus=models.default_token_bonus(backend),
    )
    recognizer.load(grammar_path)
    return LanguageHarness(
        recognizer=recognizer,
        matcher=build_matcher(language, fixtures),
        max_score=models.default_max_score(backend),
    )


def transcribe_clip(recognizer: Recognizer, wav_path: Path):
    """Resample an audio clip and run the production audio front end."""
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = resample_audio(audio, sample_rate, SAMPLE_RATE)
    audio = audio_frontend.prepare_audio(audio)
    return recognizer.transcribe(audio)
