import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import app as web_app  # noqa: E402
import intent_matcher  # noqa: E402
import numeric_ranges  # noqa: E402
import s2p_intents  # noqa: E402
import settings  # noqa: E402
import training  # noqa: E402


def test_parse_union_and_canonicalize_overlaps():
    selection = numeric_ranges.parse("3,5,8, 10-100/10", (0, 100, 1))

    assert selection.values == (3, 5, 8, *range(10, 101, 10))
    assert selection.expression == "3, 5, 8, 10-100/10"
    assert selection.grammar == "3,5,8,10..100/10"

    overlap = numeric_ranges.parse("1-10, 10-100/5", (0, 100, 1))
    assert overlap.values == (*range(1, 11), *range(15, 101, 5))
    assert overlap.expression == "1-10, 15-100/5"


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("", "Full range"),
        ("10-1", "ascending"),
        ("1-101", "outside"),
        ("1-10/0", "greater than zero"),
        ("1-10/4", "does not land"),
        ("1050", "package step"),
        ("1000-2000/150", "multiple of the package step"),
    ],
)
def test_parse_rejects_invalid_restrictions(expression, message):
    definition = (1000, 10000, 100) if "package step" in message else (0, 100, 1)
    with pytest.raises(numeric_ranges.NumericRangeError, match=message):
        numeric_ranges.parse(expression, definition)


def test_payload_rejects_unknown_lists_and_preserves_explicit_full_range():
    definitions = {"brightness": (0, 100, 1)}
    assert numeric_ranges.parse_payload({"brightness": "0-100"}, definitions) == {
        "brightness": None
    }
    assert numeric_ranges.active_selections({"brightness": None}, definitions) == {}
    with pytest.raises(numeric_ranges.NumericRangeError, match="unknown"):
        numeric_ranges.parse_payload({"mystery": "1-10"}, definitions)


def test_persistence_uses_structured_segments_and_preserves_settings(tmp_path):
    definitions = {"brightness": (0, 100, 1)}
    settings.set_max_score(tmp_path, "en", 4.5)
    selection = numeric_ranges.parse("10-100/10", definitions["brightness"])

    numeric_ranges.save(tmp_path, "en", {"brightness": selection})

    stored = json.loads(settings.path(tmp_path, "en").read_text())
    assert stored["max_score"] == 4.5
    assert stored["numeric_ranges"] == {
        "brightness": [{"start": 10, "stop": 100, "step": 10}]
    }
    assert numeric_ranges.load(tmp_path, "en", definitions) == {"brightness": selection}

    numeric_ranges.save(tmp_path, "en", {"brightness": None})
    assert settings.load(tmp_path, "en")["numeric_ranges"] == {"brightness": None}
    assert numeric_ranges.load(tmp_path, "en", definitions) == {}

    numeric_ranges.save(tmp_path, "en", {})
    assert settings.load(tmp_path, "en") == {
        "max_score": 4.5,
        "numeric_ranges": {},
    }
    assert numeric_ranges.load(tmp_path, "en", definitions) == {
        "brightness": numeric_ranges.recommended(
            "brightness", definitions["brightness"]
        )
    }


def test_restriction_changes_grammar_cost_and_hassil_matcher():
    selection = numeric_ranges.parse("10-100/10", (0, 100, 1))
    ranges = {"brightness": selection}

    templates, _referenced = s2p_intents.grammar_templates(
        ["set brightness to {brightness:brightness}"], "en", ranges
    )
    assert templates == ["set brightness to {10..100/10}"]
    assert training.phrase_count(templates, {}) == 10

    matcher = intent_matcher.build_matcher(
        ROOT,
        "en",
        [["HassLightSet", "name_brightness"]],
        training.DEV_ENTITY_RECORDS,
        training.DEV_SLOT_LISTS,
        range_overrides=ranges,
    )
    assert matcher is not None
    accepted = matcher.match("set the kitchen lamp brightness to twenty percent")
    rejected = matcher.match("set the kitchen lamp brightness to twenty one percent")
    assert accepted is not None
    assert intent_matcher.result_slots(accepted)["brightness"] == 20
    assert rejected is None


def test_restricted_matcher_preserves_package_multiplier():
    selection = numeric_ranges.parse("10-100/10", (0, 100, 1))
    matcher = intent_matcher.build_matcher(
        ROOT,
        "en",
        [["HassSetVolumeRelative", "default"]],
        [{"name": "speaker", "domain": "media_player"}],
        training.DEV_SLOT_LISTS,
        range_overrides={"volume_step_down": selection},
    )
    assert matcher is not None

    result = matcher.match("decrease the volume by twenty percent")
    assert result is not None
    assert intent_matcher.result_slots(result)["volume_step"] == -20


def test_presets_start_with_recommended_and_remain_package_subsets():
    presets = numeric_ranges.presets("timer_minutes", (1, 100, 1))
    by_id = {preset["id"]: preset for preset in presets}

    assert presets[0]["id"] == "recommended"
    assert by_id["recommended"]["expression"] == "1-10, 15-100/5"
    assert by_id["recommended"]["count"] == 28
    assert by_id["full"]["count"] == 100
    assert by_id["multiples_5"]["expression"] == "5-100/5"
    assert by_id["multiples_10"]["expression"] == "10-100/10"
    assert by_id["custom"]["expression"] == ""


def test_recommended_ranges_cover_each_current_numeric_list():
    expected = {
        "temperature": "5-35",
        "position": "0-100/10",
        "fan_speed": "10-100/10",
        "brightness": "10-100/10",
        "color_temperature": "1000-10000/500",
        "volume": "10-100/10",
        "volume_level": "10-100/10",
        "volume_step_up": "5-50/5",
        "volume_step_down": "5-50/5",
        "timer_seconds": "1-10, 15-100/5",
        "timer_minutes": "1-10, 15-100/5",
        "timer_hours": "1-12, 24-96/24",
        "timer_range_seconds": "1-10, 15-120/5",
        "timer_range_minutes": "1-10, 15-120/5",
        "timer_range_hours": "1-12, 24",
    }
    seen = {}
    for language in s2p_intents.languages():
        for name, definition in s2p_intents.range_list_definitions(language).items():
            selection = numeric_ranges.recommended(name, definition)
            seen[name] = selection.expression
            assert set(selection.values) <= set(
                numeric_ranges.package_values(definition)
            )

    assert seen == expected


def test_api_rejects_invalid_ranges_before_writing_and_round_trips(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web_app.models, "resolve", lambda *_args, **_kwargs: None)
    cfg = Namespace(
        backend="citrinet",
        data=str(tmp_path),
        language="en",
        model=None,
        models_dir=str(tmp_path / "models"),
        max_score=None,
        token_bonus=None,
        default_importance="usable",
        hass_token=None,
        hass_api="http://unused",
        entities_file=None,
        slot_lists_file=None,
        sentence_triggers=False,
        question_answers=False,
        refresh_interval=0,
    )
    client = web_app.create_app(cfg).test_client()

    initial_state = client.get("/api/state").get_json()
    initial_brightness = next(
        row for row in initial_state["numeric_ranges"] if row["name"] == "brightness"
    )
    assert initial_brightness["expression"] == "10-100/10"
    assert initial_brightness["presets"][0]["id"] == "recommended"

    invalid = client.post(
        "/api/save",
        json={
            "lang": "en",
            "enabled": [["HassLightSet", "name_brightness"]],
            "commands": [],
            "numeric_ranges": {"brightness": "10-101/10"},
        },
    )
    assert invalid.status_code == 400
    assert not (tmp_path / "en" / "enabled.json").exists()

    valid = client.post(
        "/api/save",
        json={
            "lang": "en",
            "enabled": [["HassLightSet", "name_brightness"]],
            "commands": [],
            "overrides": {},
            "numeric_ranges": {"brightness": "10-100/10"},
        },
    )
    assert valid.status_code == 200
    assert valid.get_json()["ok"] is True

    state = client.get("/api/state").get_json()
    brightness = next(
        row for row in state["numeric_ranges"] if row["name"] == "brightness"
    )
    assert brightness["expression"] == "10-100/10"
    assert brightness["count"] == 10

    monkeypatch.setattr(web_app, "_model_dir_for", lambda *_args: tmp_path / "model")
    monkeypatch.setattr(
        web_app,
        "_ensure_trained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile failed")),
    )
    failed = client.post(
        "/api/save",
        json={
            "lang": "en",
            "enabled": [["HassLightSet", "name_brightness"]],
            "commands": [],
            "numeric_ranges": {"brightness": "20-100/20"},
        },
    )
    assert failed.get_json()["ok"] is False
    loaded = numeric_ranges.load(
        tmp_path, "en", s2p_intents.range_list_definitions("en")
    )
    assert loaded["brightness"].expression == "10-100/10"
