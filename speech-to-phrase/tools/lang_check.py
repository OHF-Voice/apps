#!/usr/bin/env python3
"""Round-trip check for a language's Speech-to-Phrase templates.

Answers one question per language: do the ``speech_to_phrase``-tagged blocks in
home-assistant-intents actually produce a grammar this language's acoustic model
can decode? It runs the real path end to end --

    package JSON --> training.assemble --> FST grammar --> Recognizer
    example sentence --> HA Cloud TTS (that language's voice) --> decode

-- and classifies each decode the way the production score gate would.

This is a sanity check, not an accuracy benchmark: TTS is not human speech and
one clip per block is a small sample. It catches the failures that matter when
adding a language -- templates that do not transpile, a grammar that will not
compile, and commands the model cannot hear at all.

Usage:

    export HA_TOKEN=<long-lived token>          # HA_URL defaults to homeassistant.local
    python3 tools/lang_check.py --language de \\
        --s2p-json-dir /path/to/built/speech_to_phrase \\
        --intents-repo /path/to/intent-sentences \\
        --model /path/to/models/stt_de_citrinet_1024

``--s2p-json-dir`` points at ``speech_to_phrase/`` as built by the
intents-package ``script/merged_output.py``; omit it to use the installed
``home_assistant_intents`` package as-is.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
from vendored_lib import bind as _bind_vendored_lib  # noqa: E402

_bind_vendored_lib()

from speech_to_phrase.audio import resample as resample_audio  # noqa: E402

HA_URL = os.environ.get("HA_URL", "http://homeassistant.local:8123")
TOKEN = os.environ.get("HA_TOKEN", "")
SAMPLE_RATE = 16000

# Per-backend score gate: at or below this a transcript is accepted locally,
# above it the utterance is handed to the cloud fallback (models.DEFAULT_MAX_SCORE).
GATE = {"citrinet": 5.0, "coqui": 2.0}

# HA Cloud TTS locale per language.
TTS_LANGUAGE = {
    "ca": "ca-ES",
    "cs": "cs-CZ",
    "de": "de-DE",
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "nl": "nl-NL",
}


def norm(text: str) -> str:
    """Case/whitespace fold only.

    Deliberately does not fold anything else: what the recognizer emits is
    exactly what Home Assistant will try to match, so rewriting it here would
    hide real failures (and inventing a fold -- e.g. German ss/ß -- invents
    failures too: the model emits "schließe", spelled correctly).
    """
    # lower(), not casefold(): casefold maps "ß" to "ss", which would rewrite a
    # correct German transcript into one Home Assistant cannot match and make
    # the check report failures that do not exist.
    return " ".join(unicodedata.normalize("NFC", text.strip().lower()).split())


def expected_decode(example: str, language: str) -> str:
    """What a correct decode of ``example`` looks like.

    The grammar spells numbers out (a range is compiled to its number words), so
    an example written "5 Minuten" can only ever come back as "fünf minuten".
    Comparing the raw digits would score every numeric command as a failure.
    """
    from speech_to_phrase.templates import _spellout_words  # noqa: PLC0415

    words = []
    for word in example.split():
        stripped = word.strip(".,!?%")
        if stripped.isdigit():
            try:
                words.extend(_spellout_words(int(stripped), language))
                continue
            except Exception:  # noqa: BLE001
                pass
        words.append(word)
    return norm(" ".join(words))


# --------------------------------------------------------------------------
# Fixtures: {name}/{area}/{floor} values in the target language, taken from the
# intents repo's own test fixtures so the grammar is trained on realistic
# localized names rather than English placeholders.
# --------------------------------------------------------------------------
WANTED_DOMAINS = (
    "light", "fan", "cover", "lock", "valve", "switch", "media_player",
    "climate", "sensor",
)


def load_fixtures(intents_repo: Path, language: str):
    """Every entity of an addressable domain, not a sample of them.

    The examples are written against the whole fixture set, so training on a
    per-domain sample makes an example name an entity the grammar cannot emit
    -- which then fails as an out-of-grammar utterance and looks like a template
    problem. Taking all of them also builds a more realistic grammar: a real
    install has many devices per domain, not two.
    """
    sys.path.insert(0, str(intents_repo))
    from script.intentfest.util import load_fixtures as _load  # noqa: PLC0415

    fixtures = _load(language)
    entities: Dict[str, str] = {}
    for entity in fixtures.get("entities") or []:
        domain = str(entity.get("id", "")).split(".")[0]
        name = str(entity.get("name") or "").strip()
        if domain in WANTED_DOMAINS and name:
            entities[name] = domain

    areas = [str(a.get("name")) for a in (fixtures.get("areas") or [])][:3]
    floors = [str(f.get("name")) for f in (fixtures.get("floors") or [])][:2]
    return entities, areas, floors


# --------------------------------------------------------------------------
# Grammar
# --------------------------------------------------------------------------
def build_grammar(language: str, entities: Dict[str, str], areas, floors):
    import presets as bi  # noqa: PLC0415
    import training  # noqa: PLC0415

    slot_lists = {"area": list(areas), "floor": list(floors)}
    meta = bi.load_intents_meta()
    combos = bi.available_combos(language, meta)
    enabled = bi.default_enabled(combos, "optional")
    templates, list_values = training.assemble(
        language, enabled, [], entities, slot_lists
    )
    return templates, list_values


def build_matcher(language: str, entities: Dict[str, str], areas, floors):
    """A hassil recognizer over the same Speech-to-Phrase blocks the grammar was
    built from -- i.e. what Home Assistant does with the transcript we emit."""
    import intent_matcher  # noqa: PLC0415
    import presets as bi  # noqa: PLC0415

    meta = bi.load_intents_meta()
    combos = bi.available_combos(language, meta)
    enabled = bi.default_enabled(combos, "optional")
    return intent_matcher.build_matcher(
        language, enabled, entities,
        {"area": list(areas), "floor": list(floors)},
    )


def matched_combo(matcher, text: str) -> Optional[Tuple[str, str]]:
    """(intent, slot_combination) hassil assigns to ``text``, or None."""
    if matcher is None or not text:
        return None
    try:
        result = matcher.match(text)
    except Exception:  # noqa: BLE001
        return None
    if result is None:
        return None
    metadata = getattr(result, "intent_metadata", None) or {}
    return (result.intent.name, str(metadata.get("combo") or ""))


def lean_examples(intents_repo: Path, language: str) -> List[Tuple[str, str]]:
    """(combo_key, example) for every speech_to_phrase-tagged block."""
    out: List[Tuple[str, str]] = []
    lang_dir = intents_repo / "sentences" / language
    for path in sorted(lang_dir.glob("*/*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for block in doc.get("data") or []:
            if not block.get("speech_to_phrase"):
                continue
            example = str(block.get("example") or "").strip()
            if example:
                out.append((f"{path.parent.name}/{path.stem}", example))
    return out


def names_in_grammar(list_values: Dict[str, List[str]]) -> set:
    """Every entity name the compiled grammar can actually produce."""
    names = set()
    for key, values in list_values.items():
        if key.startswith("name"):
            names.update(norm(v) for v in values)
    return names


def unreachable_name(example: str, entities: Dict[str, str], in_grammar: set) -> str:
    """The entity an example names, when the grammar cannot produce it.

    A tagged block can be dropped from the Speech-to-Phrase grammar and still
    belong in home-assistant-intents: capability gating removes, say, the valve
    branch of HassSetPosition when no valve supports set_position. An example
    built on that block names an entity no template can emit, so decoding it
    measures nothing about the templates -- it just samples what out-of-grammar
    audio happens to hit. Report those separately instead of scoring them.
    """
    spoken = norm(example)
    for name in entities:
        if norm(name) in spoken and norm(name) not in in_grammar:
            return name
    return ""


# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------
def tts_wav(message: str, engine_id: str, tts_language: str, cache_dir: Path) -> np.ndarray:
    key = hashlib.sha1(f"{engine_id}|{tts_language}|{message}".encode()).hexdigest()
    cached = cache_dir / f"{key}.wav"
    if cached.exists():
        return sf.read(cached, dtype="float32")[0]

    if not TOKEN:
        raise SystemExit("HA_TOKEN is not set and this clip is not cached")

    req = urllib.request.Request(
        f"{HA_URL}/api/tts_get_url",
        data=json.dumps(
            {"engine_id": engine_id, "message": message, "language": tts_language}
        ).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        url = json.loads(resp.read())["url"]
    with urllib.request.urlopen(url, timeout=30) as resp:
        data, sr = sf.read(io.BytesIO(resp.read()), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = resample_audio(data, sr, SAMPLE_RATE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sf.write(cached, data, SAMPLE_RATE)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--backend", default="citrinet", choices=["citrinet", "coqui"])
    ap.add_argument("--intents-repo", required=True, type=Path)
    ap.add_argument("--s2p-json-dir", type=Path, default=None,
                    help="speech_to_phrase/ dir built by intents-package (overrides the "
                         "installed home_assistant_intents data)")
    ap.add_argument("--engine-id", default="tts.home_assistant_cloud")
    ap.add_argument("--cache-dir", type=Path, default=Path("tests/wav/.tts_cache"))
    ap.add_argument("--token-bonus", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if args.s2p_json_dir:
        import home_assistant_intents  # noqa: PLC0415

        home_assistant_intents._SPEECH_TO_PHRASE_DIR = args.s2p_json_dir  # noqa: SLF001

    import s2p_intents  # noqa: PLC0415

    if not s2p_intents.has_language(args.language):
        print(f"!! no Speech-to-Phrase templates for {args.language}")
        return 1

    entities, areas, floors = load_fixtures(args.intents_repo, args.language)
    print(f"fixtures: {len(entities)} entities, areas={areas}, floors={floors}")

    templates, list_values = build_grammar(args.language, entities, areas, floors)
    print(f"grammar: {len(templates)} templates, "
          f"{sum(len(v) for v in list_values.values())} list values")
    if not templates:
        print("!! grammar is empty")
        return 1

    from speech_to_phrase import load_recognizer  # noqa: PLC0415
    from audio_frontend import prepare_audio  # noqa: PLC0415

    token_bonus = args.token_bonus
    if token_bonus is None:
        token_bonus = 2.0 if args.backend == "citrinet" else 0.0

    rec = load_recognizer(
        args.backend, args.model, language=args.language, token_bonus=token_bonus
    )
    rec.train(templates, list_values=list_values)
    print(f"trained ({args.backend}, token_bonus={token_bonus})")

    matcher = build_matcher(args.language, entities, areas, floors)
    print(f"matcher: {'built' if matcher else 'UNAVAILABLE'}")

    gate = GATE[args.backend]
    tts_language = TTS_LANGUAGE.get(args.language, f"{args.language}-{args.language.upper()}")

    cases = lean_examples(args.intents_repo, args.language)
    if args.limit:
        cases = cases[: args.limit]

    in_grammar = names_in_grammar(list_values)
    results = []
    skipped = []
    exact = usable = accepted = 0
    for combo, example in cases:
        unreachable = unreachable_name(example, entities, in_grammar)
        if unreachable:
            skipped.append((combo, example, unreachable))
            continue
        audio = tts_wav(example, args.engine_id, tts_language, args.cache_dir)
        result = rec.transcribe(prepare_audio(audio))
        heard = norm(result.text if result else "")
        score = float(result.score) if result else float("inf")
        gated = score > gate

        is_exact = heard == expected_decode(example, args.language)
        # What actually matters is not the string but whether Home Assistant
        # would resolve the transcript to the command that was spoken. The
        # decoder legitimately picks a different in-grammar realization -- Dutch
        # drops the article ("sluit woonkamerraam" for "sluit de Woonkamerraam")
        # -- and that is the same command, not an error.
        want = matched_combo(matcher, expected_decode(example, args.language))
        got = matched_combo(matcher, heard)
        same_command = bool(got) and got == want

        if is_exact:
            exact += 1
        if same_command:
            usable += 1
            if not gated:
                accepted += 1

        if is_exact:
            status = "EXACT"
        elif same_command:
            status = "SAME_CMD"
        elif heard:
            status = "WRONG_CMD"
        else:
            status = "NO_PARSE"
        if gated:
            status += "/gated"
        results.append(
            {"combo": combo, "spoke": example, "heard": heard,
             "score": round(score, 2), "status": status,
             "want": want, "got": got}
        )
        flag = " " if same_command and not gated else "!"
        print(f"{flag} {combo:38} score={score:6.2f} {status:14} "
              f"spoke={example!r} heard={heard!r}")

    total = len(results)
    for combo, example, name in skipped:
        print(f"~ {combo:38} {'NOT_IN_GRAMMAR':14} "
              f"'{name}' is gated out of the grammar; not scored ({example!r})")
    print(f"\n{args.language}: EXACT {exact}/{total}, "
          f"same command {usable}/{total}, "
          f"accepted (same command and score<={gate}) {accepted}/{total}"
          + (f"; {len(skipped)} not scored (block gated out of the grammar)"
             if skipped else ""))
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"language": args.language, "backend": args.backend,
                        "templates": len(templates), "results": results},
                       ensure_ascii=False, indent=1)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
