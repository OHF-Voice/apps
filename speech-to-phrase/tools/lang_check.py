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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
    # Fold the orthographic differences between what TTS is handed and what a
    # lowercase acoustic vocab emits (German "schließe" decodes as "schliesse").
    text = unicodedata.normalize("NFC", text.strip().casefold()).replace("ß", "ss")
    text = text.replace("-", " ")
    return " ".join(text.split())


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
WANTED_DOMAINS = ("light", "fan", "cover", "lock", "switch", "media_player", "climate", "sensor")


def load_fixtures(intents_repo: Path, language: str, per_domain: int = 2):
    sys.path.insert(0, str(intents_repo))
    from script.intentfest.util import load_fixtures as _load  # noqa: PLC0415

    fixtures = _load(language)
    entities: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for entity in fixtures.get("entities") or []:
        domain = str(entity.get("id", "")).split(".")[0]
        name = str(entity.get("name") or "").strip()
        if domain not in WANTED_DOMAINS or not name:
            continue
        if counts.get(domain, 0) >= per_domain:
            continue
        entities[name] = domain
        counts[domain] = counts.get(domain, 0) + 1

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
    repo_root = Path(__file__).resolve().parent.parent
    combos = bi.available_combos(repo_root, language, meta)
    enabled = bi.default_enabled(combos, "optional")
    templates, list_values = training.assemble(
        repo_root, language, enabled, [], entities, slot_lists
    )
    return templates, list_values


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
    if sr != SAMPLE_RATE:
        from librosa import resample  # noqa: PLC0415

        data = resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
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
    from vad import normalize_level, trim_silence  # noqa: PLC0415

    token_bonus = args.token_bonus
    if token_bonus is None:
        token_bonus = 2.0 if args.backend == "citrinet" else 0.0

    rec = load_recognizer(
        args.backend, args.model, language=args.language, token_bonus=token_bonus
    )
    rec.train(templates, list_values=list_values)
    print(f"trained ({args.backend}, token_bonus={token_bonus})")

    gate = GATE[args.backend]
    tts_language = TTS_LANGUAGE.get(args.language, f"{args.language}-{args.language.upper()}")
    in_grammar = {norm(t) for t in templates}

    cases = lean_examples(args.intents_repo, args.language)
    if args.limit:
        cases = cases[: args.limit]

    results = []
    exact = accepted = 0
    for combo, example in cases:
        audio = tts_wav(example, args.engine_id, tts_language, args.cache_dir)
        audio = trim_silence(normalize_level(audio))
        result = rec.transcribe(audio)
        heard = norm(result.text if result else "")
        score = float(result.score) if result else float("inf")
        is_exact = heard == expected_decode(example, args.language)
        gated = score > gate
        if is_exact:
            exact += 1
            if not gated:
                accepted += 1
        status = "EXACT" if is_exact else ("OTHER" if heard else "NO_PARSE")
        if gated:
            status += "/gated"
        results.append(
            {"combo": combo, "spoke": example, "heard": heard,
             "score": round(score, 2), "status": status}
        )
        flag = " " if is_exact and not gated else "!"
        print(f"{flag} {combo:38} score={score:6.2f} {status:12} "
              f"spoke={example!r} heard={heard!r}")

    total = len(results)
    print(f"\n{args.language}: EXACT {exact}/{total}, "
          f"accepted (exact and score<={gate}) {accepted}/{total}")
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"language": args.language, "backend": args.backend,
                        "templates": len(templates), "results": results},
                       ensure_ascii=False, indent=1)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
