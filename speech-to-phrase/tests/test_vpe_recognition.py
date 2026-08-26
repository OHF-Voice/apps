#!/usr/bin/env python3
"""Regression check for human Voice Preview Edition recordings.

The local Citrinet model is intentionally not committed. When it is available,
this builds the same selected English production grammar and decodes every WAV
under ``tests/wav/mike`` through the production audio front-end.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audio_frontend import prepare_audio  # noqa: E402
from models import default_max_score  # noqa: E402
from overrides import Overrides  # noqa: E402
import training  # noqa: E402
from wyoming_server import build_info  # noqa: E402

from speech_to_phrase import load_recognizer  # noqa: E402

MODEL = ROOT / "local/models/stt_en_citrinet_512"
ENABLED = ROOT / "local/data/en/enabled.json"
WAV_ROOT = ROOT / "tests/wav/mike"
OOV_ROOT = ROOT / "tests/wav/oov"

# Canonical Home Assistant registry names represented by the recordings.
ENTITIES = {
    "basement lights": "light",
    "office lamp": "light",
    "overhead light": "light",
    "standing light": "light",
}
CUSTOM_COMMANDS = [{"sentences": ["start oliver workout"], "mode": "stt"}]

CONTRACTIONS = {"whats": "what's"}


def expected_transcript(path: Path) -> str:
    """Read a slug from a descriptive filename or a numbered file's directory."""
    slug = path.parent.name if path.stem.isdigit() else path.stem
    words = slug.split("-")
    words = [CONTRACTIONS.get(word, word) for word in words]
    return " ".join(words)


def main() -> int:
    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"{'ok  ' if condition else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
        ok &= condition

    # A generated inflection maps back to the canonical HA entity. It must not be
    # generated when another entity already owns that spoken form.
    variants = Overrides().pairs("entities", ["Basement lights"], language="en")
    check(
        "English light inflection",
        ("Basement light", "Basement lights") in variants,
        repr(variants),
    )
    collision = Overrides().pairs(
        "entities", ["Desk light", "Desk lights"], language="en"
    )
    check(
        "canonical-name collision is not aliased",
        collision == [("Desk light", "Desk light"), ("Desk lights", "Desk lights")],
        repr(collision),
    )

    quiet = np.array([0.0, 0.001, -0.002], dtype=np.float32)
    check(
        "production front-end preserves quiet samples",
        np.array_equal(prepare_audio(quiet), quiet),
    )
    program = build_info("en", MODEL.name).asr[0]
    check(
        "VPE processing preferences",
        not program.prefers_auto_gain_enabled
        and not program.prefers_noise_reduction_enabled,
    )

    if not MODEL.is_dir() or not ENABLED.exists():
        print("SKIP recognition: local English Citrinet artifacts are unavailable")
        return 0 if ok else 1

    wavs = sorted(WAV_ROOT.rglob("*.wav"))
    check("recording inventory", bool(wavs), f"{len(wavs)} WAVs")
    enabled = json.loads(ENABLED.read_text())
    templates, list_values = training.assemble(
        ROOT,
        "en",
        enabled,
        CUSTOM_COMMANDS,
        ENTITIES,
        training.DEV_SLOT_LISTS,
    )
    recognizer = load_recognizer(
        "citrinet", MODEL, language="en", token_bonus=2.0
    )
    recognizer.train(templates, list_values)
    gate = default_max_score("citrinet")

    recognized = 0
    for wav in wavs:
        samples, sample_rate = sf.read(wav, dtype="float32", always_2d=False)
        check(f"{wav.relative_to(WAV_ROOT)} sample rate", sample_rate == 16000)
        result = recognizer.transcribe(prepare_audio(samples))
        expected = expected_transcript(wav)
        passed = result.text == expected and result.score <= gate
        recognized += int(passed)
        check(
            str(wav.relative_to(WAV_ROOT)),
            passed,
            f"expected={expected!r}, got={result.text!r}, score={result.score:.3f}",
        )

    rejected = 0
    oov_wavs = sorted(OOV_ROOT.glob("*.wav"))
    for wav in oov_wavs:
        samples, _sample_rate = sf.read(wav, dtype="float32", always_2d=False)
        result = recognizer.transcribe(prepare_audio(samples))
        rejected += int(result.score > gate)
        check(
            f"OOV {wav.name}",
            result.score > gate,
            f"got={result.text!r}, score={result.score:.3f}",
        )

    print(
        f"\nVPE exact+accepted: {recognized}/{len(wavs)}; "
        f"OOV rejected: {rejected}/{len(oov_wavs)}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
