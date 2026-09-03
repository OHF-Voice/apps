"""Command-line transcription: ``python -m speech_to_phrase``.

Pick a backend and model, compile a grammar from a sentence-template file, then
transcribe one or more WAV files. One JSON object is written to stdout per audio
file (JSONL), in input order:

    {"file": "clip.wav", "text": "turn on the lights", "score": 1.83, "margin": 4.2}

``score`` is the constrained-vs-greedy penalty per token (lower is better) and
``margin`` the gap to the second-best parse; both are ``null`` when there was no
parse. Example:

    python -m speech_to_phrase \\
        --backend citrinet --model local/stt_en_citrinet_512_gamma_0_25 \\
        --templates tests/en/sentences.txt --language en \\
        tests/en/*.wav
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

from . import load_recognizer


def read_templates(path: Path) -> List[str]:
    """One template per line; blank lines and ``#`` comments are ignored."""
    templates: List[str] = []
    with open(path, "r", encoding="utf-8") as templates_file:
        for line in templates_file:
            line = line.strip()
            if line and not line.startswith("#"):
                templates.append(line)
    if not templates:
        raise ValueError(f"No templates found in {path}")
    return templates


def _finite_or_none(value: float) -> Optional[float]:
    """JSON has no infinity; emit ``null`` for a missing score/margin."""
    return value if math.isfinite(value) else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m speech_to_phrase", description=__doc__
    )
    parser.add_argument(
        "--backend",
        default="citrinet",
        choices=["citrinet", "coqui"],
        help="Acoustic backend (default: citrinet)",
    )
    parser.add_argument("--model", required=True, help="Path to the model directory")
    parser.add_argument(
        "--templates",
        required=True,
        help="Sentence-template file (one template per line)",
    )
    parser.add_argument(
        "--language", default="en", help="Locale for number spellout (default: en)"
    )
    parser.add_argument(
        "--beam",
        type=float,
        default=None,
        help="Decode pruning beam (log-prob units); default per backend",
    )
    parser.add_argument(
        "--stt-binary", help="Coqui: path to the stt_onlyprobs helper binary"
    )
    parser.add_argument("wavs", nargs="+", help="One or more WAV files to transcribe")
    args = parser.parse_args(argv)

    backend_kwargs = {}
    if args.backend == "coqui" and args.stt_binary:
        backend_kwargs["stt_binary"] = args.stt_binary

    print(f"Loading {args.backend} model from {args.model} ...", file=sys.stderr)
    recognizer = load_recognizer(
        args.backend,
        args.model,
        language=args.language,
        beam=args.beam,
        **backend_kwargs,
    )

    print(f"Building grammar from {args.templates} ...", file=sys.stderr)
    recognizer.train(read_templates(Path(args.templates)))

    exit_code = 0
    for wav in args.wavs:
        record = {"file": wav}
        try:
            result = recognizer.transcribe(wav)
            record["text"] = result.text
            record["score"] = _finite_or_none(result.score)
            record["margin"] = _finite_or_none(result.margin)
        except Exception as err:  # noqa: BLE001 - report and keep going
            exit_code = 1
            record["text"] = ""
            record["error"] = str(err)
            print(f"ERROR {wav}: {err}", file=sys.stderr)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
