"""Benchmark the Citrinet + FST constrained speech-to-text pipeline.

Given a file of sentence templates and a dataset of audio files with their
expected transcripts, this builds the grammar FST once (training) and then
transcribes every audio file, reporting accuracy (sentence-level exact match,
word error rate, character error rate) and speed (per-utterance latency and
real-time factor).

Usage:

    python -m citrinet.benchmark \\
        --templates templates.txt \\
        --dataset dataset.jsonl \\
        [--model stt_en_citrinet_512] [--language en]

Templates file: one template per line (same syntax as the `stt_local.sentences`
config), blank lines and lines starting with '#' are ignored.

Dataset (--dataset), auto-detected:
  * a directory   : every *.wav inside is a sample; the expected transcript is
                    the filename stem with '-'/'_' turned into spaces
                    (e.g. turn-on-the-lights.wav -> "turn on the lights")
  * .jsonl / .json : one JSON object per line, e.g.
        {"audio_filepath": "a.wav", "text": "what time is it"}
    (keys `audio`/`audio_path` and `transcript`/`expected` are also accepted)
  * .tsv           : <audio_path>\\t<expected transcript>
  * .csv           : <audio_path>,<expected transcript>

Relative audio paths are resolved against the dataset file's directory.

Because the grammar emits spelled-out numbers ("fifty"), digits in the expected
transcripts are spelled out the same way before scoring (disable with
--no-spell-numbers).
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import load_recognizer

# ----------------------------------------------------------------------------
# Dataset / template loading
# ----------------------------------------------------------------------------


@dataclass
class Sample:
    audio_path: Path
    expected: str
    negative: bool = False  # out-of-grammar; success means the gating rejects it


# Directory names whose audio is treated as out-of-grammar negatives.
OOV_DIR_NAMES = {"oov", "out-of-grammar", "out_of_grammar", "negative", "negatives"}


def load_templates(path: Path) -> List[str]:
    templates: List[str] = []
    with open(path, "r", encoding="utf-8") as templates_file:
        for line in templates_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            templates.append(line)
    if not templates:
        raise ValueError(f"No templates found in {path}")
    return templates


def _first_key(obj: Dict[str, str], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in obj and obj[key]:
            return str(obj[key])
    return None


def load_dataset(path: Path) -> List[Sample]:
    if path.is_dir():
        samples = []
        for wav in sorted(path.rglob("*.wav")):
            parent_dirs = {p.lower() for p in wav.relative_to(path).parts[:-1]}
            negative = bool(parent_dirs & OOV_DIR_NAMES)
            samples.append(
                Sample(
                    audio_path=wav.resolve(),
                    expected=wav.stem.replace("-", " ").replace("_", " "),
                    negative=negative,
                )
            )
        if not samples:
            raise ValueError(f"No .wav files found in directory {path}")
        return samples

    base_dir = path.parent
    samples = []
    suffix = path.suffix.lower()

    def add(audio: Optional[str], text: Optional[str], line_no: int) -> None:
        if not audio:
            raise ValueError(f"{path}:{line_no}: missing audio path")
        if text is None:
            raise ValueError(f"{path}:{line_no}: missing expected transcript")
        audio_path = Path(audio)
        if not audio_path.is_absolute():
            audio_path = (base_dir / audio_path).resolve()
        samples.append(Sample(audio_path=audio_path, expected=text))

    if suffix in (".jsonl", ".json", ".ndjson"):
        with open(path, "r", encoding="utf-8") as dataset_file:
            for line_no, line in enumerate(dataset_file, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                add(
                    _first_key(obj, ("audio_filepath", "audio", "audio_path", "wav")),
                    _first_key(obj, ("text", "transcript", "expected", "sentence")),
                    line_no,
                )
    elif suffix in (".tsv", ".csv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(path, "r", encoding="utf-8", newline="") as dataset_file:
            reader = csv.reader(dataset_file, delimiter=delimiter)
            for line_no, row in enumerate(reader, start=1):
                if not row or (row[0].strip().startswith("#")):
                    continue
                if len(row) < 2:
                    raise ValueError(f"{path}:{line_no}: expected 2 columns")
                add(row[0].strip(), delimiter.join(row[1:]).strip(), line_no)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix!r}")

    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


# ----------------------------------------------------------------------------
# Audio helpers (Recognizer.transcribe loads/resamples audio itself)
# ----------------------------------------------------------------------------


def audio_seconds(path: Path) -> float:
    """Duration in seconds of any soundfile-readable audio."""
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def spell_out_numbers(text: str, locale: str) -> str:
    """Replace standalone integer tokens with words, matching the grammar.

    Mirrors templates._expand_ref so "50 percent" scores against "fifty
    percent" and "1 hour 7 minutes" against "one hour seven minutes".
    """
    from icu_rbnf import spellout

    out: List[str] = []
    for token in text.split():
        if token.lstrip("-").isdigit():
            spelled = spellout(int(token), locale)
            # icu inserts Unicode format (Cf) chars (e.g. soft hyphens) as
            # syllable hints in some locales; drop them so the expected text
            # matches the grammar's templates._spellout_words, then treat a
            # real hyphen as a word boundary.
            spelled = "".join(ch for ch in spelled if unicodedata.category(ch) != "Cf")
            out.append(unicodedata.normalize("NFC", spelled).replace("-", " "))
        else:
            out.append(token)
    return " ".join(out)


def normalize_text(text: str, locale: Optional[str] = None) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    # Drop punctuation that the constrained decoder never emits.
    cleaned = [ch for ch in text if ch.isalnum() or ch.isspace()]
    text = " ".join("".join(cleaned).split())
    if locale:
        text = spell_out_numbers(text, locale)
    return text


def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, ref_tok in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, hyp_tok in enumerate(hyp, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


@dataclass
class Result:
    sample: Sample
    expected: str
    predicted: str
    correct: bool
    word_errors: int
    word_count: int
    char_errors: int
    char_count: int
    latency: float
    audio_seconds: float
    penalty: float
    margin: float
    accepted: bool
    negative: bool = False


# ----------------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> int:
    templates = load_templates(Path(args.templates))
    samples = load_dataset(Path(args.dataset))
    print(
        f"Loaded {len(templates)} templates and {len(samples)} samples", file=sys.stderr
    )

    # Per-token penalty scale differs by backend (subword vs character), so the
    # gating threshold does too. Swept on tests/en; validate on more data.
    default_threshold = {"citrinet": 4.0, "coqui": 1.25}[args.backend]
    if args.close_score is None:
        args.close_score = default_threshold
    if args.max_score is None:
        args.max_score = default_threshold
    if args.min_margin is None:
        args.min_margin = 0.0

    backend_kwargs = {}
    if args.backend == "coqui" and args.stt_binary:
        backend_kwargs["stt_binary"] = args.stt_binary

    print(f"Loading {args.backend} model from {args.model_dir} ...", file=sys.stderr)
    load_start = time.perf_counter()
    recognizer = load_recognizer(
        args.backend,
        args.model_dir,
        language=args.language,
        beam=args.beam,
        **backend_kwargs,
    )
    load_seconds = time.perf_counter() - load_start

    print("Building grammar FST (training) ...", file=sys.stderr)
    train_start = time.perf_counter()
    recognizer.train(templates)
    train_seconds = time.perf_counter() - train_start

    score_locale = args.language if args.spell_numbers else None
    results: List[Result] = []
    for index, sample in enumerate(samples, start=1):
        try:
            seconds = audio_seconds(sample.audio_path)
            decode_start = time.perf_counter()
            out = recognizer.transcribe(sample.audio_path)
            latency = time.perf_counter() - decode_start
        except Exception as err:  # pylint: disable=broad-except
            print(
                f"[{index}/{len(samples)}] SKIP {sample.audio_path}: {err}",
                file=sys.stderr,
            )
            continue

        predicted = normalize_text(out.text, score_locale)
        expected = normalize_text(sample.expected, score_locale)

        ref_words = expected.split()
        hyp_words = predicted.split()
        word_errors = _edit_distance(ref_words, hyp_words)
        char_errors = _edit_distance(list(expected), list(predicted))

        accepted = _would_accept(out.score, out.margin, args)

        result = Result(
            sample=sample,
            expected=expected,
            predicted=predicted,
            correct=(predicted == expected),
            word_errors=word_errors,
            word_count=len(ref_words),
            char_errors=char_errors,
            char_count=len(expected),
            latency=latency,
            audio_seconds=seconds,
            penalty=out.score,
            margin=out.margin,
            accepted=accepted,
            negative=sample.negative,
        )
        results.append(result)

        if sample.negative:
            # Success for a negative is being rejected by the gating.
            mark = "OK " if not accepted else "ERR"
            exp_field = "[out-of-grammar]"
        else:
            mark = "OK " if result.correct else "ERR"
            exp_field = repr(expected)
        print(
            f"[{index}/{len(samples)}] {mark} "
            f"{latency * 1000:6.0f}ms "
            f"pen={out.score:5.2f} mar={out.margin:5.2f} "
            f"acc={'Y' if accepted else 'n'} | "
            f"exp={exp_field} got={predicted!r}",
            file=sys.stderr,
        )

    _report(results, load_seconds, train_seconds, args)
    return 0


def _would_accept(penalty: float, margin: float, args: argparse.Namespace) -> bool:
    """Replicate vad_stt.py's local-vs-remote gating, for diagnostics."""
    max_score = math.inf if args.max_score is None else args.max_score
    close_score = math.inf if args.close_score is None else args.close_score
    min_margin = 0.0 if args.min_margin is None else args.min_margin

    if penalty <= close_score:
        return True
    if (penalty < max_score) and (margin > min_margin):
        return True
    return False


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = pct / 100.0 * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _report(
    results: List[Result],
    load_seconds: float,
    train_seconds: float,
    args: argparse.Namespace,
) -> None:
    if not results:
        print("No results to report.")
        return

    positives = [r for r in results if not r.negative]
    negatives = [r for r in results if r.negative]

    latencies = [r.latency for r in results]
    audio_total = sum(r.audio_seconds for r in results)
    decode_total = sum(latencies)

    print()
    print("=" * 64)
    print(f"ACCURACY (in-grammar samples: {len(positives)})")
    print("=" * 64)
    if positives:
        correct = sum(1 for r in positives if r.correct)
        word_errors = sum(r.word_errors for r in positives)
        word_count = sum(r.word_count for r in positives)
        char_errors = sum(r.char_errors for r in positives)
        char_count = sum(r.char_count for r in positives)
        wer = word_errors / word_count if word_count else 0.0
        cer = char_errors / char_count if char_count else 0.0
        n_pos = len(positives)
        print(f"  Sentence accuracy  : {correct}/{n_pos} = {correct / n_pos:6.2%}")
        print(f"  Word error rate    : {wer:6.2%}  ({word_errors}/{word_count} words)")
        print(f"  Char error rate    : {cer:6.2%}  ({char_errors}/{char_count} chars)")

        # Mistakes that the grammar *could* produce but the model got wrong.
        misrecognized = [r for r in positives if not r.correct]
        if misrecognized:
            print("  Misrecognitions:")
            for r in misrecognized:
                print(
                    f"    - {r.sample.audio_path.name}: "
                    f"exp={r.expected!r} got={r.predicted!r}"
                )

    print()
    print("=" * 64)
    print("GATING (local-vs-remote decision)")
    print("=" * 64)
    print(
        f"  thresholds         : max_score={args.max_score} "
        f"close_score={args.close_score} min_margin={args.min_margin}"
    )

    # Confusion matrix over the gating decision (accept = keep local result).
    #   positives  -> want accepted AND correct
    #   negatives  -> want rejected (deferred to remote)
    pos_keep_good = sum(1 for r in positives if r.accepted and r.correct)
    pos_keep_bad = sum(1 for r in positives if r.accepted and not r.correct)
    pos_defer_good = sum(1 for r in positives if not r.accepted and not r.correct)
    pos_defer_bad = sum(1 for r in positives if not r.accepted and r.correct)
    neg_reject = sum(1 for r in negatives if not r.accepted)
    neg_accept = sum(1 for r in negatives if r.accepted)

    if positives:
        print(f"  In-grammar ({len(positives)}):")
        print(f"    kept & correct   : {pos_keep_good}   (correct local answer)")
        print(f"    kept & wrong     : {pos_keep_bad}   <- FALSE ACCEPT (bad)")
        print(f"    deferred & wrong : {pos_defer_good}   (correctly sent to remote)")
        print(f"    deferred & right : {pos_defer_bad}   (missed local opportunity)")
    if negatives:
        rej_rate = neg_reject / len(negatives)
        print(f"  Out-of-grammar ({len(negatives)}):")
        print(
            f"    rejected         : {neg_reject}/{len(negatives)} = {rej_rate:6.2%}  (good)"
        )
        print(f"    accepted         : {neg_accept}   <- FALSE ACCEPT (bad)")
        if neg_accept:
            for r in negatives:
                if r.accepted:
                    print(
                        f"      ! {r.sample.audio_path.name}: "
                        f"got={r.predicted!r} pen={r.penalty:.2f} mar={r.margin:.2f}"
                    )

    # Headline: a local answer is "trusted" only when kept; it's good iff correct.
    false_accepts = pos_keep_bad + neg_accept
    total_kept = pos_keep_good + pos_keep_bad + neg_accept
    precision = pos_keep_good / total_kept if total_kept else 0.0
    print(
        f"  Kept-answer precision: {pos_keep_good}/{total_kept} = {precision:6.2%}  "
        f"({false_accepts} false accepts overall)"
    )

    print()
    print("=" * 64)
    print("SPEED")
    print("=" * 64)
    print(f"  Model load         : {load_seconds:8.2f} s")
    print(f"  Grammar build      : {train_seconds:8.2f} s")
    print(f"  Audio total        : {audio_total:8.2f} s")
    print(f"  Decode total       : {decode_total:8.2f} s")
    print(f"  Real-time factor   : {decode_total / audio_total:8.3f}  (decode/audio)")
    print(f"  Latency mean       : {statistics.mean(latencies) * 1000:8.0f} ms")
    print(f"  Latency median     : {statistics.median(latencies) * 1000:8.0f} ms")
    print(f"  Latency p95        : {_percentile(latencies, 95) * 1000:8.0f} ms")
    print(f"  Latency max        : {max(latencies) * 1000:8.0f} ms")

    if args.output:
        _write_csv(Path(args.output), results)
        print(f"\nPer-sample results written to {args.output}")


def _write_csv(path: Path, results: List[Result]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(
            [
                "audio",
                "negative",
                "expected",
                "predicted",
                "correct",
                "word_errors",
                "word_count",
                "char_errors",
                "char_count",
                "latency_ms",
                "audio_seconds",
                "penalty",
                "margin",
                "accepted",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    str(r.sample.audio_path),
                    int(r.negative),
                    r.expected,
                    r.predicted,
                    int(r.correct),
                    r.word_errors,
                    r.word_count,
                    r.char_errors,
                    r.char_count,
                    f"{r.latency * 1000:.1f}",
                    f"{r.audio_seconds:.3f}",
                    f"{r.penalty:.4f}",
                    f"{r.margin:.4f}",
                    int(r.accepted),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--templates", required=True, help="File with one sentence template per line"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Audio + transcript manifest (.jsonl/.tsv/.csv)",
    )
    parser.add_argument(
        "--backend",
        default="citrinet",
        choices=["citrinet", "coqui"],
        help="Acoustic backend",
    )
    parser.add_argument(
        "--model-dir", required=True, help="Path to the model directory"
    )
    parser.add_argument("--language", default="en", help="Locale for number spellout")
    parser.add_argument(
        "--beam",
        type=float,
        default=None,
        help="Decode pruning beam (log-prob units); default per backend",
    )
    parser.add_argument(
        "--stt-binary", help="Coqui: path to the stt_onlyprobs helper binary"
    )
    parser.add_argument("--output", help="Optional CSV path for per-sample results")
    parser.add_argument(
        "--no-spell-numbers",
        dest="spell_numbers",
        action="store_false",
        help="Do not spell out digits in expected transcripts before scoring",
    )
    # Gating thresholds (per-token penalty). Default per backend when unset.
    parser.add_argument("--max-score", type=float, default=None)
    parser.add_argument("--close-score", type=float, default=None)
    parser.add_argument("--min-margin", type=float, default=None)
    args = parser.parse_args()

    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
