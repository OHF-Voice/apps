#!/usr/bin/env python3
"""Audio pipeline sanity check for Speech-to-Phrase.

NOT a coverage/accuracy oracle (TTS != human speech). It exercises the full
pipeline end to end and catches regressions:

  template --(sample K realizations)--> example sentence
          --> HA TTS (/api/tts_get_url, cached to disk)
          --> [clean] and [reverb with device RIR (+ optional noise) @ SNR sweep]
          --> Recognizer (trained on ALL templates)
          --> score-aware, same-template classification.

Outcome per (sample, condition) is classified, not pass/fail:
  EXACT       heard == spoke
  SAME_TMPL   heard is a different valid realization of the SAME template
  CONFUSION   heard is in-grammar but from another template (real error / ambiguity)
  NO_PARSE    no decode
and each is annotated GATED when score exceeds the backend's gate threshold
(the recognizer would defer to cloud, so a wrong+gated result is acceptable).

Needs a Home Assistant instance for TTS. Point it at one with:

    export HA_URL=http://homeassistant.local:8123      # optional, this is the default
    export HA_TOKEN=<long-lived access token>

TTS language is en-US.
"""
import argparse
import hashlib
import io
import json
import os
import random
import sys
import unicodedata
import urllib.request
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vad import normalize_level, trim_silence  # noqa: E402  (src on path above)

from speech_to_phrase import load_recognizer
from speech_to_phrase.templates import (
    AlternativesNode,
    ListRefNode,
    LiteralNode,
    NumberRangeNode,
    Node,
    OptionalNode,
    SequenceNode,
    TemplateParser,
    _spellout_words,
)

HA_URL = os.environ.get("HA_URL", "http://homeassistant.local:8123")
TOKEN = os.environ.get("HA_TOKEN", "")
TTS_LANGUAGE = "en-US"
SAMPLE_RATE = 16000
# citrinet gate re-fit 8.0 -> 5.0 (2026-07-01): the expanded ~26-intent grammar
# has short commands ("next"/"stop"/"go back") that noise/OOV audio false-matches
# at 6-8, so 8.0 gave ~18/112 OOV false-accepts. Youden-optimal for the VAD-on
# pipeline is 5.5, but we ship a more conservative 5.0 default (OOV FA -> 2/112,
# legit acceptance 93% clean/97% all). Users can override per-language in the web
# UI. Re-fit with this tool (see the "gate fitting" sweep it prints).
# coqui gate re-fit 1.25 -> 2.0 (2026-07-01): evaluated sl_SL-coqui on Common
# Voice with a grammar-size sweep (fixed in-grammar probe set + disjoint OOV
# set, real human speech). The 1.25 default rejected ~24% of correctly
# recognized commands (needless cloud fallback); usable accuracy only reaches
# the decode ceiling (94% short / 97% all) around gate ~2.5. 2.0 keeps ~89%
# usable on short/command-like utterances at ~4% OOV false-accept.
GATE_THRESHOLD = {"citrinet": 5.0, "coqui": 2.0}

# Whether to VAD-trim each clip before decoding, mirroring the production STT
# path (wyoming_server trims via vad.trim_silence before transcribe). Toggled by
# --no-vad so the gate can be re-fit against the same pipeline HA actually runs.
USE_VAD = True


def transcribe(rec, audio: np.ndarray):
    """Recognize a clip through the same front-end as production (level
    normalization, optional VAD trim, then decode)."""
    audio = normalize_level(audio)
    if USE_VAD:
        audio = trim_silence(audio)
    return rec.transcribe(audio)

# Test entity registry (name -> domain). Mirrors what the add-on's trainer reads
# from the live HA registry. {name} is bound PER sentence-set to only the
# entities whose domain is in that set's name_domains -- otherwise a lock/cover
# name leaks into an on-able template and "front door on" becomes recognizable.
ENTITIES: Dict[str, str] = {
    "overhead light": "light",
    "kitchen lamp": "light",
    "kitchen fan": "fan",
    "garage door": "cover",
    "front door": "lock",
}
STATIC_LISTS: Dict[str, List[str]] = {
    "area": ["kitchen", "office"],
    "floor": ["first floor"],
    "color": ["red", "blue"],
    "brightness_level": ["maximum"],
    # Per-domain state lists + volume step (mirrors training.DEV_SLOT_LISTS).
    # Needed so the full curated grammar (HassGetState/cover/lock, volume) compiles.
    "on_off_state": ["on", "off"],
    "cover_state": ["open", "closed"],
    "lock_state": ["locked", "unlocked"],
    "volume_step": ["up", "down"],
}
# Populated with domain-scoped name__<domains> lists as templates are loaded.
LIST_VALUES: Dict[str, List[str]] = dict(STATIC_LISTS)


def _name_list(domains: Sequence[str]) -> str:
    """Register (if new) a name list scoped to `domains` and return its key."""
    key = "name__" + "_".join(sorted(domains))
    if key not in LIST_VALUES:
        LIST_VALUES[key] = [n for n, d in ENTITIES.items() if d in domains]
    return key


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


# --------------------------------------------------------------------------
# template -> realizations (one random sample, or full enumeration for oracle)
# --------------------------------------------------------------------------
def _range_values(item: Tuple[int, ...]) -> List[int]:
    if len(item) == 1:
        return [item[0]]
    start, end, step = item
    return list(range(start, end + 1, step)) if step > 0 else [start]


def sample_words(node: Node, rng: random.Random, locale: str) -> List[str]:
    if isinstance(node, LiteralNode):
        return [node.text]
    if isinstance(node, OptionalNode):
        return sample_words(node.child, rng, locale) if rng.random() < 0.5 else []
    if isinstance(node, AlternativesNode):
        return sample_words(rng.choice(node.options), rng, locale)
    if isinstance(node, NumberRangeNode):
        item = rng.choice(node.items)
        # _spellout_words is memoized and returns an immutable tuple; copy it.
        return list(_spellout_words(rng.choice(_range_values(item)), locale))
    if isinstance(node, ListRefNode):
        return rng.choice(LIST_VALUES[node.name]).split()
    if isinstance(node, SequenceNode):
        out: List[str] = []
        for part in node.parts:
            out.extend(sample_words(part, rng, locale))
        return out
    raise TypeError(type(node).__name__)


def enumerate_realizations(
    node: Node, locale: str, cap: int = 4000
) -> Optional[Set[str]]:
    """All normalized realizations of a template, or None if it exceeds `cap`."""
    def expand(n: Node) -> Optional[List[List[str]]]:
        if isinstance(n, LiteralNode):
            return [[n.text]]
        if isinstance(n, OptionalNode):
            inner = expand(n.child)
            return None if inner is None else [[]] + inner
        if isinstance(n, AlternativesNode):
            out: List[List[str]] = []
            for opt in n.options:
                e = expand(opt)
                if e is None:
                    return None
                out.extend(e)
            return out
        if isinstance(n, NumberRangeNode):
            out = []
            for item in n.items:
                for v in _range_values(item):
                    # Memoized -> tuple; the sequence builder below concatenates
                    # onto lists, so copy rather than aliasing the cache entry.
                    out.append(list(_spellout_words(v, locale)))
            return out
        if isinstance(n, ListRefNode):
            return [v.split() for v in LIST_VALUES[n.name]]
        if isinstance(n, SequenceNode):
            acc: List[List[str]] = [[]]
            for part in n.parts:
                e = expand(part)
                if e is None:
                    return None
                acc = [a + b for a in acc for b in e]
                if len(acc) > cap:
                    return None
            return acc
        raise TypeError(type(n).__name__)

    seqs = expand(node)
    if seqs is None:
        return None
    return {norm(" ".join(s)) for s in seqs}


def load_templates(s2p_repo: Path, language: str) -> List[str]:
    """Templates for the whole enabled grammar, built by the add-on's own
    trainer.

    This used to walk a ``sentences/<lang>/*.yaml`` tree and do its own {name}
    scoping. That tree is gone -- templates come from the home-assistant-intents
    package now -- and a second copy of the scoping rules would drift from the
    real one anyway. Going through training.assemble means the sweep measures
    the grammar production actually compiles, and picks up the domain/capability
    gating for free. Everything is enabled ("optional" tier) so the test hits the
    widest grammar, which is the hardest case for the decoder."""
    import presets as bi
    import training

    meta = bi.load_intents_meta()
    combos = bi.available_combos(s2p_repo, language, meta)
    enabled = bi.default_enabled(combos, "optional")
    templates, list_values = training.assemble(
        s2p_repo, language, enabled, [], ENTITIES, STATIC_LISTS
    )
    # sample_words/enumerate_realizations resolve {list} refs through this.
    LIST_VALUES.clear()
    LIST_VALUES.update({k: list(v) for k, v in list_values.items()})
    return templates


# --------------------------------------------------------------------------
# Home Assistant TTS, cached to disk by (engine, language, text)
# --------------------------------------------------------------------------
def tts_wav(message: str, engine_id: str, cache_dir: Path) -> np.ndarray:
    key = hashlib.sha1(f"{engine_id}|{TTS_LANGUAGE}|{message}".encode()).hexdigest()
    cached = cache_dir / f"{key}.wav"
    if cached.exists():
        return sf.read(cached, dtype="float32")[0]

    if not TOKEN:
        raise SystemExit(
            "HA_TOKEN is not set: this clip is not cached and synthesizing it "
            "needs a Home Assistant long-lived access token.\n"
            "  export HA_TOKEN=<token>   (and HA_URL if not homeassistant.local:8123)"
        )
    req = urllib.request.Request(
        f"{HA_URL}/api/tts_get_url",
        data=json.dumps(
            {"engine_id": engine_id, "message": message, "language": TTS_LANGUAGE}
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
        from librosa import resample

        data = resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sf.write(cached, data, SAMPLE_RATE)
    return data


# --------------------------------------------------------------------------
# Degradation: reverb (device RIR) first, then additive noise at target SNR.
# --------------------------------------------------------------------------
def _mono16k(path: Path) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        from librosa import resample

        data = resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return data


def _active_power(x: np.ndarray) -> float:
    # SNR over the voiced region only (ignore TTS leading/trailing silence)
    frame = 400
    e = np.array([np.mean(x[i:i + frame] ** 2) for i in range(0, len(x), frame)])
    if not len(e):
        return float(np.mean(x**2) + 1e-12)
    active = e[e > 0.1 * e.max()]
    return float(active.mean() + 1e-12)


def degrade(speech: np.ndarray, rir: Optional[np.ndarray],
            noise: Optional[np.ndarray], snr_db: Optional[float]) -> np.ndarray:
    sig = speech
    if rir is not None:
        sig = fftconvolve(sig, rir)[: len(speech) + len(rir)]
    if noise is not None and snr_db is not None:
        if len(noise) < len(sig):
            noise = np.tile(noise, int(np.ceil(len(sig) / len(noise))))
        off = 0 if len(noise) == len(sig) else np.random.randint(0, len(noise) - len(sig))
        noise = noise[off: off + len(sig)]
        gain = np.sqrt(_active_power(speech) / (_active_power(noise) * 10 ** (snr_db / 10)))
        sig = sig + gain * noise
    # attenuate only (never boost) so silent/quiet OOV clips aren't amplified
    peak = np.max(np.abs(sig))
    if peak > 0.95:
        sig = sig / peak * 0.95
    return sig.astype(np.float32)


Condition = Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[float]]


def build_conditions(rirs: Dict[str, np.ndarray], noises: Dict[str, np.ndarray],
                     snrs: Sequence[float]) -> List[Condition]:
    """clean baseline + per-RIR reverb-only + per-(RIR x noise x SNR) stress."""
    conds: List[Condition] = [("clean", None, None, None)]
    for rname, rir in rirs.items():
        conds.append((f"rir:{rname}", rir, None, None))  # reverb only
        for nname, noise in noises.items():
            for snr in snrs:
                conds.append((f"rir:{rname}+{nname}@{snr:g}dB", rir, noise, snr))
    return conds


# --------------------------------------------------------------------------
def run_oov(rec, oov_dir: Path, conditions: List["Condition"], gate: float):
    """Out-of-vocabulary rejection: non-command audio (other speech, cough,
    silence) must NOT be confidently matched to a command. A confident in-grammar
    decode of OOV audio is a FALSE_ACCEPT — the worst failure for a voice agent.
    A high score / empty decode is correct (the recognizer defers to cloud)."""
    files = sorted(oov_dir.glob("*.wav"))
    tally: Dict[str, Dict[str, int]] = {c[0]: {"REJECTED": 0, "FALSE_ACCEPT": 0}
                                        for c in conditions}
    false_accepts: List[Tuple[str, str, str, float]] = []
    oov_scores: List[float] = []  # in-grammar decodes of OOV audio (gate-independent)
    for f in files:
        clip = _mono16k(f)
        for cname, rir, noise, snr in conditions:
            r = transcribe(rec, degrade(clip, rir, noise, snr))
            in_grammar = bool(r.text) and r.score != float("inf")
            if in_grammar:
                oov_scores.append(r.score)
            confident = in_grammar and r.score <= gate
            if confident:
                tally[cname]["FALSE_ACCEPT"] += 1
                false_accepts.append((cname, f.name, norm(r.text), r.score))
            else:
                tally[cname]["REJECTED"] += 1
    return false_accepts, tally, len(files), oov_scores


def classify(result, spoke: str, same_tmpl: Optional[Set[str]], gate: float) -> str:
    if not result.text or result.score == float("inf"):
        tag = "NO_PARSE"
    elif norm(result.text) == spoke:
        tag = "EXACT"
    elif same_tmpl is not None and norm(result.text) in same_tmpl:
        tag = "SAME_TMPL"
    else:
        tag = "CONFUSION"
    return tag + ("/GATED" if result.score > gate else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2p-repo", required=True, type=Path)
    ap.add_argument("--language", default="en")
    ap.add_argument("--backend", default="citrinet")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--engine-id", default="tts.home_assistant_cloud")
    ap.add_argument("--samples", type=int, default=3, help="realizations per template")
    ap.add_argument("--snr-db", type=float, nargs="+", default=[15.0, 5.0])
    ap.add_argument("--rir-dir", type=Path, help="dir of RIR wavs (default: <repo>/tests/wav/rir)")
    ap.add_argument("--noise-dir", type=Path, help="dir of noise wavs (default: <repo>/tests/wav/noise; empty = no stress)")
    ap.add_argument("--cache-dir", type=Path, default=Path("tests/wav/.tts_cache"))
    ap.add_argument("--oov-dir", type=Path, help="dir of OOV wavs (default: <repo>/tests/wav/oov)")
    ap.add_argument("--token-bonus", type=float, default=0.0,
                    help="word-insertion reward per emitted token (0 = off); "
                         "sweep this when long commands decode as short ones")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-vad", action="store_true",
                    help="disable VAD trimming (default: trim, matching production STT)")
    ap.add_argument("--dump-scores", type=Path, default=None,
                    help="write raw legit/confusion/OOV scores to this JSON path")
    args = ap.parse_args()

    global USE_VAD
    USE_VAD = not args.no_vad

    wav = args.s2p_repo / "tests" / "wav"
    rir_dir = args.rir_dir or (wav / "rir")
    noise_dir = args.noise_dir if args.noise_dir is not None else (wav / "noise")
    rirs = {p.stem: _mono16k(p) for p in sorted(rir_dir.glob("*.wav"))}
    if not rirs:
        raise SystemExit(f"No RIR wavs found in {rir_dir}")
    noises = {p.stem: _mono16k(p) for p in sorted(noise_dir.glob("*.wav"))} if noise_dir.is_dir() else {}
    conditions = build_conditions(rirs, noises, args.snr_db)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    templates = load_templates(args.s2p_repo, args.language)
    if args.limit:
        templates = templates[: args.limit]

    rec = load_recognizer(args.backend, args.model, language=args.language,
                          token_bonus=args.token_bonus)
    rec.train(templates, list_values=LIST_VALUES)
    gate = GATE_THRESHOLD.get(args.backend, 4.0)

    # per-condition tally of category -> count
    tally: Dict[str, Dict[str, int]] = {c[0]: {} for c in conditions}
    # gate-fitting: raw scores of correct in-grammar decodes vs wrong (confusion)
    # ones, tagged by condition so we can separate clean from degraded.
    legit_scores: List[Tuple[str, float]] = []
    confusion_scores: List[Tuple[str, float]] = []
    for tmpl in templates:
        ast = TemplateParser(tmpl).parse()
        same = enumerate_realizations(ast, args.language)
        seen: Set[str] = set()
        for _ in range(args.samples):
            spoke = norm(" ".join(sample_words(ast, rng, args.language)))
            if spoke in seen:
                continue
            seen.add(spoke)
            try:
                speech = tts_wav(spoke, args.engine_id, args.cache_dir)
            except Exception as e:  # noqa: BLE001
                print(f"TTS-ERROR {spoke!r}: {e}")
                continue
            for cname, rir, noise, snr in conditions:
                audio = degrade(speech, rir, noise, snr)
                r = transcribe(rec, audio)
                cat = classify(r, spoke, same, gate)
                tally[cname][cat] = tally[cname].get(cat, 0) + 1
                base = cat.split("/")[0]
                if r.score != float("inf"):
                    if base in ("EXACT", "SAME_TMPL"):
                        legit_scores.append((cname, r.score))
                    elif base == "CONFUSION":
                        confusion_scores.append((cname, r.score))

    # report
    cats = ["EXACT", "SAME_TMPL", "CONFUSION", "NO_PARSE"]
    print(f"\n{'condition':<28}  " + "  ".join(f"{c:>9}" for c in cats) + "   ok%")
    worst_ok = 1.0
    for cname, counts in tally.items():
        agg = {c: 0 for c in cats}
        for k, v in counts.items():
            agg[k.split("/")[0]] += v
        n = sum(agg.values()) or 1
        ok = (agg["EXACT"] + agg["SAME_TMPL"]) / n
        worst_ok = min(worst_ok, ok)
        gated = sum(v for k, v in counts.items() if k.endswith("/GATED"))
        print(f"{cname:<28}  " + "  ".join(f"{agg[c]:>9}" for c in cats)
              + f"   {ok:5.0%}  ({gated} gated)")
    # Fail only if a CONFIDENT (non-gated) confusion/no-parse appears in clean.
    clean_bad = sum(v for k, v in tally['clean'].items()
                    if k.split('/')[0] in ('CONFUSION', 'NO_PARSE') and '/GATED' not in k)
    print(f"\nclean confident errors: {clean_bad}")

    # OOV rejection pass
    oov_dir = args.oov_dir or (wav / "oov")
    clean_fa = 0
    oov_scores: List[float] = []
    if oov_dir.is_dir():
        fa, oov_tally, n_oov, oov_scores = run_oov(rec, oov_dir, conditions, gate)
        print(f"\n=== OOV rejection ({n_oov} clips; false-accept = confident "
              f"command match of non-command audio) ===")
        print(f"{'condition':<28}  {'REJECTED':>9}  {'FALSE_ACCEPT':>12}")
        for cname, c in oov_tally.items():
            print(f"{cname:<28}  {c['REJECTED']:>9}  {c['FALSE_ACCEPT']:>12}")
        if fa:
            print("\nfalse-accepts (DANGEROUS — would act on a command never spoken):")
            for cname, fname, text, score in fa[:20]:
                print(f"  [{cname}] {fname} -> {text!r} (score={score:.3f})")
        clean_fa = oov_tally["clean"]["FALSE_ACCEPT"]
        print(f"\nclean OOV false-accepts: {clean_fa}")

    # ---- Gate fitting: where should --max-score sit? -----------------------
    # A good gate accepts legitimate in-grammar decodes (esp. clean) and rejects
    # in-grammar decodes of OOV audio. Report the score distributions and the
    # gap between them so the threshold can be re-fit (e.g. after enabling VAD).
    def _summ(label: str, scores: Sequence[float]) -> None:
        if not scores:
            print(f"  {label:<26} (none)")
            return
        a = np.sort(np.asarray(scores, dtype=float))
        pct = lambda p: float(np.percentile(a, p))  # noqa: E731
        print(f"  {label:<26} n={len(a):<4} min={a[0]:5.2f}  p50={pct(50):5.2f}  "
              f"p90={pct(90):5.2f}  p99={pct(99):5.2f}  max={a[-1]:5.2f}")

    clean_legit = [s for c, s in legit_scores if c == "clean"]
    all_legit = [s for _, s in legit_scores]
    all_conf = [s for _, s in confusion_scores]
    print(f"\n=== gate fitting (VAD={'on' if USE_VAD else 'off'}) — score "
          f"distributions (lower = more confident) ===")
    _summ("legit correct (clean)", clean_legit)
    _summ("legit correct (all cond)", all_legit)
    _summ("confusion (wrong tmpl)", all_conf)
    _summ("OOV in-grammar", oov_scores)

    if clean_legit and oov_scores:
        # The distributions overlap in the tails, so a hard threshold trades
        # legit acceptance against OOV false-accepts. Sweep candidate gates and
        # report both, plus a score that maximizes (legit accepted - OOV
        # accepted): recall of real commands minus the dangerous false-accepts.
        cl = np.asarray(clean_legit); al = np.asarray(all_legit)
        oo = np.asarray(oov_scores); cf = np.asarray(all_conf) if all_conf else np.array([])
        print("\n  gate   legit_clean%  legit_all%   OOV_FA   confusion_acc   "
              "youden(clean-OOV)")
        best = (None, -2.0)
        for g in [round(x, 1) for x in np.arange(3.0, 9.01, 0.5)]:
            lc = float((cl <= g).mean())
            la = float((al <= g).mean())
            oof = int((oo <= g).sum())
            oofr = float((oo <= g).mean())
            ca = int((cf <= g).sum()) if cf.size else 0
            j = lc - oofr  # true-accept rate (clean legit) - false-accept rate (OOV)
            if j > best[1]:
                best = (g, j)
            print(f"  {g:4.1f}   {lc:10.0%}   {la:9.0%}   {oof:4d}/{len(oo)}   "
                  f"{ca:5d}/{len(cf) if cf.size else 0}        {j:+.3f}")
        print(f"\n  --> recommended --max-score = {best[0]:.1f} "
              f"(maximizes clean-legit acceptance minus OOV false-accept rate)")

    # Optionally dump raw scores so the gate can be re-fit offline (no re-run).
    if args.dump_scores:
        args.dump_scores.write_text(json.dumps({
            "vad": USE_VAD,
            "legit_clean": clean_legit,
            "legit_all": [s for _, s in legit_scores],
            "confusion": [s for _, s in confusion_scores],
            "oov": oov_scores,
        }))
        print(f"\n  raw scores dumped to {args.dump_scores}")

    return 1 if (clean_bad or clean_fa) else 0


if __name__ == "__main__":
    raise SystemExit(main())
