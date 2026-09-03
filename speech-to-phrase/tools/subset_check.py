#!/usr/bin/env python3
"""Subset gate: verify Speech-to-Phrase curated sentences are a subset of the
home-assistant-intents templates, per (language, intent, slot_combination).

For each combo file present on BOTH sides we build two word-level finite-state
acceptors with the SAME hassil AST->FST compiler (hassil/fst.py), then use
OpenFST to test language containment exactly:

    L(S2P)  subset of  L(HA)   <=>   L(S2P) intersect complement(L(HA)) = empty

OpenFST's `fstdifference A B` computes A minus B (requires B deterministic,
epsilon-free, arc-sorted). If the difference is non-empty, S2P accepts a
phrasing HA does not -> the surviving paths ARE the counterexamples we print.

Open slots ({name}/{area}/{floor}/{domain}/{device_class}) and list/range slots
are rendered as single placeholder tokens by hassil/fst.py, so this validates
the *phrasing structure* around slots without enumerating real entities.
"""
import argparse
import copy
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml
from hassil import Intents, merge_dict
from hassil.fst import EPS, SPACE, Fst, intents_to_fst

# A {ref} inside a template. We rewrite RANGE refs (only) to a bounds-encoding
# sentinel word so the FST check validates numeric bounds, not just the slot
# name. Open slots / text lists keep flowing through hassil as placeholders.
_REF = re.compile(r"\{([^}]+)\}")
# hassil inline range syntax: {from..to[,step][:slot_name]}  (step is comma)
_INLINE_RANGE = re.compile(r"^(\d+)\.\.(\d+)(?:,(\d+))?(?::(\w+))?$")


def range_bounds(globals_dict: dict) -> Dict[str, Tuple[int, int, int]]:
    """slot/list name -> (from, to, step) for every RangeSlotList in scope."""
    out: Dict[str, Tuple[int, int, int]] = {}
    for name, spec in (globals_dict.get("lists") or {}).items():
        rng = spec.get("range") if isinstance(spec, dict) else None
        if rng:
            out[name] = (rng["from"], rng["to"], int(rng.get("step", 1)))
    return out


def _range_sentinel(slot: str, lo: int, hi: int, step: int) -> str:
    return f"RANGE·{slot}·{lo}..{hi}/{step}"  # bareword, no spaces


def normalize_ranges(text: str, ranges: Dict[str, Tuple[int, int, int]]) -> str:
    """Rewrite range refs to a bounds sentinel, on BOTH sides identically.

    - named range ref      {brightness}            (HA)  -> bounds from lists
    - named range ref      {volume:brightness}     (HA)  -> bounds of `volume`
    - inline range ref     {0..100:brightness}     (S2P) -> bounds from the ref
    Anything else (open slots, text lists) is left for hassil to placeholder.
    """
    def repl(m: re.Match) -> str:
        body = m.group(1)
        im = _INLINE_RANGE.match(body)
        if im:
            lo, hi, step, slot = im.groups()
            slot = slot or "value"
            return _range_sentinel(slot, int(lo), int(hi), int(step or 1))
        list_name, _, slot = body.partition(":")
        slot = slot or list_name
        if list_name in ranges:
            lo, hi, step = ranges[list_name]
            return _range_sentinel(slot, lo, hi, step)
        return m.group(0)

    return _REF.sub(repl, text)


def _apply_ranges(data_sets: list, ranges: Dict[str, Tuple[int, int, int]]) -> list:
    data_sets = copy.deepcopy(data_sets)
    for ss in data_sets:
        ss["sentences"] = [normalize_ranges(s, ranges) for s in ss.get("sentences", [])]
    return data_sets


def load_lang_globals(intents_repo: Path, lang: str) -> dict:
    """Merge expansion_rules + lists (per-language + shared) for a language."""
    d: dict = {}
    for f in sorted((intents_repo / "rules" / lang).glob("*.yaml")):
        merge_dict(d, yaml.safe_load(f.read_text()) or {})
    for f in sorted((intents_repo / "lists" / lang).glob("*.yaml")):
        merge_dict(d, yaml.safe_load(f.read_text()) or {})
    for f in sorted((intents_repo / "lists").glob("*.yaml")):  # shared ranges
        merge_dict(d, yaml.safe_load(f.read_text()) or {})
    return d


def combo_fst(
    globals_dict: dict,
    lang: str,
    intent: str,
    data_sets: list,
    ranges: Dict[str, Tuple[int, int, int]],
) -> Fst:
    """Build a pruned acceptor for one intent/combo's sentence sets, resolving
    expansion rules and lists from the language globals."""
    data_sets = _apply_ranges(data_sets, ranges)
    spec = {
        "language": lang,
        "intents": {intent: {"data": data_sets}},
        "expansion_rules": dict(globals_dict.get("expansion_rules", {})),
        "lists": dict(globals_dict.get("lists", {})),
    }
    fst = intents_to_fst(Intents.from_dict(spec), intent_names={intent})
    fst.prune()
    return fst


def _label(in_label: str) -> str:
    # Compare WORD sequences: <space> is just a separator -> epsilon (collapsed
    # by fstrmepsilon). Word tokens stay distinct, so no word-merging occurs.
    return EPS if in_label == SPACE else in_label


def collect_symbols(*fsts: Fst) -> Dict[str, int]:
    syms: Dict[str, int] = {EPS: 0, "<unk>": 1}
    for fst in fsts:
        for arcs in fst.arcs.values():
            for arc in arcs:
                lbl = _label(arc.in_label)
                if lbl not in syms:
                    syms[lbl] = len(syms)
    return syms


def write_syms(syms: Dict[str, int], path: Path) -> None:
    path.write_text("".join(f"{s} {i}\n" for s, i in syms.items()))


def write_acceptor_txt(fst: Fst, path: Path) -> None:
    """AT&T text acceptor, input-projected (drops intent:/output overrides).
    State 0 (start) must be emitted first."""
    lines: List[str] = []
    states = [0] + [s for s in sorted(fst.arcs) if s != 0]
    for state in states:
        for arc in fst.arcs.get(state, []):
            lines.append(f"{state} {arc.to_state} {_label(arc.in_label)}")
    for s in sorted(fst.final_states):
        lines.append(f"{s}")
    path.write_text("\n".join(lines) + "\n")


def _run(cmd: str, stdin: bytes = b"") -> bytes:
    p = subprocess.run(cmd, shell=True, input=stdin, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{p.stderr.decode()}")
    return p.stdout


def check_combo(s2p: Fst, ha: Fst, n_examples: int = 10) -> List[str]:
    """Return [] if L(s2p) subset L(ha), else up to n counterexample sentences."""
    syms = collect_symbols(s2p, ha)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        write_syms(syms, d / "syms")
        write_acceptor_txt(s2p, d / "a.txt")
        write_acceptor_txt(ha, d / "b.txt")
        comp = (
            f"fstcompile --acceptor --isymbols={d}/syms "
            f"--keep_isymbols=true {{src}}"
        )
        _run(f"{comp.format(src=f'{d}/a.txt')} | fstrmepsilon > {d}/a.fst")
        _run(
            f"{comp.format(src=f'{d}/b.txt')} | fstrmepsilon | fstdeterminize "
            f"| fstminimize | fstarcsort --sort_type=ilabel > {d}/b.fst"
        )
        _run(f"fstdifference {d}/a.fst {d}/b.fst | fstconnect > {d}/diff.fst")
        info = _run(f"fstinfo {d}/diff.fst").decode()
        n_states = next(
            int(l.split()[-1]) for l in info.splitlines() if "# of states" in l
        )
        if n_states == 0:
            return []
        printed = _run(
            f"fstshortestpath --nshortest={n_examples} {d}/diff.fst "
            f"| fstprint --acceptor"
        ).decode()
        return _paths_to_sentences(printed)


def _paths_to_sentences(fstprint: str) -> List[str]:
    """Reconstruct sentences from fstprint of a (forest of) linear path(s)."""
    nxt: Dict[int, tuple] = {}
    finals: Set[int] = set()
    starts: Set[int] = set()
    dsts: Set[int] = set()
    for line in fstprint.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            src, dst, lbl = int(parts[0]), int(parts[1]), parts[2]
            nxt[src] = nxt.get(src, []) + [(dst, lbl)]
            starts.add(src)
            dsts.add(dst)
        elif parts and parts[0]:
            finals.add(int(parts[0]))
    roots = sorted(starts - dsts) or ([0] if nxt or finals else [])
    out: List[str] = []

    def pretty(w: str) -> str:
        if w.startswith("RANGE·"):
            _, slot, bounds = w.split("·")
            lo_hi, step = bounds.split("/")
            return f"{{{lo_hi}{'' if step == '1' else ',' + step}:{slot}}}"
        return w

    def walk(state: int, acc: List[str]) -> None:
        if state in finals and acc:
            out.append(" ".join(pretty(w) for w in acc if w not in (EPS, "<eps>")))
        for dst, lbl in nxt.get(state, []):
            walk(dst, acc + [lbl])

    for r in roots:
        walk(r, [])
    return sorted(set(out))


def load_combo_data(yaml_path: Path) -> list:
    return (yaml.safe_load(yaml_path.read_text()) or {}).get("data", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intents-repo", required=True, type=Path)
    ap.add_argument("--s2p-repo", required=True, type=Path)
    ap.add_argument("--language", default="en")
    ap.add_argument("--intent", help="limit to one intent")
    args = ap.parse_args()

    g = load_lang_globals(args.intents_repo, args.language)
    ranges = range_bounds(g)
    s2p_lang = args.s2p_repo / "sentences" / args.language
    ha_lang = args.intents_repo / "sentences" / args.language

    failures = 0
    intent_dirs = sorted(p for p in s2p_lang.iterdir() if p.is_dir())
    for idir in intent_dirs:
        intent = idir.name
        if args.intent and intent != args.intent:
            continue
        for combo_file in sorted(idir.glob("*.yaml")):
            combo = combo_file.stem
            ha_file = ha_lang / intent / f"{combo}.yaml"
            if not ha_file.exists():
                print(f"FAIL {intent}/{combo}: no matching HA combo file")
                failures += 1
                continue
            s2p = combo_fst(g, args.language, intent, load_combo_data(combo_file), ranges)
            ha = combo_fst(g, args.language, intent, load_combo_data(ha_file), ranges)
            cx = check_combo(s2p, ha)
            if cx:
                failures += 1
                print(f"FAIL {intent}/{combo}: S2P accepts {len(cx)}+ phrasing(s) "
                      f"not in HA (counterexamples):")
                for s in cx:
                    print(f"      - {s}")
            else:
                print(f"OK   {intent}/{combo}: subset verified")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
