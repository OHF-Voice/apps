"""Graph building for one configured wake word.

Four grammar-constrained HCLG graphs are built per wake word, mirroring the
verifier they are derived from:

  kw     : the accepted wake phrases + a costed filler loop (which absorbs
           the command); its 1-best alignment yields the split point
  bg     : a free per-phone filler loop (the background hypothesis)
  prefix : the accepted phrases MINUS their last word (isolates the last
           word's evidence; a one-word confusable dilutes to ~nothing in the
           whole-phrase likelihood ratio but shows up undiluted per word)
  kwc    : kw plus the configured confusable phrases; how much the best
           confusable beats the wake phrase by catches near-homophones that
           per-word thresholds cannot

Filler arcs carry the SAME insertion cost in every graph, so the cost of the
command tail cancels between graphs. Graphs are cached on disk and rebuilt
when the config (or BUILD_VERSION) changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import sqlite3
from pathlib import Path

from .config import WakeWordConfig
from .kaldi import SpeechTools
from .lexicon import ModelLexicon
from .model import PreparedModel

_LOGGER = logging.getLogger(__name__)

# Uniform per-filler-word insertion cost in EVERY graph (cancellation), and
# a small confusable-arc cost so acoustic ties break toward the wake phrase.
FILLER_COST = 2.0
COMPETITOR_COST = 1.0

# Bump when grammar/scoring semantics change in a way that requires
# rebuilding cached graphs even for identical config.
BUILD_VERSION = 1

UNK = "<unk>"
SIL = "<sil>"

GRAPH_TAGS = ("kw", "bg", "prefix", "kwc")
CTM_TAGS = ("kw", "kwc")

# Grammar states: 0 is the start, 1 is the shared accepting state where the
# filler loop lives, and 2.. are chain internals.
_START = 0
_END = 1


def graph_dir_for(train_dir: Path, model_id: str, wake_id: str) -> Path:
    return (Path(train_dir) / model_id / wake_id).absolute()


def build_signature(wake_word: WakeWordConfig) -> str:
    """Everything the built graphs depend on; a mismatch forces a rebuild."""
    payload = json.dumps(
        {
            "version": BUILD_VERSION,
            "model": wake_word.model,
            "phrases": [list(p.words) for p in wake_word.phrases],
            "confusables": [list(p.words) for p in wake_word.confusables],
            "lexicon": {w: list(p) for w, p in sorted(wake_word.lexicon.items())},
            "filler_cost": FILLER_COST,
            "competitor_cost": COMPETITOR_COST,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def train_wake_word(
    wake_word: WakeWordConfig,
    model: PreparedModel,
    train_dir: Path,
    tools: SpeechTools,
    force: bool = False,
) -> Path:
    """Build (or reuse cached) graphs for `wake_word`; returns their directory."""
    v_dir = graph_dir_for(train_dir, model.id, wake_word.id)
    signature = build_signature(wake_word)
    manifest_path = v_dir / "manifest.json"

    if v_dir.exists():
        old_signature = None
        try:
            old_signature = json.loads(manifest_path.read_text(encoding="utf-8"))[
                "signature"
            ]
        except (OSError, ValueError, KeyError):
            pass
        if old_signature == signature and not force:
            _LOGGER.debug("Graphs already built for %s", wake_word.id)
            return v_dir
        _LOGGER.info("Build inputs changed for %s; rebuilding graphs", wake_word.id)
        shutil.rmtree(v_dir, ignore_errors=True)

    _LOGGER.info(
        "Building graphs for %s (model=%s, %d phrase(s), %d confusable(s))",
        wake_word.id,
        model.id,
        len(wake_word.phrases),
        len(wake_word.confusables),
    )
    try:
        await asyncio.to_thread(_train, wake_word, model, v_dir, tools)
        manifest_path.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "wake_word": wake_word.id,
                    "model": model.id,
                    "phrases": [p.text for p in wake_word.phrases],
                    "confusables": [p.text for p in wake_word.confusables],
                    "lexicon": {w: list(p) for w, p in sorted(wake_word.lexicon.items())},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(v_dir, ignore_errors=True)
        raise

    _LOGGER.info("Graphs built for %s", wake_word.id)
    return v_dir


def write_grammar(
    path: Path,
    tag: str,
    wake_word: WakeWordConfig,
    filler_words: list[str],
) -> None:
    """Write one grammar in OpenFST text (acceptor) format."""
    with open(path, "w", encoding="utf-8") as grammar_file:
        if tag == "bg":
            for filler in filler_words:
                print(f"{_START} {_START} {filler} {FILLER_COST}", file=grammar_file)
            print(_START, file=grammar_file)
            return

        chains: list[tuple[tuple[str, ...], float]] = []
        seen: set[tuple[str, ...]] = set()

        def add(words: tuple[str, ...], cost: float) -> None:
            # First writer wins, so a word sequence that is both an accepted
            # phrase and (a prefix of) a confusable keeps its zero cost.
            if words and words not in seen:
                seen.add(words)
                chains.append((words, cost))

        for phrase in wake_word.phrases:
            if tag == "prefix":
                # A single-word phrase has no prefix; keep the whole phrase so
                # lr_prefix degenerates to the whole-phrase ratio.
                add(phrase.words[:-1] or phrase.words, 0.0)
            else:
                add(phrase.words, 0.0)

        if tag == "kwc":
            for confusable in wake_word.confusables:
                add(confusable.words, COMPETITOR_COST)
            # The bare last word in first position catches "<last> <last>"
            # and a bare "<last>" being accepted as the full phrase.
            for phrase in wake_word.phrases:
                if len(phrase.words) > 1:
                    add(phrase.words[-1:], COMPETITOR_COST)

        next_state = _END + 1
        for words, cost in chains:
            state = _START
            for index, word in enumerate(words):
                last = index == len(words) - 1
                target = _END if last else next_state
                if not last:
                    next_state += 1
                weight = f" {cost}" if index == 0 and cost else ""
                print(f"{state} {target} {word}{weight}", file=grammar_file)
                state = target

        for filler in filler_words:
            print(f"{_END} {_END} {filler} {FILLER_COST}", file=grammar_file)
        print(_END, file=grammar_file)


def _train(
    wake_word: WakeWordConfig,
    model: PreparedModel,
    v_dir: Path,
    tools: SpeechTools,
) -> None:
    v_dir.mkdir(parents=True, exist_ok=True)

    conf_dir = v_dir / "conf"
    if conf_dir.exists():
        shutil.rmtree(conf_dir)
    shutil.copytree(model.model_dir / "conf", conf_dir)

    # utils/*.sh source ./path.sh and resolve helpers through ./utils
    (v_dir / "path.sh").write_text("", encoding="utf-8")
    utils_link = v_dir / "utils"
    utils_link.unlink(missing_ok=True)
    utils_link.symlink_to(tools.egs_utils_dir, target_is_directory=True)

    # --- lexicon: wake/confusable words + one filler word per phone ---
    dict_local_dir = v_dir / "data" / "local" / "dict"
    dict_local_dir.mkdir(parents=True, exist_ok=True)
    for phone_file in model.phones_dir.glob("*.txt"):
        shutil.copy(phone_file, dict_local_dir / phone_file.name)

    lexicon = ModelLexicon(model.data_dir)
    try:
        fillers = lexicon.filler_inventory()
    finally:
        lexicon.close()

    with open(dict_local_dir / "lexicon.txt", "w", encoding="utf-8") as dict_file:
        for word, prons in sorted(wake_word.lexicon.items()):
            for pron in prons:
                print(word, pron, file=dict_file)
        for name, variants in fillers:
            for variant in variants:
                print(name, variant, file=dict_file)
        print(f"{UNK} SPN", file=dict_file)
        print(f"{SIL} SIL", file=dict_file)

    # --- prepare_lang once, then one lang copy + graph per grammar ---
    data_dir = v_dir / "data"
    lang_dir = data_dir / "lang"
    tools.run(
        "bash",
        [
            str(tools.egs_utils_dir / "prepare_lang.sh"),
            str(dict_local_dir),
            UNK,
            str(data_dir / "local" / "lang"),
            str(lang_dir),
        ],
        cwd=v_dir,
    )

    filler_words = [name for name, _ in fillers]
    for tag in GRAPH_TAGS:
        lang_tag_dir = data_dir / f"lang_{tag}"
        if lang_tag_dir.exists():
            shutil.rmtree(lang_tag_dir)
        shutil.copytree(lang_dir, lang_tag_dir, symlinks=True)

        grammar_path = lang_tag_dir / "grammar.fst.txt"
        write_grammar(grammar_path, tag, wake_word, filler_words)

        words_txt = lang_tag_dir / "words.txt"
        tools.run_pipeline(
            [
                "fstcompile",
                f"--isymbols={words_txt}",
                f"--osymbols={words_txt}",
                "--acceptor",
                str(grammar_path),
            ],
            ["fstarcsort", "--sort_type=ilabel", "-", str(lang_tag_dir / "G.fst")],
            cwd=v_dir,
        )

        tools.run(
            "bash",
            [
                str(tools.egs_utils_dir / "mkgraph.sh"),
                "--self-loop-scale",
                "1.0",
                str(lang_tag_dir),
                str(model.am_dir),
                str(v_dir / f"graph_{tag}"),
            ],
            cwd=v_dir,
        )
