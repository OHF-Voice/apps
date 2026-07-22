"""Speech-to-Phrase templates/responses sourced from the ``home-assistant-intents``
package (the ``speech_to_phrase`` tagged subset) instead of the add-on's own
``sentences/`` and ``responses/`` trees.

``home_assistant_intents.get_speech_to_phrase_intents(lang)`` returns the tagged
blocks in hassil's *converted* format (domain info under ``slots`` /
``requires_context``). This module:

  * adapts each block back to the add-on's authored shape
    (``name_domains`` / ``inferred_domain`` / ``context_area`` / ``response``),
    so the existing training / matcher / preset code keeps working, and
  * transpiles a template into the speech-to-phrase-lib grammar dialect
    (``<rules>`` substituted inline, range lists inlined to ``{from..to:slot}``)
    for the FST trainer, which -- unlike hassil -- has no rule support.

The hassil matcher needs no transpile: it is handed the raw templates plus the
package's ``lists`` and ``expansion_rules``.
"""
import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple

from hassil import Intents, normalize_whitespace
from hassil.sample import sample_sentence
from home_assistant_intents import (
    get_speech_to_phrase_intents,
    get_speech_to_phrase_languages,
)

_LOGGER = logging.getLogger("speech-to-phrase.s2p_intents")

_REF_RE = re.compile(r"\{([^{}]+)\}")
_INLINE_RANGE_RE = re.compile(r"^-?\d+\s*\.\.")


@lru_cache(maxsize=None)
def _load(lang: str) -> Optional[dict]:
    return get_speech_to_phrase_intents(lang)


def languages() -> List[str]:
    """Languages that ship Speech-to-Phrase templates in the package."""
    return get_speech_to_phrase_languages()


def has_language(lang: str) -> bool:
    return _load(lang) is not None


# --- block adaptation (converted -> authored) --------------------------------


def _authored_block(block: dict) -> dict:
    """Rebuild the add-on's authored block shape from a converted package block.

    Converted -> authored:
      * ``requires_context.domain`` (list)  -> ``name_domains``
      * ``slots.domain`` (str)              -> ``inferred_domain``
      * ``requires_context.area == {slot}`` -> ``context_area``
    ``response`` and any remaining fixed ``slots`` are carried through.
    """
    out: dict = {"sentences": list(block.get("sentences", []))}

    rc = dict(block.get("requires_context") or {})
    slots = dict(block.get("slots") or {})

    name_domains = rc.get("domain")
    if isinstance(name_domains, list):
        out["name_domains"] = list(name_domains)

    inferred_domain = slots.pop("domain", None)
    if isinstance(inferred_domain, str):
        out["inferred_domain"] = inferred_domain

    area = rc.get("area")
    if isinstance(area, dict) and area.get("slot"):
        out["context_area"] = True

    if slots:  # any fixed slot values that are not the inferred domain
        out["slots"] = slots

    if "response" in block:
        out["response"] = block["response"]

    return out


@lru_cache(maxsize=None)
def _combo_map(lang: str) -> Dict[Tuple[str, str], Tuple[dict, ...]]:
    """{(intent, combo): (authored_block, ...)} for a language."""
    data = _load(lang)
    if not data:
        return {}
    out: Dict[Tuple[str, str], List[dict]] = {}
    for intent, info in data.get("intents", {}).items():
        for block in info.get("data", []):
            combo = (block.get("metadata") or {}).get("slot_combination")
            if not combo:
                continue
            out.setdefault((intent, combo), []).append(_authored_block(block))
    return {key: tuple(blocks) for key, blocks in out.items()}


def combo_blocks(lang: str, intent: str, combo: str) -> List[dict]:
    """Authored data blocks for a combo (drop-in for the old per-combo YAML)."""
    return [dict(b) for b in _combo_map(lang).get((intent, combo), ())]


def combos(lang: str) -> List[Tuple[str, str]]:
    """Every (intent, combo) the package ships for a language, sorted."""
    return sorted(_combo_map(lang).keys())


# --- responses ---------------------------------------------------------------


def responses(lang: str) -> Dict[str, Dict[str, str]]:
    """{intent: {response_key: template}} from the package (or {})."""
    data = _load(lang)
    if not data:
        return {}
    intents = (data.get("responses") or {}).get("intents") or {}
    return {
        intent: dict(keys)
        for intent, keys in intents.items()
        if isinstance(keys, dict)
    }


# --- list definitions --------------------------------------------------------


@lru_cache(maxsize=None)
def _list_defs(lang: str) -> Tuple[Dict[str, Tuple[int, int, int]], Dict[str, Tuple[str, ...]]]:
    """(range_lists, text_lists) from the package's ``lists``.

    range_lists: name -> (from, to, step); text_lists: name -> (spoken value, ...).
    Wildcard lists are omitted (no tagged Speech-to-Phrase combo uses them).
    """
    data = _load(lang) or {}
    ranges: Dict[str, Tuple[int, int, int]] = {}
    texts: Dict[str, Tuple[str, ...]] = {}
    for name, spec in (data.get("lists") or {}).items():
        if not isinstance(spec, dict):
            continue
        if "range" in spec:
            rng = spec["range"]
            ranges[name] = (rng["from"], rng["to"], int(rng.get("step", 1)))
        elif "values" in spec:
            vals = []
            for v in spec["values"]:
                text = v.get("in") if isinstance(v, dict) else v
                if text:
                    vals.append(text)
            texts[name] = tuple(vals)
    return ranges, texts


def text_list_values(lang: str) -> Dict[str, List[str]]:
    """{list_name: [spoken values]} for every text list in the package."""
    _ranges, texts = _list_defs(lang)
    return {name: list(vals) for name, vals in texts.items()}


def _flatten_value(value: str) -> str:
    """Reduce a list value that is itself a template fragment to plain words
    (first alternative of each group/optional), e.g. ``(up|increase)`` -> ``up``,
    ``[securely] locked`` -> ``securely locked``."""
    while re.search(r"\([^()]*\)", value):
        value = re.sub(r"\(([^()]*)\)", lambda m: m.group(1).split("|")[0], value)
    while re.search(r"\[[^\[\]]*\]", value):
        value = re.sub(r"\[([^\[\]]*)\]", lambda m: m.group(1).split("|")[0], value)
    return " ".join(value.split())


def example_text_values(lang: str) -> Dict[str, List[str]]:
    """{text_list_name: [one representative value]} for rendering UI examples.

    First value of each text list, flattened to plain words. Numeric (range)
    slots are not included here -- their example values are chosen per slot name
    by the caller (see presets.load_example_values), so ``50 hours`` can be a
    sensible ``2 hours`` instead.
    """
    _ranges, texts = _list_defs(lang)
    return {
        name: [_flatten_value(vals[0])] for name, vals in texts.items() if vals
    }


def list_defs_dict(lang: str) -> dict:
    """The raw package ``lists`` dict (for the hassil matcher)."""
    return dict((_load(lang) or {}).get("lists") or {})


def expansion_rules(lang: str) -> Dict[str, str]:
    """The package ``expansion_rules`` dict (for the hassil matcher)."""
    return dict((_load(lang) or {}).get("expansion_rules") or {})


_RULE_REF_RE = re.compile(r"<([a-z0-9_]+)>")


def resolve_rules(text: str, lang: str) -> str:
    """Substitute ``<rule>`` -> ``(body)`` recursively so a template contains no
    expansion-rule references (used to render example sentences for the UI, which
    must not show raw ``<rules>``). Optionals/alternatives are left intact."""
    rules = expansion_rules(lang)

    def _sub(current: str, depth: int = 0) -> str:
        if depth > 25:
            return current
        expanded = _RULE_REF_RE.sub(
            lambda m: f"({rules[m.group(1)]})" if m.group(1) in rules else m.group(0),
            current,
        )
        if expanded != current and _RULE_REF_RE.search(expanded):
            return _sub(expanded, depth + 1)
        return expanded

    return _sub(text)


# --- grammar-dialect templates (for the FST trainer) -------------------------
#
# speech-to-phrase-lib's template parser is a strict subset of hassil: it has no
# ``<rule>`` support, no ``[a|b]`` (alternatives inside an optional), and no
# ``(a;b)`` permutations. Rather than transpile every hassil construct, we let
# hassil *sample* each template into its flat phrasings -- rules resolved,
# optionals/alternatives/permutations expanded -- while keeping ``{list}`` and
# range refs as placeholders (``expand_lists``/``expand_ranges`` off). The
# resulting strings contain only literals plus ``{...}`` refs, which the trainer
# parses trivially. Lean tagged blocks are small by design, so the expansion is
# bounded (tens of phrasings per combo).


def _rewrite_ref(
    content: str,
    range_lists: Dict[str, Tuple[int, int, int]],
    text_lists: Dict[str, Sequence[str]],
    referenced: Set[str],
) -> str:
    content = content.strip()
    if _INLINE_RANGE_RE.match(content):
        return "{" + content + "}"  # already an inline range
    name = content.split(":", 1)[0].strip()
    if name in range_lists:
        lo, hi, step = range_lists[name]
        body = f"{lo}..{hi}" if step in (1, 0, None) else f"{lo}..{hi},{step}"
        return "{" + body + "}"
    if name in text_lists or name in ("name", "area", "floor"):
        referenced.add(name)
        return "{" + name + "}"  # slot suffix stripped
    return "{" + content + "}"


def grammar_templates(sentences: Sequence[str], lang: str) -> Tuple[List[str], Set[str]]:
    """Flat, lib-dialect templates for a block's hassil sentences.

    Returns ``(templates, referenced_lists)``: every phrasing the sentences
    produce (rules/optionals/alternatives/permutations expanded, lists/ranges
    kept as refs), with range refs inlined to ``{lo..hi}`` and text/``name``/
    ``area``/``floor`` refs reduced to ``{name}``. ``referenced_lists`` names the
    text lists whose values the caller must add to ``list_values``.
    """
    rules = expansion_rules(lang)
    range_lists, text_lists = _list_defs(lang)
    intents = Intents.from_dict(
        {
            "language": lang,
            "intents": {"_G": {"data": [{"sentences": list(sentences)}]}},
            "expansion_rules": rules,
        }
    )
    parsed_rules = intents.expansion_rules
    referenced: Set[str] = set()
    out: List[str] = []
    for intent_data in intents.intents["_G"].data:
        for sentence in intent_data.sentences:
            for text in sample_sentence(
                sentence,
                slot_lists=None,
                expansion_rules=parsed_rules,
                expand_lists=False,
                expand_ranges=False,
            ):
                flat = normalize_whitespace(text).strip()
                rewritten = _REF_RE.sub(
                    lambda m: _rewrite_ref(m.group(1), range_lists, text_lists, referenced),
                    flat,
                )
                out.append(normalize_whitespace(rewritten).strip())
    return list(dict.fromkeys(out)), referenced
