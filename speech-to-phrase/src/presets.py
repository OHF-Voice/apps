"""Discover the built-in (pre-defined) slot-combinations the add-on ships, and
join them with metadata from home-assistant-intents' intents.yaml.

The curated Speech-to-Phrase templates live at
    sentences/<lang>/<Intent>/<slot_combination>.yaml
and intents.yaml supplies each combo's description / example / importance /
domains for display and default-enable decisions.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

IMPORTANCE_ORDER = ["required", "usable", "complete", "optional"]

# Functional grouping for the web UI (ordered). Intents not listed fall in "Other".
INTENT_GROUPS = [
    ("Device on", ["HassTurnOn"]),
    ("Device off", ["HassTurnOff"]),
    ("Lights & brightness", ["HassLightSet"]),
    ("Covers", ["HassSetPosition"]),
    ("Fans", ["HassFanSetSpeed"]),
    ("Climate", ["HassClimateGetTemperature", "HassClimateSetTemperature"]),
    ("Media & volume", [
        "HassMediaPause", "HassMediaUnpause", "HassMediaNext", "HassMediaPrevious",
        "HassMediaPlayerMute", "HassMediaPlayerUnmute", "HassSetVolume",
        "HassSetVolumeRelative", "HassMediaSearchAndPlay",
    ]),
    ("Timers", [
        "HassStartTimer", "HassCancelTimer", "HassCancelAllTimers", "HassPauseTimer",
        "HassUnpauseTimer", "HassIncreaseTimer", "HassDecreaseTimer", "HassTimerStatus",
    ]),
    ("Information", [
        "HassGetState", "HassGetCurrentTime", "HassGetCurrentDate", "HassGetWeather",
    ]),
    ("Other", ["HassNevermind", "HassRespond", "HassBroadcast"]),
]
_INTENT_TO_GROUP = {i: label for label, intents in INTENT_GROUPS for i in intents}
_OTHER = "Other"


def intent_group(intent: str) -> str:
    return _INTENT_TO_GROUP.get(intent, _OTHER)


def group_order() -> List[str]:
    labels = [label for label, _ in INTENT_GROUPS]
    if _OTHER not in labels:
        labels.append(_OTHER)
    return labels


# Representative numbers for inline ranges, by slot name (for example sentences).
# Chosen so the plural "[s]" in templates reads correctly (e.g. "2 hours").
_NUM_SAMPLE = {
    "minutes": "5", "seconds": "30", "hours": "2",
    "brightness": "50", "volume_level": "50", "position": "50",
    "percentage": "50", "temperature": "70",
}

# Friendlier nouns when no real entity exists for a {name} domain.
_DOMAIN_NOUN = {
    "climate": "thermostat", "media_player": "media player",
    "binary_sensor": "sensor", "input_boolean": "switch",
}


def _slot_class(slot: str) -> str:
    """Colour category for a slot (mirrors the intent-sentences website)."""
    if slot == "name":
        return "name"
    if slot in ("area", "floor", "color"):
        return slot
    if slot == "state":
        return "state"
    if slot in ("domain", "device_class", "media_class"):
        return "class"
    if slot in _NUM_SAMPLE:
        return "number"
    return "other"


def _span(value: str, slot: str) -> str:
    import html
    return (f'<span class="slot slot-{_slot_class(slot)}" title="{html.escape(slot)}">'
            f"{html.escape(value)}</span>")


def _resolve_slot(
    content: str, domains: Optional[Sequence[str]],
    entities: Dict[str, str], slot_lists: Dict[str, List[str]],
):
    """(value, slot_name) for one ``{...}`` token."""
    m = re.fullmatch(
        r"-?\d+\s*\.\.\s*-?\d+(?:\s*[,/]\s*-?\d+)?(?::([a-z_]+))?", content
    )
    if m:
        slot = m.group(1) or "number"
        return _NUM_SAMPLE.get(slot, "50"), slot
    if content == "name":
        if domains:
            rep = next((n for n, d in sorted(entities.items()) if d in domains), None)
            rep = rep or _DOMAIN_NOUN.get(domains[0], domains[0].replace("_", " "))
        else:
            rep = next(iter(sorted(entities)), "device")
        return rep, "name"
    # {list} or {list:slot}: look up by list name, label by slot name.
    list_name, _, slot_name = content.partition(":")
    slot_name = slot_name or list_name
    vals = slot_lists.get(list_name)
    if vals:
        return str(vals[0]), slot_name
    return {"area": "kitchen", "floor": "first floor"}.get(list_name, list_name), slot_name


def _example_html(
    sentence: str, domains: Optional[Sequence[str]],
    entities: Dict[str, str], slot_lists: Dict[str, List[str]],
) -> str:
    """Render one example as HTML with each slot value wrapped in a span."""
    import html
    s = sentence
    # Resolve structure first; slot tokens ({...}) contain no []/() so survive.
    # Pick the first alternative of each group and the content of each optional
    # (also first alternative, e.g. "[the|my]" -> "the").
    while re.search(r"\([^()]*\)", s):
        s = re.sub(r"\(([^()]*)\)", lambda m: m.group(1).split("|")[0], s)
    while re.search(r"\[[^\[\]]*\]", s):
        s = re.sub(r"\[([^\[\]]*)\]", lambda m: m.group(1).split("|")[0], s)
    s = " ".join(s.split())  # safe: slot tokens have no spaces
    out: List[str] = []
    pos = 0
    for m in re.finditer(r"\{([^}]*)\}", s):
        out.append(html.escape(s[pos:m.start()]))
        value, slot = _resolve_slot(
            m.group(1).strip(), domains, entities, slot_lists
        )
        out.append(_span(value, slot))
        pos = m.end()
    out.append(html.escape(s[pos:]))
    return "".join(out).strip()


def sample_sentence(
    sentence: str, domains: Optional[Sequence[str]],
    entities: Dict[str, str], slot_lists: Dict[str, List[str]],
) -> str:
    """Plain-text concrete utterance from a template (for live validation)."""
    import html
    return html.unescape(re.sub(r"<[^>]+>", "", _example_html(
        sentence, domains, entities, slot_lists)))


def combo_examples(
    s2p_repo: Path, lang: str, intent: str, combo: str,
    entities: Dict[str, str], slot_lists: Dict[str, List[str]],
) -> dict:
    """Highlighted examples for a combo:
      * ``examples``  -- one per data block (used when the combo isn't split), and
      * ``by_domain`` -- one per targeted domain (for per-domain checkboxes),
                         each rendered with a {name} of that domain.
    Plus the ordered ``domains`` list.
    """
    import s2p_intents
    import training
    blocks = s2p_intents.combo_blocks(lang, intent, combo)
    if not blocks:
        return {"domains": [], "by_domain": {}, "examples": []}
    # Sample values for the package's lists (states, colors, numeric ranges)
    # so {state}/{brightness}/... render as words; caller lists (area/floor)
    # take precedence.
    slot_lists = {**s2p_intents.example_slot_values(lang), **(slot_lists or {})}
    domains: List[str] = []
    by_domain: Dict[str, str] = {}
    examples: List[str] = []
    for block in blocks:
        sents = block.get("sentences") or []
        if not sents:
            continue
        # Package templates use <rules>; resolve them so examples read as plain
        # sentences (the structural renderer only handles []/()/{...}).
        first = s2p_intents.resolve_rules(sents[0], lang)
        ex = _example_html(first, block.get("name_domains"), entities, slot_lists)
        if ex and ex not in examples:
            examples.append(ex)
        for d in training.block_domains(block):
            if d not in domains:
                domains.append(d)
            if d not in by_domain:
                by_domain[d] = _example_html(first, [d], entities, slot_lists)
    return {"domains": domains, "by_domain": by_domain, "examples": examples}


def load_intents_meta(intents_yaml: Path) -> dict:
    return yaml.safe_load(intents_yaml.read_text()) or {}


def _combo_importance(combo_def: dict) -> str:
    """Explicit `importance`, else the most-important bucket present across the
    domain maps (required < usable < complete < optional)."""
    if combo_def.get("importance") in IMPORTANCE_ORDER:
        return combo_def["importance"]
    best = None
    for key in ("name_domains", "inferred_domains"):
        m = combo_def.get(key)
        if isinstance(m, dict):
            for imp in m:
                if imp in IMPORTANCE_ORDER and (
                    best is None
                    or IMPORTANCE_ORDER.index(imp) < IMPORTANCE_ORDER.index(best)
                ):
                    best = imp
    return best or "optional"


def _combo_domains(combo_def: dict) -> List[str]:
    out = set()
    for key in ("name_domains", "inferred_domains"):
        m = combo_def.get(key)
        if isinstance(m, dict):
            for vals in m.values():
                out.update(vals)
        elif isinstance(m, list):
            out.update(m)
    return sorted(out)


def available_combos(s2p_repo: Path, lang: str, meta: dict) -> List[dict]:
    """Every (intent, combo) the package ships tagged templates for, in `lang`."""
    import s2p_intents

    combos: List[dict] = []
    for intent, combo in s2p_intents.combos(lang):
        cdef = (meta.get(intent) or {}).get("slot_combinations", {}).get(combo, {})
        example = cdef.get("example", "")
        if isinstance(example, list):
            example = example[0] if example else ""
        combos.append(
            {
                "intent": intent,
                "combo": combo,
                "description": cdef.get("description", ""),
                "example": example,
                "importance": _combo_importance(cdef),
                "domains": _combo_domains(cdef),
            }
        )
    return combos


def languages(s2p_repo: Path) -> List[str]:
    import s2p_intents

    return s2p_intents.languages()


def intent_catalog(meta: dict) -> List[dict]:
    """All known HA intents (name, description, slots) from intents.yaml, for the
    custom intent-mode picker."""
    out: List[dict] = []
    for name, d in meta.items():
        if not isinstance(d, dict) or "slot_combinations" not in d:
            continue  # only entries that look like intents
        slots = d.get("slots") or {}
        out.append({
            "name": name,
            "description": d.get("description", ""),
            "slots": [
                {
                    "name": s,
                    "description": (sd or {}).get("description", ""),
                    "required": bool((sd or {}).get("required")),
                }
                for s, sd in slots.items()
            ],
        })
    return sorted(out, key=lambda x: x["name"])


def default_enabled(combos: List[dict], threshold: str = "usable") -> List[List[str]]:
    """Enable combos at or above the importance threshold (required, usable, ...)."""
    cut = IMPORTANCE_ORDER.index(threshold)
    return [
        [c["intent"], c["combo"]]
        for c in combos
        if IMPORTANCE_ORDER.index(c["importance"]) <= cut
    ]
