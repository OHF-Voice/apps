"""Discover the built-in (pre-defined) slot-combinations the add-on ships, and
join them with metadata from home-assistant-intents.

The curated Speech-to-Phrase templates live at
    sentences/<lang>/<Intent>/<slot_combination>.yaml
and ``home_assistant_intents.get_intent_info()`` supplies each combo's
description / example / importance / domains for display and default-enable
decisions.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

IMPORTANCE_ORDER = ["required", "usable", "complete", "optional"]

# Display order for the web UI (sort of slot combinations by importance). This
# intentionally differs from IMPORTANCE_ORDER, which governs default-enable
# thresholds; here "optional" sorts ahead of "complete".
IMPORTANCE_SORT = ["required", "usable", "optional", "complete"]


def _importance_sort_key(importance: str) -> int:
    try:
        return IMPORTANCE_SORT.index(importance)
    except ValueError:
        return len(IMPORTANCE_SORT)

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


# Representative numbers for numeric slots (example sentences only). Chosen so
# the plural "[s]" in templates reads correctly (e.g. "2 hours", not "50 hours").
# Users can override/extend these via example_values.yaml (see below).
_NUM_SAMPLE = {
    "minutes": "5", "seconds": "30", "hours": "2",
    "brightness": "50", "volume_level": "50", "position": "50",
    "percentage": "50", "temperature": "70",
}


def load_example_values(s2p_repo: Optional[Path]) -> Dict[str, str]:
    """Numeric example values by slot name, built-in defaults merged with the
    user's ``example_values.yaml`` (if present at the add-on root).

    The file is a simple ``slot: value`` map, optionally nested under an
    ``example_values:`` key, e.g.::

        example_values:
          hours: 2
          temperature: 68
    """
    values = dict(_NUM_SAMPLE)
    if s2p_repo is not None:
        path = s2p_repo / "example_values.yaml"
        if path.exists():
            try:
                doc = yaml.safe_load(path.read_text()) or {}
            except Exception:  # noqa: BLE001
                doc = {}
            overrides = doc.get("example_values", doc)
            if isinstance(overrides, dict):
                for key, val in overrides.items():
                    values[str(key)] = str(val)
    return values


# Friendlier nouns when no real entity exists for a {name} domain.
_DOMAIN_NOUN = {
    "climate": "thermostat", "media_player": "media player",
    "binary_sensor": "sensor", "input_boolean": "switch",
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
        vals = slot_lists.get(slot)
        return (str(vals[0]) if vals else _NUM_SAMPLE.get(slot, "50")), slot
    if content == "name":
        if domains:
            rep = next((n for n, d in sorted(entities.items()) if d in domains), None)
            rep = rep or _DOMAIN_NOUN.get(domains[0], domains[0].replace("_", " "))
        else:
            rep = next(iter(sorted(entities)), "device")
        return rep, "name"
    # {list} or {list:slot}: label by slot name; resolve by slot name (numeric
    # example values are keyed this way, e.g. timer_hours:hours -> "hours") and
    # fall back to the list name (text lists are keyed by list name).
    list_name, _, slot_name = content.partition(":")
    slot_name = slot_name or list_name
    vals = slot_lists.get(slot_name) or slot_lists.get(list_name)
    if vals:
        return str(vals[0]), slot_name
    return {"area": "kitchen", "floor": "first floor"}.get(list_name, list_name), slot_name


def _canonical(sentence: str) -> str:
    """Collapse ``()``/``[]`` structure to the canonical wording: first
    alternative of each group, first alternative of each optional (e.g.
    ``[the|my]`` -> ``the``). Slot tokens ({...}) contain no []/() so survive."""
    s = sentence
    while re.search(r"\([^()]*\)", s):
        s = re.sub(r"\(([^()]*)\)", lambda m: m.group(1).split("|")[0], s)
    while re.search(r"\[[^\[\]]*\]", s):
        s = re.sub(r"\[([^\[\]]*)\]", lambda m: m.group(1).split("|")[0], s)
    return " ".join(s.split())  # safe: slot tokens have no spaces


def _render(sentence: str, slot_fn) -> str:
    """Render a (structure-resolved) sentence to HTML, mapping each ``{...}``
    token through ``slot_fn(content) -> html``."""
    import html
    s = _canonical(sentence)
    out: List[str] = []
    pos = 0
    for m in re.finditer(r"\{([^}]*)\}", s):
        out.append(html.escape(s[pos:m.start()]))
        out.append(slot_fn(m.group(1).strip()))
        pos = m.end()
    out.append(html.escape(s[pos:]))
    return "".join(out).strip()


def _example_html(
    sentence: str, domains: Optional[Sequence[str]],
    entities: Dict[str, str], slot_lists: Dict[str, List[str]],
) -> str:
    """Render one example as HTML, each slot value wrapped in a highlight span
    (one representative value per slot)."""
    def span_fn(content: str) -> str:
        value, slot = _resolve_slot(content, domains, entities, slot_lists)
        return _span(value, slot)

    return _render(sentence, span_fn)


# Slots whose values are an enumerable vocabulary become a <select> in the
# interactive example; numeric/range slots stay a single representative value.
_SELECT_CAP = 40


def _slot_options(
    content: str, domains: Optional[Sequence[str]],
    entities: Dict[str, str], slot_lists: Dict[str, List[str]],
):
    """(values, slot_name) for an enumerable slot, or (None, slot_name) when the
    slot isn't a discrete list (numeric range) -- rendered as a span instead."""
    if re.fullmatch(
        r"-?\d+\s*\.\.\s*-?\d+(?:\s*[,/]\s*-?\d+)?(?::([a-z_]+))?", content
    ):
        return None, "number"  # numeric range: not a picklist
    if content == "name":
        if domains:
            vals = [n for n, d in sorted(entities.items()) if d in domains]
        else:
            vals = sorted(entities)
        return vals, "name"
    list_name, _, slot_name = content.partition(":")
    slot_name = slot_name or list_name
    vals = slot_lists.get(slot_name) or slot_lists.get(list_name) or []
    return [str(v) for v in vals], slot_name


def _slot_field(
    content: str, domains: Optional[Sequence[str]],
    entities: Dict[str, str], slot_lists: Dict[str, List[str]],
) -> str:
    """A <select> of the (filtered) values for an enumerable slot, so the user
    sees the actual vocabulary in context. Falls back to a highlight span for
    numeric ranges and single-option slots."""
    import html
    values, slot = _slot_options(content, domains, entities, slot_lists)
    if not values or len(values) <= 1:
        value, slot = _resolve_slot(content, domains, entities, slot_lists)
        return _span(value, slot)
    cls = _slot_class(slot)
    shown = values[:_SELECT_CAP]
    opts = "".join(f"<option>{html.escape(v)}</option>" for v in shown)
    extra = len(values) - len(shown)
    if extra > 0:
        opts += f'<option disabled>… and {extra} more</option>'
    return (
        f'<select class="slot slot-{cls} slot-select" title="{html.escape(slot)}" '
        f'aria-label="{html.escape(slot)}" onclick="event.stopPropagation()">'
        f'{opts}</select>'
    )


def _example_interactive(
    sentence: str, domains: Optional[Sequence[str]],
    entities: Dict[str, str], slot_lists: Dict[str, List[str]],
) -> str:
    """Like _example_html, but enumerable slots render as a <select> of their
    filtered values."""
    return _render(
        sentence, lambda c: _slot_field(c, domains, entities, slot_lists)
    )


def _find_group(s: str):
    """(open_idx, close_idx, open_char) of the first top-level ``(``/``[`` group,
    or None. Nesting of the same bracket type is respected."""
    for i, ch in enumerate(s):
        if ch in "([":
            close = ")" if ch == "(" else "]"
            depth = 1
            for j in range(i + 1, len(s)):
                if s[j] == ch:
                    depth += 1
                elif s[j] == close:
                    depth -= 1
                    if depth == 0:
                        return i, j, ch
            break
    return None


def _split_alts(inner: str) -> List[str]:
    """Split on top-level ``|`` (ignoring ``|`` inside nested groups)."""
    parts: List[str] = []
    depth = 0
    buf = ""
    for ch in inner:
        if ch in "([":
            depth += 1
            buf += ch
        elif ch in ")]":
            depth -= 1
            buf += ch
        elif ch == "|" and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    return parts


def _phrasings(sentence: str, cap: int) -> List[str]:
    """Distinct wordings from the ``()`` alternatives (canonical first, slot
    tokens intact, deduped). Optionals are kept as their canonical branch, so the
    variants surface real word choices ("turn on" / "switch on") rather than a
    combinatorial blowup of "[please]"/"[the]". Grows to at most `cap`."""
    out: List[str] = []

    def rec(s: str) -> None:
        if len(out) >= cap:
            return
        g = _find_group(s)
        if g is None:
            w = " ".join(s.split())
            if w and w not in out:
                out.append(w)
            return
        i, j, ch = g
        prefix, inner, suffix = s[:i], s[i + 1:j], s[j + 1:]
        opts = [_split_alts(inner)[0]] if ch == "[" else _split_alts(inner)
        for opt in opts:
            if len(out) >= cap:
                return
            rec(prefix + opt + suffix)

    rec(sentence)
    return out


def sample_sentence(
    sentence: str, domains: Optional[Sequence[str]],
    entities: Dict[str, str], slot_lists: Dict[str, List[str]],
) -> str:
    """Plain-text concrete utterance from a template (for live validation)."""
    import html
    return html.unescape(re.sub(r"<[^>]+>", "", _example_html(
        sentence, domains, entities, slot_lists)))


def _example_card(
    sentences: Sequence[str], domains: Optional[Sequence[str]],
    entities: Dict[str, str], slot_lists: Dict[str, List[str]],
) -> str:
    """A combo example: the first template's canonical wording with <select>s for
    its enumerable slots, plus a collapsed "N more ways to say this" list of the
    other phrasings (plain text, one representative value each).

    Alternate phrasings come from BOTH the sibling sentence templates in the
    block ("turn off {name}" / "switch off {name}" / "{name} off") and the ``()``
    alternatives inside each template."""
    import html
    if not sentences:
        return ""
    main = _example_interactive(sentences[0], domains, entities, slot_lists)
    if not main:
        return ""
    variants: List[str] = []
    seen = {sample_sentence(sentences[0], domains, entities, slot_lists)}
    for tmpl in sentences:
        for w in _phrasings(tmpl, 64):
            txt = sample_sentence(w, domains, entities, slot_lists)
            if txt and txt not in seen:
                seen.add(txt)
                variants.append(txt)
    parts = [f'<div class="ex-main">{main}</div>']
    if variants:
        shown = variants[:6]
        items = "".join(f'<div class="ex-alt">{html.escape(v)}</div>' for v in shown)
        if len(variants) > len(shown):
            items += f'<div class="ex-alt more">… and {len(variants) - len(shown)} more</div>'
        plural = "s" if len(variants) != 1 else ""
        parts.append(
            f'<details class="phrasings"><summary>{len(variants)} more way{plural} '
            f'to say this</summary>{items}</details>'
        )
    return "".join(parts)


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
    import gating
    import s2p_intents
    import training

    info = training.as_entity_info(entities)
    blocks = s2p_intents.combo_blocks(lang, intent, combo)
    if not blocks:
        return {"domains": [], "by_domain": {}, "examples": []}
    # Example slot values: numeric samples by slot name (user-overridable via
    # example_values.yaml) + text-list samples by list name; caller-supplied
    # lists (area/floor) win.
    base_slots = {
        **{k: [v] for k, v in load_example_values(s2p_repo).items()},
        **s2p_intents.example_text_values(lang),
        **(slot_lists or {}),
    }
    name_domain = {r.name: r.domain for r in info.records}
    capability = gating.required_capability(intent, combo)

    def _entity_map(domains):
        """{name: domain} restricted to entities of `domains` that support the
        combo's capability -- so an example never names an incapable device."""
        allowed = set(info.names(domains, capability))
        return {n: name_domain[n] for n in allowed if n in name_domain}

    domains: List[str] = []
    by_domain: Dict[str, str] = {}
    examples: List[str] = []
    for block in blocks:
        sents = block.get("sentences") or []
        if not sents:
            continue
        nd = block.get("name_domains")
        inferred = block.get("inferred_domain")
        # Same capability/domain gate as the grammar: no capable entity -> the
        # combo isn't trained, so it gets no example either.
        if not gating.keep_block(
            nd, inferred, capability, gating.capability_domains(intent), info
        ):
            continue

        # {area}/{floor} are not narrowed by domain co-occurrence (the command
        # stays speakable everywhere), so the example just draws from the full
        # area/floor lists.
        block_slots = dict(base_slots)

        # {name} must name a capable entity of the block's domains.
        if nd:
            ent_map = _entity_map(nd)
            if not ent_map:
                continue
        else:
            ent_map = name_domain

        # Package templates use <rules>; resolve them so examples read as plain
        # sentences (the structural renderer only handles []/()/{...}). Every
        # sentence in the block is a phrasing alternative, so resolve them all.
        resolved: List[str] = []
        for s in sents:
            try:
                resolved.append(s2p_intents.resolve_rules(s, lang))
            except Exception:  # noqa: BLE001
                resolved.append(s)
        ex = _example_card(resolved, nd, ent_map, block_slots)
        if ex and ex not in examples:
            examples.append(ex)
        for d in training.block_domains(block):
            d_map = _entity_map([d])
            if nd and not d_map:  # name-based domain with no capable entity
                continue
            if d not in domains:
                domains.append(d)
            if d not in by_domain:
                # Per-domain rows stay compact: canonical wording + selects, no
                # phrasing expander.
                by_domain[d] = _example_interactive(
                    resolved[0], [d], d_map or name_domain, block_slots
                )
    return {"domains": domains, "by_domain": by_domain, "examples": examples}


def load_intents_meta() -> dict:
    """Intent metadata (descriptions, examples, importance, domains) straight
    from ``home_assistant_intents.get_intent_info()``."""
    from home_assistant_intents import get_intent_info

    return get_intent_info() or {}


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
    # Sort by importance for the web UI (stable: equal-importance combos keep
    # their discovery order). Grouping in the frontend preserves this order, so
    # each group lists its combos required -> usable -> optional -> complete.
    combos.sort(key=lambda c: _importance_sort_key(c["importance"]))
    return combos


def languages(s2p_repo: Path) -> List[str]:
    import s2p_intents

    return s2p_intents.languages()


def intent_catalog(meta: dict) -> List[dict]:
    """All known HA intents (name, description, slots) from get_intent_info(),
    for the custom intent-mode picker."""
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
