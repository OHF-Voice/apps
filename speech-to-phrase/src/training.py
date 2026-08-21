"""Training pipeline for Speech-to-Phrase.

The grammar for a language is assembled from two sources:
  1. enabled built-in slot-combinations (curated S2P templates), and
  2. the user's free-text custom sentences,
then compiled by speech-to-phrase-lib's Recognizer.

DOMAIN-SCOPED NAMES (the important bit)
---------------------------------------
speech-to-phrase-lib's `train(sentences, list_values)` takes ONE list_values
dict for the whole grammar, so a bare `{name}` -> all-entities flattens away the
entity<->domain binding and makes nonsense like "front door on" (a lock spoken
in an on-able phrasing) recognizable.

Each curated sentence-set declares `name_domains`. We therefore bind its `{name}`
to a list scoped to exactly those domains (`{name__light_switch_...}`), populated
from the user's real entities. Two consequences:
  * an entity can only be spoken in phrasings valid for ITS domain, and
  * a sentence whose name_domains match no entity is dropped entirely -- which is
    also requirement #1 ("no vacuum sentences if you have no vacuum").

In the add-on, `entities` and the area/floor lists come from the live Home
Assistant registry (SUPERVISOR_TOKEN + /core/api); for local dev they come from a
fixture file (--entities-file) or the built-in DEV defaults below.
"""
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import aiohttp
import yaml

_LOGGER = logging.getLogger("speech-to-phrase.training")


def _norm_value(s: str) -> str:
    """Normalize a slot value to match the acoustic vocab and template
    normalization: NFC, lowercase, collapsed whitespace. Without this, a
    mixed-case name like 'Office Lamp' tokenizes to '<unk>ffice <unk>amp' on a
    lowercase-subword model and never matches."""
    return " ".join(unicodedata.normalize("NFC", s).lower().split())


def _norm_values(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(v for v in (_norm_value(x) for x in values) if v))

# Local-dev fallbacks. The container replaces these with the live HA registry +
# home-assistant-intents lists. Entities live in DEV_ENTITY_RECORDS (below).
DEV_SLOT_LISTS: Dict[str, List[str]] = {
    # Only area/floor are dev fallbacks now (the container overrides them from
    # the live HA registry). Text lists (color, states, volume_step, ...) come
    # from the home-assistant-intents package via s2p_intents.
    "area": ["kitchen", "office", "living room"],
    "floor": ["first floor", "second floor"],
}

# Enriched dev fallback (name -> domain + capabilities + area/floor), so the
# entity-aware gating can be exercised without Home Assistant. Deliberately
# uneven: only kitchen has lights/a fan, the cover can't be positioned.
DEV_ENTITY_RECORDS: List[dict] = [
    {"name": "overhead light", "domain": "light",
     "features": ["brightness", "color"], "area": "kitchen", "floor": "first floor"},
    {"name": "kitchen lamp", "domain": "light",
     "features": ["brightness"], "area": "kitchen", "floor": "first floor"},
    {"name": "kitchen fan", "domain": "fan",
     "features": ["set_speed"], "area": "kitchen", "floor": "first floor"},
    {"name": "garage door", "domain": "cover",
     "features": [], "area": "living room", "floor": "first floor"},
    {"name": "front door", "domain": "lock",
     "features": [], "area": "living room", "floor": "first floor"},
]


def _ws_url(api_url: str) -> str:
    """Derive the HA websocket URL from the REST api URL. Direct HA:
    .../api -> .../api/websocket. Supervisor: /core/api -> /core/websocket."""
    from urllib.parse import urlparse, urlunparse

    p = urlparse(api_url)
    scheme = "wss" if p.scheme == "https" else "ws"
    base = p.path[:-4] if p.path.endswith("/api") else p.path
    ws_path = (base + "/websocket") if base else "/api/websocket"
    return urlunparse((scheme, p.netloc, ws_path, "", "", ""))


def _registry_names(items: list) -> List[str]:
    """name + aliases for each registry entry, de-duped."""
    out: List[str] = []
    for it in items:
        for n in [it.get("name")] + list(it.get("aliases") or []):
            if n:
                out.append(n)
    return list(dict.fromkeys(out))


async def _areas_floors(api_url: str, token: str) -> Tuple[List[str], List[str]]:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(_ws_url(api_url), max_msg_size=0) as ws:
            assert (await ws.receive_json())["type"] == "auth_required"
            await ws.send_json({"type": "auth", "access_token": token})
            assert (await ws.receive_json())["type"] == "auth_ok"
            await ws.send_json({"id": 1, "type": "config/area_registry/list"})
            am = await ws.receive_json()
            assert am.get("success"), am
            await ws.send_json({"id": 2, "type": "config/floor_registry/list"})
            fm = await ws.receive_json()
            assert fm.get("success"), fm
    return _registry_names(am["result"]), _registry_names(fm["result"])


def areas_floors_from_hass(api_url: str, token: str) -> Tuple[List[str], List[str]]:
    """(area names, floor names) — all of them, incl. aliases — from HA."""
    import asyncio

    return asyncio.run(_areas_floors(api_url, token))


async def _entity_records(api_url: str, token: str) -> List[dict]:
    """Enriched records for conversation-exposed entities: one per name/alias,
    carrying domain, device_class, capability tokens, and area/floor names.

    Capabilities come from the live state attributes (supported_features /
    supported_color_modes); area comes from the entity's own area_id, falling
    back to its device's area, resolved to a name via the area registry (and its
    floor). Best-effort: any piece that can't be fetched is simply left unknown,
    which the gating treats permissively.
    """
    import gating

    async with aiohttp.ClientSession() as session:
        # Live states -> attributes by entity_id.
        async with session.get(
            f"{api_url}/states", headers={"Authorization": f"Bearer {token}"}
        ) as resp:
            states = await resp.json()
        attrs = {
            s["entity_id"]: (s.get("attributes") or {})
            for s in states
            if s.get("entity_id")
        }

        async with session.ws_connect(_ws_url(api_url), max_msg_size=0) as ws:
            assert (await ws.receive_json())["type"] == "auth_required"
            await ws.send_json({"type": "auth", "access_token": token})
            assert (await ws.receive_json())["type"] == "auth_ok"

            async def call(msg_id, type_, **kw):
                await ws.send_json({"id": msg_id, "type": type_, **kw})
                m = await ws.receive_json()
                return m["result"] if m.get("success") else None

            expose = await call(1, "homeassistant/expose_entity/list")
            exposed = [
                eid
                for eid, info in ((expose or {}).get("exposed_entities") or {}).items()
                if info.get("conversation")
            ]
            if not exposed:
                return []
            entries = await call(
                2, "config/entity_registry/get_entries", entity_ids=exposed
            ) or {}
            devices = await call(3, "config/device_registry/list") or []
            areas = await call(4, "config/area_registry/list") or []
            floors = await call(5, "config/floor_registry/list") or []

    device_area = {d["id"]: d.get("area_id") for d in devices}
    floor_name = {f["floor_id"]: (f.get("name") or "") for f in floors}
    area_name = {a["area_id"]: (a.get("name") or "") for a in areas}
    area_floor = {a["area_id"]: floor_name.get(a.get("floor_id")) for a in areas}

    records: List[dict] = []
    for eid in exposed:
        info = entries.get(eid) or {}
        if info.get("disabled_by") is not None:
            continue
        domain = eid.split(".", 1)[0] if "." in eid else ""
        if not domain:
            continue
        names: List[str] = []
        primary = info.get("name") or info.get("original_name")
        if primary:
            names.append(primary)
        names.extend(a for a in (info.get("aliases") or []) if a)
        friendly = attrs.get(eid, {}).get("friendly_name")
        if not names and friendly:
            names.append(friendly)

        area_id = info.get("area_id") or device_area.get(info.get("device_id"))
        area = area_name.get(area_id) if area_id else None
        floor = area_floor.get(area_id) if area_id else None
        features = sorted(gating.capabilities_from_attributes(domain, attrs.get(eid, {})))
        device_class = attrs.get(eid, {}).get("device_class")

        for name in names:
            name = name.strip()
            if name:
                records.append({
                    "name": name,
                    "domain": domain,
                    "device_class": device_class,
                    "features": features,
                    "area": area,
                    "floor": floor,
                })
    return records


def entity_records_from_hass(api_url: str, token: str) -> List[dict]:
    """Enriched entity records for gating (see _entity_records)."""
    import asyncio

    return asyncio.run(_entity_records(api_url, token))


def _name_list_key(domains: Sequence[str]) -> str:
    return "name__" + "_".join(sorted(domains))


def block_domains(block: dict) -> List[str]:
    """Domains a data block targets: its name_domains, else [inferred_domain]."""
    nd = block.get("name_domains")
    if nd:
        return list(nd)
    inf = block.get("inferred_domain")
    return [inf] if inf else []


def combo_domains(s2p_repo: Path, lang: str, intent: str, combo: str) -> List[str]:
    """Distinct domains a combo targets across its blocks, in first-seen order."""
    import s2p_intents

    out: List[str] = []
    for block in s2p_intents.combo_blocks(lang, intent, combo):
        for d in block_domains(block):
            if d not in out:
                out.append(d)
    return out


def enabled_domain_map(
    entries: Sequence,
) -> Dict[Tuple[str, str], Optional[frozenset]]:
    """Parse enabled.json entries into {(intent, combo): allowed_domains}.

    Each entry is ``[intent, combo]`` (all domains -> value None) or
    ``[intent, combo, [domains...]]`` (only those domains -> a frozenset)."""
    out: Dict[Tuple[str, str], Optional[frozenset]] = {}
    for e in entries:
        if len(e) >= 3 and e[2] is not None:
            out[(e[0], e[1])] = frozenset(e[2])
        else:
            out[(e[0], e[1])] = None
    return out


def combo_blocks(doc: dict, extras_for_combo: Optional[Sequence[str]]) -> List[dict]:
    """A combo's data blocks (from its file) plus, if the user added extra
    phrasings, one synthesized block that inherits the first block's metadata
    (name_domains / inferred_domain / context_area / response)."""
    blocks = list(doc.get("data", []))
    extras = [s for s in (extras_for_combo or []) if s and s.strip()]
    if extras:
        tmpl = blocks[0] if blocks else {}
        extra = {"sentences": extras}
        for k in ("name_domains", "inferred_domain", "context_area", "response"):
            if k in tmpl:
                extra[k] = tmpl[k]
        blocks.append(extra)
    return blocks


def _effective_name_domains(
    block: dict, allowed: Optional[frozenset]
) -> Tuple[bool, Optional[List[str]]]:
    """(include_block, name_domains_to_use) after applying the allowed-domain
    filter. allowed=None means everything is allowed."""
    nd = block.get("name_domains")
    if nd:
        eff = list(nd) if allowed is None else [d for d in nd if d in allowed]
        return (bool(eff), eff)
    inf = block.get("inferred_domain")
    if inf is not None and allowed is not None and inf not in allowed:
        return (False, None)
    return (True, None)


def as_entity_info(entities):
    """Normalise the entities argument into a gating.EntityInfo.

    Accepts a plain ``{name: domain}`` dict (features/area unknown -> no
    capability/area gating), a list of enriched record dicts, or an EntityInfo.
    """
    import gating

    if isinstance(entities, gating.EntityInfo):
        return entities
    if isinstance(entities, dict):
        return gating.EntityInfo(gating.records_from_mapping(entities))
    return gating.EntityInfo(gating.records_from_dicts(entities))


def _expand_block(
    sentences: Sequence[str],
    name_domains: Optional[Sequence[str]],
    capability: Optional[str],
    info,
    templates: List[str],
    list_values: Dict[str, List[str]],
    ov=None,
    combo_key: str = "",
    canonical_lists: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Append a block's sentences to `templates`, rewriting `{name}` to a
    domain-scoped list and applying the capability gate (see
    gating.scope_sentence). Slot values are normalised to the acoustic vocab; a
    sentence with no matching capable entity is dropped.

    `ov` (overrides.Overrides) applies the user's per-command exclusions and
    aliases: an excluded slot is rebound to a command-scoped list so narrowing it
    can't leak into other commands, and every value is expanded to its spoken
    forms (the grammar only ever needs those -- the canonical name is recovered
    by the matcher)."""
    import gating
    import overrides as ovr

    ov = ov or ovr.EMPTY

    for sentence in sentences:
        rewritten, lists = gating.scope_sentence(
            sentence, name_domains, capability, info
        )
        if rewritten is None:
            continue
        dropped = False
        for key, values in lists.items():
            narrowed_key, kept = ov.narrow(combo_key, "name", values)
            if not kept:
                dropped = True  # every entity excluded from this command
                break
            if narrowed_key != "name":  # exclusions apply: give it its own list
                scoped = _rebind(key, narrowed_key)
                rewritten = rewritten.replace("{" + key + "}", "{" + scoped + "}")
                key = scoped
            list_values[key] = _norm_values(ov.spoken_values("entities", kept))
        if dropped:
            continue
        for slot, kind in (("area", "areas"), ("floor", "floors")):
            token = "{" + slot + "}"
            if token not in rewritten:
                continue
            scoped_key, kept = ov.narrow(combo_key, slot, (canonical_lists or {}).get(slot, []))
            if scoped_key == slot:
                continue  # nothing excluded: the shared list already covers it
            if not kept:
                dropped = True
                break
            rewritten = rewritten.replace(token, "{" + scoped_key + "}")
            list_values[scoped_key] = _norm_values(ov.spoken_values(kind, kept))
        if dropped:
            continue
        templates.append(rewritten)


def _rebind(key: str, scoped: str) -> str:
    """Command-scoped variant of a domain-scoped ``{name}`` list key: keep the
    domain/capability scope (``name__light__brightness``) and append the command
    suffix that ``Overrides.narrow`` produced (``...__x_hassturnon_name_only``)."""
    return key + scoped[len("name"):]


def assemble(
    s2p_repo: Path,
    lang: str,
    enabled: Sequence[Tuple[str, str]],
    custom_commands: Sequence[dict],
    entities: Dict[str, str],
    slot_lists: Dict[str, List[str]],
    extra_sentences: Optional[Dict[str, List[str]]] = None,
    ov=None,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Build (templates, list_values) for the enabled built-ins + custom commands.

    `ov` (overrides.Overrides) supplies the user's aliases and per-command target
    exclusions; the default changes nothing."""
    import custom_commands as cc
    import gating
    import overrides as ovr
    import s2p_intents

    ov = ov or ovr.EMPTY
    info = as_entity_info(entities)
    extras = extra_sentences or {}
    templates: List[str] = []
    # Slot values are normalized to match the lowercase acoustic vocab. The
    # shared area/floor lists carry their aliases (the matcher maps them back).
    list_values: Dict[str, List[str]] = {k: _norm_values(v) for k, v in slot_lists.items()}
    for slot, kind in (("area", "areas"), ("floor", "floors")):
        if slot in slot_lists:
            list_values[slot] = _norm_values(ov.spoken_values(kind, slot_lists[slot]))
    # Text lists (color, on/off states, ...) come from the package; name/area/
    # floor stay from the registry-provided slot_lists above.
    for name, values in s2p_intents.text_list_values(lang).items():
        list_values[name] = _norm_values(values)

    for (intent, combo), allowed in enabled_domain_map(enabled).items():
        si_blocks = s2p_intents.combo_blocks(lang, intent, combo)
        if not si_blocks:
            continue
        capability = gating.required_capability(intent, combo)
        for ss in combo_blocks({"data": si_blocks}, extras.get(f"{intent}/{combo}")):
            include, eff_nd = _effective_name_domains(ss, allowed)
            if not include:
                continue
            inferred = ss.get("inferred_domain")
            # Capability/domain gate for combos with no {name}/{area} slot to
            # narrow (e.g. context-area combos).
            if not gating.keep_block(
                eff_nd, inferred, capability, gating.capability_domains(intent), info
            ):
                continue
            # Package templates are hassil dialect -> expand into the trainer's
            # flat dialect. User extra sentences are already trainer-dialect, so
            # if expansion fails, fall back to using them verbatim.
            try:
                flat_templates, _ref = s2p_intents.grammar_templates(
                    ss.get("sentences", []), lang
                )
            except Exception:  # noqa: BLE001
                flat_templates = list(ss.get("sentences", []))
            _expand_block(
                flat_templates, eff_nd, capability,
                info, templates, list_values,
                ov=ov, combo_key=ovr.combo_key(intent, combo),
                canonical_lists=slot_lists,
            )

    # Custom commands (all modes contribute their sentences to the grammar).
    for block in cc.grammar_sentences(list(custom_commands or [])):
        _expand_block(
            block["sentences"], block.get("name_domains") or None, None,
            info, templates, list_values, ov=ov, canonical_lists=slot_lists,
        )

    templates = list(dict.fromkeys(templates))
    # Keep only the lists actually referenced, so the grammar (and its retrain
    # fingerprint) depends only on values that affect it -- e.g. changing areas
    # only matters if some enabled sentence uses {area}.
    referenced: Set[str] = set()
    for t in templates:
        referenced.update(re.findall(r"\{([^}]+)\}", t))
    list_values = {k: v for k, v in list_values.items() if k in referenced}
    return templates, list_values


# --- grammar size (UI cost indicator) ----------------------------------------
#
# The reason to disable a command is grammar size: every active command widens
# the FST search space and costs recognition accuracy. "Phrases" is the number of
# distinct utterances a set of templates can produce -- templates multiplied out
# by the size of each list they reference -- which tracks that cost far better
# than a template count ("turn on {name}" is one template but 40 phrases).

_RANGE_REF_RE = re.compile(r"^(-?\d+)\s*\.\.\s*(-?\d+)(?:\s*[,/]\s*(-?\d+))?$")


def phrase_count(templates: Sequence[str], list_values: Dict[str, List[str]]) -> int:
    """Distinct utterances `templates` can produce with `list_values` bound."""
    total = 0
    for template in templates:
        n = 1
        for ref in re.findall(r"\{([^}]+)\}", template):
            ref = ref.strip()
            m = _RANGE_REF_RE.match(ref)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                step = abs(int(m.group(3) or 1)) or 1
                n *= max(1, (abs(hi - lo) // step) + 1)
            else:
                n *= max(1, len(list_values.get(ref.split(":", 1)[0], [])))
        total += n
    return total


def combo_cost(
    s2p_repo: Path,
    lang: str,
    intent: str,
    combo: str,
    domain: Optional[str],
    entities,
    slot_lists: Dict[str, List[str]],
    extra_sentences: Optional[Dict[str, List[str]]] = None,
    ov=None,
) -> Dict[str, int]:
    """Grammar cost of one combo (optionally narrowed to a single domain), as
    ``{"sentences": n_templates, "phrases": n_utterances}``."""
    entry = [intent, combo, [domain]] if domain else [intent, combo]
    templates, list_values = assemble(
        s2p_repo, lang, [entry], [], entities, slot_lists,
        extra_sentences=extra_sentences, ov=ov,
    )
    return {
        "sentences": len(templates),
        "phrases": phrase_count(templates, list_values),
    }


def train(
    backend: str,
    model_dir: Path,
    lang: str,
    templates: List[str],
    list_values: Dict[str, List[str]],
    out_path: Path,
    beam: float = None,
) -> int:
    """Compile and save the grammar. Returns the number of templates."""
    from speech_to_phrase import load_recognizer

    rec = load_recognizer(backend, model_dir, language=lang, beam=beam)
    rec.train(templates, list_values=list_values)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec.save(out_path)
    return len(templates)
