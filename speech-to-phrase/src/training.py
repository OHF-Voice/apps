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

# Local-dev fallback registry (name -> domain) and slot lists. The container
# replaces these with the live HA registry + home-assistant-intents lists.
DEV_ENTITIES: Dict[str, str] = {
    "overhead light": "light",
    "kitchen lamp": "light",
    "kitchen fan": "fan",
    "garage door": "cover",
    "front door": "lock",
}
DEV_SLOT_LISTS: Dict[str, List[str]] = {
    "area": ["kitchen", "office", "living room"],
    "floor": ["first floor", "second floor"],
    "color": ["red", "green", "blue", "white"],
    "brightness_level": ["maximum", "minimum"],
    # Language constants (English). Used by both the grammar and the matcher.
    # Per-domain state lists (kept separate so domain-scoped {name} can't accept
    # cross-domain nonsense like "is the lock on"). All map to the `state` slot
    # via intent_matcher.canonical_slot().
    "on_off_state": ["on", "off"],
    "cover_state": ["open", "closed"],
    "lock_state": ["locked", "unlocked"],
    "volume_step": ["up", "down"],
}


def _ws_url(api_url: str) -> str:
    """Derive the HA websocket URL from the REST api URL. Direct HA:
    .../api -> .../api/websocket. Supervisor: /core/api -> /core/websocket."""
    from urllib.parse import urlparse, urlunparse

    p = urlparse(api_url)
    scheme = "wss" if p.scheme == "https" else "ws"
    base = p.path[:-4] if p.path.endswith("/api") else p.path
    ws_path = (base + "/websocket") if base else "/api/websocket"
    return urlunparse((scheme, p.netloc, ws_path, "", "", ""))


async def _exposed_conversation_ids(api_url: str, token: str) -> Set[str]:
    """entity_ids exposed to the `conversation` assistant, via the websocket
    command homeassistant/expose_entity/list."""
    import aiohttp

    ids: Set[str] = set()
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(_ws_url(api_url), max_msg_size=0) as ws:
            assert (await ws.receive_json())["type"] == "auth_required"
            await ws.send_json({"type": "auth", "access_token": token})
            assert (await ws.receive_json())["type"] == "auth_ok"
            await ws.send_json({"id": 1, "type": "homeassistant/expose_entity/list"})
            msg = await ws.receive_json()
            assert msg.get("success"), msg
            for eid, info in msg["result"]["exposed_entities"].items():
                if info.get("conversation"):
                    ids.add(eid)
    return ids


def exposed_conversation_ids(api_url: str, token: str) -> Set[str]:
    import asyncio

    return asyncio.run(_exposed_conversation_ids(api_url, token))


async def _exposed_entity_names(
    api_url: str, token: str, friendly: Dict[str, str]
) -> Dict[str, str]:
    """{name: domain} for conversation-exposed entities, INCLUDING aliases.

    Names come from the entity registry (name override / original_name / aliases,
    like hass_api.py), falling back to the live friendly_name. Disabled entities
    are skipped. Each alias maps to the entity's domain so it's recognizable."""
    import aiohttp

    out: Dict[str, str] = {}
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(_ws_url(api_url), max_msg_size=0) as ws:
            assert (await ws.receive_json())["type"] == "auth_required"
            await ws.send_json({"type": "auth", "access_token": token})
            assert (await ws.receive_json())["type"] == "auth_ok"
            await ws.send_json({"id": 1, "type": "homeassistant/expose_entity/list"})
            em = await ws.receive_json()
            assert em.get("success"), em
            exposed = [
                eid
                for eid, info in em["result"]["exposed_entities"].items()
                if info.get("conversation")
            ]
            if not exposed:
                return out
            await ws.send_json({
                "id": 2,
                "type": "config/entity_registry/get_entries",
                "entity_ids": exposed,
            })
            rm = await ws.receive_json()
            entries = rm["result"] if rm.get("success") else {}

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
        if not names and friendly.get(eid):
            names.append(friendly[eid])
        for name in names:
            name = name.strip()
            if name:
                out[name] = domain
    return out


def exposed_entity_names(api_url: str, token: str, friendly: Dict[str, str]) -> Dict[str, str]:
    import asyncio

    return asyncio.run(_exposed_entity_names(api_url, token, friendly))


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


def entities_from_hass(api_url: str, token: str) -> Dict[str, str]:
    """Build {name: domain} for entities **exposed to the conversation
    integration**, including their aliases. Used in the container (api_url=
    http://supervisor/core/api, token=SUPERVISOR_TOKEN)."""
    import json
    import urllib.request

    req = urllib.request.Request(
        f"{api_url}/states", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        states = json.loads(resp.read())
    friendly = {
        s["entity_id"]: (s.get("attributes") or {}).get("friendly_name", "")
        for s in states
        if s.get("entity_id")
    }

    try:
        out = exposed_entity_names(api_url, token, friendly)
        _LOGGER.info("%d exposed entity names (including aliases)", len(out))
        return out
    except Exception:  # noqa: BLE001
        _LOGGER.warning(
            "Could not fetch exposed entities/aliases; using ALL friendly names",
            exc_info=True,
        )

    out = {}
    for eid, name in friendly.items():
        domain = eid.split(".", 1)[0] if "." in eid else ""
        if domain and (name or eid):
            out[name or eid] = domain
    return out


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
    f = s2p_repo / "sentences" / lang / intent / f"{combo}.yaml"
    if not f.exists():
        return []
    out: List[str] = []
    for block in (yaml.safe_load(f.read_text()) or {}).get("data", []):
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


def _expand_block(
    sentences: Sequence[str],
    domains: Optional[Sequence[str]],
    entities: Dict[str, str],
    templates: List[str],
    list_values: Dict[str, List[str]],
) -> None:
    """Append a block's sentences to `templates`, rewriting `{name}` to a
    domain-scoped list and applying entity gating. A bare `{name}` with no
    `name_domains` binds to every entity (custom commands own the meaning)."""
    for sentence in sentences:
        if "{name}" in sentence:
            if domains:
                values = _norm_values([n for n, d in entities.items() if d in domains])
                if not values:
                    continue  # entity gating: no such device -> drop sentence
                key = _name_list_key(domains)
                list_values[key] = values
                sentence = sentence.replace("{name}", "{" + key + "}")
            elif entities:
                list_values.setdefault("name", _norm_values(list(entities.keys())))
            else:
                continue  # {name} but nothing to fill it with
        templates.append(sentence)


def assemble(
    s2p_repo: Path,
    lang: str,
    enabled: Sequence[Tuple[str, str]],
    custom_commands: Sequence[dict],
    entities: Dict[str, str],
    slot_lists: Dict[str, List[str]],
    extra_sentences: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Build (templates, list_values) for the enabled built-ins + custom commands."""
    import custom_commands as cc

    extras = extra_sentences or {}
    templates: List[str] = []
    # Slot values are normalized to match the lowercase acoustic vocab.
    list_values: Dict[str, List[str]] = {k: _norm_values(v) for k, v in slot_lists.items()}
    lang_dir = s2p_repo / "sentences" / lang

    for (intent, combo), allowed in enabled_domain_map(enabled).items():
        f = lang_dir / intent / f"{combo}.yaml"
        if not f.exists():
            continue
        doc = yaml.safe_load(f.read_text()) or {}
        for ss in combo_blocks(doc, extras.get(f"{intent}/{combo}")):
            include, eff_nd = _effective_name_domains(ss, allowed)
            if not include:
                continue
            _expand_block(
                ss.get("sentences", []), eff_nd,
                entities, templates, list_values,
            )

    # Custom commands (all modes contribute their sentences to the grammar).
    for block in cc.grammar_sentences(list(custom_commands or [])):
        _expand_block(
            block["sentences"], block.get("name_domains") or None,
            entities, templates, list_values,
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
