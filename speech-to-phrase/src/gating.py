"""Entity-aware sentence gating.

Speech-to-Phrase should only put a command in the grammar if the user actually
has an entity it can act on. The library already drops ``{name}`` sentences
whose domain matches no entity; this module adds a finer **capability** gate,
driven by an enriched entity model (domain + device_class + supported features +
area/floor):

  * drop a combo/block when no exposed entity of its domain supports the required
    feature -- e.g. ``HassSetPosition`` for a cover with no ``SET_POSITION``,
    ``HassLightSet`` brightness for a light that can't dim, ``HassFanSetSpeed``
    for a fan with no speed control. ``{name}`` lists are narrowed to the capable
    entities.

The gate is **conservative**: when a signal is unknown (no feature data for an
entity) nothing is dropped, so an incomplete registry never silently removes
valid commands.

Note: ``{area}``/``{floor}`` are **not** narrowed by domain co-occurrence. A
sentence like "turn on the lights in the basement" stays in the grammar even
when the basement has no lights -- it must remain speakable so Home Assistant
can respond with a "no entities" error rather than the utterance being silently
unrecognisable.
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set


# Feature a combo needs, by intent (refined by combo name for HassLightSet).
# The capability tokens match those produced by capabilities_from_attributes().
def required_capability(intent: str, combo: str) -> Optional[str]:
    if intent == "HassSetPosition":
        return "set_position"
    if intent == "HassFanSetSpeed":
        return "set_speed"
    if intent == "HassLightSet":
        if "brightness" in combo:
            return "brightness"
        if "color" in combo:
            return "color"
        if "temperature" in combo:
            return "color_temp"
        return None
    if intent in ("HassMediaPlayerMute", "HassMediaPlayerUnmute"):
        return "volume_mute"
    if intent in ("HassSetVolume", "HassSetVolumeRelative"):
        return "volume_set"
    return None


# Domain(s) an intent's capability applies to. Used to gate combos that carry no
# {name}/{area} slot to narrow (e.g. context-area combos such as
# HassFanSetSpeed/default, which has no inferred_domain of its own).
_CAPABILITY_DOMAINS = {
    "HassSetPosition": ["cover", "valve"],
    "HassFanSetSpeed": ["fan"],
    "HassLightSet": ["light"],
    "HassMediaPlayerMute": ["media_player"],
    "HassMediaPlayerUnmute": ["media_player"],
    "HassSetVolume": ["media_player"],
    "HassSetVolumeRelative": ["media_player"],
}


def capability_domains(intent: str) -> List[str]:
    return list(_CAPABILITY_DOMAINS.get(intent, []))


# supported_features bits / color modes -> capability tokens, by domain.
_FEATURE_BITS = {
    "cover": {"set_position": 4},
    "valve": {"set_position": 4},
    "fan": {"set_speed": 1},
    "media_player": {"volume_set": 4, "volume_mute": 8},
}
_COLOR_MODES_BRIGHTNESS = {
    "brightness", "color_temp", "hs", "rgb", "rgbw", "rgbww", "xy", "white"
}
_COLOR_MODES_COLOR = {"hs", "rgb", "rgbw", "rgbww", "xy"}


def capabilities_from_attributes(domain: str, attributes: dict) -> Set[str]:
    """Capability tokens an entity supports, from its state attributes."""
    caps: Set[str] = set()
    sf = attributes.get("supported_features") or 0
    try:
        sf = int(sf)
    except (TypeError, ValueError):
        sf = 0
    for cap, bit in _FEATURE_BITS.get(domain, {}).items():
        if sf & bit:
            caps.add(cap)
    if domain == "light":
        modes = set(attributes.get("supported_color_modes") or [])
        if modes & _COLOR_MODES_BRIGHTNESS:
            caps.add("brightness")
        if modes & _COLOR_MODES_COLOR:
            caps.add("color")
        if "color_temp" in modes:
            caps.add("color_temp")
    return caps


@dataclass(frozen=True)
class EntityRecord:
    name: str
    domain: str
    device_class: Optional[str] = None
    # None => features unknown (treated permissively, never gated out).
    features: Optional[FrozenSet[str]] = None
    area: Optional[str] = None
    floor: Optional[str] = None


class EntityInfo:
    """Aggregates over the enriched entity records for the gating decisions."""

    def __init__(self, records: Sequence[EntityRecord]):
        self._records = list(records)

    @property
    def records(self) -> List[EntityRecord]:
        return self._records

    def _match(self, domains: Sequence[str], capability: Optional[str]):
        dom = set(domains)
        for r in self._records:
            if r.domain not in dom:
                continue
            # Unknown features (None) are permissive.
            if capability and r.features is not None and capability not in r.features:
                continue
            yield r

    def supports(self, domains: Sequence[str], capability: Optional[str]) -> bool:
        """True if >=1 entity in `domains` exists (and supports `capability`)."""
        return next(self._match(domains, capability), None) is not None

    def names(self, domains: Sequence[str], capability: Optional[str]) -> List[str]:
        """Original-case names in `domains` (supporting `capability`), de-duped."""
        return list(dict.fromkeys(r.name for r in self._match(domains, capability)))


def _scoped_key(
    kind: str, domains: Sequence[str], capability: Optional[str] = None
) -> str:
    # The capability is part of the key: two blocks over the same domains but
    # with different capability requirements (e.g. the ungated "is {cover} open"
    # state query vs the set_position-gated "open {cover} to 50%") must NOT share
    # a name list, or the narrower one would silently shrink the wider one (a
    # garage door with no set_position vanishing from state queries) -- or the
    # wider one would leak position-incapable covers into the position template.
    base = kind + "__" + "_".join(sorted(domains))
    return f"{base}__{capability}" if capability else base


def scope_sentence(
    sentence: str,
    name_domains: Optional[Sequence[str]],
    capability: Optional[str],
    info: "EntityInfo",
):
    """Rewrite ``{name}`` to a domain-scoped list ref and apply the capability
    gate. Returns ``(rewritten, lists)`` or ``(None, {})`` if the sentence must be
    dropped (no capable entity of the name's domain).

    ``lists`` maps scoped-list-name -> original-case values; the caller registers
    them (normalising as needed). ``{area}``/``{floor}`` are left untouched (bound
    to the full area/floor lists by the caller): area/floor sentences stay
    speakable even when no entity of the domain lives there, so Home Assistant can
    report the "no entities" error instead of the utterance going unrecognised.
    """
    lists: Dict[str, List[str]] = {}

    if "{name}" in sentence:
        if name_domains:
            names = info.names(name_domains, capability)
            if not names:
                return None, {}  # capability/domain gate: no such entity
            key = _scoped_key("name", name_domains, capability)
            lists[key] = names
            sentence = sentence.replace("{name}", "{" + key + "}")
        else:
            # Bare {name} (e.g. custom commands): bind to every entity.
            allnames = list(dict.fromkeys(r.name for r in info.records))
            if not allnames:
                return None, {}
            lists["name"] = allnames

    return sentence, lists


def keep_block(
    name_domains: Optional[Sequence[str]],
    inferred_domain: Optional[str],
    capability: Optional[str],
    cap_domains: Optional[Sequence[str]],
    info: "EntityInfo",
) -> bool:
    """Block-level gate: is there any entity this block could act on?

    Covers combos with no ``{name}``/``{area}`` slot (e.g. context-area combos
    like "set the brightness to 50%") where scope_sentence can't drop per-slot.
    ``cap_domains`` (from capability_domains) supplies the target domain when the
    block itself carries none. Conservative: a block with no domain context at
    all (bare custom command) is kept.
    """
    domains = list(name_domains or ([inferred_domain] if inferred_domain else []))
    if not domains:
        domains = list(cap_domains or [])
    if not domains:
        return True
    return info.supports(domains, capability)


def block_target_domains(block: dict) -> List[str]:
    """Domains a block acts on: its name_domains, else [inferred_domain]."""
    nd = block.get("name_domains")
    if nd:
        return list(nd)
    inf = block.get("inferred_domain")
    return [inf] if inf else []


def records_from_mapping(mapping: Dict[str, str]) -> List[EntityRecord]:
    """Build records from a plain {name: domain} dict (features/area unknown)."""
    return [EntityRecord(name=n, domain=d) for n, d in mapping.items()]


def records_from_dicts(items: Sequence[dict]) -> List[EntityRecord]:
    """Build records from a list of dicts (enriched fixture / registry form)."""
    out: List[EntityRecord] = []
    for it in items:
        feats = it.get("features")
        out.append(
            EntityRecord(
                name=it["name"],
                domain=it["domain"],
                device_class=it.get("device_class"),
                features=frozenset(feats) if feats is not None else None,
                area=it.get("area"),
                floor=it.get("floor"),
            )
        )
    return out
