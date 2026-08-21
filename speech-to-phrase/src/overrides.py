"""What voice commands target, and what those targets are called.

Persisted per language in ``<data>/<lang>/targets.json``. Two things live here:

  * **Voice targeting.** An entity/area/floor can be switched off for voice
    entirely (global -- it disappears from every command), or excluded from a
    single command (per-command opt-out). Global off is applied where the entity
    records and area/floor lists are read, so gating, examples and the UI all
    agree; per-command exclusions are applied while assembling the grammar and
    the matcher, by binding that command's slot to a narrowed, command-scoped
    list.

  * **Aliases.** Extra spoken names for an entity/area/floor, either *added* to
    the Home Assistant name or *replacing* it. Replacing is what shrinks a
    grammar: an entity called "Kitchen Ceiling Light Bulb 2" costs a lot of
    phrases and mis-recognizes; aliasing it to "kitchen light" costs one.

Aliases never reach Home Assistant. HA resolves the ``name``/``area``/``floor``
slot against its own registry, so a spoken alias has to be mapped back to the
canonical name before the intent is handed over -- an alias HA has never heard
of would recognize perfectly and then fail to execute. The grammar is therefore
trained on the *spoken* forms, while the matcher binds them as hassil in/out
value pairs so a match still emits the canonical name. The consequence to know:
these aliases only work through Speech-to-Phrase, not through Home Assistant's
own conversation agent or a cloud fallback.
"""
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Slot name -> the section of the document that governs it.
SLOT_KINDS = {"name": "entities", "area": "areas", "floor": "floors"}
KINDS = ("entities", "areas", "floors")

FILENAME = "targets.json"


def combo_key(intent: str, combo: str) -> str:
    return f"{intent}/{combo}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


class Overrides:
    """Read-only view of a ``targets.json`` document.

    Absent entries mean "default": on for voice, no aliases, nothing excluded --
    so an empty document changes nothing.
    """

    def __init__(self, doc: Optional[dict] = None):
        doc = doc or {}
        self._by_kind: Dict[str, Dict[str, dict]] = {
            kind: {k: v for k, v in (doc.get(kind) or {}).items() if isinstance(v, dict)}
            for kind in KINDS
        }
        self._exclude: Dict[str, Dict[str, Set[str]]] = {}
        for key, spec in (doc.get("exclude") or {}).items():
            if isinstance(spec, dict):
                self._exclude[key] = {
                    slot: set(spec.get(slot) or []) for slot in SLOT_KINDS
                }

    # ---- global targeting ---------------------------------------------------

    def voice_on(self, kind: str, value: str) -> bool:
        return bool(self._by_kind.get(kind, {}).get(value, {}).get("voice", True))

    def filter_records(self, records: Sequence[dict]) -> List[dict]:
        """Drop entities switched off for voice (records are plain dicts here)."""
        return [r for r in records if self.voice_on("entities", r.get("name", ""))]

    def filter_slot_lists(self, slot_lists: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Drop areas/floors switched off for voice."""
        out = {k: list(v) for k, v in slot_lists.items()}
        for slot, kind in (("area", "areas"), ("floor", "floors")):
            if slot in out:
                out[slot] = [v for v in out[slot] if self.voice_on(kind, v)]
        return out

    # ---- aliases ------------------------------------------------------------

    def spoken(self, kind: str, value: str) -> List[str]:
        """Every way to say `value`: its aliases, plus the Home Assistant name
        unless the user chose to replace it. Replacing with no alias would make
        the thing unreachable, so the HA name survives that case."""
        spec = self._by_kind.get(kind, {}).get(value) or {}
        aliases = [a.strip() for a in (spec.get("aliases") or []) if a and a.strip()]
        if aliases and spec.get("replace"):
            return list(dict.fromkeys(aliases))
        return list(dict.fromkeys([value] + aliases))

    def pairs(self, kind: str, values: Iterable[str]) -> List[Tuple[str, str]]:
        """``[(spoken, canonical)]`` for `values` -- what the matcher binds, so a
        match on an alias still yields the name Home Assistant knows."""
        out: List[Tuple[str, str]] = []
        for value in values:
            for form in self.spoken(kind, value):
                out.append((form, value))
        return out

    def spoken_values(self, kind: str, values: Iterable[str]) -> List[str]:
        """Just the spoken forms -- what the grammar is trained on."""
        return [form for form, _canonical in self.pairs(kind, values)]

    # ---- per-command exclusions --------------------------------------------

    def excluded(self, key: str, slot: str) -> Set[str]:
        return self._exclude.get(key, {}).get(slot, set())

    def narrow(self, key: str, slot: str, values: Sequence[str]) -> Tuple[str, List[str]]:
        """``(list_key, values)`` for `slot` under command `key`.

        With nothing excluded the shared list is reused as-is. With exclusions the
        command gets its own list -- narrowing the shared one would silently strip
        the entity from every other command that uses it.
        """
        drop = self.excluded(key, slot)
        if not drop:
            return slot, list(values)
        return f"{slot}__x_{_slug(key)}", [v for v in values if v not in drop]


EMPTY = Overrides()


def load(data_dir: Path, lang: str) -> Overrides:
    path = data_dir / lang / FILENAME
    if not path.exists():
        return EMPTY
    try:
        return Overrides(json.loads(path.read_text()))
    except Exception:  # noqa: BLE001 (a corrupt file must not break training)
        return EMPTY


def load_doc(data_dir: Path, lang: str) -> dict:
    """The raw document, for round-tripping through the web UI."""
    path = data_dir / lang / FILENAME
    if not path.exists():
        return {kind: {} for kind in KINDS} | {"exclude": {}}
    try:
        doc = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        doc = {}
    for kind in KINDS:
        doc.setdefault(kind, {})
    doc.setdefault("exclude", {})
    return doc


def save(data_dir: Path, lang: str, doc: dict) -> None:
    """Persist, dropping entries that say nothing (all defaults), so the file
    stays readable and an untouched install has no file at all."""
    clean: dict = {}
    for kind in KINDS:
        section = {}
        for value, spec in (doc.get(kind) or {}).items():
            if not isinstance(spec, dict):
                continue
            aliases = [a.strip() for a in (spec.get("aliases") or []) if a and a.strip()]
            voice = bool(spec.get("voice", True))
            replace = bool(spec.get("replace")) and bool(aliases)
            if voice and not aliases:
                continue  # default
            entry: dict = {}
            if not voice:
                entry["voice"] = False
            if aliases:
                entry["aliases"] = aliases
            if replace:
                entry["replace"] = True
            section[value] = entry
        if section:
            clean[kind] = section
    exclude = {}
    for key, spec in (doc.get("exclude") or {}).items():
        narrowed = {
            slot: sorted(set(spec.get(slot) or []))
            for slot in SLOT_KINDS
            if spec.get(slot)
        }
        if narrowed:
            exclude[key] = narrowed
    if exclude:
        clean["exclude"] = exclude

    path = data_dir / lang / FILENAME
    if not clean:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2, sort_keys=True))
