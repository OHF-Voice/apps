"""Intent matcher for Speech-to-Phrase.

The constrained STT grammar can only ever emit one of the curated sentences, so
recognizing the *intent* is just a second, deterministic pass: feed the
transcript back through hassil against the **same** curated sentences and read
off the intent + slots. No fuzzy matching, no second model.

PARITY WITH THE STT GRAMMAR (the important bit)
-----------------------------------------------
We build the hassil ``Intents`` from the exact same ``sentences/<lang>/<Intent>/
<combo>.yaml`` files that ``training.assemble`` compiles into the FST, using the
identical domain-scoped ``{name}`` trick (``{name__light_switch_...}``) and the
identical entity gating (drop a sentence whose ``name_domains`` match no exposed
entity). That guarantees every transcript the STT can produce is matchable here.

Each curated data-block may carry extra fields the grammar ignores but the
matcher uses, surfaced after a match via ``RecognizeResult.intent_metadata``:
  * ``inferred_domain``  -> emitted as the ``domain`` slot (e.g. "lights" -> light)
  * ``context_area: true`` -> inject the voice satellite's area as the ``area`` slot
  * ``response``         -> response-template key (selection TODO; see intent_server)
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
from hassil import Intents, RecognizeResult, TextSlotList, recognize_best

from training import (
    _effective_name_domains,
    _name_list_key,
    combo_blocks,
    enabled_domain_map,
)

_LOGGER = logging.getLogger("speech-to-phrase.intent")

# Synthetic intent name for custom action-mode commands (no HA intent of their
# own). The intent server routes this to the action executor, not to HA.
CUSTOM_ACTION_INTENT = "_CustomAction"

# Internal list names that map onto a single HA intent slot. Per-domain state
# lists keep "is the lock on" out of the grammar but all feed the `state` slot.
_STATE_LISTS = frozenset({"on_off_state", "cover_state", "lock_state"})


def canonical_slot(key: str) -> str:
    """Map an internal slot-list name to the HA intent slot name it fills."""
    if key.startswith("name__"):
        return "name"
    if key in _STATE_LISTS:
        return "state"
    return key


class IntentMatcher:
    """Holds a compiled hassil ``Intents`` + slot lists and matches text."""

    def __init__(self, intents: Intents, slot_lists: Dict[str, TextSlotList],
                 language: str):
        self._intents = intents
        self._slot_lists = slot_lists
        self.language = language

    def match(self, text: str) -> Optional[RecognizeResult]:
        return recognize_best(
            text,
            self._intents,
            slot_lists=self._slot_lists,
            language=self.language,
        )


def build_matcher(
    s2p_repo: Path,
    lang: str,
    enabled: Sequence[Tuple[str, str]],
    entities: Dict[str, str],
    slot_lists: Dict[str, List[str]],
    custom_commands: Optional[Sequence[dict]] = None,
    extra_sentences: Optional[Dict[str, List[str]]] = None,
) -> Optional[IntentMatcher]:
    """Build an :class:`IntentMatcher` for the enabled combos + custom commands,
    or ``None`` if nothing is matchable."""
    extras = extra_sentences or {}
    lang_dir = s2p_repo / "sentences" / lang
    intents_dict: Dict[str, dict] = {}
    name_lists: Dict[str, List[str]] = {}

    def scope(sentences: Sequence[str], domains) -> List[str]:
        """Rewrite `{name}` -> a domain-scoped list (original-case names, so the
        emitted slot value is HA-friendly), dropping a sentence whose domains
        match no entity -- the same gating the grammar applies."""
        out: List[str] = []
        for sentence in sentences:
            if "{name}" in sentence:
                if domains:
                    names = [n for n, d in entities.items() if d in domains]
                    key = _name_list_key(domains)
                elif entities:
                    names = list(entities.keys())
                    key = "name"
                else:
                    continue
                if not names:
                    continue
                name_lists.setdefault(key, names)
                sentence = sentence.replace("{name}", "{" + key + "}")
            out.append(sentence)
        return out

    for (intent, combo), allowed in enabled_domain_map(enabled).items():
        f = lang_dir / intent / f"{combo}.yaml"
        if not f.exists():
            continue
        doc = yaml.safe_load(f.read_text()) or {}
        data_blocks: List[dict] = []
        for ss in combo_blocks(doc, extras.get(f"{intent}/{combo}")):
            include, eff_nd = _effective_name_domains(ss, allowed)
            if not include:
                continue
            sentences = scope(ss.get("sentences", []), eff_nd)
            if not sentences:
                continue
            metadata: Dict[str, object] = {
                "combo": combo,
                "response_key": ss.get("response", "default"),
            }
            if ss.get("inferred_domain"):
                metadata["domain"] = ss["inferred_domain"]
            if ss.get("context_area"):
                metadata["context_area"] = True
            if ss.get("slots"):  # fixed slot values (e.g. timer "half" -> 30)
                metadata["slots"] = ss["slots"]
            data_blocks.append({"sentences": sentences, "metadata": metadata})

        if data_blocks:
            intents_dict.setdefault(intent, {"data": []})["data"].extend(data_blocks)

    # Custom commands: intent-mode under the chosen HA intent, action-mode under
    # a synthetic bucket. STT-only commands are grammar-only (skipped here).
    for idx, cmd in enumerate(custom_commands or []):
        mode = cmd.get("mode", "stt")
        if mode not in ("intent", "action"):
            continue
        sentences = scope(cmd.get("sentences") or [], cmd.get("name_domains"))
        if not sentences:
            continue
        metadata = {"source": "custom", "mode": mode, "id": idx}
        if cmd.get("response"):
            metadata["response"] = cmd["response"]
        if mode == "intent":
            spec = cmd.get("intent") or {}
            intent_name = spec.get("name")
            if not intent_name:
                continue
            if spec.get("slots"):
                metadata["slots"] = spec["slots"]
        else:  # action
            metadata["action"] = cmd.get("action") or {}
            intent_name = CUSTOM_ACTION_INTENT
        intents_dict.setdefault(intent_name, {"data": []})["data"].append(
            {"sentences": sentences, "metadata": metadata}
        )

    if not intents_dict:
        return None

    hassil_slot_lists: Dict[str, TextSlotList] = {}
    for key, names in name_lists.items():
        hassil_slot_lists[key] = TextSlotList.from_strings(
            sorted(set(names)), name=key
        )
    # Text slot lists from the language/registry (area, floor, color, state, ...).
    # Numeric slots use inline ranges ({0..100:brightness}) -- hassil expands
    # those itself, so they need no list here. Harmless if unreferenced.
    for key, values in (slot_lists or {}).items():
        if values and key not in hassil_slot_lists:
            hassil_slot_lists[key] = TextSlotList.from_strings(
                sorted(set(values)), name=key
            )

    intents = Intents.from_dict({"language": lang, "intents": intents_dict})
    n_sentences = sum(
        len(b["sentences"]) for v in intents_dict.values() for b in v["data"]
    )
    _LOGGER.info(
        "Built intent matcher for '%s': %d intents, %d sentence templates",
        lang, len(intents_dict), n_sentences,
    )
    return IntentMatcher(intents, hassil_slot_lists, lang)
