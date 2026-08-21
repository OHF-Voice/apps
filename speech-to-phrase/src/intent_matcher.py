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


def canonical_slot(key: str) -> str:
    """Map an internal slot-list name to the HA intent slot name it fills.

    ``{name}``/``{area}``/``{floor}`` are rewritten to domain-scoped internal
    lists (``name__<domains>`` / ``area__<domain>`` / ``floor__<domain>``) for
    gating; map them back. Every other slot (including per-domain ``state``
    lists, which bind via ``{...states:state}``) already carries its HA name.
    """
    for prefix, canonical in (("name__", "name"), ("area__", "area"), ("floor__", "floor")):
        if key.startswith(prefix):
            return canonical
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
    ov=None,
) -> Optional[IntentMatcher]:
    """Build an :class:`IntentMatcher` for the enabled combos + custom commands,
    or ``None`` if nothing is matchable."""
    import gating
    import overrides as ovr
    import s2p_intents
    import training

    ov = ov or ovr.EMPTY
    info = training.as_entity_info(entities)
    extras = extra_sentences or {}
    intents_dict: Dict[str, dict] = {}
    # Domain-scoped lists (name__*/area__*/floor__*) collected while scoping, as
    # (spoken, canonical) pairs: the user may have aliased a device, and what
    # comes back from a match has to be the name Home Assistant knows.
    scoped_lists: Dict[str, List[Tuple[str, str]]] = {}

    def scope(sentences, name_domains, capability, key: str = "") -> List[str]:
        """Rewrite {name} to a domain-scoped list and apply the same capability
        gate as the grammar (gating.scope_sentence), then the user's per-command
        exclusions and aliases -- exactly as training._expand_block does, so the
        matcher accepts precisely what the grammar can produce."""
        out: List[str] = []
        for sentence in sentences:
            rewritten, lists = gating.scope_sentence(
                sentence, name_domains, capability, info
            )
            if rewritten is None:
                continue
            dropped = False
            for list_key, values in lists.items():
                narrowed_key, kept = ov.narrow(key, "name", values)
                if not kept:
                    dropped = True
                    break
                if narrowed_key != "name":
                    scoped = training._rebind(list_key, narrowed_key)
                    rewritten = rewritten.replace(
                        "{" + list_key + "}", "{" + scoped + "}")
                    list_key = scoped
                scoped_lists.setdefault(list_key, ov.pairs("entities", kept))
            if dropped:
                continue
            for slot, kind in (("area", "areas"), ("floor", "floors")):
                token = "{" + slot + "}"
                if token not in rewritten:
                    continue
                narrowed_key, kept = ov.narrow(key, slot, (slot_lists or {}).get(slot, []))
                if narrowed_key == slot:
                    continue
                if not kept:
                    dropped = True
                    break
                rewritten = rewritten.replace(token, "{" + narrowed_key + "}")
                scoped_lists.setdefault(narrowed_key, ov.pairs(kind, kept))
            if dropped:
                continue
            out.append(rewritten)
        return out

    for (intent, combo), allowed in enabled_domain_map(enabled).items():
        si_blocks = s2p_intents.combo_blocks(lang, intent, combo)
        if not si_blocks:
            continue
        capability = gating.required_capability(intent, combo)
        data_blocks: List[dict] = []
        for ss in combo_blocks({"data": si_blocks}, extras.get(f"{intent}/{combo}")):
            include, eff_nd = _effective_name_domains(ss, allowed)
            if not include:
                continue
            inferred = ss.get("inferred_domain")
            if not gating.keep_block(
                eff_nd, inferred, capability, gating.capability_domains(intent), info
            ):
                continue
            sentences = scope(ss.get("sentences", []), eff_nd, capability,
                              ovr.combo_key(intent, combo))
            if not sentences:
                continue
            metadata: Dict[str, object] = {
                "combo": combo,
                "response_key": ss.get("response", "default"),
            }
            if inferred:
                metadata["domain"] = inferred
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
        sentences = scope(cmd.get("sentences") or [], cmd.get("name_domains"), None)
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
    for key, pairs in scoped_lists.items():
        # from_tuples binds (spoken, canonical): saying an alias yields the
        # Home Assistant name, which is the only thing HA can act on.
        hassil_slot_lists[key] = TextSlotList.from_tuples(
            sorted(set(pairs)), name=key
        )
    # Domain-scoped lists (above) plus the un-narrowed area/floor (for name-based
    # combos) are supplied at runtime; the other text lists (color, state, ...)
    # and numeric ranges come from the package's `lists`, and `<rules>` from its
    # `expansion_rules`, both handed to hassil below so it resolves the raw
    # templates natively.
    for key, kind in (("area", "areas"), ("floor", "floors")):
        values = (slot_lists or {}).get(key)
        if values and key not in hassil_slot_lists:
            hassil_slot_lists[key] = TextSlotList.from_tuples(
                sorted(set(ov.pairs(kind, values))), name=key
            )

    intents = Intents.from_dict(
        {
            "language": lang,
            "intents": intents_dict,
            "lists": s2p_intents.list_defs_dict(lang),
            "expansion_rules": s2p_intents.expansion_rules(lang),
        }
    )
    n_sentences = sum(
        len(b["sentences"]) for v in intents_dict.values() for b in v["data"]
    )
    _LOGGER.info(
        "Built intent matcher for '%s': %d intents, %d sentence templates",
        lang, len(intents_dict), n_sentences,
    )
    return IntentMatcher(intents, hassil_slot_lists, lang)
