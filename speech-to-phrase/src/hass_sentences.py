"""Sentences Home Assistant already knows it wants to hear.

Two sources, both configured in Home Assistant rather than in this add-on, and
both useless unless the recognizer can actually produce them:

  * **sentence triggers** -- the ``conversation: sentences:`` trigger of an
    automation. HA only fires the trigger if the transcript matches it, so a
    trigger phrase that is not in the grammar can never fire.
  * **question answers** -- the ``answers:`` sentences of an
    ``assist_satellite.ask_question`` action. The satellite asks, the user
    replies, and the reply has to be recognizable for the automation to branch
    on it.

Both are fetched over the websocket API and added to the grammar as plain
sentences (see ``training.assemble``). Each source is gated by an add-on option
(``sentence_triggers`` / ``question_answers``), because every added phrase
widens the FST search space -- and the answer crawl costs one websocket
round-trip per automation/script, so a large installation may prefer to pay for
neither.

Results are cached briefly: the training path re-reads them on every registry
refresh and on every save, the web UI reads them again to show what they cost,
and none of this changes minute to minute. They are kept grouped by source
(:func:`fetch_grouped`) so the UI can say which switch a phrase came from
without a second round-trip.
"""
import itertools
import logging
import re
import time
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple

from training import _ws_url

_LOGGER = logging.getLogger("speech-to-phrase.hass_sentences")

# HA-template sentences ("{{ states(...) }}") have no fixed spoken form.
_TEMPLATE_MARKER = "{{"

_CACHE_TTL = 60.0
_cache: Dict[tuple, Tuple[float, Dict[str, List[str]]]] = {}

# The two sources, which are also the two option/settings names gating them.
TRIGGERS = "sentence_triggers"
ANSWERS = "question_answers"
SOURCES = (TRIGGERS, ANSWERS)


def _clean(sentences) -> Generator[str, None, None]:
    """Yield usable sentences from an ``answers``/``trigger_sentences`` value,
    which may be a single string or a list of them."""
    if isinstance(sentences, str):
        sentences = [sentences]
    for sentence in sentences or []:
        if not isinstance(sentence, str):
            continue
        sentence = sentence.strip()
        if sentence and _TEMPLATE_MARKER not in sentence:
            yield sentence


def _ask_question_answers(item: Any) -> Generator[str, None, None]:
    """Walk an automation/script config and yield the answer sentences of every
    ``assist_satellite.ask_question`` action in it (at any nesting depth: they
    are usually inside a ``choose``/``if`` branch)."""
    if isinstance(item, dict):
        if item.get("action") == "assist_satellite.ask_question":
            for answer in (item.get("data") or {}).get("answers") or []:
                if isinstance(answer, dict):
                    yield from _clean(answer.get("sentences"))
        else:
            for sub_item in item.values():
                yield from _ask_question_answers(sub_item)
    elif isinstance(item, list):
        for sub_item in item:
            yield from _ask_question_answers(sub_item)


async def _fetch_async(
    api_url: str, token: str, triggers: bool, answers: bool
) -> Dict[str, List[str]]:
    import aiohttp

    found: Dict[str, List[str]] = {TRIGGERS: [], ANSWERS: []}
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(_ws_url(api_url), max_msg_size=0) as ws:
            assert (await ws.receive_json())["type"] == "auth_required"
            await ws.send_json({"type": "auth", "access_token": token})
            assert (await ws.receive_json())["type"] == "auth_ok"

            ids = itertools.count(1)

            async def call(type_: str, **kw):
                await ws.send_json({"id": next(ids), "type": type_, **kw})
                msg = await ws.receive_json()
                if msg.get("success"):
                    return msg.get("result")
                _LOGGER.debug("%s failed: %s", type_, msg.get("error"))
                return None

            if triggers:
                result = await call("conversation/sentences/list")
                found[TRIGGERS] = list(_clean((result or {}).get("trigger_sentences")))
                _LOGGER.debug(
                    "Fetched %d sentence trigger phrase(s)", len(found[TRIGGERS])
                )

            if answers:
                out: List[str] = []
                for state in (await call("get_states")) or []:
                    entity_id = state.get("entity_id") or ""
                    domain = entity_id.split(".", 1)[0]
                    if domain not in ("automation", "script"):
                        continue
                    # A disabled automation cannot ask its question, so its
                    # answers are dead weight in the grammar. Scripts have no
                    # equivalent "off" state -- they are just not running.
                    if domain == "automation" and state.get("state") != "on":
                        continue
                    result = await call(f"{domain}/config", entity_id=entity_id)
                    if result:
                        out.extend(_ask_question_answers(result.get("config") or {}))
                found[ANSWERS] = out
                _LOGGER.debug("Fetched %d question answer sentence(s)", len(out))

    # De-duped per source, and across them: a phrase that is both a trigger and
    # an answer is one path in the grammar, and is attributed to the trigger.
    seen: Set[str] = set()
    for source in SOURCES:
        kept = [s for s in found[source] if not (s in seen or seen.add(s))]
        found[source] = kept
    return found


def fetch_grouped(
    api_url: str,
    token: Optional[str],
    triggers: bool = True,
    answers: bool = True,
    ttl: float = _CACHE_TTL,
) -> Dict[str, List[str]]:
    """``{"sentence_triggers": [...], "question_answers": [...]}`` from HA, with
    a source the caller switched off left empty.

    Best-effort: returns empty lists (rather than raising) if HA is unreachable
    or the websocket commands are unsupported, so a fetch failure degrades the
    grammar instead of blocking a retrain. A stale cached result is preferred to
    nothing, so one failed poll does not shrink the grammar.
    """
    import asyncio

    empty: Dict[str, List[str]] = {source: [] for source in SOURCES}
    if not token or not (triggers or answers):
        return empty
    key = (api_url, triggers, answers)
    cached = _cache.get(key)
    now = time.monotonic()
    if cached and (now - cached[0]) < ttl:
        return {k: list(v) for k, v in cached[1].items()}
    try:
        found = asyncio.run(_fetch_async(api_url, token, triggers, answers))
    except Exception:  # noqa: BLE001
        _LOGGER.exception("could not fetch Home Assistant sentences")
        return {k: list(v) for k, v in cached[1].items()} if cached else empty
    _cache[key] = (now, found)
    return {k: list(v) for k, v in found.items()}


def fetch(
    api_url: str,
    token: Optional[str],
    triggers: bool = True,
    answers: bool = True,
    ttl: float = _CACHE_TTL,
) -> List[str]:
    """Every sentence :func:`fetch_grouped` finds, flattened (for training)."""
    grouped = fetch_grouped(api_url, token, triggers, answers, ttl)
    return [s for source in SOURCES for s in grouped[source]]


# --- grammar conversion ------------------------------------------------------

_REF_RE = re.compile(r"\{([^{}]+)\}")
_RANGE_RE = re.compile(r"^-?\d+\s*\.\.\s*-?\d+")


def grammar_templates(
    sentences: Sequence[str], lang: str, bindable: Set[str]
) -> List[str]:
    """Flat, trainer-dialect templates for HA-authored sentences.

    These are written in hassil's dialect (``[optional]``, ``(a|b)``,
    ``(a;b)`` permutations), which the FST trainer does not parse, so they go
    through the same expansion as the packaged templates.

    A ``{ref}`` that survives expansion is dropped along with its sentence:
    HA sentence triggers and question answers declare no slot lists of their
    own, so any reference left over is a wildcard or a typo -- either way there
    is nothing to bind it to, and an unbound reference is not trainable.
    ``bindable`` names the lists that *do* exist (plus inline ranges), so a
    sentence that happens to use one is kept.
    """
    import s2p_intents

    out: List[str] = []
    for sentence in sentences:
        try:
            expanded, _refs = s2p_intents.grammar_templates([sentence], lang)
        except Exception:  # noqa: BLE001 -- a malformed sentence must not break training
            _LOGGER.warning("Could not parse Home Assistant sentence %r", sentence)
            continue
        kept = [
            template
            for template in expanded
            if all(
                _RANGE_RE.match(ref.strip())
                or ref.split(":", 1)[0].strip() in bindable
                for ref in _REF_RE.findall(template)
            )
        ]
        if expanded and not kept:
            _LOGGER.warning(
                "Skipping Home Assistant sentence %r: it references a list that "
                "Speech-to-Phrase cannot fill in", sentence,
            )
        out.extend(kept)
    return list(dict.fromkeys(out))
