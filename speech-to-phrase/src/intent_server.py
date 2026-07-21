#!/usr/bin/env python3
"""Wyoming intent-recognition server for Speech-to-Phrase.

Companion to the STT server: Home Assistant's Wyoming conversation agent sends a
``Recognize`` event with the transcript, and we reply with an ``Intent`` (name +
slots + a response template for HA to render) or ``NotRecognized`` (so HA can
fall back to another agent).

Two outcomes:
  * standard intent (built-in, or custom intent-mode): hand HA the intent + a
    jinja2 response template as ``Intent.text``; HA runs its handler and renders
    the template with live state (homeassistant/components/wyoming/conversation.py).
  * custom action: WE perform it (call a script/scene or a service from YAML),
    render the response **in HA** (so live state is available), and reply with
    ``Handled(text)``.

The matcher is rebuilt from disk when ``enabled.json`` / ``custom_commands.json``
change (and on a TTL so entity/area renames are picked up), keeping it in
lock-step with the STT grammar.
"""
import asyncio
import json
import logging
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from wyoming.asr import Transcript
from wyoming.handle import Handled, NotHandled
from wyoming.info import Attribution, Info, IntentModel, IntentProgram
from wyoming.intent import Entity, Intent, NotRecognized, Recognize
from wyoming.server import AsyncEventHandler, AsyncServer

import custom_commands as cc
import extra_sentences as ex
import hass_actions
from hass_satellite import resolve_area_async
from intent_matcher import IntentMatcher, build_matcher, canonical_slot
from responses import load_responses, response_for

_LOGGER = logging.getLogger("speech-to-phrase.intent")
NAME = "speech-to-phrase-intents"


def _render_local(template: Optional[str], slots: dict) -> str:
    """Local jinja2 render -- dev fallback only (no HA token). Production renders
    action responses in HA so `states()` etc. are available."""
    if not template:
        return ""
    try:
        from jinja2 import Template

        return Template(template).render(slots=slots)
    except Exception:  # noqa: BLE001
        _LOGGER.debug("local render failed", exc_info=True)
        return template


def _read_enabled(data_dir: Path, lang: str, s2p_repo: Path) -> List[Tuple[str, str]]:
    """Enabled (intent, combo) pairs: the saved set if present, else every combo
    that has a curated sentence file (so the matcher works before first save)."""
    f = data_dir / lang / "enabled.json"
    if f.exists():
        try:
            return [tuple(e) for e in json.loads(f.read_text())]
        except Exception:  # noqa: BLE001
            _LOGGER.warning("could not parse %s; using all on-disk combos", f)
    import s2p_intents

    return list(s2p_intents.combos(lang))


class MatcherHolder:
    """Owns the current :class:`IntentMatcher`, rebuilding it (off the event
    loop) when ``enabled.json`` changes or a TTL elapses."""

    def __init__(self, s2p_repo: Path, lang: str, data_dir: Path,
                 get_entities: Callable[[], Dict[str, str]],
                 get_slot_lists: Callable[[], Dict[str, List[str]]],
                 ttl: float = 600.0):
        self._s2p_repo = s2p_repo
        self._lang = lang
        self._data_dir = data_dir
        self._get_entities = get_entities
        self._get_slot_lists = get_slot_lists
        self._ttl = ttl
        self._matcher: Optional[IntentMatcher] = None
        self._sig: Optional[tuple] = None
        self._built_at: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def s2p_repo(self) -> Path:
        return self._s2p_repo

    def _disk_sig(self) -> tuple:
        """mtimes of the files that determine the matcher."""
        d = self._data_dir / self._lang
        def mt(name):
            f = d / name
            return f.stat().st_mtime if f.exists() else None
        return (mt("enabled.json"), mt(cc.FILENAME), mt(ex.FILENAME))

    async def get(self) -> Optional[IntentMatcher]:
        async with self._lock:
            sig = self._disk_sig()
            stale = (time.monotonic() - self._built_at) > self._ttl
            if self._matcher is not None and sig == self._sig and not stale:
                _LOGGER.debug("reusing cached matcher")
                return self._matcher
            reason = ("first build" if self._matcher is None
                      else "config changed" if sig != self._sig
                      else "TTL elapsed")
            _LOGGER.info("Rebuilding intent matcher (%s)", reason)
            loop = asyncio.get_event_loop()
            t0 = time.monotonic()
            self._matcher = await loop.run_in_executor(None, self._build)
            _LOGGER.info(
                "Intent matcher %s in %.2fs",
                "ready" if self._matcher is not None else "is None (nothing matchable)",
                time.monotonic() - t0,
            )
            self._sig = sig
            self._built_at = time.monotonic()
            return self._matcher

    def _build(self) -> Optional[IntentMatcher]:
        # Runs in an executor thread (no running loop), so the blocking HA
        # fetchers in training.* are safe to call here.
        try:
            enabled = _read_enabled(self._data_dir, self._lang, self._s2p_repo)
            _LOGGER.info("matcher inputs: %d enabled combo(s) for '%s'",
                         len(enabled), self._lang)
            _LOGGER.debug("enabled combos: %s", enabled)
            entities = self._get_entities()
            _LOGGER.info("matcher inputs: %d entit(y/ies) from HA/fixture", len(entities))
            _LOGGER.debug("entities: %s", entities)
            slot_lists = self._get_slot_lists()
            _LOGGER.info("matcher inputs: %d area(s)", len(slot_lists.get("area") or []))
            commands = cc.load(self._data_dir, self._lang)
            extras = ex.load(self._data_dir, self._lang)
            _LOGGER.info("matcher inputs: %d custom command(s), %d combo(s) with extras",
                         len(commands), len(extras))
            matcher = build_matcher(
                self._s2p_repo, self._lang, enabled, entities, slot_lists,
                custom_commands=commands, extra_sentences=extras,
            )
            if matcher is None:
                _LOGGER.warning(
                    "build_matcher returned None: no enabled combo resolved to a "
                    "sentence (check enabled.json and that exposed entities exist "
                    "for the combos' name_domains)"
                )
            return matcher
        except Exception:  # noqa: BLE001
            _LOGGER.exception("matcher build failed")
            return None


class IntentEventHandler(AsyncEventHandler):
    def __init__(self, reader, writer, *, holder: MatcherHolder, info: Info,
                 api_url: str, token: Optional[str],
                 responses: Dict[str, Dict[str, str]]):
        super().__init__(reader, writer)
        self._holder = holder
        self._info = info
        self._api_url = api_url
        self._token = token
        self._responses = responses

    async def handle_event(self, event) -> bool:
        from wyoming.info import Describe

        _LOGGER.debug("received event: %s", event.type)

        if Describe.is_type(event.type):
            _LOGGER.debug("Describe -> sending Info")
            await self.write_event(self._info.event())
            return True

        # HA's Wyoming conversation agent sends the transcript to recognize as a
        # `Transcript` event (asr domain) with text + context; the generic intent
        # protocol (and our test client) uses `Recognize`. Accept both.
        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            await self._recognize(transcript.text, transcript.context)
            return True

        if Recognize.is_type(event.type):
            recognize = Recognize.from_event(event)
            await self._recognize(recognize.text, recognize.context)
            return True

        _LOGGER.debug("ignoring event type: %s", event.type)
        return True

    async def _recognize(self, text, context) -> None:
        """Match text -> Intent/NotRecognized. ALWAYS replies and never lets an
        exception escape unlogged -- a silent failure looks like a hang to HA."""
        text = (text or "").strip()
        _LOGGER.info("recognize: text=%r context=%s", text, context)
        try:
            matcher = await self._holder.get()
            if matcher is None:
                _LOGGER.warning("no matcher available -> NotRecognized")
            elif not text:
                _LOGGER.info("empty transcript -> NotRecognized")
                matcher = None

            result = matcher.match(text) if matcher else None
            if result is None:
                if matcher is not None:
                    _LOGGER.info("no intent matched for %r -> NotRecognized", text)
                await self.write_event(NotRecognized().event())
                return

            metadata = result.intent_metadata or {}
            sentence = getattr(result.intent_sentence, "text", result.intent_sentence)
            _LOGGER.debug(
                "match detail: intent=%s sentence=%r raw_entities=%s metadata=%s",
                result.intent.name, sentence,
                {k: v.value for k, v in result.entities.items()}, metadata,
            )
            entities = await self._entities(result, metadata, context)
            slots = {e.name: e.value for e in entities}

            if metadata.get("mode") == "action":
                await self._handle_action(metadata, slots)
                return

            # Standard intent (built-in or custom intent-mode): hand HA the
            # intent + a response template; HA handles + renders it with state.
            if metadata.get("source") == "custom":
                response = metadata.get("response")  # user's jinja2 template
            else:
                response = response_for(
                    self._responses, result.intent.name, metadata.get("response_key")
                )
            _LOGGER.info("matched intent %s slots=%s response=%r",
                         result.intent.name, slots, response)
            await self.write_event(
                Intent(name=result.intent.name, entities=entities, text=response).event()
            )
        except Exception:  # noqa: BLE001
            _LOGGER.exception("error handling %r -> NotRecognized", text)
            await self.write_event(NotRecognized().event())

    async def _entities(self, result, metadata, context) -> List[Entity]:
        entities: List[Entity] = []
        for key, ent in result.entities.items():
            entities.append(Entity(name=canonical_slot(key), value=ent.value))
        if metadata.get("domain"):  # built-in inferred_domain
            entities.append(Entity(name="domain", value=metadata["domain"]))
        for k, v in (metadata.get("slots") or {}).items():  # custom fixed slots
            entities.append(Entity(name=k, value=v))
        if metadata.get("context_area"):
            area = await self._resolve_area(context)
            if area:
                entities.append(Entity(name="area", value=area))
            else:
                _LOGGER.warning(
                    "context_area command but no satellite area resolved "
                    "(context=%s) -> emitting without area slot", context,
                )
        return entities

    # ---- custom action path -------------------------------------------------

    async def _handle_action(self, metadata, slots) -> None:
        action = metadata.get("action") or {}
        response_tmpl = metadata.get("response")
        _LOGGER.info("action: %s slots=%s", action, slots)
        try:
            if not self._token:
                # Dev mode: can't reach HA. Render locally, skip execution.
                text = _render_local(response_tmpl, slots)
                _LOGGER.warning("no HA token: skipping execution (dev) -> Handled %r", text)
                await self.write_event(Handled(text=text).event())
                return
            if not await self._run_action(action, slots):
                await self.write_event(
                    NotHandled(text="Sorry, that action failed.").event()
                )
                return
            text = await hass_actions.render_template_async(
                self._api_url, self._token, response_tmpl or "",
                variables={"slots": slots},
            ) if response_tmpl else ""
            _LOGGER.info("action handled -> %r", text)
            await self.write_event(Handled(text=text or "").event())
        except Exception:  # noqa: BLE001
            _LOGGER.exception("action failed -> NotHandled")
            await self.write_event(
                NotHandled(text="Sorry, that action failed.").event()
            )

    async def _run_action(self, action, slots) -> bool:
        return await hass_actions.run_action_async(
            self._api_url, self._token, action, slots
        )

    async def _resolve_area(self, context) -> Optional[str]:
        context = context or {}
        # Explicit area in context (e.g. from a test client) wins.
        explicit = context.get("area")
        if explicit:
            area = explicit["name"] if isinstance(explicit, dict) else explicit
            _LOGGER.debug("area from explicit context: %r", area)
            return area
        if not self._token:
            _LOGGER.debug("no HA token; cannot resolve satellite area")
            return None
        area = await resolve_area_async(
            self._api_url, self._token,
            device_id=context.get("device_id"),
            satellite_id=context.get("satellite_id"),
        )
        _LOGGER.debug(
            "resolved area=%r from device_id=%s satellite_id=%s",
            area, context.get("device_id"), context.get("satellite_id"),
        )
        return area


def build_info(language: str) -> Info:
    return Info(
        intent=[
            IntentProgram(
                name=NAME,
                description="Constrained intent recognition",
                installed=True,
                version="0.1.0",
                attribution=Attribution(
                    name="OHF Voice", url="https://openhomefoundation.org"
                ),
                models=[
                    IntentModel(
                        name=f"{NAME}-{language}",
                        installed=True,
                        description=f"Speech-to-Phrase intents ({language})",
                        version="0.1.0",
                        attribution=Attribution(name="", url=""),
                        languages=[language],
                    )
                ],
            )
        ]
    )


async def serve(uri: str, language: str, holder: MatcherHolder,
                api_url: str, token: Optional[str]) -> None:
    info = build_info(language)
    responses = load_responses(holder.s2p_repo, language)
    # Warm the matcher in the background so a slow/hanging HA fetch can't keep
    # the server from accepting connections.
    asyncio.create_task(holder.get())
    server = AsyncServer.from_uri(uri)
    _LOGGER.info("Wyoming intent server listening on %s (language=%s)", uri, language)
    await server.run(
        partial(IntentEventHandler, holder=holder, info=info,
                api_url=api_url, token=token, responses=responses)
    )


def start_background(uri: str, language: str, data_dir: Path, s2p_repo: Path,
                     get_entities: Callable[[], Dict[str, str]],
                     get_slot_lists: Callable[[], Dict[str, List[str]]],
                     api_url: str, token: Optional[str],
                     ttl: float = 600.0) -> "threading.Thread":
    """Run the intent server in a daemon thread with its own asyncio loop."""
    import threading

    holder = MatcherHolder(
        s2p_repo, language, data_dir, get_entities, get_slot_lists, ttl=ttl
    )

    def _runner():
        try:
            asyncio.run(serve(uri, language, holder, api_url, token))
        except Exception:  # noqa: BLE001
            _LOGGER.exception("intent server thread crashed (uri=%s)", uri)

    _LOGGER.info("Starting Wyoming intent service thread (uri=%s)", uri)
    t = threading.Thread(target=_runner, name="wyoming-intent", daemon=True)
    t.start()
    return t
