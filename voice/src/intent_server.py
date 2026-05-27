import logging
import re
import time
from collections.abc import Mapping
from typing import Any, Collection, Dict, List, Optional, Set

from jinja2 import BaseLoader
from jinja2.nativetypes import NativeEnvironment
from rapidfuzz import fuzz, process
from wyoming.asr import Transcript
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, IntentModel, IntentProgram
from wyoming.intent import Entity, Intent, IntentsStart, IntentsStop, NotRecognized
from wyoming.server import AsyncEventHandler

from const import AppState, Tool
from gemma4_recognizer import TOOL_ARGS, TOOL_CALL, Gemma4Recognizer
from lang_intents import LanguageIntents
from name_resolver import NameResolver
from util import normalize_name

_LOGGER = logging.getLogger(__name__)

AREA_SLOT = "area"
FLOOR_SLOT = "floor"
ENTITY_NAME_SLOT = "name"
DOMAIN_SLOT = "domain"

# _MIN_FUZZY_SCORE = 0.85
_RAPID_FUZZ_CUTOFF = 95
_NAME_THRESHOLD = 0.9
_TODO_THRESHOLD = 0.85
_ENV = NativeEnvironment(loader=BaseLoader())


class UnresolvedNameError(Exception):
    pass


class Gemma4EventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        state: AppState,
        recognizer: Gemma4Recognizer,
        lang_intents: LanguageIntents,
        # fuzzy_matcher: FuzzyMatcher,
        *args,
        **kwargs,
    ) -> None:
        """Initialize event handler."""
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.state = state
        self.recognizer = recognizer
        self.lang_intents = lang_intents
        # self.fuzzy_matcher = fuzzy_matcher

        self.device_id: Optional[str] = None
        self.satellite_id: Optional[str] = None

        self._info_event: Optional[Event] = None

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        try:
            return await self._handle_event(event)
        except Exception:
            _LOGGER.exception("Error handling event")

        return True

    async def _handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        if Describe.is_type(event.type):
            await self._write_info()
            return True

        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            _LOGGER.debug("Handling: %s", transcript)

            # Reset
            self.device_id = None
            self.satellite_id = None
            if transcript.context:
                self.device_id = transcript.context.get("device_id")
                self.satellite_id = transcript.context.get("satellite_id")

            try:
                await self._handle_transcript_gemma(transcript)
                # await self._handle_transcript_fuzzy(transcript)
            except Exception:
                _LOGGER.exception("Unexpected error during handling")
                await self.write_event(
                    Error(
                        text="Unexpected error during handling", code="handle-error"
                    ).event()
                )

            return True

        return True

    # async def _handle_transcript_fuzzy(
    #     self, transcript: Transcript
    # ) -> None:
    #     cand_match = self.fuzzy_matcher.match_candidate(
    #         transcript.text, language=transcript.language or "en"
    #     )
    #     command: Optional[FuzzyCommand] = None
    #     if cand_match is not None:
    #         command_text, command_idx = self.state.fuzzy_candidates[
    #             cand_match.candidate_idx
    #         ]
    #         command = self.state.fuzzy_commands[command_idx]
    #         if cand_match.score < _MIN_FUZZY_SCORE:
    #             _LOGGER.debug(
    #                 "Fuzzy command score was too low: text=%s, match=%s, command=%s",
    #                 command_text,
    #                 cand_match,
    #                 command,
    #             )
    #             command = None
    #         else:
    #             _LOGGER.debug("Matched fuzzy command: %s", command)

    #     if (cand_match is None) or (command is None):
    #         await self.write_event(
    #             NotRecognized(
    #                 text=self.lang_intents.get_error_response(
    #                     language=transcript.language or "en", key="no_intent"
    #                 ),
    #                 context=transcript.context,
    #             ).event()
    #         )
    #         return

    #     intent_name = command.intent_name
    #     intent_slots: Dict[str, Any] = {}

    #     if command.intent_slots:
    #         intent_slots.update(command.intent_slots)

    #     if command.context_area and (not intent_slots.get(AREA_SLOT)):
    #         context_area_id = hass_info.current_area_id or self.state.default_area_id
    #         if context_area_id:
    #             intent_slots[AREA_SLOT] = context_area_id

    #     template_vars: Dict[str, Any] = {"slots": intent_slots}
    #     if command.number and (cand_match.number is not None):
    #         template_vars["number"] = cand_match.number

    #     if command.duration and (cand_match.duration is not None):
    #         template_vars["duration"] = cand_match.duration

    #     if is_template_string(intent_name):
    #         intent_name = render_template(intent_name, template_vars)

    #     intent_slots = render_templates_recursive(intent_slots, template_vars)

    #     _LOGGER.debug("Intent: name=%s, slots=%s", intent_name, intent_slots)
    #     await self.write_event(
    #         Intent(
    #             name=intent_name,
    #             entities=[
    #                 Entity(name=name, value=value)
    #                 for name, value in intent_slots.items()
    #             ],
    #             text=self.lang_intents.get_intent_response(
    #                 language=transcript.language or "en",
    #                 intent_name=command.intent_name,
    #                 intent_slots=intent_slots,
    #             ),
    #             context=transcript.context,
    #         ).event()
    #     )

    async def _handle_transcript_gemma(self, transcript: Transcript) -> None:
        language = transcript.language or "en"
        tool_calls, response_text = self.recognizer.get_tool_calls(
            transcript.text, language
        )
        if not tool_calls:
            await self.write_event(
                NotRecognized(text=response_text, context=transcript.context).event()
            )
            return

        # Load name resolver
        language_family = re.split(r"[_-]", language, maxsplit=1)[0]
        if language_family == "en":
            if self.state.resolver_en is None:
                self.state.resolver_en = NameResolver(self.state.resolver_en_model)
                self.state.resolver_en.load()

            name_resolver = self.state.resolver_en
        else:
            if self.state.resolver_multilingual is None:
                self.state.resolver_multilingual = NameResolver(
                    self.state.resolver_multilingual_model
                )
                self.state.resolver_multilingual.load()

            name_resolver = self.state.resolver_multilingual

        intent_events: List[Intent] = []

        for tool_name, tool_args in tool_calls:
            tool = self.state.tools[tool_name]

            try:
                await self._resolve_names(tool, tool_args, name_resolver)
            except UnresolvedNameError:
                _LOGGER.exception(
                    "Failed to resolve names: tool_name=%s, tool_args=%s",
                    tool_name,
                    tool_args,
                )
                # Fail
                intent_events.clear()
                break

            if tool.intent:
                intent_name, intent_slots = await self._tool_call_to_intent(
                    tool, tool_args
                )
            else:
                intent_name, intent_slots = tool_name, tool_args

            if intent_name in ("HassListCompleteItem", "HassListRemoveItem"):
                todo_entity_id = intent_slots.get(ENTITY_NAME_SLOT) or ""
                if todo_entity_id.startswith("todo."):
                    todo_item = intent_slots["item"]
                    await self._resolve_todo_item(
                        todo_item, todo_entity_id, intent_slots, name_resolver
                    )

            _LOGGER.debug("Intent: name=%s, slots=%s", intent_name, intent_slots)
            intent_events.append(
                Intent(
                    name=intent_name,
                    entities=[
                        Entity(name=name, value=value)
                        for name, value in intent_slots.items()
                    ],
                    text=self.lang_intents.get_intent_response(
                        language=language,
                        intent_name=intent_name,
                        intent_slots=intent_slots,
                    ),
                    context=transcript.context,
                )
            )

        if intent_events:
            await self.write_event(IntentsStart(context=transcript.context).event())
            for intent in intent_events:
                await self.write_event(intent.event())
            await self.write_event(IntentsStop(context=transcript.context).event())
        else:
            # TODO: give a better error message for unresolved names
            await self.write_event(
                NotRecognized(
                    text=self.lang_intents.get_error_response(
                        language=language, key="handle_error"
                    ),
                    context=transcript.context,
                ).event()
            )

    async def _resolve_names(
        self, tool: Tool, tool_args: TOOL_ARGS, name_resolver: NameResolver
    ) -> None:
        location: Optional[str] = tool_args.get("location")
        if location:
            self._resolve_location(location, tool_args, name_resolver)

        entity_name: Optional[str] = tool_args.get("device_name") or tool_args.get(
            "list_name"
        )
        if entity_name:
            self._resolve_entity(
                entity_name, tool.name_domains, tool_args, name_resolver
            )

    def _resolve_location(
        self, location: str, tool_args: TOOL_ARGS, name_resolver: NameResolver
    ) -> None:
        location_norm = normalize_name(location)
        location_names: Dict[str, str] = {}
        location_names_norm: Dict[str, str] = {}

        for area in self.state.hass_info.areas.values():
            # Ensure area/floor ids are deconflicted
            mapped_id = f"area_{area.area_id}"
            for area_name in area.names:
                if area_name:
                    location_names[area_name] = mapped_id
            for area_name_norm in area.names_norm:
                if area_name_norm:
                    location_names_norm[area_name_norm] = mapped_id
        for floor in self.state.hass_info.floors.values():
            # Ensure area/floor ids are deconflicted
            mapped_id = f"floor_{floor.floor_id}"
            for floor_name in floor.names:
                if floor_name:
                    location_names[floor_name] = mapped_id
            for floor_name_norm in floor.names_norm:
                if floor_name_norm:
                    location_names_norm[floor_name_norm] = mapped_id

        best_id: Optional[str] = location_names.get(location)
        if not best_id:
            best_id = location_names_norm.get(location_norm)

        if not best_id:
            best_name = name_resolver.best_candidate(
                location, list(location_names), threshold=_NAME_THRESHOLD
            )
            if best_name:
                best_id = location_names[best_name]

        if not best_id:
            # Fuzzy match
            result = process.extractOne(
                location_norm,
                list(location_names_norm),
                scorer=fuzz.ratio,
                score_cutoff=_RAPID_FUZZ_CUTOFF,
            )
            if result:
                # Map back to original name
                best_name_norm = result[0]
                best_id = location_names_norm[best_name_norm]

        if best_id:
            location_type, location_id = best_id.split("_", maxsplit=1)
            if location_type == "area":
                best_area = self.state.hass_info.areas[location_id]
                _LOGGER.debug("Resolved %s to %s", location, best_area)
                tool_args[AREA_SLOT] = best_area.area_id
            elif location_type == "floor":
                best_floor = self.state.hass_info.floors[location_id]
                _LOGGER.debug("Resolved %s to %s", location, best_floor)
                tool_args[FLOOR_SLOT] = best_floor.floor_id
        else:
            # Try to pass directly as an area
            _LOGGER.warning(
                "Couldn't resolve location name. Assuming area: %s", location
            )
            tool_args[AREA_SLOT] = location

    def _resolve_entity(
        self,
        entity_name: str,
        name_domains: Optional[Set[str]],
        tool_args: TOOL_ARGS,
        name_resolver: NameResolver,
    ) -> None:
        entity_norm = normalize_name(entity_name)
        entity_names: Dict[str, str] = {}
        entity_names_norm: Dict[str, str] = {}

        entities: Collection[Entity] = self.state.hass_info.entities.values()
        if name_domains:
            entities = [e for e in entities if e.domain in name_domains]

        for entity in entities:
            for name in entity.names:
                if name:
                    entity_names[name] = entity.entity_id
            for entity_name_norm in entity.names_norm:
                if entity_name_norm:
                    entity_names_norm[entity_name_norm] = entity.entity_id

        best_id: Optional[str] = entity_names.get(entity_name)
        if not best_id:
            best_id = entity_names_norm.get(entity_norm)

        if not best_id:
            best_name = name_resolver.best_candidate(
                entity_name, list(entity_names), threshold=_NAME_THRESHOLD
            )
            if best_name:
                best_id = entity_names[best_name]

        if not best_id:
            # Fuzzy match
            result = process.extractOne(
                entity_norm,
                list(entity_names_norm),
                scorer=fuzz.ratio,
                score_cutoff=_RAPID_FUZZ_CUTOFF,
            )
            if result:
                # Map back to original name
                best_name_norm = result[0]
                best_id = entity_names_norm[best_name_norm]
                _LOGGER.debug("Fuzzy match: %s -> %s", entity_name, best_name_norm)

        if best_id:
            best_entity = self.state.hass_info.entities[best_id]
            _LOGGER.debug("Resolved %s to %s", entity_name, best_entity)
            tool_args[ENTITY_NAME_SLOT] = best_entity.entity_id
            tool_args[DOMAIN_SLOT] = best_entity.domain
        else:
            # Try to pass directly as a name
            _LOGGER.warning("Couldn't resolve entity name: %s", entity_name)
            tool_args[ENTITY_NAME_SLOT] = entity_name

    async def _resolve_todo_item(
        self,
        item: str,
        entity_id: str,
        intent_slots: Dict[str, Any],
        name_resolver: NameResolver,
    ) -> None:
        response = (
            await self.state.hass.call_service(
                "todo",
                "get_items",
                service_data={"status": "needs_action"},
                target={"entity_id": entity_id},
                return_response=True,
            )
            or {}
        )
        _LOGGER.debug(response)

        items = response.get(entity_id, {}).get("items")
        if not items:
            return

        best_item = name_resolver.best_candidate(
            item, [item["summary"] for item in items], threshold=_TODO_THRESHOLD
        )
        if best_item:
            _LOGGER.debug("Resolved todo item '%s' to '%s'", item, best_item)
            intent_slots["item"] = best_item

    async def _tool_call_to_intent(self, tool: Tool, tool_args: TOOL_ARGS) -> TOOL_CALL:
        assert tool.intent is not None

        intent_name = tool.intent.name
        if is_template_string(intent_name):
            intent_name = render_template(intent_name, tool_args) or ""

        intent_name = intent_name.strip()
        if not intent_name:
            raise ValueError("No intent name")

        intent_slots: TOOL_ARGS = {}
        if tool.intent.slots:
            intent_slots = (
                render_templates_recursive(tool.intent.slots, tool_args) or {}
            )

        # Remove empty values
        if intent_slots:
            intent_slots = {
                key: value for key, value in intent_slots.items() if value is not None
            }

        # Fill in area from context
        if (
            tool.context_area
            and (not intent_slots.get(AREA_SLOT))
            and (not intent_slots.get(FLOOR_SLOT))
        ):
            current_area_id = await self.state.hass.get_current_area(
                self.device_id, self.satellite_id
            )
            context_area_id = current_area_id or self.state.default_area_id
            if context_area_id:
                intent_slots[AREA_SLOT] = context_area_id

        return (intent_name, intent_slots)

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        info = Info(
            intent=[
                IntentProgram(
                    name="intent-gemma4",
                    attribution=Attribution(
                        "Open Home Foundation Voice", "https://github.com/OHF-Voice"
                    ),
                    installed=True,
                    description="Gemma 4 Intent Recognizer",
                    version="0.0.1",
                    models=[
                        IntentModel(
                            name="gemma4",
                            attribution=Attribution(
                                "Google DeepMind",
                                "https://deepmind.google/models/gemma/gemma-4/",
                            ),
                            installed=True,
                            description="gemma4",
                            version="",
                            languages=[],  # all languages
                        )
                    ],
                )
            ],
        )

        self._info_event = info.event()
        await self.write_event(self._info_event)


# -----------------------------------------------------------------------------


def render_templates_recursive(data: Any, variables: Mapping[str, Any]) -> Any:
    # Template string handling
    if isinstance(data, str) and is_template_string(data):
        return render_template(data, variables)

    # Mapping (dict-like)
    if isinstance(data, Mapping):
        return {k: render_templates_recursive(v, variables) for k, v in data.items()}

    # Sequence (but not str/bytes)
    if isinstance(data, (list, tuple)):
        rendered = [render_templates_recursive(v, variables) for v in data]
        return rendered if isinstance(data, list) else tuple(rendered)

    return data


def render_template(data: str, variables: Mapping[str, Any]) -> Any:
    result = _ENV.from_string(data).render(variables)
    _LOGGER.debug("Rendered template='%s', vars=%s, result=%s", data, variables, result)
    return result


def is_template_string(maybe_template: str) -> bool:
    """Check if the input is a Jinja2 template."""
    return "{" in maybe_template and (
        "{%" in maybe_template or "{{" in maybe_template or "{#" in maybe_template
    )
