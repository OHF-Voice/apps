#!/usr/bin/env python3

import argparse
import asyncio
import logging
import re
from collections.abc import Collection
from functools import partial
from typing import Dict, List, Optional, Set

from ruamel.yaml import YAML
from wyoming.server import AsyncServer

from const import AppState, FuzzyCommand, Tool, ToolIntent
from gemma4_recognizer import Gemma4Recognizer
from hass_api import HomeAssistant
from intent_server import Gemma4EventHandler
from lang_intents import LanguageIntents
from models import Entity
from name_resolver import NameResolver
from web_server import make_web_server, run_web_server

_LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


async def main() -> None:
    """Run app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    #
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=5000)
    #
    parser.add_argument("--hass-token", required=True)
    parser.add_argument("--hass-api", default="http://homeassistant.local:8123")
    parser.add_argument(
        "--default-area-id", help="Area id to use if no context area is available"
    )
    #
    # parser.add_argument("--recognizer-repo", default="ggml-org/gemma-4-E2B-it-GGUF")
    parser.add_argument(
        "--recognizer-repo", default="bartowski/google_gemma-4-E2B-it-GGUF"
    )
    # parser.add_argument("--recognizer-filename", default="gemma-4-E2B-it-Q8_0.gguf")
    parser.add_argument(
        "--recognizer-filename", default="google_gemma-4-E2B-it-Q5_K_M.gguf"
    )
    parser.add_argument("--tools", required=True, help="Path to tools YAML file")
    parser.add_argument(
        "--tool-call-cache-size",
        type=int,
        default=100,
        help="Number of sentences to remember for tool calls",
    )
    parser.add_argument(
        "--llama-state", required=True, help="Path to save llama.cpp state"
    )
    parser.add_argument(
        "--include-names-in-tools",
        action="store_true",
        help="Include location and device names in LLM tools (slower but more accurate)",
    )
    #
    # parser.add_argument(
    #     "--fuzzy-commands", required=True, help="Path to fuzzy commands YAML file"
    # )
    #
    parser.add_argument(
        "--resolver-en-model",
        default="intfloat/e5-small-v2",
        help="HuggingFace id of sentence transformers used for name resolution (English)",
    )
    parser.add_argument(
        "--resolver-multilingual-model",
        default="intfloat/multilingual-e5-small",
        help="HuggingFace id of sentence transformers used for name resolution (multilingual)",
    )
    parser.add_argument(
        "--resolver-language",
        default="en",
        help="Default language for name resolution (default: en)",
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    hass = HomeAssistant(token=args.hass_token, api_url=args.hass_api)
    hass_info = await hass.get_info()

    # All area/floor names
    location_names: Set[str] = set()
    for area in hass_info.areas.values():
        for area_name in area.names:
            if area_name:
                location_names.add(area_name)
    for floor in hass_info.floors.values():
        for floor_name in floor.names:
            if floor_name:
                location_names.add(floor_name)

    location_names_sorted = sorted(location_names)

    # Load tools
    yaml = YAML(typ="safe")
    with open(args.tools, "r", encoding="utf-8") as tools_file:
        yaml_tools = yaml.load(tools_file)

    tools: Dict[str, Tool] = {}
    for tool_dict in yaml_tools:
        tool_name = tool_dict["tool"]["function"]["name"]
        tool = Tool(
            tool=tool_dict["tool"], context_area=tool_dict.get("context_area", False)
        )
        tools[tool_name] = tool

        # Parse tool info
        tool_intent_dict = tool_dict.get("intent")
        if tool_intent_dict:
            tool.intent = ToolIntent(
                name=tool_intent_dict["name"], slots=tool_intent_dict.get("slots")
            )

        tool.requires_light_brightness = tool_dict.get(
            "requires_light_brightness", False
        )
        tool.requires_light_color = tool_dict.get("requires_light_color", False)

        tool_name_domains = tool_dict.get("name_domains")
        if tool_name_domains:
            tool.name_domains = set(tool_name_domains)

        tool.name_features = tool_dict.get("name_features")

        if not args.include_names_in_tools:
            continue

        # Gather list of supported entities by domain and features
        supported_entities: Optional[Collection[Entity]] = None
        if tool.name_domains:
            supported_entities = []

            if tool.name_features:
                for entity in hass_info.entities.values():
                    if entity.domain not in tool.name_domains:
                        continue

                    required_features = tool.name_features.get(entity.domain)
                    if required_features is None:
                        continue

                    if (
                        entity.supported_features & required_features
                    ) != required_features:
                        continue

                    supported_entities.append(entity)
            else:
                # No need to filter by supported features
                supported_entities = [
                    e
                    for e in hass_info.entities.values()
                    if e.domain in tool.name_domains
                ]

            if tool.requires_light_brightness or tool.requires_light_color:
                # Filter lights by supported color modes
                supported_entities = [
                    e
                    for e in supported_entities
                    if (
                        (not tool.requires_light_brightness)
                        or e.light_supports_brightness
                    )
                    and ((not tool.requires_light_color) or e.light_supports_color)
                ]

            if not supported_entities:
                _LOGGER.debug(
                    "Skipping tool %s. No entities have the correct domain or required features: domains=%s, features=%s, light_brightness=%s, light_color=%s",
                    tool_name,
                    tool.name_domains,
                    tool.name_features,
                    tool.requires_light_brightness,
                    tool.requires_light_color,
                )
                continue

        tool_area_domains = tool_dict.get("area_domains")
        if tool_area_domains:
            tool_area_domains = set(tool_area_domains)

        tool_params = (
            tool.tool.get("function", {}).get("parameters", {}).get("properties", {})
        )

        # Add area/floor names
        location_param = tool_params.get("location")
        if location_param and (not location_param.get("enum")):
            supported_location_names: Set[str]
            supported_location_names_sorted: List[str] = []
            if supported_entities is None:
                if tool_area_domains:
                    supported_location_names = set()
                    for entity in hass_info.entities.values():
                        if entity.domain not in tool_area_domains:
                            continue

                        if not entity.area_id:
                            continue

                        maybe_area = hass_info.areas.get(entity.area_id)
                        if maybe_area is None:
                            continue

                        for area_name in maybe_area.names:
                            if area_name:
                                supported_location_names.add(area_name)

                        if not maybe_area.floor_id:
                            continue

                        maybe_floor = hass_info.floors.get(maybe_area.floor_id)
                        if maybe_floor is None:
                            continue

                        for floor_name in maybe_floor.names:
                            if floor_name:
                                supported_location_names.add(floor_name)

                    supported_location_names_sorted = sorted(supported_location_names)
                else:
                    # All location names
                    supported_location_names_sorted = location_names_sorted
            else:
                # Locations filtered by supported entities
                supported_location_names = set()
                for entity in supported_entities:
                    if not entity.area_id:
                        continue

                    maybe_area = hass_info.areas.get(entity.area_id)
                    if maybe_area is None:
                        continue

                    for area_name in maybe_area.names:
                        if area_name:
                            supported_location_names.add(area_name)

                    if not maybe_area.floor_id:
                        continue

                    maybe_floor = hass_info.floors.get(maybe_area.floor_id)
                    if maybe_floor is None:
                        continue

                    for floor_name in maybe_floor.names:
                        if floor_name:
                            supported_location_names.add(floor_name)

                supported_location_names_sorted = sorted(supported_location_names)

            if supported_location_names_sorted:
                location_param["enum"] = supported_location_names_sorted

        # Add entity names
        device_name_param = tool_params.get("device_name")
        if (
            device_name_param
            and (not device_name_param.get("enum"))
            and supported_entities
        ):
            supported_device_names: Set[str] = set()
            for entity in supported_entities:
                for entity_name in entity.names:
                    if entity_name:
                        supported_device_names.add(entity_name)

            if supported_device_names:
                device_name_param["enum"] = sorted(supported_device_names)

    # DEBUG
    if args.debug:
        with open("tools.debug.yaml", "w", encoding="utf-8") as f:
            yaml.dump([t.tool for t in tools.values()], f)

    _LOGGER.debug("Loaded %s tool(s)", len(tools))

    # Load fuzzy commands
    # with open(args.fuzzy_commands, "r", encoding="utf-8") as fuzzy_commands_file:
    #     yaml_commands = yaml.load(fuzzy_commands_file)

    fuzzy_commands: List[FuzzyCommand] = []
    # for command_dict in yaml_commands:
    #     fuzzy_commands.append(
    #         FuzzyCommand(
    #             intent_name=command_dict["intent"]["name"],
    #             sentences=command_dict["sentences"],
    #             intent_slots=command_dict["intent"].get("slots"),
    #             context_area=command_dict.get("context_area"),
    #             duration=command_dict.get("duration"),
    #             number=command_dict.get("number"),
    #         )
    #     )

    # _LOGGER.debug("Loaded %s fuzzy command(s)", len(fuzzy_commands))

    state = AppState(
        hass=hass,
        hass_info=hass_info,
        http_host=args.http_host,
        http_port=args.http_port,
        tools=tools,
        resolver_en_model=args.resolver_en_model,
        resolver_multilingual_model=args.resolver_multilingual_model,
        fuzzy_commands=fuzzy_commands,
        fuzzy_candidates=[
            (s, i) for i, cmd in enumerate(fuzzy_commands) for s in cmd.sentences
        ],
        default_area_id=args.default_area_id,
    )

    _LOGGER.info("Loading LLM")
    recognizer = Gemma4Recognizer(
        repo_id=args.recognizer_repo,
        filename=args.recognizer_filename,
        state_path=args.llama_state,
        cache_size=args.tool_call_cache_size,
    )
    recognizer.load([t.tool for t in tools.values()])

    lang_intents = LanguageIntents()

    resolver_language_family = re.split(r"[_-]", args.resolver_language, maxsplit=1)[0]
    if resolver_language_family == "en":
        state.resolver_en = NameResolver(args.resolver_en_model)
        state.resolver_en.load()
    else:
        state.resolver_multilingual = NameResolver(args.resolver_multilingual)
        state.resolver_multilingual.load()

    # fuzzy_matcher = FuzzyMatcher()
    # fuzzy_matcher.model = name_resolver.model
    # fuzzy_matcher.load()
    # fuzzy_matcher.train(s for s, _ in state.fuzzy_candidates)

    flask_app = make_web_server(state)
    flask_thread = run_web_server(state, flask_app)
    flask_thread.start()

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    try:
        await server.run(
            partial(
                Gemma4EventHandler,
                state,
                recognizer,
                lang_intents,
                # fuzzy_matcher,
            )
        )
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
