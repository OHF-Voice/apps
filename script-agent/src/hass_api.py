"""Wrapper for Home Assistant REST/Websocket API."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

import aiohttp

from models import ATTR_FRIENDLY_NAME, Area, Entity, Floor, State
from overrides import AREA, ENTITY, FLOOR, NameOverrides

_LOGGER = logging.getLogger(__name__)

SEARCH_MEDIA = 4194304  # MediaPlayerEntityFeature
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)
WEBSOCKET_RECEIVE_TIMEOUT = 30.0
MULTIPLE_SELECTORS = {"area", "entity", "floor", "select", "text"}


class HomeAssistantError(Exception):
    pass


@dataclass
class HomeAssistantInfo:
    states: Dict[str, State]
    entities: Dict[str, Entity]
    areas: Dict[str, Area]
    floors: Dict[str, Floor]
    # assist_satellite entity id -> display name. Satellites do not need to be
    # exposed to Assist to be useful as test context.
    satellites: Dict[str, str] = field(default_factory=dict)


@dataclass
class SatelliteInfo:
    entity_id: Optional[str] = None
    device_id: Optional[str] = None
    area_id: Optional[str] = None
    floor_id: Optional[str] = None
    media_player_id: Optional[str] = None
    music_player_id: Optional[str] = None
    music_assistant_id: Optional[str] = None

    def as_dict(self) -> Dict[str, str]:
        info_dict: Dict[str, str] = {}
        if self.entity_id:
            info_dict["entity_id"] = self.entity_id
        if self.device_id:
            info_dict["device_id"] = self.device_id
        if self.area_id:
            info_dict["area_id"] = self.area_id
        if self.floor_id:
            info_dict["floor_id"] = self.floor_id
        if self.media_player_id:
            info_dict["media_player_id"] = self.media_player_id
        if self.music_player_id:
            info_dict["music_player_id"] = self.music_player_id
        if self.music_assistant_id:
            info_dict["music_assistant_id"] = self.music_assistant_id

        return info_dict

    def as_script_variable(self, language: str) -> Dict[str, str]:
        """Return the special variable passed to Home Assistant scripts."""
        info_dict = self.as_dict()
        info_dict["language"] = language
        return info_dict


def _apply_satellite_registry_info(
    satellite_info: SatelliteInfo,
    satellite_dict: Dict[str, Any],
    devices: Dict[str, Dict[str, Any]],
) -> None:
    """Fill location and device information from registry snapshots."""
    satellite_info.area_id = satellite_dict.get("area_id")
    satellite_info.device_id = (
        satellite_dict.get("device_id") or satellite_info.device_id
    )
    if satellite_info.device_id and not satellite_info.area_id:
        satellite_info.area_id = devices.get(satellite_info.device_id, {}).get(
            "area_id"
        )


def _domain_set(domains: Any) -> Optional[Set[str]]:
    """One or more domain names as a set, or None when there is no restriction."""
    if not domains:
        return None

    if isinstance(domains, str):
        return {domains}

    if isinstance(domains, list):
        return {domain for domain in domains if isinstance(domain, str)} or None

    return None


def _get_entity_filter_domains(
    entity_selector: Dict[str, Any],
) -> Optional[Set[str]]:
    """Get domains from any Home Assistant entity filter representation."""
    entity_filters = entity_selector.get("filter")
    if not entity_filters:
        # The shorthand that predates `filter` (``entity: {domain: calendar}``)
        # is still valid, and still what most blueprints are written with.
        # Without this, such a field is offered every exposed entity.
        return _domain_set(entity_selector.get("domain"))

    if isinstance(entity_filters, dict):
        entity_filters = [entity_filters]

    domains: Set[str] = set()
    for entity_filter in entity_filters:
        if not isinstance(entity_filter, dict):
            return None
        filter_domains = _domain_set(entity_filter.get("domain"))
        if filter_domains is None:
            # This alternative does not restrict the domain, so neither can we.
            return None

        domains.update(filter_domains)

    return domains or None


def _get_select_options(select_selector: Dict[str, Any]) -> List[Any]:
    """Get the submitted values from plain or labelled select options."""
    return [
        option.get("value") if isinstance(option, dict) else option
        for option in select_selector.get("options", [])
    ]


def _number_property(number_selector: Dict[str, Any]) -> Dict[str, Any]:
    """Build the schema Home Assistant enforces for a number selector."""
    prop: Dict[str, Any] = {"type": "number"}
    if "min" in number_selector:
        prop["minimum"] = number_selector["min"]
    if "max" in number_selector:
        prop["maximum"] = number_selector["max"]
    return prop


def _select_property(select_selector: Dict[str, Any]) -> Dict[str, Any]:
    """Build a select schema, preserving support for custom values."""
    prop: Dict[str, Any] = {"type": "string"}
    if not select_selector.get("custom_value"):
        prop["enum"] = _get_select_options(select_selector)
    return prop


def _multiple_property(item_property: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap one selector value as a unique array of values."""
    return {
        "type": "array",
        "items": item_property,
        "uniqueItems": True,
    }


@dataclass
class Tool:
    name: str
    tool: Dict[str, Any]
    # Field -> display name -> the Home Assistant ids that name refers to.
    # More than one id means the name is ambiguous; the resolver picks.
    name_map: Dict[str, Dict[str, List[str]]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # Field -> what its names refer to ("entity", "area", or "floor").
    field_kinds: Dict[str, str] = field(default_factory=dict)
    # Field -> every id it could refer to, whatever the names currently are.
    # This is what the per-field name editor lists, so a target whose names were
    # cleared can still be found and given some back.
    field_targets: Dict[str, List[str]] = field(default_factory=dict)
    # Script name and description as shown in Home Assistant (the model-facing
    # copy lives in ``tool``, where a missing description falls back to the name).
    friendly_name: str = ""
    description: str = ""
    # True when the script is exposed to voice in Home Assistant. Tools are built
    # for every script so the web UI can show them, but only exposed ones are
    # given to the model.
    exposed: bool = False


class HomeAssistant:
    """API to Home Assistant."""

    def __init__(
        self,
        token: str,
        api_url: str = "http://homeassistant.local:8123/api",
    ) -> None:
        self.token = token
        self.api_url = api_url.rstrip("/")

        # Get websocket API URL
        parsed = urlparse(self.api_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        # Convert scheme
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = f"{parsed.path}/websocket"
        self.websocket_api_url = urlunparse(
            parsed._replace(
                scheme=scheme,
                path=path,
                params="",
                query="",
                fragment="",
            )
        )

    async def get_satellite_info(
        self, device_id: Optional[str] = None, satellite_id: Optional[str] = None
    ) -> SatelliteInfo:
        satellite_info = SatelliteInfo(device_id=device_id, entity_id=satellite_id)

        if (satellite_info.device_id is None) and (satellite_info.entity_id is None):
            # Can't get any more info
            return satellite_info

        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as session:
            async with session.ws_connect(
                self.websocket_api_url,
                max_msg_size=0,
                receive_timeout=WEBSOCKET_RECEIVE_TIMEOUT,
            ) as websocket:
                # Authenticate
                msg = await websocket.receive_json()
                assert msg["type"] == "auth_required", msg

                await websocket.send_json(
                    {
                        "type": "auth",
                        "access_token": self.token,
                    },
                )

                msg = await websocket.receive_json()
                assert msg["type"] == "auth_ok", msg

                # Devices
                await websocket.send_json(
                    {"id": next_id(), "type": "config/device_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                devices = {
                    device_info["id"]: device_info for device_info in msg["result"]
                }

                # Areas
                await websocket.send_json(
                    {"id": next_id(), "type": "config/area_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                areas = {area_info["area_id"]: area_info for area_info in msg["result"]}

                # Floors
                # await websocket.send_json(
                #     {"id": next_id(), "type": "config/floor_registry/list"}
                # )
                # msg = await websocket.receive_json()
                # assert msg["success"], msg
                # floors = {
                #     floor_info["floor_id"]: floor_info for floor_info in msg["result"]
                # }

                # States
                # await websocket.send_json({"id": next_id(), "type": "get_states"})
                # msg = await websocket.receive_json()
                # assert msg["success"], msg
                # states = {state["entity_id"]: state for state in msg["result"]}

                # Media players
                await websocket.send_json(
                    {"id": next_id(), "type": "config/entity_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                media_players = {
                    mp_info["entity_id"]: mp_info
                    for mp_info in msg["result"]
                    if mp_info["entity_id"].startswith("media_player.")
                    and (mp_info.get("disabled_by") is None)
                }

                # Add area/floor info to media players
                for mp_id, mp_info in media_players.items():
                    mp_area_id = mp_info.get("area_id")
                    mp_floor_id: Optional[str] = None
                    if not mp_area_id:
                        mp_device_id = mp_info.get("device_id")
                        if mp_device_id:
                            mp_area_id = devices.get(mp_device_id, {}).get("area_id")

                    if mp_area_id:
                        mp_info["area_id"] = mp_area_id
                        mp_floor_id = areas.get(mp_area_id, {}).get("floor_id")
                        if mp_floor_id:
                            mp_info["floor_id"] = mp_floor_id

                # Music players (that support SEARCH_MEDIA for Music Assistant)
                music_players: Dict[str, Dict[str, Any]] = {}
                for mp_id, mp_info in media_players.items():
                    if mp_info.get("platform") != "music_assistant":
                        continue

                    if not mp_info.get("config_entry_id"):
                        continue

                    music_players[mp_id] = mp_info

                if satellite_info.entity_id:
                    # Get area of assist_satellite entity
                    await websocket.send_json(
                        {
                            "id": next_id(),
                            "type": "config/entity_registry/get_entries",
                            "entity_ids": [satellite_info.entity_id],
                        }
                    )
                    msg = await websocket.receive_json()
                    assert msg["success"], msg
                    satellite_dict: Dict[str, Any] = next(
                        iter(msg["result"].values()), {}
                    )
                    _apply_satellite_registry_info(
                        satellite_info, satellite_dict, devices
                    )

                if satellite_info.device_id:
                    # Look for media/music player on the same device
                    for mp_id, mp_info in media_players.items():
                        if mp_info.get("device_id") != satellite_info.device_id:
                            continue

                        if not satellite_info.media_player_id:
                            satellite_info.media_player_id = mp_id
                            _LOGGER.debug("Selected media player by device: %s", mp_id)

                        if (not satellite_info.music_player_id) and (
                            mp_id in music_players
                        ):
                            satellite_info.music_player_id = mp_id
                            satellite_info.music_assistant_id = mp_info.get(
                                "config_entry_id"
                            )
                            _LOGGER.debug("Selected music player by device: %s", mp_id)

                if satellite_info.area_id:
                    satellite_info.floor_id = areas.get(satellite_info.area_id, {}).get(
                        "floor_id"
                    )

                # Look for media/music player in the same area
                if (
                    (not satellite_info.media_player_id)
                    or (not satellite_info.music_player_id)
                ) and satellite_info.area_id:
                    for mp_id, mp_info in media_players.items():
                        if mp_info.get("area_id") != satellite_info.area_id:
                            continue

                        if not satellite_info.media_player_id:
                            satellite_info.media_player_id = mp_id
                            _LOGGER.debug("Selected media player by area: %s", mp_id)

                        if (not satellite_info.music_player_id) and (
                            mp_id in music_players
                        ):
                            satellite_info.music_player_id = mp_id
                            satellite_info.music_assistant_id = mp_info.get(
                                "config_entry_id"
                            )
                            _LOGGER.debug("Selected music player by area: %s", mp_id)

                # Look for media/music player on the same floor
                if (
                    (not satellite_info.media_player_id)
                    or (not satellite_info.music_player_id)
                ) and satellite_info.floor_id:
                    for mp_id, mp_info in media_players.items():
                        if mp_info.get("floor_id") != satellite_info.floor_id:
                            continue

                        if not satellite_info.media_player_id:
                            satellite_info.media_player_id = mp_id
                            _LOGGER.debug("Selected media player by floor: %s", mp_id)

                        if (not satellite_info.music_player_id) and (
                            mp_id in music_players
                        ):
                            satellite_info.music_player_id = mp_id
                            satellite_info.music_assistant_id = mp_info.get(
                                "config_entry_id"
                            )
                            _LOGGER.debug("Selected music player by floor: %s", mp_id)

        return satellite_info

    async def _get_blueprint_descriptions(
        self, websocket: aiohttp.ClientWebSocketResponse, next_id: Callable[[], int]
    ) -> Dict[str, str]:
        """Blueprint path -> description, for every script blueprint."""
        await websocket.send_json(
            {"id": next_id(), "type": "blueprint/list", "domain": "script"}
        )
        msg = await websocket.receive_json()
        if not msg.get("success"):
            _LOGGER.debug("Could not list script blueprints: %s", msg.get("error"))
            return {}

        descriptions: Dict[str, str] = {}
        for path, blueprint_info in (msg.get("result") or {}).items():
            metadata = (blueprint_info or {}).get("metadata") or {}
            description = (metadata.get("description") or "").strip()
            if description:
                descriptions[path] = description

        return descriptions

    async def _get_script_configs(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        next_id: Callable[[], int],
        script_ids: Set[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Get the full configuration exposed by each script entity."""
        configs: Dict[str, Dict[str, Any]] = {}
        for script_id in sorted(script_ids):
            await websocket.send_json(
                {
                    "id": next_id(),
                    "type": "script/config",
                    "entity_id": f"script.{script_id}",
                }
            )
            msg = await websocket.receive_json()
            if not msg.get("success"):
                error = msg.get("error") or {}
                _LOGGER.debug(
                    "Could not get config for script %s: %s",
                    script_id,
                    error,
                )
                if error.get("code") == "unknown_command":
                    break

                continue

            config = (msg.get("result") or {}).get("config")
            if config:
                configs[script_id] = config

        return configs

    async def _get_script_blueprint_paths(
        self, session: aiohttp.ClientSession, script_ids: Set[str]
    ) -> Dict[str, str]:
        """Script id -> blueprint path, for scripts built from a blueprint.

        The blueprint path is only in the script's stored config, which is
        REST-only. Scripts whose config cannot be read are skipped: the only
        cost is a missing description.
        """
        paths: Dict[str, str] = {}
        headers = {"Authorization": f"Bearer {self.token}"}
        for script_id in sorted(script_ids):
            url = f"{self.api_url}/config/script/config/{script_id}"
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        _LOGGER.debug(
                            "No stored config for script %s (HTTP %s)",
                            script_id,
                            response.status,
                        )
                        continue

                    config = await response.json()
            except (aiohttp.ClientError, ValueError) as err:
                _LOGGER.debug("Could not read config for script %s: %s", script_id, err)
                continue

            path = ((config or {}).get("use_blueprint") or {}).get("path")
            if path:
                paths[script_id] = path

        return paths

    async def get_script_tools(
        self,
        info: HomeAssistantInfo,
        name_overrides: Optional[NameOverrides] = None,
    ) -> List[Tool]:
        """Build one tool per Home Assistant script.

        Every script gets a tool so the web UI can show what targeting it would
        get; ``Tool.exposed`` says whether it is exposed to voice. Only exposed
        tools are given to the model.

        ``name_overrides`` replaces the names of individual entities, areas, and
        floors, so the enums can use what people actually say rather than only
        what Home Assistant calls things.
        """
        tools: List[Tool] = []
        names = name_overrides or NameOverrides()

        # Every area and floor, with the names Home Assistant gives it.
        area_targets = [
            (area.area_id, list(area.names)) for area in info.areas.values()
        ]
        floor_targets = [
            (floor.floor_id, list(floor.names)) for floor in info.floors.values()
        ]

        def build_name_map(
            kind: str,
            targets: List[Tuple[str, List[str]]],
            script_id: Optional[str] = None,
            field_key: Optional[str] = None,
        ) -> Dict[str, List[str]]:
            """Display name -> the ids it refers to.

            A name can belong to more than one thing (two areas called "Office",
            several media players all called "Media Player"), so every candidate
            is kept and the resolver picks between them.
            """
            name_map: Dict[str, List[str]] = defaultdict(list)
            for target_id, default_names in targets:
                for target_name in names.names_for(
                    kind, target_id, default_names, script_id, field_key
                ):
                    name_map[target_name].append(target_id)

            return dict(name_map)

        # Shared maps for fields with no per-script override, which is most of
        # them: rebuilding these per field would be wasted work.
        area_name_map = build_name_map(AREA, area_targets)
        floor_name_map = build_name_map(FLOOR, floor_targets)

        # ---
        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as session:
            async with session.ws_connect(
                self.websocket_api_url,
                max_msg_size=0,
                receive_timeout=WEBSOCKET_RECEIVE_TIMEOUT,
            ) as websocket:
                # Authenticate
                msg = await websocket.receive_json()
                assert msg["type"] == "auth_required", msg

                await websocket.send_json(
                    {
                        "type": "auth",
                        "access_token": self.token,
                    },
                )

                msg = await websocket.receive_json()
                assert msg["type"] == "auth_ok", msg

                # Get exposed entities
                await websocket.send_json(
                    {"id": next_id(), "type": "homeassistant/expose_entity/list"}
                )

                msg = await websocket.receive_json()
                assert msg["success"], msg

                exposed_scripts = set()
                for entity_id, exposed_info in msg["result"][
                    "exposed_entities"
                ].items():
                    domain, script_id = entity_id.split(".", maxsplit=1)
                    if domain != "script":
                        continue

                    if not exposed_info.get("conversation"):
                        continue

                    exposed_scripts.add(script_id)

                # Script entities are the authoritative list of scripts. The
                # script domain's own services (turn_on, toggle, reload, ...)
                # also appear in get_services and are not scripts at all.
                await websocket.send_json({"id": next_id(), "type": "get_states"})
                msg = await websocket.receive_json()
                assert msg["success"], msg

                script_names: Dict[str, str] = {}
                for state_data in msg["result"]:
                    entity_id = state_data["entity_id"]
                    domain, _, script_id = entity_id.partition(".")
                    if domain != "script":
                        continue

                    friendly_name = state_data.get("attributes", {}).get(
                        ATTR_FRIENDLY_NAME
                    )
                    script_names[script_id] = (friendly_name or "").strip()

                await websocket.send_json(
                    {
                        "id": next_id(),
                        "type": "get_services",
                    }
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg

                scripts = msg["result"].get("script") or {}

                # Newer Home Assistant versions no longer include individual
                # scripts in get_services. script/config is the authoritative
                # source for aliases, descriptions, and fields.
                script_configs = await self._get_script_configs(
                    websocket, next_id, set(script_names)
                )
                for script_id, config in script_configs.items():
                    script = dict(scripts.get(script_id) or {})
                    if config.get("alias"):
                        script["name"] = config["alias"]
                    for config_key in ("description", "fields"):
                        if config_key in config:
                            script[config_key] = config[config_key]
                    scripts[script_id] = script

                # A script built from a blueprint reports an empty description,
                # so fall back to the blueprint's own description. Without this
                # the model only sees the script's name.
                missing_description = {
                    script_id
                    for script_id in script_names
                    if not (
                        (scripts.get(script_id) or {}).get("description") or ""
                    ).strip()
                }
                blueprint_description: Dict[str, str] = {}
                if missing_description:
                    blueprint_descriptions = await self._get_blueprint_descriptions(
                        websocket, next_id
                    )
                    if blueprint_descriptions:
                        blueprint_paths = await self._get_script_blueprint_paths(
                            session, missing_description
                        )
                        blueprint_description = {
                            script_id: blueprint_descriptions[path]
                            for script_id, path in blueprint_paths.items()
                            if path in blueprint_descriptions
                        }
                        _LOGGER.debug(
                            "Using blueprint descriptions for %s script(s)",
                            len(blueprint_description),
                        )

                for script_id in sorted(script_names):
                    # A script always has a service, but tolerate it missing
                    # rather than dropping the script from the UI entirely.
                    script = scripts.get(script_id) or {}
                    friendly_name = (
                        script.get("name") or script_names[script_id] or script_id
                    )
                    description = (script.get("description") or "").strip()
                    if not description:
                        description = blueprint_description.get(script_id, "")

                    tool_dict: Dict[str, Any] = {
                        "type": "function",
                        "function": {
                            "name": script_id,
                            # The model sees the description, falling back to the
                            # name when the script has none.
                            "description": description or friendly_name,
                        },
                    }
                    tool = Tool(
                        name=script_id,
                        tool=tool_dict,
                        friendly_name=friendly_name,
                        description=description,
                        exposed=script_id in exposed_scripts,
                    )
                    tools.append(tool)

                    fields = script.get("fields")
                    if not fields:
                        continue

                    # Convert fields to tools
                    props: Dict[str, Any] = {}
                    params = tool_dict["function"].setdefault(
                        "parameters", {"type": "object", "properties": props}
                    )
                    required_fields = set()
                    for field_key, field_info in fields.items():
                        if field_info.get("required"):
                            required_fields.add(field_key)

                        field_description = field_info.get("description") or ""

                        # Default to string type
                        field_prop: Dict[str, Any] = {"type": "string"}
                        props[field_key] = field_prop

                        selector = field_info.get("selector")
                        if selector:
                            selector_type = next(iter(selector), "")
                            selector_config = selector.get(selector_type) or {}
                            if "text" in selector:
                                pass  # already a string
                            elif "number" in selector:
                                field_prop.update(_number_property(selector["number"]))
                            elif "boolean" in selector:
                                field_prop["type"] = "boolean"
                            elif "select" in selector:
                                field_prop.update(_select_property(selector_config))
                            elif "date" in selector:
                                field_prop["format"] = "date"
                                field_description += (
                                    "\nDate in YYYY-MM-DD format. Work out the date "
                                    "from the current date, which is given with the "
                                    "sentence."
                                )
                            elif "time" in selector:
                                field_prop["format"] = "time"
                                field_description += "\nTime in HH:MM:SS format, or HH:MM if seconds are not needed"
                            elif "datetime" in selector:
                                field_prop["format"] = "date-time"
                                field_description += (
                                    "\nISO 8601 datetime in YYYY-MM-DDTHH:MM:SS format. "
                                    "Work out the date from the current date, which is "
                                    "given with the sentence. Never name a weekday or "
                                    "use a word like tomorrow."
                                )
                            elif "duration" in selector:
                                field_prop["format"] = "duration"
                                # Always all three parts: Home Assistant reads a
                                # two-part duration as MM:SS, so "01:00" would
                                # quietly mean one minute rather than one hour.
                                field_description += (
                                    "\nDuration in HH:MM:SS format, including the "
                                    "hours even when they are zero"
                                )
                            elif "color_rgb" in selector:
                                field_prop["type"] = "array"
                                field_prop["items"] = {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 255,
                                }
                                field_prop["minItems"] = 3
                                field_prop["maxItems"] = 3
                                field_description += (
                                    "\nRGB color as [red, green, blue], each 0-255."
                                )
                            elif "color_temp" in selector:
                                field_prop["type"] = "integer"
                                field_prop["minimum"] = 2000
                                field_prop["maximum"] = 6500
                                field_description += "\nColor temperature in kelvin, e.g. 2700 for warm white or 6500 for cool daylight."
                            elif ("area" in selector) and area_targets:
                                scoped = names.has_field_names(script_id, field_key)
                                field_map = (
                                    build_name_map(
                                        AREA, area_targets, script_id, field_key
                                    )
                                    if scoped
                                    else area_name_map
                                )
                                field_prop["enum"] = sorted(field_map)
                                tool.name_map[field_key] = dict(field_map)
                                tool.field_kinds[field_key] = AREA
                                tool.field_targets[field_key] = [
                                    target_id for target_id, _ in area_targets
                                ]
                            elif ("floor" in selector) and floor_targets:
                                scoped = names.has_field_names(script_id, field_key)
                                field_map = (
                                    build_name_map(
                                        FLOOR, floor_targets, script_id, field_key
                                    )
                                    if scoped
                                    else floor_name_map
                                )
                                field_prop["enum"] = sorted(field_map)
                                tool.name_map[field_key] = dict(field_map)
                                tool.field_kinds[field_key] = FLOOR
                                tool.field_targets[field_key] = [
                                    target_id for target_id, _ in floor_targets
                                ]
                            elif ("entity" in selector) and info.entities:
                                filter_domains = _get_entity_filter_domains(
                                    selector["entity"]
                                )

                                if filter_domains:
                                    entity_targets = [
                                        (entity.entity_id, list(entity.names))
                                        for entity in info.entities.values()
                                        if entity.domain in filter_domains
                                    ]
                                else:
                                    entity_targets = [
                                        (entity.entity_id, list(entity.names))
                                        for entity in info.entities.values()
                                    ]
                                field_map = build_name_map(
                                    ENTITY, entity_targets, script_id, field_key
                                )
                                field_prop["enum"] = sorted(field_map)
                                tool.name_map[field_key] = dict(field_map)
                                tool.field_kinds[field_key] = ENTITY
                                tool.field_targets[field_key] = [
                                    target_id for target_id, _ in entity_targets
                                ]

                            if (
                                selector_type in MULTIPLE_SELECTORS
                                and selector_config.get("multiple")
                            ):
                                field_prop = _multiple_property(field_prop)
                                props[field_key] = field_prop

                        if field_description:
                            field_prop["description"] = field_description.strip()

                    if required_fields:
                        params["required"] = sorted(required_fields)

        return tools

    async def get_home_info(self) -> HomeAssistantInfo:
        """Get necessary information for intent recognition."""
        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        states: Dict[str, State] = {}
        entities: Dict[str, Entity] = {}
        areas: Dict[str, Area] = {}
        floors: Dict[str, Floor] = {}
        satellites: Dict[str, str] = {}

        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as session:
            async with session.ws_connect(
                self.websocket_api_url,
                max_msg_size=0,
                receive_timeout=WEBSOCKET_RECEIVE_TIMEOUT,
            ) as websocket:
                # Authenticate
                msg = await websocket.receive_json()
                assert msg["type"] == "auth_required", msg

                await websocket.send_json(
                    {
                        "type": "auth",
                        "access_token": self.token,
                    },
                )

                msg = await websocket.receive_json()
                assert msg["type"] == "auth_ok", msg

                # Get exposed entities
                await websocket.send_json(
                    {"id": next_id(), "type": "homeassistant/expose_entity/list"}
                )

                msg = await websocket.receive_json()
                assert msg["success"], msg

                exposed_entity_ids = set()
                for entity_id, exposed_info in msg["result"][
                    "exposed_entities"
                ].items():
                    if exposed_info.get("conversation"):
                        exposed_entity_ids.add(entity_id)

                await websocket.send_json(
                    {
                        "id": next_id(),
                        "type": "get_states",
                    }
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                for state_data in msg["result"]:
                    entity_id = state_data["entity_id"]
                    state_attributes = state_data.get("attributes", {})
                    if entity_id.startswith("assist_satellite."):
                        satellites[entity_id] = str(
                            state_attributes.get(ATTR_FRIENDLY_NAME) or entity_id
                        ).strip()
                    if entity_id not in exposed_entity_ids:
                        continue

                    states[entity_id] = State(
                        entity_id=entity_id,
                        state=state_data["state"],
                        attributes=state_attributes,
                    )

                # Floors
                await websocket.send_json(
                    {"id": next_id(), "type": "config/floor_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                for floor_data in msg["result"]:
                    floor_id = floor_data["floor_id"]
                    floors[floor_id] = Floor(
                        floor_id=floor_id,
                        name=floor_data["name"].strip(),
                        aliases=floor_data.get("aliases"),
                    )

                # Areas
                await websocket.send_json(
                    {"id": next_id(), "type": "config/area_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                for area_data in msg["result"]:
                    area_id = area_data["area_id"]
                    areas[area_id] = Area(
                        area_id=area_id,
                        name=area_data["name"].strip(),
                        aliases=area_data.get("aliases"),
                        floor_id=area_data.get("floor_id"),
                    )

                # Devices
                await websocket.send_json(
                    {"id": next_id(), "type": "config/device_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                devices = {
                    device_info["id"]: device_info for device_info in msg["result"]
                }

                # Contains aliases
                # Check area_id as well as area of device_id
                # Use original_device_class
                await websocket.send_json(
                    {
                        "id": next_id(),
                        "type": "config/entity_registry/get_entries",
                        "entity_ids": list(exposed_entity_ids),
                    }
                )

                msg = await websocket.receive_json()
                assert msg["success"], msg
                for entity_id, entity_info in msg["result"].items():
                    name = None
                    names: List[str] = []

                    if entity_info:
                        if entity_info.get("disabled_by") is not None:
                            # Skip disabled entities
                            continue

                        name = (
                            entity_info.get("name", "") or entity_info["original_name"]
                        )
                        if entity_info.get("aliases"):
                            names.extend(filter(None, entity_info["aliases"]))

                    entity_area_id = None
                    if entity_info:
                        entity_area_id = entity_info.get("area_id")

                        if not entity_area_id:
                            # Try to get area from device
                            entity_device_id = entity_info.get("device_id")
                            if entity_device_id:
                                device_info = devices.get(entity_device_id)
                                if device_info:
                                    entity_area_id = device_info.get("area_id")

                    attributes: Dict[str, Any] = {}
                    state_data = states.get(entity_id)
                    if state_data:
                        attributes = state_data.attributes

                    if not name:
                        # Try friendly name
                        name = attributes.get(ATTR_FRIENDLY_NAME, "")

                    if name:
                        name = name.strip()
                        if state_data:
                            state_data.entity_name = name

                    entities[entity_id] = Entity(
                        entity_id=entity_id,
                        name=name,
                        aliases=names if names else None,
                        attributes=attributes,
                        area_id=entity_area_id,
                    )

        _LOGGER.debug(
            "Loaded %s entities, %s area(s), %s floor(s), %s satellite(s)",
            len(entities),
            len(areas),
            len(floors),
            len(satellites),
        )

        return HomeAssistantInfo(
            states=states,
            entities=entities,
            areas=areas,
            floors=floors,
            satellites=satellites,
        )

    async def call_service(
        self,
        domain: str,
        service: str,
        service_data: Optional[Dict[str, Any]] = None,
        target: Optional[Dict[str, Any]] = None,
        return_response: bool = False,
    ) -> Optional[Dict[str, Any]]:
        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as session:
            async with session.ws_connect(
                self.websocket_api_url,
                max_msg_size=0,
                receive_timeout=WEBSOCKET_RECEIVE_TIMEOUT,
            ) as websocket:
                # Authenticate
                msg = await websocket.receive_json()
                assert msg["type"] == "auth_required", msg

                await websocket.send_json(
                    {
                        "type": "auth",
                        "access_token": self.token,
                    },
                )

                msg = await websocket.receive_json()
                assert msg["type"] == "auth_ok", msg

                _LOGGER.debug(
                    "Calling service %s.%s with target=%s, data=%s",
                    domain,
                    service,
                    target,
                    service_data,
                )

                await websocket.send_json(
                    {
                        "id": next_id(),
                        "type": "call_service",
                        "domain": domain,
                        "service": service,
                        "service_data": service_data or {},
                        "target": target or {},
                        "return_response": return_response,
                    },
                )
                msg = await websocket.receive_json()
                if not msg["success"]:
                    raise HomeAssistantError(msg["error"]["message"])

                return msg.get("result", {}).get("response")
