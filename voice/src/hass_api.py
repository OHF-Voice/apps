"""Wrapper for Home Assistant REST/Websocket API."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import aiohttp

from models import ATTR_FRIENDLY_NAME, Area, Entity, Floor, State

_LOGGER = logging.getLogger(__name__)


class HomeAssistantError(Exception):
    pass


@dataclass
class HomeAssistantInfo:
    states: Dict[str, State]
    entities: Dict[str, Entity]
    areas: Dict[str, Area]
    floors: Dict[str, Floor]


# @dataclass
# class InfoForRecognition:
#     """Information gathered from Home Assistant for intent recognition."""

#     current_area_id: Optional[str]
#     current_area_name: Optional[str]
#     current_floor_id: Optional[str]
#     current_device_id: Optional[str]
#     current_satellite_id: Optional[str]
#     states: Dict[str, State]
#     entities: Dict[str, Entity]
#     areas: Dict[str, Area]
#     floors: Dict[str, Floor]


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

    async def get_info(self) -> HomeAssistantInfo:
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

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self.websocket_api_url, max_msg_size=0
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
                    if entity_id not in exposed_entity_ids:
                        continue

                    states[entity_id] = State(
                        entity_id=entity_id,
                        state=state_data["state"],
                        attributes=state_data.get("attributes", {}),
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
            "Loaded %s entities, %s area(s), %s floor(s)",
            len(entities),
            len(areas),
            len(floors),
        )

        return HomeAssistantInfo(
            states=states, entities=entities, areas=areas, floors=floors
        )

    async def get_current_area(
        self, device_id: Optional[str] = None, satellite_id: Optional[str] = None
    ) -> Optional[str]:
        if (device_id is None) and (satellite_id is None):
            return None

        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self.websocket_api_url, max_msg_size=0
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

                if satellite_id:
                    # Get area of assist_satellite entity
                    await websocket.send_json(
                        {
                            "id": next_id(),
                            "type": "config/entity_registry/get_entries",
                            "entity_ids": [satellite_id],
                        }
                    )
                    msg = await websocket.receive_json()
                    assert msg["success"], msg
                    satellite_info = next(iter(msg["result"].values()))
                    satellite_area_id = satellite_info.get("area_id")
                    if satellite_area_id:
                        return satellite_area_id

                    # Use device area
                    device_id = satellite_info.get("device_id", device_id)
                    if device_id:
                        return devices.get(device_id, {}).get("area_id")

                if device_id:
                    # Get area from device instead
                    return devices.get(device_id, {}).get("area_id")

        return None

    # async def get_info(
    #     self, device_id: Optional[str] = None, satellite_id: Optional[str] = None
    # ) -> InfoForRecognition:
    #     """Get necessary information for intent recognition."""
    #     current_id = 0

    #     def next_id() -> int:
    #         nonlocal current_id
    #         current_id += 1
    #         return current_id

    #     area_names: Set[str] = set()
    #     floor_names: Set[str] = set()
    #     states: Dict[str, State] = {}
    #     entities: Dict[str, Entity] = {}
    #     areas: Dict[str, Area] = {}
    #     floors: Dict[str, Floor] = {}
    #     current_area_id: Optional[str] = None
    #     current_area_name: Optional[str] = None
    #     current_floor_id: Optional[str] = None

    #     async with aiohttp.ClientSession() as session:
    #         async with session.ws_connect(
    #             self.websocket_api_url, max_msg_size=0
    #         ) as websocket:
    #             # Authenticate
    #             msg = await websocket.receive_json()
    #             assert msg["type"] == "auth_required", msg

    #             await websocket.send_json(
    #                 {
    #                     "type": "auth",
    #                     "access_token": self.token,
    #                 },
    #             )

    #             msg = await websocket.receive_json()
    #             assert msg["type"] == "auth_ok", msg

    #             # Get exposed entities
    #             await websocket.send_json(
    #                 {"id": next_id(), "type": "homeassistant/expose_entity/list"}
    #             )

    #             msg = await websocket.receive_json()
    #             assert msg["success"], msg

    #             exposed_entity_ids = set()
    #             for entity_id, exposed_info in msg["result"][
    #                 "exposed_entities"
    #             ].items():
    #                 if exposed_info.get("conversation"):
    #                     exposed_entity_ids.add(entity_id)

    #             satellite_ids = set()
    #             await websocket.send_json(
    #                 {
    #                     "id": next_id(),
    #                     "type": "get_states",
    #                 }
    #             )
    #             msg = await websocket.receive_json()
    #             assert msg["success"], msg
    #             for state_data in msg["result"]:
    #                 entity_id = state_data["entity_id"]
    #                 domain = entity_id.split(".", maxsplit=1)[0]
    #                 if domain == "assist_satellite":
    #                     satellite_ids.add(entity_id)

    #                 if entity_id not in exposed_entity_ids:
    #                     continue

    #                 states[entity_id] = State(
    #                     entity_id=entity_id,
    #                     state=state_data["state"],
    #                     attributes=state_data.get("attributes", {}),
    #                 )

    #             # Floors
    #             await websocket.send_json(
    #                 {"id": next_id(), "type": "config/floor_registry/list"}
    #             )
    #             msg = await websocket.receive_json()
    #             assert msg["success"], msg
    #             for floor_data in msg["result"]:
    #                 floor_id = floor_data["floor_id"]
    #                 floors[floor_id] = Floor(
    #                     floor_id=floor_id,
    #                     name=floor_data["name"],
    #                     aliases=floor_data.get("aliases"),
    #                 )
    #                 names = [floor_data["name"]]
    #                 if floor_data.get("aliases"):
    #                     names.extend(floor_data["aliases"])
    #                 for name in names:
    #                     name = name.strip()
    #                     if name:
    #                         floor_names.add(name)

    #             # Areas
    #             await websocket.send_json(
    #                 {"id": next_id(), "type": "config/area_registry/list"}
    #             )
    #             msg = await websocket.receive_json()
    #             assert msg["success"], msg
    #             for area_data in msg["result"]:
    #                 area_id = area_data["area_id"]
    #                 areas[area_id] = Area(
    #                     area_id=area_id,
    #                     name=area_data["name"],
    #                     aliases=area_data.get("aliases"),
    #                     floor_id=area_data.get("floor_id"),
    #                 )
    #                 names = [area_data["name"]]
    #                 if area_data.get("aliases"):
    #                     names.extend(area_data["aliases"])
    #                 for name in names:
    #                     name = name.strip()
    #                     if name:
    #                         area_names.add(name)

    #             # Devices
    #             await websocket.send_json(
    #                 {"id": next_id(), "type": "config/device_registry/list"}
    #             )
    #             msg = await websocket.receive_json()
    #             assert msg["success"], msg
    #             devices = {
    #                 device_info["id"]: device_info for device_info in msg["result"]
    #             }

    #             # Contains aliases
    #             # Check area_id as well as area of device_id
    #             # Use original_device_class
    #             await websocket.send_json(
    #                 {
    #                     "id": next_id(),
    #                     "type": "config/entity_registry/get_entries",
    #                     "entity_ids": list(exposed_entity_ids),
    #                 }
    #             )

    #             msg = await websocket.receive_json()
    #             assert msg["success"], msg
    #             for entity_id, entity_info in msg["result"].items():
    #                 domain = entity_id.split(".")[0]
    #                 name = None
    #                 names = []

    #                 if entity_info:
    #                     if entity_info.get("disabled_by") is not None:
    #                         # Skip disabled entities
    #                         continue

    #                     name = (
    #                         entity_info.get("name", "") or entity_info["original_name"]
    #                     )
    #                     if entity_info.get("aliases"):
    #                         names.extend(filter(None, entity_info["aliases"]))

    #                 entity_area_id = None
    #                 if entity_info:
    #                     entity_area_id = entity_info.get("area_id")

    #                     if not entity_area_id:
    #                         # Try to get area from device
    #                         entity_device_id = entity_info.get("device_id")
    #                         if entity_device_id:
    #                             device_info = devices.get(entity_device_id)
    #                             if device_info:
    #                                 entity_area_id = device_info.get("area_id")

    #                 attributes: Dict[str, Any] = {}
    #                 state_data = states.get(entity_id)
    #                 if state_data:
    #                     attributes = state_data.attributes

    #                 if not name:
    #                     # Try friendly name
    #                     name = attributes.get(ATTR_FRIENDLY_NAME, "")

    #                 if name:
    #                     names.append(name)
    #                     if state_data:
    #                         state_data.entity_name = name

    #                 entities[entity_id] = Entity(
    #                     entity_id=entity_id,
    #                     name=name,
    #                     aliases=names if names else None,
    #                     attributes=attributes,
    #                     area_id=entity_area_id,
    #                 )

    #             # Get preferred area
    #             if satellite_id:
    #                 # Get area of assist_satellite entity
    #                 await websocket.send_json(
    #                     {
    #                         "id": next_id(),
    #                         "type": "config/entity_registry/get_entries",
    #                         "entity_ids": [satellite_id],
    #                     }
    #                 )
    #                 msg = await websocket.receive_json()
    #                 assert msg["success"], msg
    #                 satellite_info = next(iter(msg["result"].values()))
    #                 satellite_area_id = satellite_info.get("area_id")
    #                 if satellite_area_id:
    #                     current_area_id = satellite_area_id
    #                 else:
    #                     # Use device area
    #                     device_id = satellite_info.get("device_id", device_id)
    #                     if device_id:
    #                         current_area_id = devices.get(device_id, {}).get("area_id")
    #             elif device_id:
    #                 # Get area from device instead
    #                 current_area_id = devices.get(device_id, {}).get("area_id")

    #             if current_area_id:
    #                 current_area_info = areas.get(current_area_id)
    #                 if current_area_info:
    #                     current_area_name = current_area_info.name
    #                     current_floor_id = current_area_info.floor_id

    #     _LOGGER.debug(
    #         "Loaded %s entities, %s area(s), %s floor(s)",
    #         len(entities),
    #         len(areas),
    #         len(floors),
    #     )

    #     return InfoForRecognition(
    #         current_area_id=current_area_id,
    #         current_area_name=current_area_name,
    #         current_floor_id=current_floor_id,
    #         current_device_id=device_id,
    #         current_satellite_id=satellite_id,
    #         states=states,
    #         entities=entities,
    #         areas=areas,
    #         floors=floors,
    #     )

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

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self.websocket_api_url, max_msg_size=0
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
