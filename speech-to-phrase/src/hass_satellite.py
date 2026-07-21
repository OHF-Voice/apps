"""Resolve the voice satellite's area from a Wyoming ``Recognize`` context.

Home Assistant's Wyoming conversation agent puts ``device_id`` and/or
``satellite_id`` (an ``assist_satellite.*`` entity id) into the ``Recognize``
event's ``context`` (see homeassistant/components/wyoming/conversation.py). For
``context_area`` commands ("turn on the lights in here") we need the *area name*
to send as the ``area`` slot, so HA scopes the command to the right room.

This mirrors default-agent's hass_api.get_info area resolution, trimmed to just
what we need: satellite entity area -> its device's area -> device area, then the
area registry for the name. Best-effort: returns ``None`` on any failure, in
which case the caller simply omits the area slot.
"""
import logging
from typing import Dict, Optional

from training import _ws_url

_LOGGER = logging.getLogger("speech-to-phrase.intent")


async def resolve_area_async(
    api_url: str, token: str, *,
    device_id: Optional[str] = None, satellite_id: Optional[str] = None,
) -> Optional[str]:
    """Async area-name resolution, for callers already in an event loop."""
    if not token or (not device_id and not satellite_id):
        return None
    try:
        return await _resolve_area(
            api_url, token, device_id=device_id, satellite_id=satellite_id
        )
    except Exception:  # noqa: BLE001
        _LOGGER.debug("area resolution failed", exc_info=True)
        return None


async def _resolve_area(
    api_url: str, token: str, *,
    device_id: Optional[str], satellite_id: Optional[str],
) -> Optional[str]:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(_ws_url(api_url), max_msg_size=0) as ws:
            assert (await ws.receive_json())["type"] == "auth_required"
            await ws.send_json({"type": "auth", "access_token": token})
            assert (await ws.receive_json())["type"] == "auth_ok"

            async def cmd(msg_id: int, payload: dict):
                await ws.send_json({"id": msg_id, **payload})
                return await ws.receive_json()

            # device_id -> area_id map
            dm = await cmd(1, {"type": "config/device_registry/list"})
            devices: Dict[str, dict] = {
                d["id"]: d for d in (dm["result"] if dm.get("success") else [])
            }

            area_id: Optional[str] = None
            if satellite_id:
                em = await cmd(2, {
                    "type": "config/entity_registry/get_entries",
                    "entity_ids": [satellite_id],
                })
                entries = em["result"] if em.get("success") else {}
                info = entries.get(satellite_id) or {}
                area_id = info.get("area_id")
                if not area_id and info.get("device_id"):
                    area_id = devices.get(info["device_id"], {}).get("area_id")
            if not area_id and device_id:
                area_id = devices.get(device_id, {}).get("area_id")

            if not area_id:
                return None

            am = await cmd(3, {"type": "config/area_registry/list"})
            for area in (am["result"] if am.get("success") else []):
                if area.get("area_id") == area_id:
                    return area.get("name")
    return None


def resolve_area(
    api_url: str, token: str, *,
    device_id: Optional[str] = None, satellite_id: Optional[str] = None,
) -> Optional[str]:
    """Synchronous wrapper. Returns the area NAME or ``None``."""
    if not token or (not device_id and not satellite_id):
        return None
    import asyncio

    try:
        return asyncio.run(
            _resolve_area(api_url, token, device_id=device_id, satellite_id=satellite_id)
        )
    except Exception:  # noqa: BLE001
        _LOGGER.debug("area resolution failed", exc_info=True)
        return None
