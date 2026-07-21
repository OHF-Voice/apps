"""Home Assistant websocket helpers for the custom-*action* command path.

When a custom command is an "action" (not a standard intent), the intent server
performs it itself and returns a ``Handled`` response:

  1. run it -- either ``script.turn_on`` / ``scene.turn_on`` for an exposed
     script/scene, or an arbitrary service call from a YAML block, and
  2. render the response template **in HA** (so ``{{ states(...) }}`` etc. work),
     injecting the matched ``slots`` as template variables.

All best-effort and async (called from the intent server's event loop). The
exposed script/scene fetch is also here for the UI's action picker.
"""
import logging
from typing import Any, Dict, List, Optional

from training import _ws_url

_LOGGER = logging.getLogger("speech-to-phrase.intent")


async def call_service_async(
    api_url: str, token: str, domain: str, service: str,
    data: Optional[Dict[str, Any]] = None,
    target: Optional[Dict[str, Any]] = None,
) -> bool:
    """Call an HA service. Returns True on success."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(_ws_url(api_url), max_msg_size=0) as ws:
            assert (await ws.receive_json())["type"] == "auth_required"
            await ws.send_json({"type": "auth", "access_token": token})
            assert (await ws.receive_json())["type"] == "auth_ok"
            await ws.send_json({
                "id": 1, "type": "call_service",
                "domain": domain, "service": service,
                "service_data": data or {}, "target": target or {},
            })
            resp = await ws.receive_json()
            ok = bool(resp.get("success"))
            if not ok:
                _LOGGER.warning("call_service %s.%s failed: %s",
                                domain, service, resp.get("error"))
            return ok


async def render_template_async(
    api_url: str, token: str, template: str,
    variables: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Render a Jinja2 template in HA and return the first result.

    ``render_template`` is a subscription that re-renders on state change; we
    take the first ``event`` result and unsubscribe.
    """
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(_ws_url(api_url), max_msg_size=0) as ws:
            assert (await ws.receive_json())["type"] == "auth_required"
            await ws.send_json({"type": "auth", "access_token": token})
            assert (await ws.receive_json())["type"] == "auth_ok"
            msg: Dict[str, Any] = {
                "id": 1, "type": "render_template", "template": template,
            }
            if variables:
                msg["variables"] = variables
            await ws.send_json(msg)
            # First a {result, success} ack, then {type: "event", event:{result}}.
            result: Optional[str] = None
            for _ in range(10):
                resp = await ws.receive_json()
                if resp.get("type") == "result" and not resp.get("success", True):
                    _LOGGER.warning("render_template failed: %s", resp.get("error"))
                    break
                if resp.get("type") == "event":
                    result = resp["event"].get("result")
                    break
            await ws.send_json({"id": 2, "type": "unsubscribe_events",
                                "subscription": 1})
            return result


async def exposed_scripts_scenes_async(
    api_url: str, token: str,
) -> List[Dict[str, str]]:
    """[{entity_id, name}] for conversation-exposed script.* / scene.* entities,
    for the action picker. Best-effort; returns [] on failure."""
    import aiohttp

    out: List[Dict[str, str]] = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(_ws_url(api_url), max_msg_size=0) as ws:
                assert (await ws.receive_json())["type"] == "auth_required"
                await ws.send_json({"type": "auth", "access_token": token})
                assert (await ws.receive_json())["type"] == "auth_ok"
                await ws.send_json(
                    {"id": 1, "type": "homeassistant/expose_entity/list"}
                )
                em = await ws.receive_json()
                exposed = [
                    eid for eid, info in (
                        em.get("result", {}).get("exposed_entities", {}) if em.get("success") else {}
                    ).items()
                    if info.get("conversation")
                    and eid.split(".", 1)[0] in ("script", "scene")
                ]
                if not exposed:
                    return out
                await ws.send_json({"id": 2, "type": "config/entity_registry/get_entries",
                                    "entity_ids": exposed})
                rm = await ws.receive_json()
                entries = rm["result"] if rm.get("success") else {}
        for eid in exposed:
            info = entries.get(eid) or {}
            name = info.get("name") or info.get("original_name") or eid
            out.append({"entity_id": eid, "name": name})
    except Exception:  # noqa: BLE001
        _LOGGER.debug("script/scene fetch failed", exc_info=True)
    return sorted(out, key=lambda x: x["name"].lower())


def exposed_scripts_scenes(api_url: str, token: str) -> List[Dict[str, str]]:
    import asyncio

    if not token:
        return []
    try:
        return asyncio.run(exposed_scripts_scenes_async(api_url, token))
    except Exception:  # noqa: BLE001
        _LOGGER.debug("script/scene fetch failed", exc_info=True)
        return []
