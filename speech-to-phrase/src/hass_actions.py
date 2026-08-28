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
    api_url: str,
    token: str,
    domain: str,
    service: str,
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
            await ws.send_json(
                {
                    "id": 1,
                    "type": "call_service",
                    "domain": domain,
                    "service": service,
                    "service_data": data or {},
                    "target": target or {},
                }
            )
            resp = await ws.receive_json()
            ok = bool(resp.get("success"))
            if not ok:
                _LOGGER.warning(
                    "call_service %s.%s failed: %s", domain, service, resp.get("error")
                )
            return ok


async def render_template_async(
    api_url: str,
    token: str,
    template: str,
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
                "id": 1,
                "type": "render_template",
                "template": template,
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
            await ws.send_json(
                {"id": 2, "type": "unsubscribe_events", "subscription": 1}
            )
            return result


async def handle_intent_async(
    api_url: str,
    token: str,
    name: str,
    slots: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    """Execute a standard intent in HA via ``POST /api/intent/handle``.

    Returns ``(ok, speech)`` -- ``speech`` is HA's spoken response text (or an
    error message when ``ok`` is False)."""
    import aiohttp

    url = api_url.rstrip("/") + "/intent/handle"
    data = {k: v for k, v in (slots or {}).items() if v is not None}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json={"name": name, "data": data},
            headers={"Authorization": f"Bearer {token}"},
        ) as resp:
            try:
                body = await resp.json(content_type=None)
            except Exception:  # noqa: BLE001
                body = None
            if resp.status != 200:
                msg = body.get("message") if isinstance(body, dict) else None
                _LOGGER.warning(
                    "intent/handle %s -> HTTP %s: %s", name, resp.status, msg
                )
                return False, msg or f"HTTP {resp.status}"
            speech = ""
            if isinstance(body, dict):
                speech = (
                    ((body.get("speech") or {}).get("plain") or {}).get("speech")
                ) or ""
                if body.get("response_type") == "error":
                    return False, speech or "intent error"
            return True, speech


async def run_action_async(
    api_url: str,
    token: str,
    action: Dict[str, Any],
    slots: Optional[Dict[str, Any]] = None,
) -> bool:
    """Perform a custom "action" command in HA: run a script/scene, or a service
    call from a YAML block (slot templates rendered in HA). Returns True on
    success."""
    import yaml

    kind = (action or {}).get("kind")
    if kind in ("script", "scene"):
        entity_id = action.get("entity_id")
        if not entity_id or "." not in entity_id:
            _LOGGER.warning("action %s missing entity_id", kind)
            return False
        domain = entity_id.split(".", 1)[0]
        # Scripts can take slots as `variables`; scenes cannot.
        data = {"variables": slots} if (domain == "script" and slots) else {}
        return await call_service_async(
            api_url,
            token,
            domain,
            "turn_on",
            data=data,
            target={"entity_id": entity_id},
        )
    if kind == "service":
        yaml_text = action.get("yaml") or ""
        # Render slot templates in the YAML (in HA), then parse + call.
        rendered = await render_template_async(
            api_url,
            token,
            yaml_text,
            variables={"slots": slots or {}},
        )
        spec = yaml.safe_load(rendered or yaml_text) or {}
        service = spec.get("service") or spec.get("action")
        if not service or "." not in service:
            _LOGGER.warning("custom service action missing 'service:' (%r)", service)
            return False
        domain, svc = service.split(".", 1)
        return await call_service_async(
            api_url,
            token,
            domain,
            svc,
            data=spec.get("data") or {},
            target=spec.get("target") or {},
        )
    _LOGGER.warning("unknown action kind: %r", kind)
    return False


async def exposed_scripts_scenes_async(
    api_url: str,
    token: str,
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
                    eid
                    for eid, info in (
                        em.get("result", {}).get("exposed_entities", {})
                        if em.get("success")
                        else {}
                    ).items()
                    if info.get("conversation")
                    and eid.split(".", 1)[0] in ("script", "scene")
                ]
                if not exposed:
                    return out
                await ws.send_json(
                    {
                        "id": 2,
                        "type": "config/entity_registry/get_entries",
                        "entity_ids": exposed,
                    }
                )
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
