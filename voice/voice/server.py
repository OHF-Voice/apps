"""The HTTP and websocket server Home Assistant talks to.

Three endpoints and no page: this app has no interface of its own, and
nothing about it is configured by hand.

``GET /info``     what can be verified here, and how much audio it wants
``GET /verify``   one websocket, one wake word detection, one verdict
``GET /health``   for the container health check

The guiding rule for everything below is that a verifier which is broken,
slow or confused must not make the assistant deaf. Every failure path --
an unparseable message, a decode that raises, a decode that overruns --
ends in *accepting* the detection, because the cost of wrongly accepting is
one unwanted command and the cost of wrongly refusing is a device that no
longer answers its name. Only a verifier that ran and said no refuses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from aiohttp import WSMsgType, web

from .protocol import (
    API_VERSION,
    BYTE_RATE,
    AppInfo,
    AudioEnd,
    MoreAudio,
    Verdict,
    VerifyRequest,
    bytes_to_ms,
)
from .verify import Verifier

_LOGGER = logging.getLogger(__name__)

MAX_WINDOW_SECONDS = 10.0
"""Audio a single verification will hold before it stops accepting more.

Nothing legitimate comes near it; it is here so that a client which never
sends ``audio-end`` costs a bounded amount of memory.
"""

VERIFY_TIMEOUT = 5.0
"""How long one decode may take before the detection is accepted anyway."""

VERIFIER: web.AppKey[Verifier] = web.AppKey("verifier", Verifier)
NAME: web.AppKey[str] = web.AppKey("name", str)
VERSION: web.AppKey[str] = web.AppKey("version", str)


def build_app(verifier: Verifier, name: str, version: str) -> web.Application:
    """The aiohttp application, with ``verifier`` answering on it."""
    app = web.Application()
    app[VERIFIER] = verifier
    app[NAME] = name
    app[VERSION] = version
    app.add_routes(
        [
            web.get("/info", _info),
            web.get("/health", _health),
            web.get("/verify", _verify),
        ]
    )
    return app


async def _health(request: web.Request) -> web.Response:
    return web.Response(text="ok")


async def _info(request: web.Request) -> web.Response:
    verifier = request.app[VERIFIER]
    info = AppInfo(
        name=request.app[NAME],
        version=request.app[VERSION],
        wake_words=verifier.wake_words,
        tail_ms=verifier.tail_ms,
        api_version=API_VERSION,
    )
    return web.json_response(info.to_json())


async def _verify(request: web.Request) -> web.WebSocketResponse:
    """One detection, judged.

    The connection carries a single verification and closes: audio frames
    have nowhere to put a request id, so sharing one connection would mean
    tagging every chunk, and a connection on the loopback costs a handshake.
    """
    verifier = request.app[VERIFIER]
    websocket = web.WebSocketResponse()
    await websocket.prepare(request)

    verify_request: VerifyRequest | None = None
    audio = bytearray()
    overflowed = False
    started = time.monotonic()

    async for message in websocket:
        if message.type == WSMsgType.BINARY:
            if len(audio) < int(MAX_WINDOW_SECONDS * BYTE_RATE):
                audio.extend(message.data)
            else:
                overflowed = True
            continue

        if message.type != WSMsgType.TEXT:
            continue

        try:
            event: dict[str, Any] = json.loads(message.data)
            event_type = event.get("type")
        except ValueError:
            _LOGGER.warning("Ignoring unparseable message")
            continue

        if event_type == "verify":
            try:
                verify_request = VerifyRequest.from_json(event)
            except (KeyError, TypeError, ValueError) as err:
                await _send(websocket, Verdict(accepted=True, reason="bad_request"))
                _LOGGER.warning("Bad verify request: %s", err)
                break
            continue

        if event_type != "audio-end":
            continue

        if verify_request is None:
            await _send(websocket, Verdict(accepted=True, reason="no_request"))
            break

        end = AudioEnd.from_json(event)
        answer = await _judge(
            verifier, verify_request, bytes(audio), final=end.final or overflowed
        )
        if isinstance(answer, MoreAudio):
            await websocket.send_json(answer.to_json())
            _LOGGER.debug("Asked for %d ms more audio", answer.ms)
            continue

        _LOGGER.info(
            "%s: %s%s (%d ms of audio, %.0f ms)",
            verify_request.wake_word_phrase,
            "accepted" if answer.accepted else "rejected",
            "" if answer.reason is None else f" ({answer.reason})",
            bytes_to_ms(len(audio)),
            (time.monotonic() - started) * 1000,
        )
        await _send(websocket, answer)
        break

    await websocket.close()
    return websocket


async def _judge(
    verifier: Verifier, request: VerifyRequest, audio: bytes, final: bool
) -> Verdict | MoreAudio:
    """Run the verifier, accepting the detection if it cannot answer.

    A verifier that raises or hangs is a broken verifier, not evidence about
    the audio: refusing on its behalf would silence the satellite.
    """
    try:
        async with asyncio.timeout(VERIFY_TIMEOUT):
            answer = await verifier.verify(request, audio, final)
    except TimeoutError:
        _LOGGER.warning("Verification timed out after %.1fs; accepting", VERIFY_TIMEOUT)
        return Verdict(accepted=True, reason="timeout")
    except Exception:  # pylint: disable=broad-except
        _LOGGER.exception("Verification failed; accepting")
        return Verdict(accepted=True, reason="error")

    if isinstance(answer, MoreAudio) and final:
        # It asked for audio that is not coming. Its last word was the
        # verdict it already had, which it chose not to send; take the safe
        # reading of that rather than inventing a refusal.
        return Verdict(accepted=True, reason="no_more_audio")
    return answer


async def _send(websocket: web.WebSocketResponse, verdict: Verdict) -> None:
    if not websocket.closed:
        await websocket.send_json(verdict.to_json())
