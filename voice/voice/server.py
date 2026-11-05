"""The HTTP and websocket server Home Assistant talks to.

Three endpoints and no page: this app has no interface of its own, and
nothing about it is configured by hand.

``GET /info``    what can be verified here, and how much audio a run may hold
``GET /run``     one websocket, one interaction, audio in and events out
``GET /health``  for the container health check

Reading the audio and making sense of it are separate tasks on purpose.
Audio arrives for the whole run while a verification holds the decoder for
roughly a tenth of a second, so the handler below does nothing but drain the
socket into a buffer, and ``_Run`` consumes that buffer behind it. An app
that read only as fast as it decoded would stall the socket and grow a queue
inside Home Assistant.

The other guiding rule is that an app which is broken, slow or confused must
not make the assistant deaf. Every failure path -- an unparseable message, a
decode that raises, a decode that overruns -- ends in *accepting* the
detection, because the cost of wrongly accepting is one unwanted command and
the cost of wrongly refusing is a device that no longer answers its name.
Only a verifier that ran and said no refuses.
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
    MAX_RUN_SECONDS,
    AppInfo,
    Error,
    RunStart,
    SttVadEnd,
    SttVadStart,
    WakeWordEnd,
    WakeWordRejected,
    bytes_to_ms,
    ms_to_bytes,
)
from .vad import Endpointer, for_silence_seconds, vad_gain
from .verify import MoreAudio, Verdict, Verifier, VerifyRequest

_LOGGER = logging.getLogger(__name__)

MAX_RUN_BYTES = ms_to_bytes(MAX_RUN_SECONDS * 1000)

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
            web.get("/run", _run),
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
        max_run_ms=bytes_to_ms(MAX_RUN_BYTES),
        api_version=API_VERSION,
    )
    return web.json_response(info.to_json())


class _Audio:
    """The run's audio as it arrives, read by offset.

    One writer (the socket) and one reader (the analysis). It is bounded:
    past ``MAX_RUN_BYTES`` the audio is dropped rather than buffered, which
    the reader sees as the run having run out.
    """

    def __init__(self, max_bytes: int = MAX_RUN_BYTES) -> None:
        self._buffer = bytearray()
        self._max_bytes = max_bytes
        self._changed = asyncio.Event()
        self.closed = False
        self.overflowed = False

    def __len__(self) -> int:
        return len(self._buffer)

    def feed(self, data: bytes) -> None:
        room = self._max_bytes - len(self._buffer)
        if room <= 0:
            self.overflowed = True
        else:
            self._buffer.extend(data[:room])
            self.overflowed = self.overflowed or len(data) > room
        self._changed.set()

    def close(self) -> None:
        """No more audio is coming."""
        self.closed = True
        self._changed.set()

    @property
    def _done(self) -> bool:
        return self.closed or self.overflowed

    def data(self, start: int = 0, end: int | None = None) -> bytes:
        return bytes(self._buffer[start:end])

    async def wait_for(self, num_bytes: int) -> bool:
        """Wait until this much has arrived. False if it never will."""
        while True:
            if len(self._buffer) >= num_bytes:
                return True
            if self._done:
                return False
            self._changed.clear()
            if len(self._buffer) >= num_bytes or self._done:
                continue
            await self._changed.wait()

    async def wait_beyond(self, offset: int) -> bytes:
        """Whatever has arrived past ``offset``, waiting for it. Empty at the
        end of the run."""
        if await self.wait_for(offset + 1):
            return self.data(offset)
        return b""


async def _run(request: web.Request) -> web.WebSocketResponse:
    """One interaction, from the satellite's first audio to its last.

    The connection carries a single run and closes: audio frames have
    nowhere to put a run id, so sharing one connection would mean tagging
    every chunk of every satellite's audio, and a run that wedges then takes
    nothing else with it.

    Reading and analysing are two tasks, and the first of them to finish
    ends the run -- the analysis because it has said everything it has to
    say, the reader because Home Assistant hung up and there is nobody left
    to say it to.
    """
    websocket = web.WebSocketResponse()
    await websocket.prepare(request)

    run = _Run(request.app[VERIFIER], websocket)
    reader = asyncio.create_task(run.drain(), name="voice-run-read")
    analysis = asyncio.create_task(run.analyse(), name="voice-run-analyse")

    try:
        await asyncio.wait({reader, analysis}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        run.audio.close()
        run.give_up()
        for task in (reader, analysis):
            task.cancel()
        await asyncio.gather(reader, analysis, return_exceptions=True)
        await websocket.close()

    return websocket


class _Run:
    """One interaction: verify the wake word, then find the command's end."""

    def __init__(self, verifier: Verifier, websocket: web.WebSocketResponse) -> None:
        self.verifier = verifier
        self.websocket = websocket
        self.audio = _Audio()
        self.start: RunStart | None = None
        self._started = asyncio.Event()
        self._abandoned = False

    def begin(self, start: RunStart) -> None:
        if self.start is not None:
            _LOGGER.warning("Ignoring a second run-start")
            return
        self.start = start
        self._started.set()

    async def drain(self) -> None:
        """Read the socket into the buffer, and nothing else.

        Whatever the analysis is busy with, audio keeps arriving, so this
        task never waits on it.
        """
        async for message in self.websocket:
            if message.type == WSMsgType.BINARY:
                self.audio.feed(message.data)
                continue
            if message.type != WSMsgType.TEXT:
                continue

            try:
                event: dict[str, Any] = json.loads(message.data)
            except ValueError:
                _LOGGER.warning("Ignoring unparseable message")
                continue

            event_type = event.get("type")
            if event_type == "run-start":
                try:
                    self.begin(RunStart.from_json(event))
                except (KeyError, TypeError, ValueError) as err:
                    _LOGGER.warning("Bad run-start: %s", err)
                    await self.send(Error("bad_request", str(err)))
                    return
            elif event_type == "audio-end":
                self.audio.close()

    def give_up(self) -> None:
        """The socket is gone; stop waiting for a run that never started.

        A run that did start is left to finish what it was saying: the audio
        is closed behind it, so nothing it is waiting on can hang.
        """
        if self.start is None:
            self._abandoned = True
        self._started.set()

    async def send(self, event: Any) -> None:
        if not self.websocket.closed:
            await self.websocket.send_json(event.to_json())

    async def analyse(self) -> None:
        await self._started.wait()
        if (self.start is None) or self._abandoned:
            return

        try:
            if not await self._wake_word(self.start):
                return
            await self._endpoint(self.start)
        except asyncio.CancelledError:
            raise
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Run failed")
            await self.send(Error("internal", "the run could not be analysed"))

    # ------------------------------------------------------------------

    async def _wake_word(self, start: RunStart) -> bool:
        """Judge the detection. False when it was refused and the run ends."""
        phrase = start.wake_word_phrase
        if phrase is None:
            # A button press, or a satellite that starts at speech-to-text.
            return True

        if not self.verifier.knows(phrase):
            # Nothing to say about it, said immediately: Home Assistant would
            # otherwise sit out its deadline before falling back to exactly
            # this.
            await self.send(WakeWordEnd(phrase, start.wake_word_audio_ms))
            return True

        began = time.monotonic()
        request = VerifyRequest(
            wake_word_phrase=phrase,
            satellite_id=start.satellite_id,
            wake_word_audio_ms=start.wake_word_audio_ms,
        )
        verdict = await self._judge(request)

        _LOGGER.info(
            "%s: %s%s (%.0f ms)",
            phrase,
            "accepted" if verdict.accepted else "rejected",
            "" if verdict.reason is None else f" ({verdict.reason})",
            (time.monotonic() - began) * 1000,
        )
        if verdict.detail:
            _LOGGER.debug("%s: %s", phrase, verdict.detail)

        if not verdict.accepted:
            await self.send(WakeWordRejected(phrase, verdict.reason))
            return False

        await self.send(
            WakeWordEnd(
                phrase,
                start.wake_word_audio_ms
                if verdict.wake_word_audio_ms is None
                else verdict.wake_word_audio_ms,
            )
        )
        return True

    async def _judge(self, request: VerifyRequest) -> Verdict:
        """Run the verifier over as much of the run as it asks to see.

        A verifier that raises or hangs is a broken verifier, not evidence
        about the audio: refusing on its behalf would silence the satellite.
        """
        wanted = ms_to_bytes(request.wake_word_audio_ms)
        if wanted <= 0:
            # A detection that did not say how long it was. The window is
            # then everything there is, so there is nothing to judge until
            # the run is over.
            wanted = MAX_RUN_BYTES
        final = not await self.audio.wait_for(wanted)

        while True:
            pcm = self.audio.data()
            try:
                async with asyncio.timeout(VERIFY_TIMEOUT):
                    answer = await self.verifier.verify(request, pcm, final)
            except TimeoutError:
                _LOGGER.warning(
                    "Verification timed out after %.1fs; accepting", VERIFY_TIMEOUT
                )
                return Verdict(accepted=True, reason="timeout")
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Verification failed; accepting")
                return Verdict(accepted=True, reason="error")

            if isinstance(answer, Verdict):
                return answer
            if final:
                # It asked for audio that is not coming. Its last word was
                # the verdict it already had and chose not to send; take the
                # safe reading of that rather than inventing a refusal.
                return Verdict(accepted=True, reason="no_more_audio")

            # It wants to see further into the run, so wait for that much
            # more of it to arrive.
            wanted = len(pcm) + ms_to_bytes(answer.ms)
            final = not await self.audio.wait_for(wanted)

    async def _endpoint(self, start: RunStart) -> None:
        """Say where speech begins and where the command ends."""
        if start.silence_seconds is None:
            return

        split = ms_to_bytes(start.wake_word_audio_ms)
        split_ms = bytes_to_ms(split)
        await self.audio.wait_for(split)
        wake = self.audio.data(0, split)

        endpointer = Endpointer(
            for_silence_seconds(start.silence_seconds),
            gain=vad_gain(wake),
            warm=wake,
        )
        _LOGGER.debug(
            "Endpointing from %d ms (gain %.1f, %.2fs of silence ends it)",
            split_ms,
            endpointer.gain,
            endpointer.settings.silence_seconds,
        )

        offset = split
        spoke = False
        while True:
            chunk = await self.audio.wait_beyond(offset)
            if not chunk:
                break
            offset += len(chunk)
            ended = endpointer.process(chunk)

            if (not spoke) and (endpointer.speech_started_at is not None):
                spoke = True
                await self.send(
                    SttVadStart(split_ms + round(endpointer.speech_started_at * 1000))
                )
            if ended:
                assert endpointer.ended_at is not None
                _LOGGER.info(
                    "Command ended after %.2fs (%s)",
                    endpointer.ended_at,
                    endpointer.reason,
                )
                await self.send(
                    SttVadEnd(split_ms + round(endpointer.ended_at * 1000))
                )
                return

        if self.audio.overflowed:
            # The ceiling is the app's, so Home Assistant is told where the
            # boundary is and the reason is written down here.
            _LOGGER.info(
                "Run reached max_run_ms (%d ms); ending the command there",
                bytes_to_ms(MAX_RUN_BYTES),
            )
            await self.send(SttVadEnd(bytes_to_ms(len(self.audio))))
            return

        _LOGGER.debug("The satellite stopped streaming after %.2fs", endpointer.elapsed)
