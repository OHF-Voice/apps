"""The HTTP and websocket server Home Assistant talks to: ``/info``,
``/run``, and ``/health`` for the container check. No page and no settings.

Reading the audio and making sense of it are separate tasks on purpose. A
verification holds the decoder for roughly a tenth of a second while audio
keeps arriving, so the handler does nothing but drain the socket and
``_Run`` consumes the buffer behind it.

Every failure path accepts the detection -- an unparseable message, a decode
that raises or overruns -- because wrongly accepting costs one unwanted
command and wrongly refusing costs a device that no longer answers its name.
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
from .vad import (
    GAIN_WINDOW_MS,
    WAKE_SECONDS,
    Endpointer,
    for_silence_seconds,
    vad_gain,
)
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
    """The run's audio as it arrives, read by offset: one writer (the
    socket), one reader (the analysis). Past ``MAX_RUN_BYTES`` audio is
    dropped, which the reader sees as the run having run out."""

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
        """Whatever has arrived past ``offset``, waiting for it. Empty at
        the end of the run."""
        if await self.wait_for(offset + 1):
            return self.data(offset)
        return b""


async def _run(request: web.Request) -> web.WebSocketResponse:
    """One interaction, from the satellite's first audio to its last.

    One run per connection: audio frames have nowhere to put a run id, so
    sharing one would mean tagging every chunk of every satellite's audio.

    Reading and analysing are two tasks, and the first to finish ends the
    run -- the analysis because it has said everything, the reader because
    Home Assistant hung up and there is nobody left to say it to.
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
        """Read the socket into the buffer and nothing else: whatever the
        analysis is busy with, audio keeps arriving."""
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

        One that did start is left to finish what it was saying, with the
        audio closed behind it so nothing it waits on can hang.
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
            # Said at once, rather than leaving Home Assistant to sit out its
            # deadline before falling back to exactly this.
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

        One that raises or hangs is a broken verifier, not evidence about
        the audio: refusing on its behalf would silence the satellite.
        """
        wanted = ms_to_bytes(request.wake_word_audio_ms)
        if wanted <= 0:
            # A detection that did not say how long it was: the window is
            # everything there is, so wait for the run to end.
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
                # It asked for audio that is not coming; take the safe
                # reading rather than inventing a refusal.
                return Verdict(accepted=True, reason="no_more_audio")

            # It wants to hear further into the run: wait for that much more.
            wanted = len(pcm) + ms_to_bytes(answer.ms)
            final = not await self.audio.wait_for(wanted)

    async def _endpoint(self, start: RunStart) -> None:
        """Say where speech begins and where the command ends."""
        if start.silence_seconds is None:
            return

        split = ms_to_bytes(start.wake_word_audio_ms)
        split_ms = bytes_to_ms(split)
        head = ms_to_bytes(GAIN_WINDOW_MS)
        await self.audio.wait_for(split + head)
        # The wake word, not the whole pre-roll: what is further back than
        # this sets neither the level nor what Silero carries over.
        wake = self.audio.data(max(0, split - ms_to_bytes(WAKE_SECONDS * 1000)), split)

        # The wake word and the command are not always at the same level --
        # one said into the microphone and the other from across the room --
        # and a command too quiet to be called speech is one that never ends.
        wake_gain = vad_gain(wake)
        command_gain = vad_gain(self.audio.data(split, split + head))

        endpointer = Endpointer(
            for_silence_seconds(start.silence_seconds),
            gain=max(wake_gain, command_gain),
            warm=wake,
        )
        _LOGGER.debug(
            "Endpointing from %d ms (gain %.1f, from wake %.1f and command"
            " %.1f; %.2fs of silence ends it)",
            split_ms,
            endpointer.gain,
            wake_gain,
            command_gain,
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
            # The ceiling is the app's: Home Assistant is told where the
            # boundary is, and why is written down here.
            _LOGGER.info(
                "Run reached max_run_ms (%d ms); ending the command there",
                bytes_to_ms(MAX_RUN_BYTES),
            )
            await self.send(SttVadEnd(bytes_to_ms(len(self.audio))))
            return

        _LOGGER.debug("The satellite stopped streaming after %.2fs", endpointer.elapsed)
