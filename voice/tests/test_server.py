"""The websocket exchange, including the ways it is allowed to go wrong."""

import asyncio
import sys
from pathlib import Path

import pytest
from aiohttp import ClientSession
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice.protocol import (  # noqa: E402
    API_VERSION,
    AudioEnd,
    MoreAudio,
    Verdict,
    VerifyRequest,
    WakeWordInfo,
    normalize_wake_word,
)
from voice.server import build_app  # noqa: E402
from voice.verify import Verifier  # noqa: E402

AUDIO = b"\x00\x01" * 8000  # 1 s


class FakeVerifier(Verifier):
    """Answers with whatever the test told it to, and records what it saw."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    @property
    def wake_words(self):
        return [WakeWordInfo(id="okay_nabu", phrase="okay nabu")]

    async def verify(self, request, pcm, final):
        self.seen.append((request, len(pcm), final))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def run(coro):
    return asyncio.run(coro)


async def _exchange(verifier, chunks=(AUDIO,), request=None, rounds=1):
    """One verification, following up to ``rounds`` more-audio requests."""
    server = TestServer(build_app(verifier, "Voice", "0.1.0"))
    await server.start_server()
    try:
        async with ClientSession() as session:
            async with session.ws_connect(server.make_url("/verify")) as websocket:
                verify = request or VerifyRequest(
                    wake_word_phrase="okay nabu", wake_word_audio_ms=1000
                )
                await websocket.send_json(verify.to_json())
                for chunk in chunks:
                    await websocket.send_bytes(chunk)
                await websocket.send_json(AudioEnd(final=rounds == 0).to_json())

                while True:
                    answer = await websocket.receive_json()
                    if answer["type"] != "more-audio":
                        return answer
                    rounds -= 1
                    await websocket.send_bytes(AUDIO)
                    await websocket.send_json(AudioEnd(final=rounds <= 0).to_json())
    finally:
        await server.close()


def test_info():
    async def go():
        server = TestServer(build_app(FakeVerifier([]), "Voice", "0.1.0"))
        await server.start_server()
        try:
            async with ClientSession() as session:
                async with session.get(server.make_url("/info")) as response:
                    return await response.json()
        finally:
            await server.close()

    info = run(go())
    assert info["api_version"] == API_VERSION
    assert info["name"] == "Voice"
    assert info["wake_words"] == [{"id": "okay_nabu", "phrase": "okay nabu"}]


def test_accepted_carries_the_split():
    verifier = FakeVerifier([Verdict(accepted=True, wake_word_audio_ms=940)])
    answer = run(_exchange(verifier))
    assert answer["accepted"] is True
    assert answer["wake_word_audio_ms"] == 940

    request, num_bytes, final = verifier.seen[0]
    assert request.wake_word_phrase == "okay nabu"
    assert request.wake_word_audio_ms == 1000
    assert num_bytes == len(AUDIO)
    assert final is False


def test_rejected_keeps_its_reason():
    verifier = FakeVerifier([Verdict(accepted=False, reason="confusable")])
    answer = run(_exchange(verifier))
    assert answer["accepted"] is False
    assert answer["reason"] == "confusable"


def test_more_audio_extends_the_window():
    verifier = FakeVerifier([MoreAudio(ms=300), Verdict(accepted=True)])
    answer = run(_exchange(verifier, rounds=1))
    assert answer["accepted"] is True

    # The whole window is offered again, not just the new audio.
    assert [seen[1] for seen in verifier.seen] == [len(AUDIO), 2 * len(AUDIO)]
    assert [seen[2] for seen in verifier.seen] == [False, True]


def test_asking_for_audio_that_is_not_coming_accepts():
    verifier = FakeVerifier([MoreAudio(ms=300)])
    answer = run(_exchange(verifier, rounds=0))
    assert answer["accepted"] is True
    assert answer["reason"] == "no_more_audio"


def test_a_verifier_that_raises_accepts():
    verifier = FakeVerifier([RuntimeError("decoder died")])
    answer = run(_exchange(verifier))
    assert answer["accepted"] is True
    assert answer["reason"] == "error"


def test_audio_without_a_request_accepts():
    async def go():
        server = TestServer(build_app(FakeVerifier([]), "Voice", "0.1.0"))
        await server.start_server()
        try:
            async with ClientSession() as session:
                async with session.ws_connect(server.make_url("/verify")) as websocket:
                    await websocket.send_bytes(AUDIO)
                    await websocket.send_json(AudioEnd(final=True).to_json())
                    return await websocket.receive_json()
        finally:
            await server.close()

    answer = run(go())
    assert answer["accepted"] is True
    assert answer["reason"] == "no_request"


@pytest.mark.parametrize(
    "spelling", ["okay_nabu", "okay nabu", "Okay-Nabu", "OKAY_NABU"]
)
def test_one_wake_word_many_spellings(spelling):
    assert normalize_wake_word(spelling) == "okay_nabu"


def test_knows_only_what_it_has():
    verifier = FakeVerifier([])
    assert verifier.knows("Okay Nabu")
    assert not verifier.knows("hey_jarvis")
