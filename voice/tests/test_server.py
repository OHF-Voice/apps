"""One run over the websocket, including the ways it is allowed to go wrong."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp import ClientSession
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice import vad  # noqa: E402
from voice.protocol import (  # noqa: E402
    API_VERSION,
    RunStart,
    WakeWordInfo,
    ms_to_bytes,
    normalize_wake_word,
)
from voice.server import build_app  # noqa: E402
from voice.verify import MoreAudio, Verdict, Verifier  # noqa: E402

WAKE_MS = 1000
"""How much of every run below the satellite calls the wake word."""


def pcm(ms: int) -> bytes:
    return b"\x00\x00" * (ms_to_bytes(ms) // 2)


class FakeVerifier(Verifier):
    """Answers with whatever the test told it to, and records what it saw."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    @property
    def wake_words(self):
        return [WakeWordInfo(id="okay_nabu", phrase="okay nabu")]

    async def verify(self, request, pcm_data, final):
        self.seen.append((request, len(pcm_data), final))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class ScriptedDetector(vad.Detector):
    """Hears what the test wrote down: 10 ms a chunk, one probability each,
    the last repeating. Silero on synthetic audio would be an experiment."""

    chunk_bytes = 160 * 2

    def __init__(self, probabilities):
        self._probabilities = list(probabilities) or [0.0]
        self.chunks = 0

    def __call__(self, chunk):
        index = min(self.chunks, len(self._probabilities) - 1)
        self.chunks += 1
        return self._probabilities[index]


def hears(monkeypatch, probabilities):
    """Make the endpointer hear this, whatever the audio actually is."""
    monkeypatch.setattr(
        vad, "make_detector", lambda gain=1.0: ScriptedDetector(probabilities)
    )


def run(coro):
    return asyncio.run(coro)


async def _run(verifier, start=None, command_ms=0, chunk_ms=100, pause=0.0):
    """One whole run, returning every event the app sent. The app closes, not
    the client, which is how a test knows the run is over."""
    if start is None:
        start = RunStart(wake_word_phrase="okay nabu", wake_word_audio_ms=WAKE_MS)

    server = TestServer(build_app(verifier, "Voice", "0.1.0"))
    await server.start_server()
    try:
        async with ClientSession() as session:
            async with session.ws_connect(server.make_url("/run")) as websocket:
                await websocket.send_json(start.to_json())
                await websocket.send_bytes(pcm(start.wake_word_audio_ms))
                if pause:
                    # Let the app get as far as it can on the wake word alone
                    await asyncio.sleep(pause)
                for _ in range(command_ms // chunk_ms):
                    await websocket.send_bytes(pcm(chunk_ms))
                await websocket.send_json({"type": "audio-end"})

                return [
                    json.loads(message.data)
                    async for message in websocket
                    if message.type.name == "TEXT"
                ]
    finally:
        await server.close()


async def _info(verifier):
    server = TestServer(build_app(verifier, "Voice", "0.1.0"))
    await server.start_server()
    try:
        async with ClientSession() as session:
            async with session.get(server.make_url("/info")) as response:
                return await response.json()
    finally:
        await server.close()


# ----------------------------------------------------------------------
# What the app says it can do


def test_info():
    info = run(_info(FakeVerifier([])))
    assert info["api_version"] == API_VERSION
    assert info["wake_words"] == [{"id": "okay_nabu", "phrase": "okay nabu"}]
    # Home Assistant is told the ceiling rather than discovering it as silence
    assert info["max_run_ms"] > 0


# ----------------------------------------------------------------------
# The wake word


def test_accepted_carries_the_split():
    verifier = FakeVerifier([Verdict(accepted=True, wake_word_audio_ms=940)])
    events = run(_run(verifier, start=RunStart(wake_word_phrase="okay nabu", wake_word_audio_ms=WAKE_MS)))

    assert events == [
        {
            "type": "wake-word-end",
            "wake_word_output": {
                "wake_word_phrase": "okay nabu",
                "wake_word_audio_ms": 940,
            },
        }
    ]


def test_accepting_without_a_split_keeps_the_satellite_s():
    verifier = FakeVerifier([Verdict(accepted=True)])
    events = run(_run(verifier))

    assert events[0]["wake_word_output"]["wake_word_audio_ms"] == WAKE_MS


def test_rejected_keeps_its_reason_and_ends_the_run():
    verifier = FakeVerifier([Verdict(accepted=False, reason="last")])
    events = run(_run(verifier, command_ms=500))

    assert events == [
        {
            "type": "wake-word-rejected",
            "wake_word_output": {"wake_word_phrase": "okay nabu", "reason": "last"},
        }
    ]


def test_more_audio_is_waited_for_rather_than_asked_for():
    """The verifier hears further into the run without asking for it."""
    verifier = FakeVerifier([MoreAudio(ms=300), Verdict(accepted=True)])
    events = run(_run(verifier, command_ms=500, pause=0.05))

    assert events[0]["type"] == "wake-word-end"
    first, second = verifier.seen
    assert first[1] == ms_to_bytes(WAKE_MS)
    assert second[1] >= first[1] + ms_to_bytes(300)
    # It was never told the run was over: the audio it wanted arrived
    assert [seen[2] for seen in verifier.seen] == [False, False]


def test_asking_for_audio_that_is_not_coming_accepts():
    verifier = FakeVerifier([MoreAudio(ms=300)])
    events = run(_run(verifier))

    assert events[0]["type"] == "wake-word-end"
    assert verifier.seen[-1][2] is True  # it was told there would be no more


def test_a_verifier_that_raises_accepts():
    verifier = FakeVerifier([RuntimeError("the decoder fell over")])
    events = run(_run(verifier))

    assert events[0]["type"] == "wake-word-end"


def test_a_wake_word_it_does_not_know_is_answered_at_once():
    """Rather than leaving Home Assistant to sit out its deadline."""
    verifier = FakeVerifier([])
    events = run(
        _run(
            verifier,
            start=RunStart(wake_word_phrase="hey jarvis", wake_word_audio_ms=WAKE_MS),
        )
    )

    assert events[0] == {
        "type": "wake-word-end",
        "wake_word_output": {
            "wake_word_phrase": "hey jarvis",
            "wake_word_audio_ms": WAKE_MS,
        },
    }
    assert not verifier.seen


def test_a_run_with_no_wake_word_is_not_judged():
    verifier = FakeVerifier([])
    events = run(_run(verifier, start=RunStart()))

    assert events == []
    assert not verifier.seen


# ----------------------------------------------------------------------
# The end of the command


def test_speech_and_silence_are_reported(monkeypatch):
    # 0.3 s of speech, then silence: speech starts after 0.2 s of it, and
    # 1.0 s of silence (0.7 asked for, plus this app's offset) ends it.
    hears(monkeypatch, [1.0] * 30 + [0.0])
    verifier = FakeVerifier([Verdict(accepted=True)])

    events = run(
        _run(
            verifier,
            start=RunStart(
                wake_word_phrase="okay nabu",
                wake_word_audio_ms=WAKE_MS,
                silence_seconds=0.7,
            ),
            command_ms=2000,
        )
    )

    assert [event["type"] for event in events] == [
        "wake-word-end",
        "stt-vad-start",
        "stt-vad-end",
    ]
    # Timestamps are milliseconds from the first audio byte of the run, so
    # they count the wake word too
    assert events[1]["timestamp"] == WAKE_MS + 200
    assert events[2]["timestamp"] == WAKE_MS + 1300


def test_nothing_said_still_ends_the_command(monkeypatch):
    hears(monkeypatch, [0.0])
    verifier = FakeVerifier([Verdict(accepted=True)])

    events = run(
        _run(
            verifier,
            start=RunStart(
                wake_word_phrase="okay nabu",
                wake_word_audio_ms=WAKE_MS,
                silence_seconds=0.7,
            ),
            command_ms=8500,
            chunk_ms=500,
        )
    )

    assert [event["type"] for event in events] == ["wake-word-end", "stt-vad-end"]
    assert events[1]["timestamp"] == WAKE_MS + 8000


def test_endpointing_can_be_switched_off(monkeypatch):
    hears(monkeypatch, [1.0] * 30 + [0.0])
    verifier = FakeVerifier([Verdict(accepted=True)])

    events = run(
        _run(
            verifier,
            start=RunStart(
                wake_word_phrase="okay nabu",
                wake_word_audio_ms=WAKE_MS,
                silence_seconds=None,
            ),
            command_ms=2000,
        )
    )

    assert [event["type"] for event in events] == ["wake-word-end"]


def test_a_run_that_starts_at_speech_is_still_endpointed(monkeypatch):
    hears(monkeypatch, [1.0] * 30 + [0.0])
    verifier = FakeVerifier([])

    events = run(
        _run(
            verifier,
            start=RunStart(silence_seconds=0.7),
            command_ms=2000,
        )
    )

    assert [event["type"] for event in events] == ["stt-vad-start", "stt-vad-end"]
    assert events[0]["timestamp"] == 200


# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling", ["okay nabu", "Okay Nabu", "okay-nabu", "okay_nabu"]
)
def test_one_wake_word_many_spellings(spelling):
    assert normalize_wake_word(spelling) == "okay_nabu"


def test_knows_only_what_it_has():
    verifier = FakeVerifier([])
    assert verifier.knows("Okay Nabu")
    assert not verifier.knows("hey jarvis")


def test_the_gain_follows_the_command_as_well_as_the_wake_word(monkeypatch):
    """A command quieter than the wake word is still heard.

    One said into the microphone and the other from across the room left
    the command 23 dB down, too quiet to be called speech -- and a command
    that never starts is one that never ends.
    """
    seen: list[bytes] = []

    def _loudness(pcm: bytes) -> float:
        seen.append(pcm)
        # Pretend the wake word is loud and the command is not
        return 1.0 if len(seen) == 1 else 12.0

    monkeypatch.setattr("voice.server.vad_gain", _loudness)
    hears(monkeypatch, [1.0] * 30 + [0.0])

    captured: list[float] = []
    real = vad.Endpointer

    def _remember(*args, **kwargs):
        captured.append(kwargs["gain"])
        return real(*args, **kwargs)

    monkeypatch.setattr("voice.server.Endpointer", _remember)

    run(
        _run(
            FakeVerifier([Verdict(accepted=True)]),
            start=RunStart(
                wake_word_phrase="okay nabu",
                wake_word_audio_ms=WAKE_MS,
                silence_seconds=0.7,
            ),
            command_ms=2000,
        )
    )

    # Both windows are judged, and the louder of the two wins
    assert len(seen) == 2
    assert len(seen[0]) == ms_to_bytes(WAKE_MS)
    assert len(seen[1]) == ms_to_bytes(vad.GAIN_WINDOW_MS)
    assert captured == [12.0]


def test_a_long_preroll_does_not_set_the_level(monkeypatch):
    """Only the wake word is judged, not everything before it.

    ``vad_gain`` takes the loudest 32 ms it is given, so a door slam two
    seconds before anyone spoke would otherwise set the gain.
    """
    seen: list[int] = []
    monkeypatch.setattr(
        "voice.server.vad_gain", lambda pcm: seen.append(len(pcm)) or 1.0
    )
    hears(monkeypatch, [1.0] * 30 + [0.0])

    run(
        _run(
            FakeVerifier([Verdict(accepted=True)]),
            start=RunStart(
                wake_word_phrase="okay nabu",
                wake_word_audio_ms=2500,
                silence_seconds=0.7,
            ),
            command_ms=2000,
        )
    )

    assert seen[0] == ms_to_bytes(vad.WAKE_SECONDS * 1000)
