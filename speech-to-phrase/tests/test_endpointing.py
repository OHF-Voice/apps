"""Tests for optional pySilero end-of-command detection."""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from vendored_lib import bind as _bind_vendored_lib  # noqa: E402

_bind_vendored_lib()

from wyoming.asr import Transcript  # noqa: E402
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402

import endpointing  # noqa: E402
import wyoming_server as ws  # noqa: E402


class ProbabilityDetector:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)
        self.resets = 0

    def reset(self):
        self.resets += 1

    def process_samples(self, _samples):
        return next(self.probabilities)


def pcm_chunks(count):
    return b"\x00\x00" * endpointing.SAMPLES_PER_VAD_CHUNK * count


def test_silero_endpoint_requires_speech_then_silence():
    probabilities = [0.0] * 20 + [0.9] * 10 + [0.0] * 22
    detector = ProbabilityDetector(probabilities)
    endpoint = endpointing.SileroEndpointDetector(0.7, detector)
    endpoint.reset(16000, 2, 1)

    assert endpoint.process(pcm_chunks(20), 16000, 2, 1)
    assert endpoint.process(pcm_chunks(10), 16000, 2, 1)
    assert not endpoint.process(pcm_chunks(22), 16000, 2, 1)
    assert detector.resets == 1


def test_speech_resets_silence_countdown():
    probabilities = [0.9] * 10 + [0.0] * 15 + [0.9] + [0.0] * 22
    endpoint = endpointing.SileroEndpointDetector(
        0.7, ProbabilityDetector(probabilities)
    )
    endpoint.reset(16000, 2, 1)

    assert endpoint.process(pcm_chunks(10), 16000, 2, 1)
    assert endpoint.process(pcm_chunks(15), 16000, 2, 1)
    assert endpoint.process(pcm_chunks(1), 16000, 2, 1)
    assert endpoint.process(pcm_chunks(21), 16000, 2, 1)
    assert not endpoint.process(pcm_chunks(1), 16000, 2, 1)


def test_pcm_is_downmixed_for_vad():
    probabilities = [0.0]
    endpoint = endpointing.SileroEndpointDetector(
        0.7, ProbabilityDetector(probabilities)
    )
    endpoint.reset(16000, 2, 2)
    stereo = np.zeros(endpointing.SAMPLES_PER_VAD_CHUNK * 2, dtype=np.int16).tobytes()
    assert endpoint.process(stereo, 16000, 2, 2)


def test_server_vad_disables_home_assistant_endpointing():
    assert ws.build_info("en", "model").asr[0].requires_external_vad
    assert (
        not ws.build_info("en", "model", requires_external_vad=False)
        .asr[0]
        .requires_external_vad
    )


@dataclass
class FakeResult:
    text: str = "turn on the kitchen lamp"
    score: float = 1.0
    margin: float = 1.0


class FakeHolder:
    ready = True
    max_score = 5.0
    debug_mode = False
    language = "en"

    def __init__(self):
        self.transcriptions = 0

    async def maybe_reload(self):
        return None

    async def transcribe(self, _samples):
        self.transcriptions += 1
        return FakeResult()


class ImmediateEndpoint:
    def reset(self, _rate, _width, _channels):
        return None

    def process(self, _audio, _rate, _width, _channels):
        return False


class CapturingHandler(ws.S2PEventHandler):
    def __init__(self, holder):
        self._holder = holder
        self._info = None
        self._endpoint_detector = ImmediateEndpoint()
        self._buf = bytearray()
        self._rate, self._width, self._channels = 16000, 2, 1
        self._full = False
        self._finished = False
        self.written = []

    async def write_event(self, event):
        self.written.append(event)


def test_early_endpoint_emits_only_one_transcript():
    async def run():
        holder = FakeHolder()
        handler = CapturingHandler(holder)
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(
                rate=16000,
                width=2,
                channels=1,
                audio=b"\x00\x00" * 8000,
            ).event()
        )
        await handler.handle_event(
            AudioChunk(
                rate=16000,
                width=2,
                channels=1,
                audio=b"\x00\x00" * 8000,
            ).event()
        )
        await handler.handle_event(AudioStop().event())

        transcripts = [
            event for event in handler.written if Transcript.is_type(event.type)
        ]
        assert len(transcripts) == 1
        assert Transcript.from_event(transcripts[0]).text == FakeResult.text
        assert holder.transcriptions == 1

    asyncio.run(run())
