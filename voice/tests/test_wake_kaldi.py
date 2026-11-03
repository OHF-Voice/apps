"""The retry the Kaldi backend adds over the decoder.

What the four decodes mean is the vendored code's business and is tested
where it came from. What is tested here is the one decision this app makes
with them: when the alignment puts the end of the wake phrase against the
end of the window, the detector fired before the speaker finished, and the
window has to slide rather than the phrase be judged truncated.

The decodes themselves are scripted, so this runs without the 255 MB of
Kaldi binaries and model data that staging brings in.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice.protocol import MoreAudio, Verdict, VerifyRequest, ms_to_bytes  # noqa: E402
from voice.wake_kaldi import TAIL_MS, KaldiVerifier  # noqa: E402

PREROLL_MS = 1000
PREROLL = ms_to_bytes(PREROLL_MS)
TAIL = ms_to_bytes(TAIL_MS)
REQUEST = VerifyRequest(wake_word_id="okay_nabu", preroll_ms=PREROLL_MS)


def result(accepted, split=None, rejected_by=None):
    """One of the library's verdicts, as far as this module reads it."""
    return SimpleNamespace(
        accepted=accepted,
        rejected_by=rejected_by,
        split=split,
        split_ms=None if split is None else int(split * 1000),
        to_dict=lambda: {"accepted": accepted, "rejected_by": rejected_by},
    )


class Verifier(KaldiVerifier):
    """A KaldiVerifier whose decodes are scripted."""

    def __init__(self, results):
        self._results = list(results)
        self.windows = []
        self.decoded = []
        self._decodes = {"okay_nabu": None}
        self._pools = []
        self.config = SimpleNamespace(
            max_window_seconds=1.4,
            wake_words={},
            resolve=lambda name: SimpleNamespace(id="okay_nabu"),
        )

    async def _decode(self, wake_id, window):
        self.windows.append(len(window))
        self.decoded.append(window)
        return self._results.pop(0)


async def verify(verifier, audio_ms, final=False):
    return await verifier.verify(REQUEST, bytes(ms_to_bytes(audio_ms)), final)


@pytest.mark.asyncio
async def test_accepted_at_once():
    verifier = Verifier([result(True, split=0.5)])
    answer = await verify(verifier, PREROLL_MS)
    assert isinstance(answer, Verdict)
    assert answer.accepted
    assert answer.split_ms == 500
    assert verifier.windows == [PREROLL]


@pytest.mark.asyncio
async def test_a_rejection_that_did_not_run_off_the_end_stands():
    verifier = Verifier([result(False, split=0.5, rejected_by="confusable")])
    answer = await verify(verifier, PREROLL_MS)
    assert isinstance(answer, Verdict)
    assert not answer.accepted
    assert answer.reason == "confusable"
    # No retry: the phrase finished inside the window.
    assert verifier.windows == [PREROLL]


@pytest.mark.asyncio
async def test_a_rejection_with_no_alignment_stands():
    verifier = Verifier([result(False, split=None, rejected_by="prefix")])
    answer = await verify(verifier, PREROLL_MS)
    assert isinstance(answer, Verdict)
    assert not answer.accepted
    assert verifier.windows == [PREROLL]


@pytest.mark.asyncio
async def test_a_phrase_against_the_edge_asks_for_more():
    verifier = Verifier([result(False, split=1.0, rejected_by="last")])
    answer = await verify(verifier, PREROLL_MS)
    assert answer == MoreAudio(ms=TAIL_MS)


@pytest.mark.asyncio
async def test_the_window_slides_once_the_audio_is_there():
    verifier = Verifier(
        [result(False, split=1.0, rejected_by="last"), result(True, split=0.8)]
    )
    answer = await verify(verifier, PREROLL_MS + TAIL_MS)

    assert isinstance(answer, Verdict)
    assert answer.accepted
    # Same length of window, started a tail later, and the split with it
    assert verifier.windows == [PREROLL, PREROLL]
    assert answer.split_ms == 800 + TAIL_MS
    assert answer.detail["offset_ms"] == TAIL_MS


@pytest.mark.asyncio
async def test_the_first_rejection_stands_if_the_retry_also_fails():
    verifier = Verifier(
        [
            result(False, split=1.0, rejected_by="last"),
            result(False, split=None, rejected_by="prefix"),
        ]
    )
    answer = await verify(verifier, PREROLL_MS + TAIL_MS)

    assert isinstance(answer, Verdict)
    assert not answer.accepted
    assert answer.reason == "last"


@pytest.mark.asyncio
async def test_audio_that_is_not_coming_is_not_waited_for():
    verifier = Verifier([result(False, split=1.0, rejected_by="last")])
    answer = await verify(verifier, PREROLL_MS, final=True)

    assert isinstance(answer, Verdict)
    assert not answer.accepted


@pytest.mark.asyncio
async def test_the_window_is_capped_and_ends_at_the_detection():
    verifier = Verifier([result(True, split=1.2)])
    answer = await verifier.verify(
        VerifyRequest(wake_word_id="okay_nabu", preroll_ms=3000),
        # Count the samples, so the window's position shows in them.
        b"".join(i.to_bytes(2, "little") for i in range(ms_to_bytes(3000) // 2)),
        False,
    )

    assert verifier.windows == [ms_to_bytes(1400)]
    # It is the last 1.4 s of the pre-roll, not the first
    assert verifier.decoded[0][:2] == (ms_to_bytes(1600) // 2).to_bytes(2, "little")
    # ...and the split it reports is still measured from the audio's start
    assert isinstance(answer, Verdict)
    assert answer.split_ms == 1200 + 1600


@pytest.mark.asyncio
async def test_a_wake_word_it_was_never_offered():
    verifier = Verifier([])
    verifier.config.resolve = lambda name: None
    answer = await verify(verifier, PREROLL_MS)

    assert isinstance(answer, Verdict)
    assert answer.accepted
    assert answer.reason == "unknown_wake_word"
