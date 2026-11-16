"""Unusable Home Assistant names and failures must not take STT down.

OHF-Voice/apps#39: an area whose name was a single NUL byte broke training, and
the failure left Home Assistant waiting on a transcript that never came.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from vendored_lib import bind as _bind_vendored_lib  # noqa: E402

_bind_vendored_lib()

from wyoming.asr import Transcript  # noqa: E402
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402

import s2p_intents  # noqa: E402
import training  # noqa: E402
import wyoming_server as ws  # noqa: E402

UNSPEAKABLE = ["\0", "", "   ", "!!!", "🙂", "kit\0chen", "\x07", "​", "-"]


def test_unspeakable_names_are_skipped():
    assert training._norm_values(UNSPEAKABLE) == []


def test_ordinary_names_are_kept():
    names = ["Lamp 2", "Bob's room", "Café", "टाइमर", "2"]
    assert training._norm_values(names) == [
        "lamp 2",
        "bob's room",
        "café",
        "टाइमर",
        "2",
    ]


def test_unspeakable_area_never_reaches_the_grammar():
    enabled = [[intent, combo] for intent, combo in s2p_intents.combos("en")]
    slot_lists = {
        **training.DEV_SLOT_LISTS,
        "area": ["Kitchen", *UNSPEAKABLE],
        "floor": ["\0"],
    }
    _, list_values = training.assemble(
        "en", enabled, [], training.DEV_ENTITY_RECORDS, slot_lists
    )
    assert list_values["area"] == ["kitchen"]
    assert list_values["floor"] == []
    for key, values in list_values.items():
        assert all(training._speakable(v) for v in values), key


class _FakeRecognizer:
    """Saves like the library: the grammar plus a ``.tokens`` sidecar."""

    fail = False

    def train(self, _templates, list_values=None):
        return None

    def save(self, path):
        Path(f"{path}.tokens").write_bytes(b"new tokens")
        Path(path).write_bytes(b"trunc" if self.fail else b"new")
        if self.fail:
            raise OSError("disk full")


def _train_with(monkeypatch, tmp_path, rec):
    import speech_to_phrase

    monkeypatch.setattr(speech_to_phrase, "load_recognizer", lambda *_a, **_k: rec)
    grammar = tmp_path / "grammar.fst"
    training.train("nemo", tmp_path, "en", ["hello"], {}, grammar)
    return grammar


def test_save_moves_the_sidecar_with_the_grammar(monkeypatch, tmp_path):
    grammar = _train_with(monkeypatch, tmp_path, _FakeRecognizer())
    assert grammar.read_bytes() == b"new"
    assert (tmp_path / "grammar.fst.tokens").read_bytes() == b"new tokens"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "grammar.fst",
        "grammar.fst.tokens",
    ]


def test_failed_save_keeps_the_previous_grammar(monkeypatch, tmp_path):
    (tmp_path / "grammar.fst").write_bytes(b"previous")
    (tmp_path / "grammar.fst.tokens").write_bytes(b"previous tokens")
    rec = _FakeRecognizer()
    rec.fail = True
    try:
        _train_with(monkeypatch, tmp_path, rec)
    except OSError:
        pass
    else:
        raise AssertionError("save error was swallowed")
    assert (tmp_path / "grammar.fst").read_bytes() == b"previous"
    assert (tmp_path / "grammar.fst.tokens").read_bytes() == b"previous tokens"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "grammar.fst",
        "grammar.fst.tokens",
    ]


class _Holder:
    ready = True
    max_score = 5.0
    debug_mode = False
    language = "en"

    async def maybe_reload(self):
        return None

    async def transcribe(self, _samples):
        raise RuntimeError("decoder blew up")


class _CapturingHandler(ws.S2PEventHandler):
    def __init__(self, holder):  # pylint: disable=super-init-not-called
        self._holder = holder
        self._info = None
        self._endpoint_detector = None
        self._vad_silence_seconds = None
        self._buf = bytearray()
        self._rate, self._width, self._channels = 16000, 2, 1
        self._full = False
        self._finished = False
        self.written = []

    async def write_event(self, event):
        self.written.append(event)


def test_transcription_error_still_answers():
    async def run():
        handler = _CapturingHandler(_Holder())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=b"\0\0" * 8000).event()
        )
        await handler.handle_event(AudioStop().event())
        transcripts = [e for e in handler.written if Transcript.is_type(e.type)]
        assert len(transcripts) == 1
        assert Transcript.from_event(transcripts[0]).text == ""

    asyncio.run(run())


def test_unloadable_grammar_keeps_the_previous_one(tmp_path):
    class Recognizer:
        grammar = "previous"
        loads = 0

        def load(self, _path):
            Recognizer.loads += 1
            raise RuntimeError("bad FST header")

    grammar = tmp_path / "grammar.fst"
    grammar.write_bytes(b"garbage")
    holder = ws.GrammarHolder.__new__(ws.GrammarHolder)
    holder._rec = Recognizer()
    holder._grammar_path = grammar
    holder._settings_path = tmp_path / "settings.json"
    holder._default_max_score = 5.0
    holder.max_score = 5.0
    holder._mtime = None
    holder._lock = asyncio.Lock()

    async def run():
        await holder.maybe_reload()
        await holder.maybe_reload()

    asyncio.run(run())
    assert holder.ready and holder._rec.grammar == "previous"
    assert Recognizer.loads == 1  # not retried until the file changes
