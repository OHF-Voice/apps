#!/usr/bin/env python3
"""Debug mode: what gets logged, and what Home Assistant is told.

Drives the real Wyoming event handler with a stubbed recognizer, so the
acoustics are out of the way and the decision under test is visible: in debug
mode Home Assistant must receive an empty transcript whatever the score gate
decided, and every utterance must reach the log with its verdict.

    /path/to/.venv/bin/python tests/test_debug_mode.py
"""
import asyncio
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wyoming.asr import Transcript  # noqa: E402
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402

import debug_log  # noqa: E402
import settings  # noqa: E402
import sources  # noqa: E402
import training  # noqa: E402
import wyoming_server as ws  # noqa: E402


@dataclass
class FakeResult:
    text: str
    score: float
    margin: float = 1.0


class FakeHolder:
    """Stands in for GrammarHolder: the handler only asks it whether it is
    ready, what the gate is, whether debug mode is on, and for a decode."""

    def __init__(self, result: FakeResult, max_score=5.0, debug_mode=False):
        self.result = result
        self.max_score = max_score
        self.debug_mode = debug_mode
        self.language = "en"
        self.ready = True

    async def maybe_reload(self):
        return None

    async def transcribe(self, samples):
        return self.result


class CapturingHandler(ws.S2PEventHandler):
    """The real handler with the socket taken out."""

    def __init__(self, holder):
        # Skip AsyncEventHandler.__init__ (it wants a reader/writer pair).
        self._holder = holder
        self._info = None
        self._buf = bytearray()
        self._rate, self._width, self._channels = 16000, 2, 1
        self.written = []

    async def write_event(self, event):
        self.written.append(event)


async def utterance(holder) -> str:
    """Run one full audio exchange and return the transcript HA receives."""
    handler = CapturingHandler(holder)
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    # 0.5 s of (silent) audio: the handler only needs samples to exist.
    await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00\x00" * 8000).event()
    )
    await handler.handle_event(AudioStop().event())
    transcripts = [e for e in handler.written if Transcript.is_type(e.type)]
    assert len(transcripts) == 1, transcripts
    return Transcript.from_event(transcripts[0]).text


async def main() -> int:
    ok = True

    def check(label, got, want):
        nonlocal ok
        bad = got != want
        print(f"{'FAIL' if bad else 'ok  '} {label}: {got!r}")
        if bad:
            print(f"      wanted {want!r}")
            ok = False

    good = FakeResult(text="turn on the kitchen lamp", score=1.0)
    poor = FakeResult(text="turn on the kitchen lamp", score=40.0)
    nothing = FakeResult(text="", score=math.inf, margin=math.inf)

    # --- debug mode off: unchanged behaviour ---------------------------------
    debug_log.clear()
    check("off + accepted -> HA gets the transcript",
          await utterance(FakeHolder(good)), "turn on the kitchen lamp")
    check("off + rejected -> HA gets nothing",
          await utterance(FakeHolder(poor)), "")
    check("off -> nothing logged", len(debug_log.entries()), 0)

    # --- debug mode on: HA is told nothing, everything is logged -------------
    debug_log.clear()
    check("on + accepted -> HA still gets nothing",
          await utterance(FakeHolder(good, debug_mode=True)), "")
    check("on + rejected -> HA gets nothing",
          await utterance(FakeHolder(poor, debug_mode=True)), "")
    check("on + no parse -> HA gets nothing",
          await utterance(FakeHolder(nothing, debug_mode=True)), "")

    entries = debug_log.entries()
    check("all three logged", len(entries), 3)
    check("verdicts", [e["accepted"] for e in entries], [True, False, False])
    check("text is logged even when gated",
          [e["text"] for e in entries],
          ["turn on the kitchen lamp", "turn on the kitchen lamp", ""])
    check("scores", [e["score"] for e in entries], [1.0, 40.0, None])
    check("gate recorded alongside", [e["max_score"] for e in entries],
          [5.0, 5.0, 5.0])
    check("ids increase", [e["id"] for e in entries],
          sorted(e["id"] for e in entries))
    # inf must not reach the UI as a bare float: it isn't valid JSON.
    json.dumps(entries)
    check("since= returns only newer", len(debug_log.entries(since=entries[1]["id"])), 1)

    # --- the toggle is read from settings.json, per language -----------------
    with tempfile.TemporaryDirectory() as data_dir:
        p = settings.path(data_dir, "en")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        check("default off", settings.read_bool_file(p, "debug_mode", False), False)
        settings.set_bool(data_dir, "en", "debug_mode", True)
        check("hot-read after the UI writes it",
              settings.read_bool_file(p, "debug_mode", False), True)
        check("max_score survives the write",
              json.loads(p.read_text()).get("debug_mode"), True)

    # --- attribution: which source produced the transcript -------------------
    by_source, list_values = training.assemble_sources(
        ROOT, "en", [["HassTurnOn", "name_only"], ["HassLightSet", "name_brightness"]],
        [{"sentences": ["movie time please"], "mode": "stt"}],
        training.DEV_ENTITY_RECORDS, training.DEV_SLOT_LISTS,
        hass_sentences={"sentence_triggers": ["goodnight house"],
                        "question_answers": ["pepperoni"]},
    )
    check("sources are kept apart",
          sorted(k.split(":")[0] for k in by_source),
          ["builtin", "builtin", "custom", "question_answers", "sentence_triggers"])

    att = sources.build(by_source, list_values, "en")
    def origin(text):
        got = att.attribute(text)
        return got and got["source"]

    check("built-in", origin("turn on the kitchen lamp"),
          "builtin:HassTurnOn/name_only")
    check("custom command", origin("movie time please"), "custom:0")
    check("sentence trigger", origin("goodnight house"), "sentence_triggers")
    check("question answer", origin("pepperoni"), "question_answers")
    # Numbers are spelled out by the grammar, so attribution must use the
    # library's own spellout rather than guessing.
    check("built-in with a number",
          origin("set the kitchen lamp brightness to fifty percent"),
          "builtin:HassLightSet/name_brightness")
    check("out-of-grammar", origin("do a barrel roll"), None)
    check("empty", origin(""), None)

    # A grammar the assembler can't attribute must not crash the debug view.
    check("no sources at all", sources.build({}, {}, "en").attribute("anything"), None)

    print("\n" + ("PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
