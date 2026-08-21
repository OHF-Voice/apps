#!/usr/bin/env python3
"""End-to-end round trip for the Wyoming intent service.

Starts the intent server in-process against the DEV entity fixture and sends
Recognize events over TCP, checking the Intent/NotRecognized replies.

    /path/to/.venv/bin/python tests/test_intent_server.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import training  # noqa: E402
import intent_server  # noqa: E402
from wyoming.asr import Transcript  # noqa: E402
from wyoming.client import AsyncTcpClient  # noqa: E402
from wyoming.info import Describe, Info  # noqa: E402
from wyoming.intent import Intent, NotRecognized, Recognize  # noqa: E402

URI = "tcp://127.0.0.1:10599"
HOST, PORT = "127.0.0.1", 10599


async def _recognize(text, context=None, *, via="transcript"):
    """Send text to recognize. `via` picks the event HA uses (Transcript) vs the
    generic intent protocol (Recognize); both must work."""
    event = (Transcript(text=text, context=context) if via == "transcript"
             else Recognize(text=text, context=context))
    async with AsyncTcpClient(HOST, PORT) as client:
        await client.write_event(event.event())
        return await client.read_event()


async def main() -> int:
    intent_server.start_background(
        URI, "en", ROOT / "tests" / "_intent_data", ROOT,
        get_entities=lambda: training.DEV_ENTITY_RECORDS,
        get_slot_lists=lambda: training.DEV_SLOT_LISTS,
        api_url="http://unused", token=None,
    )

    # Wait for the server to accept connections.
    for _ in range(50):
        try:
            async with AsyncTcpClient(HOST, PORT) as c:
                await c.write_event(Describe().event())
                info = Info.from_event(await c.read_event())
                assert info.intent and info.intent[0].name == "speech-to-phrase-intents"
                break
        except (ConnectionError, OSError):
            await asyncio.sleep(0.1)
    else:
        print("FAIL: server never came up")
        return 1

    ok = True

    def check(label, event, want_type, want_name=None, want_slots=None):
        nonlocal ok
        if want_type == "intent":
            if not Intent.is_type(event.type):
                print(f"FAIL {label}: expected Intent, got {event.type}"); ok = False; return
            got = Intent.from_event(event)
            slots = {e.name: e.value for e in got.entities}
            bad = got.name != want_name or (
                want_slots is not None and slots != want_slots
            )
            print(f"{'FAIL' if bad else 'ok  '} {label}: {got.name} {slots} text={got.text!r}")
            if bad:
                print(f"      wanted {want_name} {want_slots}"); ok = False
        elif want_type == "not-recognized":
            if not NotRecognized.is_type(event.type):
                print(f"FAIL {label}: expected NotRecognized, got {event.type}"); ok = False; return
            print(f"ok   {label}: NotRecognized")

    check("name (light, via Transcript)", await _recognize("turn on the kitchen lamp"),
          "intent", "HassTurnOn", {"name": "kitchen lamp"})
    check("name (light, via Recognize)",
          await _recognize("turn on the kitchen lamp", via="recognize"),
          "intent", "HassTurnOn", {"name": "kitchen lamp"})
    check("name (off)", await _recognize("turn off the overhead light"),
          "intent", "HassTurnOff", {"name": "overhead light"})
    check("name (cover)", await _recognize("close the garage door"),
          "intent", "HassTurnOff", {"name": "garage door"})
    check("domain in area (no ctx)", await _recognize("turn on the lights"),
          "intent", "HassTurnOn", {"domain": "light"})
    check("domain in area (ctx)",
          await _recognize("lights off", context={"area": "Office"}),
          "intent", "HassTurnOff", {"domain": "light", "area": "Office"})
    check("gibberish", await _recognize("do a barrel roll"), "not-recognized")

    # Entity-aware gating (DEV_ENTITY_RECORDS: lights/fan only in kitchen; the
    # cover has no set_position; the fan supports speed).
    check("feature gate: cover has no set_position",
          await _recognize("set the garage door to 50 percent"), "not-recognized")
    check("feature gate: fan supports speed",
          await _recognize("set the kitchen fan to 50 percent"),
          "intent", "HassFanSetSpeed")
    check("area: lights in an area that has them",
          await _recognize("turn on the lights in the kitchen"),
          "intent", "HassTurnOn", {"domain": "light", "area": "kitchen"})
    # Areas are NOT gated by domain co-occurrence: the command must stay
    # speakable even where no light lives, so HA can report the "no entities"
    # error rather than the phrase going unrecognised.
    check("area: lights in an area that has none is still recognised",
          await _recognize("turn on the lights in the living room"),
          "intent", "HassTurnOn", {"domain": "light", "area": "living room"})

    print("\n" + ("PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
