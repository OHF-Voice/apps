#!/usr/bin/env python3
"""Sentence triggers / ask_question answers fetched from Home Assistant.

Runs a fake Home Assistant websocket API in-process and checks that
``hass_sentences`` finds the phrases HA is listening for, honours its two
gates, and lands them in the trained grammar.

    /path/to/.venv/bin/python tests/test_hass_sentences.py
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aiohttp import web  # noqa: E402

import hass_sentences as hs  # noqa: E402
import training  # noqa: E402

PORT = 18199
API = f"http://127.0.0.1:{PORT}/api"

TRIGGER_SENTENCES = ["movie time", "start [the] movie", "{{ states('x') }}"]

STATES = [
    {"entity_id": "automation.pizza", "state": "on"},
    {"entity_id": "automation.disabled", "state": "off"},
    {"entity_id": "script.pick_film", "state": "off"},
    {"entity_id": "light.kitchen", "state": "on"},
]


def _ask(*sentence_lists):
    return {
        "action": "assist_satellite.ask_question",
        "data": {"answers": [{"sentences": s} for s in sentence_lists]},
    }


CONFIGS = {
    # Nested in a choose/sequence, the way HA actually writes them.
    "automation.pizza": {
        "actions": [{"choose": [{"conditions": [], "sequence": [
            _ask(["pepperoni", "cheese [please]"], "no thanks",
                 ["{{ states('sensor.topping') }}"]),
        ]}]}],
    },
    # A disabled automation can never ask its question.
    "automation.disabled": {"actions": [_ask(["should not appear"])]},
    # Scripts have no "off" state to filter on -- they are just not running.
    "script.pick_film": {"sequence": [_ask(["the godfather"])]},
}


async def _ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    await ws.send_json({"type": "auth_required"})
    await ws.receive_json()
    await ws.send_json({"type": "auth_ok"})
    async for msg in ws:
        m = json.loads(msg.data)
        type_, msg_id = m["type"], m["id"]

        def result(value, success=True):
            return {"id": msg_id, "type": "result", "success": success,
                    "result": value, "error": None if success else {"code": "x"}}

        if type_ == "conversation/sentences/list":
            await ws.send_json(result({"trigger_sentences": TRIGGER_SENTENCES}))
        elif type_ == "get_states":
            await ws.send_json(result(STATES))
        elif type_.endswith("/config"):
            config = CONFIGS.get(m["entity_id"])
            await ws.send_json(
                result({"config": config}) if config else result(None, success=False)
            )
        else:
            await ws.send_json(result(None, success=False))
    return ws


async def main() -> int:
    app = web.Application()
    app.router.add_get("/api/websocket", _ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", PORT).start()

    ok = True

    def check(label, got, want):
        nonlocal ok
        bad = got != want
        print(f"{'FAIL' if bad else 'ok  '} {label}: {got}")
        if bad:
            print(f"      wanted {want}")
            ok = False

    triggers = ["movie time", "start [the] movie"]
    answers = ["pepperoni", "cheese [please]", "no thanks", "the godfather"]

    # A Jinja2 template has no fixed spoken form, and the disabled automation's
    # answer is dead weight: both are dropped. The two sources stay apart so the
    # UI can say which switch a phrase belongs to.
    check("triggers only", await hs._fetch_async(API, "tok", True, False),
          {hs.TRIGGERS: triggers, hs.ANSWERS: []})
    check("answers only (script included, disabled automation skipped)",
          await hs._fetch_async(API, "tok", False, True),
          {hs.TRIGGERS: [], hs.ANSWERS: answers})
    check("both", await hs._fetch_async(API, "tok", True, True),
          {hs.TRIGGERS: triggers, hs.ANSWERS: answers})

    # hs.fetch is the synchronous entry point the training path uses (from the
    # Flask request thread or the watch thread), so drive it off-loop the same
    # way. The gates are checked before the connection, so an unreachable HA is
    # not what makes these empty.
    fetch = lambda **kw: asyncio.to_thread(hs.fetch, API, "tok", **kw)  # noqa: E731
    check("both gates off", await fetch(triggers=False, answers=False), [])
    check("no token", await asyncio.to_thread(hs.fetch, API, None), [])
    check("one gate off fetches only that source",
          await fetch(answers=False), triggers)
    check("sync fetch (flattened)", await fetch(), triggers + answers)

    await runner.cleanup()

    # Cached, so it survives the server going away.
    check("cached fetch", await fetch(), triggers + answers)

    # HA writes hassil ("[the] movie"), the trainer needs flat phrasings. A
    # reference to a list HA never declared cannot be bound, so it goes.
    check("grammar templates",
          hs.grammar_templates(
              triggers + answers + ["turn on {mystery}"], "en",
              {"name", "area", "floor"},
          ),
          ["movie time", "start the movie", "start movie", "pepperoni",
           "cheese please", "cheese", "no thanks", "the godfather"])

    # ...and they reach the assembled grammar.
    enabled = [["HassTurnOn", "name_only"]]
    args = ("en", enabled, [], training.DEV_ENTITY_RECORDS,
            training.DEV_SLOT_LISTS)
    base, _ = training.assemble(*args)
    with_ha, _ = training.assemble(*args, hass_sentences=triggers + answers)
    check("added to the grammar", sorted(set(with_ha) - set(base)),
          sorted(["movie time", "start the movie", "start movie", "pepperoni",
                  "cheese please", "cheese", "no thanks", "the godfather"]))

    # Per-sentence grammar cost, for the UI. A sentence that costs 0 is one the
    # UI has to flag: it is in Home Assistant but cannot be recognized.
    costs = training.hass_sentence_costs(
        "en",
        ["movie time", "start [the] movie", "turn on {mystery}", "lights in the {area}"],
        training.DEV_ENTITY_RECORDS, training.DEV_SLOT_LISTS,
    )
    check("per-sentence cost",
          [(c["text"], c["phrases"]) for c in costs],
          [("movie time", 1), ("start [the] movie", 2), ("turn on {mystery}", 0),
           # {area} costs one phrase per area in the (dev) registry.
           ("lights in the {area}", len(training.DEV_SLOT_LISTS["area"]))])

    # The per-language switches: the add-on option is the default, the UI
    # overrides one language, and a non-bool is refused rather than coerced.
    import settings  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as data_dir:
        check("flag defaults to the add-on option",
              settings.get_bool(data_dir, "en", hs.TRIGGERS, True), True)
        settings.set_bool(data_dir, "en", hs.TRIGGERS, False)
        check("flag override persists",
              settings.get_bool(data_dir, "en", hs.TRIGGERS, True), False)
        check("other language keeps the default",
              settings.get_bool(data_dir, "de", hs.TRIGGERS, True), True)
        check("non-bool is refused",
              settings.set_bool(data_dir, "en", hs.TRIGGERS, "false"), None)
        check("...and the previous value stands",
              settings.get_bool(data_dir, "en", hs.TRIGGERS, True), False)

    print("\n" + ("PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
