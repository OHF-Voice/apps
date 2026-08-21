"""Custom commands: the user-authored counterpart to the built-in combos.

A custom command is one object (persisted per language in
``<data>/<lang>/custom_commands.json``)::

    {
      "sentences": ["movie time", "start movie night"],
      "mode": "intent" | "action" | "stt",
      "name_domains": ["light", ...],     # only if a sentence uses {name}
      "response": "<jinja2>",             # spoken reply (HA-rendered)

      # mode == "intent": fire a standard HA intent
      "intent": {"name": "HassTurnOn", "slots": {"domain": "light"}},

      # mode == "action": the add-on performs it and returns Handled
      "action": {"kind": "script",  "entity_id": "script.movie_night"}
              |  {"kind": "service", "yaml": "service: light.turn_on\n..."}
    }

``mode == "stt"`` means "recognize the phrase but don't handle it" -- the
sentences go into the STT grammar only, so Home Assistant's own agent (or
another) deals with the transcript. All three modes contribute their sentences
to the grammar; only intent/action are added to the matcher.

The add-on currently ships speech-to-text only, so ``stt`` is the only mode the
web UI writes and the only one with any effect: ``intent`` and ``action`` are
handled by ``intent_server.py``, which is not started unless app.py is given
``--intent``. Loading still accepts all three so an older file keeps working.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

_LOGGER = logging.getLogger("speech-to-phrase")

FILENAME = "custom_commands.json"
_LEGACY = "custom_sentences.txt"


def load(data_dir: Path, lang: str) -> List[dict]:
    """Load custom commands, migrating a legacy custom_sentences.txt (one phrase
    per line) into STT-only commands."""
    f = data_dir / lang / FILENAME
    if f.exists():
        try:
            cmds = json.loads(f.read_text())
            return cmds if isinstance(cmds, list) else []
        except Exception:  # noqa: BLE001
            _LOGGER.exception("could not parse %s", f)
            return []

    legacy = data_dir / lang / _LEGACY
    if legacy.exists():
        lines = [
            ln.strip() for ln in legacy.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        return [{"sentences": [ln], "mode": "stt"} for ln in lines]
    return []


def save(data_dir: Path, lang: str, commands: List[dict]) -> None:
    d = data_dir / lang
    d.mkdir(parents=True, exist_ok=True)
    (d / FILENAME).write_text(json.dumps(commands, indent=2, ensure_ascii=False))


def sentences_of(cmd: dict) -> List[str]:
    return [s.strip() for s in (cmd.get("sentences") or []) if s and s.strip()]


def name_domains_of(cmd: dict) -> List[str]:
    return list(cmd.get("name_domains") or [])


def grammar_sentences(commands: List[dict]) -> List[Dict[str, object]]:
    """Sentence blocks (all modes) for the STT grammar, in the shape
    training.assemble consumes: ``[{"sentences": [...], "name_domains": [...]}]``."""
    blocks: List[Dict[str, object]] = []
    for cmd in commands:
        ss = sentences_of(cmd)
        if not ss:
            continue
        blocks.append({"sentences": ss, "name_domains": name_domains_of(cmd)})
    return blocks
