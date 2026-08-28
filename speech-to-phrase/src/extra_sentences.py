"""User 'extra phrasings' added to built-in commands.

Stored per language in ``<data>/<lang>/extra_sentences.json`` as
``{"<intent>/<combo>": ["template", ...]}``. Each extra phrasing is merged into
that combo as one more data block (see ``training.combo_blocks``), inheriting the
combo's intent, response, and slot semantics -- so e.g. adding
``{0..100:minutes} minute timer`` to ``HassStartTimer/minutes_only`` makes
"5 minute timer" work without re-specifying the intent.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

_LOGGER = logging.getLogger("speech-to-phrase")

FILENAME = "extra_sentences.json"


def key(intent: str, combo: str) -> str:
    return f"{intent}/{combo}"


def load(data_dir: Path, lang: str) -> Dict[str, List[str]]:
    f = data_dir / lang / FILENAME
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text())
    except Exception:  # noqa: BLE001
        _LOGGER.exception("could not parse %s", f)
        return {}
    if not isinstance(d, dict):
        return {}
    return {
        k: [s.strip() for s in v if isinstance(s, str) and s.strip()]
        for k, v in d.items()
        if isinstance(v, list)
    }


def save(data_dir: Path, lang: str, extras: Dict[str, List[str]]) -> None:
    clean = {
        k: [s.strip() for s in (v or []) if s and s.strip()]
        for k, v in (extras or {}).items()
    }
    clean = {k: v for k, v in clean.items() if v}  # drop empty
    d = data_dir / lang
    d.mkdir(parents=True, exist_ok=True)
    (d / FILENAME).write_text(json.dumps(clean, indent=2, ensure_ascii=False))
