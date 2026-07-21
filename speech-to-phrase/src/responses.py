"""Load the home-assistant-intents response templates bundled under
``responses/<lang>/<Intent>.yaml``.

These are HA jinja2 templates (referencing ``state`` / ``slots`` / ``query``)
that we hand to Home Assistant as ``Intent.text``; HA renders them *after*
handling, with live state. A matched built-in combo selects one by its response
*key* (the ``response:`` field in the sentence file, default ``"default"``).
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import yaml

_LOGGER = logging.getLogger("speech-to-phrase.intent")


def load_responses(s2p_repo: Path, lang: str) -> Dict[str, Dict[str, str]]:
    """Return {intent: {response_key: template}} for a language.

    Sourced from the home-assistant-intents package (the ``speech_to_phrase``
    tagged responses), not the add-on's ``responses/`` tree. ``s2p_repo`` is kept
    for signature compatibility.
    """
    import s2p_intents

    out = s2p_intents.responses(lang)
    if not out:
        _LOGGER.warning("no packaged Speech-to-Phrase responses for '%s'", lang)
    else:
        _LOGGER.info("Loaded responses for %d intents (%s)", len(out), lang)
    return out


def response_for(
    responses: Dict[str, Dict[str, str]], intent: str, key: Optional[str]
) -> Optional[str]:
    """Template for (intent, key), falling back to the intent's "default"."""
    table = responses.get(intent) or {}
    return table.get(key or "default") or table.get("default")
