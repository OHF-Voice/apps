"""Per-language settings persisted to ``<data>/<lang>/settings.json``.

Each one has an add-on option supplying the default for every language, and a
web-UI override for one language:

  * ``max_score`` -- the max per-token score at/below which a decode is accepted
    (lower = more confident; above it the utterance is gated to an empty
    transcript so Home Assistant can fall back to cloud STT). The Wyoming STT
    server re-reads it on each utterance, so a change takes effect without a
    restart or a retrain.
  * ``sentence_triggers`` / ``question_answers`` -- whether to pull the phrases
    Home Assistant is already listening for into the grammar (see
    ``hass_sentences.py``). These *are* the grammar, so changing one retrains.

Debug mode is deliberately *not* here: it stops the add-on answering Home
Assistant, so it must not outlive the session that turned it on (see
``debug_log``).
"""
import json
import logging
from pathlib import Path
from typing import Optional, Union

_LOGGER = logging.getLogger(__name__)

FILENAME = "settings.json"

# Guard rails for a user-entered gate. The fitted default is ~5.0 (citrinet);
# well below ~1 nothing matches, well above ~15 even OOV noise is accepted.
MIN_MAX_SCORE = 0.1
MAX_MAX_SCORE = 50.0


def path(data_dir: Union[str, Path], lang: str) -> Path:
    return Path(data_dir) / lang / FILENAME


def _read(p: Path) -> dict:
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001 -- corrupt file shouldn't break serving
            _LOGGER.warning("Ignoring unreadable settings file %s", p)
    return {}


def load(data_dir: Union[str, Path], lang: str) -> dict:
    return _read(path(data_dir, lang))


def _coerce_max_score(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return min(MAX_MAX_SCORE, max(MIN_MAX_SCORE, v))


def get_max_score(data_dir: Union[str, Path], lang: str, default: float) -> float:
    """Persisted gate for `lang`, or `default` if unset/invalid."""
    v = load(data_dir, lang).get("max_score")
    return default if v is None else _coerce_max_score(v, default)


def read_max_score_file(settings_path: Path, default: float) -> float:
    """Same as get_max_score but from an explicit file path (for the Wyoming
    server, which knows its grammar dir but not data_dir/lang)."""
    v = _read(settings_path).get("max_score")
    return default if v is None else _coerce_max_score(v, default)


def get_bool(data_dir: Union[str, Path], lang: str, key: str, default: bool) -> bool:
    """Persisted flag for `lang`, or `default` if unset. Anything stored that
    isn't a bool is ignored rather than coerced -- `"false"` reading as True is
    exactly the kind of surprise a grammar-affecting switch shouldn't have."""
    v = load(data_dir, lang).get(key)
    return v if isinstance(v, bool) else default


def set_bool(data_dir: Union[str, Path], lang: str, key: str, value) -> Optional[bool]:
    """Persist a flag for `lang`. Returns the stored value, or None if `value`
    was not a bool, in which case nothing is written and the previous setting
    (or the add-on option's default) stands."""
    if not isinstance(value, bool):
        _LOGGER.warning("Ignoring non-boolean %s=%r for '%s'", key, value, lang)
        return None
    p = path(data_dir, lang)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _read(p)
    data[key] = value
    p.write_text(json.dumps(data, indent=2))
    return value


def set_max_score(data_dir: Union[str, Path], lang: str, value) -> Optional[float]:
    """Persist the gate for `lang` (clamped to the valid range). Returns the
    stored value, or None if `value` could not be parsed, in which case nothing
    is written and the previous setting (or the per-backend default) stands.

    Writing on a parse failure used to pin the gate at MIN_MAX_SCORE, i.e. 0.1 --
    a value that accepts nothing, silently killing recognition."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        _LOGGER.warning("Ignoring unparseable max_score %r for '%s'", value, lang)
        return None
    p = path(data_dir, lang)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _read(p)
    data["max_score"] = stored = min(MAX_MAX_SCORE, max(MIN_MAX_SCORE, parsed))
    p.write_text(json.dumps(data, indent=2))
    return stored
