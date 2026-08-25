"""Debug mode: the switch, and the log of what the recognizer heard.

The Wyoming STT server asks :func:`enabled` per utterance and, when it is on,
appends an entry here and tells Home Assistant nothing (see
``wyoming_server.py``); the web UI polls ``/api/transcriptions`` and annotates
each entry with the sentence source it came from (see ``sources.py``).

Both live in memory and neither is written to disk, so **debug mode does not
survive a restart**. That is deliberate: while it is on the add-on answers Home
Assistant with an empty transcript, so voice does nothing. A diagnostic that
silently outlives the session that turned it on would leave someone with a
broken assistant and no memory of why -- and a restart is the first thing they
would try. Off is the only safe state to come back up in.

Unlike ``max_score`` there is nothing per-language here: one recognizer runs, and
debug mode observes it.

Written from the Wyoming thread and read from Flask request threads, hence the
lock. The log is bounded, so leaving debug mode on cannot grow without limit --
the oldest entries are dropped, which is what a live view wants anyway.
"""
import threading
import time
from collections import deque
from typing import Dict, List, Optional

MAX_ENTRIES = 200

_lock = threading.Lock()
_entries: "deque[dict]" = deque(maxlen=MAX_ENTRIES)
_next_id = 1
_enabled = False


def enabled() -> bool:
    return _enabled


def set_enabled(on: bool) -> bool:
    """Turn debug mode on or off. Switching off discards the log: it described a
    session that has ended, and keeping it would make a stale feed look live."""
    global _enabled  # noqa: PLW0603
    _enabled = bool(on)
    if not _enabled:
        clear()
    return _enabled


def record(
    language: str,
    text: str,
    score: float,
    margin: float,
    accepted: bool,
    max_score: float,
    duration: Optional[float] = None,
) -> dict:
    """Append one recognition. ``accepted`` is what the score gate decided, not
    what Home Assistant received -- in debug mode HA always gets nothing."""
    global _next_id  # noqa: PLW0603
    with _lock:
        entry = {
            "id": _next_id,
            "at": time.time(),
            "language": language,
            "text": text,
            # inf is not valid JSON; the UI shows "no match" for a null score.
            "score": None if score != score or score in (float("inf"),) else round(score, 3),
            "margin": None if margin in (float("inf"),) or margin != margin else round(margin, 3),
            "accepted": bool(accepted),
            "max_score": max_score,
            "duration": round(duration, 2) if duration is not None else None,
        }
        _next_id += 1
        _entries.append(entry)
    return entry


def entries(since: int = 0) -> List[dict]:
    """Entries with id > `since`, oldest first, so the UI can poll for deltas."""
    with _lock:
        return [dict(e) for e in _entries if e["id"] > since]


def last_id() -> int:
    with _lock:
        return _entries[-1]["id"] if _entries else 0


def clear() -> None:
    with _lock:
        _entries.clear()


def stats() -> Dict[str, int]:
    with _lock:
        return {"count": len(_entries), "last_id": _entries[-1]["id"] if _entries else 0}
