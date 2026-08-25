"""In-memory log of recent recognitions, for the web UI's debug mode.

The Wyoming STT server appends one entry per utterance (see
``wyoming_server.py``); the web UI polls ``/api/transcriptions`` and annotates
each with the sentence source it came from (see ``sources.py``). Nothing is
written to disk: this is a live view for someone watching the UI, not a record.

Written from the Wyoming thread and read from Flask request threads, hence the
lock. Bounded, so leaving debug mode on cannot grow without limit -- the oldest
entries are dropped, which is what a live view wants anyway.
"""
import threading
import time
from collections import deque
from typing import Dict, List, Optional

MAX_ENTRIES = 200

_lock = threading.Lock()
_entries: "deque[dict]" = deque(maxlen=MAX_ENTRIES)
_next_id = 1


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
