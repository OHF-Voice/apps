"""Latency + correctness benchmark over a fixed tool/sentence fixture.

The fixture (tools + sentences) is independent of any live Home Assistant
instance so results are reproducible and comparable across machines and across
model/config changes. The actual model calls happen in
``Gemma4Recognizer.run_sentences`` (which snapshots and restores the live state
so benchmarking never disturbs the running assistant); this module loads the
fixture and scores the parsed tool calls against the expected ones.
"""

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import yaml

TOOL_CALL = Tuple[str, Dict[str, Any]]


@dataclass
class Fixture:
    tools: List[Dict[str, Any]]
    sentences: List[Dict[str, Any]]
    language: str = "en"


def load_fixture(path: Union[str, Path]) -> Fixture:
    with open(path, "r", encoding="utf-8") as fixture_file:
        data = yaml.safe_load(fixture_file) or {}

    return Fixture(
        tools=data.get("tools") or [],
        sentences=data.get("sentences") or [],
        language=data.get("language") or "en",
    )


def _normalize_calls(calls: List[Dict[str, Any]]) -> List[TOOL_CALL]:
    """Order-independent, comparable representation of a list of tool calls."""
    normalized = [(c["name"], c.get("args") or {}) for c in calls]
    normalized.sort(key=lambda c: (c[0], json.dumps(c[1], sort_keys=True)))
    return normalized


def run(recognizer: Any, fixture: Fixture, passes: int = 3) -> Dict[str, Any]:
    """Run the fixture through the recognizer and score it.

    Returns a JSON-serializable dict with a ``summary`` and per-``sentences``
    results (median/min/max latency, tokens/sec, expected vs. parsed calls).
    """
    raw = recognizer.run_sentences(
        fixture.tools,
        fixture.sentences,
        passes=passes,
        default_language=fixture.language,
    )

    sentence_results: List[Dict[str, Any]] = []
    median_latencies: List[float] = []
    total_completion_tokens = 0
    num_passed = 0

    for fixture_sentence, result in zip(fixture.sentences, raw["sentences"]):
        latencies = [p["latency"] for p in result["passes"]]
        median_latency = statistics.median(latencies)
        # Greedy decoding is deterministic, so output is identical across passes.
        completion_tokens = result["passes"][-1]["completion_tokens"] or 0
        tps = (completion_tokens / median_latency) if median_latency > 0 else None

        expected = _normalize_calls(fixture_sentence.get("expected") or [])
        got = _normalize_calls(result["tool_calls"])
        passed = expected == got
        if passed:
            num_passed += 1

        median_latencies.append(median_latency)
        total_completion_tokens += completion_tokens

        sentence_results.append(
            {
                "text": result["text"],
                "language": result["language"],
                "passed": passed,
                "expected": [{"name": n, "args": a} for n, a in expected],
                "got": result["tool_calls"],
                "median_latency_ms": median_latency * 1000.0,
                "min_latency_ms": min(latencies) * 1000.0,
                "max_latency_ms": max(latencies) * 1000.0,
                "completion_tokens": completion_tokens,
                "tokens_per_second": tps,
                "content": result["content"],
            }
        )

    num_sentences = len(sentence_results)
    total_latency = sum(median_latencies)
    summary = {
        "num_sentences": num_sentences,
        "num_passed": num_passed,
        "pass_rate": (num_passed / num_sentences) if num_sentences else None,
        "passes": raw["passes"],
        "num_tools": raw["num_tools"],
        "rebuild_seconds": raw["rebuild_seconds"],
        "mean_latency_ms": (
            statistics.fmean(median_latencies) * 1000.0 if median_latencies else None
        ),
        "median_latency_ms": (
            statistics.median(median_latencies) * 1000.0 if median_latencies else None
        ),
        "max_latency_ms": (
            max(median_latencies) * 1000.0 if median_latencies else None
        ),
        "overall_tokens_per_second": (
            (total_completion_tokens / total_latency) if total_latency > 0 else None
        ),
        "config": raw["config"],
    }

    return {"summary": summary, "sentences": sentence_results}
