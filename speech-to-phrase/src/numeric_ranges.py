"""Validated per-language restrictions for package numeric slot lists."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import settings

_LOGGER = logging.getLogger("speech-to-phrase.numeric_ranges")

Definition = Tuple[int, int, int]
Definitions = Mapping[str, Definition]

_ITEM_RE = re.compile(
    r"^\s*(-?\d+)(?:\s*(?:-|\.\.)\s*(-?\d+)" r"(?:\s*/\s*(\d+))?)?\s*$"
)
_REF_RE = re.compile(r"\{([^{}]+)\}")
_SINGLE_INLINE_RE = re.compile(
    r"^\s*(-?\d+)\s*\.\.\s*(-?\d+)" r"(?:\s*[,/]\s*(\d+))?\s*$"
)
_UNION_INLINE_ITEM_RE = re.compile(
    r"^\s*(-?\d+)(?:\s*\.\.\s*(-?\d+)" r"(?:\s*/\s*(\d+))?)?\s*$"
)
_MULTIPLIED_SLOT_PREFIX = "__numeric__"


class NumericRangeError(ValueError):
    """A user-entered numeric range restriction is invalid."""


@dataclass(frozen=True)
class Segment:
    """One inclusive arithmetic progression."""

    start: int
    stop: int
    step: int = 1

    def values(self) -> range:
        return range(self.start, self.stop + 1, self.step)

    def expression(self) -> str:
        if self.start == self.stop:
            return str(self.start)
        suffix = "" if self.step == 1 else f"/{self.step}"
        return f"{self.start}-{self.stop}{suffix}"

    def grammar(self) -> str:
        if self.start == self.stop:
            return str(self.start)
        suffix = "" if self.step == 1 else f"/{self.step}"
        return f"{self.start}..{self.stop}{suffix}"

    def hassil(self, slot: str) -> str:
        step = "" if self.step == 1 else f",{self.step}"
        return f"{{{self.start}..{self.stop}{step}:{slot}}}"

    def stored(self) -> Dict[str, int]:
        return {"start": self.start, "stop": self.stop, "step": self.step}


@dataclass(frozen=True)
class Selection:
    """A canonical, non-empty subset of a package numeric range."""

    values: Tuple[int, ...]
    segments: Tuple[Segment, ...]

    @property
    def expression(self) -> str:
        return ", ".join(segment.expression() for segment in self.segments)

    @property
    def grammar(self) -> str:
        return ",".join(segment.grammar() for segment in self.segments)

    def hassil(self, slot: str) -> str:
        alternatives = "|".join(segment.hassil(slot) for segment in self.segments)
        return alternatives if len(self.segments) == 1 else f"({alternatives})"

    def stored(self) -> Sequence[Dict[str, int]]:
        return [segment.stored() for segment in self.segments]


Selections = Mapping[str, Selection]
Choices = Mapping[str, Optional[Selection]]

_PERCENT_WITH_ZERO = {"position"}
_PERCENT_WITHOUT_ZERO = {
    "brightness",
    "fan_speed",
    "volume",
    "volume_level",
}
_VOLUME_STEPS = {"volume_step_up", "volume_step_down"}
_TIMER_SHORT = {
    "timer_seconds",
    "timer_minutes",
    "timer_range_seconds",
    "timer_range_minutes",
}
_TIMER_HOURS = {"timer_hours", "timer_range_hours"}


def _segments(values: Iterable[int]) -> Tuple[Segment, ...]:
    """Compress sorted values into stable arithmetic runs."""
    ordered = sorted(set(values))
    out = []
    index = 0
    while index < len(ordered):
        if index + 2 < len(ordered):
            step = ordered[index + 1] - ordered[index]
            end = index + 2
            while end < len(ordered) and ordered[end] - ordered[end - 1] == step:
                end += 1
            if end - index >= 3:
                out.append(Segment(ordered[index], ordered[end - 1], step))
                index = end
                continue
        out.append(Segment(ordered[index], ordered[index]))
        index += 1
    return tuple(out)


def selection_from_values(values: Iterable[int]) -> Selection:
    """Create a canonical selection from explicit values."""
    ordered = tuple(sorted(set(values)))
    if not ordered:
        raise NumericRangeError("at least one value is required")
    return Selection(ordered, _segments(ordered))


def package_values(definition: Definition) -> Tuple[int, ...]:
    """Every value permitted by the package range, including its base step."""
    start, stop, step = definition
    return tuple(range(start, stop + 1, step))


def recommended(name: str, definition: Definition) -> Selection:
    """Voice-friendly values for one package list, intersected with its bounds."""
    allowed = package_values(definition)
    allowed_set = set(allowed)
    start, stop, base_step = definition
    candidates: Iterable[int]

    if name in _PERCENT_WITH_ZERO:
        candidates = range(0, stop + 1, 10)
    elif name in _PERCENT_WITHOUT_ZERO:
        candidates = range(10, stop + 1, 10)
    elif name in _VOLUME_STEPS:
        candidates = range(5, min(stop, 50) + 1, 5)
    elif name in _TIMER_SHORT:
        candidates = (*range(1, 11), *range(15, stop + 1, 5))
    elif name in _TIMER_HOURS:
        candidates = (*range(1, 13), 24, 48, 72, 96)
    elif name == "temperature":
        candidates = range(5, min(stop, 35) + 1)
    elif name == "color_temperature":
        candidates = range(1000, stop + 1, 500)
    else:
        # Keep roughly 10–20 evenly spaced values for future package lists.
        step = max(base_step, ((stop - start) // 15 // base_step) * base_step)
        candidates = range(start, stop + 1, step or base_step)

    values = [value for value in candidates if value in allowed_set]
    if not values:
        values = list(allowed)
    return selection_from_values(values)


def parse(expression: str, definition: Definition) -> Selection:
    """Parse ``3, 5, 8, 10-100/10`` within a package range."""
    if not isinstance(expression, str) or not expression.strip():
        raise NumericRangeError("enter one or more values, or choose Full range")

    package_start, package_stop, package_step = definition
    values: set[int] = set()
    for raw_item in expression.split(","):
        match = _ITEM_RE.fullmatch(raw_item)
        if not match:
            raise NumericRangeError(
                f"invalid item {raw_item.strip()!r}; use N or START-END[/STEP]"
            )
        start = int(match.group(1))
        stop = int(match.group(2)) if match.group(2) is not None else start
        step = int(match.group(3) or 1)
        if start > stop:
            raise NumericRangeError(f"{start}-{stop} must be ascending")
        if step <= 0:
            raise NumericRangeError("step must be greater than zero")
        if start < package_start or stop > package_stop:
            raise NumericRangeError(
                f"{start}-{stop} is outside {package_start}-{package_stop}"
            )
        if (start - package_start) % package_step:
            raise NumericRangeError(
                f"{start} is not on the package step of {package_step}"
            )
        if (stop - package_start) % package_step:
            raise NumericRangeError(
                f"{stop} is not on the package step of {package_step}"
            )
        if start != stop and step % package_step:
            raise NumericRangeError(
                f"step {step} must be a multiple of the package step " f"{package_step}"
            )
        if start != stop and (stop - start) % step:
            raise NumericRangeError(
                f"{start}-{stop}/{step} does not land on its end value"
            )
        values.update(range(start, stop + 1, step))

    return selection_from_values(values)


def parse_payload(
    payload: Any, definitions: Definitions
) -> Dict[str, Optional[Selection]]:
    """Validate API choices; omitted entries retain the Recommended default."""
    if not isinstance(payload, dict):
        raise NumericRangeError("numeric_ranges must be an object")
    unknown = sorted(set(payload) - set(definitions))
    if unknown:
        raise NumericRangeError(f"unknown numeric list: {unknown[0]}")

    parsed: Dict[str, Optional[Selection]] = {}
    for name, expression in payload.items():
        if expression is None:
            parsed[name] = None
            continue
        if not isinstance(expression, str):
            raise NumericRangeError(f"{name} must be a range expression")
        try:
            selection = parse(expression, definitions[name])
        except NumericRangeError as err:
            raise NumericRangeError(f"{name}: {err}") from err
        parsed[name] = (
            None if selection.values == package_values(definitions[name]) else selection
        )
    return parsed


def active_selections(
    choices: Choices, definitions: Definitions
) -> Dict[str, Selection]:
    """Resolve omitted choices to Recommended and explicit ``None`` to Full."""
    active = {}
    for name, definition in definitions.items():
        selection = choices.get(name, recommended(name, definition))
        if selection is not None and selection.values != package_values(definition):
            active[name] = selection
    return active


def _selection_from_stored(value: Any, definition: Definition) -> Selection:
    if not isinstance(value, list) or not value:
        raise NumericRangeError("stored restriction must be a non-empty list")
    parts = []
    for item in value:
        if not isinstance(item, dict):
            raise NumericRangeError("stored segment must be an object")
        try:
            segment = Segment(
                start=int(item["start"]),
                stop=int(item["stop"]),
                step=int(item.get("step", 1)),
            )
        except (KeyError, TypeError, ValueError) as err:
            raise NumericRangeError("stored segment is malformed") from err
        parts.append(segment.expression())
    return parse(", ".join(parts), definition)


def load(
    data_dir: Union[str, Path], lang: str, definitions: Definitions
) -> Dict[str, Selection]:
    """Load valid restrictions, ignoring stale/corrupt entries with a warning."""
    raw = settings.load(data_dir, lang).get("numeric_ranges", {})
    if not isinstance(raw, dict):
        _LOGGER.warning("Ignoring malformed numeric_ranges setting for '%s'", lang)
        raw = {}

    choices: Dict[str, Optional[Selection]] = {}
    for name, value in raw.items():
        definition = definitions.get(name)
        if definition is None:
            _LOGGER.warning(
                "Ignoring numeric range '%s' no longer provided for '%s'", name, lang
            )
            continue
        if value is None:
            choices[name] = None
            continue
        try:
            selection = _selection_from_stored(value, definition)
        except NumericRangeError as err:
            _LOGGER.warning(
                "Ignoring invalid numeric range '%s' for '%s': %s", name, lang, err
            )
            continue
        choices[name] = selection
    return active_selections(choices, definitions)


def save(data_dir: Union[str, Path], lang: str, choices: Choices) -> None:
    """Persist canonical choices; ``None`` explicitly selects Full range."""
    settings.update(
        data_dir,
        lang,
        {
            "numeric_ranges": {
                name: selection.stored() if selection is not None else None
                for name, selection in choices.items()
            }
        },
    )


def presets(name: str, definition: Definition) -> Sequence[Dict[str, Any]]:
    """Recommended, full, and useful generic choices for one package list."""
    full = package_values(definition)
    recommended_selection = recommended(name, definition)
    choices: List[Dict[str, Any]] = [
        {
            "id": "recommended",
            "label": "Recommended",
            "expression": recommended_selection.expression,
            "count": len(recommended_selection.values),
        },
        {"id": "full", "label": "Full range", "expression": None, "count": len(full)},
    ]
    seen = {full, recommended_selection.values}
    for divisor in (5, 10):
        values = tuple(value for value in full if value % divisor == 0)
        if not values or values in seen:
            continue
        selection = selection_from_values(values)
        choices.append(
            {
                "id": f"multiples_{divisor}",
                "label": f"Multiples of {divisor}",
                "expression": selection.expression,
                "count": len(values),
            }
        )
        seen.add(values)
    choices.append(
        {"id": "custom", "label": "Advanced…", "expression": "", "count": None}
    )
    return choices


def rewrite_hassil_refs(
    text: str,
    selections: Selections,
    multipliers: Optional[Mapping[str, float]] = None,
) -> str:
    """Replace restricted named ranges with equivalent Hassil alternatives."""

    def replace(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        name, separator, slot = content.partition(":")
        name = name.strip()
        selection = selections.get(name)
        if selection is None:
            return match.group(0)
        output_slot = slot.strip() if separator else name
        multiplier = (multipliers or {}).get(name)
        if multiplier is not None:
            output_slot = f"{_MULTIPLIED_SLOT_PREFIX}{multiplier:g}__{output_slot}"
        return selection.hassil(output_slot)

    return _REF_RE.sub(replace, text)


def decode_hassil_slot(slot: str) -> Tuple[str, Optional[float]]:
    """Recover a slot name and package multiplier from a synthetic range slot."""
    if not slot.startswith(_MULTIPLIED_SLOT_PREFIX):
        return slot, None
    encoded = slot[len(_MULTIPLIED_SLOT_PREFIX) :]
    multiplier_text, separator, canonical = encoded.partition("__")
    if not separator:
        return slot, None
    try:
        multiplier = float(multiplier_text)
    except ValueError:
        return slot, None
    return canonical, multiplier


def inline_values(content: str) -> Optional[Tuple[int, ...]]:
    """Parse a trainer inline range/union for phrase-cost calculation."""
    match = _SINGLE_INLINE_RE.fullmatch(content)
    if match:
        start, stop = int(match.group(1)), int(match.group(2))
        step = int(match.group(3) or 1)
        if step <= 0 or start > stop or (stop - start) % step:
            return None
        return tuple(range(start, stop + 1, step))

    values: set[int] = set()
    for item in content.split(","):
        match = _UNION_INLINE_ITEM_RE.fullmatch(item)
        if not match:
            return None
        start = int(match.group(1))
        stop = int(match.group(2)) if match.group(2) is not None else start
        step = int(match.group(3) or 1)
        if step <= 0 or start > stop or (stop - start) % step:
            return None
        values.update(range(start, stop + 1, step))
    return tuple(sorted(values)) if values else None
