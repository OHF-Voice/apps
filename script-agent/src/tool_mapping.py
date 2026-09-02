"""Map model tool calls back to Home Assistant script calls."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from hass_api import HomeAssistantInfo, SatelliteInfo, Tool

_LOGGER = logging.getLogger(__name__)


@dataclass
class HomeGeometry:
    """Where things are, for picking between things that share a name."""

    # Entity id -> the area it is in (directly, or via its device).
    entity_areas: Dict[str, Optional[str]] = field(default_factory=dict)
    # Area id -> the floor it is on.
    area_floors: Dict[str, Optional[str]] = field(default_factory=dict)

    @classmethod
    def from_info(cls, info: HomeAssistantInfo) -> "HomeGeometry":
        return cls(
            entity_areas={
                entity_id: entity.area_id for entity_id, entity in info.entities.items()
            },
            area_floors={
                area_id: area.floor_id for area_id, area in info.areas.items()
            },
        )

    def area_of(self, target_id: str) -> Optional[str]:
        """The area a candidate is in. An area is its own area."""
        if "." in target_id:
            # An entity id; anything else is an area or floor id.
            return self.entity_areas.get(target_id)

        if target_id in self.area_floors:
            return target_id

        return None

    def floor_of(self, target_id: str) -> Optional[str]:
        """The floor a candidate is on. A floor is its own floor."""
        if "." in target_id:
            area_id = self.entity_areas.get(target_id)
            return self.area_floors.get(area_id) if area_id else None

        if target_id in self.area_floors:
            return self.area_floors.get(target_id)

        # Not an area, so assume a floor id.
        return target_id


@dataclass
class MappedCall:
    """One tool call resolved against the tool that produced it."""

    tool_id: str
    # Variables to pass to the script, with names resolved to ids.
    variables: Dict[str, Any] = field(default_factory=dict)
    # Field -> value the model produced that is not in that field's enum.
    unresolved: Dict[str, Any] = field(default_factory=dict)
    # Field -> the candidate ids of a name that refers to more than one thing
    # and could not be narrowed down.
    ambiguous: Dict[str, List[str]] = field(default_factory=dict)
    # Fields the script requires that could not be filled in.
    missing_required: List[str] = field(default_factory=list)

    @property
    def script_id(self) -> str:
        return f"script.{self.tool_id}"

    @property
    def can_run(self) -> bool:
        """False when a field the script requires could not be resolved."""
        return not self.missing_required


def required_fields(tool: Tool) -> Set[str]:
    """Field names the script marks as required."""
    params = (tool.tool.get("function") or {}).get("parameters") or {}
    return set(params.get("required") or [])


def _narrow(
    candidates: List[str],
    satellite: Optional[SatelliteInfo],
    geometry: Optional[HomeGeometry],
) -> List[str]:
    """Narrow candidates that share a name using where the command came from.

    Several things can legitimately share a name -- "Media Player" in three
    rooms, an "Office" area and an office speaker. The satellite that heard the
    command is the best available hint about which one was meant: prefer one in
    the same area, then one on the same floor. Returns the candidates that
    survive, which is more than one when the hint does not settle it.
    """
    if (len(candidates) <= 1) or (satellite is None) or (geometry is None):
        return candidates

    for hint, locate in (
        (satellite.area_id, geometry.area_of),
        (satellite.floor_id, geometry.floor_of),
    ):
        if not hint:
            continue

        nearby = [target for target in candidates if locate(target) == hint]
        if len(nearby) == 1:
            return nearby

        if nearby:
            # Narrowed but still tied; keep going with the smaller set.
            candidates = nearby

    return candidates


def _matches_type(value: Any, expected_type: Optional[str]) -> bool:
    """Check the JSON types used by script field schemas."""
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True


def _valid_value(schema: Dict[str, Any], value: Any, check_enum: bool = True) -> bool:
    """Validate a model value against the generated script field schema."""
    if not _matches_type(value, schema.get("type")):
        return False

    allowed = schema.get("enum")
    if check_enum and (allowed is not None) and (value not in allowed):
        return False

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if (minimum is not None) and (value < minimum):
            return False
        maximum = schema.get("maximum")
        if (maximum is not None) and (value > maximum):
            return False
        multiple_of = schema.get("multipleOf")
        if multiple_of is not None:
            quotient = value / multiple_of
            if abs(quotient - round(quotient)) > 1e-9:
                return False

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if (min_items is not None) and (len(value) < min_items):
            return False
        max_items = schema.get("maxItems")
        if (max_items is not None) and (len(value) > max_items):
            return False
        if schema.get("uniqueItems"):
            for item_index, item in enumerate(value):
                if item in value[:item_index]:
                    return False

        item_schema = schema.get("items") or {}
        return all(
            _valid_value(item_schema, item, check_enum=check_enum) for item in value
        )

    return True


def map_tool_call(
    tool: Tool,
    tool_args: Dict[str, Any],
    satellite: Optional[SatelliteInfo] = None,
    geometry: Optional[HomeGeometry] = None,
) -> MappedCall:
    """Resolve the names a model produced into Home Assistant ids.

    Fields with no name map (free text, numbers, dates, ...) pass through
    unchanged. Fields with a name map hold a display name that has to resolve to
    an entity/area/floor id, and enums only steer generation, so the model can
    invent a name that is not in the list. An unresolvable name is dropped rather
    than passed along: forwarding the display name would make the script fail on
    its own target, and forwarding ``None`` would look to the script like a
    deliberately empty value.

    A name that refers to several things is narrowed using the satellite's area
    or floor when possible, and otherwise dropped as well -- picking arbitrarily
    would act on whichever one happened to be listed first.
    """
    call = MappedCall(tool_id=tool.name)
    required = required_fields(tool)
    properties = ((tool.tool.get("function") or {}).get("parameters") or {}).get(
        "properties"
    ) or {}

    for var_key, var_value in tool_args.items():
        if var_key not in properties:
            _LOGGER.warning(
                "Dropping unknown field %s from tool %s", var_key, tool.name
            )
            continue

        field_schema = properties[var_key]
        expects_array = field_schema.get("type") == "array"
        name_map = tool.name_map.get(var_key)

        if not _valid_value(field_schema, var_value, check_enum=not bool(name_map)):
            _LOGGER.warning(
                "Dropping malformed field %s=%r from tool %s",
                var_key,
                var_value,
                tool.name,
            )
            call.unresolved[var_key] = var_value
            continue

        # An enum and its name map are always generated together, so an empty
        # map means this field is not name-mapped at all.
        if not name_map:
            call.variables[var_key] = var_value
            continue

        if expects_array:
            if not isinstance(var_value, list):
                _LOGGER.warning(
                    "Dropping scalar value for array field %s=%r from tool %s",
                    var_key,
                    var_value,
                    tool.name,
                )
                call.unresolved[var_key] = var_value
                continue

            mapped_values: List[str] = []
            unresolved_values: List[Any] = []
            ambiguous_values: List[str] = []
            for item in var_value:
                try:
                    candidates = name_map.get(item)
                except TypeError:
                    candidates = None
                if not candidates:
                    unresolved_values.append(item)
                    continue

                narrowed = _narrow(sorted(candidates), satellite, geometry)
                if len(narrowed) > 1:
                    ambiguous_values.extend(narrowed)
                    continue

                mapped_value = narrowed[0]
                if mapped_value not in mapped_values:
                    mapped_values.append(mapped_value)

            if unresolved_values:
                _LOGGER.debug(
                    "No mapping for %s=%r in tool %s",
                    var_key,
                    unresolved_values,
                    tool.name,
                )
                call.unresolved[var_key] = unresolved_values
            if ambiguous_values:
                _LOGGER.warning(
                    "%s in tool %s contains ambiguous names referring to: %s",
                    var_key,
                    tool.name,
                    ambiguous_values,
                )
                call.ambiguous[var_key] = sorted(set(ambiguous_values))
            if unresolved_values or ambiguous_values:
                continue

            call.variables[var_key] = mapped_values
            continue

        try:
            candidates = name_map.get(var_value)
        except TypeError:
            # A malformed model response may put a list or object in a scalar
            # enum field. Treat it like any other unknown name.
            candidates = None
        if not candidates:
            _LOGGER.debug(
                "No mapping for %s=%r in tool %s", var_key, var_value, tool.name
            )
            call.unresolved[var_key] = var_value
            continue

        narrowed = _narrow(sorted(candidates), satellite, geometry)
        if len(narrowed) > 1:
            _LOGGER.warning(
                "%s=%r in tool %s refers to %s things and the satellite did not "
                "narrow it down: %s",
                var_key,
                var_value,
                tool.name,
                len(narrowed),
                narrowed,
            )
            call.ambiguous[var_key] = narrowed
            continue

        mapped_value = narrowed[0]
        _LOGGER.debug("Mapping '%s' -> '%s'", var_value, mapped_value)
        call.variables[var_key] = mapped_value

    # This catches both fields that were supplied but could not be resolved and
    # required fields the model omitted entirely.
    call.missing_required = sorted(required - set(call.variables))
    return call
