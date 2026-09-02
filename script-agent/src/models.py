"""Basic versions of Home Assistant models."""

import itertools
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, List, Optional

from util import normalize_name

ATTR_SUPPORTED_FEATURES: Final = "supported_features"
ATTR_DEVICE_CLASS: Final = "device_class"
ATTR_FRIENDLY_NAME: Final = "friendly_name"


class ColorMode(str, Enum):
    """Possible light color modes."""

    UNKNOWN = "unknown"
    """Ambiguous color mode"""
    ONOFF = "onoff"
    """Must be the only supported mode"""
    BRIGHTNESS = "brightness"
    """Must be the only supported mode"""
    COLOR_TEMP = "color_temp"
    HS = "hs"
    XY = "xy"
    RGB = "rgb"
    RGBW = "rgbw"
    RGBWW = "rgbww"
    WHITE = "white"
    """Must *NOT* be the only supported mode"""


VALID_COLOR_MODES = {
    ColorMode.ONOFF,
    ColorMode.BRIGHTNESS,
    ColorMode.COLOR_TEMP,
    ColorMode.HS,
    ColorMode.XY,
    ColorMode.RGB,
    ColorMode.RGBW,
    ColorMode.RGBWW,
    ColorMode.WHITE,
}
COLOR_MODES_BRIGHTNESS = VALID_COLOR_MODES - {ColorMode.ONOFF}
COLOR_MODES_COLOR = {
    ColorMode.HS,
    ColorMode.RGB,
    ColorMode.RGBW,
    ColorMode.RGBWW,
    ColorMode.XY,
}


@dataclass
class State:
    entity_id: str
    state: Any
    attributes: Dict[str, Any] = field(default_factory=dict)
    domain: str = field(init=False)
    entity_name: Optional[str] = None

    def __post_init__(self) -> None:
        self.domain = self.entity_id.split(".", maxsplit=1)[0]

    @property
    def name(self) -> str:
        if self.entity_name:
            return self.entity_name

        return self.entity_id.replace("_", " ")

    @property
    def state_with_unit(self) -> Any:
        unit = self.attributes.get("unit_of_measurement")
        if unit:
            return f"{self.state} {unit}"
        return self.state


@dataclass
class Entity:
    entity_id: str
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    domain: str = field(init=False)
    aliases: Optional[List[str]] = None
    device_id: Optional[str] = None
    area_id: Optional[str] = None

    _names_norm: Optional[List[str]] = None
    _light_color_modes: Optional[List[ColorMode]] = None

    def __post_init__(self) -> None:
        self.domain = self.entity_id.split(".", maxsplit=1)[0]
        if self.aliases:
            self.aliases = [a for a in self.aliases if a.strip()]

        color_modes = self.attributes.get("supported_color_modes")
        if color_modes:
            self._light_color_modes = []
            for color_mode_str in color_modes:
                try:
                    self._light_color_modes.append(ColorMode(color_mode_str))
                except ValueError:
                    continue

    @property
    def names(self) -> Iterable[str]:
        if not self.aliases:
            return [self.name]

        return itertools.chain([self.name], self.aliases)

    @property
    def names_norm(self) -> List[str]:
        if self._names_norm is None:
            self._names_norm = [normalize_name(name) for name in self.names]

        return self._names_norm

    @property
    def supported_features(self) -> int:
        if not self.attributes:
            return 0

        return self.attributes.get(ATTR_SUPPORTED_FEATURES, 0)

    @property
    def device_class(self) -> Optional[str]:
        if not self.attributes:
            return None

        return self.attributes.get(ATTR_DEVICE_CLASS)

    @property
    def light_supports_brightness(self) -> bool:
        if not self._light_color_modes:
            return False

        for color_mode in self._light_color_modes:
            if color_mode in COLOR_MODES_BRIGHTNESS:
                return True

        return False

    @property
    def light_supports_color(self) -> bool:
        if not self._light_color_modes:
            return False

        for color_mode in self._light_color_modes:
            if color_mode in COLOR_MODES_COLOR:
                return True

        return False


@dataclass
class Area:
    area_id: str
    name: str
    aliases: Optional[List[str]] = None
    floor_id: Optional[str] = None

    _names_norm: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.aliases:
            self.aliases = [a for a in self.aliases if a.strip()]

    @property
    def names(self) -> Iterable[str]:
        if not self.aliases:
            return [self.name]

        return itertools.chain([self.name], self.aliases)

    @property
    def names_norm(self) -> List[str]:
        if self._names_norm is None:
            self._names_norm = [normalize_name(name) for name in self.names]

        return self._names_norm


@dataclass
class Floor:
    floor_id: str
    name: str
    aliases: Optional[List[str]] = None
    name_norm: str = field(init=False)

    _names_norm: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.name_norm = normalize_name(self.name)
        if self.aliases:
            self.aliases = [a for a in self.aliases if a.strip()]

    @property
    def names(self) -> Iterable[str]:
        if not self.aliases:
            return [self.name]

        return itertools.chain([self.name], self.aliases)

    @property
    def names_norm(self) -> List[str]:
        if self._names_norm is None:
            self._names_norm = [normalize_name(name) for name in self.names]

        return self._names_norm
