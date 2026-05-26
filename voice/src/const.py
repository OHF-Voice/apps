from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from hass_api import HomeAssistant

if TYPE_CHECKING:
    from name_resolver import NameResolver

BASE_DIR = Path(__file__).parent


@dataclass
class ToolIntent:
    name: str
    slots: Optional[Dict[str, Any]] = None


@dataclass
class Tool:
    tool: Dict[str, Any]
    intent: Optional[ToolIntent] = None
    name_domains: Optional[Set[str]] = None
    context_area: bool = False


@dataclass
class FuzzyCommand:
    intent_name: str
    sentences: List[str]
    intent_slots: Optional[Dict[str, Any]] = None
    context_area: bool = False
    duration: bool = False
    number: bool = False


@dataclass
class AppState:
    hass: HomeAssistant
    # http_host: str
    # http_port: int
    tools: Dict[str, Tool]

    fuzzy_commands: List[FuzzyCommand]
    fuzzy_candidates: List[Tuple[str, int]]

    resolver_en_model: str
    resolver_multilingual_model: str
    resolver_en: "Optional[NameResolver]" = None
    resolver_multilingual: "Optional[NameResolver]" = None

    default_area_id: Optional[str] = None
