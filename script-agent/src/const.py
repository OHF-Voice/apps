import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from hass_api import HomeAssistant, HomeAssistantInfo, Tool
from overrides import Overrides
from tool_mapping import HomeGeometry

BASE_DIR = Path(__file__).parent
APP_NAME = "Script Agent"
APP_SLUG = "script-agent"
APP_VERSION = "1.1.0"

if TYPE_CHECKING:
    from gemma4_recognizer import Gemma4Recognizer


@dataclass
class AppState:
    hass: HomeAssistant
    hass_info: HomeAssistantInfo
    # Tools the model can call, by tool name. Only targeted scripts belong here:
    # this is what the Wyoming handler looks a tool call up in before running it.
    tools: Dict[str, Tool]
    recognizer: "Gemma4Recognizer"
    # Single worker serializes all access to the (non-thread-safe) model, shared
    # between live recognition and the benchmark endpoint.
    llama_executor: ThreadPoolExecutor
    # Every script in Home Assistant, targeted or not, for the web UI to show.
    all_tools: List[Tool] = field(default_factory=list)
    # Local corrections to exposure and names, and where they are persisted
    # (None when no path was configured).
    overrides: Overrides = field(default_factory=Overrides)
    overrides_path: Optional[Path] = None
    # Held while the tool set is being swapped, so two applies cannot overlap.
    reload_lock: threading.Lock = field(default_factory=threading.Lock)
    # Set before a model-prefix rebuild is queued. Voice requests use this to
    # fail immediately instead of waiting behind a rebuild and running later.
    model_rebuilding: threading.Event = field(default_factory=threading.Event)
    # The app's event loop, so the web thread can re-read Home Assistant.
    loop: Optional[asyncio.AbstractEventLoop] = None
    # Where things are, for picking between things that share a name.
    geometry: HomeGeometry = field(default_factory=HomeGeometry)

    def set_home_info(self, info: HomeAssistantInfo) -> None:
        """Adopt a new Home Assistant snapshot."""
        self.hass_info = info
        self.geometry = HomeGeometry.from_info(info)

    def set_targeted(self, tools: List[Tool]) -> None:
        """Replace the set of tools the model may call."""
        self.tools = {tool.name: tool for tool in tools}
