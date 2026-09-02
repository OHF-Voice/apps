"""Local corrections to what the model is offered, on top of Home Assistant.

Three independent things live here.

**Which scripts are targeted.** Home Assistant's "expose to voice" is the
default: a script exposed there is targeted here. Only the deliberate exceptions
are recorded, so exposing something new in Home Assistant is still picked up
automatically::

    scripts:
      enabled:  [peppa_pig_sound]   # not exposed in HA, but targeted here
      disabled: [get_date]          # exposed in HA, but not targeted here

**What things are called.** Enum values come from Home Assistant names and
aliases, which are not always what someone says out loud, and sometimes include
entities whose names only confuse the model. Listing a target here replaces its
names entirely; an empty list keeps it out of the enums altogether::

    names:
      entity:
        light.desk_lamp: [Desk Lamp, the big lamp]
        media_player.built_in_audio: []          # never offered to the model
      area:
        kitchen: [Kitchen, the cookhouse]
      floor:
        first_floor: [Downstairs]

Both are keyed by Home Assistant id, so entries can be checked against what
still exists and dropped when it does not.

**Web settings.** Values changed in the app's web UI are stored under
``settings`` and override their startup defaults.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set

import yaml

if TYPE_CHECKING:
    from hass_api import HomeAssistantInfo, Tool

_LOGGER = logging.getLogger(__name__)

# The kinds of thing a field's names can refer to.
ENTITY = "entity"
AREA = "area"
FLOOR = "floor"
NAME_KINDS = (ENTITY, AREA, FLOOR)


@dataclass
class ScriptOverrides:
    """Deliberate exceptions to Home Assistant's voice exposure."""

    # Scripts targeted even though they are not exposed to voice.
    enabled: Set[str] = field(default_factory=set)
    # Scripts not targeted even though they are exposed to voice.
    disabled: Set[str] = field(default_factory=set)

    def targets(self, tool: "Tool") -> bool:
        """True when this script should be given to the model."""
        if tool.name in self.disabled:
            return False

        return tool.exposed or (tool.name in self.enabled)

    def select(self, all_tools: List["Tool"]) -> List["Tool"]:
        """The tools the model should be given, in the given order."""
        return [tool for tool in all_tools if self.targets(tool)]

    def as_dict(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "scripts": {
                "enabled": sorted(self.enabled),
                "disabled": sorted(self.disabled),
            }
        }

    def prune(self, known_scripts: Set[str]) -> Set[str]:
        """Drop entries naming scripts that no longer exist.

        Returns the names that were dropped. Callers must only pass a script
        list they know to be complete: an empty or partial list (Home Assistant
        still starting, an integration not yet loaded) is indistinguishable from
        every script having been deleted, and pruning against it would quietly
        discard the user's choices.
        """
        dropped = (self.enabled | self.disabled) - known_scripts
        if dropped:
            self.enabled -= dropped
            self.disabled -= dropped

        return dropped


@dataclass
class NameOverrides:
    """What things are called, replacing Home Assistant's names and aliases."""

    # Home Assistant id -> the names the model may use for it. An empty list
    # keeps the target out of the enums entirely.
    by_kind: Dict[str, Dict[str, List[str]]] = field(
        default_factory=lambda: {kind: {} for kind in NAME_KINDS}
    )
    # Script -> field -> id -> names, for when one script's field needs different
    # names than everywhere else. Beats ``by_kind`` for that field only.
    per_script: Dict[str, Dict[str, Dict[str, List[str]]]] = field(default_factory=dict)

    def names_for(
        self,
        kind: str,
        target_id: str,
        default_names: Iterable[str],
        script: Optional[str] = None,
        field_key: Optional[str] = None,
    ) -> List[str]:
        """The names to offer for one target, honouring any override.

        A per-script override for this field wins, then a global one for this
        kind, then Home Assistant's own names.
        """
        if (script is not None) and (field_key is not None):
            scoped = self.field_names(script, field_key).get(target_id)
            if scoped is not None:
                return [name for name in scoped if name]

        override = self.by_kind.get(kind, {}).get(target_id)
        if override is None:
            return [name for name in default_names if name]

        return [name for name in override if name]

    def field_names(self, script: str, field_key: str) -> Dict[str, List[str]]:
        """Per-script overrides for one field, empty when there are none."""
        return self.per_script.get(script, {}).get(field_key, {})

    def has_field_names(self, script: str, field_key: str) -> bool:
        return bool(self.field_names(script, field_key))

    def set_field_names(
        self, script: str, field_key: str, names: Dict[str, List[str]]
    ) -> None:
        """Replace the per-script overrides for one field."""
        fields = self.per_script.setdefault(script, {})
        if names:
            fields[field_key] = names
        else:
            fields.pop(field_key, None)

        if not fields:
            self.per_script.pop(script, None)

    def is_overridden(self, kind: str, target_id: str) -> bool:
        return target_id in self.by_kind.get(kind, {})

    def set_names(self, kind: str, target_id: str, names: List[str]) -> None:
        self.by_kind.setdefault(kind, {})[target_id] = names

    def clear(self, kind: str, target_id: str) -> None:
        """Go back to Home Assistant's names for one target."""
        self.by_kind.get(kind, {}).pop(target_id, None)

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            kind: {
                target_id: list(names)
                for target_id, names in sorted(self.by_kind.get(kind, {}).items())
            }
            for kind in NAME_KINDS
        }
        data["per_script"] = {
            script: {
                field_key: {
                    target_id: list(names)
                    for target_id, names in sorted(targets.items())
                }
                for field_key, targets in sorted(fields.items())
            }
            for script, fields in sorted(self.per_script.items())
        }
        return data

    def prune(
        self, known_ids: Dict[str, Set[str]], known_scripts: Optional[Set[str]] = None
    ) -> Dict[str, Set[str]]:
        """Drop overrides for targets and scripts that no longer exist.

        ``known_ids`` maps kind to the ids that exist now, and must be complete
        for every kind it lists -- see ``ScriptOverrides.prune``. Kinds missing
        from it are left alone.
        """
        dropped: Dict[str, Set[str]] = {}
        for kind, ids in known_ids.items():
            gone = set(self.by_kind.get(kind, {})) - ids
            if gone:
                for target_id in gone:
                    del self.by_kind[kind][target_id]
                dropped[kind] = gone

        # Per-script overrides die with their script, and lose entries for
        # targets that are gone. The field's kind is not recorded, so check
        # against every id we were given.
        all_ids: Set[str] = set()
        for ids in known_ids.values():
            all_ids |= ids

        gone_scripts = (
            set(self.per_script) - known_scripts if known_scripts is not None else set()
        )
        for script in gone_scripts:
            del self.per_script[script]
        if gone_scripts:
            dropped["script fields"] = gone_scripts

        gone_targets: Set[str] = set()
        for script, fields in list(self.per_script.items()):
            for field_key, targets in list(fields.items()):
                stale = set(targets) - all_ids
                for target_id in stale:
                    del targets[target_id]
                gone_targets |= stale
                if not targets:
                    del fields[field_key]

            if not fields:
                del self.per_script[script]

        if gone_targets:
            dropped["script field targets"] = gone_targets

        return dropped


@dataclass
class Overrides:
    """Everything the user has chosen to override locally."""

    scripts: ScriptOverrides = field(default_factory=ScriptOverrides)
    names: NameOverrides = field(default_factory=NameOverrides)
    max_tokens: Optional[int] = None
    # Prompts as edited in the web UI. None means "use the startup default",
    # which is how a reset is recorded.
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            **self.scripts.as_dict(),
            "names": self.names.as_dict(),
        }
        settings: Dict[str, object] = {}
        if self.max_tokens is not None:
            settings["max_tokens"] = self.max_tokens
        if self.system_prompt is not None:
            settings["system_prompt"] = self.system_prompt
        if self.user_prompt is not None:
            settings["user_prompt"] = self.user_prompt
        if settings:
            data["settings"] = settings
        return data


def load(path: Optional[Path]) -> Overrides:
    """Read the overrides file, falling back to "no overrides"."""
    if (path is None) or (not path.exists()):
        _LOGGER.debug("No overrides file: %s", path)
        return Overrides()

    try:
        with open(path, "r", encoding="utf-8") as overrides_file:
            data = yaml.safe_load(overrides_file) or {}
    except (OSError, yaml.YAMLError) as err:
        # A corrupt file must not stop the app from starting: falling back to
        # Home Assistant's own exposure and names is always a safe answer.
        _LOGGER.warning("Ignoring unreadable overrides file %s: %s", path, err)
        return Overrides()

    if not isinstance(data, dict):
        _LOGGER.warning("Ignoring overrides file %s: not a mapping", path)
        return Overrides()

    scripts = data.get("scripts") or {}
    script_overrides = ScriptOverrides(
        enabled=_name_set(scripts.get("enabled")),
        disabled=_name_set(scripts.get("disabled")),
    )

    # Disabling wins, so a name in both sets is meaningless. Keep the file
    # honest rather than silently applying a rule the user cannot see.
    both = script_overrides.enabled & script_overrides.disabled
    if both:
        _LOGGER.warning(
            "Script(s) %s are both enabled and disabled in %s; treating as disabled",
            sorted(both),
            path,
        )
        script_overrides.enabled -= both

    names = data.get("names") or {}
    name_overrides = NameOverrides()
    if isinstance(names, dict):
        for kind in NAME_KINDS:
            for target_id, target_names in (names.get(kind) or {}).items():
                if not isinstance(target_names, list):
                    _LOGGER.warning(
                        "Ignoring %s name override for %s in %s: not a list",
                        kind,
                        target_id,
                        path,
                    )
                    continue

                name_overrides.set_names(
                    kind,
                    str(target_id),
                    [str(name).strip() for name in target_names if str(name).strip()],
                )

        for script, fields in (names.get("per_script") or {}).items():
            if not isinstance(fields, dict):
                _LOGGER.warning(
                    "Ignoring per-script names for %s in %s: not a mapping",
                    script,
                    path,
                )
                continue

            for field_key, targets in fields.items():
                if not isinstance(targets, dict):
                    _LOGGER.warning(
                        "Ignoring per-script names for %s.%s in %s: not a mapping",
                        script,
                        field_key,
                        path,
                    )
                    continue

                scoped = {
                    str(target_id): [
                        str(name).strip() for name in target_names if str(name).strip()
                    ]
                    for target_id, target_names in targets.items()
                    if isinstance(target_names, list)
                }
                name_overrides.set_field_names(str(script), str(field_key), scoped)

    _LOGGER.debug(
        "Loaded overrides: %s script(s) enabled, %s disabled, %s name override(s)",
        len(script_overrides.enabled),
        len(script_overrides.disabled),
        sum(len(v) for v in name_overrides.by_kind.values()),
    )
    _LOGGER.debug(
        "Per-script name overrides: %s",
        {s: sorted(f) for s, f in name_overrides.per_script.items()},
    )
    settings = data.get("settings") or {}
    max_tokens: Optional[int] = None
    prompts: Dict[str, Optional[str]] = {"system_prompt": None, "user_prompt": None}
    if isinstance(settings, dict):
        configured_max_tokens = settings.get("max_tokens")
        if (
            isinstance(configured_max_tokens, int)
            and not isinstance(configured_max_tokens, bool)
            and configured_max_tokens > 0
        ):
            max_tokens = configured_max_tokens
        elif configured_max_tokens is not None:
            _LOGGER.warning(
                "Ignoring invalid max_tokens setting in %s: %r",
                path,
                configured_max_tokens,
            )

        # Only the shape is checked here; what makes a prompt usable is the
        # recognizer's business, and the app falls back to the default when it
        # rejects one.
        for setting_key in prompts:
            configured_prompt = settings.get(setting_key)
            if isinstance(configured_prompt, str) and configured_prompt.strip():
                prompts[setting_key] = configured_prompt
            elif configured_prompt is not None:
                _LOGGER.warning(
                    "Ignoring invalid %s setting in %s: %r",
                    setting_key,
                    path,
                    configured_prompt,
                )

    return Overrides(
        scripts=script_overrides,
        names=name_overrides,
        max_tokens=max_tokens,
        system_prompt=prompts["system_prompt"],
        user_prompt=prompts["user_prompt"],
    )


def prune_all(
    all_overrides: Overrides,
    all_tools: List["Tool"],
    info: "HomeAssistantInfo",
) -> bool:
    """Drop overrides for anything that no longer exists, and log what went.

    Returns True when something was dropped, so the caller can rewrite the file.
    Only call this with a Home Assistant snapshot you know to be complete.
    """
    changed = False

    dropped_scripts = all_overrides.scripts.prune({tool.name for tool in all_tools})
    if dropped_scripts:
        _LOGGER.info(
            "Forgetting override(s) for deleted script(s): %s", sorted(dropped_scripts)
        )
        changed = True

    dropped_names = all_overrides.names.prune(
        {
            ENTITY: set(info.entities),
            AREA: set(info.areas),
            FLOOR: set(info.floors),
        },
        known_scripts={tool.name for tool in all_tools},
    )
    for kind, gone in sorted(dropped_names.items()):
        _LOGGER.info(
            "Forgetting name override(s) for deleted %s: %s", kind, sorted(gone)
        )
        changed = True

    return changed


def save(path: Path, all_overrides: Overrides) -> None:
    """Write the overrides file, replacing it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as overrides_file:
        yaml.safe_dump(
            all_overrides.as_dict(),
            overrides_file,
            default_flow_style=False,
            sort_keys=True,
        )

    tmp_path.replace(path)
    _LOGGER.debug("Saved overrides: %s", path)


def _name_set(value: object) -> Set[str]:
    """A set of script names from whatever the YAML held."""
    if not isinstance(value, list):
        return set()

    return {str(name).strip() for name in value if str(name).strip()}
