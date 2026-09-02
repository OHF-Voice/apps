import asyncio
import copy
import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from flask import Flask, Response, jsonify, render_template, request, url_for
from flask.typing import ResponseReturnValue
from werkzeug.middleware.proxy_fix import ProxyFix

import benchmark
import overrides
from benchmark import Fixture
from const import AppState
from gemma4_recognizer import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    PROMPT_PLACEHOLDERS,
    normalize_prompt,
    validate_prompts,
)
from hass_api import HomeAssistantInfo, SatelliteInfo, Tool
from overrides import AREA, ENTITY, FLOOR, NAME_KINDS, NameOverrides, ScriptOverrides
from tool_mapping import map_tool_call, required_fields

if TYPE_CHECKING:
    # WSGI types live in typeshed only, so they cannot be imported at runtime.
    from _typeshed.wsgi import StartResponse, WSGIApplication, WSGIEnvironment

_LOGGER = logging.getLogger(__name__)

MAX_PASSES = 10
MIN_MAX_TOKENS = 16
MAX_MAX_TOKENS = 512
PENDING_RUN_TTL = 300.0
MAX_PENDING_RUNS = 100
SCRIPT_CALL_TIMEOUT = 30.0

# Names shown per enum field before collapsing into "+N more".
ENUM_SAMPLE = 5

# Seconds to wait for Home Assistant when re-reading it.
HASS_TIMEOUT = 60.0


def _call_script(state: AppState, script_id: str, variables: Dict[str, Any]) -> None:
    """Call a script from the web thread on the app's event loop."""
    if state.loop is None:
        raise RuntimeError("Home Assistant event loop is unavailable")

    service_call = state.hass.call_service(
        "script",
        "turn_on",
        service_data={"variables": variables},
        target={"entity_id": script_id},
    )
    future = asyncio.run_coroutine_threadsafe(service_call, state.loop)
    future.result(timeout=SCRIPT_CALL_TIMEOUT)


def _apply_candidate(
    state: AppState,
    hass_info: HomeAssistantInfo,
    all_tools: List[Tool],
    candidate_overrides: overrides.Overrides,
) -> List[Tool]:
    """Atomically swap the model and app state to a prepared snapshot."""
    overrides.prune_all(candidate_overrides, all_tools, hass_info)
    targeted = candidate_overrides.scripts.select(all_tools)

    def reload_and_commit() -> None:
        # This runs on the model worker. Committing here closes the small window
        # where a queued recognition could use the new prefix but still map its
        # result against the old Tool objects.
        state.recognizer.reload([tool.tool for tool in targeted])
        state.overrides = candidate_overrides
        state.set_home_info(hass_info)
        state.all_tools = all_tools
        state.set_targeted(targeted)

    state.model_rebuilding.set()
    try:
        future = state.llama_executor.submit(reload_and_commit)
        future.result()
    finally:
        state.model_rebuilding.clear()

    if state.overrides_path is not None:
        overrides.save(state.overrides_path, state.overrides)

    return targeted


def _replacement_global_names(
    current: NameOverrides,
    changes: Dict[str, Any],
    known: Dict[str, Set[str]],
) -> NameOverrides:
    """Build global name replacements without losing per-field exceptions."""
    new_names = NameOverrides(per_script=copy.deepcopy(current.per_script))
    for kind in NAME_KINDS:
        kind_changes = changes.get(kind) or {}
        if not isinstance(kind_changes, dict):
            raise ValueError(f"Names for {kind} must be a mapping")

        for target_id, target_names in kind_changes.items():
            if target_id not in known[kind]:
                # Ignore anything that is not a real target: a name has to
                # resolve to a Home Assistant id to be usable.
                _LOGGER.debug("Ignoring unknown %s: %s", kind, target_id)
                continue

            if not isinstance(target_names, list):
                raise ValueError(f"Names for {target_id} must be a list")

            new_names.set_names(
                kind,
                str(target_id),
                [str(name).strip() for name in target_names if str(name).strip()],
            )

    return new_names


def _describe_fields(tool: Tool, names: NameOverrides) -> List[Dict[str, Any]]:
    """Summarize a tool's fields for the web UI."""
    function = tool.tool.get("function") or {}
    params = function.get("parameters") or {}
    props = params.get("properties") or {}
    required = required_fields(tool)

    fields: List[Dict[str, Any]] = []
    for field_name, schema in sorted(props.items()):
        schema = schema or {}
        value_schema = (
            schema.get("items") or {} if schema.get("type") == "array" else schema
        )
        enum = value_schema.get("enum") or []
        name_map = tool.name_map.get(field_name) or {}
        # Names that refer to more than one thing. The satellite's area can often
        # narrow one down at recognition time, but not always, so they are worth
        # showing: giving them distinct names on the Names page always works.
        shared = sorted(name for name, ids in name_map.items() if len(ids) > 1)
        fields.append(
            {
                "name": field_name,
                "type": schema.get("type", "string"),
                "format": value_schema.get("format"),
                "required": field_name in required,
                "enum_count": len(enum),
                "enum_sample": ", ".join(str(value) for value in enum[:ENUM_SAMPLE]),
                "enum_more": max(0, len(enum) - ENUM_SAMPLE),
                # True when the field's names resolve to Home Assistant ids.
                "mapped": bool(name_map),
                "shared_names": shared,
                "shared_sample": ", ".join(shared[:ENUM_SAMPLE]),
                # Set when this field takes names, so the UI can offer the
                # per-field editor and show how many are overridden there.
                "kind": tool.field_kinds.get(field_name),
                "scoped_count": len(names.field_names(tool.name, field_name)),
            }
        )

    return fields


def _describe_script(
    tool: Tool, targeted: bool, names: NameOverrides
) -> Dict[str, Any]:
    function = tool.tool.get("function") or {}
    return {
        "name": tool.name,
        "friendly_name": tool.friendly_name,
        "description": tool.description,
        # What the model is actually told about this script. Falls back to the
        # script's name when it has no description (and no blueprint to borrow
        # one from), which gives the model very little to match on.
        "model_description": function.get("description") or "",
        "has_description": bool(tool.description),
        "exposed": tool.exposed,
        "targeted": targeted,
        # Targeted despite not being exposed to voice in Home Assistant: worth
        # calling out, since this app runs scripts.
        "force_enabled": targeted and not tool.exposed,
        "fields": _describe_fields(tool, names),
    }


def make_web_server(state: AppState, benchmark_fixture_path: Union[str, Path]) -> Flask:
    flask_app = Flask(__name__)
    flask_app.secret_key = "e603c0d4-18d8-4018-ad45-4c89a0aa9941"

    flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app, x_proto=1, x_host=1)  # type: ignore[method-assign]
    flask_app.wsgi_app = IngressPrefixMiddleware(flask_app.wsgi_app)  # type: ignore[method-assign]

    fixture: Optional[Fixture] = None
    fixture_error: Optional[str] = None
    try:
        fixture = benchmark.load_fixture(benchmark_fixture_path)
    except (OSError, ValueError) as err:
        fixture_error = f"Could not load benchmark fixture: {err}"
        _LOGGER.warning(fixture_error)

    pending_runs: Dict[str, tuple[float, str, Dict[str, Any]]] = {}
    pending_runs_lock = threading.Lock()

    def issue_run_token(script_id: str, variables: Dict[str, Any]) -> str:
        """Keep a resolved call server-side and return its opaque handle."""
        now = time.monotonic()
        token = secrets.token_urlsafe(24)
        with pending_runs_lock:
            expired = [
                run_id
                for run_id, (created, _, _) in pending_runs.items()
                if (now - created) > PENDING_RUN_TTL
            ]
            for run_id in expired:
                del pending_runs[run_id]
            if len(pending_runs) >= MAX_PENDING_RUNS:
                oldest = min(pending_runs, key=lambda run_id: pending_runs[run_id][0])
                del pending_runs[oldest]
            pending_runs[token] = (now, script_id, copy.deepcopy(variables))
        return token

    @flask_app.context_processor
    def inject_url_for() -> Dict[str, Any]:
        return dict(url_for=url_for)  # pylint: disable=use-dict-literal

    @flask_app.route("/", methods=["GET"])
    def index() -> ResponseReturnValue:
        """Show every script, split by whether the model can call it."""
        scripts = [
            _describe_script(
                tool, state.overrides.scripts.targets(tool), state.overrides.names
            )
            for tool in state.all_tools
        ]
        areas = sorted(
            ({"id": a.area_id, "name": a.name} for a in state.hass_info.areas.values()),
            key=lambda a: a["name"],
        )
        return render_template(
            "home.html",
            targeted=[s for s in scripts if s["targeted"]],
            untargeted=[s for s in scripts if not s["targeted"]],
            can_save=state.overrides_path is not None,
            areas=areas,
        )

    @flask_app.route("/test", methods=["GET"])
    def test_page() -> ResponseReturnValue:
        """Show the interactive sentence tester."""
        satellites = sorted(
            (
                {"id": entity_id, "name": name}
                for entity_id, name in state.hass_info.satellites.items()
            ),
            key=lambda satellite: (
                str(satellite["name"]).casefold(),
                str(satellite["id"]),
            ),
        )
        return render_template("test.html", satellites=satellites)

    @flask_app.route("/settings", methods=["GET"])
    def settings_page() -> ResponseReturnValue:
        """Show runtime model information and editable recognition settings."""
        model = state.recognizer.describe()
        return render_template(
            "settings.html",
            model=model,
            context_mode="fixed" if state.recognizer.n_ctx is not None else "automatic",
            max_tokens_source=(
                "web setting"
                if state.overrides.max_tokens is not None
                else "app startup option"
            ),
            can_save=state.overrides_path is not None,
            min_max_tokens=MIN_MAX_TOKENS,
            max_max_tokens=MAX_MAX_TOKENS,
            system_prompt=normalize_prompt(state.recognizer.system_prompt),
            user_prompt=normalize_prompt(state.recognizer.user_prompt),
            default_system_prompt=normalize_prompt(DEFAULT_SYSTEM_PROMPT),
            default_user_prompt=normalize_prompt(DEFAULT_USER_PROMPT),
            prompts_customized=(state.overrides.system_prompt is not None)
            or (state.overrides.user_prompt is not None),
            placeholders=sorted(PROMPT_PLACEHOLDERS.items()),
        )

    @flask_app.route("/settings/prompts", methods=["POST"])
    def settings_prompts_apply() -> ResponseReturnValue:
        """Apply and persist the system and user prompts."""
        if state.overrides_path is None:
            return jsonify({"error": "No settings file is configured"}), 503
        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        body = request.get_json(silent=True) or {}
        system_prompt = body.get("system_prompt")
        user_prompt = body.get("user_prompt")
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            return jsonify({"error": "Both prompts are required"}), 400

        try:
            system_prompt, user_prompt = validate_prompts(system_prompt, user_prompt)
        except ValueError as err:
            return jsonify({"error": str(err)}), 400

        if not state.reload_lock.acquire(blocking=False):
            return jsonify({"error": "Already applying a change"}), 409

        old_system_prompt = state.recognizer.system_prompt
        old_user_prompt = state.recognizer.user_prompt
        candidate_overrides = copy.deepcopy(state.overrides)
        # A prompt that matches the default is not recorded, so the app keeps
        # following the default if a later version improves it.
        candidate_overrides.system_prompt = (
            None
            if system_prompt == normalize_prompt(DEFAULT_SYSTEM_PROMPT)
            else system_prompt
        )
        candidate_overrides.user_prompt = (
            None
            if user_prompt == normalize_prompt(DEFAULT_USER_PROMPT)
            else user_prompt
        )

        state.model_rebuilding.set()
        try:
            future = state.llama_executor.submit(
                state.recognizer.set_prompts, system_prompt, user_prompt
            )
            future.result()
            try:
                overrides.save(state.overrides_path, candidate_overrides)
            except Exception:
                # Keep the persisted and effective prompts aligned if writing fails.
                rollback = state.llama_executor.submit(
                    state.recognizer.set_prompts, old_system_prompt, old_user_prompt
                )
                rollback.result()
                raise
            state.overrides = candidate_overrides
        except ValueError as err:
            return jsonify({"error": str(err)}), 400
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Failed to apply prompts")
            return jsonify({"error": f"Could not apply prompts: {err}"}), 500
        finally:
            state.model_rebuilding.clear()
            state.reload_lock.release()

        return jsonify(
            {
                "system_prompt": state.recognizer.system_prompt,
                "user_prompt": state.recognizer.user_prompt,
                "customized": (candidate_overrides.system_prompt is not None)
                or (candidate_overrides.user_prompt is not None),
                "n_ctx": (
                    state.recognizer.llm.n_ctx()
                    if state.recognizer.llm is not None
                    else None
                ),
            }
        )

    @flask_app.route("/settings", methods=["POST"])
    def settings_apply() -> ResponseReturnValue:
        """Apply and persist the maximum generation length."""
        if state.overrides_path is None:
            return jsonify({"error": "No settings file is configured"}), 503
        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        body = request.get_json(silent=True) or {}
        max_tokens = body.get("max_tokens")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not (MIN_MAX_TOKENS <= max_tokens <= MAX_MAX_TOKENS)
        ):
            return (
                jsonify(
                    {
                        "error": (
                            f"Maximum tokens must be an integer from "
                            f"{MIN_MAX_TOKENS} to {MAX_MAX_TOKENS}"
                        )
                    }
                ),
                400,
            )

        if not state.reload_lock.acquire(blocking=False):
            return jsonify({"error": "Already applying a change"}), 409

        old_max_tokens = state.recognizer.max_tokens
        candidate_overrides = copy.deepcopy(state.overrides)
        candidate_overrides.max_tokens = max_tokens
        state.model_rebuilding.set()
        try:
            future = state.llama_executor.submit(
                state.recognizer.set_max_tokens, max_tokens
            )
            future.result()
            try:
                overrides.save(state.overrides_path, candidate_overrides)
            except Exception:
                # Keep the persisted and effective values aligned if writing fails.
                rollback = state.llama_executor.submit(
                    state.recognizer.set_max_tokens, old_max_tokens
                )
                rollback.result()
                raise
            state.overrides = candidate_overrides
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Failed to apply maximum tokens")
            return jsonify({"error": f"Could not apply setting: {err}"}), 500
        finally:
            state.model_rebuilding.clear()
            state.reload_lock.release()

        return jsonify(state.recognizer.describe())

    def _fetch_from_hass(
        candidate_overrides: overrides.Overrides,
    ) -> tuple[HomeAssistantInfo, List[Tool]]:
        """Re-read Home Assistant and build tools without changing live state.

        Runs the fetch on the app's event loop, since the web server is a
        separate thread. Candidate name overrides are applied while the tools
        are built, but nothing is committed until the model accepts the prefix.
        """
        assert state.loop is not None, "No event loop"

        async def fetch() -> Tuple[HomeAssistantInfo, List[Tool]]:
            hass_info = await state.hass.get_home_info()
            all_tools = await state.hass.get_script_tools(
                hass_info, candidate_overrides.names
            )
            return hass_info, all_tools

        hass_info, all_tools = asyncio.run_coroutine_threadsafe(
            fetch(), state.loop
        ).result(timeout=HASS_TIMEOUT)

        if not all_tools:
            # Nothing came back, so this is not a trustworthy snapshot: keep
            # what we have rather than pruning against it.
            raise RuntimeError("Home Assistant returned no scripts")

        return hass_info, all_tools

    @flask_app.route("/reload", methods=["POST"])
    def reload_from_hass() -> ResponseReturnValue:
        """Re-read Home Assistant, then rebuild the tools and model prefix.

        The app otherwise reads Home Assistant once, at start, so this is how a
        newly exposed script or a renamed entity is picked up.
        """
        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        if not state.reload_lock.acquire(blocking=False):
            return jsonify({"error": "Already applying a change"}), 409

        try:
            candidate_overrides = copy.deepcopy(state.overrides)
            hass_info, all_tools = _fetch_from_hass(candidate_overrides)
            targeted = _apply_candidate(
                state, hass_info, all_tools, candidate_overrides
            )
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Failed to reload from Home Assistant")
            return jsonify({"error": f"Could not reload: {err}"}), 500
        finally:
            state.reload_lock.release()

        return jsonify(
            {
                "num_scripts": len(state.all_tools),
                "num_targeted": len(targeted),
            }
        )

    def _requested_names(body: Dict[str, Any]) -> Optional[Set[str]]:
        """The script names the page wants targeted, or None if malformed."""
        names = body.get("targeted")
        if not isinstance(names, list):
            return None

        known = {tool.name for tool in state.all_tools}
        return {str(name) for name in names} & known

    @flask_app.route("/overrides/estimate", methods=["POST"])
    def overrides_estimate() -> ResponseReturnValue:
        """Report what a prospective tool set would cost, before applying it."""
        body = request.get_json(silent=True) or {}
        names = _requested_names(body)
        if names is None:
            return jsonify({"error": "No script list given"}), 400

        tools = [tool.tool for tool in state.all_tools if tool.name in names]
        return jsonify(
            {
                "num_tools": len(tools),
                "n_ctx": state.recognizer.required_n_ctx(tools),
                "current_n_ctx": (
                    state.recognizer.llm.n_ctx()
                    if state.recognizer.llm is not None
                    else None
                ),
                "fixed_n_ctx": state.recognizer.n_ctx is not None,
            }
        )

    @flask_app.route("/overrides", methods=["POST"])
    def overrides_apply() -> ResponseReturnValue:
        """Change which scripts are targeted, then rebuild the model prefix.

        Recognition is unavailable while the prefix is rebuilt, which can take
        minutes on slow hardware, so this is deliberately an explicit action
        rather than something that happens per checkbox.
        """
        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        body = request.get_json(silent=True) or {}
        names = _requested_names(body)
        if names is None:
            return jsonify({"error": "No script list given"}), 400

        # A second apply while one is in flight would race on the model.
        if not state.reload_lock.acquire(blocking=False):
            return jsonify({"error": "Already applying a change"}), 409

        try:
            new_script_overrides = ScriptOverrides(
                enabled={
                    tool.name
                    for tool in state.all_tools
                    if (tool.name in names) and not tool.exposed
                },
                disabled={
                    tool.name
                    for tool in state.all_tools
                    if (tool.name not in names) and tool.exposed
                },
            )
            try:
                candidate_overrides = copy.deepcopy(state.overrides)
                candidate_overrides.scripts = new_script_overrides
                targeted = _apply_candidate(
                    state,
                    state.hass_info,
                    state.all_tools,
                    candidate_overrides,
                )
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.exception("Failed to apply script overrides")
                return jsonify({"error": f"Could not apply: {err}"}), 500

            return jsonify(
                {
                    "num_targeted": len(targeted),
                    "enabled": sorted(new_script_overrides.enabled),
                    "disabled": sorted(new_script_overrides.disabled),
                    "n_ctx": (
                        state.recognizer.llm.n_ctx()
                        if state.recognizer.llm is not None
                        else None
                    ),
                }
            )
        finally:
            state.reload_lock.release()

    @flask_app.route("/tools.json", methods=["GET"])
    def tools_json() -> ResponseReturnValue:
        """The tools as given to the model (OpenAI function spec), for debugging."""
        tools = [tool.tool for tool in state.tools.values()]
        return Response(json.dumps(tools, indent=2), mimetype="application/json")

    @flask_app.route("/test", methods=["POST"])
    def test() -> ResponseReturnValue:
        """Recognize one sentence and report what it would run, without running it."""
        body = request.get_json(silent=True) or {}
        text = str(body.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text given"}), 400
        language = str(body.get("language") or "").strip() or "en"

        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        satellite_id = str(body.get("satellite_id") or "").strip() or None
        satellite: Optional[SatelliteInfo] = None
        if satellite_id:
            if satellite_id not in state.hass_info.satellites:
                return jsonify({"error": "Unknown satellite"}), 400
            if state.loop is None:
                return (
                    jsonify({"error": "Home Assistant event loop is unavailable"}),
                    503,
                )

            try:
                satellite_future = asyncio.run_coroutine_threadsafe(
                    state.hass.get_satellite_info(satellite_id=satellite_id),
                    state.loop,
                )
                satellite = satellite_future.result(timeout=HASS_TIMEOUT)
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.exception("Could not resolve test satellite %s", satellite_id)
                return jsonify({"error": f"Could not resolve satellite: {err}"}), 502

        # Serialize on the model worker, like live recognition and the benchmark.
        start_time = time.monotonic()
        future = state.llama_executor.submit(
            state.recognizer.get_tool_calls, text, language
        )
        tool_calls, response_text = future.result()
        latency_ms = (time.monotonic() - start_time) * 1000.0

        calls = []
        for tool_id, tool_args in tool_calls:
            tool = state.tools.get(tool_id)
            if tool is None:
                # The model can emit a name that is not a tool at all.
                calls.append(
                    {
                        "script_id": tool_id,
                        "unknown_tool": True,
                        "can_run": False,
                        "variables": {},
                        "unresolved": {},
                        "ambiguous": {},
                        "missing_required": [],
                    }
                )
                continue

            call = map_tool_call(tool, tool_args, satellite, state.geometry)
            variables = dict(call.variables)
            if call.can_run and satellite is not None:
                variables["satellite"] = satellite.as_script_variable(language)

            call_info = {
                "script_id": call.script_id,
                "unknown_tool": False,
                "can_run": call.can_run,
                "variables": variables,
                "unresolved": call.unresolved,
                "ambiguous": call.ambiguous,
                "missing_required": call.missing_required,
            }
            if call.can_run:
                call_info["run_id"] = issue_run_token(call.script_id, variables)
            calls.append(call_info)

        return jsonify(
            {
                "text": response_text,
                "latency_ms": latency_ms,
                "calls": calls,
            }
        )

    @flask_app.route("/test/run", methods=["POST"])
    def test_run() -> ResponseReturnValue:
        """Run one resolved test result exactly once."""
        body = request.get_json(silent=True) or {}
        run_id = body.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return jsonify({"error": "No test run given"}), 400
        if state.model_rebuilding.is_set():
            return jsonify({"error": "Model configuration is changing"}), 409

        now = time.monotonic()
        with pending_runs_lock:
            pending = pending_runs.pop(run_id, None)
        if pending is None or (now - pending[0]) > PENDING_RUN_TTL:
            return jsonify({"error": "This test result expired; test it again"}), 404

        _, script_id, variables = pending
        tool_id = script_id.removeprefix("script.")
        if tool_id not in state.tools:
            return jsonify({"error": "This script is no longer targeted"}), 409

        try:
            _call_script(state, script_id, variables)
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Failed to run tested script %s", script_id)
            return jsonify({"error": f"Could not run {script_id}: {err}"}), 502

        return jsonify({"script_id": script_id})

    def _name_rows() -> Dict[str, List[Dict[str, Any]]]:
        """Every nameable target, with the names the model is currently offered.

        Only entities exposed to Assist reach the enums, so those are the only
        ones worth listing; all areas and floors are always available.
        """
        info = state.hass_info
        names = state.overrides.names
        rows: Dict[str, List[Dict[str, Any]]] = {}

        def row(
            kind: str, target_id: str, default_names: List[str], extra: str = ""
        ) -> Dict[str, Any]:
            current = names.names_for(kind, target_id, default_names)
            return {
                "id": target_id,
                "default_names": default_names,
                "names": current,
                "overridden": names.is_overridden(kind, target_id),
                "excluded": not current,
                "extra": extra,
            }

        rows[ENTITY] = [
            row(ENTITY, entity.entity_id, [n for n in entity.names if n], entity.domain)
            for entity in sorted(info.entities.values(), key=lambda e: e.entity_id)
        ]
        rows[AREA] = [
            row(AREA, area.area_id, [n for n in area.names if n])
            for area in sorted(info.areas.values(), key=lambda a: a.area_id)
        ]
        rows[FLOOR] = [
            row(FLOOR, floor.floor_id, [n for n in floor.names if n])
            for floor in sorted(info.floors.values(), key=lambda f: f.floor_id)
        ]
        return rows

    @flask_app.route("/names", methods=["GET"])
    def names_page() -> ResponseReturnValue:
        """Edit what the model may call each entity, area, and floor."""
        return render_template(
            "names.html",
            rows=_name_rows(),
            can_save=state.overrides_path is not None,
        )

    def _default_names(kind: str, target_id: str) -> List[str]:
        """Home Assistant's own names for one target."""
        info = state.hass_info
        if kind == ENTITY:
            target: Any = info.entities.get(target_id)
        elif kind == AREA:
            target = info.areas.get(target_id)
        else:
            target = info.floors.get(target_id)

        return [name for name in target.names if name] if target else []

    @flask_app.route("/names/field", methods=["GET"])
    def names_field_page() -> ResponseReturnValue:
        """Edit names for one script's field only.

        Reached from that field on the Scripts page, so the list is already the
        one that misbehaved: the field's own candidates, after its domain filter.
        """
        script_name = str(request.args.get("script") or "")
        field_key = str(request.args.get("field") or "")
        tool = next((t for t in state.all_tools if t.name == script_name), None)
        if (tool is None) or (field_key not in tool.field_kinds):
            if request.args.get("format") == "json":
                return jsonify({"error": "Unknown script field"}), 404
            return (
                render_template(
                    "names_field.html", tool=None, script=script_name, field=field_key
                ),
                404,
            )

        kind = tool.field_kinds[field_key]
        scoped = state.overrides.names.field_names(script_name, field_key)
        rows: List[Dict[str, Any]] = []
        for target_id in tool.field_targets.get(field_key, []):
            default_names = _default_names(kind, target_id)
            # What this target is called everywhere else, which is what an empty
            # per-field override falls back to.
            global_names = state.overrides.names.names_for(
                kind, target_id, default_names
            )
            current = scoped.get(target_id, global_names)
            rows.append(
                {
                    "id": target_id,
                    "global_names": global_names,
                    "names": current,
                    "overridden": target_id in scoped,
                    "excluded": not current,
                }
            )

        rows = sorted(rows, key=lambda r: str(r["id"]))
        can_save = state.overrides_path is not None
        if request.args.get("format") == "json":
            return jsonify(
                {
                    "script": script_name,
                    "friendly_name": tool.friendly_name,
                    "field": field_key,
                    "kind": kind,
                    "rows": rows,
                    "can_save": can_save,
                }
            )

        return render_template(
            "names_field.html",
            tool=tool,
            script=script_name,
            field=field_key,
            kind=kind,
            rows=rows,
            can_save=can_save,
        )

    @flask_app.route("/names/field", methods=["POST"])
    def names_field_apply() -> ResponseReturnValue:
        """Replace one script field's name overrides, then rebuild."""
        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        body = request.get_json(silent=True) or {}
        script_name = str(body.get("script") or "")
        field_key = str(body.get("field") or "")
        changes = body.get("names")
        if not isinstance(changes, dict):
            return jsonify({"error": "No names given"}), 400

        tool = next((t for t in state.all_tools if t.name == script_name), None)
        if (tool is None) or (field_key not in tool.field_kinds):
            return jsonify({"error": "Unknown script field"}), 404

        # Only ids this field could actually refer to; anything else could never
        # be resolved and would just rot in the file.
        candidates = set(tool.field_targets.get(field_key, []))
        scoped = {
            str(target_id): [
                str(name).strip() for name in target_names if str(name).strip()
            ]
            for target_id, target_names in changes.items()
            if (target_id in candidates) and isinstance(target_names, list)
        }

        if not state.reload_lock.acquire(blocking=False):
            return jsonify({"error": "Already applying a change"}), 409

        try:
            candidate_overrides = copy.deepcopy(state.overrides)
            candidate_overrides.names.set_field_names(script_name, field_key, scoped)
            hass_info, all_tools = _fetch_from_hass(candidate_overrides)
            _apply_candidate(state, hass_info, all_tools, candidate_overrides)
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Failed to apply names for %s.%s", script_name, field_key)
            return jsonify({"error": f"Could not apply: {err}"}), 500
        finally:
            state.reload_lock.release()

        return jsonify({"num_overridden": len(scoped)})

    @flask_app.route("/names", methods=["POST"])
    def names_apply() -> ResponseReturnValue:
        """Replace name overrides, then rebuild the tools and model prefix.

        Names feed the enums, which are built while reading Home Assistant, so
        this re-reads it rather than trying to patch the tools in place.
        """
        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        body = request.get_json(silent=True) or {}
        changes = body.get("names")
        if not isinstance(changes, dict):
            return jsonify({"error": "No names given"}), 400

        info = state.hass_info
        known: Dict[str, Set[str]] = {
            ENTITY: set(info.entities),
            AREA: set(info.areas),
            FLOOR: set(info.floors),
        }

        # Global edits must not erase the field-specific exceptions introduced
        # in 1.6. They are independent layers of the naming configuration.
        try:
            new_names = _replacement_global_names(state.overrides.names, changes, known)
        except ValueError as err:
            return jsonify({"error": str(err)}), 400

        if not state.reload_lock.acquire(blocking=False):
            return jsonify({"error": "Already applying a change"}), 409

        try:
            candidate_overrides = copy.deepcopy(state.overrides)
            candidate_overrides.names = new_names
            hass_info, all_tools = _fetch_from_hass(candidate_overrides)
            targeted = _apply_candidate(
                state, hass_info, all_tools, candidate_overrides
            )
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Failed to apply name overrides")
            return jsonify({"error": f"Could not apply: {err}"}), 500
        finally:
            state.reload_lock.release()

        return jsonify(
            {
                "num_overridden": sum(len(v) for v in new_names.by_kind.values()),
                "num_targeted": len(targeted),
            }
        )

    @flask_app.route("/health")
    def health() -> ResponseReturnValue:
        # Deliberately "ok" while the model is still loading: the first boot can
        # spend many minutes downloading it, and the container should not be
        # restarted for being slow. Use /status to see readiness.
        return {"status": "ok"}, 200

    @flask_app.route("/status")
    def status() -> ResponseReturnValue:
        return {"model_loaded": state.recognizer.ready}, 200

    @flask_app.route("/benchmark", methods=["GET"])
    def benchmark_page() -> ResponseReturnValue:
        return render_template("benchmark.html", max_passes=MAX_PASSES)

    @flask_app.route("/benchmark/run", methods=["POST"])
    def benchmark_run() -> ResponseReturnValue:
        if fixture is None:
            return {"error": fixture_error or "No benchmark fixture"}, 500
        if not state.recognizer.ready:
            return {"error": "Model is still loading"}, 503

        body = request.get_json(silent=True) or {}
        try:
            passes = int(body.get("passes", 3))
        except (TypeError, ValueError):
            passes = 3
        passes = max(1, min(MAX_PASSES, passes))

        _LOGGER.info("Running benchmark (passes=%s)", passes)
        # Serialize on the single model worker so a benchmark can't race a live
        # recognition request (the model is not thread-safe). Blocks live voice
        # handling for the benchmark's duration.
        future = state.llama_executor.submit(
            benchmark.run, state.recognizer, fixture, passes
        )
        result = future.result()
        return Response(json.dumps(result), mimetype="application/json")

    return flask_app


def run_web_server(flask_app: Flask, host: str, port: int) -> threading.Thread:
    def run_flask() -> None:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        flask_app.run(host=host, port=port, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)

    return flask_thread


class IngressPrefixMiddleware:
    """Ingress fix for Home Assistant app web UI."""

    def __init__(self, app: "WSGIApplication") -> None:
        self.app = app

    def __call__(
        self, environ: "WSGIEnvironment", start_response: "StartResponse"
    ) -> Iterable[bytes]:
        ingress_path = environ.get("HTTP_X_INGRESS_PATH", "")
        if ingress_path:
            environ["SCRIPT_NAME"] = ingress_path
            path_info = environ.get("PATH_INFO", "")
            if path_info.startswith(ingress_path):
                environ["PATH_INFO"] = path_info[len(ingress_path) :] or "/"
        return self.app(environ, start_response)
