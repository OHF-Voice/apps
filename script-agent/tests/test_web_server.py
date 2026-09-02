import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from const import AppState
from gemma4_recognizer import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    validate_prompts,
)
from hass_api import HomeAssistantInfo, SatelliteInfo, Tool
from models import Entity
from overrides import AREA, ENTITY, FLOOR, NameOverrides, Overrides, load
from web_server import (
    _apply_candidate,
    _call_script,
    _replacement_global_names,
    make_web_server,
)


def _info():
    return HomeAssistantInfo(states={}, entities={}, areas={}, floors={})


def _tool(name):
    return Tool(
        name=name,
        exposed=True,
        tool={"type": "function", "function": {"name": name}},
    )


class _Recognizer:
    def __init__(self, error=None):
        self.error = error
        self.reload_calls = []
        self.max_token_calls = []
        self.prompt_calls = []
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.user_prompt = DEFAULT_USER_PROMPT
        self.max_tokens = 128
        self.n_ctx = None
        self.ready = True
        self.llm = _Llama()
        self.repo_id = "bartowski/google_gemma-4-E2B-it-GGUF"
        self.filename = "google_gemma-4-E2B-it-Q5_K_M.gguf"
        self.flash_attn = True
        self.tool_calls = []

    def reload(self, tools):
        self.reload_calls.append(tools)
        if self.error:
            raise self.error

    def set_max_tokens(self, max_tokens):
        self.max_token_calls.append(max_tokens)
        if self.error:
            raise self.error
        self.max_tokens = max_tokens

    def set_prompts(self, system_prompt, user_prompt):
        self.prompt_calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        # The real recognizer validates and normalizes before it commits.
        self.system_prompt, self.user_prompt = validate_prompts(
            system_prompt, user_prompt
        )

    def describe(self):
        return {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "n_ctx": self.llm.n_ctx(),
            "n_threads": 4,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "flash_attn": self.flash_attn,
            "draft_model": None,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }

    def get_tool_calls(self, _text, _language):
        return self.tool_calls, "model response"


class _Llama:
    def n_ctx(self):
        return 768


class NameReplacementTests(unittest.TestCase):
    def test_global_names_preserve_per_script_names(self):
        current = NameOverrides(
            per_script={"demo": {"target": {"light.desk": ["Desk only"]}}}
        )
        known = {
            ENTITY: {"light.desk"},
            AREA: set(),
            FLOOR: set(),
        }

        replacement = _replacement_global_names(
            current,
            {ENTITY: {"light.desk": ["Desk"]}, AREA: {}, FLOOR: {}},
            known,
        )

        self.assertEqual(
            {"demo": {"target": {"light.desk": ["Desk only"]}}},
            replacement.per_script,
        )
        self.assertEqual(["Desk"], replacement.by_kind[ENTITY]["light.desk"])


class ScriptCallTests(unittest.TestCase):
    def test_calls_script_with_resolved_variables(self):
        hass = Mock()
        service_call = object()
        hass.call_service.return_value = service_call
        state = Mock(hass=hass, loop=object())
        future = Mock()

        with patch(
            "web_server.asyncio.run_coroutine_threadsafe", return_value=future
        ) as submit:
            _call_script(
                state,
                "script.notify_mike",
                {"target": ["light.office"]},
            )

        hass.call_service.assert_called_once_with(
            "script",
            "turn_on",
            service_data={"variables": {"target": ["light.office"]}},
            target={"entity_id": "script.notify_mike"},
        )
        submit.assert_called_once_with(service_call, state.loop)
        future.result.assert_called_once()


class TransactionTests(unittest.TestCase):
    def test_failed_model_reload_does_not_commit_candidate_state(self):
        old_info = _info()
        old_tool = _tool("old")
        old_overrides = Overrides()
        executor = ThreadPoolExecutor(max_workers=1)
        state = AppState(
            hass=object(),  # type: ignore[arg-type]
            hass_info=old_info,
            tools={old_tool.name: old_tool},
            recognizer=_Recognizer(RuntimeError("reload failed")),  # type: ignore[arg-type]
            llama_executor=executor,
            all_tools=[old_tool],
            overrides=old_overrides,
        )
        new_tool = _tool("new")
        candidate = Overrides()

        try:
            with self.assertRaisesRegex(RuntimeError, "reload failed"):
                _apply_candidate(state, _info(), [new_tool], candidate)

            self.assertIs(old_info, state.hass_info)
            self.assertIs(old_overrides, state.overrides)
            self.assertEqual([old_tool], state.all_tools)
            self.assertEqual({"old": old_tool}, state.tools)
        finally:
            executor.shutdown()

    def test_successful_model_reload_commits_one_snapshot(self):
        old_tool = _tool("old")
        recognizer = _Recognizer()
        executor = ThreadPoolExecutor(max_workers=1)
        state = AppState(
            hass=object(),  # type: ignore[arg-type]
            hass_info=_info(),
            tools={old_tool.name: old_tool},
            recognizer=recognizer,  # type: ignore[arg-type]
            llama_executor=executor,
            all_tools=[old_tool],
            overrides=Overrides(),
        )
        new_info = _info()
        new_tool = _tool("new")
        candidate = Overrides()

        try:
            targeted = _apply_candidate(state, new_info, [new_tool], candidate)

            self.assertEqual([new_tool], targeted)
            self.assertIs(new_info, state.hass_info)
            self.assertIs(candidate, state.overrides)
            self.assertEqual({"new": new_tool}, state.tools)
        finally:
            executor.shutdown()


class WebPageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.executor = ThreadPoolExecutor(max_workers=1)
        tool = _tool("notify_mike")
        tool.friendly_name = "Notify Mike"
        tool.description = "Notify Mike with a message"
        tool.tool["function"]["description"] = tool.description
        tool.tool["function"]["parameters"] = {
            "type": "object",
            "properties": {
                "target": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["Office Light", "Office"],
                    },
                    "uniqueItems": True,
                }
            },
            "required": ["target"],
        }
        tool.field_kinds["target"] = ENTITY
        tool.field_targets["target"] = ["light.office"]
        tool.name_map["target"] = {
            "Office Light": ["light.office"],
            "Office": ["light.office"],
        }
        info = _info()
        info.satellites["assist_satellite.kitchen"] = "Kitchen Voice"
        info.entities["light.office"] = Entity(
            entity_id="light.office",
            name="Office Light",
            aliases=["Office"],
        )
        self.state = AppState(
            hass=object(),  # type: ignore[arg-type]
            hass_info=info,
            tools={tool.name: tool},
            recognizer=_Recognizer(),  # type: ignore[arg-type]
            llama_executor=self.executor,
            all_tools=[tool],
            overrides=Overrides(),
            overrides_path=Path(self.temp_dir.name) / "overrides.yaml",
        )
        fixture_path = Path(__file__).parents[1] / "src" / "benchmark.yaml"
        self.client = make_web_server(self.state, fixture_path).test_client()

    def tearDown(self):
        self.executor.shutdown()
        self.temp_dir.cleanup()

    def test_scripts_page_explains_sync_does_not_train(self):
        response = self.client.get("/")

        self.assertEqual(200, response.status_code)
        self.assertIn(b"Sync &amp; rebuild", response.data)
        self.assertIn(b"does <b>not</b> train or fine-tune", response.data)
        self.assertIn(b'id="model-change-dialog"', response.data)
        self.assertIn(b"<progress", response.data)
        self.assertIn(b"function beginModelChange(message)", response.data)
        self.assertIn(b'aria-busy", "true"', response.data)
        self.assertNotIn(b'id="sentence"', response.data)
        self.assertIn(b"white-space: nowrap", response.data)
        self.assertIn(b'href="#main-content">Skip to main content', response.data)
        self.assertIn(b'aria-label="Target Notify Mike"', response.data)
        self.assertIn(b'aria-labelledby="dialog-title"', response.data)
        self.assertNotIn(b">Benchmark</a>", response.data)
        self.assertNotIn(b">Tools JSON</a>", response.data)
        self.assertLess(
            response.data.index(b">Names</a>"), response.data.index(b">Test</a>")
        )
        self.assertIn(b">Details</button>", response.data)
        self.assertNotIn(b"Office Light, Office", response.data)
        self.assertNotIn(b"Shared by", response.data)
        self.assertIn(b"array", response.data)
        self.assertIn(b"2 names", response.data)

    def test_sentence_tester_has_its_own_page(self):
        response = self.client.get("/test")

        self.assertEqual(200, response.status_code)
        self.assertIn(b'id="sentence"', response.data)
        self.assertIn(b'for="sentence">Command to test', response.data)
        self.assertIn(b'for="test-satellite">Satellite', response.data)
        self.assertIn(b'for="language">Response language', response.data)
        self.assertIn(
            b"Kitchen Voice (assist_satellite.kitchen)",
            response.data,
        )
        self.assertNotIn(b"Satellite area", response.data)
        self.assertIn(b"Testing alone never runs the script", response.data)
        self.assertIn(b"Run in Home Assistant", response.data)
        self.assertIn(b'aria-current="page">Test</a>', response.data)
        self.assertIn(
            b"#test-status { min-height: 1.4rem; margin-top: 0.75rem; }", response.data
        )
        self.assertIn(b"function renderValue(value)", response.data)
        self.assertIn(b"class='variable-object'", response.data)
        self.assertIn(
            b'"<tr><td>" + esc(k) + "</td><td>" + renderValue(v)', response.data
        )
        self.assertIn(b'key !== "satellite"', response.data)
        self.assertIn(b"class='satellite-variable'", response.data)
        self.assertIn(b"<code>satellite</code>", response.data)
        self.assertIn(b"Special variable", response.data)

    def test_names_page_has_search_and_collapsible_sections(self):
        response = self.client.get("/names")

        self.assertEqual(200, response.status_code)
        self.assertIn(b'id="names-search"', response.data)
        self.assertIn(
            b'placeholder="Search IDs, names, aliases, or entity types"', response.data
        )
        for kind in ("entity", "area", "floor"):
            self.assertIn(
                f'<details class="names-section" data-kind="{kind}" open>'.encode(),
                response.data,
            )
        self.assertIn(b'class="name-row"', response.data)
        self.assertIn(b'role="status" aria-live="polite"', response.data)
        self.assertIn(b'class="target-checkbox"', response.data)
        self.assertIn(b"Allow the model to target light.office", response.data)
        self.assertIn(
            b'beginModelChange("Applying names and rebuilding model context',
            response.data,
        )

    def test_excluded_name_uses_unchecked_targetable_control(self):
        self.state.overrides.names.by_kind[ENTITY]["light.office"] = []

        response = self.client.get("/names")

        self.assertEqual(200, response.status_code)
        self.assertIn(b'class="name-row excluded"', response.data)
        self.assertIn(b'value="Office Light, Office"', response.data)
        checkbox_start = response.data.index(b'class="target-checkbox"')
        checkbox_end = response.data.index(b">", checkbox_start)
        self.assertNotIn(b"checked", response.data[checkbox_start:checkbox_end])

    def test_unnamed_entity_uses_unchecked_targetable_control(self):
        self.state.hass_info.entities["sensor.unnamed"] = Entity(
            entity_id="sensor.unnamed",
            name="",
        )

        response = self.client.get("/names")

        self.assertEqual(200, response.status_code)
        row_start = response.data.index(b"<code>sensor.unnamed</code>")
        checkbox_start = response.data.index(b'class="target-checkbox"', row_start)
        checkbox_end = response.data.index(b">", checkbox_start)
        self.assertNotIn(b"checked", response.data[checkbox_start:checkbox_end])

    def test_successful_test_call_can_run_once(self):
        self.state.recognizer.tool_calls = [("notify_mike", {"target": ["Office"]})]
        test_response = self.client.post(
            "/test",
            json={"text": "notify Mike", "language": "en"},
        )

        self.assertEqual(200, test_response.status_code)
        test_call = test_response.get_json()["calls"][0]
        self.assertTrue(test_call["can_run"])
        self.assertIn("run_id", test_call)

        with patch("web_server._call_script") as call_script:
            run_response = self.client.post(
                "/test/run", json={"run_id": test_call["run_id"]}
            )
            reused_response = self.client.post(
                "/test/run", json={"run_id": test_call["run_id"]}
            )

        self.assertEqual(200, run_response.status_code)
        self.assertEqual({"script_id": "script.notify_mike"}, run_response.get_json())
        call_script.assert_called_once_with(
            self.state,
            "script.notify_mike",
            {"target": ["light.office"]},
        )
        self.assertEqual(404, reused_response.status_code)

    def test_selected_satellite_is_resolved_shown_and_passed_to_script(self):
        self.state.recognizer.tool_calls = [("notify_mike", {"target": ["Office"]})]
        self.state.hass = Mock()
        self.state.loop = object()  # type: ignore[assignment]
        satellite_request = object()
        self.state.hass.get_satellite_info.return_value = satellite_request
        satellite = SatelliteInfo(
            entity_id="assist_satellite.kitchen",
            device_id="device-1",
            area_id="kitchen",
            floor_id="ground_floor",
            media_player_id="media_player.kitchen",
        )
        satellite_future = Mock()
        satellite_future.result.return_value = satellite

        with patch(
            "web_server.asyncio.run_coroutine_threadsafe",
            return_value=satellite_future,
        ) as submit:
            test_response = self.client.post(
                "/test",
                json={
                    "text": "notify Mike",
                    "language": "de",
                    "satellite_id": "assist_satellite.kitchen",
                },
            )

        self.assertEqual(200, test_response.status_code)
        self.state.hass.get_satellite_info.assert_called_once_with(
            satellite_id="assist_satellite.kitchen"
        )
        submit.assert_called_once_with(satellite_request, self.state.loop)
        satellite_future.result.assert_called_once_with(timeout=60.0)
        test_call = test_response.get_json()["calls"][0]
        self.assertEqual(
            {
                "entity_id": "assist_satellite.kitchen",
                "device_id": "device-1",
                "area_id": "kitchen",
                "floor_id": "ground_floor",
                "media_player_id": "media_player.kitchen",
                "language": "de",
            },
            test_call["variables"]["satellite"],
        )

        with patch("web_server._call_script") as call_script:
            run_response = self.client.post(
                "/test/run", json={"run_id": test_call["run_id"]}
            )

        self.assertEqual(200, run_response.status_code)
        call_script.assert_called_once_with(
            self.state,
            "script.notify_mike",
            {
                "target": ["light.office"],
                "satellite": test_call["variables"]["satellite"],
            },
        )

    def test_unknown_satellite_is_rejected(self):
        response = self.client.post(
            "/test",
            json={
                "text": "notify Mike",
                "satellite_id": "assist_satellite.unknown",
            },
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual({"error": "Unknown satellite"}, response.get_json())

    def test_unresolved_test_call_has_no_run_token(self):
        self.state.recognizer.tool_calls = [("notify_mike", {"target": ["Unknown"]})]

        response = self.client.post(
            "/test",
            json={"text": "notify someone unknown", "language": "en"},
        )

        self.assertEqual(200, response.status_code)
        test_call = response.get_json()["calls"][0]
        self.assertFalse(test_call["can_run"])
        self.assertNotIn("run_id", test_call)

    def test_settings_page_shows_model_runtime(self):
        response = self.client.get("/settings")

        self.assertEqual(200, response.status_code)
        self.assertIn(b"google_gemma-4-E2B-it-Q5_K_M.gguf", response.data)
        self.assertIn(b"768", response.data)
        self.assertIn(b">4</div>", response.data)
        self.assertIn(b"Enabled", response.data)
        self.assertIn(b'value="128"', response.data)
        self.assertIn(b'aria-current="page">Settings</a>', response.data)

    def test_settings_page_updates_and_persists_max_tokens(self):
        response = self.client.post("/settings", json={"max_tokens": 256})

        self.assertEqual(200, response.status_code)
        self.assertEqual(256, response.get_json()["max_tokens"])
        self.assertEqual([256], self.state.recognizer.max_token_calls)
        self.assertEqual(256, self.state.overrides.max_tokens)
        self.assertEqual(256, load(self.state.overrides_path).max_tokens)

    def test_settings_page_rejects_invalid_max_tokens(self):
        response = self.client.post("/settings", json={"max_tokens": 8})

        self.assertEqual(400, response.status_code)
        self.assertEqual([], self.state.recognizer.max_token_calls)

    def test_settings_page_rolls_back_when_persistence_fails(self):
        with patch("web_server.overrides.save", side_effect=OSError("disk full")):
            response = self.client.post("/settings", json={"max_tokens": 256})

        self.assertEqual(500, response.status_code)
        self.assertEqual([256, 128], self.state.recognizer.max_token_calls)
        self.assertEqual(128, self.state.recognizer.max_tokens)
        self.assertIsNone(self.state.overrides.max_tokens)

    def test_settings_page_shows_the_prompts_and_their_placeholders(self):
        response = self.client.get("/settings")

        self.assertEqual(200, response.status_code)
        self.assertIn(b"Sentence: &#34;{text}&#34;", response.data)
        self.assertIn(b"<code>{weekday}</code>", response.data)
        self.assertIn(b"Using the built-in defaults.", response.data)

    def test_prompts_are_applied_and_persisted(self):
        response = self.client.post(
            "/settings/prompts",
            json={
                "system_prompt": "Only call one tool.\r\n",
                "user_prompt": '"{text}" on {weekday}',
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertTrue(response.get_json()["customized"])
        self.assertEqual("Only call one tool.", self.state.recognizer.system_prompt)
        self.assertEqual(
            "Only call one tool.", load(self.state.overrides_path).system_prompt
        )
        self.assertEqual(
            '"{text}" on {weekday}', load(self.state.overrides_path).user_prompt
        )

    def test_default_prompts_are_not_recorded_as_overrides(self):
        self.state.overrides.system_prompt = "Only call one tool."

        response = self.client.post(
            "/settings/prompts",
            json={
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "user_prompt": DEFAULT_USER_PROMPT,
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertFalse(response.get_json()["customized"])
        self.assertIsNone(self.state.overrides.system_prompt)
        self.assertIsNone(load(self.state.overrides_path).system_prompt)

    def test_prompt_without_the_sentence_is_rejected(self):
        response = self.client.post(
            "/settings/prompts",
            json={
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "user_prompt": "Language: {language}",
            },
        )

        self.assertEqual(400, response.status_code)
        self.assertIn("{text}", response.get_json()["error"])
        self.assertEqual([], self.state.recognizer.prompt_calls)

    def test_prompts_roll_back_when_persistence_fails(self):
        with patch("web_server.overrides.save", side_effect=OSError("disk full")):
            response = self.client.post(
                "/settings/prompts",
                json={
                    "system_prompt": "Only call one tool.",
                    "user_prompt": DEFAULT_USER_PROMPT,
                },
            )

        self.assertEqual(500, response.status_code)
        self.assertEqual(
            [
                ("Only call one tool.", DEFAULT_USER_PROMPT),
                (DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT),
            ],
            self.state.recognizer.prompt_calls,
        )
        self.assertEqual(DEFAULT_SYSTEM_PROMPT, self.state.recognizer.system_prompt)
        self.assertIsNone(self.state.overrides.system_prompt)

    def test_field_details_are_available_as_json_for_dialog(self):
        response = self.client.get(
            "/names/field?script=notify_mike&field=target&format=json"
        )

        self.assertEqual(200, response.status_code)
        data = response.get_json()
        self.assertEqual("target", data["field"])
        self.assertEqual("entity", data["kind"])
        self.assertEqual(["Office Light", "Office"], data["rows"][0]["names"])


if __name__ == "__main__":
    unittest.main()
