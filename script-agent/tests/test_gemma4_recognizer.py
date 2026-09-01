import pickle
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

try:
    import llama_cpp  # noqa: F401
except ModuleNotFoundError:
    llama_cpp_stub = ModuleType("llama_cpp")
    llama_cpp_stub.Llama = object
    llama_cpp_stub.__version__ = "test"
    sys.modules["llama_cpp"] = llama_cpp_stub

from gemma4_recognizer import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    LLAMA_CPP_VERSION,
    Gemma4Recognizer,
    _get_tools_hash,
    _parse_tool_calls,
    format_prompt,
    prompt_values,
    validate_prompts,
)


class _FakeLlama:
    def __init__(self, n_ctx=64, load_error=None, response=None):
        self._n_ctx = n_ctx
        self.load_error = load_error
        self.response = response
        self.create_calls = 0
        self.loaded_states = []

    def n_ctx(self):
        return self._n_ctx

    def create_chat_completion(self, **kwargs):
        self.create_calls += 1
        return self.response or {}

    def save_state(self):
        return {"state": "good"}

    def load_state(self, state):
        self.loaded_states.append(state)
        if self.load_error:
            raise self.load_error


class ToolCallParserTests(unittest.TestCase):
    def test_single_quoted_value_can_contain_comma(self):
        text = (
            "<|tool_call>call:play{artist:<|'|>Earth, Wind & Fire<|'|>}" "<tool_call|>"
        )

        self.assertEqual(
            [("play", {"artist": "Earth, Wind & Fire"})],
            _parse_tool_calls(text),
        )

    def test_malformed_argument_drops_call_instead_of_raising(self):
        text = "<|tool_call>call:demo{broken}<tool_call|>"

        self.assertEqual([], _parse_tool_calls(text))


class RecognitionTests(unittest.TestCase):
    def test_truncated_response_is_not_partially_executed(self):
        recognizer = Gemma4Recognizer(state_path="unused.bin", max_tokens=64)
        recognizer.tools = [{"type": "function", "function": {"name": "demo"}}]
        recognizer.llm = _FakeLlama(
            response={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": (
                                "<|tool_call>call:demo{message:<|'|>partial<|'|>}"
                                "<tool_call|><|tool_call>call:other{value:"
                            )
                        },
                    }
                ]
            }
        )

        calls, text = recognizer.get_tool_calls("test")

        self.assertEqual([], calls)
        self.assertIn("64-token generation limit", text)


class ModelCreationTests(unittest.TestCase):
    @patch("gemma4_recognizer.Llama")
    @patch("gemma4_recognizer.hf_hub_download", return_value="/model")
    def test_gpu_layers_are_passed_to_llama(self, _download, llama):
        recognizer = Gemma4Recognizer(
            state_path="unused.bin",
            n_gpu_layers=-1,
        )

        recognizer._create_llm(256)  # pylint: disable=protected-access

        self.assertEqual(-1, llama.call_args.kwargs["n_gpu_layers"])


class PromptTests(unittest.TestCase):
    def test_default_user_prompt_carries_the_current_date(self):
        # Without it the model cannot turn "Saturday" into a date, and answers a
        # datetime field with the word instead.
        content = format_prompt(
            DEFAULT_USER_PROMPT,
            prompt_values("book it for saturday", "en", datetime(2026, 8, 31, 9, 5)),
        )

        self.assertIn('Sentence: "book it for saturday"', content)
        self.assertIn("Monday", content)
        self.assertIn("2026-08-31", content)

    def test_unknown_placeholder_is_left_alone_rather_than_raising(self):
        self.assertEqual(
            "say {something} about hi",
            format_prompt("say {something} about {text}", {"text": "hi"}),
        )

    def test_default_prompts_are_valid_and_already_normalized(self):
        # Written normalized so that restoring them in the web UI leaves the
        # saved and running prompts identical, rather than one rebuild apart.
        system_prompt, user_prompt = validate_prompts(
            DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
        )

        self.assertEqual(DEFAULT_SYSTEM_PROMPT, system_prompt)
        self.assertEqual(DEFAULT_USER_PROMPT, user_prompt)

    def test_user_prompt_without_the_sentence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"must include \{text\}"):
            validate_prompts(DEFAULT_SYSTEM_PROMPT, "Language: {language}")

    def test_misspelled_placeholder_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\{weekdya\}"):
            validate_prompts(DEFAULT_SYSTEM_PROMPT, "{text} on {weekdya}")

    def test_empty_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "System prompt cannot be empty"):
            validate_prompts("  \n ", DEFAULT_USER_PROMPT)

    def test_cache_key_follows_the_date_the_prompt_carried(self):
        recognizer = Gemma4Recognizer(state_path="unused.bin", cache_size=10)
        recognizer.tools = []
        recognizer.llm = _FakeLlama(
            response={"choices": [{"message": {"content": "no tools"}}]}
        )

        with patch("gemma4_recognizer.datetime") as clock:
            clock.now.return_value = datetime(2026, 8, 31, 9, 5)
            recognizer.get_tool_calls("book it for saturday")
            recognizer.get_tool_calls("book it for saturday")
            self.assertEqual(1, recognizer.llm.create_calls)

            # Same sentence, next day: the answer would be a day early.
            clock.now.return_value = datetime(2026, 9, 1, 9, 5)
            recognizer.get_tool_calls("book it for saturday")

        self.assertEqual(2, recognizer.llm.create_calls)


class PromptSettingsTests(unittest.TestCase):
    def _recognizer(self):
        recognizer = Gemma4Recognizer(state_path="unused.bin", cache_size=10)
        recognizer.tools = [{"type": "function", "function": {"name": "demo"}}]
        recognizer.llm = _FakeLlama(n_ctx=1024)
        recognizer.ready = True
        return recognizer

    def test_form_round_trip_does_not_rebuild(self):
        recognizer = self._recognizer()
        recognizer.cache.set("en: cached", ([], "cached"))

        with patch.object(recognizer, "_restore_or_build_state") as restore:
            # What a textarea gives back: no leading newline, CRLF line endings.
            recognizer.set_prompts(
                DEFAULT_SYSTEM_PROMPT.strip().replace("\n", "\r\n"),
                DEFAULT_USER_PROMPT,
            )

        restore.assert_not_called()
        self.assertEqual(DEFAULT_SYSTEM_PROMPT, recognizer.system_prompt)
        self.assertIsNotNone(recognizer.cache.get("en: cached"))

    def test_user_prompt_change_clears_the_cache_without_rebuilding(self):
        recognizer = self._recognizer()
        recognizer.cache.set("en: cached", ([], "cached"))

        with patch.object(recognizer, "_restore_or_build_state") as restore:
            recognizer.set_prompts(DEFAULT_SYSTEM_PROMPT, "{text} ({language})")

        restore.assert_not_called()
        self.assertEqual("{text} ({language})", recognizer.user_prompt)
        self.assertIsNone(recognizer.cache.get("en: cached"))
        self.assertTrue(recognizer.ready)

    def test_system_prompt_change_rebuilds_the_prefix(self):
        recognizer = self._recognizer()

        with patch.object(recognizer, "_restore_or_build_state") as restore:
            recognizer.set_prompts("Only call one tool.", DEFAULT_USER_PROMPT)

        restore.assert_called_once_with()
        self.assertEqual("Only call one tool.", recognizer.system_prompt)
        self.assertEqual("Only call one tool.", recognizer.system_message["content"])
        self.assertTrue(recognizer.ready)

    def test_rejected_prompt_leaves_the_running_ones_alone(self):
        recognizer = self._recognizer()

        with self.assertRaises(ValueError):
            recognizer.set_prompts("Only call one tool.", "no sentence here")

        self.assertEqual(DEFAULT_SYSTEM_PROMPT, recognizer.system_prompt)
        self.assertEqual(DEFAULT_USER_PROMPT, recognizer.user_prompt)

    def test_failed_rebuild_rolls_the_prompts_back(self):
        recognizer = self._recognizer()
        old_llm = recognizer.llm

        with patch.object(
            recognizer,
            "_restore_or_build_state",
            side_effect=RuntimeError("prefix failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "prefix failed"):
                recognizer.set_prompts("Only call one tool.", DEFAULT_USER_PROMPT)

        self.assertEqual(DEFAULT_SYSTEM_PROMPT, recognizer.system_prompt)
        self.assertEqual(DEFAULT_SYSTEM_PROMPT, recognizer.system_message["content"])
        self.assertIs(old_llm, recognizer.llm)
        self.assertEqual([{"state": "good"}], old_llm.loaded_states)
        self.assertTrue(recognizer.ready)

    def test_fixed_context_rejects_prompts_that_will_not_fit(self):
        recognizer = self._recognizer()
        recognizer.n_ctx = 1024

        with patch.object(recognizer, "required_n_ctx", return_value=2048):
            with self.assertRaisesRegex(ValueError, "fixed context is 1024"):
                recognizer.set_prompts("Only call one tool.", DEFAULT_USER_PROMPT)

        self.assertEqual(DEFAULT_SYSTEM_PROMPT, recognizer.system_prompt)


class RuntimeSettingsTests(unittest.TestCase):
    def test_max_tokens_grows_automatic_context(self):
        recognizer = Gemma4Recognizer(
            state_path="unused.bin", max_tokens=64, n_ctx=None
        )
        recognizer.tools = [{"type": "function", "function": {"name": "demo"}}]
        recognizer.llm = _FakeLlama(n_ctx=64)
        recognizer.ready = True

        def create_larger(n_ctx):
            recognizer.llm = _FakeLlama(n_ctx=n_ctx)

        with patch.object(recognizer, "required_n_ctx", return_value=128), patch.object(
            recognizer, "_create_llm", side_effect=create_larger
        ), patch.object(recognizer, "_restore_or_build_state") as restore:
            recognizer.set_max_tokens(128)

        self.assertEqual(128, recognizer.max_tokens)
        self.assertEqual(128, recognizer.llm.n_ctx())
        self.assertTrue(recognizer.ready)
        restore.assert_called_once_with()

    def test_max_tokens_rolls_back_when_context_growth_fails(self):
        recognizer = Gemma4Recognizer(
            state_path="unused.bin", max_tokens=64, n_ctx=None
        )
        recognizer.tools = [{"type": "function", "function": {"name": "demo"}}]
        old_llm = _FakeLlama(n_ctx=64)
        recognizer.llm = old_llm
        recognizer.ready = True

        def create_larger(n_ctx):
            recognizer.llm = _FakeLlama(n_ctx=n_ctx)

        with patch.object(recognizer, "required_n_ctx", return_value=128), patch.object(
            recognizer, "_create_llm", side_effect=create_larger
        ), patch.object(
            recognizer,
            "_restore_or_build_state",
            side_effect=RuntimeError("prefix failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "prefix failed"):
                recognizer.set_max_tokens(128)

        self.assertEqual(64, recognizer.max_tokens)
        self.assertIs(old_llm, recognizer.llm)
        self.assertEqual([{"state": "good"}], old_llm.loaded_states)
        self.assertTrue(recognizer.ready)

    def test_fixed_context_rejects_token_limit_that_will_not_fit(self):
        recognizer = Gemma4Recognizer(state_path="unused.bin", max_tokens=64, n_ctx=64)
        recognizer.tools = [{"type": "function", "function": {"name": "demo"}}]
        recognizer.llm = _FakeLlama(n_ctx=64)

        with patch.object(recognizer, "required_n_ctx", return_value=128):
            with self.assertRaisesRegex(ValueError, "fixed context is 64"):
                recognizer.set_max_tokens(128)

        self.assertEqual(64, recognizer.max_tokens)


class StateCacheTests(unittest.TestCase):
    def test_state_parent_directory_is_created(self):
        tools = [{"type": "function", "function": {"name": "demo"}}]
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "missing" / "nested" / "state.bin"
            recognizer = Gemma4Recognizer(state_path=state_path)
            recognizer.tools = tools
            recognizer.llm = _FakeLlama()
            recognizer.model_path = Path("/model")

            recognizer._restore_or_build_state()  # pylint: disable=protected-access

            self.assertTrue(state_path.is_file())
            self.assertTrue(state_path.with_suffix(".sha256").is_file())
            self.assertFalse(state_path.with_suffix(".tmp").exists())

    def test_corrupt_matching_state_is_rebuilt(self):
        tools = [{"type": "function", "function": {"name": "demo"}}]
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.bin"
            state_path.write_bytes(b"not a pickle")

            recognizer = Gemma4Recognizer(state_path=state_path)
            recognizer.tools = tools
            recognizer.llm = _FakeLlama()
            recognizer.model_path = Path("/model")
            runtime_model_id = (
                f"llama-cpp-python/{LLAMA_CPP_VERSION};"
                "n_ctx=64;n_gpu_layers=0;flash_attn=1;model_path=/model;"
                f"{recognizer.repo_id}/{recognizer.filename}"
            )
            state_path.with_suffix(".sha256").write_text(
                _get_tools_hash(runtime_model_id, tools, recognizer.system_prompt),
                encoding="utf-8",
            )

            recognizer._restore_or_build_state()  # pylint: disable=protected-access

            self.assertEqual(1, recognizer.llm.create_calls)
            with open(state_path, "rb") as state_file:
                self.assertEqual({"state": "good"}, pickle.load(state_file))

    def test_failed_context_growth_restores_old_model(self):
        old_tools = [{"type": "function", "function": {"name": "old"}}]
        new_tools = [{"type": "function", "function": {"name": "new"}}]
        recognizer = Gemma4Recognizer(state_path="unused.bin")
        old_llm = _FakeLlama(n_ctx=64)
        recognizer.llm = old_llm
        recognizer.tools = old_tools
        recognizer.ready = True

        def create_larger(_n_ctx):
            recognizer.llm = _FakeLlama(n_ctx=128)

        with patch.object(recognizer, "required_n_ctx", return_value=128), patch.object(
            recognizer, "_create_llm", side_effect=create_larger
        ), patch.object(
            recognizer,
            "_restore_or_build_state",
            side_effect=RuntimeError("prefix failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "prefix failed"):
                recognizer.reload(new_tools)

        self.assertIs(old_llm, recognizer.llm)
        self.assertEqual(old_tools, recognizer.tools)
        self.assertTrue(recognizer.ready)
        self.assertEqual([{"state": "good"}], old_llm.loaded_states)


if __name__ == "__main__":
    unittest.main()
