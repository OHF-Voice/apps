import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

from wyoming.asr import Transcript
from wyoming.error import Error

from hass_api import Tool
from intent_server import (
    ScriptAgentEventHandler,
    _missing_required_error_text,
    _wait_for_service_calls,
)
from tool_mapping import MappedCall


class ServiceCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_successful_call(self):
        completed = []

        async def succeed():
            completed.append(True)

        await _wait_for_service_calls([succeed()])

        self.assertEqual([True], completed)

    async def test_all_failed_calls_raise(self):
        async def fail(message):
            raise RuntimeError(message)

        with self.assertRaisesRegex(RuntimeError, "first"):
            await _wait_for_service_calls([fail("first"), fail("second")])

    async def test_partial_failure_still_counts_as_handled(self):
        async def succeed():
            return None

        async def fail():
            raise RuntimeError("partial")

        await _wait_for_service_calls([fail(), succeed()])

    async def test_stalled_call_times_out(self):
        async def stall():
            await asyncio.Event().wait()

        with self.assertRaises(TimeoutError):
            await _wait_for_service_calls([stall()], timeout=0.01)

    async def test_partial_timeout_still_counts_as_handled(self):
        async def stall():
            await asyncio.Event().wait()

        async def succeed():
            return None

        await _wait_for_service_calls([stall(), succeed()], timeout=0.01)


class MissingRequiredErrorTests(unittest.TestCase):
    def test_single_script_names_missing_parameters(self):
        call = MappedCall(tool_id="device_on_off", missing_required=["device_name"])

        self.assertEqual(
            "Cannot run script.device_on_off: missing required parameter(s): "
            "device_name",
            _missing_required_error_text([call]),
        )

    def test_multiple_scripts_name_each_missing_parameter(self):
        calls = [
            MappedCall(tool_id="first", missing_required=["target"]),
            MappedCall(tool_id="second", missing_required=["action", "device_name"]),
        ]

        self.assertEqual(
            "Cannot run scripts because required parameters are missing: "
            "script.first (target); script.second (action, device_name)",
            _missing_required_error_text(calls),
        )


class MissingRequiredHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_required_parameter_returns_error_without_calling_script(
        self,
    ):
        tool = Tool(
            name="device_on_off",
            tool={
                "type": "function",
                "function": {
                    "name": "device_on_off",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "device_name": {"type": "string"},
                        },
                        "required": ["action", "device_name"],
                    },
                },
            },
        )
        recognizer = SimpleNamespace(
            ready=True,
            get_tool_calls=lambda _text, _language: (
                [("device_on_off", {"action": "on"})],
                "Would run device_on_off with action on",
            ),
        )
        hass = SimpleNamespace(call_service=AsyncMock())

        with ThreadPoolExecutor(max_workers=1) as executor:
            state = SimpleNamespace(
                hass=hass,
                recognizer=recognizer,
                llama_executor=executor,
                tools={"device_on_off": tool},
                geometry=None,
                model_rebuilding=threading.Event(),
            )
            handler = object.__new__(ScriptAgentEventHandler)
            handler.state = state
            handler.write_event = AsyncMock()

            await handler._handle_event(
                Transcript(text="turn on the lights", language="en").event()
            )

        hass.call_service.assert_not_called()
        handler.write_event.assert_awaited_once()
        error = Error.from_event(handler.write_event.await_args.args[0])
        self.assertEqual("missing-required-parameters", error.code)
        self.assertEqual(
            "Cannot run script.device_on_off: missing required parameter(s): "
            "device_name",
            error.text,
        )

    async def test_rebuild_returns_error_without_queuing_recognition(self):
        recognizer = SimpleNamespace(
            ready=True,
            get_tool_calls=unittest.mock.Mock(),
        )
        rebuilding = threading.Event()
        rebuilding.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            state = SimpleNamespace(
                recognizer=recognizer,
                model_rebuilding=rebuilding,
                llama_executor=executor,
            )
            handler = object.__new__(ScriptAgentEventHandler)
            handler.state = state
            handler.write_event = AsyncMock()

            await handler._handle_event(
                Transcript(text="turn on the lights", language="en").event()
            )

        recognizer.get_tool_calls.assert_not_called()
        error = Error.from_event(handler.write_event.await_args.args[0])
        self.assertEqual("model-rebuilding", error.code)


if __name__ == "__main__":
    unittest.main()
