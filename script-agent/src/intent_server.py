import asyncio
import contextlib
import logging
import time
from typing import Any, Dict, List, Optional

from wyoming.asr import Transcript
from wyoming.error import Error
from wyoming.event import Event
from wyoming.handle import Handled, NotHandled
from wyoming.info import Attribution, Describe, HandleModel, HandleProgram, Info
from wyoming.server import AsyncEventHandler

from const import APP_NAME, APP_SLUG, APP_VERSION, AppState
from hass_api import SatelliteInfo
from tool_mapping import MappedCall, map_tool_call

_LOGGER = logging.getLogger(__name__)

# Satellite context is helpful for disambiguation, but must never hold a voice
# command open indefinitely if Home Assistant's registry API stalls.
SATELLITE_TIMEOUT = 10.0
SERVICE_CALL_TIMEOUT = 10.0
MODEL_REBUILDING_TEXT = (
    "The script agent is loading or rebuilding its model cache. "
    "Please try again shortly."
)


async def _wait_for_service_calls(
    service_calls: List[Any], timeout: float = SERVICE_CALL_TIMEOUT
) -> None:
    """Wait until Home Assistant accepts at least one script call."""
    results = await asyncio.gather(
        *(
            asyncio.wait_for(service_call, timeout=timeout)
            for service_call in service_calls
        ),
        return_exceptions=True,
    )
    errors: List[Exception] = [
        result for result in results if isinstance(result, Exception)
    ]
    if len(errors) == len(results):
        raise errors[0]
    for err in errors:
        if isinstance(err, TimeoutError):
            _LOGGER.error("A script call timed out after %s second(s)", timeout)
        else:
            _LOGGER.error("A script call failed: %s", err)


def _missing_required_error_text(calls: List[MappedCall]) -> str:
    """Describe script calls that are missing required parameters."""
    if len(calls) == 1:
        call = calls[0]
        return (
            f"Cannot run {call.script_id}: missing required parameter(s): "
            f"{', '.join(call.missing_required)}"
        )

    details = "; ".join(
        f"{call.script_id} ({', '.join(call.missing_required)})" for call in calls
    )
    return f"Cannot run scripts because required parameters are missing: {details}"


class ScriptAgentEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        state: AppState,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize event handler."""
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.state = state

        self._info_event: Optional[Event] = None

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        try:
            return await self._handle_event(event)
        except Exception:
            _LOGGER.exception("Error handling event")

        return True

    async def _handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        if Describe.is_type(event.type):
            await self._write_info()
            return True

        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            _LOGGER.debug("Handling: %s", transcript)

            try:
                if self.state.model_rebuilding.is_set() or (
                    not self.state.recognizer.ready
                ):
                    await self.write_event(
                        Error(
                            text=MODEL_REBUILDING_TEXT,
                            code="model-rebuilding",
                        ).event()
                    )
                    return True

                language = transcript.language or "en"
                satellite_info: Optional[SatelliteInfo] = None
                satellite_info_task: Optional[asyncio.Task] = None
                try:
                    if transcript.context:
                        device_id = transcript.context.get("device_id")
                        satellite_id = transcript.context.get("satellite_id")
                        satellite_info_task = asyncio.create_task(
                            self.state.hass.get_satellite_info(device_id, satellite_id)
                        )

                    model_future = self.state.llama_executor.submit(
                        self.state.recognizer.get_tool_calls,
                        transcript.text,
                        language,
                    )
                    # Close the small cross-thread race between the check above
                    # and a rebuild being queued. A queued recognition can still
                    # be cancelled; one already running will finish first.
                    if self.state.model_rebuilding.is_set() and model_future.cancel():
                        await self.write_event(
                            Error(
                                text=MODEL_REBUILDING_TEXT,
                                code="model-rebuilding",
                            ).event()
                        )
                        return True

                    tool_calls, response_text = await asyncio.wrap_future(model_future)
                    if not tool_calls:
                        await self.write_event(
                            NotHandled(
                                text=response_text, context=transcript.context
                            ).event()
                        )
                        return True

                    if satellite_info_task:
                        try:
                            satellite_info = await asyncio.wait_for(
                                satellite_info_task, timeout=SATELLITE_TIMEOUT
                            )
                        except Exception as err:  # pylint: disable=broad-except
                            _LOGGER.warning(
                                "Could not get satellite context; continuing: %s", err
                            )

                    mapped_calls: List[MappedCall] = []
                    calls_missing_required: List[MappedCall] = []
                    for tool_id, tool_args in tool_calls:
                        # Only targeted tools may run; models can still invent a
                        # name that is not a tool at all.
                        tool = self.state.tools.get(tool_id)
                        if tool is None:
                            _LOGGER.warning("Ignoring unknown tool: %s", tool_id)
                            continue

                        # The satellite's area is how a name shared by several
                        # things gets narrowed down.
                        call = map_tool_call(
                            tool, tool_args, satellite_info, self.state.geometry
                        )
                        if call.unresolved:
                            _LOGGER.warning(
                                "Dropping unresolvable field(s) of %s: %s",
                                call.script_id,
                                call.unresolved,
                            )
                        if call.ambiguous:
                            _LOGGER.warning(
                                "Dropping ambiguous field(s) of %s: %s",
                                call.script_id,
                                call.ambiguous,
                            )

                        if not call.can_run:
                            _LOGGER.warning(
                                "Not running %s: required field(s) %s could not be "
                                "resolved to anything in this home",
                                call.script_id,
                                call.missing_required,
                            )
                            calls_missing_required.append(call)
                            continue

                        mapped_calls.append(call)

                    if calls_missing_required:
                        error_text = _missing_required_error_text(
                            calls_missing_required
                        )
                        await self.write_event(
                            Error(
                                text=error_text,
                                code="missing-required-parameters",
                            ).event()
                        )
                        return True

                    service_calls = []
                    for call in mapped_calls:
                        variables: Dict[str, Any] = dict(call.variables)
                        if satellite_info:
                            variables["satellite"] = satellite_info.as_script_variable(
                                language
                            )

                        _LOGGER.debug(
                            "Calling script %s with variables %s",
                            call.script_id,
                            variables,
                        )
                        service_calls.append(
                            self.state.hass.call_service(
                                "script",
                                "turn_on",
                                service_data={"variables": variables},
                                target={"entity_id": call.script_id},
                            )
                        )

                    if not service_calls:
                        # Every call was unknown or unresolvable, so nothing ran.
                        await self.write_event(
                            NotHandled(
                                text=response_text, context=transcript.context
                            ).event()
                        )
                        return True

                    # Do not claim the request was handled until Home Assistant
                    # has accepted at least one script call. Multiple calls are
                    # dispatched concurrently.
                    await _wait_for_service_calls(service_calls)

                    await self.write_event(
                        Handled(text="", context=transcript.context).event()
                    )
                finally:
                    # A no-match response returns before satellite context is
                    # needed. Do not leave its registry task orphaned.
                    if satellite_info_task and not satellite_info_task.done():
                        satellite_info_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await satellite_info_task
            except Exception:
                _LOGGER.exception("Unexpected error during handling")
                await self.write_event(
                    Error(
                        text="Unexpected error during handling", code="handle-error"
                    ).event()
                )

            return True

        return True

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        info = Info(
            handle=[
                HandleProgram(
                    name=APP_SLUG,
                    attribution=Attribution(
                        "Open Home Foundation Voice", "https://github.com/OHF-Voice"
                    ),
                    installed=True,
                    description=APP_NAME,
                    version=APP_VERSION,
                    models=[
                        HandleModel(
                            name="gemma4",
                            attribution=Attribution(
                                "Google DeepMind",
                                "https://deepmind.google/models/gemma/gemma-4/",
                            ),
                            installed=True,
                            description="gemma4",
                            version="",
                            languages=[],  # all languages
                        )
                    ],
                    supports_home_control=True,
                )
            ],
        )

        self._info_event = info.event()
        await self.write_event(self._info_event)
