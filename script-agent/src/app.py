#!/usr/bin/env python3

import argparse
import asyncio
import contextlib
import logging
import signal
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from wyoming.server import AsyncServer

import overrides
from const import BASE_DIR, AppState
from gemma4_recognizer import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    Gemma4Recognizer,
    validate_prompts,
)
from hass_api import HomeAssistant
from intent_server import ScriptAgentEventHandler
from web_server import make_web_server, run_web_server

_LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


def _log_arguments(args: argparse.Namespace) -> None:
    """Log startup arguments without exposing the Home Assistant credential."""
    values = vars(args).copy()
    values["hass_token"] = "<redacted>"
    _LOGGER.debug("Arguments: %s", values)


async def main() -> None:
    """Run app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    #
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=5000)
    #
    parser.add_argument("--hass-token", required=True)
    parser.add_argument("--hass-api", default="http://homeassistant.local:8123")
    #
    parser.add_argument(
        "--hf-repo",
        default="bartowski/google_gemma-4-E2B-it-GGUF",
        help="Hugging Face repo for Gemma 4 (official: ggml-org/gemma-4-E2B-it-GGUF)",
    )
    parser.add_argument(
        "--hf-filename",
        default="google_gemma-4-E2B-it-Q5_K_M.gguf",
        help="Gemma 4 model filename (official: gemma-4-E2B-it-Q8_0.gguf)",
    )
    parser.add_argument(
        "--tool-call-cache-size",
        type=int,
        default=100,
        help="Number of sentences to remember for tool calls",
    )
    parser.add_argument(
        "--llama-state", required=True, help="Path to save llama.cpp state"
    )
    parser.add_argument(
        "--n-ctx", type=int, default=0, help="Size of model context (0 = auto)"
    )
    parser.add_argument(
        "--n-ctx-overhead",
        type=int,
        default=128,
        help="Number of tokens expected beyond system prompt (only when n_ctx = auto)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=0,
        help="CPU threads for the model (0 = llama.cpp default). "
        "Throughput is memory-bandwidth bound, so leave headroom for Home "
        "Assistant on the same box rather than using every core.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of model layers to offload to the GPU (-1 = all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens the model can generate",
    )
    parser.add_argument(
        "--flash-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use llama.cpp flash attention (default: enabled)",
    )
    parser.add_argument(
        "--benchmark-fixture",
        default=None,
        help="Path to benchmark fixture YAML (default: bundled benchmark.yaml)",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="Path to the YAML file recording which scripts are targeted when "
        "that differs from Home Assistant's voice exposure",
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _log_arguments(args)

    _LOGGER.info("Loading scripts from Home Assistant")
    hass = HomeAssistant(token=args.hass_token, api_url=args.hass_api)
    hass_info = await hass.get_home_info()

    # Home Assistant's voice exposure and names are the defaults; the overrides
    # file records the deliberate exceptions. Names are applied while the tools
    # are built, so load them first.
    overrides_path = Path(args.overrides) if args.overrides else None
    all_overrides = overrides.load(overrides_path)

    all_tools = await hass.get_script_tools(hass_info, all_overrides.names)

    # Scripts that are not targeted are still kept for the web UI, which shows
    # what targeting they would get.
    if all_tools:
        # Only prune against a list we actually got: an empty list means
        # something is wrong with Home Assistant, not that everything is gone.
        if overrides.prune_all(all_overrides, all_tools, hass_info):
            if overrides_path is not None:
                overrides.save(overrides_path, all_overrides)

    script_tools = all_overrides.scripts.select(all_tools)
    _LOGGER.debug(
        "Loaded %s script(s), %s targeted (%s exposed to voice)",
        len(all_tools),
        len(script_tools),
        sum(1 for tool in all_tools if tool.exposed),
    )

    if not script_tools:
        _LOGGER.warning(
            "No scripts are targeted. Expose a script to voice, or enable one in "
            "the web UI. Agent will not function."
        )

    # Prompts edited in the web UI replace the defaults. They are validated here
    # rather than trusted: a prompt saved by an older version, or hand-edited in
    # the overrides file, must not leave the app unable to recognize anything.
    system_prompt = DEFAULT_SYSTEM_PROMPT
    user_prompt = DEFAULT_USER_PROMPT
    if (all_overrides.system_prompt is not None) or (
        all_overrides.user_prompt is not None
    ):
        try:
            system_prompt, user_prompt = validate_prompts(
                all_overrides.system_prompt or system_prompt,
                all_overrides.user_prompt or user_prompt,
            )
        except ValueError as err:
            _LOGGER.warning("Using the default prompts: %s", err)
            all_overrides.system_prompt = None
            all_overrides.user_prompt = None
            system_prompt = DEFAULT_SYSTEM_PROMPT
            user_prompt = DEFAULT_USER_PROMPT

    _LOGGER.info(
        "Loading Gemma 4 (repo=%s, filename=%s)", args.hf_repo, args.hf_filename
    )
    recognizer = Gemma4Recognizer(
        repo_id=args.hf_repo.strip(),
        filename=args.hf_filename.strip(),
        state_path=args.llama_state,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cache_size=args.tool_call_cache_size,
        n_ctx=args.n_ctx if args.n_ctx > 0 else None,
        n_ctx_overhead=args.n_ctx_overhead,
        n_threads=args.n_threads if args.n_threads > 0 else None,
        n_gpu_layers=args.n_gpu_layers,
        max_tokens=all_overrides.max_tokens or args.max_tokens,
        flash_attn=args.flash_attention,
        debug=args.debug,
    )

    # Single worker serializes all model access (live recognition + benchmark).
    llama_executor = ThreadPoolExecutor(max_workers=1)

    state = AppState(
        hass=hass,
        hass_info=hass_info,
        tools={t.name: t for t in script_tools},
        recognizer=recognizer,
        llama_executor=llama_executor,
        all_tools=all_tools,
        overrides=all_overrides,
        overrides_path=overrides_path,
    )
    state.set_home_info(hass_info)

    # The web thread schedules Home Assistant reads on this loop.
    loop = asyncio.get_running_loop()
    state.loop = loop

    # Start the web UI before the model loads: the first boot can spend many
    # minutes downloading the model and building the cached state, and the UI
    # shows a "model loading" notice until it is ready. This also lets the
    # container's health check pass while the download is in progress.
    benchmark_fixture_path = args.benchmark_fixture or (BASE_DIR / "benchmark.yaml")
    flask_app = make_web_server(state, benchmark_fixture_path)
    flask_thread = run_web_server(flask_app, host=args.http_host, port=args.http_port)
    flask_thread.start()

    # Load on the model worker thread so all model access stays on one thread
    # (a benchmark request arriving during load queues behind it).
    tools_list = [t.tool for t in script_tools]
    await loop.run_in_executor(llama_executor, recognizer.load, tools_list)

    # Only serve Wyoming once the model is ready: the discovery service waits
    # for this port before telling Home Assistant about the agent.
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    # Handle graceful termination
    stop_event = asyncio.Event()

    def request_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, request_stop)

    server_task = asyncio.create_task(
        server.run(partial(ScriptAgentEventHandler, state))
    )
    stop_task = asyncio.create_task(stop_event.wait())

    try:
        done, _ = await asyncio.wait(
            (server_task, stop_task), return_when=asyncio.FIRST_COMPLETED
        )
        if server_task in done:
            # Do not leave a healthy-looking web UI behind if Wyoming failed.
            await server_task
    finally:
        _LOGGER.info("Shutting down")

        stop_task.cancel()
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_task
        with contextlib.suppress(asyncio.CancelledError):
            await server_task
        llama_executor.shutdown(wait=False, cancel_futures=True)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
