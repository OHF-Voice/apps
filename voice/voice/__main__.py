"""Run the app.

There is nothing to configure: the port is fixed, the verifier is whichever
one finds its models, and what Home Assistant needs to know it asks for.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from aiohttp import web

from . import NAME, __version__
from .server import build_app
from .verify import SPLIT_NONE, SPLITS, load_verifier

_LOGGER = logging.getLogger(__name__)

DEFAULT_PORT = 10800

_DIR = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(prog="voice")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--wake-config",
        default=str(_DIR / "wake.yaml"),
        help="Wake word verifier configuration",
    )
    parser.add_argument(
        "--train-dir",
        help="Where wake word decoding graphs are built (default: beside the config)",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default=SPLIT_NONE,
        help=(
            "Which boundary to send back for speech-to-text to start at: "
            "none keeps the satellite's own (default), kaldi sends where the "
            "decode aligned the end of the wake phrase"
        ),
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    verifier = load_verifier(
        Path(args.wake_config) if args.wake_config else None,
        Path(args.train_dir) if args.train_dir else None,
        split=args.split,
    )
    try:
        await verifier.start()
    except Exception:  # pylint: disable=broad-except
        # Better to serve an app that verifies nothing than to leave Home
        # Assistant waiting on a container that restarts forever.
        _LOGGER.exception("Verifier failed to start; nothing will be verified")
        from .verify import NoVerifier

        verifier = NoVerifier()

    words = ", ".join(w.id for w in verifier.wake_words)
    _LOGGER.info("Verifying: %s", words or "(nothing)")

    app = build_app(verifier, NAME, __version__)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    _LOGGER.info("%s %s listening on %s:%s", NAME, __version__, args.host, args.port)

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await verifier.stop()


if __name__ == "__main__":
    main()
