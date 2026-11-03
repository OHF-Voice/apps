"""Kaldi wake word verification, vendored from wyoming-wake-kaldi.

Four grammar-constrained decodes of one acoustic model -- the wake phrase, a
free background loop, the phrase minus its last word, and the phrase's known
confusables -- give a verdict and an alignment. See ``verify.py`` for what
the four costs mean and ``train.py`` for how the graphs are built.

This is a copy, not a dependency, for three reasons. The original is a
Wyoming server, and this app does not speak Wyoming to it: it reached past
the server into ``WakeService`` and its internals, which is not an interface
anyone promised to keep. Those internals need no Wyoming at all -- dropping
``handler.py`` and ``__main__.py``, which were the server, left code that
imports nothing but the standard library and PyYAML. And a copy under a name
of its own cannot be shadowed by an installed or editable copy of the
original, which is a failure that looks like a wrong answer rather than an
import error.

What is here is the original's computation, unchanged except for the
decoder daemon's default path (``decoderd/`` moved in with the code) and
three comments that said "the service". What is not here is the Wyoming
server (``handler.py``, ``__main__.py``) and the object that held one
verifier per wake word (``service.py``), whose work ``KaldiVerifier`` now
does directly.

It still needs the Kaldi and OpenFST binaries and an acoustic model, which
are staged into ``local/`` -- see the README. Nothing here downloads them.
"""

from .config import ConfigError, ServiceConfig, WakeWordConfig, load_config
from .kaldi import SpeechTools
from .model import PreparedModel, prepare_model
from .pool import ResidentDecoder, maybe_start_decoder
from .train import train_wake_word
from .verify import Verifier, VerifyResult

__all__ = [
    "ConfigError",
    "PreparedModel",
    "ResidentDecoder",
    "ServiceConfig",
    "SpeechTools",
    "VerifyResult",
    "Verifier",
    "WakeWordConfig",
    "load_config",
    "maybe_start_decoder",
    "prepare_model",
    "train_wake_word",
]
