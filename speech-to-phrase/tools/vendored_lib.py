"""Bind ``speech_to_phrase`` to the add-on's vendored ``lib/``.

Import this (and call :func:`bind`) from anything under ``tools/`` or ``tests/``
before importing ``speech_to_phrase`` or a module from ``src/`` that does.

The image installs ``lib/`` and nothing else, so that is the code such a script
has to exercise. A development machine usually also has the upstream library
installed editable, and scikit-build's editable install hooks ``sys.meta_path``
-- which is consulted *before* ``sys.path``, so it shadows ``lib/`` however the
path is ordered. Measuring that copy instead is silent and misleading in both
directions: it has no subword-segmentation lattice, so clear commands decode as
nothing and a language looks broken; and it lacks anything added to ``lib/``
since, so a test fails on an ImportError for a function that does exist.

So: put ``lib/`` on ``sys.path`` *and* drop any finder that claims to own the
name, leaving the ordinary path-based import as the only one that can answer.
"""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def bind() -> None:
    """Make ``import speech_to_phrase`` resolve to ``lib/``. Idempotent."""
    pkg_dir = REPO_ROOT / "lib" / "speech_to_phrase"

    already = sys.modules.get("speech_to_phrase")
    if already is not None and str(pkg_dir) in str(getattr(already, "__file__", "")):
        return

    path_entry = str(pkg_dir.parent)
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

    for finder in list(sys.meta_path):
        if finder is importlib.machinery.PathFinder:
            continue  # the one that honours sys.path, i.e. finds lib/
        try:
            spec = finder.find_spec("speech_to_phrase", None)
        except Exception:  # noqa: BLE001  (a finder that objects can't shadow us)
            continue
        if spec is not None:
            sys.meta_path.remove(finder)

    # The native extension is a build artifact, so in a checkout it sits under
    # lib/build/<wheel-tag>/ rather than next to the sources. Pre-register it so
    # grammar.py's "from . import _fst" finds it there.
    if "speech_to_phrase._fst" not in sys.modules and not any(pkg_dir.glob("_fst*.so")):
        built = sorted((REPO_ROOT / "lib" / "build").glob("*/_fst*.so"))
        if not built:
            raise SystemExit(
                "lib/ has no compiled _fst extension: build it with "
                "`pip install ./lib` (or `pip wheel ./lib`) and re-run"
            )
        spec = importlib.util.spec_from_file_location(
            "speech_to_phrase._fst", built[-1]
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules["speech_to_phrase._fst"] = module
        spec.loader.exec_module(module)
