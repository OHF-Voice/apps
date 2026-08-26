"""Acoustic-model provisioning.

Models are downloaded from the rhasspy-speech HuggingFace dataset
    https://huggingface.co/datasets/rhasspy/rhasspy-speech/tree/main/models
as ``<name>.tar.gz`` (e.g. ``en_US-coqui``) and extracted into a local models
directory given on the command line (``--models-dir``, default ``/data/models``
in the add-on). Re-download is skipped if the model is already present.

The container image may also ship a model under ``<addon_root>/models`` (see the
Dockerfile's ``BUNDLE_MODEL``). That copy is used in preference to downloading,
so a fresh install starts without network access; it is read-only and never
written to, and ``--models-dir`` still wins if the same model is present there.
"""
import logging
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Optional

_LOGGER = logging.getLogger("speech-to-phrase.models")

HF_BASE = "https://huggingface.co/datasets/rhasspy/rhasspy-speech/resolve/main/models"
# Models baked into the image at build time, if any.
BUNDLED_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TOOLS_BASE = "https://huggingface.co/datasets/rhasspy/rhasspy-speech/resolve/main/tools"

# machine() -> stt_onlyprobs binary name (needed only for the coqui backend).
STT_BINARIES = {
    "x86_64": "stt_onlyprobs.x86_64.bin",
    "amd64": "stt_onlyprobs.x86_64.bin",
    "aarch64": "stt_onlyprobs.arm64.bin",
    "arm64": "stt_onlyprobs.arm64.bin",
}
# A directory is "a model" if it holds an acoustic-model file: *.tflite (coqui),
# *.onnx (citrinet, often named <model>.onnx), or *.fst (kaldi).
MODEL_GLOBS = ("*.tflite", "*.onnx", "*.fst")

# Per-backend default score gate: the max per-token penalty at/below which a
# local transcript is accepted (above it the utterance is treated as
# out-of-grammar and handed to the cloud fallback). The scales differ because
# Citrinet is subword and Coqui is character. Citrinet 5.0 was re-fit on
# tests/en; Coqui 2.0 was re-fit on Common Voice (sl) — the old 1.25 rejected
# ~24% of correctly recognized commands. Users can override globally (add-on
# ``max_score`` option) or per-language in the web UI.
DEFAULT_MAX_SCORE = {"citrinet": 5.0, "coqui": 2.0}


def default_max_score(backend: str) -> float:
    return DEFAULT_MAX_SCORE.get(backend, 5.0)


# Word-insertion reward per emitted token. The recognizer compares the candidate
# it generates with the unbiased decode using the per-token acoustic score, so
# the bonus can recover a long command without winning merely by adding optional
# words. A rejected short candidate gets a stronger rescue attempt only when the
# corrected phrase closely matches the unconstrained CTC transcript. Citrinet
# 2.0 and the rescue thresholds were fit against human English commands and the
# OOV corpus in tests/wav.
# Coqui is 0 because it has not been measured -- its cost scale differs from
# Citrinet's, so borrowing the number would be a guess.
DEFAULT_TOKEN_BONUS = {"citrinet": 2.0, "coqui": 0.0}


def default_token_bonus(backend: str) -> float:
    return DEFAULT_TOKEN_BONUS.get(backend, 0.0)

# language -> {backend: HuggingFace model name}. The repo ships NeMo CTC models
# (citrinet/conformer, ONNX -> "citrinet" backend, runs on onnxruntime with no
# extra binary) and Coqui TFLite models ("coqui" backend, needs stt_onlyprobs).
# Citrinet is preferred where available. Extend as languages are validated.
MODEL_NAMES = {
    "en": {"citrinet": "stt_en_citrinet_512", "coqui": "en_US-coqui"},
    "de": {"citrinet": "stt_de_citrinet_1024", "coqui": "de_DE-coqui"},
    # Spanish is Citrinet-only on purpose: es_ES-coqui does not load at all
    # ("Expected [T, 30] probs, got (T, 36)" -- its alphabet has 36 symbols and
    # the stt_onlyprobs decode path expects 30). Listing it only gave anyone who
    # set backend=coqui a model that downloads and then refuses to run.
    "es": {"citrinet": "stt_es_citrinet_512"},
    # French uses the Conformer, not stt_fr_citrinet_1024_gamma_0_25: that model
    # cannot resolve the "verrouille"/"déverrouille" prefix, decoding "unlock the
    # front door" as "lock the front door" at 0.73 -- confidently, so the score
    # gate does not catch it. Choosing a different lock verb only moves the error
    # to the more dangerous direction. The Conformer decodes both correctly
    # (0.17/0.23) and takes the language from 55/56 to 56/56 commands resolved.
    "fr": {"citrinet": "stt_fr_conformer_ctc_large", "coqui": "fr_FR-rhasspy"},
    "it": {"citrinet": "stt_it_conformer_ctc_large", "coqui": "it_IT-coqui"},
    "zh": {"citrinet": "stt_zh_citrinet_512"},
    "ru": {"citrinet": "stt_ru_conformer_ctc_large"},
    "hr": {"citrinet": "stt_hr_conformer_ctc_large"},
    "hi": {"citrinet": "stt_hi_conformer_ctc_medium"},
    "ca": {"citrinet": "stt_ca_conformer_ctc_large", "coqui": "ca_ES-coqui"},
    # Dutch prefers Citrinet: nl_NL-coqui misrecognises below the score gate
    # ("doe de lichten uit" decoding as "...aan"), so it acts on the wrong
    # command instead of deferring to the cloud. The Citrinet model does not
    # have that failure.
    "nl": {"citrinet": "stt_nl_citrinet_256", "coqui": "nl_NL-coqui"},
    "cs": {"coqui": "cs_CZ-coqui"},
    "sl": {"coqui": "sl_SL-coqui"},
}


def _present(d: Path) -> bool:
    return d.is_dir() and any(any(d.glob(pat)) for pat in MODEL_GLOBS)


def _find_model_dir(root: Path) -> Path:
    """Locate the extracted dir actually containing the model files (the tarball
    may or may not wrap them in a top-level directory)."""
    if _present(root):
        return root
    for d in (p for p in root.rglob("*") if p.is_dir()):
        if _present(d):
            return d
    return root


def model_name_for(language: str, backend: str) -> Optional[str]:
    """The model for exactly ``(language, backend)``, or None if there isn't one.

    Deliberately does not substitute another backend's model. It used to fall
    back to whatever the language had, which meant ``language: cs`` with
    ``backend: citrinet`` downloaded the Coqui model and then died loading it
    (``FileNotFoundError: cs_CZ-coqui/tokens.txt``) on every start. A missing
    model is a configuration answer -- None -- not a different model.
    ``resolve_backend`` is what picks a backend that exists.
    """
    return MODEL_NAMES.get(language, {}).get(backend)


def backends_for(language: str) -> List[str]:
    """Backends that have a model for ``language`` (may be empty)."""
    return list(MODEL_NAMES.get(language, {}))


def resolve_backend(language: str, requested: str) -> str:
    """Turn ``backend="auto"`` into a concrete backend that actually has a model
    for ``language``. Citrinet is preferred (no extra binary, subword scale);
    Coqui is used for languages that ship only a Coqui model (``cs``; Dutch
    moved to Citrinet). Non-auto values pass through unchanged -- and if that
    pairing has no model, ``resolve`` says so rather than substituting one."""
    if requested != "auto":
        return requested
    by_backend = MODEL_NAMES.get(language, {})
    if "citrinet" in by_backend:
        return "citrinet"
    if "coqui" in by_backend:
        return "coqui"
    return "citrinet"


def _extract_safely(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract a downloaded model archive, refusing anything a model has no
    business containing.

    This is the only place a remote source writes to disk, so a member must not
    be able to escape ``dest`` (an absolute path, or one climbing out with
    "..") and must be a plain file or directory -- never a link, device or
    fifo.

    tarfile's ``filter="data"`` does that and is used where available. The
    add-on image has it, but it only arrived in Python 3.12 (backported as far
    as 3.11.4), and the library's floor is 3.11 -- where passing ``filter=`` is
    a TypeError. Since the same code runs on a Debian 12 base and on
    contributors' machines, the rules are applied by hand where the filter is
    absent rather than letting the hardening depend on the interpreter.

    The hand-rolled path is not identical: it refuses an absolute-path member
    outright where the real filter strips the leading separator and extracts it
    under ``dest``, and it does not scrub setuid bits. Both are safe, and no
    model archive contains either.
    """
    if hasattr(tarfile, "data_filter"):
        tf.extractall(dest, filter="data")
        return

    root = dest.resolve()
    for member in tf.getmembers():
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(
                f"refusing archive member {member.name!r}: not a file or "
                f"directory (type {member.type!r})"
            )
        if not (root / member.name).resolve().is_relative_to(root):
            raise RuntimeError(
                f"refusing archive member {member.name!r}: it would extract "
                f"outside {dest}"
            )
    tf.extractall(dest)


def ensure_model(name: str, models_dir: Path) -> Path:
    """Return <models_dir>/<name>, downloading + extracting it if absent.

    A copy bundled into the image is used before reaching for the network, so a
    fresh install with the default language never waits on a download."""
    models_dir = Path(models_dir)
    target = models_dir / name
    if _present(target):
        return target

    bundled = BUNDLED_MODELS_DIR / name
    if _present(bundled):
        _LOGGER.debug("Using acoustic model '%s' bundled in the image", name)
        return bundled

    url = f"{HF_BASE}/{name}.tar.gz"
    _LOGGER.info("Downloading acoustic model '%s' from %s", name, url)
    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "model.tar.gz"
        urllib.request.urlretrieve(url, tar_path)
        extract_dir = Path(td) / "x"
        with tarfile.open(tar_path) as tf:
            _extract_safely(tf, extract_dir)
        src = _find_model_dir(extract_dir)
        if not _present(src):
            raise RuntimeError(f"No model files found in archive for '{name}'")
        tmp_target = models_dir / f".{name}.tmp"
        if tmp_target.exists():
            shutil.rmtree(tmp_target)
        shutil.move(str(src), str(tmp_target))
        tmp_target.rename(target)  # atomic publish
    _LOGGER.info("Acoustic model '%s' ready at %s", name, target)
    return target


def ensure_stt_binary(tools_dir: Path) -> Path:
    """Download the architecture-appropriate stt_onlyprobs binary (used by the
    coqui backend) and point $STT_ONLYPROBS at it. Idempotent."""
    arch = platform.machine().lower()
    name = STT_BINARIES.get(arch)
    if not name:
        raise RuntimeError(f"No stt_onlyprobs binary for architecture {arch!r}")
    tools_dir = Path(tools_dir)
    target = tools_dir / name
    if not target.exists():
        url = f"{TOOLS_BASE}/{name}"
        _LOGGER.info("Downloading stt_onlyprobs (%s) from %s", arch, url)
        tools_dir.mkdir(parents=True, exist_ok=True)
        tmp = tools_dir / f".{name}.tmp"
        urllib.request.urlretrieve(url, tmp)
        tmp.chmod(0o755)
        tmp.rename(target)
        _LOGGER.info("stt_onlyprobs ready at %s", target)
    os.environ["STT_ONLYPROBS"] = str(target)
    return target


def resolve(model: Optional[str], models_dir: Path, language: str,
            backend: str, tools_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve a usable model directory.

    * ``model`` is an existing model dir -> use as-is (dev: point at a checkout).
    * ``model`` is a name -> download/extract it.
    * ``model`` unset -> derive the name from (language, backend) and download.
    Returns None if no mapping exists and nothing was given (caller decides).
    The coqui backend additionally needs the stt_onlyprobs binary, which is
    fetched here and exported via $STT_ONLYPROBS.
    """
    if backend == "coqui":
        ensure_stt_binary(tools_dir or (Path(models_dir).parent / "tools"))
    if model:
        p = Path(model)
        if _present(p):
            return p
        return ensure_model(model, models_dir)
    name = model_name_for(language, backend)
    if not name:
        available = backends_for(language)
        if available:
            _LOGGER.error(
                "No %s model for '%s'; that language ships a model for: %s. "
                "Set backend to one of those (or 'auto') in the add-on options.",
                backend, language, ", ".join(available),
            )
        else:
            _LOGGER.error("No acoustic model for language '%s'", language)
        return None
    return ensure_model(name, models_dir)
