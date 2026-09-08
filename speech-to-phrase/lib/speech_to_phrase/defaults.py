"""Shared backend names and calibrated recognition defaults."""

from pathlib import Path
from typing import Optional, Union

ENGLISH_MODEL = "stt_en_parakeet_tdt_ctc_110m"
ENGLISH_MODEL_ALIASES = {ENGLISH_MODEL, "parakeet-tdt-ctc-110m"}
LEGACY_ENGLISH_MODELS = {
    "stt_en_citrinet_512",
    "stt_en_citrinet_512_gamma_0_25",
}

# A decode is accepted at or below this per-token penalty. Parakeet's 3.8 gate
# retained 62/67 VPE commands while rejecting every evaluation OOV clip.
# Other NeMo models use the prior 5.0 calibration; Coqui uses its separately
# calibrated character-token scale.
DEFAULT_MAX_SCORE = {"nemo": 5.0, "coqui": 2.0}
MODEL_MAX_SCORE = {model_name: 3.8 for model_name in ENGLISH_MODEL_ALIASES}

# CTC's blank bias favors shorter grammar paths. This reward is used only to
# generate a competing candidate; final selection still uses acoustic penalty.
DEFAULT_TOKEN_BONUS = {"nemo": 2.0, "coqui": 0.0}


def normalize_backend(backend: str) -> str:
    """Return the canonical backend name, accepting the former public spelling."""
    backend = backend.lower()
    return "nemo" if backend == "citrinet" else backend


def default_max_score(backend: str, model: Optional[Union[str, Path]] = None) -> float:
    """Return the calibrated acceptance gate for a model/backend."""
    if model is not None:
        model_name = Path(model).name
        if model_name in MODEL_MAX_SCORE:
            return MODEL_MAX_SCORE[model_name]
    return DEFAULT_MAX_SCORE.get(normalize_backend(backend), 5.0)


def default_token_bonus(backend: str) -> float:
    """Return the word-insertion reward for a backend."""
    return DEFAULT_TOKEN_BONUS.get(normalize_backend(backend), 0.0)
