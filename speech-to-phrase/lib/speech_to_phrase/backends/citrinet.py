"""Citrinet CTC backend via onnxruntime.

The exported ONNX takes log-mel features (``audio_signal`` ``[B, 80, T]``) plus a
``length`` and returns ``logprobs`` (already log-softmax). The mel preprocessor
is reproduced in :mod:`speech_to_phrase.features`.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..features import MelFeaturizer
from ..tokenizer import SubwordTokenizer
from .base import AcousticModel


def _find_onnx(model_dir: Path) -> Path:
    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx model found in {model_dir}")
    return onnx_files[0]


class CitrinetModel(AcousticModel):
    def __init__(
        self,
        model_dir: Path,
        onnx_path: Optional[Path] = None,
        spm_model: Optional[Path] = None,
        providers: Optional[List[str]] = None,
    ):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        tokens_path = model_dir / "tokens.json"
        if not tokens_path.exists():
            tokens_path = model_dir / "tokens.txt"
        if spm_model is None:
            for candidate in model_dir.glob("*.model"):
                spm_model = candidate
                break
        self.tokenizer = SubwordTokenizer.from_model_dir(tokens_path, spm_model)

        if onnx_path is None:
            onnx_path = _find_onnx(model_dir)
        self._session = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        self._signal_name = inputs[0].name
        self._length_name = inputs[1].name
        self._featurizer = MelFeaturizer()

    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        feats = self._featurizer(audio)  # [n_mels, T]
        signal = feats[np.newaxis, :, :]
        length = np.array([feats.shape[1]], dtype=np.int64)
        out = self._session.run(
            None, {self._signal_name: signal, self._length_name: length}
        )[0]
        return np.ascontiguousarray(out[0], dtype=np.float32)  # [T, V]
