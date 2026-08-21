"""Coqui STT CTC backend via the ``stt_onlyprobs`` TFLite helper.

``stt_onlyprobs`` loads a Coqui ``model.tflite`` and streams per-frame acoustic
probabilities over a simple stdin/stdout protocol:

  * write ``uint32`` length (native endianness) followed by that many bytes of
    16-bit PCM audio (repeatable);
  * write ``uint32`` 0 to flush and request decoding;
  * read one line of space-separated floats per frame, terminated by a blank
    line. Each line has ``alphabet_size + 1`` values (blank is the last class).

The process is kept alive across calls. Build the helper separately (see the
project README); its path is given via ``stt_binary`` or ``$STT_ONLYPROBS``.
"""

import os
import struct
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..tokenizer import CharTokenizer
from .base import AcousticModel

_LOG_FLOOR = 1e-12


class CoquiModel(AcousticModel):
    def __init__(
        self,
        model_dir: Path,
        stt_binary: Optional[Path] = None,
        model_path: Optional[Path] = None,
    ):
        model_dir = Path(model_dir)
        self.tokenizer = CharTokenizer.from_alphabet_file(model_dir / "alphabet.txt")
        self._model_path = Path(model_path or (model_dir / "model.tflite"))

        binary = stt_binary or os.environ.get("STT_ONLYPROBS")
        if not binary:
            raise ValueError(
                "Coqui backend needs the stt_onlyprobs binary: pass stt_binary= "
                "or set $STT_ONLYPROBS"
            )
        self._binary = Path(binary)
        if not self._binary.exists():
            raise FileNotFoundError(f"stt_onlyprobs not found: {self._binary}")
        self._num_classes = self.tokenizer.num_tokens + 1
        self._proc: Optional[subprocess.Popen] = None

    def _ensure_proc(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [str(self._binary), str(self._model_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        return self._proc

    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        proc = self._ensure_proc()
        assert proc.stdin is not None and proc.stdout is not None

        pcm = np.clip(audio, -1.0, 1.0)
        pcm_bytes = (pcm * 32767.0).astype("<i2").tobytes()

        proc.stdin.write(struct.pack("=I", len(pcm_bytes)))
        proc.stdin.write(pcm_bytes)
        proc.stdin.write(struct.pack("=I", 0))  # flush / decode
        proc.stdin.flush()

        rows: List[List[float]] = []
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("stt_onlyprobs exited unexpectedly")
            text = line.decode("utf-8").strip()
            if not text:
                break  # blank line ends the utterance
            rows.append([float(v) for v in text.split()])

        probs = np.asarray(rows, dtype=np.float32)
        if probs.ndim != 2 or probs.shape[1] != self._num_classes:
            raise RuntimeError(
                f"Expected [T, {self._num_classes}] probs, got {probs.shape}"
            )
        # Convert probabilities to natural-log domain for the FST decoder.
        return np.log(np.maximum(probs, _LOG_FLOOR)).astype(np.float32)

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:  # pylint: disable=broad-except
                self._proc.kill()
        self._proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass
