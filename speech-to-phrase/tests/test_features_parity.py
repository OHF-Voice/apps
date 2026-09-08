#!/usr/bin/env python3
"""Gate on the log-mel front end: assert it still matches librosa exactly.

``lib/speech_to_phrase/_dsp.py`` replaced ``librosa.filters.mel`` and
``librosa.stft`` with local implementations, to drop a dependency that pulls
numba, scikit-learn and llvmlite -- around 250 MB, including one 170 MB shared
object -- for three function calls. The acoustic models were trained on
librosa's features, so "close enough" is not the standard: a front end that
differs even slightly is feeding the network something it never saw. This test
compares the two directly and requires them to be **bit-identical**, at three
levels:

  1. the mel filterbank matrix,
  2. the STFT of real audio, and
  3. the featurizer's whole output, and the acoustic model's log-probabilities
     computed from it -- which is the number that actually decides a transcript.

Resampling is checked separately and only for closeness, because there is
nothing to be exact against: librosa hands that work to soxr as well, and the
only difference is who calls it.

Run it with librosa installed (a development machine); it skips otherwise, since
the whole point of the change is that the add-on image no longer has it.

    python3 tests/test_features_parity.py
"""
import glob
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from vendored_lib import bind as _bind_vendored_lib  # noqa: E402

_bind_vendored_lib()

from speech_to_phrase import load_recognizer  # noqa: E402
from speech_to_phrase._dsp import mel_filterbank, stft  # noqa: E402
from speech_to_phrase.audio import resample  # noqa: E402
from speech_to_phrase.features import MelFeaturizer  # noqa: E402

CLIPS = sorted(glob.glob(str(REPO_ROOT / "tests/wav/.tts_cache/*.wav")))
MODEL = Path("/home/hansenm/opt/speech-to-phrase-lib/local/stt_de_citrinet_1024")

_failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)


def main() -> int:
    try:
        import librosa
    except ImportError:
        print("SKIP: librosa is not installed, so there is nothing to compare against")
        return 0

    import soundfile as sf

    f = MelFeaturizer()

    # 1. The filterbank. Built once at startup, so an error here is systematic.
    theirs = librosa.filters.mel(
        sr=f.sample_rate, n_fft=f.n_fft, n_mels=f.n_mels,
        fmin=0.0, fmax=f.sample_rate / 2.0, norm="slaney", htk=False,
    ).astype(np.float32)
    mine = mel_filterbank(
        sample_rate=f.sample_rate, n_fft=f.n_fft, n_mels=f.n_mels,
        fmin=0.0, fmax=f.sample_rate / 2.0,
    )
    check("mel filterbank is bit-identical", np.array_equal(mine, theirs),
          f"shape {mine.shape}, max|diff| {np.abs(mine - theirs).max():.1e}")

    if not CLIPS:
        print("SKIP: no cached audio under tests/wav/.tts_cache to compare on")
        return 1 if _failures else 0

    # 2. The transform, over enough real clips to catch a length/padding edge
    #    case rather than one lucky alignment.
    clips = CLIPS[:25]
    worst_stft = 0.0
    exact_stft = 0
    for path in clips:
        audio, _ = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        t = librosa.stft(
            audio, n_fft=f.n_fft, hop_length=f.hop_length, win_length=f.win_length,
            window=f._window, center=True, pad_mode="reflect",  # noqa: SLF001
        )
        m = stft(
            audio, n_fft=f.n_fft, hop_length=f.hop_length,
            win_length=f.win_length, window=f._window,  # noqa: SLF001
        )
        if m.shape != t.shape:
            check(f"stft shape matches for {Path(path).name}", False,
                  f"{m.shape} vs {t.shape}")
            break
        exact_stft += int(np.array_equal(m, t))
        worst_stft = max(worst_stft, float(np.abs(m - t).max()))
    check("stft is bit-identical on every clip", exact_stft == len(clips),
          f"{exact_stft}/{len(clips)} exact, worst max|diff| {worst_stft:.1e}")

    # 3. The whole featurizer, and the log-probs the decoder actually consumes.
    #    Re-deriving the old featurizer here (rather than trusting 1+2 to
    #    compose) is what makes this a test of the front end and not of _dsp.
    def librosa_features(audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        x = np.concatenate(([x[0]], x[1:] - f.preemph * x[:-1])).astype(np.float32)
        spec = librosa.stft(
            x, n_fft=f.n_fft, hop_length=f.hop_length, win_length=f.win_length,
            window=f._window, center=True, pad_mode="reflect",  # noqa: SLF001
        )
        mel = theirs @ (np.abs(spec) ** f.mag_power)
        feats = np.log(mel + f.log_zero_guard)
        mean = feats.mean(axis=1, keepdims=True)
        std = feats.std(axis=1, ddof=1, keepdims=True)
        return ((feats - mean) / (std + f.norm_eps)).astype(np.float32)

    worst_feat = 0.0
    exact_feat = 0
    for path in clips:
        audio, _ = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        a, b = f(audio), librosa_features(audio)
        exact_feat += int(np.array_equal(a, b))
        worst_feat = max(worst_feat, float(np.abs(a - b).max()))
    check("featurizer output is bit-identical", exact_feat == len(clips),
          f"{exact_feat}/{len(clips)} exact, worst max|diff| {worst_feat:.1e}")

    if MODEL.is_dir():
        rec = load_recognizer("nemo", MODEL, language="de")
        model = rec.model
        sample = clips[:8]
        worst_lp = 0.0
        exact_lp = 0
        for path in sample:
            audio, _ = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            ours = model.log_probs(audio)
            # Swap only the featurizer, so the ONNX graph, its input names and
            # the length vector are identical and the sole variable is the front
            # end. Restored in a finally, or every later check would be measuring
            # librosa against itself.
            original = model._featurizer  # noqa: SLF001
            try:
                model._featurizer = librosa_features  # noqa: SLF001
                theirs_lp = model.log_probs(audio)
            finally:
                model._featurizer = original  # noqa: SLF001
            exact_lp += int(np.array_equal(ours, theirs_lp))
            worst_lp = max(worst_lp, float(np.abs(ours - theirs_lp).max()))
        check("acoustic log-probs are bit-identical", exact_lp == len(sample),
              f"{exact_lp}/{len(sample)} exact, worst max|diff| {worst_lp:.1e}")
    else:
        print(f"skip log-prob check: no model at {MODEL}")

    # 4. Resampling: closeness only (see the module docstring).
    rng = np.random.default_rng(0)
    worst_rs = 0.0
    for orig in (8000, 22050, 44100, 48000):
        x = rng.standard_normal(orig).astype(np.float32)
        a = resample(x, orig, 16000)
        b = librosa.resample(x, orig_sr=orig, target_sr=16000)
        if a.shape != b.shape:
            check(f"resample {orig} -> 16000 length matches", False,
                  f"{a.shape} vs {b.shape}")
            continue
        worst_rs = max(worst_rs, float(np.abs(a - b).max()))
    check("resample matches librosa", worst_rs < 1e-6, f"max|diff| {worst_rs:.1e}")

    print()
    if _failures:
        print(f"FAIL ({len(_failures)}): " + "; ".join(_failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
