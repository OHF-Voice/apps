"""The two DSP primitives the log-mel front end needs: a Slaney mel filterbank
and a short-time Fourier transform.

These reproduce ``librosa.filters.mel(..., htk=False, norm="slaney")`` and
``librosa.stft(..., center=True, pad_mode="reflect")`` exactly, for the
parameters NeMo's Citrinet preprocessor uses. They exist so the library does not
depend on librosa, which hard-depends on numba and scikit-learn and therefore on
llvmlite -- around 250 MB of install, including one 170 MB shared object, for
three function calls. ``tests/test_features_parity.py`` asserts that the two
agree bit for bit and is the gate on any change here.

Numerics matter more than they look. Both halves are float32 end to end (a
float32 window times float32 frames, transformed by an rfft that returns
complex64), which is what librosa does, so the features fed to the acoustic
model are the same values it saw during training rather than merely close ones.
The transform comes from ``scipy.fft`` because that is the backend librosa uses,
and it is what makes the parity exact: ``numpy.fft`` is correct too but differs
in the last bit (~1e-7 relative), and a front end that "almost" matches the one
the model was trained on is not worth the 40 MB scipy would save.
"""

import numpy as np
import scipy.fft

# Slaney's mel scale: linear below 1 kHz, logarithmic above. The constants are
# the ones in librosa (and in Malcolm Slaney's Auditory Toolbox before it) --
# 200/3 Hz per mel in the linear region, and a log region that reaches 6.4x the
# break frequency over 27 mel steps.
_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = float(np.log(6.4) / 27.0)


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float64)
    # np.maximum keeps log() off zero; np.where discards that branch below the
    # break frequency anyway, but evaluating it would still warn.
    return np.where(
        freq >= _MIN_LOG_HZ,
        _MIN_LOG_MEL + np.log(np.maximum(freq, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOGSTEP,
        freq / _F_SP,
    )


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    return np.where(
        mel >= _MIN_LOG_MEL,
        _MIN_LOG_HZ * np.exp(_LOGSTEP * (mel - _MIN_LOG_MEL)),
        mel * _F_SP,
    )


def mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float = None,
) -> np.ndarray:
    """``[n_mels, 1 + n_fft // 2]`` triangular mel filterbank, Slaney-normalized.

    Each row is a triangle spanning three consecutive mel-spaced centre
    frequencies, scaled by ``2 / (f[i+2] - f[i])`` so that filters have equal
    area rather than equal height -- without which the wide high-frequency bands
    would dominate the log-mel vector.
    """
    if fmax is None:
        fmax = sample_rate / 2.0

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    # n_mels + 2 edges: every filter needs a left and right neighbour.
    edges = _mel_to_hz(
        np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    )
    widths = np.diff(edges)
    # ramps[i, k] = edges[i] - fft_freqs[k]
    ramps = np.subtract.outer(edges, fft_freqs)

    # float32 from here, and the normalization applied in place afterwards, so
    # the triangles are rounded to single precision at the same point librosa
    # rounds them. Computing in float64 and casting once at the end differs in
    # the last bit, which is enough to fail an exact-parity test.
    weights = np.zeros((n_mels, fft_freqs.size), dtype=np.float32)
    for i in range(n_mels):
        rising = -ramps[i] / widths[i]
        falling = ramps[i + 2] / widths[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(rising, falling))

    weights *= (2.0 / (edges[2 : n_mels + 2] - edges[:n_mels]))[:, np.newaxis]
    return weights


def stft(
    x: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """``[1 + n_fft // 2, n_frames]`` complex64 STFT, centred, reflect-padded.

    Centred means frame ``t`` is aligned on sample ``t * hop_length`` rather
    than starting there, which is what the preprocessor's frame count assumes;
    it is implemented, as librosa does it, by reflecting ``n_fft // 2`` samples
    onto each end before framing.
    """
    if window.shape[0] != win_length:
        raise ValueError(
            f"window is {window.shape[0]} samples, expected win_length={win_length}"
        )

    # Centre the (shorter) analysis window inside the transform length, so the
    # zero padding is split evenly and the window stays symmetric about the
    # frame centre.
    left = (n_fft - win_length) // 2
    padded_window = np.pad(window, (left, n_fft - win_length - left))

    pad = n_fft // 2
    padded = np.pad(np.asarray(x, dtype=np.float32).reshape(-1), (pad, pad), mode="reflect")

    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    return scipy.fft.rfft(frames * padded_window, axis=-1).T
