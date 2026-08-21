"""Backend abstraction shared by the Citrinet and Coqui acoustic models."""

import abc

import numpy as np

from ..tokenizer import Tokenizer


class AcousticModel(abc.ABC):
    """A CTC acoustic model: 16 kHz mono float32 audio -> per-frame log-probs.

    Implementations expose a :class:`~speech_to_phrase.tokenizer.Tokenizer` whose
    ``num_tokens``/``blank_id`` describe the output layout: ``log_probs`` returns
    a ``[T, num_tokens + 1]`` float32 array of natural-log probabilities, with the
    blank class at column ``tokenizer.blank_id``.
    """

    tokenizer: Tokenizer

    @abc.abstractmethod
    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        """Return ``[T, num_tokens + 1]`` natural-log probabilities."""
        raise NotImplementedError
