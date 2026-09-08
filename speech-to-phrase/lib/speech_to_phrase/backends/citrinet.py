"""Compatibility import for the former Citrinet-specific backend name."""

from .nemo import NemoCtcModel

CitrinetModel = NemoCtcModel

__all__ = ["CitrinetModel", "NemoCtcModel"]
