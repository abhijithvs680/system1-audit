"""Adapters that let a real or synthetic model answer audit questions."""

from .mock import SyntheticDecider

__all__ = ["SyntheticDecider"]
