"""HDPA — Sauvola binarization for scanned historical documents."""

from .sauvola import DEFAULT_K, DEFAULT_R, DEFAULT_WINDOW, sauvola_binarize

__all__ = ["sauvola_binarize", "DEFAULT_WINDOW", "DEFAULT_K", "DEFAULT_R"]
__version__ = "1.0.0"
