"""Plattli writer and tools."""

from .writer import PlattliWriter

try:
    from ._version import version as __version__
except Exception:
    __version__ = "0+unknown"

__all__ = ("PlattliWriter",)
