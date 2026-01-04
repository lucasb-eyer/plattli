"""Plattli writer and tools."""

from .bulk_writer import PlattliBulkWriter
from .writer import PlattliWriter

try:
    from ._version import version as __version__
except Exception:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = ("PlattliBulkWriter", "PlattliWriter")
