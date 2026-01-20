"""Plattli writer and tools."""

from .bulk_writer import PlattliBulkWriter
from .reader import Reader, is_run, is_run_dir, resolve_run_dir
from .writer import CompactingWriter, DirectWriter

try:
    from ._version import version as __version__
except Exception:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = ("CompactingWriter", "DirectWriter", "PlattliBulkWriter", "Reader", "is_run", "is_run_dir", "resolve_run_dir")
