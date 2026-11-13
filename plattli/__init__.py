"""Plattli writer and tools."""

from ._numeric import DTYPE_ITEMSIZE, DTYPE_TO_NUMPY, _encode_numeric as encode_numeric
from .bulk_writer import BulkWriter
from .reader import Reader, is_run, is_run_dir, resolve_run_dir
from .writer import CompactingWriter, DirectWriter

try:
    from ._version import version as __version__
except Exception:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = ("CompactingWriter", "DirectWriter", "BulkWriter", "Reader", "is_run", "is_run_dir", "resolve_run_dir",
           "DTYPE_ITEMSIZE", "DTYPE_TO_NUMPY", "encode_numeric")
