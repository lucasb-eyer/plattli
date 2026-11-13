"""Numeric storage codecs. Signed 128-bit values use Python ints in memory."""

import struct

import numpy as np


DTYPE_TO_NUMPY = {
    "f32": np.float32, "f64": np.float64,
    "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64,
    "u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64,
    "i128": object,
}
DTYPE_ITEMSIZE = {name: int(name[1:]) // 8 for name in DTYPE_TO_NUMPY}
INT_LIMITS = {
    name: (-(1 << (int(name[1:]) - 1)), (1 << (int(name[1:]) - 1)) - 1)
    if name[0] == "i" else (0, (1 << int(name[1:])) - 1)
    for name in DTYPE_TO_NUMPY if name[0] in "iu"
}


def _encode_numeric(values, dtype):
    """Encode a one-dimensional numeric column in its Plattli storage dtype."""
    if dtype == "i128":
        return b"".join(int(value).to_bytes(16, "little", signed=True) for value in values)
    return np.asarray(values, dtype=DTYPE_TO_NUMPY[dtype]).tobytes()


def _decode_numeric(data, dtype):
    if dtype == "i128":
        return np.fromiter((low | (high << 64) for low, high in struct.iter_unpack("<Qq", data)),
                           dtype=object, count=len(data) // 16)
    return np.frombuffer(data, dtype=DTYPE_TO_NUMPY[dtype])


def _read_numeric(path, dtype):
    if dtype == "i128":
        data = path.read_bytes()
        return _decode_numeric(data[:len(data) // 16 * 16], dtype)
    return np.fromfile(path, dtype=DTYPE_TO_NUMPY[dtype])


def _numeric_dtype(values):
    return "i128" if values.dtype == object else f"{values.dtype.kind}{values.dtype.itemsize * 8}"
