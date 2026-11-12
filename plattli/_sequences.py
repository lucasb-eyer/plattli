"""Arithmetic shared by index ranges and numeric value sequences."""

import numpy as np


def _arithmetic_array(start, step, offset, count, dtype):
    dtype = np.dtype(dtype)
    if step == 0:
        return np.full(count, start, dtype=dtype)
    if dtype.kind == 'O':
        return np.fromiter((start + row * step for row in range(offset, offset + count)), dtype=object, count=count)
    if dtype.kind not in 'iu':
        raise ValueError('arithmetic sequences require an integer dtype')
    # Modular intermediates remain exact even when a signed subtraction or
    # multiplication would overflow, provided the resulting values fit.
    unsigned = np.dtype(f'u{dtype.itemsize}')
    modulus = 1 << (8 * dtype.itemsize)
    result = np.arange(count, dtype=unsigned)
    result *= unsigned.type(step % modulus)
    result += unsigned.type((start + offset * step) % modulus)
    return result.view(dtype)
