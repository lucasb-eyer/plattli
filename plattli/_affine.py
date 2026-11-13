"""Exact integer sequences affine in the metric's logging steps."""

import numpy as np

from ._numeric import DTYPE_TO_NUMPY, INT_LIMITS


def _step_sequence_values(indices, values, dtype):
    if dtype not in INT_LIMITS:
        return None
    indices = np.asarray(indices)
    values = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    if not indices.size:
        indices = np.empty(0, dtype=np.uint32)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or values.ndim != 1 or len(indices) != len(values):
        return None
    if len(indices) and (np.any(indices > 0xFFFFFFFF) or np.any(indices < 0) or np.any(indices[1:] <= indices[:-1])):
        return None
    if values.dtype.kind not in 'iuO':
        return None
    if values.dtype.kind == 'O' and any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in values):
        return None
    if len(values):
        lower, upper = INT_LIMITS[dtype]
        if int(values.min()) < lower or int(values.max()) > upper:
            return None
    return indices, values


def _find_step_sequence(indices, values, dtype):
    checked = _step_sequence_values(indices, values, dtype)
    if checked is None:
        return None
    indices, values = checked
    if len(values) < 2 or len(values) > 2**32:
        return None
    delta = int(values[1]) - int(values[0])
    distance = int(indices[1]) - int(indices[0])
    step, remainder = divmod(delta, distance)
    if remainder:
        return None
    start = int(values[0]) - int(indices[0]) * step
    if any(int(value) != start + int(index) * step for index, value in zip(indices, values)):
        return None
    return {'start': start, 'step': step, 'count': len(values), 'on': 'step'}


def _extend_step_sequence(sequence, indices, values, dtype):
    checked = _step_sequence_values(indices, values, dtype)
    if checked is None or sequence.get('on') != 'step':
        return None
    indices, values = checked
    if type(sequence.get('count')) is not int or not 0 <= sequence['count'] <= 2**32 - len(values):
        return None
    if type(sequence.get('start')) is not int or type(sequence.get('step')) is not int:
        return None
    if any(int(value) != sequence['start'] + int(index) * sequence['step'] for index, value in zip(indices, values)):
        return None
    return {**sequence, 'count': sequence['count'] + len(values)}


def _step_sequence_array(sequence, indices, dtype):
    indices = np.asarray(indices)
    if not indices.size:
        indices = np.empty(0, dtype=np.uint32)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu':
        raise ValueError('step-affine values require integer logging steps')
    if len(indices) and (np.any(indices < 0) or np.any(indices > 0xFFFFFFFF) or np.any(indices[1:] <= indices[:-1])):
        raise ValueError('logging steps are out of order or outside uint32')
    start, step = sequence['start'], sequence['step']
    if type(start) is not int or type(step) is not int:
        raise ValueError('step-affine values require integer start and step')
    if dtype not in INT_LIMITS:
        raise ValueError(f'invalid step-affine dtype: {dtype}')
    if len(indices):
        lower, upper = INT_LIMITS[dtype]
        first, last = start + int(indices[0]) * step, start + int(indices[-1]) * step
        if min(first, last) < lower or max(first, last) > upper:
            raise ValueError(f'step-affine values exceed {dtype}')
    if dtype == 'i128':
        return np.fromiter((start + int(index) * step for index in indices), dtype=object, count=len(indices))
    target = np.dtype(DTYPE_TO_NUMPY[dtype])
    unsigned = np.dtype(f'u{target.itemsize}')
    modulus = 1 << (8 * target.itemsize)
    result = indices.astype(unsigned)
    result *= unsigned.type(step % modulus)
    result += unsigned.type(start % modulus)
    return result.view(target)
