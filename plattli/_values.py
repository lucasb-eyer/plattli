"""Inline constant and arithmetic values, indexed by physical row or logging step."""

import operator

import numpy as np

from ._numeric import DTYPE_TO_NUMPY, INT_LIMITS
from ._sequences import _arithmetic_array
from ._affine import _extend_step_sequence, _step_sequence_array


def _sequence_dtype(dtype):
    if dtype not in DTYPE_TO_NUMPY:
        raise ValueError(f'invalid value sequence dtype: {dtype}')
    return np.dtype(DTYPE_TO_NUMPY[dtype])


def _value_sequence(spec):
    if 'values' not in spec:
        return None
    sequence = spec['values']
    if not isinstance(sequence, dict) or set(sequence) not in ({'start', 'step', 'count'}, {'start', 'step', 'count', 'on'}):
        raise ValueError(f'invalid value sequence: {sequence}')
    if 'on' in sequence and sequence['on'] != 'step':
        raise ValueError(f'invalid value sequence basis: {sequence["on"]}')
    start, step, count = sequence['start'], sequence['step'], sequence['count']
    if type(count) is not int or not 0 <= count <= 2**32:
        raise ValueError(f'invalid value sequence count: {count}')
    dtype = _sequence_dtype(spec['dtype'])
    if dtype.kind in 'iuO':
        if type(start) is not int or type(step) is not int:
            raise ValueError('integer value sequences require integer start and step')
        if sequence.get('on') != 'step':
            lower, upper = INT_LIMITS[spec['dtype']]
            last = start + max(0, count - 1) * step
            if min(start, last) < lower or max(start, last) > upper:
                raise ValueError(f'value sequence exceeds {spec["dtype"]}')
    else:
        if 'on' in sequence:
            raise ValueError('step-based sequences require integer values')
        if type(step) not in (int, float) or step != 0:
            raise ValueError('float value sequences must be constant')
        if type(start) not in (int, float):
            raise ValueError(f'invalid float constant: {start}')
        try:
            with np.errstate(over='ignore', invalid='ignore'):
                stored = dtype.type(start)
        except OverflowError as exc:
            raise ValueError(f'value sequence exceeds {spec["dtype"]}') from exc
        if not np.isfinite(stored) or stored.item() != start:
            raise ValueError(f'value sequence exceeds {spec["dtype"]}')
    return sequence


def _sequence_array(spec, offset=0, count=None, indices=None):
    sequence = _value_sequence(spec)
    if sequence is None:
        raise ValueError('missing value sequence')
    if isinstance(offset, (bool, np.bool_)) or isinstance(count, (bool, np.bool_)):
        raise TypeError('value sequence slice requires integer offset and count')
    offset = operator.index(offset)
    count = sequence['count'] - offset if count is None else operator.index(count)
    if offset < 0 or count < 0 or offset + count > sequence['count']:
        raise ValueError('value sequence slice out of bounds')
    if sequence.get('on') == 'step':
        if indices is None or len(indices) != count:
            raise ValueError('step-based values require the selected logging steps')
        return _step_sequence_array(sequence, indices, spec['dtype'])
    return _arithmetic_array(sequence['start'], sequence['step'], offset, count, _sequence_dtype(spec['dtype']))


def _sequence_value(spec, row, indices=None):
    return _sequence_array(spec, row, 1, indices=indices)[0]


def _sequence_values(values, dtype):
    """Validate before casting: a compact encoding must never round its input."""
    try:
        # A signed/unsigned Python list can otherwise become float64 before we
        # inspect it, losing integer bits even when the destination is exact.
        source = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
        target = _sequence_dtype(dtype)
        if source.ndim != 1:
            return None
        if not source.size:
            return np.empty(0, dtype=target)
        if source.dtype.kind not in 'iufO':
            return None
        if target.kind in 'iuO':
            if source.dtype.kind not in 'iuO':
                return None
            if source.dtype.kind == 'O' and any(not isinstance(value, (int, np.integer))
                                               or isinstance(value, (bool, np.bool_)) for value in source):
                return None
            if source.size:
                lower, upper = INT_LIMITS[dtype]
                if int(source.min()) < lower or int(source.max()) > upper:
                    return None
        else:
            if source.dtype.kind == 'O' and any(not isinstance(value, (int, float, np.integer, np.floating))
                                               or isinstance(value, (bool, np.bool_)) for value in source):
                return None
        with np.errstate(over='ignore', invalid='ignore'):
            actual = source.astype(target, copy=False)
        if target.kind == 'f':
            if not np.isfinite(actual).all():
                return None
            # NumPy mixed integer/float comparisons can round large integers first.
            # Object comparison keeps Python's exact integer-versus-float semantics.
            if source.dtype.kind in 'iuO':
                if not np.array_equal(source.astype(object), actual.astype(object)):
                    return None
            elif not np.array_equal(source, actual):
                return None
        return actual
    except (TypeError, ValueError, OverflowError):
        return None


def _float_constant_matches(values, start):
    bits = values.view(f'u{values.dtype.itemsize}')
    return bool(np.all(bits == np.asarray(start, dtype=values.dtype).view(bits.dtype)))


def _extend_value_sequence(sequence, values, dtype, indices=None):
    """Return a new sequence, or None when the values break its exact pattern."""
    try:
        _value_sequence({'dtype': dtype, 'values': sequence})
        if sequence['count'] == 0:
            return _find_value_sequence(values, dtype)
        if sequence.get('on') == 'step':
            return _extend_step_sequence(sequence, indices, values, dtype) if indices is not None else None
        actual = _sequence_values(values, dtype)
        if actual is None or sequence['count'] + len(actual) > 2**32:
            return None
        if not len(actual):
            return dict(sequence)
        result = {**sequence, 'count': sequence['count'] + len(actual)}
        if actual.dtype.kind == 'f':
            return result if _float_constant_matches(actual, sequence['start']) else None
        if sequence['count'] == 1:
            result['step'] = (actual[0].item() if isinstance(actual[0], np.generic) else actual[0]) - sequence['start']
        expected = _sequence_array({'dtype': dtype, 'values': result}, sequence['count'], len(actual))
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.array_equal(expected, actual) else None


def _find_value_sequence(values, dtype):
    actual = _sequence_values(values, dtype)
    if actual is None or len(actual) > 2**32:
        return None
    if not len(actual):
        return {'start': 0, 'step': 0, 'count': 0}
    start = actual[0].item() if isinstance(actual[0], np.generic) else actual[0]
    if actual.dtype.kind == 'f':
        return {'start': start, 'step': 0.0, 'count': len(actual)} if _float_constant_matches(actual, start) else None
    step = (actual[1].item() if isinstance(actual[1], np.generic) else actual[1]) - start if len(actual) > 1 else 0
    sequence = {'start': start, 'step': step, 'count': len(actual)}
    try:
        expected = _sequence_array({'dtype': dtype, 'values': sequence})
    except (TypeError, ValueError, OverflowError):
        return None
    return sequence if np.array_equal(expected, actual) else None
