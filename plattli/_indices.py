import numpy as np


def _segments_from_spec(indices_spec):
    if isinstance(indices_spec, dict):
        return [indices_spec]
    if isinstance(indices_spec, list):
        return indices_spec
    raise RuntimeError(f"invalid indices spec: {indices_spec}")


def _segment_values(segment):
    if not isinstance(segment, dict):
        raise ValueError(f"invalid segment: {segment}")
    if "start" not in segment or "stop" not in segment or "step" not in segment:
        raise ValueError(f"segment missing keys: {segment}")
    start = int(segment["start"])
    stop = int(segment["stop"])
    step = int(segment["step"])
    if start < 0 or stop < 0:
        raise ValueError(f"segment out of range: {segment}")
    if step <= 0 or stop <= start:
        raise ValueError(f"invalid segment range: {segment}")
    return start, stop, step


def _segments_count_and_last(segments):
    if not segments:
        return 0, None
    total = 0
    last = None
    for segment in segments:
        start, stop, step = _segment_values(segment)
        count = (stop - start + step - 1) // step
        last_val = start + (count - 1) * step
        if last is not None and start <= last:
            raise ValueError(f"segments out of order: {segment}")
        total += count
        last = last_val
    return int(total), int(last) if last is not None else None


def _segments_length(segments):
    total, _ = _segments_count_and_last(segments)
    return total


def _segments_to_array(segments):
    if not segments:
        return np.asarray([], dtype=np.uint32)
    arrays = []
    last = None
    for segment in segments:
        start, stop, step = _segment_values(segment)
        if last is not None and start <= last:
            raise ValueError(f"segments out of order: {segment}")
        arrays.append(np.arange(start, stop, step, dtype=np.uint32))
        count = (stop - start + step - 1) // step
        last = start + (count - 1) * step
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays)


def _segments_truncate(segments, step):
    step = int(step)
    if step < 0:
        raise ValueError(f"step out of range: {step}")
    if not segments:
        return [], 0
    kept = []
    total = 0
    last = None
    for segment in segments:
        start, stop, stride = _segment_values(segment)
        if last is not None and start <= last:
            raise ValueError(f"segments out of order: {segment}")
        count = (stop - start + stride - 1) // stride
        last_val = start + (count - 1) * stride
        if step <= start:
            break
        if step >= stop:
            kept.append({"start": start, "stop": stop, "step": stride})
            total += count
            last = last_val
            continue
        keep = (step - start + stride - 1) // stride
        if keep > 0:
            trunc_stop = start + keep * stride
            kept.append({"start": start, "stop": trunc_stop, "step": stride})
            total += keep
            last = start + (keep - 1) * stride
        break
    return kept, int(total)


def _segments_too_many(segments):
    total, _ = _segments_count_and_last(segments)
    max_segments = max(4, total // 10)
    return len(segments) > max_segments


def _piecewise_segments(indices):
    if indices.size == 0:
        return []
    if indices.size == 1:
        val = int(indices[0])
        return [{"start": val, "stop": val + 1, "step": 1}]
    values = indices.astype(np.int64, copy=False)
    diffs = np.diff(values)
    if (diffs <= 0).any():
        return None
    segments = []
    size = values.size
    i = 0
    while i < size - 1:
        step = int(values[i + 1] - values[i])
        j = i + 1
        while j + 1 < size and int(values[j + 1] - values[j]) == step:
            j += 1
        start = int(values[i])
        stop = int(values[j]) + step
        segments.append({"start": start, "stop": stop, "step": step})
        i = j + 1
    if i == size - 1:
        val = int(values[i])
        segments.append({"start": val, "stop": val + 1, "step": 1})
    return segments


def _find_piecewise_params(indices):
    if indices.size < 2:
        return None
    segments = _piecewise_segments(indices)
    if not segments:
        return None
    max_segments = max(4, indices.size // 10)
    if len(segments) > max_segments:
        return None
    return segments


def _append_step_to_segments(segments, step):
    if not segments:
        segments.append({"start": int(step), "stop": int(step) + 1, "step": 1})
        return segments
    last = segments[-1]
    start, stop, stride = _segment_values(last)
    count = (stop - start + stride - 1) // stride
    last_val = start + (count - 1) * stride
    step = int(step)
    if step <= last_val:
        raise ValueError(f"step out of order: {step} after {last_val}")
    if count == 1:
        if step == start + stride:
            last["stop"] = step + stride
            return segments
        stride = step - start
        last["step"] = stride
        last["stop"] = step + stride
        return segments
    if step == last_val + stride:
        last["stop"] = step + stride
        return segments
    segments.append({"start": step, "stop": step + 1, "step": 1})
    return segments


def _append_step_to_indices_spec(indices_spec, step):
    segments = _segments_from_spec(indices_spec)
    return _append_step_to_segments(segments, step)


def _append_steps_to_indices_spec(indices_spec, steps):
    for step in steps:
        indices_spec = _append_step_to_indices_spec(indices_spec, step)
    return indices_spec
