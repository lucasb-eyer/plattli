import numpy as np


def _segments_from_spec(indices_spec):
    if isinstance(indices_spec, dict):
        return [indices_spec]
    if isinstance(indices_spec, list):
        return indices_spec
    raise RuntimeError(f"invalid indices spec: {indices_spec}")


def _segment_start_step(segment):
    if not isinstance(segment, dict):
        raise ValueError(f"invalid segment: {segment}")
    if "start" not in segment or "step" not in segment:
        raise ValueError(f"segment missing keys: {segment}")
    start = int(segment["start"])
    step = int(segment["step"])
    if start < 0:
        raise ValueError(f"segment out of range: {segment}")
    if step <= 0:
        raise ValueError(f"invalid segment range: {segment}")
    return start, step


def _segment_values(segment):
    start, step = _segment_start_step(segment)
    if "stop" not in segment:
        raise ValueError(f"segment missing stop: {segment}")
    stop = int(segment["stop"])
    if stop < 0:
        raise ValueError(f"segment out of range: {segment}")
    if stop <= start:
        raise ValueError(f"invalid segment range: {segment}")
    return start, stop, step


def _segments_have_open_tail(segments):
    for idx, segment in enumerate(segments):
        if "stop" not in segment:
            if idx != len(segments) - 1:
                raise ValueError(f"only the final segment may omit stop: {segment}")
            return True
    return False


def _segments_with_counts(segments, total_count=None):
    total = 0
    last = None
    total_count = None if total_count is None else int(total_count)
    for idx, segment in enumerate(segments):
        start, step = _segment_start_step(segment)
        if last is not None and start <= last:
            raise ValueError(f"segments out of order: {segment}")
        if "stop" in segment:
            stop = int(segment["stop"])
            if stop < 0:
                raise ValueError(f"segment out of range: {segment}")
            if stop <= start:
                raise ValueError(f"invalid segment range: {segment}")
            count = (stop - start + step - 1) // step
            is_open = False
        else:
            if idx != len(segments) - 1:
                raise ValueError(f"only the final segment may omit stop: {segment}")
            if total_count is None:
                raise ValueError(f"open final segment requires total count: {segment}")
            if total_count < total:
                raise ValueError(f"total count {total_count} shorter than closed segments ({total})")
            count = total_count - total
            is_open = True
        if count < 0:
            raise ValueError(f"invalid segment count: {segment}")
        if count:
            last = start + (count - 1) * step
        yield start, count, step, is_open
        total += count


def _segments_count_and_last(segments, total_count=None):
    if not segments:
        return 0, None
    total = 0
    last = None
    for start, count, step, _ in _segments_with_counts(segments, total_count):
        if count:
            last = start + (count - 1) * step
        total += count
    return int(total), int(last) if last is not None else None


def _segments_length(segments):
    total, _ = _segments_count_and_last(segments)
    return total


def _segments_to_array(segments, total_count=None):
    if not segments:
        return np.asarray([], dtype=np.uint32)
    arrays = []
    for start, count, step, _ in _segments_with_counts(segments, total_count):
        if count:
            arrays.append(start + np.arange(count, dtype=np.uint32) * np.uint32(step))
    if not arrays:
        return np.asarray([], dtype=np.uint32)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays)


def _segments_close_open_tail(segments, total_count):
    closed = []
    for start, count, step, _ in _segments_with_counts(segments, total_count):
        if count:
            closed.append({"start": start, "stop": start + count * step, "step": step})
    return closed


def _segments_truncate(segments, step, total_count=None):
    step = int(step)
    if step < 0:
        raise ValueError(f"step out of range: {step}")
    if not segments:
        return [], 0
    kept = []
    total = 0
    for start, count, stride, is_open in _segments_with_counts(segments, total_count):
        if not count:
            continue
        stop = start + count * stride
        if step <= start:
            break
        if step >= stop:
            if is_open:
                kept.append({"start": start, "step": stride})
            else:
                kept.append({"start": start, "stop": stop, "step": stride})
            total += count
            continue
        keep = (step - start + stride - 1) // stride
        if keep > 0:
            if is_open:
                kept.append({"start": start, "step": stride})
            else:
                kept.append({"start": start, "stop": start + keep * stride, "step": stride})
            total += keep
        break
    return kept, int(total)


def _segments_too_many(segments, total_count=None):
    total, _ = _segments_count_and_last(segments, total_count)
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


def _append_step_to_segments(segments, step, count=None, open_tail=False):
    step = int(step)
    if not segments:
        if open_tail:
            segments.append({"start": step, "step": 1})
        else:
            segments.append({"start": step, "stop": step + 1, "step": 1})
        return segments

    total, last_val = _segments_count_and_last(segments, count)
    last = segments[-1]
    start, stride = _segment_start_step(last)
    last_open = "stop" not in last
    if last_open:
        closed_total, _ = _segments_count_and_last(segments[:-1]) if len(segments) > 1 else (0, None)
        last_count = total - closed_total
    else:
        stop = int(last["stop"])
        last_count = (stop - start + stride - 1) // stride

    if last_count == 0:
        if step != start:
            raise ValueError(f"step out of order: {step} for empty open segment starting at {start}")
        return segments
    if step <= last_val:
        raise ValueError(f"step out of order: {step} after {last_val}")

    if last_count == 1:
        if step == start + stride:
            if not last_open:
                last["stop"] = step + stride
            return segments
        stride = step - start
        last["step"] = stride
        if not last_open:
            last["stop"] = step + stride
        return segments
    if step == last_val + stride:
        if not last_open:
            last["stop"] = step + stride
        return segments

    if last_open:
        last["stop"] = start + last_count * stride
    if open_tail:
        segments.append({"start": step, "step": 1})
    else:
        segments.append({"start": step, "stop": step + 1, "step": 1})
    return segments


def _append_step_to_indices_spec(indices_spec, step, count=None, open_tail=False):
    segments = _segments_from_spec(indices_spec)
    return _append_step_to_segments(segments, step, count=count, open_tail=open_tail)


def _append_steps_to_indices_spec(indices_spec, steps, count=None, open_tail=False):
    for step in steps:
        indices_spec = _append_step_to_indices_spec(indices_spec, step, count=count, open_tail=open_tail)
        if count is not None:
            count += 1
    return indices_spec
