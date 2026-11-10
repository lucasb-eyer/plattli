import operator
import struct

import numpy as np

from ._indices import _segments_from_spec, _segments_with_counts


RANGE_DTYPE = np.dtype([("start", "<u4"), ("step", "<u4"), ("end_row", "<u8")])


def _ranges_parse(data):
    """Decode complete records and an optional open head, ignoring a partial append."""
    records = np.frombuffer(data, dtype=RANGE_DTYPE, count=len(data) // 16)
    last = None
    if len(records):
        ends = records["end_row"]
        if ends[0] == 0 or ends[-1] > 2**32 or np.any(ends[1:] <= ends[:-1]):
            raise ValueError("range end_row values must increase within uint32 row capacity")
        if np.any(records["step"] == 0):
            raise ValueError("range step must be positive")
        counts = np.empty(len(records), dtype=np.uint64)
        counts[0] = ends[0]
        counts[1:] = ends[1:] - ends[:-1]
        last = records["start"].astype(np.uint64) + (counts - 1) * records["step"]
        if np.any(last > 0xFFFFFFFF):
            raise ValueError("range step values exceed uint32")
        if np.any(records["start"][1:] <= last[:-1]):
            raise ValueError("ranges are out of order")

    tail = None
    if len(data) % 16 >= 8:
        tail = struct.unpack_from("<II", data, len(records) * 16)
        if tail[1] == 0:
            raise ValueError("range step must be positive")
        if last is not None and tail[0] <= last[-1]:
            raise ValueError("open range is out of order")
        if len(records) and records[-1]["end_row"] == 2**32:
            raise ValueError("open range exceeds uint32 row capacity")
    return records, tail


def _ranges_records(data, count):
    """Resolve an open head against a value count captured before reading data."""
    count = operator.index(count)
    if not 0 <= count <= 2**32:
        raise ValueError(f"range row count out of bounds: {count}")
    records, tail = _ranges_parse(data)
    closed = int(records[-1]["end_row"]) if len(records) else 0
    if tail is None or count <= closed:
        return records
    start, step = tail
    if start + (count - closed - 1) * step > 0xFFFFFFFF:
        raise ValueError("open range step values exceed uint32")
    result = np.empty(len(records) + 1, dtype=RANGE_DTYPE)
    result[:-1] = records
    result[-1] = (start, step, count)
    return result


def _ranges_encode(segments, total_count=None, open_tail=False):
    """Encode inline segments, optionally clipping them to the available values."""
    if total_count is not None:
        total_count = operator.index(total_count)
        if not 0 <= total_count <= 2**32:
            raise ValueError(f"range row count out of bounds: {total_count}")
    records = []
    rows = 0
    for start, count, step, _ in _segments_with_counts(_segments_from_spec(segments), total_count):
        if total_count is not None:
            count = min(count, max(total_count - rows, 0))
        if count:
            if start > 0xFFFFFFFF or not 0 < step <= 0xFFFFFFFF or start + (count - 1) * step > 0xFFFFFFFF:
                raise ValueError("range step values exceed uint32")
            rows += count
            if rows > 2**32:
                raise ValueError("range row count exceeds uint32 capacity")
            records.append((start, step, rows))
    data = np.asarray(records, dtype=RANGE_DTYPE).tobytes()
    return data[:-8] if open_tail and records else data


def _ranges_to_segments(data, total_count):
    """Convert the available prefix to finite inline segments without expanding steps."""
    if total_count is None:
        records, tail = _ranges_parse(data)
        if tail is not None:
            raise ValueError("open range requires a value count")
        total_count = int(records[-1]["end_row"]) if len(records) else 0
    else:
        total_count = operator.index(total_count)
        records = _ranges_records(data, total_count)
    result = []
    offset = 0
    for record in records:
        end = int(record["end_row"])
        count = min(end, total_count) - offset
        if count > 0:
            start = int(record["start"])
            step = int(record["step"])
            result.append({"start": start, "stop": start + count * step, "step": step})
        if end >= total_count:
            break
        offset = end
    return result


def _ranges_append(tail, count, steps):
    """Build an append; tail is (start, step, first_row), with an immutable stride."""
    count = operator.index(count)
    if not 0 <= count <= 2**32:
        raise ValueError(f"range row count out of bounds: {count}")
    steps = np.asarray(steps)
    if steps.ndim != 1:
        raise ValueError("range steps must be one-dimensional")
    if len(steps):
        if steps.dtype.kind not in "iu":
            raise TypeError("range steps must be integers")
        if np.any(steps < 0) or np.any(steps > 0xFFFFFFFF):
            raise ValueError("range steps exceed uint32")
        if np.any(steps[1:] <= steps[:-1]):
            raise ValueError("range steps are out of order")
        if count + len(steps) > 2**32:
            raise ValueError("range row count exceeds uint32 capacity")
    if tail is not None:
        start, step, first_row = map(operator.index, tail)
        if not 0 <= start <= 0xFFFFFFFF or not 0 < step <= 0xFFFFFFFF or not 0 <= first_row <= count:
            raise ValueError(f"invalid open range state: {tail}")
        tail = (start, step, first_row)
        if count > first_row:
            last = start + (count - first_row - 1) * step
            if last > 0xFFFFFFFF:
                raise ValueError("open range step values exceed uint32")
            if len(steps) and int(steps[0]) <= last:
                raise ValueError("range steps are out of order")
        elif len(steps) and int(steps[0]) != start:
            raise ValueError("first step does not match the empty open range")

    payload = bytearray()
    for idx, value in enumerate(steps):
        value = int(value)
        if tail is not None:
            start, step, first_row = tail
            if value == start + (count - first_row) * step:
                count += 1
                continue
            payload.extend(struct.pack("<Q", count))
        step = int(steps[idx + 1]) - value if idx + 1 < len(steps) else 1
        tail = (value, step, count)
        payload.extend(struct.pack("<II", value, step))
        count += 1
    return bytes(payload), tail
