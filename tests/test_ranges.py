import struct
import unittest

import numpy as np

from plattli._indices import _segments_to_array
from plattli._ranges import (
    RANGE_DTYPE,
    _ranges_append,
    _ranges_encode,
    _ranges_parse,
    _ranges_records,
    _ranges_to_segments,
)


class TestRanges(unittest.TestCase):
    def test_roundtrip_and_clipped_prefix(self):
        segments = [
            {"start": 0, "stop": 5, "step": 2},
            {"start": 10, "stop": 17, "step": 3},
            {"start": 100, "stop": 101, "step": 1},
        ]
        for open_tail in (False, True):
            with self.subTest(open_tail=open_tail):
                data = _ranges_encode(segments, open_tail=open_tail)
                self.assertEqual(len(data), 48 - 8 * open_tail)
                closed, tail = _ranges_parse(data)
                self.assertEqual(closed["end_row"].tolist(), [3, 6] if open_tail else [3, 6, 7])
                self.assertEqual(tail, (100, 1) if open_tail else None)
                for count in range(8):
                    np.testing.assert_array_equal(
                        _segments_to_array(_ranges_to_segments(data, count)),
                        np.asarray([0, 2, 4, 10, 13, 16, 100][:count], dtype=np.uint32),
                    )
        self.assertEqual(RANGE_DTYPE.itemsize, 16)
        self.assertEqual(_ranges_encode([]), b"")
        closed, tail = _ranges_parse(b"")
        self.assertEqual(closed.tolist(), [])
        self.assertIsNone(tail)

    def test_encoding_clips_closed_and_open_specs(self):
        for segments in (
            [{"start": 0, "stop": 3, "step": 1}, {"start": 10, "stop": 16, "step": 2}],
            [{"start": 0, "stop": 3, "step": 1}, {"start": 10, "step": 2}],
        ):
            for count in (0, 1, 3, 4):
                with self.subTest(segments=segments, count=count):
                    encoded = _ranges_encode(segments, total_count=count, open_tail=True)
                    self.assertEqual(
                        _segments_to_array(_ranges_to_segments(encoded, count)).tolist(),
                        [0, 1, 2, 10][:count],
                    )

    def test_every_interrupted_append_preserves_published_rows(self):
        data = _ranges_encode([{"start": 0, "stop": 8, "step": 2}], open_tail=True)
        next_steps = [8, 10, 20, 22, 24, 40]
        payload, tail = _ranges_append((0, 2, 0), 4, next_steps)
        self.assertEqual(tail, (40, 1, 9))
        for length in range(len(payload) + 1):
            with self.subTest(index_bytes=length):
                self.assertEqual(
                    _segments_to_array(_ranges_to_segments(data + payload[:length], 4)).tolist(),
                    [0, 2, 4, 6],
                )
        for count in range(4, 11):
            with self.subTest(value_rows=count):
                self.assertEqual(
                    _segments_to_array(_ranges_to_segments(data + payload, count)).tolist(),
                    ([0, 2, 4, 6] + next_steps)[:count],
                )

    def test_appending_keeps_singleton_stride_and_reuses_open_head(self):
        data = struct.pack("<II", 5, 1)
        payload, tail = _ranges_append((5, 1, 0), 1, [9, 13])
        self.assertEqual(payload, struct.pack("<QII", 1, 9, 4))
        self.assertEqual(tail, (9, 4, 1))
        more, tail = _ranges_append(tail, 3, np.asarray([17, 21], dtype=np.uint32))
        self.assertEqual(more, b"")
        self.assertEqual(tail, (9, 4, 1))
        self.assertEqual(
            _segments_to_array(_ranges_to_segments(data + payload, 5)).tolist(),
            [5, 9, 13, 17, 21],
        )
        closed, tail = _ranges_parse(data + payload + struct.pack("<Q", 5))
        self.assertIsNone(tail)
        self.assertEqual(closed["end_row"].tolist(), [1, 5])

    def test_empty_file_and_append_after_closed_prefix(self):
        payload, tail = _ranges_append(None, 0, [1, 4, 7])
        self.assertEqual(payload, struct.pack("<II", 1, 3))
        self.assertEqual(tail, (1, 3, 0))
        data = payload + struct.pack("<Q", 3)
        payload, tail = _ranges_append(None, 3, [20])
        self.assertEqual(tail, (20, 1, 3))
        self.assertEqual(
            _segments_to_array(_ranges_to_segments(data + payload, 4)).tolist(),
            [1, 4, 7, 20],
        )

    def test_uint32_step_and_row_boundaries_without_expansion(self):
        data = _ranges_encode({"start": 0xFFFFFFFF, "stop": 2**32, "step": 1})
        self.assertEqual(_ranges_parse(data)[0].tolist(), [(0xFFFFFFFF, 1, 1)])
        self.assertEqual(_ranges_to_segments(data, 1), [{"start": 0xFFFFFFFF, "stop": 2**32, "step": 1}])
        data = _ranges_encode({"start": 0, "stop": 2**32, "step": 1})
        self.assertEqual(_ranges_parse(data)[0]["end_row"].tolist(), [2**32])
        self.assertEqual(_ranges_to_segments(data, None), [{"start": 0, "stop": 2**32, "step": 1}])
        self.assertEqual(_ranges_records(struct.pack("<II", 0, 1), 2**32)["end_row"].tolist(), [2**32])
        self.assertEqual(_ranges_records(struct.pack("<II", 0xFFFFFFFF, 1), 1)["start"].tolist(), [0xFFFFFFFF])
        with self.assertRaisesRegex(ValueError, "exceed uint32"):
            _ranges_records(struct.pack("<II", 0xFFFFFFFF, 1), 2)

    def test_closed_metadata_can_be_ahead_of_values(self):
        data = struct.pack("<IIQII", 0, 1, 50, 100, 1)
        self.assertEqual(_ranges_records(data, 25)["end_row"].tolist(), [50])
        self.assertEqual(_ranges_to_segments(data, 25), [{"start": 0, "stop": 25, "step": 1}])
        self.assertEqual(_ranges_records(data, 52)["end_row"].tolist(), [50, 52])

    def test_invalid_complete_records_fail(self):
        for records in (
            [(0, 1, 0)],
            [(0, 0, 1)],
            [(0, 1, 2**32 + 1)],
            [(0xFFFFFFFF, 1, 2)],
            [(0, 1, 3), (10, 1, 2)],
            [(0, 1, 3), (10, 1, 3)],
            [(0, 1, 3), (2, 1, 4)],
        ):
            with self.subTest(records=records), self.assertRaises(ValueError):
                _ranges_parse(np.asarray(records, dtype=RANGE_DTYPE).tobytes())
        for data in (
            struct.pack("<II", 0, 0),
            struct.pack("<IIQII", 0, 1, 3, 2, 1),
            struct.pack("<IIQII", 0, 1, 2**32, 0xFFFFFFFF, 1),
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                _ranges_parse(data)

    def test_invalid_append_state_and_points_fail(self):
        for steps in ([0], [2, 1], [-1], [2**32]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                _ranges_append((0, 1, 0), 1, steps)
        for steps in ([1.0], [True]):
            with self.subTest(steps=steps), self.assertRaises(TypeError):
                _ranges_append(None, 0, steps)
        for tail, count, steps in (
            ((0, 0, 0), 1, [1]),
            ((0, 1, 2), 1, [1]),
            ((0xFFFFFFFF, 1, 0), 2, []),
            ((10, 1, 0), 0, [11]),
        ):
            with self.subTest(tail=tail, count=count, steps=steps), self.assertRaises(ValueError):
                _ranges_append(tail, count, steps)


if __name__ == "__main__":
    unittest.main()
