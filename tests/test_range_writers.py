import unittest

import numpy as np

from plattli._indices import _piecewise_segments, _segments_truncate, _segments_to_array


class TestRangeWriters(unittest.TestCase):
    def test_truncate_segment_overhang(self):
        steps = np.asarray([3, 40, 45, 50, 55, 60, 65, 70], dtype=np.uint32)
        segments = _piecewise_segments(steps)
        for cutoff in range(80):
            with self.subTest(cutoff=cutoff):
                kept, count = _segments_truncate(segments, cutoff)
                np.testing.assert_array_equal(_segments_to_array(kept), steps[steps < cutoff])
                self.assertEqual(count, sum(steps < cutoff))
