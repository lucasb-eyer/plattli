import unittest
from unittest import mock

import numpy as np

from plattli import _values
from plattli._sequences import _arithmetic_array
from plattli._values import _extend_value_sequence, _find_value_sequence, _sequence_array, _value_sequence


class TestValueSequences(unittest.TestCase):
    def test_integer_limits_and_steps_wider_than_the_storage_dtype(self):
        for dtype, values in (("i8", [-128, 127]), ("i64", [-(2**63), 2**63 - 1]),
                              ("u64", [2**64 - 1, 0]), ("i128", [10**38, 0, -(10**38)]), ("i128", [-1, 2**63 + 1])):
            with self.subTest(dtype=dtype):
                sequence = _find_value_sequence(values, dtype)
                self.assertIsNotNone(sequence)
                self.assertEqual(_sequence_array({"dtype": dtype, "values": sequence}).tolist(), values)
                prefix = _find_value_sequence(values[:1], dtype)
                self.assertEqual(_extend_value_sequence(prefix, values[1:], dtype), sequence)

    def test_empty_sequences_and_batch_extension(self):
        for dtype in ("i128", "i8", "u64", "f32", "f64"):
            with self.subTest(dtype=dtype):
                empty = {"start": 0, "step": 0, "count": 0}
                self.assertEqual(_find_value_sequence([], dtype), empty)
                self.assertEqual(_extend_value_sequence(empty, [], dtype), empty)
                self.assertEqual(_extend_value_sequence(empty, [7, 7, 7], dtype), {"start": 7, "step": 0, "count": 3})
        prefix = {"start": -10, "step": 3, "count": 10}
        with mock.patch.object(_values, "_arithmetic_array", wraps=_arithmetic_array) as expand:
            result = _extend_value_sequence(prefix, np.arange(20, 300020, 3, dtype=np.int32), "i32")
        self.assertEqual(result, {"start": -10, "step": 3, "count": 100010})
        self.assertEqual(expand.call_count, 1)
        self.assertEqual(prefix, {"start": -10, "step": 3, "count": 10})

    def test_float_constants_preserve_bits_in_full_and_sliced_reads(self):
        for dtype in ("f32", "f64"):
            for values in ([-0.0] * 4, [0.0] * 4, [3.5] * 4, [0.1] * 4):
                with self.subTest(dtype=dtype, values=values):
                    target = np.dtype(dtype[0] + str(int(dtype[1:]) // 8))
                    actual = np.asarray(values, dtype=target)
                    sequence = _find_value_sequence(actual, dtype)
                    self.assertIsNotNone(sequence)
                    spec = {"dtype": dtype, "values": sequence}
                    self.assertEqual(_sequence_array(spec).tobytes(), actual.tobytes())
                    self.assertEqual(_sequence_array(spec, 1, 2).tobytes(), actual[1:3].tobytes())
                    prefix = _find_value_sequence(actual[:1], dtype)
                    self.assertEqual(_extend_value_sequence(prefix, actual[1:], dtype), sequence)
            for values in ([3.5, 3.25, 3.0, 2.75], [-0.0, 0.5, 1.0],
                           [0.0, -0.0], [-0.0, -0.0, 0.0]):
                with self.subTest(dtype=dtype, values=values):
                    self.assertIsNone(_find_value_sequence(values, dtype))
                    prefix = _find_value_sequence(values[:1], dtype)
                    self.assertIsNone(_extend_value_sequence(prefix, values[1:], dtype))
        self.assertIsNone(_find_value_sequence([1.0, 1.1], "f32"))
        self.assertIsNone(_find_value_sequence([2**53 + 1], "f64"))

    def test_float_arithmetic_specs_are_rejected(self):
        for dtype in ("f32", "f64"):
            for count in (0, 1, 2):
                with self.subTest(dtype=dtype, count=count), self.assertRaises(ValueError):
                    _value_sequence({"dtype": dtype, "values": {"start": 0.5, "step": 0.25, "count": count}})

    def test_invalid_input_never_wraps_or_rounds_into_a_sequence(self):
        for dtype, values in (("i8", [128]), ("u64", [-1]), ("i64", [2**63]),
                              ("i128", [1.5]), ("i128", [True]), ("i128", [[1]]),
                              ("f32", [float("nan")]), ("f64", [float("inf")]),
                              ("f32", [10**1000]), ("f64", ["1"])):
            with self.subTest(dtype=dtype, values=values):
                self.assertIsNone(_find_value_sequence(values, dtype))
                self.assertIsNone(_extend_value_sequence({"start": 0, "step": 0, "count": 0}, values, dtype))
        self.assertIsNone(_extend_value_sequence({"start": 0, "step": 1, "count": 2}, [3], "i64"))

    def test_counts_slices_and_finite_endpoints_are_validated(self):
        with self.assertRaisesRegex(ValueError, "invalid value sequence"):
            _value_sequence({"dtype": "i64", "values": None})
        valid = {"dtype": "u32", "values": {"start": 0, "step": 1, "count": 2**32}}
        self.assertEqual(_sequence_array(valid, 2**32 - 2, 2).tolist(), [2**32 - 2, 2**32 - 1])
        self.assertIsNone(_extend_value_sequence(valid["values"], [2**32], "u32"))
        for count in (-1, True, 1.5, 2**32 + 1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                _value_sequence({"dtype": "i128", "values": {"start": 0, "step": 0, "count": count}})
        for offset, count in ((1.5, 1), (False, 1), (0, True)):
            with self.subTest(offset=offset, count=count), self.assertRaises(TypeError):
                _sequence_array(valid, offset, count)
        for offset, count in ((-1, 1), (0, -1), (2**32, 1)):
            with self.subTest(offset=offset, count=count), self.assertRaises(ValueError):
                _sequence_array(valid, offset, count)
        for dtype, start, step in (("i8", 127, 1), ("f32", 1e100, 0),
                                   ("f64", 1e308, 1e308), ("f64", 10**1000, 0),
                                   ("i128", True, 1), ("i128", 1, 0.5)):
            with self.subTest(dtype=dtype, start=start, step=step), self.assertRaises(ValueError):
                _value_sequence({"dtype": dtype, "values": {"start": start, "step": step, "count": 2}})
