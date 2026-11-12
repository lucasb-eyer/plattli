import unittest

import numpy as np

from plattli._affine import _extend_step_sequence, _find_step_sequence, _step_sequence_array


class TestStepSequences(unittest.TestCase):
    def test_irregular_cadence_preserves_exact_integer_values(self):
        indices = [0, 100, 200, 251]
        for dtype, start, stride in (('int', 10**20, 10**6), ('int', -(10**80), -7), ('i64', 10, 3), ('u64', 2**64 - 1, -100)):
            with self.subTest(dtype=dtype, start=start, stride=stride):
                values = [start + index * stride for index in indices]
                sequence = _find_step_sequence(indices, values, dtype)
                self.assertEqual(sequence, {'start': start, 'step': stride, 'count': 4, 'on': 'step'})
                self.assertEqual(_step_sequence_array(sequence, indices, dtype).tolist(), values)
                prefix = _find_step_sequence(indices[:2], values[:2], dtype)
                self.assertEqual(_extend_step_sequence(prefix, indices[2:], values[2:], dtype), sequence)
                self.assertIsNone(_extend_step_sequence(sequence, [300], [start + 300 * stride + 1], dtype))

    def test_nonintegral_slope_and_invalid_inputs_do_not_encode(self):
        for indices, values, dtype in (([0, 100], [0, 1], 'int'), ([0, 0], [1, 1], 'int'),
                                      ([0, -1], [0, 1], 'int'), ([0, 1], [0, 256], 'u8'),
                                      ([0, 1], [True, 2], 'int'), ([0, 1], [0.0, 1.0], 'f64')):
            with self.subTest(indices=indices, values=values, dtype=dtype):
                self.assertIsNone(_find_step_sequence(indices, values, dtype))

    def test_intercepts_and_intermediates_need_not_fit_storage_dtype(self):
        for dtype, indices, values in (('i8', [100, 101], [-128, 127]),
                                      ('i64', [1, 2], [-(2**63), 2**63 - 1]),
                                      ('u64', [100, 101], [2**64 - 1, 0])):
            with self.subTest(dtype=dtype):
                sequence = _find_step_sequence(indices, values, dtype)
                self.assertEqual(_step_sequence_array(sequence, indices, dtype).tolist(), values)
        with self.assertRaises(ValueError):
            _step_sequence_array({'start': 127, 'step': 1}, [1], 'i8')


class TestStepAffineWriters(unittest.TestCase):
    def writer(self, kind, path, **kwargs):
        import plattli
        if kind == 'direct':
            return plattli.DirectWriter(path, write_threads=0, **kwargs)
        if kind == 'bulk':
            return plattli.BulkWriter(path, **kwargs)
        return plattli.CompactingWriter(path, hotsize=int(kind), **kwargs)

    def test_every_n_steps_and_extra_final_step(self):
        import tempfile
        import plattli
        indices = [0, 100, 200, 251]
        for kind in ('direct', '1', '2', '100', 'bulk'):
            for zipped in (False, True):
                with self.subTest(kind=kind, zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                    writer = self.writer(kind, tmp)
                    for row, step in enumerate(indices):
                        writer.step = step
                        writer.write(counter=10**20 + step * 10**6, rows=10**30 + row * 3)
                        writer.end_step()
                    writer.finish(optimize=False, zip=zipped)
                    with plattli.Reader(tmp) as reader:
                        self.assertEqual(reader.metric_indices('counter').tolist(), indices)
                        self.assertEqual(reader.metric_values('counter').tolist(), [10**20 + step * 10**6 for step in indices])
                        self.assertEqual(reader.manifest()['counter']['values']['on'], 'step')
                        self.assertEqual(reader.metric_values('rows').tolist(), [10**30 + row * 3 for row in range(len(indices))])
                        self.assertNotIn('on', reader.manifest()['rows']['values'])
                        self.assertEqual(reader.metric_values('counter', start=100, stop=251).tolist(),
                                         [10**20 + step * 10**6 for step in indices[1:]])
                        self.assertEqual(reader.metric_values('counter', istart=-1).tolist(), [10**20 + 251 * 10**6])

    def test_resume_step_sequence_and_restart_after_rewind_to_zero(self):
        import tempfile
        import plattli
        for kind in ('direct', '1', '100'):
            for cutoff in (0, 100, 252):
                with self.subTest(kind=kind, cutoff=cutoff), tempfile.TemporaryDirectory() as tmp:
                    writer = self.writer(kind, tmp)
                    for step in (0, 100, 200, 251):
                        writer.step = step
                        writer.write(counter=10**20 + step * 10**6)
                        writer.end_step()
                    writer.finish(optimize=False, zip=False)
                    resumed = self.writer(kind, tmp, step=cutoff)
                    resumed.step = cutoff
                    resumed.write(counter=(-10**25 if cutoff == 0 else 10**20 + cutoff * 10**6))
                    resumed.end_step()
                    resumed.finish(optimize=False, zip=False)
                    with plattli.Reader(tmp) as reader:
                        expected_steps = [step for step in (0, 100, 200, 251) if step < cutoff] + [cutoff]
                        expected = [-10**25] if cutoff == 0 else [10**20 + step * 10**6 for step in expected_steps]
                        self.assertEqual(reader.metric_indices('counter').tolist(), expected_steps)
                        self.assertEqual(reader.metric_values('counter').tolist(), expected)

    def test_off_pattern_integer_write_leaves_batch_unapplied(self):
        import copy
        import tempfile
        import plattli
        for kind in ('direct', '1', '100', 'bulk'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                writer = self.writer(kind, tmp)
                for step in (0, 100, 200, 251):
                    writer.step = step
                    writer.write(counter=10**20 + step * 10**6)
                    writer.end_step()
                writer.step = 300
                if hasattr(writer, '_wait_for_compaction'):
                    writer._wait_for_compaction()
                before = copy.deepcopy(getattr(writer, '_manifest', None))
                with self.assertRaises(ValueError):
                    writer.write(other=12, counter=10**20 + 300 * 10**6 + 1)
                self.assertEqual(getattr(writer, '_manifest', None), before)
                writer.write(counter=10**20 + 300 * 10**6)
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(tmp) as reader:
                    self.assertNotIn('other', reader.metrics())
                    self.assertEqual(reader.metric_values('counter').tolist(),
                                     [10**20 + step * 10**6 for step in (0, 100, 200, 251, 300)])

    def test_rewind_step_sequence_can_continue_by_row(self):
        import tempfile
        import plattli
        for kind in ('direct', '1', '100'):
            for recover_hot in (False, True) if kind != 'direct' else (False,):
                with self.subTest(kind=kind, recover_hot=recover_hot), tempfile.TemporaryDirectory() as tmp:
                    writer = self.writer(kind, tmp)
                    for step in (0, 100, 200, 251):
                        writer.step = step
                        writer.write(counter=10**20 + step)
                        writer.end_step()
                    writer.finish(optimize=False, zip=False)
                    with self.writer(kind, tmp, step=150) as resumed:
                        resumed.write(counter=10**20 + 200)
                        resumed.end_step()
                        if not recover_hot:
                            resumed.finish(optimize=False, zip=False)
                    if recover_hot:
                        with self.writer(kind, tmp, step=151) as resumed:
                            resumed.write(counter=10**20 + 300)
                            resumed.end_step()
                            resumed.finish(optimize=False, zip=False)
                    with plattli.Reader(tmp) as reader:
                        self.assertEqual(reader.metric_indices('counter').tolist(),
                                         [0, 100, 150, 151] if recover_hot else [0, 100, 150])
                        self.assertEqual(reader.metric_values('counter').tolist(),
                                         [10**20 + row * 100 for row in range(4 if recover_hot else 3)])
                        self.assertNotIn('on', reader.manifest()['counter']['values'])

    def test_irregular_fixed_width_history_promotes_to_exact_step_sequence(self):
        import tempfile
        import plattli
        for kind in ('direct', '1', '100', 'bulk'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                writer = self.writer(kind, tmp)
                for step in (0, 1, 3, 5):
                    writer.step = step
                    writer.write(counter=2**63 - 5 + step)
                    writer.end_step()
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(tmp) as reader:
                    self.assertEqual(reader.manifest()['counter']['dtype'], 'int')
                    self.assertEqual(reader.metric_values('counter').tolist(), [2**63 - 5 + step for step in (0, 1, 3, 5)])
