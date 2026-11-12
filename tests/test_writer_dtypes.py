import json
import copy
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

import plattli
import plattli.writer as writer_module


def _writer(kind, root, **kwargs):
    if kind == 'bulk':
        return plattli.BulkWriter(root, **kwargs)
    if kind == 'direct':
        return plattli.DirectWriter(root, write_threads=4, **kwargs)
    return plattli.CompactingWriter(root, hotsize=2 if kind == 'cold' else 100, **kwargs)


def _stored(root, name):
    with plattli.Reader(root) as reader:
        return reader.manifest()[name]['dtype'], reader.metric_values(name, dtype='storage')


class TestWriterDtypes(unittest.TestCase):
    def test_invalid_batches_leave_values_and_dtypes_unchanged(self):
        for kind in ('direct', 'cold', 'hot', 'bulk'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                writer = _writer(kind, root)
                writer.write(a=np.uint8(1), b=1, loss=1.0, note={'ok': True})
                writer.end_step()
                writer.write(already=9)

                def snapshot():
                    if kind == 'bulk':
                        state = {name: (column.dtype, list(column.i), list(column.v))
                                 for name, column in writer._columns.items()}
                    else:
                        state = (writer._manifest, getattr(writer, '_current_row', {}),
                                 getattr(writer, '_hot_rows', []))
                    files = {str(path.relative_to(root)): path.read_bytes()
                             for path in root.rglob('*') if path.is_file()}
                    return copy.deepcopy((state, writer._step_metrics)), files

                before = snapshot()
                invalid_batches = [
                    {'a': 300, 'b': 1.5},
                    {'loss': np.float64(2.0), 'b': 1.5},
                    {'new': 4, 'bad': object()},
                    {'a': 300, 'note': {'bad': {1}}},
                    {'a': 300, 'note': np.array([1, 2])},
                    {'a': 300, 'note': '\ud800'},
                    {'a': 300, '\ud800': 1},
                    {'a': 300, 'step': 5},
                    {'a': 300, 'already': 10},
                    {'loss': np.float64(2.0), 'bad': 1e300},
                ]
                for batch in invalid_batches:
                    with self.subTest(batch=repr(batch)):
                        with self.assertRaises((TypeError, ValueError, OverflowError, RuntimeError)):
                            writer.write(batch)
                        self.assertEqual(snapshot(), before)
                writer.finish(optimize=False, zip=False)
                self.assertEqual(_stored(root, 'a')[0], 'u8')
                self.assertEqual(_stored(root, 'a')[1].tolist(), [1])
                self.assertEqual(_stored(root, 'loss')[0], 'f32')
                self.assertEqual(_stored(root, 'loss')[1].tolist(), [1.0])

    def test_integer_inputs_to_float_columns_are_exact(self):
        for kind in ('direct', 'cold', 'hot', 'bulk'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                writer = _writer(kind, root)
                writer.write(loss=0.5, wide=np.float64(0.5))
                writer.end_step()
                for value in (0, 1, 2**24, 2**100):
                    writer.write(loss=value)
                    writer.end_step()
                dtype = writer._columns['loss'].dtype if kind == 'bulk' else writer._manifest['loss']['dtype']
                self.assertEqual(dtype, 'f32')
                writer.write(loss=2**24 + 1, wide=np.uint64(2**63))
                writer.end_step()
                dtype = writer._columns['loss'].dtype if kind == 'bulk' else writer._manifest['loss']['dtype']
                self.assertEqual(dtype, 'f64')
                for value in (2**53 + 1, -(2**53 + 1), np.uint64(2**64 - 1)):
                    with self.subTest(value=value), self.assertRaisesRegex(OverflowError, 'exactly'):
                        writer.write(loss=value)
                writer.write(loss=2**53, wide=1)
                writer.finish(optimize=False, zip=False)
                self.assertEqual(_stored(root, 'loss')[1].tolist(),
                                 [0.5, 0.0, 1.0, float(2**24), float(2**100), float(2**24 + 1), float(2**53)])
                self.assertEqual(_stored(root, 'wide')[1].tolist(), [0.5, float(2**63), 1.0])

    def test_finite_float_overflow_rejected_but_nonfinite_values_allowed(self):
        maximum = float(np.finfo(np.float32).max)
        for kind in ('direct', 'cold', 'hot', 'bulk'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                writer = _writer(kind, root)
                for value in (1e300, -1e300):
                    with self.assertRaises(OverflowError):
                        writer.write(loss=value)
                writer.write(loss=maximum, wide=np.float64(1e300))
                writer.end_step()
                # Values just above f32's maximum can still round to a finite f32.
                writer.write(loss=np.nextafter(maximum, float('inf')).item())
                writer.end_step()
                with self.assertRaises(OverflowError):
                    writer.write(loss=1e300)
                for value in (float('inf'), float('-inf'), float('nan')):
                    writer.write(loss=value)
                    writer.end_step()
                writer.finish(optimize=False, zip=False)
                dtype, values = _stored(root, 'loss')
                self.assertEqual(dtype, 'f32')
                self.assertEqual(values[:4].tolist(), [maximum, maximum, float('inf'), float('-inf')])
                self.assertTrue(np.isnan(values[-1]))
                self.assertEqual(_stored(root, 'wide')[1].tolist(), [1e300])

    def test_step_validation_survives_optimized_python_without_mutation(self):
        script = r'''
import json
import tempfile
from pathlib import Path
import plattli

def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}

factories = [(plattli.DirectWriter, {'write_threads': 0}),
             (plattli.CompactingWriter, {'hotsize': 100}),
             (plattli.BulkWriter, {})]
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    for cls, kwargs in factories:
        for zipped in (False, True):
            root = base / (cls.__name__ + str(zipped))
            writer = cls(root, **kwargs)
            writer.write(value=7)
            writer.end_step()
            writer.finish(optimize=False, zip=zipped)
            before = snapshot(root)
            resume = {'overwrite': True} if cls is plattli.BulkWriter else {'allow_resume_finalized': True}
            for invalid in (-1, 2**32):
                try:
                    cls(root, step=invalid, **kwargs, **resume)
                except ValueError:
                    pass
                else:
                    raise RuntimeError('invalid constructor step accepted')
                if snapshot(root) != before:
                    raise RuntimeError('invalid resume changed existing files')
                fresh = root / str(invalid)
                try:
                    cls(fresh, step=invalid, **kwargs)
                except ValueError:
                    pass
                else:
                    raise RuntimeError('invalid new step accepted')
                if fresh.exists():
                    raise RuntimeError('invalid constructor created files')
        root = base / (cls.__name__ + '-boundary')
        writer = cls(root, step=2**32 - 1, **kwargs)
        writer.write(value=7)
        writer.end_step()
        for invalid in (2**32, -1):
            writer.step = invalid
            try:
                writer.write(value=8)
            except ValueError:
                pass
            else:
                raise RuntimeError('invalid write step accepted')
        writer.step = 2**32
        writer.finish(optimize=False, zip=False)
        with plattli.Reader(root) as reader:
            if reader.metric_indices('value').tolist() != [2**32 - 1]:
                raise RuntimeError('maximum valid step did not survive finish')
print('ok')
'''
        result = subprocess.run([sys.executable, '-O', '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'ok')

    def test_rewrite_rejects_lossy_integer_cast_before_publication(self):
        for source, target, values in [('u64', 'i64', [2**64 - 5, 7]),
                                       ('i64', 'u8', [-1, 7]), ('i64', 'i8', [128])]:
            with self.subTest(source=source, target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / f'counter.{source}'
                np.asarray(values, dtype=writer_module.DTYPE_TO_NUMPY[source]).tofile(path)
                manifest = {'counter': {'dtype': source, 'indices': 'indices'}}
                before = path.read_bytes()
                publish = mock.Mock()
                with self.assertRaises(OverflowError):
                    writer_module._rewrite_stored_dtype(root, manifest, 'counter', target, publish)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(manifest['counter']['dtype'], source)
                self.assertFalse((root / f'counter.{target}').exists())
                publish.assert_not_called()

    def test_live_dtype_reconciliation_preserves_values(self):
        for kind in ['direct', 'cold', 'hot']:
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    writer = _writer(kind, tmp_path)
                    precise = np.float64(1.123456789012345)
                    rows = [
                        dict(counter=np.uint8(100), narrow=np.uint8(2), loss=1.123456789, signed=np.int8(-1)),
                        dict(counter=7, narrow=np.uint8(3), loss=precise, signed=np.uint8(255)),
                        dict(counter=np.uint8(200), narrow=np.uint8(4), loss=2.123456789, signed=np.int8(-2)),
                    ]
                    for row in rows:
                        writer.write(row)
                        writer.end_step()
                    writer.finish(optimize=False, zip=False)
                    dtype, values = _stored(tmp_path, 'counter')
                    self.assertEqual(dtype, 'i64')
                    self.assertEqual(values.tolist(), [100, 7, 200])
                    self.assertEqual(_stored(tmp_path, 'narrow')[0], 'u8')
                    dtype, values = _stored(tmp_path, 'loss')
                    self.assertEqual(dtype, 'f64')
                    self.assertEqual(values.tolist(), [float(np.float32(rows[0]['loss'])), precise, rows[2]['loss']])
                    dtype, values = _stored(tmp_path, 'signed')
                    self.assertEqual(dtype, 'i16')
                    self.assertEqual(values.tolist(), [-1, 255, -2])

    def test_preemption_keeps_explicit_narrow_dtype(self):
        for kind in ['direct', 'cold', 'hot']:
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    with _writer(kind, tmp_path) as writer:
                        for value in (100, 150, 200):
                            writer.write(counter=np.uint8(value))
                            writer.end_step()
                    with mock.patch.object(writer_module, '_rewrite_stored_dtype', side_effect=AssertionError('unexpected widening')):
                        resumed = _writer(kind, tmp_path, step=3)
                        self.assertEqual(resumed._manifest['counter']['dtype'], 'u8')
                        resumed.write(counter=np.uint8(201))
                        resumed.end_step()
                        resumed.finish(optimize=False, zip=False)
                    dtype, values = _stored(tmp_path, 'counter')
                    self.assertEqual(dtype, 'u8')
                    self.assertEqual(values.tolist(), [100, 150, 200, 201])
                    self.assertEqual((tmp_path / 'plattli/counter.u8').stat().st_size, 4)

    def test_finalized_resume_promotes_only_on_next_measurement(self):
        for kind in ['direct', 'cold', 'hot']:
            for zipped in [False, True]:
                with self.subTest(kind=kind, zipped=zipped):
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp_path = Path(tmp)
                        writer = _writer(kind, tmp_path)
                        for value in (100, 200):
                            writer.write(counter=value)
                            writer.end_step()
                        writer.finish(zip=zipped)
                        resumed = _writer(kind, tmp_path, step=2, allow_resume_finalized=True)
                        self.assertEqual(resumed._manifest['counter']['dtype'], 'u8')
                        resumed.write(counter=300)
                        self.assertEqual(resumed._manifest['counter']['dtype'], 'i64')
                        resumed.end_step()
                        resumed.write(counter=-1)
                        resumed.finish(optimize=False, zip=False)
                        self.assertEqual(_stored(tmp_path, 'counter')[1].tolist(), [100, 200, 300, -1])

    def test_uint64_reconciliation_is_exact(self):
        for kind in ['direct', 'cold', 'hot']:
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    maximum = 2 ** 64 - 1
                    writer = _writer(kind, tmp_path)
                    writer.write(large=np.uint64(maximum), small=np.uint64(10))
                    writer.end_step()
                    writer.write(large=1, small=-1)
                    writer.end_step()
                    with self.assertRaisesRegex(OverflowError, 'cannot share'):
                        writer.write(large=-1)
                    writer.write(large=np.uint64(2 ** 63 + 1))
                    writer.end_step()
                    writer.finish(zip=False)
                    dtype, values = _stored(tmp_path, 'large')
                    self.assertEqual(dtype, 'u64')
                    self.assertEqual(values.tolist(), [maximum, 1, 2 ** 63 + 1])
                    self.assertEqual(_stored(tmp_path, 'small')[1].tolist(), [10, -1])

    def test_incompatible_numeric_values_fail_before_writing(self):
        for kind in ['direct', 'cold', 'hot']:
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    writer = _writer(kind, tmp_path)
                    writer.write(counter=-1, loss=1.0)
                    writer.end_step()
                    with self.assertRaisesRegex(TypeError, 'cannot mix'):
                        writer.write(counter=1.5)
                    with self.assertRaisesRegex(TypeError, 'cannot mix'):
                        writer.write(loss='not numeric')
                    with self.assertRaisesRegex(OverflowError, 'cannot share'):
                        writer.write(counter=np.uint64(2 ** 63))
                    writer.write(counter=2, loss=2.0)
                    writer.finish(optimize=False, zip=False)
                    self.assertEqual(_stored(tmp_path, 'counter')[1].tolist(), [-1, 2])

    def test_promotion_waits_for_background_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            writer = plattli.CompactingWriter(tmp_path, hotsize=1)
            for value in (100, 200, 150, 175):
                writer.write(counter=np.uint8(value))
                writer.end_step()
            writer._wait_for_compaction()
            writer._flush_hot()
            writer._wait_for_compaction()
            started = threading.Event()
            release = threading.Event()
            append = writer_module._append_numeric

            def blocked_append(path, values, dtype):
                if dtype == 'u8':
                    started.set()
                    self.assertTrue(release.wait(5))
                append(path, values, dtype)
            with mock.patch.object(writer_module, '_append_numeric', side_effect=blocked_append):
                writer.write(counter=np.uint8(180))
                writer.end_step()
                writer.write(counter=np.uint8(190))
                writer.end_step()
                self.assertTrue(started.wait(5))
                with ThreadPoolExecutor(max_workers=1) as executor:
                    pending = executor.submit(writer.write, counter=300)
                    self.assertFalse(pending.done())
                    release.set()
                    pending.result(timeout=5)
            writer.finish(optimize=False, zip=False)
            self.assertEqual(_stored(tmp_path, 'counter')[1].tolist(), [100, 200, 150, 175, 180, 190, 300])

    def test_promotion_keeps_old_column_when_manifest_publish_fails(self):
        for kind in ['direct', 'cold', 'hot']:
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    writer = _writer(kind, tmp_path)
                    writer.write(counter=np.uint8(100))
                    writer.end_step()
                    with mock.patch.object(writer_module, '_replace_text_checked', side_effect=OSError('interrupted promotion')):
                        with self.assertRaisesRegex(OSError, 'interrupted promotion'):
                            writer.write(counter=300)
                    self.assertEqual(json.loads((tmp_path / 'plattli/plattli.json').read_text())['counter']['dtype'], 'u8')
                    resumed = _writer(kind, tmp_path, step=1)
                    resumed.write(counter=300)
                    resumed.finish(optimize=False, zip=False)
                    self.assertEqual(_stored(tmp_path, 'counter')[1].tolist(), [100, 300])

    def test_resume_after_interrupted_finish_truncates_before_promotion(self):
        for kind in ['direct', 'cold', 'hot']:
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    writer = _writer(kind, tmp_path)
                    for value in (100, 200, 150):
                        writer.write(counter=value)
                        writer.end_step()
                    with mock.patch.object(writer_module, '_optimize_indices', side_effect=OSError('interrupted finish')):
                        with self.assertRaisesRegex(OSError, 'interrupted finish'):
                            writer.finish(zip=False)
                    self.assertEqual(_stored(tmp_path, 'counter')[0], 'u8')
                    resumed = _writer(kind, tmp_path, step=1)
                    self.assertEqual(resumed._manifest['counter']['dtype'], 'u8')
                    resumed.write(counter=-300)
                    resumed.finish(optimize=False, zip=False)
                    self.assertEqual(_stored(tmp_path, 'counter')[1].tolist(), [100, -300])

    def test_uint64_reconciliation_checks_retained_bounds_once(self):
        for kind in ['direct', 'cold', 'hot']:
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    writer = _writer(kind, tmp_path)
                    writer.write(counter=np.uint64(2 ** 64 - 1))
                    writer.end_step()
                    with mock.patch.object(writer_module, '_integer_bounds', wraps=writer_module._integer_bounds) as bounds:
                        for value in range(10):
                            writer.write(counter=value)
                            writer.end_step()
                        self.assertEqual(bounds.call_count, 1)
                        with self.assertRaises(OverflowError):
                            writer.write(counter=-1)
                        self.assertEqual(bounds.call_count, 1)
                    writer.finish(optimize=False, zip=False)
                    self.assertEqual(_stored(tmp_path, 'counter')[1].tolist(), [2 ** 64 - 1, *range(10)])

    def test_integer_compaction_preserves_boundaries(self):
        for value in [2 ** 63 - 1, 2 ** 63, 2 ** 64 - 1]:
            with self.subTest(value=value):
                values = np.asarray([0, value], dtype=np.uint64)
                self.assertEqual(writer_module._tight_dtype(values).tolist(), [0, value])

    def test_checked_integer_coercion(self):
        for value in [256, -1, 1.5, float('nan'), float('inf')]:
            with self.subTest(value=value):
                with self.assertRaises((ValueError, OverflowError)):
                    writer_module._coerce_stored_value(value, 'u8', 'counter', 'run')

    def test_array_like_scalars_reconcile_signed_uint64(self):
        class ArrayScalar:
            def __init__(self, value):
                self.value = value

            def __array__(self, dtype=None, copy=None):
                return np.asarray(self.value, dtype=dtype)

        for kind in ('direct', 'cold', 'hot', 'bulk'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                writer = _writer(kind, root)
                writer.write(signed=np.int64(-1), unsigned=np.uint64(7))
                writer.end_step()
                writer.write(signed=ArrayScalar(np.uint64(7)), unsigned=ArrayScalar(np.int64(-1)))
                writer.finish(optimize=False, zip=False)
                self.assertEqual(_stored(root, 'signed')[1].tolist(), [-1, 7])
                self.assertEqual(_stored(root, 'unsigned')[1].tolist(), [7, -1])
