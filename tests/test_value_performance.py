import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import plattli
import plattli.writer as writer_module


class TestValuePerformance(unittest.TestCase):
    def test_direct_continuations_do_not_publish_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = plattli.DirectWriter(tmp, write_threads=0)
            for row in range(3):
                writer.write(constant=0.5, arithmetic=10 + row * 3,
                             huge=10**20 + row * 10**6, fraction=np.float64(row * 0.25))
                writer.end_step()
            root = Path(tmp) / 'plattli'
            self.assertNotIn('values', writer._manifest['fraction'])
            self.assertFalse((root / 'fraction.values').exists())
            before = {path.name: path.read_bytes() for path in root.glob('*.values')}
            manifest = (root / 'plattli.json').read_bytes()
            with mock.patch.object(writer_module, '_write_manifest', side_effect=AssertionError('continuation rewrites manifest')), \
                    mock.patch.object(writer_module, '_replace_text_checked', side_effect=AssertionError('continuation rewrites sequence head')):
                for row in range(3, 35):
                    writer.write(constant=0.5, arithmetic=10 + row * 3,
                                 huge=10**20 + row * 10**6, fraction=np.float64(row * 0.25))
                    writer.end_step()
            self.assertEqual((root / 'plattli.json').read_bytes(), manifest)
            self.assertEqual({path.name: path.read_bytes() for path in root.glob('*.values')}, before)
            with plattli.Reader(tmp) as reader:
                self.assertEqual(reader.metric_values('constant').tolist(), [0.5] * 35)
                self.assertEqual(reader.metric_values('arithmetic').tolist(), [10 + row * 3 for row in range(35)])
                self.assertEqual(reader.metric_values('huge').tolist(), [10**20 + row * 10**6 for row in range(35)])
                self.assertEqual(reader.metric_values('fraction').tolist(), [row * 0.25 for row in range(35)])
            writer.finish(optimize=False, zip=False)

    def test_direct_step_affine_continuations_do_not_publish_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = plattli.DirectWriter(tmp, write_threads=0)
            indices = [0, 100, 200, 251]
            for step in indices:
                writer.step = step
                writer.write(counter=10**20 + step * 10**6)
                writer.end_step()
            with mock.patch.object(writer_module, '_write_manifest', side_effect=AssertionError('continuation rewrites manifest')), \
                    mock.patch.object(writer_module, '_replace_text_checked', side_effect=AssertionError('continuation rewrites sequence head')):
                for step in (300, 500, 501, 1000):
                    indices.append(step)
                    writer.step = step
                    writer.write(counter=10**20 + step * 10**6)
                    writer.end_step()
            with plattli.Reader(tmp) as reader:
                self.assertEqual(reader.metric_indices('counter').tolist(), indices)
                self.assertEqual(reader.metric_values('counter').tolist(), [10**20 + step * 10**6 for step in indices])
            writer.finish(optimize=False, zip=False)

    def test_direct_fallback_freezes_head_before_publishing_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = plattli.DirectWriter(tmp, write_threads=0)
            for _ in range(3):
                writer.write(value=5)
                writer.end_step()
            root = Path(tmp) / 'plattli'
            replace = writer_module._replace_text_checked
            publication = []

            def inspect_publication(path, payload, label):
                if path == root / 'value.values':
                    self.assertEqual(json.loads(payload)['values'], {'start': 5, 'step': 0, 'count': 3})
                    publication.append('head')
                elif path == root / 'plattli.json':
                    self.assertNotIn('values', json.loads(payload)['value'])
                    self.assertEqual(json.loads((root / 'value.values').read_text())['values']['count'], 3)
                    self.assertEqual(np.fromfile(root / 'value.i64', dtype=np.int64).tolist(), [5, 5, 5, 6])
                    publication.append('manifest')
                return replace(path, payload, label)

            with plattli.Reader(tmp) as reader:
                self.assertEqual(reader.metric_values('value').tolist(), [5, 5, 5])
                with mock.patch.object(writer_module, '_replace_text_checked', side_effect=inspect_publication):
                    writer.write(value=6)
                    writer.end_step()
                self.assertEqual(publication, ['head', 'manifest'])
                self.assertEqual(reader.metric_values('value').tolist(), [5, 5, 5, 6])
            writer.finish(optimize=False, zip=False)
