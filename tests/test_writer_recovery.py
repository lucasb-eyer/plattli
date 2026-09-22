import copy
import json
import os
import tempfile
import threading
import unittest
import zipfile
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np

import plattli
import plattli.writer as writer_module


def _writer(kind, root, **kwargs):
    defaults = {'write_threads': 2} if kind == 'direct' else {'hotsize': 100} if kind == 'compact' else {}
    defaults.update(kwargs)
    cls = {'direct': plattli.DirectWriter, 'compact': plattli.CompactingWriter, 'bulk': plattli.BulkWriter}[kind]
    return cls(root, **defaults)


def _snapshot(root):
    return {str(path.relative_to(root)): ('link', os.readlink(path)) if path.is_symlink()
            else ('file', path.read_bytes()) if path.is_file() else ('dir',)
            for path in root.rglob('*')}


class TestWriterRecovery(unittest.TestCase):
    def test_steps_require_integers_before_any_mutation(self):
        for kind in ('direct', 'compact', 'bulk'):
            for invalid in (1.5, 1.0, True, np.bool_(False), '1', np.float32(1)):
                with self.subTest(kind=kind, invalid=repr(invalid)), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / 'run'
                    with self.assertRaises(TypeError):
                        _writer(kind, root, step=invalid)
                    self.assertFalse(root.exists())
                    writer = _writer(kind, root)
                    writer.write(counter=1)
                    writer.end_step()
                    writer.step = invalid
                    before = _snapshot(root)
                    with self.assertRaises(TypeError):
                        writer.write(other=2)
                    with self.assertRaises(TypeError):
                        writer.end_step()
                    self.assertEqual(_snapshot(root), before)
                    writer.step = 1
                    writer.finish(optimize=False, zip=False)

    def test_numpy_step_increment_cannot_wrap(self):
        for kind in ('direct', 'compact', 'bulk'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                writer = _writer(kind, root, step=np.uint32(2**32 - 1))
                self.assertIs(type(writer.step), int)
                writer.write(counter=1)
                writer.step = np.uint32(2**32 - 1)
                writer.end_step()
                self.assertIs(type(writer.step), int)
                self.assertEqual(writer.step, 2**32)
                with self.assertRaises(ValueError):
                    writer.write(counter=2)
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(root) as reader:
                    self.assertEqual(reader.metric_indices('counter').tolist(), [2**32 - 1])

    def test_resume_config_future_directory_rejected_before_extraction(self):
        for kind in ('direct', 'compact'):
            for config in ('plattli', 'plattli/nested', 'config-link'):
                with self.subTest(kind=kind, config=config), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / 'run'
                    original = _writer('bulk', root)
                    original.write(counter=1)
                    original.end_step()
                    original.finish()
                    with zipfile.ZipFile(root / 'metrics.plattli', 'a') as archive:
                        archive.writestr('nested/sidecar.txt', 'keep')
                    if config == 'config-link':
                        (root / config).symlink_to(root / 'plattli/nested')
                    before = _snapshot(root)
                    with self.assertRaises(FileNotFoundError):
                        _writer(kind, root, config=config, allow_resume_finalized=True)
                    self.assertEqual(_snapshot(root), before)

    def test_interrupted_config_replacement_preserves_previous_config_and_sources(self):
        for kind in ('direct', 'compact'):
            for linked in (False, True):
                for replacement in ('value', 'link'):
                    with self.subTest(kind=kind, linked=linked, replacement=replacement), tempfile.TemporaryDirectory() as tmp:
                        base = Path(tmp)
                        source, other = base / 'source.json', base / 'other.json'
                        source.write_text('{"old":true}')
                        other.write_text('{"other":true}')
                        writer = _writer(kind, base / 'run', config=str(source) if linked else {'old': True})
                        config = writer.root / 'config.json'
                        old_bytes = config.read_bytes()
                        original_replace = Path.replace

                        def interrupt(path, target):
                            if path == config.with_name('config.json.tmp'):
                                raise OSError('interrupted config publication')
                            return original_replace(path, target)

                        with mock.patch.object(Path, 'replace', interrupt), self.assertRaises(OSError):
                            writer.set_config({'new': True} if replacement == 'value' else str(other))
                        self.assertEqual(config.read_bytes(), old_bytes)
                        self.assertEqual(config.is_symlink(), linked)
                        self.assertEqual(source.read_text(), '{"old":true}')
                        self.assertEqual(other.read_text(), '{"other":true}')
                        # A leftover temporary link must not redirect a later value write.
                        writer.set_config({'done': True})
                        self.assertFalse(config.is_symlink())
                        self.assertEqual(json.loads(config.read_text()), {'done': True})
                        self.assertEqual(source.read_text(), '{"old":true}')
                        self.assertEqual(other.read_text(), '{"other":true}')
                        writer.finish(optimize=False, zip=False)

    def test_resume_preserves_unknown_pairs_until_finalization(self):
        for kind in ('direct', 'compact'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                writer = _writer(kind, root)
                writer.write(value=1.0)
                writer.end_step()
                writer.finish(optimize=False, zip=False)
                storage = root / 'plattli'
                for rel in ('value.f64', 'value.f64.tmp', 'plattli.json.tmp', 'lost.u8', 'lost.indices', 'notes.u8'):
                    (storage / rel).write_bytes(b'keep until authorized cleanup')
                resumed = _writer(kind, root, step=1)
                for rel in ('value.f64', 'value.f64.tmp', 'plattli.json.tmp'):
                    self.assertFalse((storage / rel).exists())
                for rel in ('lost.u8', 'lost.indices', 'notes.u8'):
                    self.assertEqual((storage / rel).read_bytes(), b'keep until authorized cleanup')
                resumed.finish(optimize=False, zip=False)
                self.assertFalse((storage / 'lost.u8').exists())
                self.assertFalse((storage / 'lost.indices').exists())
                self.assertEqual((storage / 'notes.u8').read_bytes(), b'keep until authorized cleanup')

    def test_reserved_config_temporary_sources_are_rejected_without_mutation(self):
        for kind in ('direct', 'compact'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                writer = _writer(kind, root / 'run', config={'old': True})
                config = writer.root / 'config.json'
                source = writer.root / 'config.json.tmp'
                source.write_text('{"source":true}')
                alias = root / 'alias.json'
                alias.symlink_to(source)
                before = _snapshot(root)
                for target in (source, alias):
                    with self.assertRaisesRegex(ValueError, 'reserved temporary path'):
                        writer.set_config(str(target))
                    self.assertEqual(_snapshot(root), before)
                config.unlink()
                config.symlink_to(source)
                before = _snapshot(root)
                with self.assertRaisesRegex(ValueError, 'reserved temporary path'):
                    writer.set_config({'new': True})
                self.assertEqual(_snapshot(root), before)
                # The reservation is location-specific, not a basename restriction.
                config.unlink()
                external = root / 'config.json.tmp'
                external.write_text('{"external":true}')
                writer.set_config(str(external))
                self.assertEqual(json.loads(config.read_text()), {'external': True})
                writer.finish(optimize=False, zip=False)
                self.assertEqual(external.read_text(), '{"external":true}')

    def test_invalid_finish_overrides_do_not_modify_existing_outputs(self):
        for kind in ('direct', 'compact', 'bulk'):
            for zipped in (False, True):
                with self.subTest(kind=kind, zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    original = _writer(kind, root, config={'keep': True})
                    original.write(counter=1)
                    original.end_step()
                    original.finish(optimize=False, zip=zipped)
                    resume = {'overwrite': True} if kind == 'bulk' else {'step': 1, 'allow_resume_finalized': True}
                    writer = _writer(kind, root, **resume)
                    writer.write(counter=2)
                    writer.end_step()
                    before = _snapshot(root)
                    for override in ({'bad': object()}, {'bad': '\ud800'}):
                        for output_zip in (False, True):
                            with self.subTest(override=repr(override), output_zip=output_zip):
                                with self.assertRaises((TypeError, UnicodeError)):
                                    writer.finish(config_overrides=override, zip=output_zip)
                                self.assertEqual(_snapshot(root), before)
                                self.assertFalse(writer._finished)
                    writer.finish(optimize=False, zip=zipped)

    def test_invalid_constructor_options_do_not_modify_runs(self):
        for kind in ('direct', 'compact', 'bulk'):
            for zipped in (False, True):
                with self.subTest(kind=kind, zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / 'run'
                    writer = _writer(kind, root)
                    writer.write(counter=1)
                    writer.end_step()
                    writer.finish(optimize=False, zip=zipped)
                    (root / 'configdir').mkdir()
                    before = _snapshot(root)
                    options = [{'config': {'bad': object()}}, {'config': {'bad': '\ud800'}},
                               {'config': 'configdir'}]
                    if kind == 'direct':
                        options.extend([{'write_threads': -1}, {'write_threads': 1.5}])
                    if kind == 'compact':
                        options.extend([{'hotsize': 0}, {'hotsize': -1}, {'hotsize': 'invalid'}])
                    resume = {'overwrite': True} if kind == 'bulk' else {'allow_resume_finalized': True}
                    for option in options:
                        with self.subTest(option=repr(option)):
                            with self.assertRaises((TypeError, ValueError, FileNotFoundError)):
                                _writer(kind, root, step=0, **resume, **option)
                            self.assertEqual(_snapshot(root), before)

    def test_memory_config_is_snapshotted_and_self_source_is_preserved(self):
        for kind in ('direct', 'compact', 'bulk'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = {'nested': {'values': [1, 2]}}
                writer = _writer(kind, root, config=config)
                config['nested']['values'].append(object())
                writer.write(counter=1)
                writer.end_step()
                writer.finish(optimize=False, zip=False)
                self.assertEqual(json.loads((root / 'plattli/config.json').read_text()),
                                 {'nested': {'values': [1, 2]}})
                if kind != 'bulk':
                    writer = _writer(kind, root, step=1, config='plattli/config.json')
                    self.assertFalse((root / 'plattli/config.json').is_symlink())
                    writer.finish(optimize=False, zip=False)

    def test_interrupted_promotion_orphans_removed_on_resume(self):
        for kind in ('direct', 'compact'):
            for window in ('before_manifest', 'after_manifest'):
                with self.subTest(kind=kind, window=window), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    writer = _writer(kind, root)
                    for value in (1.0, 2.0, 3.0):
                        writer.write(value=value)
                        writer.end_step()
                    writer.finish(optimize=False, zip=False)
                    reader = plattli.Reader(root)
                    self.assertEqual(reader.metric_values('value').tolist(), [1.0, 2.0, 3.0])
                    writer = _writer(kind, root, step=3)
                    unlink = Path.unlink

                    def fail_unlink(path, *args, **kwargs):
                        if path.name == 'value.f32':
                            raise OSError('interrupted old-file deletion')
                        return unlink(path, *args, **kwargs)

                    failure = mock.patch.object(writer_module, '_replace_text_checked', side_effect=OSError('interrupted manifest')) \
                        if window == 'before_manifest' else mock.patch.object(Path, 'unlink', fail_unlink)
                    with failure, self.assertRaises(OSError):
                        writer.write(value=np.float64(4.123456789))
                    self.assertTrue((root / 'plattli/value.f32').exists())
                    self.assertTrue((root / 'plattli/value.f64').exists())
                    resumed = _writer(kind, root, step=1)
                    self.assertEqual(len(list((root / 'plattli').glob('value.f*'))), 1)
                    resumed.write(value=np.float64(7.123456789))
                    resumed.end_step()
                    self.assertEqual(reader.metric_values('value').tolist(), [1.0, 7.123456789])
                    reader.close()
                    resumed.finish(optimize=False, zip=True)
                    with zipfile.ZipFile(root / 'metrics.plattli') as archive:
                        self.assertNotIn('value.f32', archive.namelist())
                        self.assertEqual(np.frombuffer(archive.read('value.f64'), dtype=np.float64).tolist(),
                                         [1.0, 7.123456789])

    def test_finish_cleans_proven_orphans_and_preserves_ancillary_files(self):
        for kind in ('direct', 'compact'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                writer = _writer(kind, root)
                writer.write({'detail/value': 1.0})
                writer.end_step()
                writer.write({'detail/value': 2.0})
                writer.end_step()
                storage = root / 'plattli'
                artifacts = ['detail/value.f64', 'detail/value.f64.tmp', 'lost.u8', 'lost.indices',
                             'plattli.json.tmp', 'config.json.tmp']
                ancillary = ['notes.jsonl', 'notes.u8', 'notes.txt', 'notes.txt.tmp', 'standalone.indices']
                for rel in artifacts + ancillary:
                    (storage / rel).write_bytes(b'orphan' if rel in artifacts else b'keep')
                writer.finish(optimize=True, zip=True)
                with zipfile.ZipFile(root / 'metrics.plattli') as archive:
                    for rel in artifacts + ['detail/value.indices']:
                        self.assertNotIn(rel, archive.namelist())
                    for rel in ancillary:
                        self.assertEqual(archive.read(rel), b'keep')
                    self.assertIn('config.json', archive.namelist())

    def test_cleanup_preserves_internal_config_target_and_recovery_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = _writer('compact', root)
            writer.write(counter=np.uint8(1))
            writer.end_step()
            writer.write(counter=np.uint8(2))
            writer.end_step()
            writer.__exit__(None, None, None)
            storage = root / 'plattli'
            (storage / 'hot.jsonl').rename(storage / 'hot.compacting.jsonl')
            (storage / 'hot.jsonl').write_text('{"step":2,"counter":3}\n')
            (storage / 'settings.jsonl').write_text('{"keep":true}')
            (storage / 'settings.indices').write_bytes(b'orphan index')
            (storage / 'config.json').unlink()
            (storage / 'config.json').symlink_to(storage / 'settings.jsonl')
            (storage / 'counter.i64').write_bytes(b'orphan')
            (storage / 'notes.jsonl').write_text('keep')
            resumed = _writer('compact', root, step=3, config=None)
            self.assertFalse((storage / 'counter.i64').exists())
            self.assertFalse((storage / 'hot.compacting.jsonl').exists())
            self.assertTrue((storage / 'hot.jsonl').exists())
            self.assertEqual((storage / 'notes.jsonl').read_text(), 'keep')
            self.assertEqual(json.loads((storage / 'config.json').read_text()), {'keep': True})
            with plattli.Reader(root) as reader:
                self.assertEqual(reader.metric_values('counter').tolist(), [1, 2, 3])
            resumed.finish(optimize=False, zip=True)
            with zipfile.ZipFile(root / 'metrics.plattli') as archive:
                self.assertEqual(json.loads(archive.read('config.json')), {'keep': True})
                self.assertEqual(json.loads(archive.read('settings.jsonl')), {'keep': True})

    def test_cleanup_refuses_missing_authoritative_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([0, 1], dtype=np.uint32).tofile(root / 'counter.indices')
            np.asarray([1, 2], dtype=np.uint8).tofile(root / 'counter.u8')
            before = _snapshot(root)
            with self.assertRaises(FileNotFoundError):
                writer_module._cleanup_stored_files(root, {'counter': {'dtype': 'i64', 'indices': 'indices'}})
            self.assertEqual(_snapshot(root), before)

    def test_mutating_json_after_write_cannot_change_accepted_values(self):
        for kind in ('direct', 'compact', 'bulk'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                writer = _writer(kind, root)
                writer.write(note={'initial': True})
                writer.end_step()
                payload = {'nested': {'values': [1, {'x': 2}]}}
                expected = copy.deepcopy(payload)
                started, release = threading.Event(), threading.Event()
                append = writer_module._append_jsonl

                def delayed_append(path, values):
                    started.set()
                    self.assertTrue(release.wait(5))
                    append(path, values)

                patch = mock.patch.object(writer_module, '_append_jsonl', delayed_append) if kind == 'direct' else nullcontext()
                with patch:
                    writer.write(note=payload)
                    if kind == 'direct':
                        self.assertTrue(started.wait(5))
                    payload['nested']['values'][1]['x'] = 99
                    payload['nested']['values'].append(object())
                    release.set()
                    writer.end_step()
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(root) as reader:
                    self.assertEqual(reader.metric_values('note').tolist(), [{'initial': True}, expected])

    def test_pending_step_validated_before_every_compacting_publication(self):
        for invalid, error in ((-1, ValueError), (2**32, ValueError), (1.5, TypeError), (True, TypeError), (np.bool_(False), TypeError)):
            for operation in ('flush', 'end_step', 'context_exit', 'finish'):
                with self.subTest(invalid=invalid, operation=operation), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    writer = _writer('compact', root)
                    writer.write(counter=1, flush=True)
                    before = _snapshot(root)
                    writer.step = invalid
                    calls = {'flush': lambda: writer.write(flush=True), 'end_step': writer.end_step,
                             'context_exit': lambda: writer.__exit__(None, None, None),
                             'finish': lambda: writer.finish(optimize=False, zip=False)}
                    with self.assertRaises(error):
                        calls[operation]()
                    self.assertEqual(_snapshot(root), before)
                    self.assertEqual(writer._current_row, {'counter': 1})
                    self.assertEqual(writer.step, invalid)
                    writer.step = 0
                    writer.finish(optimize=False, zip=False)
                    with plattli.Reader(root) as reader:
                        self.assertEqual(reader.metric_indices('counter').tolist(), [0])
