import zipfile
from unittest import mock

import numpy as np
import pytest

from plattli import BulkWriter, CompactingWriter, DirectWriter, Reader
from plattli._values import _sequence_array, _value_sequence


def writer(kind, path, **kwargs):
    if kind == 'bulk':
        return BulkWriter(path, config={}, **kwargs)
    if kind == 'direct':
        return DirectWriter(path, config={}, write_threads=0, **kwargs)
    return CompactingWriter(path, config={}, hotsize=1 if kind == 'cold' else 100, **kwargs)


@pytest.mark.parametrize('kind', ['bulk', 'direct', 'cold', 'hot'])
@pytest.mark.parametrize('zipped', [False, True])
@pytest.mark.parametrize('optimize', [False, True])
def test_irregular_i128_roundtrip_promotion_and_selection(tmp_path, kind, zipped, optimize):
    values = [0, 1, 3, 10**23 + 2, 10**23 + 9, -(2**127), 2**127 - 1]
    w = writer(kind, tmp_path)
    for row, value in enumerate(values):
        w.step = row * 3
        w.write(flops_6nd=value, loss=float(row + 1))
        w.end_step()
    w.finish(optimize=optimize, zip=zipped)

    expected_bytes = b''.join(value.to_bytes(16, 'little', signed=True) for value in values)
    if zipped:
        with zipfile.ZipFile(tmp_path / 'metrics.plattli') as archive:
            assert archive.read('flops_6nd.i128') == expected_bytes
    else:
        assert (tmp_path / 'plattli/flops_6nd.i128').read_bytes() == expected_bytes
    with Reader(tmp_path) as r:
        assert r.manifest()['flops_6nd']['dtype'] == 'i128'
        assert 'values' not in r.manifest()['flops_6nd']
        for dtype in ['standard', 'storage', object]:
            actual = r.metric_values('flops_6nd', dtype=dtype)
            assert actual.dtype == object and actual.tolist() == values
            assert all(type(value) is int for value in actual)
        assert r.metric('flops_6nd', idx=-1)[1] == values[-1]
        assert r.metric_values('flops_6nd', start=9, stop=12).tolist() == values[3:5]
        assert r.metric_values('flops_6nd', istart=3, istop=5).tolist() == values[3:5]
        assert r.metric_values('flops_6nd', vstart=10**23, vstop=10**23 + 10).tolist() == values[3:5]
        assert r.metric_values('flops_6nd', istart=0, istop=3, dtype=np.int64).tolist() == values[:3]
        assert r.metric_values('flops_6nd', istart=0, istop=0).dtype == object
        with pytest.raises(OverflowError):
            r.metric_values('flops_6nd', dtype=np.int64)
        steps, cols = r.table(['flops_6nd', 'loss'], on='flops_6nd', vstart=10**23, vstop=10**23 + 10)
        assert steps.tolist() == [9, 12]
        assert cols['flops_6nd'].tolist() == values[3:5]
        assert cols['loss'].tolist() == [4., 5.]


@pytest.mark.parametrize('kind', ['bulk', 'direct', 'cold', 'hot'])
def test_step_sequence_can_break_and_remain_exact(tmp_path, kind):
    w = writer(kind, tmp_path)
    steps = [0, 100, 200, 251, 300, 310]
    values = [10**23 + 2 + step * 100 for step in steps]
    values[-2:] = [10**23 + 30003, -(10**23 + 7)]
    for step, value in zip(steps, values):
        w.step = step
        w.write(counter=value)
        w.end_step()
    w.finish(zip=False)
    with Reader(tmp_path) as r:
        assert r.metric_indices('counter').tolist() == steps
        assert r.metric_values('counter').tolist() == values
        assert 'values' not in r.manifest()['counter']


@pytest.mark.parametrize('kind', ['direct', 'cold', 'hot'])
@pytest.mark.parametrize('zipped', [False, True])
@pytest.mark.parametrize('cutoff', [0, 3, 6])
def test_i128_resume_and_rewind(tmp_path, kind, zipped, cutoff):
    values = [10**23 + 2, 10**23 + 4, 10**23 + 7, -10**23, 2**127 - 1, -(2**127)]
    with writer(kind, tmp_path) as w:
        for value in values:
            w.write(counter=value)
            w.end_step()
    if zipped:
        w.finish(zip=True)
    with Reader(tmp_path) as r:
        assert r.metric_values('counter').tolist() == values
    resumed = writer(kind, tmp_path, step=cutoff, allow_resume_finalized=zipped)
    resumed.write(counter=10**23 + 17)
    resumed.end_step()
    resumed.write(counter=-(10**23 + 27))
    resumed.finish(zip=zipped)
    with Reader(tmp_path) as r:
        assert r.metric_values('counter').tolist() == values[:cutoff] + [10**23 + 17, -(10**23 + 27)]


@pytest.mark.parametrize('kind', ['bulk', 'direct', 'cold', 'hot'])
@pytest.mark.parametrize('overflow', [-(2**127) - 1, 2**127])
def test_overflow_does_not_create_or_promote_columns(tmp_path, kind, overflow):
    w = writer(kind, tmp_path)
    w.write(counter=1)
    w.end_step()
    with pytest.raises(OverflowError, match='out of range for i128'):
        w.write(new_metric=42, counter=overflow)
    w.write(counter=10**23 + 2)
    w.finish(optimize=False, zip=False)
    with Reader(tmp_path) as r:
        assert r.metrics() == ['counter']
        assert r.metric_values('counter').tolist() == [1, 10**23 + 2]


@pytest.mark.parametrize('zipped', [False, True])
def test_raw_i128_slices_only_read_requested_bytes(tmp_path, zipped):
    w = writer('bulk', tmp_path)
    values = [10**23 + row * row for row in range(100)]
    for value in values:
        w.write(counter=value)
        w.end_step()
    w.finish(zip=zipped)
    with Reader(tmp_path) as r:
        assert r.metric_values('counter', vstart=values[40], vstop=values[42]).tolist() == values[40:43]
        with mock.patch.object(r, '_columnar_values', side_effect=AssertionError('full column read')):
            if not zipped:
                assert r.metric_values('counter', vstart=values[40], vstop=values[42]).tolist() == values[40:43]
            assert r.metric('counter', idx=50)[1] == values[50]
        if zipped:
            with mock.patch.object(r, '_read_zip_slice', wraps=r._read_zip_slice) as reads:
                assert r.metric_values('counter', istart=50, istop=52).tolist() == values[50:52]
            assert mock.call('counter.i128', 50 * 16, 2 * 16) in reads.call_args_list


def test_incomplete_i128_tail_is_ignored_live_and_removed_on_resume(tmp_path):
    with writer('direct', tmp_path) as w:
        for value in [10**23 + 2, 10**23 + 4, 10**23 + 7]:
            w.write(counter=value)
            w.end_step()
    path = tmp_path / 'plattli/counter.i128'
    with path.open('ab') as stream:
        stream.write(b'partial')
    with Reader(tmp_path) as r:
        assert r.metric_values('counter').tolist() == [10**23 + 2, 10**23 + 4, 10**23 + 7]
    resumed = writer('direct', tmp_path, step=3)
    resumed.write(counter=10**23 + 12)
    resumed.finish(zip=False)
    assert path.stat().st_size == 4 * 16
    with Reader(tmp_path) as r:
        assert r.metric_values('counter')[-1] == 10**23 + 12


def test_sequence_coefficients_may_exceed_i128_but_values_may_not():
    spec = {'dtype': 'i128', 'values': {'start': -(2**127), 'step': 2**128 - 1, 'count': 2}}
    assert _sequence_array(spec).tolist() == [-(2**127), 2**127 - 1]
    with pytest.raises(ValueError, match='exceeds i128'):
        _value_sequence({**spec, 'values': {**spec['values'], 'count': 3}})
    spec = {'dtype': 'i128', 'values': {'start': -(2**128), 'step': 2**127, 'count': 2, 'on': 'step'}}
    assert _sequence_array(spec, indices=[1, 2]).tolist() == [-(2**127), 0]
    with pytest.raises(ValueError, match='exceed i128'):
        _sequence_array(spec, indices=[2, 3])


@pytest.mark.parametrize('kind', ['bulk', 'direct', 'cold', 'hot'])
def test_optimization_does_not_make_python_big_ints_unreadable(tmp_path, kind):
    w = writer(kind, tmp_path)
    values = [2**63 + 2, 2**63 + 4, 2**63 + 7]
    for value in values:
        w.write(counter=value)
        w.end_step()
    w.finish(zip=False)
    with Reader(tmp_path) as r:
        assert r.manifest()['counter']['dtype'] == 'u64'
        assert r.metric_values('counter').tolist() == values


@pytest.mark.parametrize('zipped', [False, True])
def test_float64_conversion_and_exact_reads(tmp_path, zipped):
    values = [10**23 + 2, -(10**23 + 7), -1, -(2**127), 2**127 - 1]
    w = writer('bulk', tmp_path)
    for value in values:
        w.write(counter=value, loss=1.)
        w.end_step()
    w.finish(optimize=False, zip=zipped)
    expected = np.asarray(values, dtype=np.float64)
    with Reader(tmp_path) as r:
        np.testing.assert_array_equal(r.metric_values('counter', dtype='float64'), expected)
        np.testing.assert_array_equal(r.metric('counter', dtype=float)[1], expected)
        assert r.metric('counter', idx=2, dtype=float)[1] == -1.
        np.testing.assert_array_equal(r.metric_values('counter', istart=1, istop=3, dtype=float), expected[1:3])
        for on in ['step', 'counter']:
            _, columns = r.table(['counter', 'loss'], on=on, dtype={'counter': np.float64})
            np.testing.assert_array_equal(columns['counter'], expected)
            assert columns['loss'].dtype == np.float32
        assert r.metric_values('counter').tolist() == values
        assert r.table(['counter', 'loss'])[1]['counter'].tolist() == values


@pytest.mark.parametrize('monotonic', [False, True])
@pytest.mark.parametrize('zipped', [False, True])
def test_float64_reads_filter_before_rounding(tmp_path, monotonic, zipped):
    values = [10**23 + offset for offset in ([2, 4, 7, 9] if monotonic else [7, 2, 9, 4])]
    w = writer('bulk', tmp_path)
    for value in values:
        w.write(counter=value, loss=1.)
        w.end_step()
    w.finish(zip=zipped)
    with Reader(tmp_path) as r:
        expected_steps = [i for i, value in enumerate(values) if 10**23 + 4 <= value <= 10**23 + 7]
        for table in [False, True]:
            args = dict(vstart=10**23 + 4, vstop=10**23 + 7, dtype=np.float64)
            steps, actual = (r.table(['counter', 'loss'], on='counter', **args) if table else r.metric('counter', **args))
            assert steps.tolist() == expected_steps
            actual = actual['counter'] if table else actual
            assert actual.dtype == np.float64
        assert len(r.metric_values('counter', vstart=10**23 + 4, vstop=10**23 + 7, dtype=float)) == 2


def test_float64_reads_merge_raw_and_hot_values(tmp_path):
    import json
    w = writer('direct', tmp_path)
    values = [10**23 + 2, 10**23 + 4, 10**23 + 7]
    for value in values:
        w.write(counter=value)
        w.end_step()
    (tmp_path / 'plattli/hot.jsonl').write_text(json.dumps({'step': 3, 'counter': -1}) + '\n')
    with Reader(tmp_path) as r:
        for read in [lambda: r.metric_values('counter', dtype=float), lambda: r.metric('counter', dtype=float)[1],
                     lambda: r.metric_values('counter', start=0, stop=3, dtype=float)]:
            actual = read()
            assert actual.dtype == np.float64
            np.testing.assert_array_equal(actual, np.asarray(values + [-1], dtype=float))
        assert r.metric_values('counter').tolist() == values + [-1]


@pytest.mark.parametrize('kind', ['bulk', 'direct', 'cold', 'hot'])
@pytest.mark.parametrize('numpy_input', [False, True])
def test_python_and_numpy_wide_integers_read_the_same(tmp_path, kind, numpy_input):
    values = [1, 5, 2, 2**63]
    w = writer(kind, tmp_path)
    for value in values:
        w.write(counter=np.uint64(value) if numpy_input else value)
        w.end_step()
    w.finish(zip=False)
    with Reader(tmp_path) as r:
        assert r.manifest()['counter']['dtype'] == 'u64'
        assert r.metric_values('counter').dtype == object
        assert r.metric_values('counter').tolist() == values
        with pytest.raises(OverflowError):
            r.metric_values('counter', dtype=np.int64)


@pytest.mark.parametrize('zipped', [False, True])
@pytest.mark.parametrize('dtype', ['i128', 'i64', 'f32'])
def test_encoded_numeric_reads_skip_decoding_and_bound_to_indices(tmp_path, zipped, dtype):
    import json
    from plattli import encode_numeric

    values = [10**23 + 2, -(10**23 + 7), -1] if dtype == 'i128' else [2, 7, -1]
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'plattli.json').write_text(json.dumps({'counter': {'dtype': dtype, 'indices': 'indices'}}))
    np.asarray([2, 5, 9], dtype=np.uint32).tofile(tmp_path / 'counter.indices')
    raw = encode_numeric(values, dtype)
    (tmp_path / f'counter.{dtype}').write_bytes(raw + encode_numeric([13], dtype) + b'x')
    if zipped:
        with zipfile.ZipFile(tmp_path / 'metrics.plattli', 'w') as archive:
            for name in ['config.json', 'plattli.json', 'counter.indices', f'counter.{dtype}']:
                archive.write(tmp_path / name, name)
    with Reader(tmp_path / 'metrics.plattli' if zipped else tmp_path) as r:
        with mock.patch('plattli.reader._decode_numeric', side_effect=AssertionError('decoded raw values')):
            assert r.metric_value_bytes('counter') == (dtype, raw)


@pytest.mark.parametrize('sequence', [False, True])
def test_encoded_numeric_reads_include_hot_tail(tmp_path, sequence):
    import json
    from plattli import encode_numeric

    values = [10**23 + offset for offset in ([2, 4, 6] if sequence else [2, 4, 7])]
    with writer('direct', tmp_path) as w:
        for value in values:
            w.write(counter=value)
            w.end_step()
    (tmp_path / 'plattli/hot.jsonl').write_text(json.dumps({'step': 3, 'counter': -1}) + '\n')
    with Reader(tmp_path) as r:
        assert r.metric_value_bytes('counter') == ('i128', encode_numeric(values + [-1], 'i128'))
    with Reader(tmp_path, exclude_hot=True) as r:
        assert r.metric_value_bytes('counter') == ('i128', encode_numeric(values, 'i128'))


@pytest.mark.parametrize('kind', ['bulk', 'direct', 'cold', 'hot'])
def test_integer_range_check_applies_to_the_chosen_storage_dtype(tmp_path, kind):
    w = writer(kind, tmp_path)
    w.write(x=1.5)
    w.end_step()
    w.write(x=2**130)
    w.end_step()
    with pytest.raises(OverflowError, match='exactly'):
        w.write(x=2**130 + 1)
    with pytest.raises(OverflowError, match='out of range for i128'):
        w.write(integer_column=2**130)
    w.finish(zip=False)
    with Reader(tmp_path) as r:
        assert r.manifest()['x']['dtype'] == 'f64'
        assert r.metric_values('x').tolist() == [1.5, float(2**130)]
        assert 'integer_column' not in r.metrics()
