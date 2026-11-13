import json
import math
import operator
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from shutil import rmtree

import numpy as np

from ._values import _value_sequence, _sequence_array, _extend_value_sequence, _find_value_sequence
from ._affine import _find_step_sequence

from ._indices import (
    _append_step_to_indices_spec,
    _append_steps_to_indices_spec,
    _piecewise_segments,
    _segments_close_open_tail,
    _segments_count_and_last,
    _segments_from_spec,
    _segments_have_open_tail,
    _segments_to_array,
    _segments_too_many,
    _segments_truncate,
    _segments_with_counts,
)
from ._ranges import (
    _ranges_append,
    _ranges_encode,
    _ranges_parse,
    _ranges_records,
    _ranges_to_segments,
)

from ._numeric import (
    DTYPE_TO_NUMPY, DTYPE_ITEMSIZE, INT_LIMITS as _INT_LIMITS,
    _encode_numeric, _read_numeric, _numeric_dtype,
)

_F32_MAX = float(np.finfo(np.float32).max)


def _value_dtype(dtype):
    return DTYPE_TO_NUMPY[dtype]

JSONL_DTYPE = "jsonl"
HOT_FILENAME = "hot.jsonl"
HOT_COMPACTING_FILENAME = "hot.compacting.jsonl"
_RESERVED_METRIC_NAMES = frozenset({"step", "run_rows", "when_exported", "hot", "hot.compacting"})


def _validate_metric_name(name, run_name):
    if not isinstance(name, str):
        raise TypeError(f"metric names must be strings in run {run_name} (got {type(name).__name__})")
    name.encode("utf-8")  # Reject invalid filenames before any schema/data is published.
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"invalid metric name {name!r} in run {run_name}")
    if name in _RESERVED_METRIC_NAMES:
        raise ValueError(f"metric name {name!r} is reserved in run {run_name}")
    if PurePosixPath(name).is_absolute():
        raise ValueError(f"absolute metric path {name!r} is not allowed in run {run_name}")
    windows_name = PureWindowsPath(name)
    if windows_name.is_absolute() or windows_name.drive:
        raise ValueError(f"absolute metric path {name!r} is not allowed in run {run_name}")
    if any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"invalid metric path {name!r} in run {run_name}")
    return name


def _validate_metric_names(names, run_name):
    for name in names:
        _validate_metric_name(name, run_name)


def _coerce_step(step, run_name):
    if type(step) is int:
        return step
    if isinstance(step, (bool, np.bool_)):
        raise TypeError(f"step must be an integer for run {run_name}, not bool")
    try:
        return operator.index(step)
    except TypeError as exc:
        raise TypeError(f"step must be an integer for run {run_name}: {step!r}") from exc


def _validate_step(step, run_name):
    step = _coerce_step(step, run_name)
    # TODO: auto-promote step storage and returned indices to uint64.
    if step < 0 or step > 0xFFFFFFFF:
        raise ValueError(f"step out of uint32 range for run {run_name}: {step}")
    return step


def _archive_member_path(root, name, run_name):
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise RuntimeError(f"invalid archive member {name!r} in run {run_name}")
    member_name = name[:-1] if name.endswith("/") else name
    if not member_name or PurePosixPath(member_name).is_absolute():
        raise RuntimeError(f"unsafe archive member {name!r} in run {run_name}")
    windows_name = PureWindowsPath(member_name)
    if windows_name.is_absolute() or windows_name.drive:
        raise RuntimeError(f"unsafe archive member {name!r} in run {run_name}")
    if any(part in ("", ".", "..") for part in member_name.split("/")):
        raise RuntimeError(f"unsafe archive member {name!r} in run {run_name}")
    root = Path(root).resolve()
    out_path = (root / member_name).resolve()
    try:
        out_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"unsafe archive member {name!r} in run {run_name}") from exc
    return out_path


def _zip_path_for_root(root):
    return Path(root) / "metrics.plattli"


def _run_name_for_root(root):
    root = Path(root)
    if root.name == "plattli":
        return root.parent.name
    return root.name


def _manifest_payload(manifest, run_rows=None, run_name=None):
    _validate_metric_names(manifest, run_name or "manifest")
    payload = {
        **manifest,
        "when_exported": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    if run_rows is not None:
        payload["run_rows"] = run_rows
    return json.dumps(payload, ensure_ascii=False)


def _write_manifest(path, manifest, run_rows=None):
    path = Path(path)
    _replace_text_checked(path, _manifest_payload(manifest, run_rows, _run_name_for_root(path.parent)), "manifest")


def _check_temp_source(path, source):
    temporary = path.parent.resolve() / (path.name + ".tmp")
    if source.resolve() == temporary or source.parent.resolve() / source.name == temporary:
        raise ValueError(f"source uses the writer's reserved temporary path: {temporary}")


def _replace_text_checked(path, payload, label):
    if not payload:
        raise RuntimeError(f"{label} payload must not be empty for {path}")
    path = Path(path)
    if path.is_symlink():
        _check_temp_source(path, path.parent / os.readlink(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.is_symlink():
        tmp_path.unlink()  # A failed config-link replacement must not redirect this write.
    with tmp_path.open("w", encoding="utf-8") as fh:  # Do this whole dance to be atomic on NFS.
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    if tmp_path.stat().st_size <= 0:
        raise RuntimeError(f"{label} tmp file is empty after write: {tmp_path}")
    tmp_path.replace(path)
    _fsync_dir(path.parent)  # Again, for NFS-atomicity.
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{label} file is empty after replace: {path}")


def _replace_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        temporary.unlink()
    with temporary.open("wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    temporary.replace(path)
    _fsync_dir(path.parent)


def _append_range_bytes(path, payload):
    if payload:
        with path.open("ab") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())


def _choose_indices(segments, count, *, live=False):
    if not _segments_too_many(segments, total_count=count):
        return segments
    if live:
        return "ranges" if count >= 32 * len(segments) else "indices"
    return "ranges" if 16 * len(segments) < 4 * count else "indices"


def _store_indices(root, name, segments, count, *, open_tail=False):
    selected = _choose_indices(segments, count, live=True)
    if selected == "ranges":
        _replace_bytes(root / f"{name}.ranges", _ranges_encode(segments, total_count=count, open_tail=open_tail))
    elif selected == "indices":
        _replace_bytes(root / f"{name}.indices", _segments_to_array(segments, total_count=count).tobytes())
    return selected


def _extend_inline_indices(segments, steps, count):
    # Only the old tail and first two incoming points need incremental inference.
    # Compress the rest in one pass instead of rescanning growing history per row.
    segments = _append_steps_to_indices_spec(segments, steps[:2], count=count, open_tail=True)
    count += min(2, len(steps))
    if not segments:
        return segments
    segments[-1].pop("stop", None)
    remaining = np.asarray(steps[2:], dtype=np.uint64)
    if len(remaining):
        closed, _ = _segments_count_and_last(segments[:-1])
        tail = segments[-1]
        expected = tail["start"] + (count - closed + np.arange(len(remaining), dtype=np.uint64)) * tail["step"]
        mismatch = np.flatnonzero(remaining != expected)
        if len(mismatch):
            matched = int(mismatch[0])
            tail["stop"] = tail["start"] + (count - closed + matched) * tail["step"]
            segments.extend(_piecewise_segments(remaining[matched:].astype(np.uint32)))
            segments[-1].pop("stop")
    return segments


def _coalesce_segments(segments):
    merged = []
    for segment in segments:
        start, stop, step = segment["start"], segment["stop"], segment["step"]
        count = (stop - start + step - 1) // step
        if merged and merged[-1]["step"] == step and merged[-1]["stop"] == start:
            merged[-1]["stop"] = start + count * step
        else:
            merged.append({"start": start, "stop": start + count * step, "step": step})
    return merged


def _fsync_dir(path):
    if not hasattr(os, "O_DIRECTORY"):
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_config(run_root, path, config):
    run_name = _run_name_for_root(run_root)
    if config is None:
        if path.exists() or path.is_symlink():
            return
        config = {}
    if isinstance(config, str):
        target = (run_root / config).expanduser()
        if target.exists():
            if not target.is_file():
                raise FileNotFoundError(f"config target is not a file: {target} (run {run_name})")
            target = target.resolve()
            if target == path.resolve():
                return
            tmp_path = path.with_name(path.name + ".tmp")
            if tmp_path.exists() or tmp_path.is_symlink():
                tmp_path.unlink()
            tmp_path.symlink_to(target)
            tmp_path.replace(path)
            _fsync_dir(path.parent)
            return
        if path.exists() or path.is_symlink():
            return
        config = {}
    # Replace the link itself when switching from an external config to a value.
    _replace_text_checked(path, json.dumps(config, ensure_ascii=False), "config")


def _prepare_config(run_root, config):
    path = run_root / "plattli" / "config.json"
    if path.is_symlink():
        _check_temp_source(path, path.parent / os.readlink(path))
    if config is None:
        return None
    if isinstance(config, str):
        target = (run_root / config).expanduser()
        _check_temp_source(path, target)
        if (target.exists() and not target.is_file()) or target.resolve() == (run_root / "plattli").resolve():
            raise FileNotFoundError(f"config target is not a file: {target} (run {_run_name_for_root(run_root)})")
        return config
    payload = json.dumps(config, ensure_ascii=False)
    payload.encode("utf-8")  # JSON permits lone surrogates; UTF-8 files do not.
    return json.loads(payload)


def _config_payload_with_overrides(path, config_overrides, run_name):
    if config_overrides is None:
        return None
    if not isinstance(config_overrides, dict):
        raise TypeError(f"config_overrides must be a dict in run {run_name}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"config must be a JSON object to apply overrides in run {run_name}")
    payload = json.dumps({**config, **config_overrides}, ensure_ascii=False)
    payload.encode("utf-8")  # Validate file encoding before finalization mutates data.
    return payload


def _truncate_to_step(root, manifest, step, allow_missing):
    run_name = _run_name_for_root(root)
    for name, spec in manifest.items():
        if spec.get("values") == "sequence":
            spec["values"] = _stored_value_sequence(root, name, spec)
        dtype = spec["dtype"]
        indices_spec = spec["indices"]
        if indices_spec == "indices":
            idx_path = root / f"{name}.indices"
            if not idx_path.exists():
                if allow_missing:
                    continue
                raise FileNotFoundError(f"missing indices file for {name} in run {run_name}")
            indices = np.fromfile(idx_path, dtype=np.uint32)
            keep = int(np.searchsorted(indices, step, side="left")) if indices.size else 0
            if "values" in spec:
                keep = min(keep, _value_sequence(spec)["count"])
            with idx_path.open("r+b") as fh:
                fh.truncate(keep * 4)
        elif indices_spec == "ranges":
            path = root / f"{name}.ranges"
            if not path.exists():
                if allow_missing:
                    continue
                raise FileNotFoundError(f"missing ranges file for {name} in run {run_name}")
            original_ranges = path.read_bytes()
            stored = _stored_values_count(root, name, spec, allow_missing)
            segments = _ranges_to_segments(original_ranges, stored)
            kept, keep = _segments_truncate(segments, step)
        else:
            try:
                segments = _segments_from_spec(indices_spec)
                total_count = _stored_values_count(root, name, spec, allow_missing) if _segments_have_open_tail(segments) else None
                kept, keep = _segments_truncate(segments, step, total_count=total_count)
            except (ValueError, RuntimeError) as exc:
                raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
            spec["indices"] = kept

        if "values" in spec:
            spec["values"] = {**_value_sequence(spec), "count": min(keep, spec["values"]["count"])}
        elif dtype == JSONL_DTYPE:
            path = root / f"{name}.jsonl"
            if not path.exists():
                if allow_missing:
                    continue
                raise FileNotFoundError(f"missing values file for {name} in run {run_name}")
            offset = 0
            with path.open("rb") as fh:  # Truncate in place (atomic), like the numeric branch.
                for count, line in enumerate(fh):
                    if count >= keep:
                        break
                    offset += len(line)
            if path.stat().st_size != offset:
                with path.open("r+b") as fh:
                    fh.truncate(offset)
                    if indices_spec == "ranges":
                        fh.flush()
                        os.fsync(fh.fileno())
        else:
            value_path = root / f"{name}.{dtype}"
            if not value_path.exists():
                if allow_missing:
                    continue
                raise FileNotFoundError(f"missing values file for {name} in run {run_name}")
            size = keep * DTYPE_ITEMSIZE[dtype]
            if value_path.stat().st_size != size:
                with value_path.open("r+b") as fh:
                    fh.truncate(size)
                    if indices_spec == "ranges":
                        fh.flush()
                        os.fsync(fh.fileno())
        if indices_spec == "ranges":
            # Preserve an open tail only when no values were cut. Never reopen a
            # closed tail: hot recovery can truncate twice, and the second pass
            # must retain the first pass's bound for readers holding an old count.
            data = _ranges_encode(kept, total_count=keep,
                                  open_tail=keep == stored and _ranges_parse(original_ranges)[1] is not None)
            if data != original_ranges:
                _replace_bytes(root / f"{name}.ranges", data)
    _write_manifest(root / "plattli.json", manifest)


def _force_indices_files(root, manifest):
    run_name = _run_name_for_root(root)
    updated = False
    retired = []
    for name, spec in manifest.items():
        indices_spec = spec.get("indices")
        if indices_spec == "indices":
            continue
        try:
            if indices_spec == "ranges":
                segments = _ranges_to_segments((root / f"{name}.ranges").read_bytes(),
                                               _stored_values_count(root, name, spec, allow_missing=False))
                total_count = None
            else:
                segments = _segments_from_spec(indices_spec)
                total_count = _stored_values_count(root, name, spec, allow_missing=False) if _segments_have_open_tail(segments) else None
            indices = _segments_to_array(segments, total_count=total_count)
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
        idx_path = root / f"{name}.indices"
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        _replace_bytes(idx_path, indices.tobytes())
        if indices_spec == "ranges":
            retired.append(root / f"{name}.ranges")
        spec["indices"] = "indices"
        updated = True
    if updated:
        _write_manifest(root / "plattli.json", manifest)
        for path in retired:
            path.unlink()
            _fsync_dir(path.parent)


def _migrate_oversized_piecewise_indices(root, manifest):
    updated = False
    for name, spec in manifest.items():
        if spec["indices"] in ("indices", "ranges"):
            continue
        segments = _segments_from_spec(spec["indices"])
        if not segments:
            continue
        count = _stored_values_count(root, name, spec, allow_missing=False)
        if _segments_too_many(segments, total_count=count):
            spec["indices"] = _store_indices(root, name, segments, count, open_tail=True)
            updated = True
    return updated


def _plan_indices(manifest, read_bytes, values_count, run_name):
    replacements = {}
    retired = set()
    changed = 0
    metrics = {name: spec for name, spec in manifest.items() if name not in ("run_rows", "when_exported")}
    _validate_metric_names(metrics, run_name)
    for name, spec in metrics.items():
        original = spec["indices"]
        try:
            count = values_count(name, spec)
        except KeyError as exc:
            raise FileNotFoundError(f"missing values file for {name} in run {run_name}") from exc
        data = None
        if original in ("indices", "ranges"):
            try:
                data = read_bytes(f"{name}.{original}")
            except KeyError as exc:
                raise FileNotFoundError(f"missing {original} file for {name} in run {run_name}") from exc
        if original == "indices":
            indices = np.frombuffer(data, dtype="<u4", count=len(data) // 4)[:count]
            segments = _piecewise_segments(indices)
            if segments is None:
                raise ValueError(f"indices out of order for {name} in run {run_name}")
        elif original == "ranges":
            segments = _ranges_to_segments(data, count)
        else:
            segments = []
            offset = 0
            for start, size, step, _ in _segments_with_counts(_segments_from_spec(original), total_count=count):
                size = min(size, max(count - offset, 0))
                if size:
                    segments.append({"start": start, "stop": start + size * step, "step": step})
                    offset += size
        segments = _coalesce_segments(segments)
        count, _ = _segments_count_and_last(segments)
        selected = _choose_indices(segments, count)
        payload = None
        if selected == "ranges":
            payload = _ranges_encode(segments, total_count=count)
        elif selected == "indices":
            payload = _segments_to_array(segments).astype("<u4", copy=False).tobytes()
        if selected == original and (payload is None or payload == data):
            continue
        changed += 1
        spec["indices"] = selected
        if payload is not None:
            replacements[f"{name}.{selected}"] = payload
        if original in ("indices", "ranges") and selected != original:
            retired.add(f"{name}.{original}")
    return replacements, retired, changed


def _optimize_indices(root, manifest):
    planned = {name: dict(spec) for name, spec in manifest.items()}
    replacements, retired, changed = _plan_indices(
        planned, lambda name: (root / name).read_bytes(),
        lambda name, spec: _stored_values_count(root, name, spec, allow_missing=False),
        _run_name_for_root(root),
    )
    if not changed:
        return
    for name, data in replacements.items():
        _replace_bytes(root / name, data)
    _write_manifest(root / "plattli.json", planned)
    manifest.update(planned)
    directories = set()
    for name in retired:
        path = root / name
        path.unlink()
        directories.add(path.parent)
    for directory in directories:
        _fsync_dir(directory)


def _indices_length(root, name, spec):
    indices_spec = spec["indices"]
    if isinstance(indices_spec, (dict, list)):
        try:
            segments = _segments_from_spec(indices_spec)
            total_count = _stored_values_count(root, name, spec, allow_missing=False) if _segments_have_open_tail(segments) else None
            total, _ = _segments_count_and_last(segments, total_count=total_count)
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f"{exc} (metric {name}, run {_run_name_for_root(root)})") from exc
        return total
    if indices_spec == "indices":
        idx_path = root / f"{name}.indices"
        return idx_path.stat().st_size // 4
    if indices_spec == "ranges":
        count = _stored_values_count(root, name, spec, allow_missing=False)
        records = _ranges_records((root / f"{name}.ranges").read_bytes(), count)
        return min(count, int(records[-1]["end_row"])) if len(records) else 0
    return 0  # pragma: no cover


def _stored_values_count(root, name, spec, allow_missing):
    if spec.get("values") == "sequence":
        return _stored_value_sequence(root, name, spec)["count"]
    sequence = _value_sequence(spec)
    if sequence is not None:
        return sequence["count"]
    dtype = spec["dtype"]
    if dtype == JSONL_DTYPE:
        path = root / f"{name}.jsonl"
        if not path.exists():
            if allow_missing:
                return 0
            raise FileNotFoundError(f"missing values file for {name} in run {_run_name_for_root(root)}")
        count = 0
        lines = path.read_bytes().splitlines()
        for idx, line in enumerate(lines):
            try:
                json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if idx == len(lines) - 1:
                    break
                raise
            count += 1
        return count
    path = root / f"{name}.{dtype}"
    if not path.exists():
        if allow_missing:
            return 0
        raise FileNotFoundError(f"missing values file for {name} in run {_run_name_for_root(root)}")
    return path.stat().st_size // DTYPE_ITEMSIZE[dtype]


def _close_open_indices(root, manifest):
    changed = False
    for name, spec in manifest.items():
        indices_spec = spec["indices"]
        if indices_spec == "ranges":
            path = root / f"{name}.ranges"
            records, tail = _ranges_parse(path.read_bytes())
            if tail is not None:
                count = _stored_values_count(root, name, spec, allow_missing=False)
                closed = int(records[-1]["end_row"]) if len(records) else 0
                if count <= closed:
                    _replace_bytes(path, records.tobytes())
                else:
                    _append_range_bytes(path, np.asarray([count], dtype="<u8").tobytes())
                changed = True
            continue
        if not isinstance(indices_spec, (dict, list)):
            continue
        segments = _segments_from_spec(indices_spec)
        if not _segments_have_open_tail(segments):
            continue
        total_count = _stored_values_count(root, name, spec, allow_missing=False)
        spec["indices"] = _segments_close_open_tail(segments, total_count)
        changed = True
    return changed


def _merge_metrics(args, kwargs, run_name):
    if len(args) > 1:
        raise TypeError(f"write accepts at most one positional metrics dict in run {run_name}")
    metrics = args[0] if args else None
    if metrics is None:
        metrics = {}
    if not isinstance(metrics, dict):
        raise TypeError(f"metrics must be a dict in run {run_name} (got {type(metrics).__name__})")
    if kwargs:
        overlap = metrics.keys() & kwargs.keys()
        if overlap:
            raise ValueError(f"duplicate metric names in run {run_name}: {sorted(overlap)}")
        metrics = {**metrics, **kwargs}
    return metrics


def _coerce_stored_value(value, dtype, name, run_name):
    if type(value) not in (int, float, bool) and hasattr(value, "__array__"):
        value = np.asarray(value)
    if isinstance(value, (np.ndarray, np.generic)):
        if value.shape != ():
            raise ValueError(f"only scalar values are supported for {name} in run {run_name} (shape {value.shape})")
        value = value.item()
    if dtype == JSONL_DTYPE:
        # Validate the complete batch before any writer publishes data or schema.
        payload = json.dumps(value, ensure_ascii=False)
        payload.encode("utf-8")
        return value if value is None or type(value) in (str, int, float, bool) else json.loads(payload)
    if dtype in _INT_LIMITS:
        if not isinstance(value, (int, float)) or (isinstance(value, float) and not np.isfinite(value)):
            raise ValueError(f"expected an integer for {name} in run {run_name}")
        integer = int(value)
        if integer != value:
            raise ValueError(f"fractional value cannot be stored as {dtype} for {name} in run {run_name}")
        lo, hi = _INT_LIMITS[dtype]
        if not lo <= integer <= hi:
            raise OverflowError(f"value {integer} is out of range for {dtype} for {name} in run {run_name}")
        return integer
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected a numeric scalar for {name} in run {run_name}")
    # Python float -> f32 may round, but a finite value must not become infinity.
    if dtype == "f32" and abs(value) > _F32_MAX and math.isfinite(value):
        with np.errstate(over="ignore"):
            converted = np.float32(value).item()
        if math.isinf(converted):
            raise OverflowError(f"finite value is out of range for {dtype} for {name} in run {run_name}")
        return converted
    return DTYPE_TO_NUMPY[dtype](value).item()


def _prepare_metrics(metrics, manifest, step_metrics, step, run_name, integer_bounds):
    """Validate and snapshot the whole batch before publishing any changes."""
    prepared = {}
    for name, value in metrics.items():
        if name in step_metrics:
            raise RuntimeError(f"metric already written in step {step} for {name} in run {run_name}")
        if name not in manifest:
            dtype = _resolve_dtype(value, name=name, run_name=run_name)
        else:
            dtype = _reconcile_dtype(manifest[name]["dtype"], value, name, run_name,
                                     lambda name=name: integer_bounds(name))
        value = _coerce_stored_value(value, dtype, name, run_name)
        prepared[name] = dtype, value
    return prepared


def _reconcile_dtype(dtype, value, name, run_name, integer_bounds):
    """Use new scalar dtype evidence; recovered JSON rows must not call this."""
    if dtype == JSONL_DTYPE:
        return dtype
    incoming = _resolve_dtype(value, name=name, run_name=run_name)
    if dtype == "u64" and incoming == "i128" and 0 <= value <= _INT_LIMITS["u64"][1]:
        return "u64"
    if "i128" in (dtype, incoming) and dtype[0] in "iu" and incoming[0] in "iu":
        return "i128"
    if incoming == dtype:
        return dtype
    if incoming == JSONL_DTYPE and isinstance(value, (list, tuple)):
        raise ValueError(f"only scalar values are supported for {name} in run {run_name}")
    if incoming == JSONL_DTYPE or (dtype[0] != "f" and incoming[0] == "f"):
        raise TypeError(f"cannot mix {dtype} and {incoming} values for {name} in run {run_name}")
    if dtype[0] == "f":
        if incoming[0] in "iu":
            integer = int(value) if isinstance(value, (int, np.integer)) else int(np.asarray(value).item())
            # Equality between a NumPy float and int can itself round the int.
            # Convert back to a Python integer before testing exactness.
            candidate = float(integer)
            if dtype == "f32" and abs(candidate) <= _F32_MAX:
                candidate = float(np.float32(candidate))
                if int(candidate) == integer:
                    return dtype
            if int(float(integer)) == integer:
                return "f64"
            raise OverflowError(f"integer cannot be represented exactly as a float for {name} in run {run_name}")
        return "f64" if "f64" in (dtype, incoming) else "f32"
    if dtype[0] == incoming[0]:
        return dtype if int(dtype[1:]) >= int(incoming[1:]) else incoming

    # Unlike np.promote_types, signed/unsigned reconciliation must never use floats.
    unsigned = dtype if dtype[0] == "u" else incoming
    signed = dtype if dtype[0] == "i" else incoming
    if unsigned != "u64":
        bits = max(int(signed[1:]), int(unsigned[1:]) * 2)
        return f"i{bits}"
    scalar = int(value) if isinstance(value, (int, np.integer)) else int(np.asarray(value).item())
    if dtype == "i64" and scalar <= _INT_LIMITS["i64"][1]:
        return dtype
    lo, hi = integer_bounds()
    lo, hi = min(lo, scalar), max(hi, scalar)
    if hi <= _INT_LIMITS["i64"][1]:
        return "i64"
    if lo >= 0:
        return "u64"
    return "i128"


def _stored_value_sequence(root, name, spec):
    if spec.get("values") == "sequence":
        # Capture indices before the sequence head: newer coefficients may close
        # an old head, but an old head must never extrapolate over newer rows.
        count = (root / f"{name}.indices").stat().st_size // 4
        stored = json.loads((root / f"{name}.values").read_text())
        sequence = stored["values"]
        sequence = {**sequence, "count": min(count, sequence.get("count", count))}
        _value_sequence({"dtype": stored["dtype"], "values": sequence})
        # Promotion publishes the head before the manifest, and appends the new
        # index afterward. A crash between publications leaves only the retained
        # prefix, which must still fit the manifest's dtype. Recover at that dtype.
        return dict(_value_sequence({**spec, "values": sequence}))
    if "values" in spec:
        return dict(_value_sequence(spec))
    raise ValueError(f"missing value sequence for {name}")


def _stored_indices(root, name, spec, count):
    if spec["indices"] == "indices":
        return np.fromfile(root / f"{name}.indices", dtype=np.uint32, count=count)
    if spec["indices"] == "ranges":
        segments = _ranges_to_segments((root / f"{name}.ranges").read_bytes(), count)
    else:
        segments = _segments_from_spec(spec["indices"])
    return _segments_to_array(segments, total_count=count)[:count]


def _sequence_endpoints(root, name, spec):
    sequence = _value_sequence(spec)
    if not sequence["count"]:
        return [0]
    if sequence.get("on") == "step":
        if spec["indices"] == "indices":
            with (root / f"{name}.indices").open("rb") as source:
                first = int(np.frombuffer(source.read(4), dtype=np.uint32)[0])
                source.seek((sequence["count"] - 1) * 4)
                last = int(np.frombuffer(source.read(4), dtype=np.uint32)[0])
        else:
            indices = _stored_indices(root, name, spec, sequence["count"])
            first, last = int(indices[0]), int(indices[-1])
        return [sequence["start"] + first * sequence["step"], sequence["start"] + last * sequence["step"]]
    return [sequence["start"], sequence["start"] + (sequence["count"] - 1) * sequence["step"]]


def _append_stored_values(root, name, spec, values, indices=None):
    """Extend an inline sequence or publish its complete raw replacement."""
    sequence = _value_sequence(spec)
    if sequence is None:
        raise ValueError(f"missing value sequence for {name}")
    extended = _extend_value_sequence(sequence, values, spec["dtype"], indices=indices)
    previous_indices = None
    if extended is None and indices is not None and spec["dtype"][0] in "iu":
        previous_indices = _stored_indices(root, name, spec, sequence["count"])
        previous = _sequence_array(spec, indices=previous_indices)
        combined = np.concatenate((previous.astype(object), np.asarray(values, dtype=object)))
        extended = _find_value_sequence(combined, spec["dtype"]) or _find_step_sequence(
            np.concatenate((previous_indices, np.asarray(indices, dtype=np.uint32))), combined, spec["dtype"])
    if extended is not None:
        spec["values"] = extended
    else:
        if sequence.get("on") == "step" and previous_indices is None:
            previous_indices = _stored_indices(root, name, spec, sequence["count"])
        array = np.concatenate((_sequence_array(spec, indices=previous_indices), np.asarray(values, dtype=_value_dtype(spec["dtype"]))))
        _replace_bytes(root / f"{name}.{spec['dtype']}", _encode_numeric(array, spec["dtype"]))
        spec.pop("values")


def _optimize_values(root, manifest):
    retired = []
    for name, spec in manifest.items():
        if "values" in spec or spec["dtype"] == JSONL_DTYPE:
            continue
        path = root / f"{name}.{spec['dtype']}"
        values = _read_numeric(path, spec["dtype"])
        sequence = _find_value_sequence(values, spec["dtype"])
        if sequence is None and spec["dtype"][0] in "iu":
            sequence = _find_step_sequence(_stored_indices(root, name, spec, len(values)), values, spec["dtype"])
        if sequence is not None:
            spec["values"] = sequence
            retired.append(path)
    if retired:
        try:
            _write_manifest(root / "plattli.json", manifest)
        except BaseException:
            for path in retired:
                name = path.relative_to(root).as_posix().rsplit(".", 1)[0]
                manifest[name].pop("values")
            raise
        for path in retired:
            path.unlink()
            _fsync_dir(path.parent)


def _integer_bounds(root, name, spec, hot_rows=()):
    sequence = _value_sequence(spec)
    if sequence is not None:
        endpoints = _sequence_endpoints(root, name, spec)
        values = endpoints + [int(row[name]) for row in hot_rows if name in row]
        return min(values), max(values)
    dtype = spec["dtype"]
    path = root / f"{name}.{dtype}"
    lo = hi = 0
    if path.exists() and path.stat().st_size:
        values = _read_numeric(path, dtype)
        lo, hi = int(values.min()), int(values.max())
    for row in hot_rows:
        if name in row:
            value = int(row[name])
            lo, hi = min(lo, value), max(hi, value)
    return lo, hi


def _publish_stored_dtype(root, manifest, name, dtype, values, publish):
    """Publish a complete replacement before removing the old column."""
    spec = manifest[name]
    old_dtype = spec["dtype"]
    old_path = root / f"{name}.{old_dtype}"
    if values is not None:
        new_path = root / f"{name}.{dtype}"
        tmp_path = new_path.with_name(new_path.name + ".tmp")
        with tmp_path.open("wb") as fh:
            fh.write(_encode_numeric(values, dtype))
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(new_path)
        _fsync_dir(new_path.parent)
    spec["dtype"] = dtype
    try:
        publish()
    except BaseException:
        spec["dtype"] = old_dtype
        raise
    if old_path.exists():
        old_path.unlink()
        _fsync_dir(old_path.parent)


def _rewrite_stored_dtype(root, manifest, name, dtype, publish):
    spec = manifest[name]
    if "values" in spec:
        original = dict(spec)
        sequence = dict(_value_sequence(spec))
        if spec["dtype"][0] in "iu" and dtype[0] in "iu":
            lo, hi = _INT_LIMITS[dtype]
            endpoints = _sequence_endpoints(root, name, spec)
            if min(endpoints) < lo or max(endpoints) > hi:
                raise OverflowError(f"stored values are out of range for {dtype} for {name} in run {_run_name_for_root(root)}")
        if sequence.get("on") != "step" and dtype != "i128":
            sequence["start"] = _value_dtype(dtype)(sequence["start"]).item()
        _value_sequence({"dtype": dtype, "values": sequence})
        spec["dtype"] = dtype
        spec["values"] = sequence
        try:
            publish()
        except BaseException:
            spec.clear()
            spec.update(original)
            raise
        return
    old_path = root / f"{name}.{manifest[name]['dtype']}"
    values = None
    if old_path.exists():
        values = _read_numeric(old_path, manifest[name]["dtype"])
        target = np.dtype(DTYPE_TO_NUMPY[dtype])
        if spec["dtype"][0] in "iu" and dtype[0] in "iu":
            if values.size and not np.can_cast(values.dtype, target, casting="safe"):
                lo, hi = _INT_LIMITS[dtype]
                if int(values.min()) < lo or int(values.max()) > hi:
                    raise OverflowError(f"stored values are out of range for {dtype} for {name} in run {_run_name_for_root(root)}")
        elif values.dtype.kind != "f" or target.kind != "f" or target.itemsize < values.dtype.itemsize:
            raise TypeError(f"cannot rewrite {values.dtype} values as {dtype} for {name} in run {_run_name_for_root(root)}")
        values = values.astype(target, copy=False)
    _publish_stored_dtype(root, manifest, name, dtype, values, publish)


class _Monotonic:
    def __init__(self, dtype):
        self.last = None
        self.direction = None
        self.broken = dtype not in DTYPE_TO_NUMPY
        self.seen = False

    def add(self, value):
        if self.broken:
            return
        if value != value:
            self.broken = True
            self.direction = None
            return
        if not self.seen:
            self.last = value
            self.seen = True
            return
        if value == self.last:
            return
        direction = "inc" if value > self.last else "dec"
        if self.direction is None:
            self.direction = direction
        elif self.direction != direction:
            self.broken = True
            self.direction = None
        self.last = value

    def add_array(self, values):
        if self.broken or values.size == 0:
            return
        self.add(values[0].item() if isinstance(values[0], np.generic) else values[0])
        if self.broken or values.size == 1:
            return
        if (values != values).any():  # NaN anywhere breaks monotonicity.
            self.broken = True
            self.direction = None
            return
        inc = bool((values[1:] > values[:-1]).any())
        dec = bool((values[1:] < values[:-1]).any())
        if inc and dec:
            self.broken = True
            self.direction = None
            return
        if inc or dec:
            direction = "inc" if inc else "dec"
            if self.direction is None:
                self.direction = direction
            elif self.direction != direction:
                self.broken = True
                self.direction = None
                return
        self.last = values[-1].item() if isinstance(values[-1], np.generic) else values[-1]

    def apply(self, spec):
        direction = None if self.broken or not self.seen else self.direction or "inc"
        if direction is None:
            if "monotonic" not in spec:
                return False
            spec.pop("monotonic")
            return True
        if spec.get("monotonic") == direction:
            return False
        spec["monotonic"] = direction
        return True


def _set_monotonic(spec, values):
    state = _Monotonic(spec["dtype"])
    state.add_array(np.asarray(values))
    state.apply(spec)


def _refresh_monotonic_metadata(root, manifest, hot_rows=()):
    states = {}
    changed = False
    run_name = _run_name_for_root(root)
    sorted_hot_rows = sorted(hot_rows, key=lambda row: int(row["step"]))
    for name, spec in manifest.items():
        dtype = spec["dtype"]
        state = _Monotonic(dtype)
        if "values" in spec:
            sequence = _value_sequence(spec)
            if sequence["count"]:
                for value in _sequence_endpoints(root, name, spec):
                    state.add(value)
            for row in sorted_hot_rows:
                if name in row:
                    state.add(_coerce_stored_value(row[name], dtype, name, run_name))
        elif dtype in DTYPE_TO_NUMPY:
            path = root / f"{name}.{dtype}"
            if path.exists():
                state.add_array(_read_numeric(path, dtype))
            for row in sorted_hot_rows:
                if name in row:
                    state.add(_coerce_stored_value(row[name], dtype, name, run_name))
        states[name] = state
        changed = state.apply(spec) or changed
    return states, changed


def _track_monotonic_value(manifest, states, name, value):
    spec = manifest[name]
    state = states.get(name)
    if state is None:
        state = _Monotonic(spec["dtype"])
        states[name] = state
    state.add(value)
    return state.apply(spec)


def _tighten_dtypes(root, manifest):
    for name, spec in manifest.items():
        dtype = spec["dtype"]
        if dtype == JSONL_DTYPE:
            continue
        if "values" in spec:
            if dtype[0] == "f" or not spec["values"]["count"]:
                continue
            sequence = _value_sequence(spec)
            endpoints = np.asarray(_sequence_endpoints(root, name, spec), dtype=_value_dtype(dtype))
            tightened = _tight_dtype(endpoints)
            spec["dtype"] = _numeric_dtype(tightened)
            continue
        path = root / f"{name}.{dtype}"
        arr = _read_numeric(path, dtype)
        if arr.size == 0:
            continue  # pragma: no cover
        tightened = _tight_dtype(arr)
        if tightened is None:
            continue  # pragma: no cover - unreachable
        new_dtype = _numeric_dtype(tightened)
        if new_dtype == dtype:
            continue
        _publish_stored_dtype(root, manifest, name, new_dtype, tightened,
                              lambda: _write_manifest(root / "plattli.json", manifest))


def _zip_output(run_root, root, config_payload=None):
    zip_path = _zip_path_for_root(run_root)
    tmp_path = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if config_payload is not None and rel.as_posix() == "config.json":
                zf.writestr("config.json", config_payload)
                continue
            if path.is_symlink():
                zf.writestr(rel.as_posix(), path.read_bytes())
                continue
            zf.write(path, rel)
    tmp_path.replace(zip_path)
    rmtree(root)


def _cleanup_stored_files(root, manifest, *, include_orphans=False):
    """Remove obsolete known files; sweep unreferenced pairs only on finalization."""
    active = {"plattli.json", "config.json", HOT_FILENAME, HOT_COMPACTING_FILENAME}
    for name, spec in manifest.items():
        value_name = f"{name}.{spec['dtype']}"
        if "values" in spec:
            _value_sequence(spec)
        else:
            active.add(value_name)
        if spec["indices"] in ("indices", "ranges"):
            suffix = spec["indices"]
            active.add(f"{name}.{suffix}")
            if not (root / f"{name}.{suffix}").is_file():
                raise FileNotFoundError(f"missing {suffix} file for {name} in run {_run_name_for_root(root)}")
        # Never delete an alternate representation to hide a missing active one.
        if "values" not in spec and not (root / value_name).is_file() and _indices_length(root, name, spec):
            raise FileNotFoundError(f"missing values file for {name} in run {_run_name_for_root(root)}")

    config = root / "config.json"
    if config.is_symlink():
        try:
            active.add(config.resolve().relative_to(root.resolve()).as_posix())
        except ValueError:
            pass  # Config links outside the metric directory are not scanned.

    files = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file() or path.is_symlink()}
    value_suffixes = {*DTYPE_TO_NUMPY, JSONL_DTYPE, "values"}
    orphan_stems = set()
    if include_orphans:
        for rel in files:
            stem, _, suffix = rel.rpartition(".")
            if suffix in value_suffixes and stem not in manifest and any(f"{stem}.{suffix}" in files for suffix in ("indices", "ranges")):
                orphan_stems.add(stem)

    changed_dirs = set()
    for rel, path in files.items():
        if rel in active:
            continue
        base = rel.removesuffix(".tmp")
        stem, _, suffix = base.rpartition(".")
        metadata_tmp = rel in {"plattli.json.tmp", "config.json.tmp", f"{HOT_FILENAME}.tmp", f"{HOT_COMPACTING_FILENAME}.tmp"}
        metric_artifact = suffix in value_suffixes | {"indices", "ranges"} and (stem in manifest or stem in orphan_stems)
        if metadata_tmp or metric_artifact:
            path.unlink()
            changed_dirs.add(path.parent)
    for directory in changed_dirs:
        _fsync_dir(directory)


def _maybe_resume_finalized(run_root, root, allow_resume_finalized, config=None):
    zip_path = _zip_path_for_root(run_root)
    if not zip_path.exists():
        return
    run_name = _run_name_for_root(run_root)
    if not allow_resume_finalized:
        raise RuntimeError(f"found finalized run {run_name} at {zip_path}; pass allow_resume_finalized=True to resume or remove the zip")
    if not zip_path.is_file():
        raise RuntimeError(f"found {zip_path} for run {run_name} but it is not a file")
    if root.exists():
        if not root.is_dir():
            raise RuntimeError(f"expected {root} to be a directory for run {run_name}")
        if any(root.iterdir()):
            raise RuntimeError(f"found both {zip_path} and {root} for run {run_name}; refuse to merge")
    with zipfile.ZipFile(zip_path) as zf:
        zf.getinfo("plattli.json")
        members = []
        for info in zf.infolist():
            out_path = _archive_member_path(root, info.filename, run_name)
            if info.filename.endswith("/"):
                continue
            members.append((info, out_path))
        if isinstance(config, str):
            target = (run_root / config).expanduser().resolve()
            if any(target in out_path.parents for _, out_path in members):
                raise FileNotFoundError(f"config target is not a file: {target} (run {run_name})")
        root.mkdir(parents=True, exist_ok=True)
        for info, out_path in members:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(zf.read(info))
    zip_path.unlink()


class DirectWriter:
    def __init__(self, outdir, step=0, write_threads=16, config="config.json", allow_resume_finalized=False):
        self.run_root = Path(outdir)
        if self.run_root.name == "plattli":
            raise ValueError(f"outdir should be a run directory, not the plattli folder: {outdir}")
        self.step = _validate_step(step, self.run_root.name)
        write_threads = operator.index(write_threads)
        if write_threads < 0:
            raise ValueError(f"write_threads must be >= 0 for run {self.run_root.name}: {write_threads}")
        config = _prepare_config(self.run_root, config)
        self.root = self.run_root / "plattli"
        _maybe_resume_finalized(self.run_root, self.root, allow_resume_finalized, config)
        self.root.mkdir(parents=True, exist_ok=True)

        self._manifest = {}
        self._executor = ThreadPoolExecutor(max_workers=write_threads) if write_threads else None
        self._futures = []
        self._sequence_files = set()
        self._step_metrics = set()
        self._monotonic = {}
        self._integer_bounds_cache = {}
        self._broken = None
        self._finished = False

        if (self.root / "plattli.json").exists():
            self._manifest = json.loads((self.root / "plattli.json").read_text(encoding="utf-8"))
            self._manifest.pop("when_exported", None)
            rewrite_manifest = self._manifest.pop("run_rows", None) is not None
            _validate_metric_names(self._manifest, self.run_root.name)
            if self._manifest:
                _truncate_to_step(self.root, self._manifest, self.step, allow_missing=False)
                _force_indices_files(self.root, self._manifest)
                self._monotonic, monotonic_changed = _refresh_monotonic_metadata(self.root, self._manifest)
                if monotonic_changed:
                    rewrite_manifest = True
            if rewrite_manifest:
                _write_manifest(self.root / "plattli.json", self._manifest)

        _write_config(self.run_root, self.root / "config.json", config)
        if (self.root / "plattli.json").exists():
            _cleanup_stored_files(self.root, self._manifest)

    def write(self, *args, **kwargs):
        self._check_finished()
        self._check_broken()
        self._drain_errors()
        metrics = _merge_metrics(args, kwargs, self.run_root.name)
        # Existing names were validated when first published or loaded on resume.
        _validate_metric_names((name for name in metrics if name not in self._manifest), self.run_root.name)
        if not metrics:
            return

        self.step = _validate_step(self.step, self.run_root.name)

        prepared = _prepare_metrics(metrics, self._manifest, self._step_metrics,
                                    self.step, self.run_root.name, self._integer_bounds)

        new_metric = False
        for name, (dtype, value) in prepared.items():
            is_new = name not in self._manifest
            if is_new:
                (self.root / name).parent.mkdir(parents=True, exist_ok=True)
                self._manifest[name] = {"indices": "indices", "dtype": dtype}
                if dtype != JSONL_DTYPE:
                    sequence = _find_value_sequence([value], dtype)
                    if sequence is not None:
                        self._manifest[name]["values"] = {**sequence, "count": 0}
                new_metric = True
            else:
                if dtype != self._manifest[name]["dtype"]:
                    wait(self._futures)
                    self._drain_errors()
                    try:
                        _rewrite_stored_dtype(self.root, self._manifest, name, dtype,
                                              lambda: self._publish_manifest(name))
                    except Exception as err:
                        self._broken = err
                        raise
            new_metric = _track_monotonic_value(self._manifest, self._monotonic, name, value) or new_metric
            if name in self._integer_bounds_cache:
                lo, hi = self._integer_bounds_cache[name]
                self._integer_bounds_cache[name] = min(lo, value), max(hi, value)
            write_indices = self._manifest[name]["indices"] == "indices"
            if "values" in self._manifest[name]:
                try:
                    if is_new:
                        (self.root / f"{name}.{dtype}").unlink(missing_ok=True)
                        (self.root / f"{name}.indices").write_bytes(b"")
                    spec = self._manifest[name]
                    previous = dict(spec["values"])
                    _append_stored_values(self.root, name, spec, [value], indices=[self.step])
                    if "values" in spec:
                        # Coefficients precede their first indexed sample. A
                        # continuing cadence only appends its explicit index.
                        changed = is_new or {k: v for k, v in previous.items() if k != "count"} != {k: v for k, v in spec["values"].items() if k != "count"}
                        needs_publish = name not in self._sequence_files
                        if changed or needs_publish:
                            self._store_sequence_head(name)
                        new_metric = new_metric or needs_publish
                    else:
                        # Freeze the old head before indices grow into a raw
                        # replacement. Cached readers can safely read its prefix.
                        _replace_text_checked(self.root / f"{name}.values", json.dumps({"dtype": dtype, "values": previous}), "value sequence")
                        new_metric = True
                    if write_indices:
                        _append_indices(self.root / f"{name}.indices", [self.step])
                except Exception as err:
                    self._broken = err
                    raise
            elif is_new:
                try:
                    # Discard files orphaned by a crash before the metric reached the
                    # manifest, then publish the first row synchronously.
                    (self.root / f"{name}.{dtype}").write_bytes(b"")
                    (self.root / f"{name}.indices").write_bytes(b"")
                    self._write_entry(name, dtype, value, self.step, write_indices)
                except Exception as err:
                    self._broken = err
                    raise
            elif self._executor:
                self._futures.append(self._executor.submit(self._write_entry, name, dtype, value, self.step, write_indices))
            else:
                try:
                    self._write_entry(name, dtype, value, self.step, write_indices)
                except Exception as err:
                    self._broken = err
                    raise
            self._step_metrics.add(name)

        if new_metric:
            try:
                self._publish_manifest()
            except Exception as err:
                self._broken = err
                raise

    def end_step(self):
        self._check_finished()
        self._check_broken()
        self.step = _coerce_step(self.step, self.run_root.name)
        wait(self._futures)
        self._drain_errors()
        self._flush_step_indices()
        self._step_metrics.clear()
        self.step += 1

    def finish(self, optimize=True, zip=True, config_overrides=None):
        self._check_finished()
        self._check_broken()
        config_payload = _config_payload_with_overrides(
            self.root / "config.json", config_overrides, self.run_root.name)
        wait(self._futures)
        self._drain_errors()
        self._flush_step_indices()
        self._step_metrics.clear()

        if optimize:
            _tighten_dtypes(self.root, self._manifest)
            _optimize_values(self.root, self._manifest)
            _optimize_indices(self.root, self._manifest)

        _close_open_indices(self.root, self._manifest)
        _write_manifest(self.root / "plattli.json", self._manifest,
                        run_rows=max((_indices_length(self.root, name, spec)
                                      for name, spec in self._manifest.items()), default=0))
        _cleanup_stored_files(self.root, self._manifest, include_orphans=True)

        if zip:
            _zip_output(self.run_root, self.root, config_payload)
        elif config_payload is not None:
            _replace_text_checked(self.root / "config.json", config_payload, "config")
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._finished = True

    def set_config(self, config):
        self._check_finished()
        _write_config(self.run_root, self.root / "config.json", _prepare_config(self.run_root, config))

    def _check_finished(self):
        if self._finished:
            raise RuntimeError(f"writer for run {self.run_root.name} is already finished")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Flush, don't finish: the run stays resumable, like dropping the writer.
        if self._broken is None:
            wait(self._futures)
            self._drain_errors()

    def _write_entry(self, name, dtype, value, step, write_indices):
        if dtype == JSONL_DTYPE:
            _append_jsonl(self.root / f"{name}.jsonl", [value])
        else:
            _append_numeric(self.root / f"{name}.{dtype}", [value], dtype)
        if write_indices:
            _append_indices((self.root / f"{name}.indices"), [step])

    def _integer_bounds(self, name):
        if name in self._integer_bounds_cache:
            return self._integer_bounds_cache[name]
        wait(self._futures)
        self._drain_errors()
        bounds = _integer_bounds(self.root, name, self._manifest[name])
        self._integer_bounds_cache[name] = bounds
        return bounds

    def _store_sequence_head(self, name):
        spec = self._manifest[name]
        head = {key: value for key, value in spec["values"].items() if key != "count"}
        _replace_text_checked(self.root / f"{name}.values", json.dumps({"dtype": spec["dtype"], "values": head}), "value sequence")
        self._sequence_files.add(name)

    def _publish_manifest(self, changed_name=None):
        previous = None
        if changed_name is not None and "values" in self._manifest[changed_name]:
            path = self.root / f"{changed_name}.values"
            if path.exists():
                previous = path.read_bytes()
        try:
            if changed_name is not None and "values" in self._manifest[changed_name]:
                self._store_sequence_head(changed_name)
            live = {name: {**spec, "values": "sequence"} if "values" in spec and name in self._sequence_files else spec
                    for name, spec in self._manifest.items()}
            _write_manifest(self.root / "plattli.json", live)
        except BaseException:
            if previous is not None:
                _replace_bytes(path, previous)
            raise
        for name in self._step_metrics:
            if "values" not in self._manifest[name]:
                (self.root / f"{name}.values").unlink(missing_ok=True)

    def _flush_step_indices(self):
        if not self._step_metrics:
            return
        updated = False
        run_name = self.run_root.name
        for name in self._step_metrics:
            indices_spec = self._manifest[name]["indices"]
            if indices_spec == "indices":
                continue
            try:
                self._manifest[name]["indices"] = _append_step_to_indices_spec(indices_spec, self.step)
            except (ValueError, RuntimeError) as exc:
                raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
            updated = True
        if updated:
            _write_manifest(self.root / "plattli.json", self._manifest)

    def _drain_errors(self):
        remaining = []
        for f in self._futures:
            if f.done():
                if (err := f.exception()) is not None:
                    self._broken = err
                    raise err
            else:
                remaining.append(f)
        self._futures = remaining

    def _check_broken(self):
        if self._broken is not None:
            raise RuntimeError(f"writer for run {self.run_root.name} is broken after an earlier write error;"
                               " the on-disk state may be partial, recreate the writer to resume") from self._broken


class CompactingWriter:
    def __init__(self, outdir, step=0, *, hotsize, config="config.json", allow_resume_finalized=False, live_indices="auto"):
        if live_indices not in ("auto", "explicit"):
            raise ValueError(f"invalid live_indices: {live_indices!r}; expected auto or explicit")
        self.live_indices = live_indices
        self.run_root = Path(outdir)
        if self.run_root.name == "plattli":
            raise ValueError(f"outdir should be a run directory, not the plattli folder: {outdir}")
        self.step = _validate_step(step, self.run_root.name)
        self.hotsize = int(hotsize)
        if self.hotsize <= 0:
            raise ValueError(f"hotsize must be > 0 for run {self.run_root.name}: {self.hotsize}")
        config = _prepare_config(self.run_root, config)
        self.root = self.run_root / "plattli"
        _maybe_resume_finalized(self.run_root, self.root, allow_resume_finalized, config)
        self.root.mkdir(parents=True, exist_ok=True)

        self._manifest = {}
        self._compact_executor = ThreadPoolExecutor(max_workers=1)
        self._compact_future = None
        self._step_metrics = set()
        self._monotonic = {}
        self._integer_bounds_cache = {}
        self._broken = None
        self._finished = False

        self._hot_rows = []
        self._hot_index = {}
        self._current_row = {}
        self._hot_lock = threading.Lock()
        self._hot_fh = None
        self._stored_counts = {}
        self._range_states = {}
        self._manifest_dirty = False
        self._manifest_version = 0
        self._manifest_disk_version = 0
        self._manifest_io_lock = threading.Lock()

        rewrite_manifest = False
        if (self.root / "plattli.json").exists():
            self._manifest = json.loads((self.root / "plattli.json").read_text(encoding="utf-8"))
            self._manifest.pop("when_exported", None)
            rewrite_manifest = self._manifest.pop("run_rows", None) is not None
            _validate_metric_names(self._manifest, self.run_root.name)
            if self._manifest:
                _truncate_to_step(self.root, self._manifest, self.step, allow_missing=True)

        self._load_hot_rows()
        if self._manifest:
            if self.live_indices == "explicit":
                _force_indices_files(self.root, self._manifest)
            elif _migrate_oversized_piecewise_indices(self.root, self._manifest):
                rewrite_manifest = True
            self._monotonic, monotonic_changed = _refresh_monotonic_metadata(self.root, self._manifest, self._hot_rows)
            if monotonic_changed:
                rewrite_manifest = True
        if rewrite_manifest:
            _write_manifest(self.root / "plattli.json", self._manifest)
        _write_config(self.run_root, self.root / "config.json", config)
        if (self.root / "plattli.json").exists():
            _cleanup_stored_files(self.root, self._manifest)

    def __del__(self):
        # Some callers intentionally drop unfinished writers to resume later.
        self._close_hot_file()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Flush, don't finish: the run stays resumable, like dropping the writer.
        if self._broken is None and not self._finished:
            self._flush_hot()
            if self._compact_future is not None:
                wait([self._compact_future])
                with self._hot_lock:
                    self._maybe_finalize_compaction_locked()
        self._close_hot_file()

    def write(self, *args, flush=False, **kwargs):
        self._check_finished()
        self._check_broken()
        self._drain_errors()
        metrics = _merge_metrics(args, kwargs, self.run_root.name)
        # Existing names were validated when first published or loaded on resume.
        _validate_metric_names((name for name in metrics if name not in self._manifest), self.run_root.name)
        if not metrics and flush:
            self._flush_hot()
            return
        if not metrics:
            return

        self.step = _validate_step(self.step, self.run_root.name)

        # Never wait for a worker while holding its manifest lock. Most writes keep
        # their dtype and don't wait; widening drains the old representation first.
        prepared = _prepare_metrics(metrics, self._manifest, self._step_metrics,
                                    self.step, self.run_root.name, self._integer_bounds)

        for name, (dtype, value) in prepared.items():
            if name in self._manifest and dtype != self._manifest[name]["dtype"]:
                self._wait_for_compaction()
                with self._hot_lock:
                    try:
                        _rewrite_stored_dtype(self.root, self._manifest, name, dtype,
                                              lambda: self._persist_manifest(*self._serialize_manifest_locked()))
                    except Exception as err:
                        self._broken = err
                        raise

        new_metric = False
        pending = None
        with self._hot_lock:  # The compaction thread reads and serializes the manifest.
            for name, (dtype, value) in prepared.items():
                if name not in self._manifest:
                    (self.root / name).parent.mkdir(parents=True, exist_ok=True)
                    self._manifest[name] = {"indices": [], "dtype": dtype}
                    if dtype != JSONL_DTYPE:
                        sequence = _find_value_sequence([value], dtype)
                        if sequence is not None:
                            self._manifest[name]["values"] = {**sequence, "count": 0}
                    new_metric = True
                if _track_monotonic_value(self._manifest, self._monotonic, name, value):
                    # Safe to defer: flags are persisted by the next compaction before any
                    # values land in columnar files, and recomputed from data on resume.
                    self._manifest_dirty = True
                self._current_row[name] = value
                if name in self._integer_bounds_cache:
                    lo, hi = self._integer_bounds_cache[name]
                    self._integer_bounds_cache[name] = min(lo, value), max(hi, value)
                self._step_metrics.add(name)

            if new_metric:
                pending = self._serialize_manifest_locked()
        if pending:
            self._persist_manifest(*pending)
        if flush:
            self._flush_hot()

    def end_step(self):
        self._check_finished()
        self._check_broken()
        self.step = _coerce_step(self.step, self.run_root.name)
        if self._current_row or self.step in self._hot_index:
            self._flush_hot()
        self._step_metrics.clear()
        self._current_row = {}
        self.step += 1

    def finish(self, optimize=True, zip=True, config_overrides=None):
        self._check_finished()
        self._check_broken()
        if self._current_row:
            self.step = _validate_step(self.step, self.run_root.name)
        config_payload = _config_payload_with_overrides(
            self.root / "config.json", config_overrides, self.run_root.name)
        if self._compact_future:
            wait([self._compact_future])
            with self._hot_lock:
                self._maybe_finalize_compaction_locked()
        with self._hot_lock:
            if self._current_row:
                self._upsert_hot_row(self.step, self._current_row)
            rows = list(self._hot_rows)
            self._hot_rows = []
            self._hot_index = {}
            self._current_row = {}
        if rows:
            self._compact_rows(rows)
        hot_path = self.root / HOT_FILENAME
        self._close_hot_file()
        if hot_path.exists():
            hot_path.unlink()

        if optimize:
            _tighten_dtypes(self.root, self._manifest)
            _optimize_values(self.root, self._manifest)
            _optimize_indices(self.root, self._manifest)

        _close_open_indices(self.root, self._manifest)
        _write_manifest(self.root / "plattli.json", self._manifest,
                        run_rows=max((_indices_length(self.root, name, spec)
                                      for name, spec in self._manifest.items()), default=0))
        _cleanup_stored_files(self.root, self._manifest, include_orphans=True)

        if zip:
            _zip_output(self.run_root, self.root, config_payload)
        elif config_payload is not None:
            _replace_text_checked(self.root / "config.json", config_payload, "config")
        if self._compact_executor:
            self._compact_executor.shutdown(wait=True)
            self._compact_executor = None
        self._finished = True

    def set_config(self, config):
        self._check_finished()
        _write_config(self.run_root, self.root / "config.json", _prepare_config(self.run_root, config))

    def _check_finished(self):
        if self._finished:
            raise RuntimeError(f"writer for run {self.run_root.name} is already finished")

    def _wait_for_compaction(self):
        if self._compact_future is not None:
            wait([self._compact_future])
            with self._hot_lock:
                self._maybe_finalize_compaction_locked()

    def _integer_bounds(self, name):
        if name in self._integer_bounds_cache:
            return self._integer_bounds_cache[name]
        self._wait_for_compaction()
        bounds = _integer_bounds(self.root, name, self._manifest[name], self._hot_rows)
        self._integer_bounds_cache[name] = bounds
        return bounds

    def _load_hot_rows(self):
        hot_path = self.root / HOT_FILENAME
        compacting_path = self.root / HOT_COMPACTING_FILENAME
        rewrite_hot = compacting_path.exists()  # Consolidate a leftover rotation into one file.
        if not hot_path.exists() and not rewrite_hot:
            return
        new_metric = False
        min_hot_step = None
        rows = {}
        for path in (compacting_path, hot_path):
            if not path.exists():
                continue
            lines = path.read_bytes().splitlines()
            for idx, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if idx == len(lines) - 1:
                        rewrite_hot = True
                        break
                    raise
                row_step = int(row["step"])
                if row_step >= self.step:  # Resuming at step N nukes N and everything beyond.
                    rewrite_hot = True
                    continue
                rows[row_step] = row  # On overlap, the active hot file wins.
                if min_hot_step is None or row_step < min_hot_step:
                    min_hot_step = row_step
        for row_step, row in rows.items():
            self._hot_index[row_step] = len(self._hot_rows)
            self._hot_rows.append(row)
            for name, value in row.items():
                if name == "step":
                    continue
                _validate_metric_name(name, self.run_root.name)
                if name not in self._manifest:
                    dtype = _resolve_dtype(value, name=name, run_name=self.run_root.name)
                    (self.root / name).parent.mkdir(parents=True, exist_ok=True)
                    self._manifest[name] = {"indices": [], "dtype": dtype}
                    if dtype != JSONL_DTYPE:
                        sequence = _find_value_sequence([value], dtype)
                        if sequence is not None:
                            self._manifest[name]["values"] = {**sequence, "count": 0}
                    new_metric = True
        if rewrite_hot:
            self._write_hot_file()
        if compacting_path.exists():
            compacting_path.unlink()
        if min_hot_step is not None:
            _truncate_to_step(self.root, self._manifest, min_hot_step, allow_missing=True)
            new_metric = False
        if new_metric:
            with self._hot_lock:
                _write_manifest(self.root / "plattli.json", self._manifest)

    def _drain_errors(self):
        if self._compact_future is None:
            return
        if self._compact_future.done():
            if (err := self._compact_future.exception()) is not None:
                with self._hot_lock:
                    self._compact_future = None
                self._broken = err
                raise err

    def _check_broken(self):
        if self._broken is not None:
            raise RuntimeError(f"writer for run {self.run_root.name} is broken after an earlier compaction error;"
                               " the on-disk state may be partial, recreate the writer to resume") from self._broken

    def _serialize_manifest_locked(self):
        self._manifest_version += 1
        self._manifest_dirty = False
        return self._manifest_version, _manifest_payload(self._manifest)

    def _persist_manifest(self, version, payload):
        # Ordinary compaction fsyncs outside _hot_lock. Dtype promotion holds it
        # after draining compaction so the replacement and its schema stay together.
        with self._manifest_io_lock:
            if version <= self._manifest_disk_version:
                return  # A newer serialization already landed.
            _replace_text_checked(self.root / "plattli.json", payload, "manifest")
            self._manifest_disk_version = version

    def _flush_hot(self):
        if self._current_row:
            self.step = _validate_step(self.step, self.run_root.name)
        with self._hot_lock:
            future = self._compact_future
            backlogged = future is not None and len(self._hot_rows) >= 10 * self.hotsize
        if backlogged:
            # Backpressure: the filesystem cannot keep up with the write rate. Block on
            # the in-flight batch (outside the lock, the worker needs it) so memory stays
            # bounded; slow beats OOM. Normal operation never enters this branch.
            wait([future])
        with self._hot_lock:
            self._maybe_finalize_compaction_locked()
            if self._compact_future is None and (batch := self._rotate_hot_locked()):
                self._compact_future = self._compact_executor.submit(self._compact_batch, batch)
            if self._current_row:
                mode, append_row = self._upsert_hot_row(self.step, self._current_row)
                if mode == "rewrite":
                    self._write_hot_file()
                elif append_row is not None:
                    self._append_hot_row(append_row)

    def _rotate_hot_locked(self):
        # _hot_rows is in step order and only its tail can hold the in-progress step, so
        # the completed batch is a prefix slice; keep this O(1)-ish, it runs every step.
        k = len(self._hot_rows)
        while k and self._hot_rows[k - 1]["step"] >= self.step:
            k -= 1
        if k < self.hotsize:
            return []
        # Rotate instead of rewriting: rename is a cheap metadata op, and compaction can
        # later just unlink the rotated file. A rewrite would fsync under the lock and
        # stall the writing thread.
        self._close_hot_file()
        (self.root / HOT_FILENAME).rename(self.root / HOT_COMPACTING_FILENAME)
        batch = self._hot_rows[:k]
        self._hot_rows = self._hot_rows[k:]
        self._hot_index = {row["step"]: idx for idx, row in enumerate(self._hot_rows)}
        if self._hot_rows:
            # Rare (mid-step flush): non-batch rows were in the rotated file too; keep
            # them durable in the fresh hot log before the rotated file goes away.
            self._write_hot_file()
        return batch

    def _compact_batch(self, rows):
        self._compact_rows(rows)
        (self.root / HOT_COMPACTING_FILENAME).unlink()

    def _upsert_hot_row(self, step, row):
        payload = {"step": int(step)}
        payload.update(row)
        if step in self._hot_index:
            idx = self._hot_index[step]
            if self._hot_rows[idx] == payload:
                return None, None
            self._hot_rows[idx] = payload
            return "rewrite", None
        self._hot_index[step] = len(self._hot_rows)
        self._hot_rows.append(payload)
        return "append", payload

    def _write_hot_file(self):
        self._close_hot_file()
        path = self.root / HOT_FILENAME
        if not self._hot_rows:
            if path.exists():
                path.unlink()
            return
        payload = "".join(f"{json.dumps(row, ensure_ascii=False, separators=(',', ':'))}\n" for row in self._hot_rows)
        _replace_text_checked(path, payload, "hot")

    def _append_hot_row(self, row):
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        fh = self._open_hot_file()
        fh.write(line.encode("utf-8"))
        fh.flush()

    def _open_hot_file(self):
        if self._hot_fh is None or self._hot_fh.closed:
            path = self.root / HOT_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            self._hot_fh = path.open("ab")
        return self._hot_fh

    def _close_hot_file(self):
        if getattr(self, "_hot_fh", None) is not None:
            self._hot_fh.close()
            self._hot_fh = None

    def _maybe_finalize_compaction_locked(self):
        future = self._compact_future
        if future is None or not future.done():
            return
        self._compact_future = None
        if (err := future.exception()) is not None:
            self._broken = err
            raise err

    def _compact_rows(self, rows):
        columns = {}
        for row in rows:
            step = int(row["step"])
            for name, value in row.items():
                if name == "step":
                    continue
                col = columns.get(name)
                if col is None:
                    col = {"indices": [], "values": []}
                    columns[name] = col
                col["indices"].append(step)
                col["values"].append(value)

        run_name = self.run_root.name
        with self._hot_lock:
            metas = {}
            value_specs = {}
            for name in columns:
                dtype = self._manifest[name]["dtype"]
                value_specs[name] = dict(self._manifest[name])
                if "values" in value_specs[name]:
                    value_specs[name]["values"] = dict(value_specs[name]["values"])
                indices_spec = self._manifest[name]["indices"]
                if indices_spec not in ("indices", "ranges"):
                    # The append helpers mutate segments in place; work on a copy so the
                    # live manifest only ever changes under the lock.
                    indices_spec = [dict(seg) for seg in _segments_from_spec(indices_spec)]
                metas[name] = (dtype, indices_spec)

        # Extend indices (files or specs) first. Values are appended only after the
        # manifest is persisted, so on-disk specs are never behind the value files:
        # with open-tail segments, "manifest ahead of values" is always resolvable.
        spec_updates = {}
        new_counts = {}
        for name, col in columns.items():
            dtype, indices_spec = metas[name]
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            if indices_spec == "indices":
                _append_indices(self.root / f"{name}.indices", col["indices"])
                continue
            if indices_spec == "ranges":
                path = self.root / f"{name}.ranges"
                current_count = self._stored_counts.get(name)
                if current_count is None:
                    current_count = _stored_values_count(self.root, name, value_specs[name], allow_missing=False)
                if name not in self._range_states:
                    data = path.read_bytes()
                    records, tail = _ranges_parse(data)
                    self._range_states[name] = (
                        (tail[0], tail[1], int(records[-1]["end_row"]) if len(records) else 0) if tail else None,
                        len(data),
                    )
                tail, size = self._range_states[name]
                had_tail = tail is not None
                payload, tail = _ranges_append(tail, current_count, col["indices"])
                new_count = current_count + len(col["indices"])
                new_size = size + len(payload)
                # Keep live ranges only when both history and this batch amortize
                # range publication over at least 32 samples per new range.
                added_ranges = (len(payload) + (0 if had_tail else 8)) // 16
                if new_count < 32 * ((new_size + 8) // 16) or len(col["indices"]) < 32 * added_ranges:
                    segments = _ranges_to_segments(path.read_bytes(), current_count)
                    indices = np.concatenate((_segments_to_array(segments), np.asarray(col["indices"], dtype=np.uint32)))
                    # Freeze the old representation before it can become stale.
                    _append_range_bytes(path, payload + np.asarray([new_count], dtype="<u8").tobytes())
                    _replace_bytes(self.root / f"{name}.indices", indices.tobytes())
                    spec_updates[name] = "indices"
                    del self._range_states[name]
                else:
                    _append_range_bytes(path, payload)
                    self._range_states[name] = tail, new_size
                new_counts[name] = new_count
                continue
            try:
                old_spec = json.dumps(indices_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                current_count = self._stored_counts.get(name)
                if current_count is None:
                    current_count = _stored_values_count(self.root, name, value_specs[name], allow_missing=True)
                spec = _extend_inline_indices(indices_spec, col["indices"], current_count)
                new_counts[name] = current_count + len(col["indices"])
            except (ValueError, RuntimeError) as exc:
                raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
            if self.live_indices == "explicit":
                _replace_bytes(self.root / f"{name}.indices", _segments_to_array(spec, total_count=new_counts[name]).tobytes())
                spec_updates[name] = "indices"
            elif _segments_too_many(spec, total_count=new_counts[name]):
                selected = _store_indices(self.root, name, spec, new_counts[name], open_tail=True)
                spec_updates[name] = selected
            elif json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")) != old_spec:
                spec_updates[name] = spec

        # Inline values and their indices become visible in the same manifest.
        # A sequence that breaks is materialized before publishing its raw encoding.
        value_updates = {}
        for name, col in columns.items():
            spec = value_specs[name]
            if "values" in spec:
                new_counts[name] = spec["values"]["count"] + len(col["values"])
                _append_stored_values(self.root, name, spec, col["values"], indices=col["indices"])
                value_updates[name] = spec

        # One coalesced manifest write per batch; serialize under the lock, fsync outside.
        pending = None
        with self._hot_lock:
            for name, spec in spec_updates.items():
                self._manifest[name]["indices"] = spec
            for name, spec in value_updates.items():
                if "values" in spec:
                    self._manifest[name]["values"] = spec["values"]
                else:
                    self._manifest[name].pop("values")
            if spec_updates or value_updates or self._manifest_dirty:
                pending = self._serialize_manifest_locked()
        if pending:
            self._persist_manifest(*pending)
        for name, selected in spec_updates.items():
            if metas[name][1] == "ranges" and selected == "indices":
                (self.root / f"{name}.ranges").unlink()
                _fsync_dir((self.root / name).parent)

        for name, col in columns.items():
            dtype, _ = metas[name]
            if name in value_updates:
                pass  # Already encoded or materialized before manifest publication.
            elif dtype == JSONL_DTYPE:
                _append_jsonl(self.root / f"{name}.jsonl", col["values"])
            else:
                _append_numeric(self.root / f"{name}.{dtype}", col["values"], dtype)
            if name in new_counts:
                self._stored_counts[name] = new_counts[name]


def _find_arange_params(array):
    if array.size in (0, 1):
        return None  # pragma: no cover
    diffs = np.diff(array)
    if not (diffs > 0).all() or (diffs != diffs[0]).any():
        return None
    step = int(diffs[0])
    start = int(array[0])
    stop = int(array[-1]) + 1
    return start, stop, step


def _tightest_int(array):
    if array.dtype.kind not in "iuO":
        return array  # pragma: no cover - unreachable

    # Python comparisons stay exact at the int64/uint64 boundary.
    amin, amax = int(array.min()), int(array.max())
    if amin >= 0:
        for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
            if amax <= np.iinfo(dt).max:
                return array.astype(dt, copy=False)
        return array

    for dt in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dt)
        if info.min <= amin and amax <= info.max:
            return array.astype(dt, copy=False)

    return array  # pragma: no cover - unreachable


def _tight_dtype(array):
    array = np.asarray(array)
    if array.dtype.kind == "f":
        return array
    if array.dtype.kind in "iuO":
        return _tightest_int(array)
    return None  # pragma: no cover - unreachable


def _append_numeric(path, values, dtype):
    arr = np.asarray(values, dtype=DTYPE_TO_NUMPY[dtype])
    if arr.ndim != 1:
        raise ValueError(f"expected 1d values for {path} (shape {arr.shape})")
    with path.open("ab") as fh:
        fh.write(_encode_numeric(arr, dtype))


def _append_indices(path, indices):
    arr = np.asarray(indices)
    if arr.ndim != 1:
        raise ValueError(f"expected 1d indices for {path} (shape {arr.shape})")
    with path.open("ab") as fh:
        np.asarray(arr, dtype=np.uint32).tofile(fh)


def _append_jsonl(path, values):
    lines = []
    for value in values:
        if isinstance(value, (np.ndarray, np.generic)):
            value = value.item()
        lines.append(json.dumps(value, ensure_ascii=False))
    if not lines:
        return
    payload = "\n".join(lines) + "\n"
    with path.open("ab") as fh:
        fh.write(payload.encode("utf-8"))


def _resolve_dtype(value, name=None, run_name=None):
    if type(value) is int:
        return "i64" if -(2**63) <= value < 2**63 else "i128"
    if type(value) is float:
        return "f32"
    if not isinstance(value, (np.ndarray, np.generic)) and hasattr(value, "__array__"):
        value = np.asarray(value)
    if isinstance(value, (np.ndarray, np.generic)):
        if value.shape != ():
            detail = "only scalar array-like values are supported"
            if name is not None:
                detail += f" for {name}"
            if run_name is not None:
                detail += f" in run {run_name}"
            detail += f" (shape {value.shape})"
            raise ValueError(detail)
        if (kind := value.dtype.kind) in "fiu":
            dtype = f"{kind}{value.dtype.itemsize * 8}"
            return dtype if dtype in DTYPE_TO_NUMPY else JSONL_DTYPE
        return JSONL_DTYPE
    if isinstance(value, bool):
        return JSONL_DTYPE
    if isinstance(value, float):
        return "f32"
    if isinstance(value, (int, np.integer)):
        return "i64"
    return JSONL_DTYPE
