import json
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from shutil import rmtree

import numpy as np

from ._indices import (
    _append_step_to_indices_spec,
    _append_steps_to_indices_spec,
    _find_piecewise_params,
    _segments_close_open_tail,
    _segments_count_and_last,
    _segments_from_spec,
    _segments_have_open_tail,
    _segments_to_array,
    _segments_too_many,
    _segments_truncate,
)

DTYPE_TO_NUMPY = {
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
}

JSONL_DTYPE = "jsonl"
HOT_FILENAME = "hot.jsonl"
HOT_COMPACTING_FILENAME = "hot.compacting.jsonl"
_RESERVED_METRIC_NAMES = frozenset({"step", "run_rows", "when_exported", "hot", "hot.compacting"})


def _validate_metric_name(name, run_name):
    if not isinstance(name, str):
        raise TypeError(f"metric names must be strings in run {run_name} (got {type(name).__name__})")
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


def _replace_text_checked(path, payload, label):
    if not payload:
        raise RuntimeError(f"{label} payload must not be empty for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
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
            if path.exists() or path.is_symlink():
                path.unlink()
            path.symlink_to(target.resolve())
            return
        if path.exists() or path.is_symlink():
            return
        config = {}
    if path.is_symlink():
        path.unlink()
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")


def _truncate_to_step(root, manifest, step, allow_missing):
    run_name = _run_name_for_root(root)
    for name, spec in manifest.items():
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
            with idx_path.open("r+b") as fh:
                fh.truncate(keep * 4)
        else:
            try:
                segments = _segments_from_spec(indices_spec)
                total_count = _stored_values_count(root, name, dtype, allow_missing) if _segments_have_open_tail(segments) else None
                kept, keep = _segments_truncate(segments, step, total_count=total_count)
            except (ValueError, RuntimeError) as exc:
                raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
            spec["indices"] = kept

        if dtype == JSONL_DTYPE:
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
            with path.open("r+b") as fh:
                fh.truncate(offset)
        else:
            value_path = root / f"{name}.{dtype}"
            if not value_path.exists():
                if allow_missing:
                    continue
                raise FileNotFoundError(f"missing values file for {name} in run {run_name}")
            with value_path.open("r+b") as fh:
                fh.truncate(keep * np.dtype(DTYPE_TO_NUMPY[dtype]).itemsize)
    _write_manifest(root / "plattli.json", manifest)


def _force_indices_files(root, manifest):
    run_name = _run_name_for_root(root)
    updated = False
    for name, spec in manifest.items():
        indices_spec = spec.get("indices")
        if indices_spec == "indices":
            continue
        try:
            segments = _segments_from_spec(indices_spec)
            total_count = _stored_values_count(root, name, spec["dtype"], allow_missing=False) if _segments_have_open_tail(segments) else None
            indices = _segments_to_array(segments, total_count=total_count)
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
        idx_path = root / f"{name}.indices"
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        with idx_path.open("wb") as fh:
            indices.tofile(fh)
        spec["indices"] = "indices"
        updated = True
    if updated:
        _write_manifest(root / "plattli.json", manifest)


def _migrate_oversized_piecewise_indices(root, manifest):
    run_name = _run_name_for_root(root)
    updated = False
    for name, spec in manifest.items():
        indices_spec = spec["indices"]
        if indices_spec == "indices":
            continue
        try:
            segments = _segments_from_spec(indices_spec)
            total_count = (
                _stored_values_count(root, name, spec["dtype"], allow_missing=False)
                if _segments_have_open_tail(segments)
                else None
            )
            if not _segments_too_many(segments, total_count=total_count):
                continue
            indices = _segments_to_array(segments, total_count=total_count)
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
        idx_path = root / f"{name}.indices"
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        with idx_path.open("wb") as fh:
            indices.tofile(fh)
        spec["indices"] = "indices"
        updated = True
    return updated


def _optimize_indices(root, manifest):
    for name, spec in manifest.items():
        if spec["indices"] != "indices":
            continue  # pragma: no cover
        idx_path = root / f"{name}.indices"
        indices = np.fromfile(idx_path, dtype=np.uint32)
        segments = _find_piecewise_params(indices)
        if segments:
            spec["indices"] = segments
            idx_path.unlink()


def _indices_length(root, name, spec):
    indices_spec = spec["indices"]
    if isinstance(indices_spec, (dict, list)):
        try:
            segments = _segments_from_spec(indices_spec)
            total_count = _stored_values_count(root, name, spec["dtype"], allow_missing=False) if _segments_have_open_tail(segments) else None
            total, _ = _segments_count_and_last(segments, total_count=total_count)
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f"{exc} (metric {name}, run {_run_name_for_root(root)})") from exc
        return total
    if indices_spec == "indices":
        idx_path = root / f"{name}.indices"
        return idx_path.stat().st_size // 4
    return 0  # pragma: no cover


def _stored_values_count(root, name, dtype, allow_missing):
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
    return path.stat().st_size // np.dtype(DTYPE_TO_NUMPY[dtype]).itemsize


def _close_open_indices(root, manifest):
    changed = False
    for name, spec in manifest.items():
        indices_spec = spec["indices"]
        if not isinstance(indices_spec, (dict, list)):
            continue
        segments = _segments_from_spec(indices_spec)
        if not _segments_have_open_tail(segments):
            continue
        total_count = _stored_values_count(root, name, spec["dtype"], allow_missing=False)
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
    if hasattr(value, "__array__"):
        value = np.asarray(value)
    if isinstance(value, (np.ndarray, np.generic)):
        if value.shape != ():
            raise ValueError(f"only scalar values are supported for {name} in run {run_name} (shape {value.shape})")
        value = value.item()
    if dtype == JSONL_DTYPE:
        return value

    arr = np.asarray(value)
    if arr.shape != ():
        raise ValueError(f"only scalar values are supported for {name} in run {run_name} (shape {arr.shape})")
    return np.asarray(arr, dtype=DTYPE_TO_NUMPY[dtype]).item()


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
        self.add(values[0].item())
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
        self.last = values[-1].item()

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
        if dtype in DTYPE_TO_NUMPY:
            path = root / f"{name}.{dtype}"
            if path.exists():
                state.add_array(np.fromfile(path, dtype=DTYPE_TO_NUMPY[dtype]))
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
        path = root / f"{name}.{dtype}"
        arr = np.fromfile(path, dtype=DTYPE_TO_NUMPY[dtype])
        if arr.size == 0:
            continue  # pragma: no cover
        tightened = _tight_dtype(arr)
        if tightened is None:
            continue  # pragma: no cover - unreachable
        new_dtype = f"{tightened.dtype.kind}{tightened.dtype.itemsize * 8}"
        if new_dtype == dtype:
            continue
        tightened.tofile(root / f"{name}.{new_dtype}")
        path.unlink()
        spec["dtype"] = new_dtype


def _zip_output(run_root, root):
    zip_path = _zip_path_for_root(run_root)
    tmp_path = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if path.is_symlink():
                zf.writestr(rel.as_posix(), path.read_bytes())
                continue
            zf.write(path, rel)
    tmp_path.replace(zip_path)
    rmtree(root)


def _maybe_resume_finalized(run_root, root, allow_resume_finalized):
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
    else:
        root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.getinfo("plattli.json")
        members = []
        for info in zf.infolist():
            out_path = _archive_member_path(root, info.filename, run_name)
            if info.filename.endswith("/"):
                continue
            members.append((info, out_path))
        for info, out_path in members:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(zf.read(info))
    zip_path.unlink()


class DirectWriter:
    def __init__(self, outdir, step=0, write_threads=16, config="config.json", allow_resume_finalized=False):
        self.run_root = Path(outdir)
        if self.run_root.name == "plattli":
            raise ValueError(f"outdir should be a run directory, not the plattli folder: {outdir}")
        self.root = self.run_root / "plattli"
        _maybe_resume_finalized(self.run_root, self.root, allow_resume_finalized)
        self.root.mkdir(parents=True, exist_ok=True)

        self.step = int(step)
        assert self.step >= 0, f"step must be >= 0 for run {self.run_root.name}: {self.step}"

        self._manifest = {}
        self._executor = ThreadPoolExecutor(max_workers=write_threads) if write_threads else None
        self._futures = []
        self._step_metrics = set()
        self._monotonic = {}
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

        self.set_config(config)

    def write(self, *args, **kwargs):
        self._check_finished()
        self._check_broken()
        self._drain_errors()
        metrics = _merge_metrics(args, kwargs, self.run_root.name)
        _validate_metric_names(metrics, self.run_root.name)
        if not metrics:
            return

        if self.step < 0 or self.step > 0xFFFFFFFF:
            raise ValueError(f"step out of uint32 range for run {self.run_root.name}: {self.step}")

        new_metric = False
        for name, value in metrics.items():
            if name == "step":
                raise ValueError(f"metric name 'step' is reserved in run {self.run_root.name}")
            if name in self._step_metrics:
                raise RuntimeError(f"metric already written in step {self.step} for {name} in run {self.run_root.name}")
            is_new = name not in self._manifest
            if is_new:
                dtype = _resolve_dtype(value, name=name, run_name=self.run_root.name)
                (self.root / name).parent.mkdir(parents=True, exist_ok=True)
                self._manifest[name] = {"indices": "indices", "dtype": dtype}
                new_metric = True
            else:
                dtype = self._manifest[name]["dtype"]
            value = _coerce_stored_value(value, dtype, name, self.run_root.name)
            new_metric = _track_monotonic_value(self._manifest, self._monotonic, name, value) or new_metric
            write_indices = self._manifest[name]["indices"] == "indices"
            if is_new:
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
                _write_manifest(self.root / "plattli.json", self._manifest)
            except Exception as err:
                self._broken = err
                raise

    def end_step(self):
        self._check_finished()
        self._check_broken()
        wait(self._futures)
        self._drain_errors()
        self._flush_step_indices()
        self._step_metrics.clear()
        self.step += 1

    def finish(self, optimize=True, zip=True):
        self._check_finished()
        self._check_broken()
        wait(self._futures)
        self._drain_errors()
        self._flush_step_indices()
        self._step_metrics.clear()

        if optimize:
            _tighten_dtypes(self.root, self._manifest)
            _optimize_indices(self.root, self._manifest)

        _close_open_indices(self.root, self._manifest)
        _write_manifest(self.root / "plattli.json", self._manifest,
                        run_rows=max((_indices_length(self.root, name, spec)
                                      for name, spec in self._manifest.items()), default=0))

        if zip:
            _zip_output(self.run_root, self.root)
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._finished = True

    def set_config(self, config):
        self._check_finished()
        _write_config(self.run_root, self.root / "config.json", config)

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
    def __init__(self, outdir, step=0, *, hotsize, config="config.json", allow_resume_finalized=False):
        self.run_root = Path(outdir)
        if self.run_root.name == "plattli":
            raise ValueError(f"outdir should be a run directory, not the plattli folder: {outdir}")
        self.root = self.run_root / "plattli"
        _maybe_resume_finalized(self.run_root, self.root, allow_resume_finalized)
        self.root.mkdir(parents=True, exist_ok=True)

        self.step = int(step)
        assert self.step >= 0, f"step must be >= 0 for run {self.run_root.name}: {self.step}"

        self.hotsize = int(hotsize)
        if self.hotsize <= 0:
            raise ValueError(f"hotsize must be > 0 for run {self.run_root.name}: {self.hotsize}")

        self._manifest = {}
        self._compact_executor = ThreadPoolExecutor(max_workers=1)
        self._compact_future = None
        self._step_metrics = set()
        self._monotonic = {}
        self._broken = None
        self._finished = False

        self._hot_rows = []
        self._hot_index = {}
        self._current_row = {}
        self._hot_lock = threading.Lock()
        self._hot_fh = None
        self._stored_counts = {}
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
            if _migrate_oversized_piecewise_indices(self.root, self._manifest):
                rewrite_manifest = True
            self._monotonic, monotonic_changed = _refresh_monotonic_metadata(self.root, self._manifest, self._hot_rows)
            if monotonic_changed:
                rewrite_manifest = True
        if rewrite_manifest:
            _write_manifest(self.root / "plattli.json", self._manifest)
        self.set_config(config)

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
        _validate_metric_names(metrics, self.run_root.name)
        if not metrics and flush:
            self._flush_hot()
            return
        if not metrics:
            return

        if self.step < 0 or self.step > 0xFFFFFFFF:
            raise ValueError(f"step out of uint32 range for run {self.run_root.name}: {self.step}")

        new_metric = False
        pending = None
        with self._hot_lock:  # The compaction thread reads and serializes the manifest.
            for name, value in metrics.items():
                if name == "step":
                    raise ValueError(f"metric name 'step' is reserved in run {self.run_root.name}")
                if name in self._step_metrics:
                    raise RuntimeError(f"metric already written in step {self.step} for {name} in run {self.run_root.name}")
                if name not in self._manifest:
                    dtype = _resolve_dtype(value, name=name, run_name=self.run_root.name)
                    (self.root / name).parent.mkdir(parents=True, exist_ok=True)
                    self._manifest[name] = {"indices": [], "dtype": dtype}
                    new_metric = True
                else:
                    dtype = self._manifest[name]["dtype"]
                value = _coerce_stored_value(value, dtype, name, self.run_root.name)
                if _track_monotonic_value(self._manifest, self._monotonic, name, value):
                    # Safe to defer: flags are persisted by the next compaction before any
                    # values land in columnar files, and recomputed from data on resume.
                    self._manifest_dirty = True
                self._current_row[name] = value
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
        if self._current_row or self.step in self._hot_index:
            self._flush_hot()
        self._step_metrics.clear()
        self._current_row = {}
        self.step += 1

    def finish(self, optimize=True, zip=True):
        self._check_finished()
        self._check_broken()
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
            _optimize_indices(self.root, self._manifest)

        _close_open_indices(self.root, self._manifest)
        _write_manifest(self.root / "plattli.json", self._manifest,
                        run_rows=max((_indices_length(self.root, name, spec)
                                      for name, spec in self._manifest.items()), default=0))

        if zip:
            _zip_output(self.run_root, self.root)
        if self._compact_executor:
            self._compact_executor.shutdown(wait=True)
            self._compact_executor = None
        self._finished = True

    def set_config(self, config):
        self._check_finished()
        _write_config(self.run_root, self.root / "config.json", config)

    def _check_finished(self):
        if self._finished:
            raise RuntimeError(f"writer for run {self.run_root.name} is already finished")

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
        # The fsyncs happen here, outside _hot_lock, so they never stall the other thread.
        with self._manifest_io_lock:
            if version <= self._manifest_disk_version:
                return  # A newer serialization already landed.
            _replace_text_checked(self.root / "plattli.json", payload, "manifest")
            self._manifest_disk_version = version

    def _flush_hot(self):
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
            for name in columns:
                dtype = self._manifest[name]["dtype"]
                indices_spec = self._manifest[name]["indices"]
                if indices_spec != "indices":
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
            try:
                old_spec = json.dumps(indices_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                current_count = self._stored_counts.get(name)
                if current_count is None:
                    current_count = _stored_values_count(self.root, name, dtype, allow_missing=True)
                spec = _append_steps_to_indices_spec(
                    indices_spec, col["indices"], count=current_count, open_tail=True
                )
                new_counts[name] = current_count + len(col["indices"])
            except (ValueError, RuntimeError) as exc:
                raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
            if _segments_too_many(spec, total_count=new_counts[name]):
                try:
                    indices = _segments_to_array(spec, total_count=new_counts[name])
                except (ValueError, RuntimeError) as exc:
                    raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
                idx_path = self.root / f"{name}.indices"
                with idx_path.open("wb") as fh:
                    indices.tofile(fh)
                spec_updates[name] = "indices"
            elif json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")) != old_spec:
                spec_updates[name] = spec

        # One coalesced manifest write per batch; serialize under the lock, fsync outside.
        pending = None
        with self._hot_lock:
            for name, spec in spec_updates.items():
                self._manifest[name]["indices"] = spec
            if spec_updates or self._manifest_dirty:
                pending = self._serialize_manifest_locked()
        if pending:
            self._persist_manifest(*pending)

        for name, col in columns.items():
            dtype, _ = metas[name]
            if dtype == JSONL_DTYPE:
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
    if not np.issubdtype(array.dtype, np.integer):
        return array  # pragma: no cover - unreachable

    amin, amax = array.min(), array.max()

    if amin >= 0:
        for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
            if amax <= np.iinfo(dt).max:
                return array.astype(dt, copy=False)
        return array.astype(np.uint64, copy=False)  # pragma: no cover - unreachable

    for dt in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dt)
        if info.min <= amin and amax <= info.max:
            return array.astype(dt, copy=False)

    return array  # pragma: no cover - unreachable


def _tight_dtype(array):
    array = np.asarray(array)
    if array.dtype.kind == "f":
        return array
    if array.dtype.kind in "iu":
        return _tightest_int(array)
    return None  # pragma: no cover - unreachable


def _append_numeric(path, values, dtype):
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"expected 1d values for {path} (shape {arr.shape})")
    with path.open("ab") as fh:
        np.asarray(arr, dtype=DTYPE_TO_NUMPY[dtype]).tofile(fh)


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
    if hasattr(value, "__array__"):
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
