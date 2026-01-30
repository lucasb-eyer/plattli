import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree

import numpy as np

from ._indices import (
    _append_step_to_indices_spec,
    _append_steps_to_indices_spec,
    _find_piecewise_params,
    _segments_count_and_last,
    _segments_from_spec,
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


def _zip_path_for_root(root):
    return Path(root) / "metrics.plattli"


def _run_name_for_root(root):
    root = Path(root)
    if root.name == "plattli":
        return root.parent.name
    return root.name


def _write_manifest(path, manifest, run_rows=None):
    payload = {
        **manifest,
        "when_exported": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    if run_rows is not None:
        payload["run_rows"] = run_rows
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


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
                kept, keep = _segments_truncate(segments, step)
            except (ValueError, RuntimeError) as exc:
                raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
            spec["indices"] = kept

        if (dtype := spec["dtype"]) == JSONL_DTYPE:
            path = root / f"{name}.jsonl"
            if not path.exists():
                if allow_missing:
                    continue
                raise FileNotFoundError(f"missing values file for {name} in run {run_name}")
            if keep <= 0:
                path.write_text("", encoding="utf-8")
                continue
            lines = []
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if len(lines) >= keep:
                        break
                    lines.append(line.rstrip("\n"))
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
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
            indices = _segments_to_array(segments)
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


def _indices_length(root, name, indices_spec):
    if isinstance(indices_spec, (dict, list)):
        try:
            segments = _segments_from_spec(indices_spec)
            total, _ = _segments_count_and_last(segments)
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f"{exc} (metric {name}, run {_run_name_for_root(root)})") from exc
        return total
    if indices_spec == "indices":
        idx_path = root / f"{name}.indices"
        return idx_path.stat().st_size // 4
    return 0  # pragma: no cover


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
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            out_path = root / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(zf.read(name))
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

        if (self.root / "plattli.json").exists():
            self._manifest = json.loads((self.root / "plattli.json").read_text(encoding="utf-8"))
            self._manifest.pop("when_exported", None)
            self._manifest.pop("run_rows", None)
            if self._manifest:
                _truncate_to_step(self.root, self._manifest, self.step, allow_missing=False)
                _force_indices_files(self.root, self._manifest)

        self.set_config(config)

    def write(self, **metrics):
        self._drain_errors()
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
            if name not in self._manifest:
                dtype = _resolve_dtype(value, name=name, run_name=self.run_root.name)
                (self.root / name).parent.mkdir(parents=True, exist_ok=True)
                self._manifest[name] = {"indices": "indices", "dtype": dtype}
                new_metric = True
            else:
                dtype = self._manifest[name]["dtype"]
            write_indices = self._manifest[name]["indices"] == "indices"
            if self._executor:
                self._futures.append(self._executor.submit(self._write_entry, name, dtype, value, self.step, write_indices))
            else:
                self._write_entry(name, dtype, value, self.step, write_indices)
            self._step_metrics.add(name)

        if new_metric:
            _write_manifest(self.root / "plattli.json", self._manifest)

    def end_step(self):
        wait(self._futures)
        self._drain_errors()
        self._flush_step_indices()
        self._step_metrics.clear()
        self.step += 1

    def finish(self, optimize=True, zip=True):
        if not self._manifest:
            return

        wait(self._futures)
        self._drain_errors()
        self._flush_step_indices()
        self._step_metrics.clear()

        if optimize:
            _tighten_dtypes(self.root, self._manifest)
            _optimize_indices(self.root, self._manifest)

        _write_manifest(self.root / "plattli.json", self._manifest,
                        run_rows=max(_indices_length(self.root, name, spec["indices"])
                                     for name, spec in self._manifest.items()))

        if zip:
            _zip_output(self.run_root, self.root)
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.write = self.end_step = self.set_config = None

    def set_config(self, config):
        _write_config(self.run_root, self.root / "config.json", config)

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
                    raise err
            else:
                remaining.append(f)
        self._futures = remaining


class CompactingWriter:
    def __init__(self, outdir, step=0, hotsize=None, config="config.json", allow_resume_finalized=False):
        self.run_root = Path(outdir)
        if self.run_root.name == "plattli":
            raise ValueError(f"outdir should be a run directory, not the plattli folder: {outdir}")
        self.root = self.run_root / "plattli"
        _maybe_resume_finalized(self.run_root, self.root, allow_resume_finalized)
        self.root.mkdir(parents=True, exist_ok=True)

        self.step = int(step)
        assert self.step >= 0, f"step must be >= 0 for run {self.run_root.name}: {self.step}"

        if hotsize is None:
            raise ValueError(f"hotsize is required for run {self.run_root.name}")
        self.hotsize = int(hotsize)
        if self.hotsize <= 0:
            raise ValueError(f"hotsize must be > 0 for run {self.run_root.name}: {self.hotsize}")

        self._manifest = {}
        self._compact_executor = ThreadPoolExecutor(max_workers=1)
        self._compact_future = None
        self._compact_steps = set()
        self._step_metrics = set()

        self._hot_rows = []
        self._hot_index = {}
        self._current_row = {}
        self._hot_lock = threading.Lock()

        if (self.root / "plattli.json").exists():
            self._manifest = json.loads((self.root / "plattli.json").read_text(encoding="utf-8"))
            self._manifest.pop("when_exported", None)
            self._manifest.pop("run_rows", None)
            if self._manifest:
                _truncate_to_step(self.root, self._manifest, self.step, allow_missing=True)

        self._load_hot_rows()
        self.set_config(config)

    def write(self, metrics=None, flush=False, **kwargs):
        self._drain_errors()
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, dict):
            raise TypeError(f"metrics must be a dict in run {self.run_root.name} (got {type(metrics).__name__})")
        if kwargs:
            overlap = metrics.keys() & kwargs.keys()
            if overlap:
                raise ValueError(f"duplicate metric names in run {self.run_root.name}: {sorted(overlap)}")
            metrics = {**metrics, **kwargs}
        if not metrics and flush:
            self._flush_hot()
            return
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
            if name not in self._manifest:
                dtype = _resolve_dtype(value, name=name, run_name=self.run_root.name)
                (self.root / name).parent.mkdir(parents=True, exist_ok=True)
                self._manifest[name] = {"indices": [], "dtype": dtype}
                new_metric = True
            else:
                dtype = self._manifest[name]["dtype"]
            if hasattr(value, "__array__"):
                value = np.asarray(value)
            if isinstance(value, (np.ndarray, np.generic)):
                if value.shape != ():
                    raise ValueError(f"only scalar values are supported for {name} in run {self.run_root.name} (shape {value.shape})")
                value = value.item()
            if dtype != JSONL_DTYPE:
                arr = np.asarray(value)
                if arr.shape != ():
                    raise ValueError(f"only scalar values are supported for {name} in run {self.run_root.name} (shape {arr.shape})")
                value = np.asarray(arr, dtype=DTYPE_TO_NUMPY[dtype]).item()
            self._current_row[name] = value
            self._step_metrics.add(name)

        if new_metric:
            with self._hot_lock:
                _write_manifest(self.root / "plattli.json", self._manifest)
        if flush:
            self._flush_hot()

    def end_step(self):
        if self._current_row or self.step in self._hot_index:
            self._flush_hot()
        self._step_metrics.clear()
        self._current_row = {}
        self.step += 1

    def finish(self, optimize=True, zip=True):
        if not self._manifest:
            return

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
        if hot_path.exists():
            hot_path.unlink()

        if optimize:
            _tighten_dtypes(self.root, self._manifest)
            _optimize_indices(self.root, self._manifest)

        _write_manifest(self.root / "plattli.json", self._manifest,
                        run_rows=max(_indices_length(self.root, name, spec["indices"])
                                     for name, spec in self._manifest.items()))

        if zip:
            _zip_output(self.run_root, self.root)
        if self._compact_executor:
            self._compact_executor.shutdown(wait=True)
            self._compact_executor = None
        self.write = self.end_step = self.set_config = None

    def set_config(self, config):
        _write_config(self.run_root, self.root / "config.json", config)

    def _load_hot_rows(self):
        hot_path = self.root / HOT_FILENAME
        if not hot_path.exists():
            return
        step_row = None
        new_metric = False
        rewrite_hot = False
        min_hot_step = None
        with hot_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                row_step = int(row["step"])
                if row_step > self.step:
                    rewrite_hot = True
                    continue
                if min_hot_step is None or row_step < min_hot_step:
                    min_hot_step = row_step
                if row_step == self.step:
                    step_row = row
                else:
                    self._hot_index[row_step] = len(self._hot_rows)
                    self._hot_rows.append(row)
                for name, value in row.items():
                    if name == "step":
                        continue
                    if name not in self._manifest:
                        dtype = _resolve_dtype(value, name=name, run_name=self.run_root.name)
                        (self.root / name).parent.mkdir(parents=True, exist_ok=True)
                        self._manifest[name] = {"indices": [], "dtype": dtype}
                        new_metric = True
        if step_row is not None:
            self._hot_index[self.step] = len(self._hot_rows)
            self._hot_rows.append(step_row)
            self._current_row = {k: v for k, v in step_row.items() if k != "step"}
            self._step_metrics = set(self._current_row.keys())
        if rewrite_hot:
            self._write_hot_file()
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
                    self._compact_steps = set()
                raise err

    def _flush_hot(self):
        wrote = False
        with self._hot_lock:
            if self._current_row:
                self._upsert_hot_row(self.step, self._current_row)
                wrote = True
            if self._maybe_finalize_compaction_locked():
                wrote = True
            if wrote:
                self._write_hot_file()
            if self._compact_future is None:
                batch = self._compact_batch_locked()
                if batch:
                    self._compact_steps = {int(row["step"]) for row in batch}
                    self._compact_future = self._compact_executor.submit(self._compact_rows, batch)

    def _upsert_hot_row(self, step, row):
        payload = {"step": int(step)}
        payload.update(row)
        if step in self._hot_index:
            idx = self._hot_index[step]
            self._hot_rows[idx] = payload
        else:
            self._hot_index[step] = len(self._hot_rows)
            self._hot_rows.append(payload)

    def _write_hot_file(self):
        path = self.root / HOT_FILENAME
        tmp_path = path.with_name(path.name + ".tmp")
        payload = "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in self._hot_rows)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)

    def _maybe_finalize_compaction_locked(self):
        future = self._compact_future
        if future is None or not future.done():
            return False
        if (err := future.exception()) is not None:
            self._compact_future = None
            self._compact_steps = set()
            raise err
        if self._compact_steps:
            steps = self._compact_steps
            self._hot_rows = [row for row in self._hot_rows if row["step"] not in steps]
            self._hot_index = {row["step"]: idx for idx, row in enumerate(self._hot_rows)}
            self._compact_steps = set()
        self._compact_future = None
        return True

    def _compact_batch_locked(self):
        completed = [row for row in self._hot_rows if row["step"] < self.step]
        if len(completed) < self.hotsize:
            return []
        completed.sort(key=lambda row: row["step"])
        return completed[:self.hotsize]

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

        updated_specs = {}
        run_name = self.run_root.name
        for name, col in columns.items():
            dtype = self._manifest[name]["dtype"]
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            indices_spec = self._manifest[name]["indices"]
            if indices_spec == "indices":
                _append_indices(self.root / f"{name}.indices", col["indices"])
            else:
                try:
                    spec = _append_steps_to_indices_spec(indices_spec, col["indices"])
                except (ValueError, RuntimeError) as exc:
                    raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
                if _segments_too_many(spec):
                    try:
                        indices = _segments_to_array(spec)
                    except (ValueError, RuntimeError) as exc:
                        raise type(exc)(f"{exc} (metric {name}, run {run_name})") from exc
                    idx_path = self.root / f"{name}.indices"
                    idx_path.parent.mkdir(parents=True, exist_ok=True)
                    with idx_path.open("wb") as fh:
                        indices.tofile(fh)
                    updated_specs[name] = "indices"
                else:
                    updated_specs[name] = spec
            if dtype == JSONL_DTYPE:
                _append_jsonl(self.root / f"{name}.jsonl", col["values"])
            else:
                _append_numeric(self.root / f"{name}.{dtype}", col["values"], dtype)

        with self._hot_lock:
            if updated_specs:
                for name, spec in updated_specs.items():
                    self._manifest[name]["indices"] = spec
                _write_manifest(self.root / "plattli.json", self._manifest)


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
        return array.astype(np.float32, copy=False)
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
