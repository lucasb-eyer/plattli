import json
import zipfile
from pathlib import Path

import numpy as np

from ._indices import _segment_values, _segments_count_and_last, _segments_from_spec, _segments_to_array
from .writer import DTYPE_TO_NUMPY, HOT_FILENAME, JSONL_DTYPE

def is_run(path):
    return _resolve_plattli(path)[0] is not None

def resolve_run_dir(path):
    target = Path(path).expanduser()
    if not target.is_dir():
        return None
    if (target / "plattli.json").is_file():
        return target.resolve()
    plattli_dir = target / "plattli"
    if (plattli_dir / "plattli.json").is_file():
        return plattli_dir.resolve()
    return None

def is_run_dir(path):
    return resolve_run_dir(path) is not None


def _is_plattli_zip(path):
    if not path or not path.is_file():
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            zf.getinfo("plattli.json")
    except Exception:
        return False
    return True


def _resolve_plattli(path):
    target = Path(path).expanduser()

    if target.is_file():
        if _is_plattli_zip(target):
            return "zip", target.resolve()
        return None, None

    if not target.is_dir():
        return None, None

    zip_path = target / "metrics.plattli"
    if _is_plattli_zip(zip_path):
        return "zip", zip_path.resolve()
    dir_path = target / "plattli"
    direct_ok = (target / "plattli.json").is_file()
    dir_ok = (dir_path / "plattli.json").is_file()

    if direct_ok:
        return "dir", target.resolve()
    if dir_ok:
        return "dir", dir_path.resolve()

    return None, None


def _run_name_for_root(root):
    if root.is_file():
        if root.name == "metrics.plattli":
            return root.parent.name
        if root.suffix == ".plattli":
            return root.stem
        return root.name
    if root.name == "plattli":
        return root.parent.name
    return root.name


class Reader:
    def __init__(self, path):
        kind, root = _resolve_plattli(path)
        if kind is None:
            raise FileNotFoundError(f"not a plattli run: {path}")
        self.kind = kind
        self.root = root
        self._run_name = _run_name_for_root(root)
        self._zip = None
        self._manifest = None
        self._config = None
        self._run_rows = None
        self._when_exported = None
        self._hot_columns = None
        self._hot_has_file = None
        self._rows_cache = {}
        if self.kind == "zip":
            self._zip = zipfile.ZipFile(self.root)

    def close(self):
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _read_text(self, name):
        if self.kind == "zip":
            return self._zip.read(name).decode("utf-8")
        return (self.root / name).read_text(encoding="utf-8")

    def _read_bytes(self, name):
        if self.kind == "zip":
            return self._zip.read(name)
        return (self.root / name).read_bytes()

    def _trim_size(self, size, unit):
        if size <= 0:
            return 0
        return size - (size % unit)

    def _read_jsonl_values(self, name):
        if self.kind == "zip":
            data = self._read_bytes(f"{name}.jsonl")
        else:
            path = self.root / f"{name}.jsonl"
            if not path.exists():
                if self._ensure_hot():
                    return []
                raise FileNotFoundError(f"missing values file for {name} in run {self._run_name}")
            data = path.read_bytes()
        if not data:
            return []
        lines = data.splitlines()
        if not lines:
            return []
        values = []
        for idx, line in enumerate(lines):
            try:
                values.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if idx == len(lines) - 1:
                    break
                raise
        return values

    def _indices_count_and_last(self, name, indices_spec):
        if isinstance(indices_spec, (list, dict)):
            try:
                segments = _segments_from_spec(indices_spec)
                return _segments_count_and_last(segments)
            except (ValueError, RuntimeError) as exc:
                raise type(exc)(f"{exc} (metric {name}, run {self._run_name})") from exc
        if indices_spec == "indices":
            if self.kind == "zip":
                data = self._read_bytes(f"{name}.indices")
                valid = self._trim_size(len(data), 4)
                count = valid // 4
                if count == 0:
                    return 0, None
                last = int(np.frombuffer(data[valid - 4:valid], dtype=np.uint32)[0])
                return count, last
            path = self.root / f"{name}.indices"
            if not path.exists():
                if self._ensure_hot():
                    return 0, None
                raise FileNotFoundError(f"missing indices file for {name} in run {self._run_name}")
            size = path.stat().st_size
            valid = self._trim_size(size, 4)
            count = valid // 4
            if count == 0:
                return 0, None
            with path.open("rb") as fh:
                fh.seek(valid - 4)
                last = int(np.frombuffer(fh.read(4), dtype=np.uint32)[0])
            return count, last
        raise RuntimeError(f"invalid indices spec for {name} in run {self._run_name}: {indices_spec}")

    def _values_count(self, name, spec):
        dtype = spec.get("dtype")
        if dtype == JSONL_DTYPE:
            return len(self._read_jsonl_values(name))
        if dtype not in DTYPE_TO_NUMPY:
            raise ValueError(f"unsupported dtype for {name} in run {self._run_name}: {dtype}")
        itemsize = np.dtype(DTYPE_TO_NUMPY[dtype]).itemsize
        if self.kind == "zip":
            data = self._read_bytes(f"{name}.{dtype}")
            valid = self._trim_size(len(data), itemsize)
            return valid // itemsize
        path = self.root / f"{name}.{dtype}"
        if not path.exists():
            if self._ensure_hot():
                return 0
            raise FileNotFoundError(f"missing values file for {name} in run {self._run_name}")
        size = path.stat().st_size
        valid = self._trim_size(size, itemsize)
        return valid // itemsize

    def _ensure_manifest(self):
        if self._manifest is not None:
            return
        manifest = json.loads(self._read_text("plattli.json"))
        self._run_rows = manifest.pop("run_rows", None)
        self._when_exported = manifest.pop("when_exported", None)
        self._manifest = manifest

    def _ensure_hot(self):
        if self._hot_columns is not None:
            return self._hot_has_file
        self._hot_columns = {}
        if self.kind != "dir":
            self._hot_has_file = False
            return False
        hot_path = self.root / HOT_FILENAME
        self._hot_has_file = hot_path.exists()
        if not self._hot_has_file:
            return False
        with hot_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                step = int(row["step"])
                for name, value in row.items():
                    if name == "step":
                        continue
                    col = self._hot_columns.get(name)
                    if col is None:
                        col = {"indices": [], "values": []}
                        self._hot_columns[name] = col
                    col["indices"].append(step)
                    col["values"].append(value)
        return True

    def _metric_spec(self, name, allow_hot=False):
        self._ensure_manifest()
        if name in self._manifest:
            return self._manifest[name]
        if allow_hot:
            self._ensure_hot()
            if name in self._hot_columns:
                return None
        raise KeyError(f"unknown metric {name} in run {self._run_name}")

    def config(self):
        if self._config is None:
            self._config = json.loads(self._read_text("config.json"))
        return self._config

    def when_exported(self):
        self._ensure_manifest()
        return self._when_exported

    def rows(self, name):
        if name in self._rows_cache:
            return self._rows_cache[name]
        self._ensure_hot()
        spec = self._metric_spec(name, allow_hot=True)
        columnar_count, last_step = self._columnar_count_and_last_step(name, spec)
        hot_count = 0
        if name in self._hot_columns:
            if last_step is None:
                hot_count = len(self._hot_columns[name]["indices"])
            else:
                hot_count = sum(1 for step in self._hot_columns[name]["indices"] if step > last_step)
        rows = columnar_count + hot_count
        self._rows_cache[name] = rows
        return rows

    def approx_max_rows(self, faster=True):
        self._ensure_manifest()
        if self._run_rows is not None:
            return self._run_rows

        max_rows = 0
        indices_metric = None
        for name, spec in self._manifest.items():
            indices_spec = spec.get("indices")
            if isinstance(indices_spec, (list, dict)):
                try:
                    segments = _segments_from_spec(indices_spec)
                    count, _ = _segments_count_and_last(segments)
                except (ValueError, RuntimeError) as exc:
                    raise type(exc)(f"{exc} (metric {name}, run {self._run_name})") from exc
                if count > max_rows:
                    max_rows = count
            elif indices_spec == "indices" and indices_metric is None:
                indices_metric = name

        if not faster:
            self._ensure_hot()
            if self._hot_columns:
                hot_metrics = sorted(self._hot_columns.items(),
                                     key=lambda item: len(item[1]["indices"]),
                                     reverse=True)
                for name, _ in hot_metrics[:2]:
                    rows = self.rows(name)
                    if rows > max_rows:
                        max_rows = rows

        if max_rows:
            return max_rows
        if indices_metric is None:
            return 0
        if self.kind == "zip":
            info = self._zip.getinfo(f"{indices_metric}.indices")
            valid = self._trim_size(info.file_size, 4)
            return valid // 4
        path = self.root / f"{indices_metric}.indices"
        if not path.exists():
            if self._ensure_hot():
                return 0
            raise FileNotFoundError(f"missing indices file for {indices_metric} in run {self._run_name}")
        size = path.stat().st_size
        valid = self._trim_size(size, 4)
        return valid // 4

    def metrics(self):
        self._ensure_manifest()
        self._ensure_hot()
        return sorted(set(self._manifest.keys()) | set(self._hot_columns.keys()))

    def _columnar_count_and_last_step(self, name, spec):
        if spec is None:
            return 0, None
        indices_spec = spec.get("indices")
        indices_count, indices_last = self._indices_count_and_last(name, indices_spec)
        if indices_count == 0:
            return 0, None
        values_count = self._values_count(name, spec)
        count = min(indices_count, values_count)
        if count <= 0:
            return 0, None
        if count == indices_count:
            return count, indices_last
        idx = count - 1
        if isinstance(indices_spec, (list, dict)):
            try:
                segments = _segments_from_spec(indices_spec)
                for segment in segments:
                    start, stop, step = _segment_values(segment)
                    seg_count = (stop - start + step - 1) // step
                    if idx < seg_count:
                        return count, int(start + idx * step)
                    idx -= seg_count
            except (ValueError, RuntimeError) as exc:
                raise type(exc)(f"{exc} (metric {name}, run {self._run_name})") from exc
            raise RuntimeError(f"indices spec shorter than expected for {name} in run {self._run_name}")
        if indices_spec == "indices":
            if self.kind == "zip":
                data = self._read_bytes(f"{name}.indices")
                valid = self._trim_size(len(data), 4)
                offset = idx * 4
                if offset + 4 > valid:
                    return count, indices_last
                last_step = int(np.frombuffer(data[offset:offset + 4], dtype=np.uint32)[0])
                return count, last_step
            path = self.root / f"{name}.indices"
            if not path.exists():
                if self._ensure_hot():
                    return 0, None
                raise FileNotFoundError(f"missing indices file for {name} in run {self._run_name}")
            size = path.stat().st_size
            valid = self._trim_size(size, 4)
            offset = idx * 4
            if offset + 4 > valid:
                return count, indices_last
            with path.open("rb") as fh:
                fh.seek(offset)
                last_step = int(np.frombuffer(fh.read(4), dtype=np.uint32)[0])
            return count, last_step
        raise RuntimeError(f"invalid indices spec for {name} in run {self._run_name}: {indices_spec}")

    def _columnar_indices(self, name, spec):
        if spec is None:
            return np.asarray([], dtype=np.uint32)
        indices_spec = spec.get("indices")
        indices_count, _ = self._indices_count_and_last(name, indices_spec)
        if indices_count == 0:
            return np.asarray([], dtype=np.uint32)
        values_count = self._values_count(name, spec)
        count = min(indices_count, values_count)
        if count <= 0:
            return np.asarray([], dtype=np.uint32)
        if isinstance(indices_spec, (list, dict)):
            try:
                segments = _segments_from_spec(indices_spec)
                indices = _segments_to_array(segments)
            except (ValueError, RuntimeError) as exc:
                raise type(exc)(f"{exc} (metric {name}, run {self._run_name})") from exc
            if count < indices.size:
                return indices[:count]
            return indices
        if indices_spec == "indices":
            if self.kind == "zip":
                data = self._read_bytes(f"{name}.indices")
                valid = self._trim_size(len(data), 4)
                max_count = valid // 4
                count = min(count, max_count)
                if count <= 0:
                    return np.asarray([], dtype=np.uint32)
                return np.frombuffer(data[:count * 4], dtype=np.uint32)
            path = self.root / f"{name}.indices"
            if not path.exists():
                if self._ensure_hot():
                    return np.asarray([], dtype=np.uint32)
                raise FileNotFoundError(f"missing indices file for {name} in run {self._run_name}")
            size = path.stat().st_size
            valid = self._trim_size(size, 4)
            max_count = valid // 4
            count = min(count, max_count)
            if count <= 0:
                return np.asarray([], dtype=np.uint32)
            with path.open("rb") as fh:
                data = fh.read(count * 4)
            data = data[:self._trim_size(len(data), 4)]
            if not data:
                return np.asarray([], dtype=np.uint32)
            return np.frombuffer(data, dtype=np.uint32)
        raise RuntimeError(f"invalid indices spec for {name} in run {self._run_name}: {indices_spec}")

    def _columnar_values(self, name, spec):
        if spec is None:
            return np.asarray([], dtype=object)
        dtype = spec.get("dtype")
        if dtype == JSONL_DTYPE:
            indices_count, _ = self._indices_count_and_last(name, spec.get("indices"))
            if indices_count == 0:
                return np.asarray([], dtype=object)
            values = self._read_jsonl_values(name)
            if not values:
                return np.asarray([], dtype=object)
            count = min(indices_count, len(values))
            if count <= 0:
                return np.asarray([], dtype=object)
            if len(values) > count:
                values = values[:count]
            return np.asarray(values, dtype=object)
        if dtype not in DTYPE_TO_NUMPY:
            raise ValueError(f"unsupported dtype for {name} in run {self._run_name}: {dtype}")
        indices_count, _ = self._indices_count_and_last(name, spec.get("indices"))
        if indices_count == 0:
            return np.asarray([], dtype=DTYPE_TO_NUMPY[dtype])
        values_count = self._values_count(name, spec)
        count = min(indices_count, values_count)
        if count <= 0:
            return np.asarray([], dtype=DTYPE_TO_NUMPY[dtype])
        itemsize = np.dtype(DTYPE_TO_NUMPY[dtype]).itemsize
        if self.kind == "zip":
            data = self._read_bytes(f"{name}.{dtype}")
            valid = self._trim_size(len(data), itemsize)
            max_count = valid // itemsize
            count = min(count, max_count)
            if count <= 0:
                return np.asarray([], dtype=DTYPE_TO_NUMPY[dtype])
            return np.frombuffer(data[:count * itemsize], dtype=DTYPE_TO_NUMPY[dtype])
        path = self.root / f"{name}.{dtype}"
        if not path.exists():
            if self._ensure_hot():
                return np.asarray([], dtype=DTYPE_TO_NUMPY[dtype])
            raise FileNotFoundError(f"missing values file for {name} in run {self._run_name}")
        with path.open("rb") as fh:
            data = fh.read(count * itemsize)
        data = data[:self._trim_size(len(data), itemsize)]
        if not data:
            return np.asarray([], dtype=DTYPE_TO_NUMPY[dtype])
        return np.frombuffer(data, dtype=DTYPE_TO_NUMPY[dtype])

    def _hot_for_metric(self, name, last_step):
        self._ensure_hot()
        col = self._hot_columns.get(name)
        if not col:
            return np.asarray([], dtype=np.uint32), []
        indices = []
        values = []
        for step, value in zip(col["indices"], col["values"]):
            if last_step is None or step > last_step:
                indices.append(step)
                values.append(value)
        return np.asarray(indices, dtype=np.uint32), values

    def metric_indices(self, name):
        spec = self._metric_spec(name, allow_hot=True)
        columnar = self._columnar_indices(name, spec)
        last_step = int(columnar[-1]) if columnar.size else None
        hot_idx, _ = self._hot_for_metric(name, last_step)
        if hot_idx.size == 0:
            return columnar
        if columnar.size == 0:
            return hot_idx
        return np.concatenate([columnar, hot_idx])

    def metric_values(self, name):
        spec = self._metric_spec(name, allow_hot=True)
        columnar = self._columnar_values(name, spec)
        last_step = None
        if spec is not None:
            indices = self._columnar_indices(name, spec)
            if indices.size:
                last_step = int(indices[-1])
        _, hot_values = self._hot_for_metric(name, last_step)
        if not hot_values:
            return columnar
        if spec is None or spec.get("dtype") == JSONL_DTYPE:
            hot_arr = np.asarray(hot_values, dtype=object)
            if columnar.size == 0:
                return hot_arr
            return np.concatenate([columnar, hot_arr])
        dtype = spec.get("dtype")
        hot_arr = np.asarray(hot_values, dtype=DTYPE_TO_NUMPY[dtype])
        if columnar.size == 0:
            return hot_arr
        return np.concatenate([columnar, hot_arr])

    def metric(self, name, idx=None):
        if idx is None:
            return self.metric_indices(name), self.metric_values(name)
        indices = self.metric_indices(name)
        values = self.metric_values(name)
        return indices[idx], values[idx]
