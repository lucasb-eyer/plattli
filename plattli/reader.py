import json
import zipfile
from pathlib import Path

import numpy as np

from ._indices import _segments_count_and_last, _segments_from_spec, _segments_to_array
from .writer import DTYPE_TO_NUMPY, HOT_FILENAME, JSONL_DTYPE


def has_plattli(path):
    return _resolve_plattli(path)[0]


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
    dir_path = target / "plattli"

    direct_ok = (target / "plattli.json").is_file()
    dir_ok = (dir_path / "plattli.json").is_file()
    zip_ok = _is_plattli_zip(zip_path)

    if zip_ok:
        return "zip", zip_path.resolve()
    if direct_ok:
        return "dir", target.resolve()
    if dir_ok:
        return "dir", dir_path.resolve()

    return None, None


class Reader:
    def __init__(self, path):
        kind, root = _resolve_plattli(path)
        if kind is None:
            raise FileNotFoundError(f"not a plattli run: {path}")
        self.kind = kind
        self.root = root
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

    def _ensure_manifest(self):
        if self._manifest is not None:
            return
        manifest = json.loads(self._read_text("plattli.json"))
        self._run_rows = manifest.pop("run_rows", None)
        self._when_exported = manifest.pop("when_exported", None)
        self._manifest = manifest

    def _ensure_hot(self):
        if self._hot_columns is not None:
            return
        self._hot_columns = {}
        if self.kind != "dir":
            self._hot_has_file = False
            return
        hot_path = self.root / HOT_FILENAME
        self._hot_has_file = hot_path.exists()
        if not self._hot_has_file:
            return
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
    def _metric_spec(self, name, allow_hot=False):
        self._ensure_manifest()
        if name in self._manifest:
            return self._manifest[name]
        if allow_hot:
            self._ensure_hot()
            if name in self._hot_columns:
                return None
        raise KeyError(f"unknown metric: {name}")

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
                segments = _segments_from_spec(indices_spec)
                count, _ = _segments_count_and_last(segments)
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
            if info.file_size % 4:
                raise ValueError(f"invalid indices file size for {indices_metric}: {info.file_size}")
            return info.file_size // 4
        path = self.root / f"{indices_metric}.indices"
        size = path.stat().st_size
        if size % 4:
            raise ValueError(f"invalid indices file size for {indices_metric}: {size}")
        return size // 4

    def metrics(self):
        self._ensure_manifest()
        self._ensure_hot()
        return sorted(set(self._manifest.keys()) | set(self._hot_columns.keys()))

    def _columnar_count_and_last_step(self, name, spec):
        if spec is None:
            return 0, None
        indices_spec = spec.get("indices")
        if isinstance(indices_spec, (list, dict)):
            segments = _segments_from_spec(indices_spec)
            return _segments_count_and_last(segments)
        if indices_spec == "indices":
            if self.kind == "zip":
                data = self._read_bytes(f"{name}.indices")
                size = len(data)
                if size % 4:
                    raise ValueError(f"invalid indices file size for {name}: {size}")
                count = size // 4
                if count == 0:
                    return 0, None
                last_step = int(np.frombuffer(data[-4:], dtype=np.uint32)[0])
                return count, last_step
            path = self.root / f"{name}.indices"
            if not path.exists():
                self._ensure_hot()
                if self._hot_has_file:
                    return 0, None
                raise FileNotFoundError(f"missing indices file for {name}")
            size = path.stat().st_size
            if size % 4:
                raise ValueError(f"invalid indices file size for {name}: {size}")
            count = size // 4
            if count == 0:
                return 0, None
            with path.open("rb") as fh:
                fh.seek(-4, 2)
                last_step = int(np.frombuffer(fh.read(4), dtype=np.uint32)[0])
            return count, last_step
        raise RuntimeError(f"invalid indices spec for {name}: {indices_spec}")

    def _columnar_indices(self, name, spec):
        if spec is None:
            return np.asarray([], dtype=np.uint32)
        indices_spec = spec.get("indices")
        if isinstance(indices_spec, (list, dict)):
            segments = _segments_from_spec(indices_spec)
            return _segments_to_array(segments)
        if indices_spec == "indices":
            if self.kind == "zip":
                return np.frombuffer(self._read_bytes(f"{name}.indices"), dtype=np.uint32)
            path = self.root / f"{name}.indices"
            if not path.exists():
                self._ensure_hot()
                if self._hot_has_file:
                    return np.asarray([], dtype=np.uint32)
                raise FileNotFoundError(f"missing indices file for {name}")
            return np.fromfile(path, dtype=np.uint32)
        raise RuntimeError(f"invalid indices spec for {name}: {indices_spec}")

    def _columnar_values(self, name, spec):
        if spec is None:
            return np.asarray([], dtype=object)
        dtype = spec.get("dtype")
        if dtype == JSONL_DTYPE:
            if self.kind == "zip":
                values = [json.loads(line) for line in self._read_bytes(f"{name}.jsonl").splitlines()]
                return np.asarray(values, dtype=object)
            path = self.root / f"{name}.jsonl"
            if not path.exists():
                self._ensure_hot()
                if self._hot_has_file:
                    return np.asarray([], dtype=object)
                raise FileNotFoundError(f"missing values file for {name}")
            with path.open("r", encoding="utf-8") as fh:
                return np.asarray([json.loads(line) for line in fh], dtype=object)
        if dtype not in DTYPE_TO_NUMPY:
            raise ValueError(f"unsupported dtype for {name}: {dtype}")
        if self.kind == "zip":
            return np.frombuffer(self._read_bytes(f"{name}.{dtype}"), dtype=DTYPE_TO_NUMPY[dtype])
        path = self.root / f"{name}.{dtype}"
        if not path.exists():
            self._ensure_hot()
            if self._hot_has_file:
                return np.asarray([], dtype=DTYPE_TO_NUMPY[dtype])
            raise FileNotFoundError(f"missing values file for {name}")
        return np.fromfile(path, dtype=DTYPE_TO_NUMPY[dtype])

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
