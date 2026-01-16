import json
import zipfile
from pathlib import Path

import numpy as np

from .writer import DTYPE_TO_NUMPY


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
        self._rows = None
        self._when_exported = None
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

    def _zip_size(self, name):
        return self._zip.getinfo(name).file_size

    def _ensure_manifest(self):
        if self._manifest is not None:
            return
        manifest = json.loads(self._read_text("plattli.json"))
        self._rows = manifest.pop("run_rows", None)
        self._when_exported = manifest.pop("when_exported", None)
        self._manifest = manifest

    def _metric_spec(self, name):
        self._ensure_manifest()
        if name not in self._manifest:
            raise KeyError(f"unknown metric: {name}")
        return self._manifest[name]

    def config(self):
        if self._config is None:
            self._config = json.loads(self._read_text("config.json"))
        return self._config

    def when_exported(self):
        self._ensure_manifest()
        return self._when_exported

    def rows(self):
        self._ensure_manifest()
        if self._rows is not None:
            return self._rows
        run_rows = 0
        for name, spec in self._manifest.items():
            indices_spec = spec.get("indices")
            if isinstance(indices_spec, dict):
                start = int(indices_spec.get("start") or 0)
                stop = int(indices_spec.get("stop") or 0)
                step = int(indices_spec.get("step") or 1)
                if step <= 0 or stop <= start:
                    rows = 0
                else:
                    rows = (stop - start + step - 1) // step
            elif indices_spec == "indices":
                if self.kind == "zip":
                    size = self._zip_size(f"{name}.indices")
                else:
                    size = (self.root / f"{name}.indices").stat().st_size
                if size % 4:
                    raise ValueError(f"invalid indices file size for {name}: {size}")
                rows = size // 4
            else:
                raise RuntimeError(f"invalid indices spec for {name}: {indices_spec}")
            if rows > run_rows:
                run_rows = rows
        self._rows = run_rows
        return run_rows

    def metrics(self):
        self._ensure_manifest()
        return sorted(self._manifest.keys())

    def metric_indices(self, name):
        spec = self._metric_spec(name)
        indices_spec = spec.get("indices")
        if isinstance(indices_spec, dict):
            start = int(indices_spec.get("start") or 0)
            stop = int(indices_spec.get("stop") or 0)
            step = int(indices_spec.get("step") or 1)
            if step <= 0 or stop <= start:
                return np.asarray([], dtype=np.uint32)
            return np.arange(start, stop, step, dtype=np.uint32)
        if indices_spec == "indices":
            if self.kind == "zip":
                return np.frombuffer(self._read_bytes(f"{name}.indices"), dtype=np.uint32)
            return np.fromfile(self.root / f"{name}.indices", dtype=np.uint32)
        raise RuntimeError(f"invalid indices spec for {name}: {indices_spec}")

    def metric_values(self, name):
        spec = self._metric_spec(name)
        dtype = spec.get("dtype")
        if dtype == "json":
            return np.asarray(json.loads(self._read_text(f"{name}.json")), dtype=object)
        if dtype not in DTYPE_TO_NUMPY:
            raise ValueError(f"unsupported dtype for {name}: {dtype}")
        if self.kind == "zip":
            return np.frombuffer(self._read_bytes(f"{name}.{dtype}"), dtype=DTYPE_TO_NUMPY[dtype])
        return np.fromfile(self.root / f"{name}.{dtype}", dtype=DTYPE_TO_NUMPY[dtype])

    def metric(self, name, idx=None):
        if idx is None:
            return self.metric_indices(name), self.metric_values(name)
        indices = self.metric_indices(name)
        values = self.metric_values(name)
        return indices[idx], values[idx]
