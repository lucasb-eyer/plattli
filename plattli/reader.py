import json
import zipfile
from pathlib import Path

import numpy as np

from ._indices import (
    _segments_count_and_last,
    _segments_from_spec,
    _segments_have_open_tail,
    _segments_to_array,
    _segments_with_counts,
)
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

    def _selector_kind(self, start, stop, istart, istop, vstart, vstop):
        kinds = []
        if start is not None or stop is not None:
            kinds.append("step")
        if istart is not None or istop is not None:
            kinds.append("position")
        if vstart is not None or vstop is not None:
            kinds.append("value")
        if len(kinds) > 1:
            raise ValueError("Use only one range selector: start/stop, istart/istop, or vstart/vstop.")
        return kinds[0] if kinds else None

    def _position_slice(self, count, istart, istop):
        istart = 0 if istart is None else int(istart)
        istop = count if istop is None else int(istop)
        if istart < 0 or istop < 0:
            raise ValueError("istart/istop must be non-negative.")
        istart = min(istart, count)
        istop = min(max(istop, istart), count)
        return [(istart, istop)] if istart < istop else []

    def _ceil_div(self, value, step):
        return -(-value // step)

    def _read_indices_file(self, name, count):
        if self.kind == "zip":
            data = self._read_bytes(f"{name}.indices")
            data = data[:self._trim_size(len(data), 4)]
            if not data:
                return np.asarray([], dtype=np.uint32)
            return np.frombuffer(data[:count * 4], dtype=np.uint32)
        path = self.root / f"{name}.indices"
        if not path.exists():
            if self._ensure_hot():
                return np.asarray([], dtype=np.uint32)
            raise FileNotFoundError(f"missing indices file for {name} in run {self._run_name}")
        with path.open("rb") as fh:
            data = fh.read(count * 4)
        data = data[:self._trim_size(len(data), 4)]
        if not data:
            return np.asarray([], dtype=np.uint32)
        return np.frombuffer(data, dtype=np.uint32)

    def _step_chunks_for_spec(self, name, indices_spec, start, stop, count):
        if start is not None and stop is not None and start > stop:
            return []
        if isinstance(indices_spec, (list, dict)):
            chunks = []
            offset = 0
            try:
                segments = _segments_from_spec(indices_spec)
                counted = _segments_with_counts(segments, total_count=count if _segments_have_open_tail(segments) else None)
                for seg_start, raw_count, seg_step, _ in counted:
                    seg_count = min(raw_count, max(0, count - offset))
                    if seg_count <= 0:
                        break
                    left = 0 if start is None else self._ceil_div(int(np.ceil(start - seg_start)), seg_step)
                    right = seg_count if stop is None else int(np.floor((stop - seg_start) / seg_step)) + 1
                    left = min(max(left, 0), seg_count)
                    right = min(max(right, 0), seg_count)
                    if left < right:
                        chunks.append((offset + left, offset + right))
                    offset += seg_count
            except (ValueError, RuntimeError) as exc:
                raise type(exc)(f"{exc} (metric {name}, run {self._run_name})") from exc
            return chunks
        if indices_spec == "indices":
            indices = self._read_indices_file(name, count)
            left = 0 if start is None else int(np.searchsorted(indices, start, side="left"))
            right = len(indices) if stop is None else int(np.searchsorted(indices, stop, side="right"))
            return [(left, right)] if left < right else []
        raise RuntimeError(f"invalid indices spec: {indices_spec}")

    def _metric_count(self, name, spec):
        if spec is None:
            return 0
        indices_count, _ = self._indices_count_and_last(name, spec.get("indices"))
        if indices_count == 0:
            return 0
        return min(indices_count, self._values_count(name, spec))

    def _read_indices_slice(self, name, offset, count):
        if count <= 0:
            return np.asarray([], dtype=np.uint32)
        if self.kind == "zip":
            data = self._read_bytes(f"{name}.indices")
            data = data[offset * 4:(offset + count) * 4]
        else:
            path = self.root / f"{name}.indices"
            if not path.exists():
                if self._ensure_hot():
                    return np.asarray([], dtype=np.uint32)
                raise FileNotFoundError(f"missing indices file for {name} in run {self._run_name}")
            with path.open("rb") as fh:
                fh.seek(offset * 4)
                data = fh.read(count * 4)
        data = data[:self._trim_size(len(data), 4)]
        if not data:
            return np.asarray([], dtype=np.uint32)
        return np.frombuffer(data, dtype=np.uint32)

    def _read_value_slice(self, name, spec, offset, count):
        dtype = spec.get("dtype")
        if count <= 0:
            if dtype == JSONL_DTYPE:
                return np.asarray([], dtype=object)
            if dtype not in DTYPE_TO_NUMPY:
                raise ValueError(f"unsupported dtype for {name} in run {self._run_name}: {dtype}")
            return np.asarray([], dtype=DTYPE_TO_NUMPY[dtype])
        if dtype == JSONL_DTYPE:
            return self._columnar_values(name, spec)[offset:offset + count]
        if dtype not in DTYPE_TO_NUMPY:
            raise ValueError(f"unsupported dtype for {name} in run {self._run_name}: {dtype}")
        target = DTYPE_TO_NUMPY[dtype]
        itemsize = np.dtype(target).itemsize
        if self.kind == "zip":
            data = self._read_bytes(f"{name}.{dtype}")
            data = data[offset * itemsize:(offset + count) * itemsize]
        else:
            path = self.root / f"{name}.{dtype}"
            if not path.exists():
                if self._ensure_hot():
                    return np.asarray([], dtype=target)
                raise FileNotFoundError(f"missing values file for {name} in run {self._run_name}")
            with path.open("rb") as fh:
                fh.seek(offset * itemsize)
                data = fh.read(count * itemsize)
        data = data[:self._trim_size(len(data), itemsize)]
        if not data:
            return np.asarray([], dtype=target)
        return np.frombuffer(data, dtype=target)

    def _concat(self, pieces, dtype):
        pieces = [piece for piece in pieces if len(piece)]
        if not pieces:
            return np.asarray([], dtype=dtype)
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces)

    def _indices_chunks_from_segments(self, indices_spec, chunks):
        segments = _segments_from_spec(indices_spec)
        total_count = max((stop for _, stop in chunks), default=0) if _segments_have_open_tail(segments) else None
        pieces = []
        for chunk_start, chunk_stop in chunks:
            offset = 0
            for start, count, step, _ in _segments_with_counts(segments, total_count=total_count):
                left = max(chunk_start - offset, 0)
                right = min(chunk_stop - offset, count)
                if left < right:
                    pieces.append(np.arange(start + left * step, start + right * step, step, dtype=np.uint32))
                offset += count
                if offset >= chunk_stop:
                    break
        return self._concat(pieces, np.uint32)

    def _columnar_indices_chunks(self, name, spec, chunks):
        if spec is None:
            return np.asarray([], dtype=np.uint32)
        indices_spec = spec.get("indices")
        if isinstance(indices_spec, (list, dict)):
            return self._indices_chunks_from_segments(indices_spec, chunks)
        if indices_spec == "indices":
            return self._concat(
                [self._read_indices_slice(name, start, stop - start) for start, stop in chunks],
                np.uint32,
            )
        raise RuntimeError(f"invalid indices spec for {name} in run {self._run_name}: {indices_spec}")

    def _columnar_values_chunks(self, name, spec, chunks):
        dtype = spec.get("dtype")
        return self._concat(
            [self._read_value_slice(name, spec, start, stop - start) for start, stop in chunks],
            object if dtype == JSONL_DTYPE else DTYPE_TO_NUMPY[dtype],
        )

    def _monotonic_value_chunks(self, name, spec, vstart, vstop, count):
        if count <= 0:
            return []
        if vstart is not None and vstop is not None and vstart > vstop:
            return []
        if self.kind != "dir":
            return None
        dtype = spec.get("dtype")
        if dtype not in DTYPE_TO_NUMPY:
            return None
        direction = spec.get("monotonic")
        if direction not in ("inc", "dec"):
            return None

        path = self.root / f"{name}.{dtype}"
        target = DTYPE_TO_NUMPY[dtype]
        itemsize = np.dtype(target).itemsize
        with path.open("rb") as fh:
            def value_at(pos):
                fh.seek(pos * itemsize)
                return np.frombuffer(fh.read(itemsize), dtype=target)[0]

            def first_true(pred):
                lo, hi = 0, count
                while lo < hi:
                    mid = (lo + hi) // 2
                    if pred(value_at(mid)):
                        hi = mid
                    else:
                        lo = mid + 1
                return lo

            if direction == "inc":
                left = 0 if vstart is None else first_true(lambda value: value >= vstart)
                right = count if vstop is None else first_true(lambda value: value > vstop)
            else:
                left = 0 if vstop is None else first_true(lambda value: value <= vstop)
                right = count if vstart is None else first_true(lambda value: value < vstart)
        return [(left, right)] if left < right else []

    def _selector_chunks(self, name, spec, start, stop, istart, istop, vstart, vstop):
        kind = self._selector_kind(start, stop, istart, istop, vstart, vstop)
        if kind is None:
            return None
        if spec is None or self._ensure_hot():
            return None
        count = self._metric_count(name, spec)
        if kind == "position":
            return self._position_slice(count, istart, istop)
        if kind == "step":
            return self._step_chunks_for_spec(name, spec.get("indices"), start, stop, count)
        return self._monotonic_value_chunks(name, spec, vstart, vstop, count)

    def _apply_selector(self, indices, values, start, stop, istart, istop, vstart, vstop):
        kind = self._selector_kind(start, stop, istart, istop, vstart, vstop)
        if kind is None:
            return indices, values
        if kind == "position":
            chunks = self._position_slice(len(values), istart, istop)
            if not chunks:
                return indices[:0], values[:0]
            start, stop = chunks[0]
            return indices[start:stop], values[start:stop]
        if kind == "step":
            mask = np.ones(len(indices), dtype=bool)
            if start is not None:
                mask &= indices >= start
            if stop is not None:
                mask &= indices <= stop
            return indices[mask], values[mask]
        mask = np.ones(len(values), dtype=bool)
        if vstart is not None:
            mask &= values >= vstart
        if vstop is not None:
            mask &= values <= vstop
        return indices[mask], values[mask]

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
                total_count = self._values_count(name, self._manifest[name]) if _segments_have_open_tail(segments) else None
                return _segments_count_and_last(segments, total_count=total_count)
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
        data = hot_path.read_bytes()
        lines = data.splitlines()
        for idx, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if idx == len(lines) - 1:
                    break
                raise
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

    def manifest(self):
        self._ensure_manifest()
        return self._manifest

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
                    total_count = self._values_count(name, spec) if _segments_have_open_tail(segments) else None
                    count, _ = _segments_count_and_last(segments, total_count=total_count)
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
                total_count = values_count if _segments_have_open_tail(segments) else None
                for start, seg_count, step, _ in _segments_with_counts(segments, total_count=total_count):
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
                total_count = values_count if _segments_have_open_tail(segments) else None
                indices = _segments_to_array(segments, total_count=total_count)
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

    def _metric_indices_full(self, name):
        spec = self._metric_spec(name, allow_hot=True)
        columnar = self._columnar_indices(name, spec)
        last_step = int(columnar[-1]) if columnar.size else None
        hot_idx, _ = self._hot_for_metric(name, last_step)
        if hot_idx.size == 0:
            return columnar
        if columnar.size == 0:
            return hot_idx
        return np.concatenate([columnar, hot_idx])

    def _metric_values_full(self, name):
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

    def _metric_full(self, name):
        return self._metric_indices_full(name), self._metric_values_full(name)

    def metric_indices(self, name, start=None, stop=None, istart=None, istop=None, vstart=None, vstop=None):
        if self._selector_kind(start, stop, istart, istop, vstart, vstop) is None:
            return self._metric_indices_full(name)
        spec = self._metric_spec(name, allow_hot=True)
        chunks = self._selector_chunks(name, spec, start, stop, istart, istop, vstart, vstop)
        if chunks is not None:
            return self._columnar_indices_chunks(name, spec, chunks)
        indices, values = self._metric_full(name)
        return self._apply_selector(indices, values, start, stop, istart, istop, vstart, vstop)[0]

    def metric_values(self, name, start=None, stop=None, istart=None, istop=None, vstart=None, vstop=None):
        if self._selector_kind(start, stop, istart, istop, vstart, vstop) is None:
            return self._metric_values_full(name)
        spec = self._metric_spec(name, allow_hot=True)
        chunks = self._selector_chunks(name, spec, start, stop, istart, istop, vstart, vstop)
        if chunks is not None:
            return self._columnar_values_chunks(name, spec, chunks)
        indices, values = self._metric_full(name)
        return self._apply_selector(indices, values, start, stop, istart, istop, vstart, vstop)[1]

    def metric(self, name, idx=None, start=None, stop=None, istart=None, istop=None, vstart=None, vstop=None):
        if self._selector_kind(start, stop, istart, istop, vstart, vstop) is None:
            indices, values = self._metric_full(name)
        else:
            spec = self._metric_spec(name, allow_hot=True)
            chunks = self._selector_chunks(name, spec, start, stop, istart, istop, vstart, vstop)
            if chunks is None:
                indices, values = self._apply_selector(
                    *self._metric_full(name),
                    start, stop, istart, istop, vstart, vstop,
                )
            else:
                indices = self._columnar_indices_chunks(name, spec, chunks)
                values = self._columnar_values_chunks(name, spec, chunks)
        if idx is None:
            return indices, values
        return indices[idx], values[idx]
