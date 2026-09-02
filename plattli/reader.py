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
from .writer import (
    DTYPE_TO_NUMPY,
    HOT_COMPACTING_FILENAME,
    HOT_FILENAME,
    JSONL_DTYPE,
    _validate_metric_names,
    _validate_metric_name,
)

def is_run(path):
    kind, _, zf, zip_fh = _resolve_plattli(path)
    if zf is not None:
        _close_plattli_zip(zf, zip_fh)
    return kind is not None

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


def _open_plattli_zip(path):
    """Returns an open (ZipFile, file handle) if path is a plattli zip, else None."""
    try:
        fh = path.open("rb", buffering=4096)
    except OSError:
        return None
    try:
        zf = zipfile.ZipFile(fh)
    except (zipfile.BadZipFile, OSError):
        fh.close()
        return None
    except BaseException:
        fh.close()
        raise
    try:
        zf.getinfo("plattli.json")
    except KeyError:
        _close_plattli_zip(zf, fh)
        return None
    except BaseException:
        _close_plattli_zip(zf, fh)
        raise
    return zf, fh


def _close_plattli_zip(zf, fh):
    try:
        zf.close()
    finally:
        fh.close()


def _resolve_plattli(path):
    target = Path(path).expanduser()

    if target.is_dir():
        zip_path = target / "metrics.plattli"
        if (archive := _open_plattli_zip(zip_path)) is not None:
            zf, fh = archive
            return "zip", zip_path.resolve(), zf, fh
        if (target / "plattli.json").is_file():
            return "dir", target.resolve(), None, None
        if (target / "plattli" / "plattli.json").is_file():
            return "dir", (target / "plattli").resolve(), None, None
    elif (archive := _open_plattli_zip(target)) is not None:
        zf, fh = archive
        return "zip", target.resolve(), zf, fh

    return None, None, None, None


def _run_name_for_root(root, kind):
    if kind == "zip":
        if root.name == "metrics.plattli":
            return root.parent.name
        if root.suffix == ".plattli":
            return root.stem
        return root.name
    if root.name == "plattli":
        return root.parent.name
    return root.name


class Reader:
    def __init__(self, path, kind=None, exclude_hot=False):
        if not isinstance(exclude_hot, bool):
            raise TypeError("exclude_hot must be a boolean")
        if kind is None:
            kind, root, zf, zip_fh = _resolve_plattli(path)
            if kind is None:
                raise FileNotFoundError(f"not a plattli run: {path}")
        elif kind == "dir":
            root = Path(path).expanduser().absolute()
            zf = zip_fh = None
        elif kind == "zip":
            root = Path(path).expanduser().absolute()
            if (archive := _open_plattli_zip(root)) is None:
                raise FileNotFoundError(f"not a plattli archive: {path}")
            zf, zip_fh = archive
        else:
            raise ValueError(f"invalid reader kind: {kind}")
        self._zip = zf
        self._zip_fh = zip_fh
        try:
            self.kind = kind
            self.root = root
            self.exclude_hot = exclude_hot
            self._run_name = _run_name_for_root(root, kind)
            self._manifest = None
            self._config = None
            self._run_rows = None
            self._when_exported = None
            self._hot_columns = None
            self._hot_has_file = None
            self._rows_cache = {}
            self._jsonl_cache = {}
        except BaseException:
            self.close()
            raise

    def refresh(self):
        """Drop cached metadata (manifest, config, hot rows, row counts, parsed jsonl)
        so subsequent reads pick up data written since. Zip readers are immutable
        snapshots, so refreshing them is unnecessary."""
        self._manifest = None
        self._config = None
        self._run_rows = None
        self._when_exported = None
        self._hot_columns = None
        self._hot_has_file = None
        self._rows_cache = {}
        self._jsonl_cache = {}

    def close(self):
        if self._zip is not None:
            zf = self._zip
            fh = self._zip_fh
            self._zip = None
            self._zip_fh = None
            _close_plattli_zip(zf, fh)

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

    def _parse_jsonl_bytes(self, data):
        lines = data.splitlines()
        entries = []
        for idx, line in enumerate(lines):
            try:
                entries.append(json.loads(line.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if idx == len(lines) - 1:
                    break
                raise
        return entries

    def _read_live_jsonl(self, path):
        for attempt in range(3):
            try:
                return self._parse_jsonl_bytes(path.read_bytes())
            except (json.JSONDecodeError, UnicodeDecodeError):
                if attempt == 2:
                    raise

    def _read_zip_slice(self, name, offset, size):
        # Members are ZIP_STORED, so seeking is O(1); never read more than asked for.
        with self._zip.open(name) as fh:
            if offset:
                fh.seek(offset)
            return fh.read(size)

    def _zip_member_size(self, name):
        return self._zip.getinfo(name).file_size

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
        if istart < 0:  # Negative positions count from the end, like Python slices.
            istart = max(count + istart, 0)
        if istop < 0:
            istop = max(count + istop, 0)
        istart = min(istart, count)
        istop = min(max(istop, istart), count)
        return [(istart, istop)] if istart < istop else []

    def _ceil_div(self, value, step):
        return -(-value // step)

    def _read_indices_file(self, name, count):
        if self.kind == "zip":
            data = self._read_zip_slice(f"{name}.indices", 0, count * 4)
            data = data[:self._trim_size(len(data), 4)]
            if not data:
                return np.asarray([], dtype=np.uint32)
            return np.frombuffer(data, dtype=np.uint32)
        path = self.root / f"{name}.indices"
        try:
            with path.open("rb") as fh:
                data = fh.read(count * 4)
        except FileNotFoundError:
            if self._missing_columnar_is_empty():
                return np.asarray([], dtype=np.uint32)
            raise FileNotFoundError(f"missing indices file for {name} in run {self._run_name}")
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
            data = self._read_zip_slice(f"{name}.indices", offset * 4, count * 4)
        else:
            path = self.root / f"{name}.indices"
            try:
                with path.open("rb") as fh:
                    fh.seek(offset * 4)
                    data = fh.read(count * 4)
            except FileNotFoundError:
                if self._missing_columnar_is_empty():
                    return np.asarray([], dtype=np.uint32)
                raise FileNotFoundError(f"missing indices file for {name} in run {self._run_name}")
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
            # offset+count never exceeds the metric count, so slice the cached list directly.
            return np.asarray(self._read_jsonl_values(name)[offset:offset + count], dtype=object)
        if dtype not in DTYPE_TO_NUMPY:
            raise ValueError(f"unsupported dtype for {name} in run {self._run_name}: {dtype}")
        target = DTYPE_TO_NUMPY[dtype]
        itemsize = np.dtype(target).itemsize
        if self.kind == "zip":
            data = self._read_zip_slice(f"{name}.{dtype}", offset * itemsize, count * itemsize)
        else:
            path = self.root / f"{name}.{dtype}"
            try:
                with path.open("rb") as fh:
                    fh.seek(offset * itemsize)
                    data = fh.read(count * itemsize)
            except FileNotFoundError:
                if self._missing_columnar_is_empty():
                    return np.asarray([], dtype=target)
                raise FileNotFoundError(f"missing values file for {name} in run {self._run_name}")
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

    def _metric_select(self, name, start, stop, istart, istop, vstart, vstop, want_indices=True, want_values=True):
        """Selector read as (indices, values): chunked columnar read plus in-memory hot tail.

        Returns None when the selector cannot be resolved to chunks (non-monotonic value
        selector); the caller then falls back to a full read + filter."""
        spec = self._metric_spec(name, allow_hot=True)
        # Freeze the readable columnar prefix before filtering the overlapping hot tail.
        count = self._metric_count(name, spec)
        hot_steps, hot_values = self._hot_tail(name, spec)
        kind = self._selector_kind(start, stop, istart, istop, vstart, vstop)
        if kind == "position":
            hot_keep = slice(0, 0)
            chunks = []
            for lo, hi in self._position_slice(count + len(hot_values), istart, istop):
                if lo < count:
                    chunks.append((lo, min(hi, count)))
                hot_keep = slice(max(lo - count, 0), max(hi - count, 0))
        elif kind == "step":
            chunks = self._step_chunks_for_spec(name, spec.get("indices"), start, stop, count) if spec is not None else []
            hot_keep = np.ones(len(hot_steps), dtype=bool)
            if start is not None:
                hot_keep &= hot_steps >= start
            if stop is not None:
                hot_keep &= hot_steps <= stop
        else:
            if spec is None:
                return None
            chunks = self._monotonic_value_chunks(name, spec, vstart, vstop, count)
            if chunks is None:
                return None
            hot_arr = np.asarray(hot_values, dtype=DTYPE_TO_NUMPY[spec["dtype"]])
            hot_keep = np.ones(len(hot_arr), dtype=bool)
            if vstart is not None:
                hot_keep &= hot_arr >= vstart
            if vstop is not None:
                hot_keep &= hot_arr <= vstop

        indices = values = None
        if want_indices:
            col = self._columnar_indices_chunks(name, spec, chunks) if spec is not None else np.asarray([], dtype=np.uint32)
            indices = self._concat([col, hot_steps[hot_keep]], np.uint32)
        if want_values:
            np_dtype = object if spec is None or spec.get("dtype") == JSONL_DTYPE else DTYPE_TO_NUMPY[spec["dtype"]]
            col = self._columnar_values_chunks(name, spec, chunks) if spec is not None else np.asarray([], dtype=np_dtype)
            if isinstance(hot_keep, slice):
                hot_sel = hot_values[hot_keep]
            else:
                hot_sel = [value for value, keep in zip(hot_values, hot_keep) if keep]
            values = self._concat([col, np.asarray(hot_sel, dtype=np_dtype)], np_dtype)
        if indices is not None and values is not None:
            count = min(len(indices), len(values))
            indices = indices[:count]
            values = values[:count]
        return indices, values

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
        if (values := self._jsonl_cache.get(name)) is not None:
            return values
        values = self._parse_jsonl_values(name)
        self._jsonl_cache[name] = values
        return values

    def _parse_jsonl_values(self, name):
        if self.kind == "zip":
            return self._parse_jsonl_bytes(self._read_bytes(f"{name}.jsonl"))
        else:
            path = self.root / f"{name}.jsonl"
            try:
                return self._read_live_jsonl(path)
            except FileNotFoundError:
                if self._missing_columnar_is_empty():
                    return []
                raise FileNotFoundError(f"missing values file for {name} in run {self._run_name}")

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
                valid = self._trim_size(self._zip_member_size(f"{name}.indices"), 4)
                count = valid // 4
                if count == 0:
                    return 0, None
                last = int(np.frombuffer(self._read_zip_slice(f"{name}.indices", valid - 4, 4), dtype=np.uint32)[0])
                return count, last
            path = self.root / f"{name}.indices"
            try:
                with path.open("rb") as fh:
                    fh.seek(0, 2)
                    valid = self._trim_size(fh.tell(), 4)
                    count = valid // 4
                    if count == 0:
                        return 0, None
                    fh.seek(valid - 4)
                    last = int(np.frombuffer(fh.read(4), dtype=np.uint32)[0])
            except FileNotFoundError:
                if self._missing_columnar_is_empty():
                    return 0, None
                raise FileNotFoundError(f"missing indices file for {name} in run {self._run_name}")
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
            return self._trim_size(self._zip_member_size(f"{name}.{dtype}"), itemsize) // itemsize
        path = self.root / f"{name}.{dtype}"
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            if self._missing_columnar_is_empty():
                return 0
            raise FileNotFoundError(f"missing values file for {name} in run {self._run_name}")
        valid = self._trim_size(size, itemsize)
        return valid // itemsize

    def _ensure_manifest(self):
        if self._manifest is not None:
            return
        manifest = json.loads(self._read_text("plattli.json"))
        self._run_rows = manifest.pop("run_rows", None)
        self._when_exported = manifest.pop("when_exported", None)
        _validate_metric_names(manifest, self._run_name)
        self._manifest = manifest

    def _ensure_hot(self):
        if self._hot_columns is not None:
            return self._hot_has_file
        self._ensure_manifest()
        self._hot_columns = {}
        self._hot_has_file = False
        if self.exclude_hot:
            return False
        # Writers remove their hot logs before publishing finalized row metadata.
        if self.kind != "dir" or self._run_rows is not None:
            return False
        rows = {}

        def merge_file(filename):
            hot_path = self.root / filename
            try:
                entries = self._read_live_jsonl(hot_path)
            except FileNotFoundError:
                return False  # A completed compaction may unlink its transient file here.
            for row in entries:
                rows[int(row["step"])] = row  # On overlap, the active hot file wins.
            return True

        for filename in (HOT_COMPACTING_FILENAME, HOT_FILENAME):
            self._hot_has_file |= merge_file(filename)
        if not self._hot_has_file:
            # hot.jsonl may have been renamed after the first compacting-log read.
            self._hot_has_file = merge_file(HOT_COMPACTING_FILENAME)
        for step, row in rows.items():
            for name, value in row.items():
                if name == "step":
                    continue
                _validate_metric_name(name, self._run_name)
                col = self._hot_columns.get(name)
                if col is None:
                    col = {"indices": [], "values": []}
                    self._hot_columns[name] = col
                col["indices"].append(step)
                col["values"].append(value)
        return self._hot_has_file

    def _missing_columnar_is_empty(self):
        """A live manifest may advertise a metric before its first compaction."""
        return self.exclude_hot or self._ensure_hot()

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

    def _probe_metric_count(self, name, spec):
        if spec.get("indices") != "indices":
            return self._values_count(name, spec)
        if self.kind == "zip":
            return self._trim_size(self._zip_member_size(f"{name}.indices"), 4) // 4
        path = self.root / f"{name}.indices"
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            if self._missing_columnar_is_empty():
                return 0
            raise FileNotFoundError(f"missing indices file for {name} in run {self._run_name}")
        return self._trim_size(size, 4) // 4

    def approx_max_rows(self, nprobes=12):
        """Estimate the largest metric row count with at most nprobes file probes.

        Closed index specs and finalized run_rows need no probes. Open numeric and
        explicit-index metrics share the budget across index-cadence groups. Hot rows
        and open JSONL metrics are excluded.
        """
        if not isinstance(nprobes, (int, np.integer)) or isinstance(nprobes, bool):
            raise TypeError("nprobes must be an integer")
        if nprobes < 0:
            raise ValueError("nprobes must be >= 0")
        nprobes = int(nprobes)
        self._ensure_manifest()
        if self._run_rows is not None:
            return self._run_rows

        max_rows = 0
        indices_candidates = []
        cadence_groups = {}
        for name, spec in self._manifest.items():
            indices_spec = spec.get("indices")
            if isinstance(indices_spec, (list, dict)):
                try:
                    segments = _segments_from_spec(indices_spec)
                    if not _segments_have_open_tail(segments):
                        count, _ = _segments_count_and_last(segments)
                        max_rows = max(max_rows, count)
                        continue
                    closed_rows, _ = _segments_count_and_last(segments[:-1])
                    _segments_count_and_last(segments, total_count=closed_rows)
                except (ValueError, RuntimeError) as exc:
                    raise type(exc)(f"{exc} (metric {name}, run {self._run_name})") from exc
                if spec.get("dtype") == JSONL_DTYPE:
                    continue
                if spec.get("dtype") not in DTYPE_TO_NUMPY:
                    raise ValueError(f"unsupported dtype for {name} in run {self._run_name}: {spec.get('dtype')}")
                cadence_groups.setdefault(int(segments[-1]["step"]), []).append((
                    (-closed_rows, int(segments[0]["start"])),
                    name,
                    spec,
                ))
            elif indices_spec == "indices":
                indices_candidates.append(((0, 0), name, spec))
            else:
                raise RuntimeError(f"invalid indices spec for {name} in run {self._run_name}: {indices_spec}")

        groups = []
        if indices_candidates:
            groups.append(indices_candidates)
        groups.extend(cadence_groups[cadence] for cadence in sorted(cadence_groups))
        for group in groups:
            group.sort(key=lambda candidate: candidate[0])

        if sum(len(group) for group in groups) <= nprobes:
            candidates = [candidate for group in groups for candidate in group]
        else:
            candidates = []
            positions = [0] * len(groups)
            while len(candidates) < nprobes:
                for idx, group in enumerate(groups):
                    if positions[idx] < len(group):
                        candidates.append(group[positions[idx]])
                        positions[idx] += 1
                        if len(candidates) == nprobes:
                            break

        for _, name, spec in candidates:
            max_rows = max(max_rows, self._probe_metric_count(name, spec))
        return max_rows

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
                valid = self._trim_size(self._zip_member_size(f"{name}.indices"), 4)
                offset = idx * 4
                if offset + 4 > valid:
                    return count, indices_last
                last_step = int(np.frombuffer(self._read_zip_slice(f"{name}.indices", offset, 4), dtype=np.uint32)[0])
                return count, last_step
            path = self.root / f"{name}.indices"
            try:
                with path.open("rb") as fh:
                    fh.seek(0, 2)
                    valid = self._trim_size(fh.tell(), 4)
                    offset = idx * 4
                    if offset + 4 > valid:
                        return count, indices_last
                    fh.seek(offset)
                    last_step = int(np.frombuffer(fh.read(4), dtype=np.uint32)[0])
            except FileNotFoundError:
                if self._missing_columnar_is_empty():
                    return 0, None
                raise FileNotFoundError(f"missing indices file for {name} in run {self._run_name}")
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
                valid = self._trim_size(self._zip_member_size(f"{name}.indices"), 4)
                count = min(count, valid // 4)
                if count <= 0:
                    return np.asarray([], dtype=np.uint32)
                return np.frombuffer(self._read_zip_slice(f"{name}.indices", 0, count * 4), dtype=np.uint32)
            path = self.root / f"{name}.indices"
            try:
                with path.open("rb") as fh:
                    fh.seek(0, 2)
                    valid = self._trim_size(fh.tell(), 4)
                    count = min(count, valid // 4)
                    if count <= 0:
                        return np.asarray([], dtype=np.uint32)
                    fh.seek(0)
                    data = fh.read(count * 4)
            except FileNotFoundError:
                if self._missing_columnar_is_empty():
                    return np.asarray([], dtype=np.uint32)
                raise FileNotFoundError(f"missing indices file for {name} in run {self._run_name}")
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
            valid = self._trim_size(self._zip_member_size(f"{name}.{dtype}"), itemsize)
            count = min(count, valid // itemsize)
            if count <= 0:
                return np.asarray([], dtype=DTYPE_TO_NUMPY[dtype])
            return np.frombuffer(self._read_zip_slice(f"{name}.{dtype}", 0, count * itemsize), dtype=DTYPE_TO_NUMPY[dtype])
        path = self.root / f"{name}.{dtype}"
        try:
            with path.open("rb") as fh:
                data = fh.read(count * itemsize)
        except FileNotFoundError:
            if self._missing_columnar_is_empty():
                return np.asarray([], dtype=DTYPE_TO_NUMPY[dtype])
            raise FileNotFoundError(f"missing values file for {name} in run {self._run_name}")
        data = data[:self._trim_size(len(data), itemsize)]
        if not data:
            return np.asarray([], dtype=DTYPE_TO_NUMPY[dtype])
        return np.frombuffer(data, dtype=DTYPE_TO_NUMPY[dtype])

    def _hot_tail(self, name, spec):
        """Hot rows for this metric beyond the columnar data, as (steps, values list)."""
        if not self._ensure_hot():
            return np.asarray([], dtype=np.uint32), []
        last_step = None
        if spec is not None:
            # Cheap: stat + tail peek, not a full indices read.
            _, last_step = self._columnar_count_and_last_step(name, spec)
        return self._hot_for_metric(name, last_step)

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
        _, hot_values = self._hot_tail(name, spec)
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
        spec = self._metric_spec(name, allow_hot=True)
        indices = self._columnar_indices(name, spec)
        values = self._columnar_values(name, spec)
        # A compaction can land between these reads. Keep their common durable prefix,
        # then merge the hot tail against that same prefix boundary.
        count = min(len(indices), len(values))
        indices = indices[:count]
        values = values[:count]
        last_step = int(indices[-1]) if count else None
        hot_indices, hot_values = self._hot_for_metric(name, last_step)
        count = min(len(hot_indices), len(hot_values))
        hot_indices = hot_indices[:count]
        hot_values = hot_values[:count]
        dtype = (
            object if spec is None or spec.get("dtype") == JSONL_DTYPE
            else DTYPE_TO_NUMPY[spec["dtype"]]
        )
        return (
            self._concat([indices, hot_indices], np.uint32),
            self._concat([values, np.asarray(hot_values, dtype=dtype)], dtype),
        )

    def metric_indices(self, name, start=None, stop=None, istart=None, istop=None, vstart=None, vstop=None):
        if self._selector_kind(start, stop, istart, istop, vstart, vstop) is None:
            return self._metric_indices_full(name)
        selected = self._metric_select(name, start, stop, istart, istop, vstart, vstop, want_values=False)
        if selected is None:
            indices, values = self._metric_full(name)
            return self._apply_selector(indices, values, start, stop, istart, istop, vstart, vstop)[0]
        return selected[0]

    def metric_values(self, name, start=None, stop=None, istart=None, istop=None, vstart=None, vstop=None):
        if self._selector_kind(start, stop, istart, istop, vstart, vstop) is None:
            return self._metric_values_full(name)
        selected = self._metric_select(name, start, stop, istart, istop, vstart, vstop, want_indices=False)
        if selected is None:
            indices, values = self._metric_full(name)
            return self._apply_selector(indices, values, start, stop, istart, istop, vstart, vstop)[1]
        return selected[1]

    def table(self, names, on="step", start=None, stop=None, istart=None, istop=None, vstart=None, vstop=None):
        """Read several metrics step-aligned: (steps, {name: values}), all equal length,
        keeping only steps present in every requested metric.

        Selectors select rows of the `on` column. With the default on="step" that is the
        aligned table itself (vstart/vstop are not meaningful there); with a metric name,
        vstart/vstop select by that metric's values and the other columns follow."""
        if not names:
            raise ValueError(f"table() needs at least one metric name in run {self._run_name}")
        kind = self._selector_kind(start, stop, istart, istop, vstart, vstop)
        if on == "step":
            if kind == "value":
                raise ValueError("vstart/vstop need a metric as the `on` column")
            # Step ranges push down to every column; positions apply to the joined table.
            sel = {"start": start, "stop": stop} if kind == "step" else {}
            columns = {name: self.metric(name, **sel) for name in names}
            column_values = iter(columns.values())
            steps = next(column_values)[0]
            # Equal-cadence columns need neither an intersection nor remapping.
            same_indices = True
            for idx, _ in column_values:
                if idx is steps or (len(idx) == len(steps) and np.array_equal(idx, steps)):
                    continue
                same_indices = False
                steps = np.intersect1d(steps, idx, assume_unique=True)
            if kind == "position":
                chunks = self._position_slice(len(steps), istart, istop)
                lo, hi = chunks[0] if chunks else (0, 0)
                if same_indices:
                    return steps[lo:hi], {name: vals[lo:hi] for name, (_, vals) in columns.items()}
                steps = steps[lo:hi]
            elif same_indices:
                return steps, {name: vals[:len(steps)] for name, (_, vals) in columns.items()}
        else:
            steps, on_values = self.metric(on, start=start, stop=stop, istart=istart, istop=istop, vstart=vstart, vstop=vstop)
            columns = {}
            for name in names:
                if name == on:
                    columns[name] = (steps, on_values)
                elif len(steps):  # Other columns only need the selected step window.
                    columns[name] = self.metric(name, start=int(steps[0]), stop=int(steps[-1]))
                else:
                    columns[name] = self.metric(name, istart=0, istop=0)
            same_indices = True
            for idx, _ in columns.values():
                if idx is steps or (len(idx) == len(steps) and np.array_equal(idx, steps)):
                    continue
                same_indices = False
                steps = np.intersect1d(steps, idx, assume_unique=True)
            if same_indices:
                return steps, {name: vals[:len(steps)] for name, (_, vals) in columns.items()}
        # A live write can extend vals after idx was read, so direct reuse still trims
        # the not-yet-indexed tail.
        return steps, {
            name: vals[:len(steps)] if idx is steps or (len(idx) == len(steps) and np.array_equal(idx, steps))
            else vals[np.searchsorted(idx, steps)]
            for name, (idx, vals) in columns.items()
        }

    def metric(self, name, idx=None, start=None, stop=None, istart=None, istop=None, vstart=None, vstop=None):
        if self._selector_kind(start, stop, istart, istop, vstart, vstop) is None:
            if isinstance(idx, (int, np.integer)) and not isinstance(idx, bool):
                # Fast path: seek-read just the one requested row instead of the whole column.
                indices, values = self._metric_select(name, None, None, int(idx), None if idx == -1 else int(idx) + 1, None, None)
                if len(indices) != 1:
                    raise IndexError(f"index {idx} out of range for metric {name} in run {self._run_name}")
                return indices[0], values[0]
            indices, values = self._metric_full(name)
        else:
            selected = self._metric_select(name, start, stop, istart, istop, vstart, vstop)
            if selected is None:
                indices, values = self._apply_selector(
                    *self._metric_full(name),
                    start, stop, istart, istop, vstart, vstop,
                )
            else:
                indices, values = selected
        if idx is None:
            return indices, values
        return indices[idx], values[idx]
