import json
import zipfile
from shutil import rmtree
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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


class PlattliWriter:
    def __init__(self, outdir, step=0, write_threads=16, config=None):
        self.root = Path(outdir)
        self.root.mkdir(parents=True, exist_ok=True)

        self.step = int(step)
        assert self.step >= 0, "step must be >= 0"

        self._manifest = {}
        self._executor = ThreadPoolExecutor(max_workers=write_threads) if write_threads else None
        self._futures = []
        self._step_metrics = set()  # Just to protect user from double-logging.

        if (self.root / "plattli.json").exists():
            self._manifest = json.loads((self.root / "plattli.json").read_text(encoding="utf-8"))
            self._manifest.pop("when_exported", None)
            self._manifest.pop("run_rows", None)
            self._truncate_to_step(self.step)

        self.set_config(config)

    def write(self, **metrics):
        self._drain_errors()
        new_metric = False
        for name, value in metrics.items():
            if name in self._step_metrics:
                raise RuntimeError(f"metric already written in step {self.step}: {name}")
            if name not in self._manifest:
                dtype = _resolve_dtype(value)
                (self.root / name).parent.mkdir(parents=True, exist_ok=True)
                self._manifest[name] = {"indices": "indices", "dtype": dtype}
                new_metric = True
            else:
                dtype = self._manifest[name]["dtype"]
            if self._executor:
                self._futures.append(self._executor.submit(self._write_entry, name, dtype, value, self.step))
            else:
                self._write_entry(name, dtype, value, self.step)
            self._step_metrics.add(name)
        if new_metric:
            self._write_manifest()

    def end_step(self):
        wait(self._futures)
        self._drain_errors()
        self._step_metrics.clear()
        self.step += 1

    def finish(self, optimize=True, zip=True):
        if not self._manifest:
            return

        wait(self._futures)
        self._drain_errors()

        if optimize:
            self._tighten_dtypes()
            self._optimize_indices()

        self._write_manifest(run_rows=max(self._indices_length(name, spec["indices"])
                                          for name, spec in self._manifest.items()))

        if zip:
            self._zip_output()
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.write = self.end_step = self.set_config = None

    def set_config(self, config):
        path = self.root / "config.json"
        if config is None:
            if path.exists():
                return
            config = {}
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    def _write_entry(self, name, dtype, value, step):
        if step < 0 or step > 0xFFFFFFFF:
            raise ValueError(f"step out of uint32 range: {step}")
        if dtype == "json":
            _append_json_value(self.root / f"{name}.json", value)
        else:
            _append_numeric(self.root / f"{name}.{dtype}", value, dtype)
        _append_indices((self.root / f"{name}.indices"), step)

    def _drain_errors(self):
        remaining = []
        for f in self._futures:
            if f.done():
                if (err := f.exception()) is not None:
                    err.add_note("Background write failed; exception surfaced while draining pending writes.")
                    raise err
            else:
                remaining.append(f)
        self._futures = remaining

    def _write_manifest(self, run_rows=None):
        manifest = {
            **self._manifest,
            "when_exported": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        if run_rows is not None:
            manifest["run_rows"] = run_rows
        (self.root / "plattli.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    def _truncate_to_step(self, step):
        for name, spec in self._manifest.items():
            # First, truncate the indices. Always turns them to an .indices file.
            indices_spec = spec["indices"]
            if indices_spec == "indices":
                idx_path = self.root / f"{name}.indices"
                if not idx_path.exists():
                    raise FileNotFoundError(f"missing indices file for {name}")  # pragma: no cover
                indices = np.fromfile(idx_path, dtype=np.uint32)
                keep = int(np.searchsorted(indices, step, side="left"))
                with idx_path.open("r+b") as fh:
                    fh.truncate(keep * 4)
            elif isinstance(indices_spec, dict):
                indices = np.arange(indices_spec["start"], indices_spec["stop"], indices_spec["step"], dtype=np.uint32)
                keep = int(np.searchsorted(indices, step, side="left"))
                with (self.root / f"{name}.indices").open("wb") as fh:
                    indices[:keep].tofile(fh)
                spec["indices"] = "indices"
            else:
                raise RuntimeError(f"invalid indices spec for {name}: {indices_spec}")  # pragma: no cover

            # And then truncate the values accordingly. For json, have one line per step for readability.
            if (dtype := spec["dtype"]) == "json":
                path = self.root / f"{name}.json"
                data = json.loads(path.read_text(encoding="utf-8"))[:keep]
                with path.open("w", encoding="utf-8") as fh:
                    if data:
                        fh.write("[\n")
                        fh.write("\n,\n".join(json.dumps(v, ensure_ascii=False) for v in data))
                        fh.write("\n]")
                    else:
                        fh.write("[]")
            else:
                with (self.root / f"{name}.{dtype}").open("r+b") as fh:
                    fh.truncate(keep * np.dtype(DTYPE_TO_NUMPY[dtype]).itemsize)
        self._write_manifest()

    def _optimize_indices(self):
        for name, spec in self._manifest.items():
            if spec["indices"] != "indices":
                continue  # pragma: no cover
            idx_path = self.root / f"{name}.indices"
            indices = np.fromfile(idx_path, dtype=np.uint32)
            if params := _find_arange_params(indices):
                spec["indices"] = {"start": params[0], "stop": params[1], "step": params[2]}
                idx_path.unlink()

    def _indices_length(self, name, indices_spec):
        if isinstance(indices_spec, dict):
            start = int(indices_spec["start"])
            stop = int(indices_spec["stop"])
            step = int(indices_spec["step"])
            if step <= 0 or stop <= start:
                return 0  # pragma: no cover
            return int((stop - start + step - 1) // step)
        if indices_spec == "indices":
            idx_path = self.root / f"{name}.indices"
            return idx_path.stat().st_size // 4  # It's always uint32
        return 0  # pragma: no cover

    def _tighten_dtypes(self):
        for name, spec in self._manifest.items():
            dtype = spec["dtype"]
            if dtype == "json":
                continue
            path = self.root / f"{name}.{dtype}"
            arr = np.fromfile(path, dtype=DTYPE_TO_NUMPY[dtype])
            if arr.size == 0:
                continue  # pragma: no cover
            tightened = _tight_dtype(arr)
            if tightened is None:
                continue  # pragma: no cover - unreachable
            new_dtype = f"{tightened.dtype.kind}{tightened.dtype.itemsize * 8}"
            if new_dtype == dtype:
                continue
            tightened.tofile(self.root / f"{name}.{new_dtype}")
            path.unlink()
            spec["dtype"] = new_dtype

    def _zip_output(self):
        with zipfile.ZipFile(f"{self.root}.zip", "w", compression=zipfile.ZIP_STORED) as zf:
            for path in sorted(self.root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(self.root))
        rmtree(self.root)


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


def _append_numeric(path, value, dtype):
    arr = np.asarray(value)
    if arr.shape != ():
        raise ValueError("only scalar values are supported")
    with path.open("ab") as fh:
        np.asarray([arr], dtype=DTYPE_TO_NUMPY[dtype]).tofile(fh)


def _append_indices(path, step):
    with path.open("ab") as fh:
        np.asarray([step], dtype=np.uint32).tofile(fh)


def _append_json_value(path, value):
    if isinstance(value, (np.ndarray, np.generic)):
        value = value.item()
    payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
    if not path.exists() or path.stat().st_size == 0:
        with path.open("wb") as fh:
            fh.write(b"[\n")
            fh.write(payload)
            fh.write(b"\n]")
        return
    with path.open("r+b") as fh:
        fh.seek(0, 2)
        pos = fh.tell()
        while pos > 0:
            pos -= 1
            fh.seek(pos)
            ch = fh.read(1)
            if ch not in b" \t\r\n":
                break
        if ch != b"]":
            raise RuntimeError(f"invalid json array in {path}")  # pragma: no cover
        pos2 = pos - 1
        while pos2 >= 0:
            fh.seek(pos2)
            ch2 = fh.read(1)
            if ch2 not in b" \t\r\n":
                break
            pos2 -= 1
        if pos2 >= 0 and ch2 == b"[":
            fh.seek(pos2 + 1)
            fh.truncate()
            fh.write(b"\n")
            fh.write(payload)
            fh.write(b"\n]")
            return
        fh.seek(pos)
        fh.truncate()
        fh.write(b",\n")
        fh.write(payload)
        fh.write(b"\n]")


def _resolve_dtype(value):
    if hasattr(value, "__array__"):
        value = np.asarray(value)
    if isinstance(value, (np.ndarray, np.generic)):
        if value.shape != ():
            raise ValueError("only scalar array-like values are supported")
        if (kind := value.dtype.kind) in "fiu":
            dtype = f"{kind}{value.dtype.itemsize * 8}"
            return dtype if dtype in DTYPE_TO_NUMPY else "json"
        return "json"
    if isinstance(value, bool):
        return "json"
    if isinstance(value, float):
        return "f32"
    if isinstance(value, (int, np.integer)):
        return "i64"
    return "json"
