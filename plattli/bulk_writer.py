import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .writer import DTYPE_TO_NUMPY, _find_arange_params, _resolve_dtype, _tight_dtype, _zip_path_for_root


class _ColumnBuffer:
    __slots__ = ("i", "v")

    def __init__(self):
        self.i = []
        self.v = []


class PlattliBulkWriter:
    def __init__(self, outdir, step=0, config=None):
        self.run_root = Path(outdir)
        if self.run_root.name == "plattli":
            raise ValueError("outdir should be a run directory, not the plattli folder")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.root = self.run_root / "plattli"
        self.step = int(step)
        assert self.step >= 0, "step must be >= 0"

        self._columns = {}
        self._step_metrics = set()
        self._config = config

    def set_config(self, config):
        self._config = config

    def write(self, **metrics):
        assert 0 <= self.step <= 0xFFFFFFFF, f"step out of uint32 range: {self.step}"
        for name, value in metrics.items():
            if name in self._step_metrics:
                raise RuntimeError(f"metric already written in step {self.step}: {name}")
            bucket = self._columns.get(name)
            if bucket is None:
                bucket = _ColumnBuffer()
                self._columns[name] = bucket
            bucket.i.append(self.step)
            bucket.v.append(value)
            self._step_metrics.add(name)

    def end_step(self):
        self._step_metrics.clear()
        self.step += 1

    def finish(self, optimize=True, zip=True):
        if not self._columns:
            return

        if zip:
            zf = zipfile.ZipFile(_zip_path_for_root(self.run_root), "w", compression=zipfile.ZIP_STORED)

            def write_bytes(name, payload):
                zf.writestr(name, payload)

            def close():
                zf.close()
        else:
            self.root.mkdir(parents=True, exist_ok=True)

            def write_bytes(name, payload):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as fh:
                    fh.write(payload)

            def close():
                return

        path = self.root / "config.json"
        config = self._config
        if config is None:
            run_config = self.run_root / "config.json"
            if run_config.exists():
                if not run_config.is_file():
                    raise FileNotFoundError(f"config target is not a file: {run_config}")
                config = str(run_config.resolve())
            else:
                config = {}
        if isinstance(config, str):
            target = Path(config).expanduser()
            check_target = target if target.is_absolute() else self.run_root / target
            if not check_target.exists():
                raise FileNotFoundError(f"config target missing: {check_target}")
            if not check_target.is_file():
                raise FileNotFoundError(f"config target is not a file: {check_target}")
            if zip:
                write_bytes("config.json", check_target.read_bytes())
            else:
                if path.exists() or path.is_symlink():
                    path.unlink()
                path.symlink_to(check_target.resolve())
            config = None
        if config is not None:
            if not zip and path.is_symlink():
                path.unlink()
            write_bytes("config.json", json.dumps(config, ensure_ascii=False).encode("utf-8"))

        manifest = {}
        run_rows = 0
        for name, column in self._columns.items():
            indices = np.asarray(column.i, dtype=np.uint32)
            if optimize and (params := _find_arange_params(indices)):
                indices_spec = {"start": params[0], "stop": params[1], "step": params[2]}
            else:
                indices_spec = "indices"
                write_bytes(f"{name}.indices", indices.tobytes())

            if indices.size > run_rows:
                run_rows = indices.size

            if optimize:
                tightened = _tight_dtype(column.v)
                if tightened is not None:
                    dtype_tag = f"{tightened.dtype.kind}{tightened.dtype.itemsize * 8}"
                    manifest[name] = {"indices": indices_spec, "dtype": dtype_tag}
                    write_bytes(f"{name}.{dtype_tag}", tightened.tobytes())
                    continue

            if (dtype := _resolve_dtype(column.v[0])) == "json":
                manifest[name] = {"indices": indices_spec, "dtype": "json"}
                values = [_jsonify(v) for v in column.v]
                write_bytes(f"{name}.json", json.dumps(values, ensure_ascii=False).encode("utf-8"))
                continue

            arr = np.asarray(column.v, dtype=DTYPE_TO_NUMPY[dtype])
            manifest[name] = {"indices": indices_spec, "dtype": dtype}
            write_bytes(f"{name}.{dtype}", arr.tobytes())

        manifest["when_exported"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        manifest["run_rows"] = run_rows
        write_bytes("plattli.json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
        close()
        self.write = self.end_step = self.set_config = None


def _jsonify(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.item()
    return value
