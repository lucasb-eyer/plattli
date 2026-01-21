import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ._indices import _find_piecewise_params
from .writer import DTYPE_TO_NUMPY, JSONL_DTYPE, _resolve_dtype, _tight_dtype, _zip_path_for_root


class _ColumnBuffer:
    __slots__ = ("i", "v")

    def __init__(self):
        self.i = []
        self.v = []


class PlattliBulkWriter:
    def __init__(self, outdir, step=0, config="config.json"):
        self.run_root = Path(outdir)
        if self.run_root.name == "plattli":
            raise ValueError(f"outdir should be a run directory, not the plattli folder: {outdir}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.root = self.run_root / "plattli"
        self.step = int(step)
        assert self.step >= 0, f"step must be >= 0 for run {self.run_root.name}: {self.step}"

        self._columns = {}
        self._step_metrics = set()
        self._config = config

    def set_config(self, config):
        self._config = config

    def write(self, **metrics):
        assert 0 <= self.step <= 0xFFFFFFFF, f"step out of uint32 range for run {self.run_root.name}: {self.step}"
        for name, value in metrics.items():
            if name == "step":
                raise ValueError(f"metric name 'step' is reserved in run {self.run_root.name}")
            if name in self._step_metrics:
                raise RuntimeError(f"metric already written in step {self.step} for {name} in run {self.run_root.name}")
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
            zip_path = _zip_path_for_root(self.run_root)
            tmp_path = zip_path.with_name(zip_path.name + ".tmp")
            zf = zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED)

            def write_bytes(name, payload):
                zf.writestr(name, payload)

            def close():
                zf.close()
                tmp_path.replace(zip_path)
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
            config = {}
        if isinstance(config, str):
            target = (self.run_root / config).expanduser()
            if target.exists():
                if not target.is_file():
                    raise FileNotFoundError(f"config target is not a file: {target} (run {self.run_root.name})")
                if zip:
                    write_bytes("config.json", target.read_bytes())
                else:
                    if path.exists() or path.is_symlink():
                        path.unlink()
                    path.symlink_to(target.resolve())
                config = None
            else:
                config = {}
        if config is not None:
            if not zip and path.is_symlink():
                path.unlink()
            write_bytes("config.json", json.dumps(config, ensure_ascii=False).encode("utf-8"))

        manifest = {}
        run_rows = 0
        for name, column in self._columns.items():
            indices = np.asarray(column.i, dtype=np.uint32)
            segments = _find_piecewise_params(indices) if optimize else None
            if segments:
                indices_spec = segments
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

            if (dtype := _resolve_dtype(column.v[0], name=name, run_name=self.run_root.name)) == JSONL_DTYPE:
                manifest[name] = {"indices": indices_spec, "dtype": JSONL_DTYPE}
                lines = "\n".join(json.dumps(v.item() if isinstance(v, (np.ndarray, np.generic)) else v,
                                             ensure_ascii=False) for v in column.v)
                if lines:
                    lines += "\n"
                write_bytes(f"{name}.jsonl", lines.encode("utf-8"))
                continue

            arr = np.asarray(column.v, dtype=DTYPE_TO_NUMPY[dtype])
            manifest[name] = {"indices": indices_spec, "dtype": dtype}
            write_bytes(f"{name}.{dtype}", arr.tobytes())

        manifest["when_exported"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        manifest["run_rows"] = run_rows
        write_bytes("plattli.json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
        close()
        self.write = self.end_step = self.set_config = None
