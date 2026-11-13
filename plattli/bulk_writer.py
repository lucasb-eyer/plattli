import json
import shutil
import tempfile
import zipfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ._indices import _piecewise_segments
from ._ranges import _ranges_encode
from ._affine import _find_step_sequence
from ._values import _find_value_sequence
from ._numeric import _encode_numeric, _numeric_dtype
from .writer import (
    DTYPE_TO_NUMPY,
    JSONL_DTYPE,
    HOT_FILENAME,
    HOT_COMPACTING_FILENAME,
    _archive_member_path,
    _choose_indices,
    _coerce_step,
    _coerce_stored_value,
    _merge_metrics,
    _prepare_config,
    _reconcile_dtype,
    _resolve_dtype,
    _set_monotonic,
    _tight_dtype,
    _value_dtype,
    _validate_metric_names,
    _validate_step,
    _zip_path_for_root,
)


def _old_artifacts(manifest, names):
    metrics = {name: spec for name, spec in manifest.items() if name not in ("when_exported", "run_rows")}
    _validate_metric_names(metrics, "previous output")
    suffixes = {*DTYPE_TO_NUMPY, JSONL_DTYPE, "values"}
    stems = set(metrics)
    for name in names:
        stem, _, suffix = name.rpartition(".")
        if suffix in suffixes and any(f"{stem}.{suffix}" in names for suffix in ("indices", "ranges")):
            stems.add(stem)
    owned = {"plattli.json", "config.json", HOT_FILENAME, HOT_COMPACTING_FILENAME}
    owned.update(name + ".tmp" for name in tuple(owned))
    for stem in stems:
        for suffix in suffixes | {"indices", "ranges"}:
            owned.add(f"{stem}.{suffix}")
            owned.add(f"{stem}.{suffix}.tmp")
    return owned


def _open_ancillary(source):
    path, member = source
    return path.open("rb") if member is None else path.open(member)


def _same_ancillary(first, second):
    sizes = [path.stat().st_size if member is None else member.file_size
             for path, member in (first, second)]
    if sizes[0] != sizes[1]:
        return False
    with _open_ancillary(first) as left, _open_ancillary(second) as right:
        while chunk := left.read(1024 * 1024):
            if chunk != right.read(len(chunk)):
                return False
        return not right.read(1)


def _scan_previous_outputs(root, zip_path, stack, workspace):
    """Collect extras; the caller keeps archive handles open through staging."""
    ancillary = {}
    previous = []
    owned = set()
    if root.exists():
        files = {path.relative_to(root).as_posix(): (path, None)
                 for path in root.rglob("*") if path.is_file() or path.is_symlink()}
        manifest_path = root / "plattli.json"
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        previous.append(files)
        owned.update(_old_artifacts(old_manifest, files))
    if zip_path.exists():
        previous_zip = stack.enter_context(zipfile.ZipFile(zip_path))
        files = {info.filename: (previous_zip, info) for info in previous_zip.infolist() if not info.is_dir()}
        old_manifest = json.loads(previous_zip.read("plattli.json"))
        previous.append(files)
        owned.update(_old_artifacts(old_manifest, files))
    for files in previous:
        for name, entry in files.items():
            _archive_member_path(workspace / "checked", name, root.parent.name)
            if name in owned:
                continue
            if name in ancillary and not _same_ancillary(ancillary[name], entry):
                raise ValueError(f"conflicting ancillary file {name!r} in existing outputs for run {root.parent.name}")
            ancillary[name] = entry
    return ancillary


def _publish_output(root, zip_path, staged, backup, zip):
    # Retire the directory before publication so a failed rename cannot leave it
    # beside a newly published archive. Any old archive stays live until replaced.
    if zip:
        try:
            if root.exists():
                root.replace(backup)
            staged.replace(zip_path)
        except BaseException:
            if not staged.exists():
                raise  # Publication completed; keep the new archive authoritative.
            try:
                if backup.exists():
                    backup.replace(root)
            except BaseException as exc:
                raise RuntimeError(f"could not restore previous output; retained files at {backup.parent}") from exc
            raise
    else:
        if root.exists():
            root.replace(backup)
        try:
            staged.replace(root)
            if zip_path.exists():
                zip_path.unlink()
        except BaseException:
            if not staged.exists() and not zip_path.exists():
                raise  # Publication completed; retain the new authoritative tree.
            try:
                if root.exists():
                    root.replace(staged)
                if backup.exists():
                    backup.replace(root)
            except BaseException as exc:
                raise RuntimeError(f"could not restore previous output; retained files at {backup.parent}") from exc
            raise
    if backup.exists():
        shutil.rmtree(backup)


class _ColumnBuffer:
    __slots__ = ("i", "v", "dtype", "_bounds", "_bounds_count")

    def __init__(self):
        self.i = []
        self.v = []
        self.dtype = None
        self._bounds = None
        self._bounds_count = 0

    def integer_bounds(self):
        # jsonl2plattli can rewind this buffer; discarded extrema must not survive.
        if self._bounds is None or self._bounds_count != len(self.v):
            self._bounds = (min(self.v), max(self.v)) if self.v else (0, 0)
            self._bounds_count = len(self.v)
        return self._bounds

    def append(self, step, value, dtype):
        if self._bounds is not None:
            if self._bounds_count == len(self.v):
                lo, hi = self._bounds
                self._bounds = min(lo, value), max(hi, value)
                self._bounds_count += 1
            else:
                # A rewind may be followed by a matching dtype, bypassing bounds().
                self._bounds = None
        self.dtype = dtype
        self.i.append(step)
        self.v.append(value)


class BulkWriter:
    def __init__(self, outdir, step=0, config="config.json", overwrite=False):
        self.run_root = Path(outdir)
        if self.run_root.name == "plattli":
            raise ValueError(f"outdir should be a run directory, not the plattli folder: {outdir}")
        self.step = _validate_step(step, self.run_root.name)
        config = _prepare_config(self.run_root, config)
        if not overwrite:
            zip_path = _zip_path_for_root(self.run_root)
            if zip_path.exists():
                raise RuntimeError(f"found existing {zip_path}; pass overwrite=True to replace it")
            root = self.run_root / "plattli"
            if root.exists():
                raise RuntimeError(f"found existing {root}; pass overwrite=True to replace it")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.root = self.run_root / "plattli"
        self._columns = {}
        self._step_metrics = set()
        self._config = config
        self._finished = False

    def set_config(self, config):
        self._check_finished()
        self._config = _prepare_config(self.run_root, config)

    def _check_finished(self):
        if self._finished:
            raise RuntimeError(f"writer for run {self.run_root.name} is already finished")

    def write(self, *args, **kwargs):
        self._check_finished()
        self.step = _validate_step(self.step, self.run_root.name)
        metrics = _merge_metrics(args, kwargs, self.run_root.name)
        # Existing column names were validated when their first value was accepted.
        _validate_metric_names((name for name in metrics if name not in self._columns), self.run_root.name)
        prepared = []
        for name, value in metrics.items():
            if name in self._step_metrics:
                raise RuntimeError(f"metric already written in step {self.step} for {name} in run {self.run_root.name}")
            bucket = self._columns.get(name)
            if bucket is None:
                dtype = _resolve_dtype(value, name=name, run_name=self.run_root.name)
                value = _coerce_stored_value(value, dtype, name, self.run_root.name)
            else:
                dtype = _reconcile_dtype(bucket.dtype, value, name, self.run_root.name,
                                         integer_bounds=bucket.integer_bounds)
                value = _coerce_stored_value(value, dtype, name, self.run_root.name)
            prepared.append((name, bucket, dtype, value))
        for name, bucket, dtype, value in prepared:
            if bucket is None:
                bucket = _ColumnBuffer()
                self._columns[name] = bucket
            bucket.append(self.step, value, dtype)
            self._step_metrics.add(name)

    def end_step(self):
        self._check_finished()
        self.step = _coerce_step(self.step, self.run_root.name)
        self._step_metrics.clear()
        self.step += 1

    def finish(self, optimize=True, zip=True, config_overrides=None):
        self._check_finished()
        if config_overrides is not None and not isinstance(config_overrides, dict):
            raise TypeError(f"config_overrides must be a dict in run {self.run_root.name}")

        config = self._config
        config_target = None
        if config is None:
            config = {}
        if isinstance(config, str):
            config_target = (self.run_root / config).expanduser()
            if config_target.exists():
                if not config_target.is_file():
                    raise FileNotFoundError(
                        f"config target is not a file: {config_target} (run {self.run_root.name})")
                config = json.loads(config_target.read_text(encoding="utf-8")) if config_overrides is not None else None
            else:
                config = {}
                config_target = None
        if config_overrides is not None:
            if not isinstance(config, dict):
                raise TypeError(f"config must be a JSON object to apply overrides in run {self.run_root.name}")
            config = {**config, **config_overrides}
        config_payload = json.dumps(config, ensure_ascii=False).encode("utf-8") if config is not None else None
        config_link = None
        if config is None:
            config_target = config_target.resolve()
            # A source inside the retired output would otherwise become a dangling
            # link, or a self-link when it is the previous config.json itself.
            if zip or config_target.is_relative_to(self.root.resolve()):
                config_payload = config_target.read_bytes()
            else:
                config_link = config_target

        if self.root.exists() and not self.root.is_dir():
            raise RuntimeError(f"expected {self.root} to be a directory for run {self.run_root.name}")
        zip_path = _zip_path_for_root(self.run_root)
        workspace = Path(tempfile.mkdtemp(prefix=".plattli-", dir=self.run_root))
        staged = workspace / ("metrics.plattli" if zip else "plattli")
        backup = workspace / "previous-plattli"
        try:
            with ExitStack() as stack:
                ancillary = _scan_previous_outputs(self.root, zip_path, stack, workspace)

                zf = stack.enter_context(zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_STORED)) if zip else None
                if not zip:
                    staged.mkdir()
                written = set()
                directories = set()

                def register(name):
                    parts = name.split("/")
                    parents = {"/".join(parts[:i]) for i in range(1, len(parts))}
                    if name in directories or parents & written:
                        raise ValueError(f"conflicting output file and directory at {name!r} in run {self.run_root.name}")
                    directories.update(parents)
                    written.add(name)

                def write_bytes(name, payload):
                    register(name)
                    if zip:
                        zf.writestr(name, payload)
                    else:
                        path = staged / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(payload)

                if config_link is not None:
                    register("config.json")
                    (staged / "config.json").symlink_to(config_link)
                else:
                    write_bytes("config.json", config_payload)

                manifest = {}
                run_rows = 0
                for name, column in self._columns.items():
                    indices = np.asarray(column.i, dtype=np.uint32)
                    segments = _piecewise_segments(indices) if optimize and indices.size >= 2 else None
                    indices_spec = _choose_indices(segments, indices.size) if segments else "indices"
                    if indices_spec == "ranges":
                        write_bytes(f"{name}.ranges", _ranges_encode(segments, total_count=indices.size))
                    elif indices_spec == "indices":
                        write_bytes(f"{name}.indices", indices.tobytes())
                    run_rows = max(run_rows, indices.size)

                    dtype = column.dtype
                    if dtype == JSONL_DTYPE:
                        manifest[name] = {"indices": indices_spec, "dtype": JSONL_DTYPE}
                        lines = "\n".join(json.dumps(value, ensure_ascii=False) for value in column.v)
                        write_bytes(f"{name}.jsonl", (lines + "\n" if lines else "").encode("utf-8"))
                        continue
                    arr = np.asarray(column.v, dtype=_value_dtype(dtype))
                    if optimize and arr.size:
                        arr = _tight_dtype(arr)
                        dtype = _numeric_dtype(arr)
                    manifest[name] = {"indices": indices_spec, "dtype": dtype}
                    _set_monotonic(manifest[name], arr)
                    sequence = _find_value_sequence(arr, dtype) or _find_step_sequence(indices, arr, dtype)
                    if sequence is not None:
                        manifest[name]["values"] = sequence
                    else:
                        write_bytes(f"{name}.{dtype}", _encode_numeric(arr, dtype))

                manifest["when_exported"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                manifest["run_rows"] = run_rows
                write_bytes("plattli.json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
                for name, entry in ancillary.items():
                    if name in written:
                        raise ValueError(f"ancillary file {name!r} conflicts with requested output for run {self.run_root.name}")
                    register(name)
                    with _open_ancillary(entry) as source:
                        if zip:
                            with zf.open(name, "w") as target:
                                shutil.copyfileobj(source, target)
                        else:
                            path = _archive_member_path(staged, name, self.run_root.name)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            with path.open("wb") as target:
                                shutil.copyfileobj(source, target)

            _publish_output(self.root, zip_path, staged, backup, zip)
            self._finished = True
        finally:
            # If restoration or retirement failed, retain the backup for recovery.
            if not backup.exists():
                shutil.rmtree(workspace)
