"""Optimize indices of an offline run: python -m plattli.optimize_indices PATH."""

import argparse
import copy
import json
import os
import shutil
import zipfile

from .reader import Reader
from .writer import (
    HOT_COMPACTING_FILENAME,
    HOT_FILENAME,
    _plan_indices,
    _fsync_dir,
    _replace_bytes,
    _replace_text_checked,
)


def optimize_indices(path):
    """Rewrite index representations without changing values or run metadata."""
    with Reader(path) as reader:
        if reader.kind == "zip":
            names = reader._zip.namelist()
            if len(names) != len(set(names)):
                raise ValueError("archive contains duplicate member names")
            has_hot = any(name in names for name in (HOT_FILENAME, HOT_COMPACTING_FILENAME))
        else:
            has_hot = any((reader.root / name).exists() for name in (HOT_FILENAME, HOT_COMPACTING_FILENAME))
        if has_hot:
            raise RuntimeError("finish the run before optimizing indices: hot logs are present")
        reader.manifest()
        manifest = json.loads(reader._read_text("plattli.json"))
        if any(spec.get("values") == "sequence" for name, spec in manifest.items() if name not in ("run_rows", "when_exported")):
            raise RuntimeError("finish the run before optimizing indices: live value sequences are present")
        replacements, retired, changed = _plan_indices(manifest, reader._read_bytes, reader._values_count, reader._run_name)
        if not changed:
            return 0
        payload = json.dumps(manifest, ensure_ascii=False)
        if reader.kind == "dir":
            for name, data in replacements.items():
                _replace_bytes(reader.root / name, data)
            _replace_text_checked(reader.root / "plattli.json", payload, "manifest")
            directories = set()
            for name in retired:
                target = reader.root / name
                target.unlink()
                directories.add(target.parent)
            for directory in directories:
                _fsync_dir(directory)
            return changed

        archive = reader.root
        temporary = archive.with_name(archive.name + ".tmp")
        if temporary.is_symlink():
            temporary.unlink()
        replacements["plattli.json"] = payload.encode("utf-8")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as output:
                for info in reader._zip.infolist():
                    if info.filename in retired:
                        continue
                    if info.filename in replacements:
                        output.writestr(info.filename, replacements.pop(info.filename))
                        continue
                    info = copy.copy(info)
                    info.compress_type = zipfile.ZIP_STORED
                    with reader._zip.open(info.filename) as source, output.open(info, "w") as target:
                        shutil.copyfileobj(source, target)
                for name, data in replacements.items():
                    output.writestr(name, data)
            with temporary.open("rb") as fh:
                os.fsync(fh.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(archive)
        _fsync_dir(archive.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Offline run directory or .plattli archive")
    args = parser.parse_args()
    print(f"Optimized indices for {optimize_indices(args.path)} metrics.")


if __name__ == "__main__":
    main()
