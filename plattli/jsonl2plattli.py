#!/usr/bin/env python3

import json
import re
import zipfile
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pathlib import Path

from .bulk_writer import PlattliBulkWriter
from .writer import _zip_path_for_root

DEFAULT_SKIP_PATTERN = r"[pg]norm.*"


def convert_run(run_dir, dest, use_named_zip, skip_cols=None, allow_rewinds=False):
    metrics_path = run_dir / "metrics.jsonl"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.jsonl in {run_dir}")
    patterns = [re.compile(pattern) for pattern in skip_cols or ()]

    def delcols(row):
        return {k: v for k, v in row.items() if not any(r.fullmatch(k) for r in patterns)}

    ncols = 0
    nrows = 0
    saw_step = False
    saw_no_step = False
    prev_step = None
    rewind_hint = " If this is expected (such as due to preemptions), use the --allow-rewinds flag."
    if dest == run_dir:
        config = "config.json"
    else:
        config_path = run_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            config = {}
    w = PlattliBulkWriter(dest, config=config)
    with metrics_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            raw = json.loads(line)
            row_step = raw.pop("step", None)
            row = delcols(raw)
            if row_step is not None:
                if saw_no_step:
                    raise ValueError(f"mixed step/no-step rows at {metrics_path}:{lineno}")
                if isinstance(row_step, bool) or not isinstance(row_step, int):
                    raise TypeError(f"step must be an int at {metrics_path}:{lineno}")
                if row_step < 0 or row_step > 0xFFFFFFFF:
                    raise ValueError(f"step out of uint32 range at {metrics_path}:{lineno}: {row_step}")
                if prev_step is not None and row_step <= prev_step:
                    if allow_rewinds and row_step < prev_step:
                        for column in w._columns.values():
                            while column.i and column.i[-1] >= row_step:
                                column.i.pop()
                                column.v.pop()
                        w._step_metrics.clear()
                    else:
                        raise ValueError(
                            f"step must strictly increase at {metrics_path}:{lineno}: {row_step} after {prev_step}."
                            f"{rewind_hint}"
                        )
                if not allow_rewinds and row_step < w.step:
                    raise ValueError(
                        f"step went backwards at {metrics_path}:{lineno}: {row_step} < {w.step}."
                        f"{rewind_hint}"
                    )
                w.step = row_step
                saw_step = True
                prev_step = row_step
            else:
                if saw_step:
                    raise ValueError(f"mixed step/no-step rows at {metrics_path}:{lineno}")
                saw_no_step = True
            ncols = max(ncols, len(row))
            nrows += 1
            if row:
                w.write(**row)
            w.end_step()
    w.finish(optimize=True, zip=True)
    outpath = _zip_path_for_root(dest)
    if use_named_zip:
        named = dest.with_name(f"{dest.name}.plattli")
        if named != outpath:
            outpath.replace(named)
            outpath = named
        if dest.is_dir():
            try:
                dest.rmdir()
            except OSError:
                pass
    with zipfile.ZipFile(outpath) as zf:
        manifest = json.loads(zf.read("plattli.json"))
    return outpath, nrows, ncols, manifest


def _convert_run(job):
    return convert_run(*job)


def discover_runs(indirs, output_root, deep=False):
    runs = []
    for root in indirs:
        if not root.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {root}")

        for metrics_file in (root.rglob("metrics.jsonl") if deep else root.glob("*/metrics.jsonl")):
            run_dir = metrics_file.parent
            if output_root is None:
                dest = run_dir
            else:
                dest = output_root / root.name / run_dir.relative_to(root)
            runs.append((run_dir, dest))

    return sorted(runs, key=lambda pair: str(pair[0]))


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve() if args.outdir else None
    indirs = [Path(p).resolve() for p in (args.indirs or [Path.cwd()])]
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(indirs, outdir, deep=args.deep)
    if not runs:
        print("No runs found. Nothing to export.")
        return

    print(f"Found {len(runs)} runs under {[root.name for root in indirs]}")

    skip_cols = None
    if args.skipcols is not None:
        skip_cols = [pattern if pattern != "DEFAULT" else DEFAULT_SKIP_PATTERN for pattern in args.skipcols]
    jobs = [(run_dir, dest, outdir is not None, skip_cols, args.allow_rewinds) for run_dir, dest in runs]

    converted = 0
    with Pool(processes=args.workers) as pool:
        for outpath, nrows, ncols, _ in pool.imap_unordered(_convert_run, jobs):
            print(f"converted {(outpath.relative_to(outdir) if outdir else outpath)} ({nrows} rows, {ncols} cols)")
            converted += 1
    if outdir:
        print(f"done. exported {converted} runs into {outdir}")
    else:
        print(f"done. exported {converted} runs in-place")


def parse_args():
    p = ArgumentParser(description="Find and convert JSONL experiment logs into Plättli format.")
    p.add_argument("indirs", nargs="*",
                   help="Experiment folders to search for metrics.jsonl/config.json (defaults to cwd)")
    p.add_argument("-o", "--outdir",
                   help="Directory that will receive converted runs (defaults to writing <run_dir>/metrics.plattli)")
    p.add_argument("-w", "--workers", type=int, default=max(cpu_count() - 2, 1),
                   help="Worker processes (default: NCPU - 2)")
    p.add_argument("--skipcols", action="append", metavar="REGEXP",
                   help=f"Skip metrics whose names fully match REGEXP; use \"DEFAULT\" for the built-in pattern ({DEFAULT_SKIP_PATTERN})."
                        " Can be repeated multiple times for skipping multiple patterns.")
    p.add_argument("--deep", action="store_true",
                   help="Recurse into subdirectories when discovering runs (default: immediate children only).")
    p.add_argument("--allow-rewinds", action="store_true",
                   help="Allow step numbers to decrease and truncate existing data to the new step.")
    return p.parse_args()


if __name__ == "__main__":
    main()
