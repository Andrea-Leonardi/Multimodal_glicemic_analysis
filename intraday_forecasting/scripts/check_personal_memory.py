"""Run small equivalence FIRST, then the full build, in memory-limited children.

python scripts/check_personal_memory.py sample
python scripts/check_personal_memory.py full

RLIMIT_AS bounds virtual address space, not just live arrays/RSS. Thread pools
are capped before numerical imports. No large build runs in the parent process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    files = [Path(__file__).resolve(), ROOT / "src/personal_data.py", ROOT / "src/data.py", ROOT / "tests/personal_dense_reference.py",
             ROOT / "tests/test_personal_memory.py", ROOT / "config/personal-v2.json",
             ROOT / "data/cleaned/cleaned-CGM.csv", ROOT / "data/cleaned/cleaned-Basal.csv",
             ROOT / "data/tandemsource_data/merged/merged-Bolo.csv"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def available_mib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable not readable; refusing an unbounded build")


def limit_memory(requested, reserve):
    available = available_mib()
    limit = min(requested, available - reserve)
    if limit < 512:
        raise RuntimeError("Insufficient available memory after reserving resources for other applications")
    old_hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    limit_bytes = limit * 1024 ** 2
    if old_hard != resource.RLIM_INFINITY:
        limit_bytes = min(limit_bytes, old_hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    return {"limit_kind": "RLIMIT_AS (virtual address space)", "limit_mib": limit_bytes / 1024 ** 2,
            "mem_available_before_mib": available, "reserve_mib": reserve}


def child(args):
    # These overrides are local to the verification child, never shell/global.
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"]:
        os.environ[name] = "1"
    memory = limit_memory(args.limit_mib, args.reserve_mib)
    print(json.dumps({"stage": args.child, **memory}), flush=True)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tests"))
    started = time.perf_counter()
    if args.child == "equivalence":
        from test_personal_memory import compare_sample
        metrics = {"sample_rows": args.sample_rows,
                   "maximum_absolute_differences": compare_sample(args.sample_rows, args.block_size),
                   "equivalent": True}
    else:
        import numpy as np
        from src.personal_data import build
        dataset = build(block_size=args.block_size, max_rows=args.sample_rows if args.mode == "sample" else None, progress=True)
        metrics = {"rows": len(dataset.times), "features": len(dataset.feature_names),
                   "feature_shape": list(dataset.features.shape), "sequence_shape": list(dataset.sequence.shape),
                   "eligible_origins": int(dataset.eligible.sum()),
                   "median_cgm_interval_seconds": float(np.median(np.diff(dataset.times))),
                   "array_mib": sum(getattr(dataset, field).nbytes for field in ["times", "glucose", "features", "targets", "target_indices", "sequence", "eligible"]) / 1024 ** 2}
        if not 250 < metrics["median_cgm_interval_seconds"] < 350:
            raise AssertionError("Invalid sensor time units")
        # Persist separate .npy arrays using bounded writes. They can later be
        # reopened using mmap_mode='r', without rebuilding or duplicating RAM.
        if args.mode == "full":
            target = Path(args.output) / "dataset"
            if target.exists():
                metadata = target / "metadata.json"
                if not metadata.exists() or json.loads(metadata.read_text()).get("fingerprint") != fingerprint():
                    raise RuntimeError("Saved dataset belongs to different code/data. Choose a new --output directory.")
            else:
                # Atomic directory publication: an interrupted write leaves no
                # apparently complete dataset at the final destination.
                with tempfile.TemporaryDirectory(prefix=".dataset-", dir=args.output) as name:
                    staging = Path(name)
                    for field in ["times", "glucose", "features", "targets", "target_indices", "sequence", "eligible"]:
                        with (staging / (field + ".npy")).open("wb") as stream:
                            np.save(stream, getattr(dataset, field), allow_pickle=False)
                    (staging / "feature_names.json").write_text(json.dumps(dataset.feature_names, indent=2))
                    (staging / "metadata.json").write_text(json.dumps(dataset.metadata | {"fingerprint": fingerprint()}, indent=2))
                    dataset.event_table.to_csv(staging / "events.csv", index=False)
                    staging.replace(target)
            metrics["saved_dataset"] = str(target)
    report = {"status": "passed", "stage": args.child, "mode": args.mode, "block_size": args.block_size,
              "seconds": time.perf_counter() - started, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              **memory, **metrics, "fingerprint": fingerprint()}
    report_path = Path(args.output) / f"{args.mode}-{args.child}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "fingerprint"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sample", "full"])
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--sample-rows", type=int, default=4096)
    parser.add_argument("--limit-mib", type=int, default=2048)
    parser.add_argument("--reserve-mib", type=int, default=4096)
    parser.add_argument("--output", default=str(ROOT / "outputs/memory-check"))
    parser.add_argument("--child", choices=["equivalence", "build"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.block_size < 1 or not 1 <= args.sample_rows <= 4096 or args.limit_mib < 512 or args.reserve_mib < 1024:
        parser.error("Require positive block size, 1..4096 sample rows, limit>=512 MiB and reserve>=1024 MiB")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.child:
        try:
            child(args)
        except MemoryError:
            print("Memory limit reached: child stopped without exhausting system RAM.", file=sys.stderr, flush=True)
            raise SystemExit(2)
        return
    if args.mode == "full":
        sample = Path(args.output) / "sample-equivalence.json"
        if not sample.exists() or json.loads(sample.read_text()).get("fingerprint") != fingerprint():
            parser.error("Run sample equivalence with the current code/data before the full build")
        previous = json.loads(sample.read_text())
        if not previous.get("equivalent") or previous["status"] != "passed":
            parser.error("Sample equivalence has not passed")
    stages = ["equivalence", "build"] if args.mode == "sample" else ["build"]
    for stage in stages:
        command = [sys.executable, str(Path(__file__).resolve()), args.mode, "--child", stage,
                   "--block-size", str(args.block_size), "--sample-rows", str(args.sample_rows),
                   "--limit-mib", str(args.limit_mib), "--reserve-mib", str(args.reserve_mib), "--output", args.output]
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
