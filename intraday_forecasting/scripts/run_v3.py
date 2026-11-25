"""Launch v3 with the established WSL memory and thread budgets."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["cache", "tabular", "trajectory", "finetune", "rolling", "ensemble", "intervals", "final", "verify", "all"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    profile = json.loads((ROOT / "config/resources.json").read_text())["training"]
    env = os.environ.copy()
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"]:
        env[key] = str(profile["threads"])
    env["HF_HUB_OFFLINE"] = "1"
    if args.device == "cuda":
        available = next(int(s.split()[1]) // 1024 for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"))
        limit = min(profile["limit_mib"], available - profile["reserve_mib"])
        if limit < 4096:
            raise RuntimeError("Insufficient RAM after reserving resources for the system")
        command = ["systemd-run", "--user", "--scope", "--quiet", "--collect", "-p", f"MemoryMax={limit}M", "-p", f"MemoryHigh={min(14336, limit-1024)}M", "-p", "MemorySwapMax=1G", str(ROOT / ".venv-gpu/bin/python")]
    else:
        command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_limited.py"), "--", str(ROOT / ".venv/bin/python")]
    command += ["-m", "src.v3.run", "--stage", args.stage, "--device", args.device]
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
