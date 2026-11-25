"""Run the optional CUDA worker in a physical-memory-limited systemd scope.

CUDA reserves large virtual address ranges: RLIMIT_AS is unsuitable here.
The existing CPU launcher and its address-space limits remain unchanged.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "timestral", "extended", "long", "final"])
    args = parser.parse_args()
    settings = json.loads((ROOT / "config/resources.json").read_text())["training"]
    available = next(int(line.split()[1]) // 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    limit = min(settings["limit_mib"], available - settings["reserve_mib"])
    if limit < 4096:
        raise SystemExit("Insufficient free RAM for bounded CUDA execution")
    env = os.environ.copy()
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"]:
        env[name] = str(settings["threads"])
    env["HF_HUB_OFFLINE"] = "1"
    command = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
               "-p", f"MemoryMax={limit}M", "-p", f"MemoryHigh={min(14336, limit - 1024)}M",
               "-p", "MemorySwapMax=1G", str(ROOT / ".venv-gpu/bin/python"), "-m", "src.personal_cuda", args.stage]
    print(json.dumps({"physical_ram_limit_mib": limit, "reserve_mib": settings["reserve_mib"], "stage": args.stage}), flush=True)
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
