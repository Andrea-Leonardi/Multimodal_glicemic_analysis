"""Execute a command using the project memory profile and available WSL RAM.

Example:
.venv/bin/python scripts/run_limited.py -- .venv/bin/python -m module

Training defaults to 16 GiB virtual address space, reduced when necessary to
leave at least 4 GiB of currently available memory for other applications.
The limit is per process: run one training at a time, using shared-memory
threads rather than multiple independent workers each with this allowance.
"""

import argparse
import json
import os
from pathlib import Path

from check_personal_memory import limit_memory

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["build", "training"], default="training")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command after --")
    settings = json.loads((ROOT / "config/resources.json").read_text())[args.profile]
    memory = limit_memory(settings["limit_mib"], settings["reserve_mib"])
    environment = os.environ.copy()
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"]:
        environment[name] = str(settings["threads"])
    print(json.dumps({"profile": args.profile, "threads": settings["threads"], **memory}), flush=True)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
