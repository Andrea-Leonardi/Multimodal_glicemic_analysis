"""Staged v3 runner; each numerical stage has its own immutable manifest."""

import argparse
from datetime import datetime, timezone
import json
import platform
try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]
import time

from src.v3.common import context, atomic_json

STAGES = ["cache", "tabular", "trajectory", "finetune", "rolling", "ensemble", "intervals", "final", "verify"]


def setup_device(device):
    import torch
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu for CPU-capable stages")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        free, total = torch.cuda.mem_get_info()
        budget = min(4 * 2**30, free - 2**30)
        if budget < 2**30:
            raise RuntimeError("Insufficient free VRAM after reserve")
        torch.cuda.set_per_process_memory_fraction(budget / total)
    return {"python": platform.python_version(), "torch": torch.__version__, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    execution = setup_device(args.device)
    dataset, base, cfg, output = context()
    for name in STAGES if args.stage == "all" else [args.stage]:
        started = time.perf_counter()
        log = output / f"execution-{name}.json"
        info = {"stage": name, "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", **execution}
        atomic_json(log, info)
        module = __import__("src.v3." + {"tabular": "training", "trajectory": "training", "intervals": "uncertainty", "final": "evaluate"}.get(name, name), fromlist=["run"])
        try:
            module.run(dataset, base, cfg, output, args.device, **({"kind": name} if name in {"tabular", "trajectory"} else {}))
        except BaseException as exc:
            info.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            atomic_json(log, info)
            raise
        info.update(status="complete", seconds=time.perf_counter() - started,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource is not None else 0.0)
        if args.device == "cuda":
            import torch
            info["peak_cuda_mib"] = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.empty_cache()
        atomic_json(log, info)


if __name__ == "__main__":
    main()
