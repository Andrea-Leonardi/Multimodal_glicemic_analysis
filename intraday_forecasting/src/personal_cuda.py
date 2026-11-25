"""Optional full-float32 CUDA execution of the unchanged Timestral pipeline.

A numerical equivalence gate is required before continuing CPU caches. Hardware
and environment provenance are recorded separately from scientific manifests.
"""

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]
import time

import numpy as np
import torch

from src.data import ROOT, sha256
from src.personal_data import configuration
from src.personal_experiment import load_dataset
from src import personal_timestral as ts

OUTPUT = ROOT / "outputs/personal-v2"
ORIGINAL_INIT = ts.Predictor.__init__
ORIGINAL_PREDICT = ts.Predictor.predict


def provenance():
    cgroup = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    folder = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    return {"python": platform.python_version(), "torch_build": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "versions": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "scipy", "catboost", "timestral"]},
            "source_sha256": {name: sha256(ROOT / name) for name in ["src/personal_cuda.py", "scripts/run_cuda.py", "src/personal_timestral.py", "config/personal-v2.json"]},
            "memory_limits": {name: (folder / name).read_text().strip() for name in ["memory.max", "memory.high", "memory.swap.max"]},
            "precision": "float32; TF32 disabled; unchanged pretrained weights and forecasting code"}


def gpu_init(self, *args, **kwargs):
    ORIGINAL_INIT(self, *args, **kwargs)
    self.adapter.forecaster.model.to("cuda")
    # The upstream rotary-frequency buffer is nonpersistent and must follow the
    # parameters when moving a forecaster initially constructed on the CPU.
    self.adapter.forecaster.model.transformer.rope.rebuild_inv_freq()


def gpu_predict(self, positions, spec, embeddings=False):
    failed = False
    try:
        return ORIGINAL_PREDICT(self, positions, spec, embeddings)
    except torch.cuda.OutOfMemoryError:
        if len(positions) == 1:
            raise
        failed = True
    if failed:
        gc.collect()
        torch.cuda.empty_cache()
        midpoint = len(positions) // 2
        first = gpu_predict(self, positions[:midpoint], spec, embeddings)
        second = gpu_predict(self, positions[midpoint:], spec, embeddings)
        return tuple(None if a is None else np.concatenate([a, b]) for a, b in zip(first, second))


def release(predictor):
    del predictor
    gc.collect()
    torch.cuda.empty_cache()


def check(dataset, config):
    cases = [("timestral-28m", "glucose", 24), ("timestral-28m", "physical", 72), ("timestral-209m", "glucose", 168)]
    details = []
    for model, mode, hours in cases:
        path = OUTPUT / "timestral" / f"sparse_{model}_{mode}_{hours}h.npz"
        with np.load(path) as saved:
            positions = saved["positions"][:4]
            expected, quantiles = saved["prediction"][:4], saved["quantiles"][:4]
        spec = {"model": model, "mode": mode, "context_hours": hours}
        embedding_error = None
        if model == "timestral-28m" and mode == "glucose":
            ts.Predictor.__init__ = ORIGINAL_INIT
            cpu = ts.Predictor(dataset, config, model)
            _, _, cpu_embedding = cpu.predict(positions, spec, embeddings=True)
            del cpu
            gc.collect()
        ts.Predictor.__init__ = gpu_init
        gpu = ts.Predictor(dataset, config, model)
        pred, q, embedding = gpu.predict(positions, spec, embeddings=model == "timestral-28m" and mode == "glucose")
        if embedding is not None:
            embedding_error = float(np.max(np.abs(embedding - cpu_embedding)))
        prediction_error = float(np.max(np.abs(pred - expected)))
        quantile_error = float(np.max(np.abs(q - quantiles)))
        details.append({"model": model, "mode": mode, "context_hours": hours, "origins": positions.tolist(),
                        "cpu_artifact_sha256": sha256(path), "max_prediction_difference_mgdl": prediction_error,
                        "max_quantile_difference_mgdl": quantile_error, "max_embedding_difference": embedding_error})
        del gpu
        gc.collect()
        torch.cuda.empty_cache()
        if prediction_error > .01 or quantile_error > .01 or (embedding_error is not None and embedding_error > .001):
            raise ValueError(f"CPU/CUDA equivalence tolerance exceeded: {details[-1]}")
    result = {"passed": True, "utc": datetime.now(timezone.utc).isoformat(), "execution": provenance(), "cases": details,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource is not None else 0.0,
              "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
    gate = OUTPUT / "cuda-equivalence.json"
    history = OUTPUT / "cuda-equivalence"
    history.mkdir(exist_ok=True)
    if gate.exists():
        (history / (sha256(gate) + ".json")).write_bytes(gate.read_bytes())
    gate.write_text(json.dumps(result, indent=2))
    (history / (sha256(gate) + ".json")).write_bytes(gate.read_bytes())
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "timestral", "extended", "long", "final"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use the original bounded CPU commands")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    free, total = torch.cuda.mem_get_info()
    budget = min(4 * 2**30, free - 2**30)
    if budget < 2**30:
        raise SystemExit("Less than 2 GiB free GPU memory; defer CUDA execution")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    dataset, config = load_dataset(), configuration()
    if args.stage == "check":
        check(dataset, config)
        return
    gate = json.loads((OUTPUT / "cuda-equivalence.json").read_text())
    if not gate["passed"] or gate["execution"]["source_sha256"] != provenance()["source_sha256"]:
        raise ValueError("Repeat CUDA equivalence check after changing accelerator code")
    ts.Predictor.__init__, ts.Predictor.predict = gpu_init, gpu_predict
    started = time.perf_counter()
    log = OUTPUT / f"execution-cuda-{args.stage}.json"
    execution = {"stage": args.stage, "started_utc": datetime.now(timezone.utc).isoformat(), "execution": provenance(),
                 "equivalence_sha256": sha256(OUTPUT / "cuda-equivalence.json"),
                 "equivalence_artifact": "cuda-equivalence/" + sha256(OUTPUT / "cuda-equivalence.json") + ".json", "status": "running"}
    log.write_text(json.dumps(execution, indent=2))
    if args.stage == "timestral":
        ts.run_timestral(dataset, config, OUTPUT)
    elif args.stage == "extended":
        from src.personal_timestral_extended import run
        run(dataset, config, OUTPUT)
    elif args.stage == "long":
        from src.personal_timestral_long import run
        run(dataset, config, OUTPUT)
    else:
        from src.personal_evaluate import finalize
        finalize(dataset, config, OUTPUT)
    execution.update(status="complete", seconds=time.perf_counter() - started,
                     peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource is not None else 0.0,
                     peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    log.write_text(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()
