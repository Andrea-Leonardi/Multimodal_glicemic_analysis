"""Extend Timestral context for glucose and physical inputs, validation only."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from src.data import ROOT, sha256
from src.personal_data import configuration
from src.personal_experiment import all_validation, init_stage, load_dataset, metrics
from src.personal_timestral import Predictor, forecast_file


def run(dataset, config, output, hours=(336, 672)):
    output = Path(output)
    plan = [[context, "glucose"] for context in hours] + [[168, "physical"], [336, "physical"]]
    fingerprint = {"config": config, "plan": plan, "dataset": dataset.metadata["fingerprint"],
                   "code": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__).resolve(), ROOT / "src/personal_timestral.py"]}}
    init_stage(output, "timestral-extended", fingerprint)
    earlier = json.loads((output / "timestral-selection.json").read_text())
    valid, fold_labels = all_validation(dataset, config)
    sparse = valid[valid % config["timestral_selection_stride_readings"] == 0]
    predictor = Predictor(dataset, config, "timestral-28m")
    choices = [r for r in earlier["candidates_sparse"] if r["spec"]["model"] == "timestral-28m"]
    additional = []
    for context, mode in plan:
        spec = {"name": f"timestral-28m_{mode}_{context}h", "family": "timestral", "model": "timestral-28m", "mode": mode, "context_hours": context}
        prediction = forecast_file(predictor, sparse, spec, output / "timestral" / ("sparse_" + spec["name"] + ".npz"))
        candidate = {"spec": spec, "metrics": metrics(dataset.targets[sparse], prediction)}
        choices.append(candidate)
        additional.append(candidate)
        print(f"Estensione {mode}/{context}h: {np.round(candidate['metrics']['mae'], 2).tolist()}", flush=True)
    chosen = min(choices, key=lambda c: np.mean(c["metrics"]["mae"]))
    dense_candidates = []
    if chosen["spec"]["name"] in [c["spec"]["name"] for c in additional]:
        spec = chosen["spec"]
        dense = forecast_file(predictor, valid, spec, output / "timestral" / ("dense_" + spec["name"] + ".npz"))
        np.savez_compressed(output / "oof" / (spec["name"] + ".npz"), positions=valid, prediction=dense, folds=fold_labels)
        dense_candidates.append({"spec": spec, "metrics": metrics(dataset.targets[valid], dense)})
    (output / "timestral-extended-selection.json").write_text(json.dumps({"candidates_sparse": additional, "candidates": dense_candidates,
        "selected_sparse": chosen, "note": "28M glucose 14/28 days and physical 7/14 days; validation only, test untouched."}, indent=2))
    del predictor
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "outputs/personal-v2"))
    parser.add_argument("--dataset", default=str(ROOT / "outputs/memory-check/dataset"))
    args = parser.parse_args()
    run(load_dataset(args.dataset), configuration(), args.output)
