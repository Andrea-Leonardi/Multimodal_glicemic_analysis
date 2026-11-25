"""Validation-only saturation check within Timestral's 32,768-step limit."""

import gc
import json
from pathlib import Path

import numpy as np

from src.data import ROOT, sha256
from src.personal_experiment import all_validation, init_stage, metrics
from src.personal_timestral import Predictor, forecast_file


def run(dataset, config, output):
    output = Path(output)
    plan = {"timestral-28m": [[1344, "glucose"], [2688, "glucose"], [672, "physical"]],
            "timestral-209m": [[336, "glucose"], [672, "glucose"]]}
    fingerprint = {"config": config, "plan": plan, "dataset": dataset.metadata["fingerprint"],
        "code": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__).resolve(), ROOT / "src/personal_timestral.py"]}}
    init_stage(output, "timestral-long", fingerprint)
    previous = json.loads((output / "timestral-selection.json").read_text())["candidates_sparse"]
    previous += json.loads((output / "timestral-extended-selection.json").read_text())["candidates_sparse"]
    valid, fold_labels = all_validation(dataset, config)
    sparse = valid[valid % config["timestral_selection_stride_readings"] == 0]
    additional, dense = [], []
    for model, contexts in plan.items():
        choices = [r for r in previous if r["spec"]["model"] == model]
        predictor = Predictor(dataset, config, model)
        new_names = []
        for hours, mode in contexts:
            if hours * 12 > 32768:
                raise ValueError("Context exceeds pretrained Timestral length")
            spec = {"name": f"{model}_{mode}_{hours}h", "family": "timestral", "model": model,
                    "mode": mode, "context_hours": hours}
            prediction = forecast_file(predictor, sparse, spec, output / "timestral" / ("sparse_" + spec["name"] + ".npz"))
            candidate = {"spec": spec, "metrics": metrics(dataset.targets[sparse], prediction)}
            choices.append(candidate)
            additional.append(candidate)
            new_names.append(spec["name"])
            print(f"Contesto lungo {spec['name']}: {np.round(candidate['metrics']['mae'], 2).tolist()}", flush=True)
        chosen = min(choices, key=lambda c: np.mean(c["metrics"]["mae"]))
        if chosen["spec"]["name"] in new_names:
            spec = chosen["spec"]
            prediction = forecast_file(predictor, valid, spec, output / "timestral" / ("dense_" + spec["name"] + ".npz"))
            np.savez_compressed(output / "oof" / (spec["name"] + ".npz"), positions=valid, prediction=prediction, folds=fold_labels)
            dense.append({"spec": spec, "metrics": metrics(dataset.targets[valid], prediction)})
        del predictor
        gc.collect()
    (output / "timestral-long-selection.json").write_text(json.dumps({"candidates_sparse": additional, "candidates": dense,
        "note": "28M glucose 8/16 weeks, physical 4 weeks, and 209M glucose 2/4 weeks; within pretrained context limit; same validation origins, no calibration/test selection."}, indent=2))
