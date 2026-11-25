"""Weekly expanding-window refits with explicitly available training labels."""

import gc
import json
import time

import joblib
import numpy as np

from src.v3.common import (LongInputs, atomic_joblib, atomic_json, folds, load_forecast,
    save_forecast, score, stage, training_positions, utc_boundary)
from src.v3.training import fit, predict


def specs(cfg):
    return [{"name": name, "family": "boost", "residual": "residual" in name,
             "weighted": False, "policy": "weekly"} for name in cfg["weekly"]["models"]]


def cutoffs(times, anchor, days):
    return anchor + np.floor((np.asarray(times) - anchor) / (days * 86400)) * days * 86400


def predict_phase(dataset, base, cfg, output, device, spec, positions, anchor, phase, scope):
    path = output / "weekly" / f"{spec['name']}_{phase}.npz"
    epochs = cutoffs(dataset.times[positions], anchor, cfg["weekly"]["update_days"])
    if path.exists():
        result = load_forecast(path, positions)
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["training_cutoff"], epochs)
        return result
    inputs = LongInputs(dataset, output)
    result = np.full((len(positions), 4), np.nan, np.float32)
    for cutoff in np.unique(epochs):
        mask = epochs == cutoff
        train = training_positions(dataset, base, float(cutoff))
        model_path = output / "weekly-models" / spec["name"] / scope / f"{int(cutoff)}.joblib"
        started = time.perf_counter()
        if model_path.exists():
            artifact = joblib.load(model_path)
            if artifact["training_cutoff"] != cutoff or artifact["spec"] != spec:
                raise ValueError("Weekly checkpoint policy differs")
            model = artifact["model"]
        else:
            model = fit(dataset, train, spec, inputs, cfg, device)
            upper = float(dataset.times[train[-1]] + 180 * 60 + base["target_tolerance_seconds"])
            if upper >= cutoff:
                raise ValueError("Weekly update used labels unavailable at its cutoff")
            atomic_joblib(model_path, {"model": model, "spec": spec, "training_cutoff": float(cutoff),
                "training_rows": len(train), "training_origin_last": float(dataset.times[train[-1]]), "label_upper_bound": upper})
        result[mask] = predict(dataset, positions[mask], spec, inputs, model)
        print(f"{spec['name']} / {phase} / {int(cutoff)}: {len(train)} training, {mask.sum()} previsioni ({time.perf_counter()-started:.1f}s)", flush=True)
        del model
        gc.collect()
    save_forecast(path, positions, result, training_cutoff=epochs)
    return result


def run(dataset, base, cfg, output, device):
    stage(output, "rolling", dataset, cfg, ["src/v3/rolling.py", "src/v3/training.py", "src/v3/models.py", "src/v3/cache.py"])
    candidates = []
    for spec in specs(cfg):
        positions, predictions, fold_ids = [], [], []
        for k, (date, train, valid) in enumerate(folds(dataset, base)):
            prediction = predict_phase(dataset, base, cfg, output, device, spec, valid,
                utc_boundary(date, base).timestamp(), date, date)
            positions.append(valid)
            predictions.append(prediction)
            fold_ids.append(np.full(len(valid), k))
        pos, pred = np.concatenate(positions), np.concatenate(predictions)
        save_forecast(output / "oof" / f"{spec['name']}.npz", pos, pred, folds=np.concatenate(fold_ids))
        candidates.append({"spec": spec, "metrics": score(dataset.targets[pos], pred)})
        print(f"{spec['name']}: MAE {np.round(candidates[-1]['metrics']['mae'],2).tolist()}", flush=True)
    atomic_json(output / "rolling-selection.json", {"candidates": candidates,
        "policy": "7-day UTC periods anchored at each validation start; expanding history; every training label purged before update cutoff."})
