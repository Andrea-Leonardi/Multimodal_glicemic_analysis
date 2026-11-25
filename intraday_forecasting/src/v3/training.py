"""Chronological comparisons for stronger corrections, rapid weighting and curves."""

import gc
import json
import time

import numpy as np

from src.v3.common import (LongInputs, atomic_joblib, atomic_json, dense_targets,
    folds, load_forecast, origin_weights, save_forecast, score, stage)
from src.v3.models import create


def specs(kind):
    if kind == "trajectory":
        return [{"name": "trajectory_huber", "family": "trajectory", "weighted": False, "residual": False},
                {"name": "trajectory_rapid_slope", "family": "trajectory", "weighted": True, "residual": False}]
    return [{"name": "residual_long", "family": "boost", "residual": True, "weighted": False},
            {"name": "residual_long_rapid", "family": "boost", "residual": True, "weighted": True},
            {"name": "direct_rapid", "family": "boost", "residual": False, "weighted": True}]


def fit(dataset, positions, spec, inputs, cfg, device, validation=None, epochs=None):
    x, offset = inputs.get(positions, spec.get("residual", False))
    model = create(spec, cfg, device)
    target = dense_targets(dataset, positions, dataset.metadata["config"], cfg["dense_steps"]) if spec["family"] == "trajectory" else dataset.targets[positions]
    kwargs = {"weights": origin_weights(dataset, positions, cfg, spec["weighted"])}
    if spec["family"] == "trajectory":
        kwargs.update(sequence=np.asarray(dataset.sequence[positions]), fixed_epochs=epochs)
        if validation is not None:
            vx, vo = inputs.get(validation)
            kwargs["validation"] = (vx, dense_targets(dataset, validation, dataset.metadata["config"], cfg["dense_steps"]), vo, np.asarray(dataset.sequence[validation]))
    model.fit(x, target, offset, **kwargs)
    return model


def predict(dataset, positions, spec, inputs, model, curves=False):
    result, full = [], []
    for start in range(0, len(positions), 1024):
        ix = positions[start:start+1024]
        x, offset = inputs.get(ix, spec.get("residual", False))
        if spec["family"] == "trajectory":
            curve = model.predict_curve(x, offset, np.asarray(dataset.sequence[ix]))
            full.append(curve)
            result.append(curve[:, [5, 11, 23, 35]])
        else:
            result.append(model.predict(x, offset))
    pred = np.concatenate(result).astype(np.float32)
    return (pred, np.concatenate(full).astype(np.float32) if full else None) if curves else pred


def run(dataset, base, cfg, output, device, kind="tabular"):
    fingerprint = stage(output, kind, dataset, cfg, ["src/v3/training.py", "src/v3/models.py", "src/v3/trajectory_network.py", "src/v3/cache.py"])
    inputs = LongInputs(dataset, output)
    candidates = []
    for spec in specs(kind):
        positions, predictions, fold_ids, details = [], [], [], []
        for k, (date, train, valid) in enumerate(folds(dataset, base)):
            path = output / "oof" / f"{spec['name']}_{date}.npz"
            if path.exists():
                prediction = load_forecast(path, valid)
                detail = json.loads(path.with_suffix(".json").read_text())
            else:
                started = time.perf_counter()
                model = fit(dataset, train, spec, inputs, cfg, device, validation=valid)
                prediction, full = predict(dataset, valid, spec, inputs, model, curves=True)
                save_forecast(path, valid, prediction, **({"curve": full} if full is not None else {}))
                detail = {"fold": date, "train_rows": len(train), "epochs": model.epochs_used,
                          "seconds": time.perf_counter()-started, "metrics": score(dataset.targets[valid], prediction)}
                atomic_json(path.with_suffix(".json"), detail)
                if hasattr(model, "to"):
                    model.to("cpu")
                atomic_joblib(output / "cv-models" / f"{spec['name']}_{date}.joblib", {"model": model, "spec": spec, "fingerprint": fingerprint})
                del model
                gc.collect()
            print(f"{spec['name']} / {date}: MAE {np.round(detail['metrics']['mae'], 2).tolist()}", flush=True)
            positions.append(valid)
            predictions.append(prediction)
            fold_ids.append(np.full(len(valid), k))
            details.append(detail)
        pos, pred = np.concatenate(positions), np.concatenate(predictions)
        save_forecast(output / "oof" / f"{spec['name']}.npz", pos, pred, folds=np.concatenate(fold_ids))
        candidates.append({"spec": spec, "metrics": score(dataset.targets[pos], pred), "folds": details})
    atomic_json(output / f"{kind}-selection.json", {"candidates": candidates})
