"""Horizon-specific MAE boosting and a validation-only 90-day recency check."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data import ROOT, sha256
from src.personal_data import configuration
from src.personal_experiment import folds, init_stage, load_dataset, metrics
from src.personal_models import create as original_create


class DirectMAEModel:
    """Four separate conditional-median estimators, fitted to observed labels."""
    def __init__(self, spec, seed=42, threads=8):
        self.spec, self.seed, self.threads = spec, seed, threads
        self.epochs_used = None

    def fit(self, x, y, current_glucose, **kwargs):
        from catboost import CatBoostRegressor
        offset = current_glucose[:, None] if current_glucose.ndim == 1 else current_glucose
        residual = y - offset
        self.models = []
        for k in range(y.shape[1]):
            valid = np.isfinite(residual[:, k])
            model = CatBoostRegressor(iterations=self.spec["iterations"], depth=self.spec["depth"],
                learning_rate=.04, l2_leaf_reg=self.spec["l2"], loss_function="MAE",
                thread_count=self.threads, random_seed=self.seed, verbose=False, allow_writing_files=False)
            model.fit(x[valid], residual[valid, k])
            self.models.append(model)
        return self

    def predict(self, x, current_glucose, **kwargs):
        offset = current_glucose[:, None] if current_glucose.ndim == 1 else current_glucose
        return np.column_stack([model.predict(x) for model in self.models]) + offset


def create(spec, seed=42, threads=8):
    return DirectMAEModel(spec, seed, threads) if spec["family"] == "direct_mae" else original_create(spec, seed, threads)


def training_window(dataset, positions, spec, boundary):
    days = spec.get("train_window_days")
    return positions if days is None else positions[dataset.times[positions] >= boundary - days * 86400]


def run(dataset, config, output):
    output = Path(output)
    fingerprint = {"config": config, "dataset": dataset.metadata["fingerprint"],
        "code": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__).resolve(), ROOT / "src/personal_experiment.py", ROOT / "src/personal_models.py"]}}
    init_stage(output, "direct-boosting", fingerprint)
    base = {"family": "direct_mae", "depth": 6, "iterations": 700, "l2": 15.}
    specs = [dict(base, name="catboost_mae_d6"), dict(base, name="catboost_mae_d6_recent90d", train_window_days=90)]
    candidates = []
    for spec in specs:
        predictions, positions, labels, details = [], [], [], []
        for k, (start, train, valid) in enumerate(folds(dataset, config)):
            boundary = pd.Timestamp(start, tz=config["timezone"]).timestamp()
            train = training_window(dataset, train, spec, boundary)
            path = output / "oof" / f"{spec['name']}_{start}.npz"
            if path.exists():
                with np.load(path) as saved:
                    if not np.array_equal(saved["positions"], valid):
                        raise ValueError("Direct boosting OOF origins differ")
                    prediction = saved["prediction"]
                detail = json.loads(path.with_suffix(".json").read_text())
            else:
                started = time.perf_counter()
                model = create(spec, config["seed"], config["threads"])
                model.fit(np.asarray(dataset.features[train]), dataset.targets[train], dataset.glucose[train])
                prediction = model.predict(np.asarray(dataset.features[valid]), dataset.glucose[valid]).astype(np.float32)
                np.savez_compressed(path, positions=valid, prediction=prediction)
                detail = {"fold": start, "train_rows": len(train), "epochs": None,
                          "seconds": time.perf_counter() - started, "metrics": metrics(dataset.targets[valid], prediction)}
                path.with_suffix(".json").write_text(json.dumps(detail, indent=2))
                del model
                gc.collect()
            print(f"{spec['name']} / {start}: MAE {np.round(detail['metrics']['mae'], 2).tolist()} ({detail['seconds']:.1f}s)", flush=True)
            predictions.append(prediction)
            positions.append(valid)
            labels.append(np.full(len(valid), k))
            details.append(detail)
        pos, pred = np.concatenate(positions), np.concatenate(predictions)
        np.savez_compressed(output / "oof" / (spec["name"] + ".npz"), positions=pos, prediction=pred, folds=np.concatenate(labels))
        candidates.append({"spec": spec, "metrics": metrics(dataset.targets[pos], pred), "folds": details})
    (output / "direct-boosting-selection.json").write_text(json.dumps({"candidates": candidates,
        "note": "Separate horizon MAE losses; full history versus trailing 90 training days, same validation origins and purging."}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "outputs/personal-v2"))
    parser.add_argument("--dataset", default=str(ROOT / "outputs/memory-check/dataset"))
    args = parser.parse_args()
    run(load_dataset(args.dataset), configuration(), args.output)
