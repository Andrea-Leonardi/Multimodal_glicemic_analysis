"""Validate whether using every 5-minute training reading improves the finalists."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from src.data import ROOT, sha256
from src.personal_data import configuration
from src.personal_experiment import folds, init_stage, load_dataset, metrics
from src.personal_models import create


def run(dataset, config, output):
    output = Path(output)
    fingerprint = {"config": config, "dataset": dataset.metadata["fingerprint"],
        "code": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__).resolve(), ROOT / "src/personal_models.py", ROOT / "src/personal_network.py", ROOT / "src/personal_experiment.py"]}}
    init_stage(output, "dense-training", fingerprint)
    earlier = json.loads((output / "tabular-selection.json").read_text())["candidates"]
    best_cat = min([c for c in earlier if c["spec"]["family"] == "catboost" and not c["spec"].get("features")], key=lambda c: np.mean(c["metrics"]["mae"]))["spec"]
    best_tcn = next(c["spec"] for c in earlier if c["spec"]["name"] == "tcn_huber")
    specs = [dict(s, name=s["name"] + "_dense", train_stride_readings=1) for s in [best_cat, best_tcn]]
    groups = folds(dataset, config | {"train_stride_readings": 1})
    candidates = []
    for spec in specs:
        predictions, positions, labels, details = [], [], [], []
        for k, (start, train, valid) in enumerate(groups):
            path = output / "oof" / f"{spec['name']}_{start}.npz"
            if path.exists():
                with np.load(path) as saved:
                    if not np.array_equal(saved["positions"], valid):
                        raise ValueError("Dense training OOF origins differ")
                    prediction = saved["prediction"]
                detail = json.loads(path.with_suffix(".json").read_text())
            else:
                started = time.perf_counter()
                x, vx = np.asarray(dataset.features[train]), np.asarray(dataset.features[valid])
                seq = np.asarray(dataset.sequence[train]) if spec.get("sequence") else None
                vseq = np.asarray(dataset.sequence[valid]) if spec.get("sequence") else None
                model = create(spec, config["seed"], config["threads"])
                model.fit(x, dataset.targets[train], dataset.glucose[train], sequence=seq,
                          validation=(vx, dataset.targets[valid], dataset.glucose[valid], vseq))
                prediction = model.predict(vx, dataset.glucose[valid], sequence=vseq).astype(np.float32)
                np.savez_compressed(path, positions=valid, prediction=prediction)
                detail = {"fold": start, "train_rows": len(train), "epochs": model.epochs_used,
                          "seconds": time.perf_counter() - started, "metrics": metrics(dataset.targets[valid], prediction)}
                path.with_suffix(".json").write_text(json.dumps(detail, indent=2))
                del model, x, vx, seq, vseq
                gc.collect()
            print(f"{spec['name']} / {start}: MAE {np.round(detail['metrics']['mae'], 2).tolist()} ({detail['seconds']:.1f}s)", flush=True)
            predictions.append(prediction)
            positions.append(valid)
            labels.append(np.full(len(valid), k))
            details.append(detail)
        pos, pred = np.concatenate(positions), np.concatenate(predictions)
        np.savez_compressed(output / "oof" / (spec["name"] + ".npz"), positions=pos, prediction=pred, folds=np.concatenate(labels))
        candidates.append({"spec": spec, "metrics": metrics(dataset.targets[pos], pred), "folds": details})
    (output / "dense-training-selection.json").write_text(json.dumps({"candidates": candidates,
        "note": "Same validation origins; compare training every reading vs every third reading; final fit respects selected training stride."}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "outputs/personal-v2"))
    parser.add_argument("--dataset", default=str(ROOT / "outputs/memory-check/dataset"))
    args = parser.parse_args()
    run(load_dataset(args.dataset), configuration(), args.output)
