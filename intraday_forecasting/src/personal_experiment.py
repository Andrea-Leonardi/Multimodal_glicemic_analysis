"""Train, select and evaluate personal models using expanding temporal folds.

All heavy commands should run through scripts/run_limited.py. Stages can resume.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]
import time

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT, sha256
from src.personal_data import Dataset, configuration, split_positions
from src.personal_models import candidate_specs, create, feature_columns


def load_dataset(folder=None):
    folder = Path(folder or ROOT / "outputs/memory-check/dataset")
    meta = json.loads((folder / "metadata.json").read_text())
    # Training need not depend on the test helper's hash, but numerical source
    # and CSV hashes must still match the verified, immutable cached dataset.
    checked = {**meta["source_sha256"], "src/personal_data.py": meta["fingerprint"]["src/personal_data.py"],
               "src/data.py": meta["fingerprint"]["src/data.py"]}
    for name, expected in checked.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Dataset source changed: {name}; rebuild safely into a new directory")
    arrays = {name: np.load(folder / (name + ".npy"), mmap_mode="r", allow_pickle=False)
              for name in ["times", "glucose", "features", "targets", "target_indices", "sequence", "eligible"]}
    return Dataset(**arrays, feature_names=json.loads((folder / "feature_names.json").read_text()),
                   event_table=pd.read_csv(folder / "events.csv"), streams={}, metadata=meta)


def utc_boundary(value, config):
    return pd.Timestamp(value, tz=config["timezone"]).tz_convert("UTC")


def folds(dataset, config):
    beginning = pd.Timestamp(float(dataset.times[0]), unit="s", tz="UTC")
    result = []
    for start, end in config["folds"]:
        a, b = utc_boundary(start, config), utc_boundary(end, config)
        train = split_positions(dataset, beginning, a, config["train_stride_readings"], training=True)
        validation = split_positions(dataset, a, b, config["evaluation_stride_readings"], training=True)
        assert (dataset.times[train] + max(config["horizons_minutes"]) * 60 + config["target_tolerance_seconds"] < a.timestamp()).all()
        result.append((start, train, validation))
    return result


def all_validation(dataset, config):
    groups = folds(dataset, config)
    return np.concatenate([v for _, _, v in groups]), np.concatenate([np.full(len(v), k) for k, (_, _, v) in enumerate(groups)])


def stage_fingerprint(dataset, config, stage):
    sources = [ROOT / "src/personal_experiment.py", ROOT / "src/personal_models.py", ROOT / "src/personal_network.py"]
    if stage == "hybrid":
        sources += [ROOT / "src/personal_timestral.py"]
    return {"config": config, "dataset": dataset.metadata["fingerprint"],
            "code": {str(p.relative_to(ROOT)): sha256(p) for p in sources}}


def init_stage(output, stage, fingerprint):
    output.mkdir(parents=True, exist_ok=True)
    path = output / (stage + "-manifest.json")
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise ValueError(f"{stage} code/config differs; choose a new output directory")
    else:
        versions = {}
        for package in ["numpy", "pandas", "scikit-learn", "catboost", "torch", "timestral"]:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        path.write_text(json.dumps({"fingerprint": fingerprint, "versions": versions}, indent=2))


def metrics(y, prediction):
    error = np.asarray(prediction) - np.asarray(y)
    return {"mae": np.nanmean(np.abs(error), axis=0).tolist(),
            "rmse": np.sqrt(np.nanmean(error ** 2, axis=0)).tolist(),
            "n": np.isfinite(error).sum(axis=0).tolist()}


class Inputs:
    def __init__(self, dataset, output):
        self.dataset, self.output = dataset, output
        self.ts_cache = None

    def get(self, spec, positions):
        d = self.dataset
        selected = feature_columns(d.feature_names, spec.get("features", "all"))
        x = np.asarray(d.features[positions][:, selected], dtype=np.float32)
        offset = np.asarray(d.glucose[positions])
        if spec.get("base") == "timestral24":
            if self.ts_cache is None:
                from src.personal_timestral import load_personal_cache
                self.ts_cache = load_personal_cache(self.output)
            cached = self.ts_cache
            index = np.searchsorted(cached["positions"], positions)
            if not np.array_equal(cached["positions"][index], positions):
                raise ValueError("Timestral personal cache misses required positions")
            offset = np.asarray(cached["prediction"][index])
            extra = [offset - np.asarray(d.glucose[positions])[:, None]]
            if spec.get("embedding"):
                extra.append(np.asarray(cached["embedding"][index]))
            x = np.column_stack([x, *extra]).astype(np.float32)
        return x, offset


def hybrid_specs(output):
    chosen = json.loads((output / "tabular-selection.json").read_text())
    tree = min([s for s in chosen["candidates"] if s["spec"]["family"] == "catboost"], key=lambda r: np.mean(r["metrics"]["mae"]))["spec"]
    return [dict(tree, name="timestral_catboost_residual", base="timestral24"),
            {"name": "timestral_readout_ridge100", "family": "ridge", "alpha": 100., "base": "timestral24", "embedding": True},
            {"name": "timestral_readout_ridge10000", "family": "ridge", "alpha": 10000., "base": "timestral24", "embedding": True}]


def cv_stage(dataset, config, output, stage="tabular"):
    init_stage(output, stage, stage_fingerprint(dataset, config, stage))
    inputs = Inputs(dataset, output)
    specs = candidate_specs() if stage == "tabular" else hybrid_specs(output)
    if stage == "tabular":
        # Same boosting parameters across these ablations distinguish raw events
        # and modeled exposure without changing the test set or target definitions.
        base = next(s for s in specs if s["name"] == "catboost_d6")
        specs += [dict(base, name="catboost_glucose_only", features="glucose"),
                  dict(base, name="catboost_raw_events", features="raw"),
                  dict(base, name="catboost_humalog360", features="humalog360")]
    oof_folder = output / "oof"
    oof_folder.mkdir(exist_ok=True)
    groups = folds(dataset, config)
    candidates = []
    for spec in specs:
        predictions, positions, fold_labels, details = [], [], [], []
        for fold_index, (start, train, valid) in enumerate(groups):
            path = oof_folder / f"{spec['name']}_{start}.npz"
            if path.exists():
                with np.load(path) as saved:
                    if not np.array_equal(saved["positions"], valid):
                        raise ValueError("OOF cache origin mismatch")
                    pred = saved["prediction"]
                detail = json.loads(path.with_suffix(".json").read_text())
            else:
                began = time.perf_counter()
                tx, tg = inputs.get(spec, train)
                vx, vg = inputs.get(spec, valid)
                model = create(spec, config["seed"], config["threads"])
                train_sequence = np.asarray(dataset.sequence[train]) if spec.get("sequence") else None
                valid_sequence = np.asarray(dataset.sequence[valid]) if spec.get("sequence") else None
                model.fit(tx, dataset.targets[train], tg, sequence=train_sequence,
                          validation=(vx, dataset.targets[valid], vg, valid_sequence))
                pred = model.predict(vx, vg, sequence=valid_sequence).astype(np.float32)
                # Save predictions before reading the validation outcome in scoring.
                np.savez_compressed(path, positions=valid, prediction=pred)
                detail = {"fold": start, "train_rows": len(train), "validation_rows": len(valid),
                          "epochs": model.epochs_used, "seconds": time.perf_counter() - began,
                          "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource is not None else 0.0,
                          "metrics": metrics(dataset.targets[valid], pred)}
                path.with_suffix(".json").write_text(json.dumps(detail, indent=2))
                del model, tx, vx, train_sequence, valid_sequence
                gc.collect()
            print(f"{spec['name']} / {start}: MAE {np.round(detail['metrics']['mae'], 2).tolist()} ({detail['seconds']:.1f}s)", flush=True)
            predictions.append(pred)
            positions.append(valid)
            fold_labels.append(np.full(len(valid), fold_index))
            details.append(detail)
        pos, pred = np.concatenate(positions), np.concatenate(predictions)
        total = metrics(dataset.targets[pos], pred)
        candidate = {"spec": spec, "metrics": total, "folds": details}
        candidates.append(candidate)
        np.savez_compressed(oof_folder / f"{spec['name']}.npz", positions=pos, prediction=pred, folds=np.concatenate(fold_labels))
    path = output / (stage + "-selection.json")
    path.write_text(json.dumps({"candidates": candidates}, indent=2))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["tabular", "timestral", "hybrid", "final", "all"], default="all")
    parser.add_argument("--output", default=str(ROOT / "outputs/personal-v2"))
    parser.add_argument("--dataset", default=str(ROOT / "outputs/memory-check/dataset"))
    args = parser.parse_args()
    output = Path(args.output)
    config = configuration()
    dataset = load_dataset(args.dataset)
    for stage in ["tabular", "timestral", "hybrid", "final"] if args.stage == "all" else [args.stage]:
        if stage in {"tabular", "hybrid"}:
            cv_stage(dataset, config, output, stage)
        elif stage == "timestral":
            from src.personal_timestral import run_timestral
            run_timestral(dataset, config, output)
        else:
            from src.personal_evaluate import finalize
            finalize(dataset, config, output)


if __name__ == "__main__":
    main()
