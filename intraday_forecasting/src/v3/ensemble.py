"""Diversity-aware convex selection over the entire candidate pool."""

import json

import numpy as np

from src.data import ROOT, sha256
from src.v3.common import (all_validation, atomic_json, load_forecast, regimes,
    save_forecast, score, stage, dense_targets)


def greedy_weights(predictions, target, steps=20):
    """Caruana-style selection with replacement; no individual top-three filter."""
    valid = np.isfinite(target)
    p, y = np.asarray(predictions)[valid], np.asarray(target)[valid]
    if not len(y):
        raise ValueError("No observed outcomes for ensemble selection")
    total = np.zeros(len(y), np.float64)
    counts = np.zeros(p.shape[1], np.int64)
    best_loss, best_weights = np.inf, None
    for step in range(1, steps + 1):
        loss = np.mean(np.abs((total[:, None] + p) / step - y[:, None]), axis=0)
        selected = int(np.argmin(loss))
        counts[selected] += 1
        total += p[:, selected]
        if loss[selected] < best_loss - 1e-12:
            best_loss, best_weights = float(loss[selected]), counts.astype(np.float64) / step
    return best_weights


def fit_policy(matrix, target, group, cfg, gated=False):
    # matrix is N x 4 x candidates; all predictions precede their outcomes.
    weights = np.empty((3, target.shape[1], matrix.shape[-1]), np.float64)
    for k in range(target.shape[1]):
        global_weights = greedy_weights(matrix[:, k], target[:, k], cfg["ensemble"]["steps"])
        for regime in range(3):
            weights[regime, k] = global_weights
            selected = group == regime
            n = int(np.count_nonzero(selected & np.isfinite(target[:, k])))
            if gated and n >= cfg["ensemble"]["minimum_regime_rows"]:
                local = greedy_weights(matrix[selected, k], target[selected, k], cfg["ensemble"]["steps"])
                share = n / (n + cfg["ensemble"]["regime_shrinkage_rows"])
                weights[regime, k] = share * local + (1 - share) * global_weights
    return weights


def apply_policy(matrix, group, weights):
    weights = np.asarray(weights)
    if not np.isfinite(weights).all() or (weights < -1e-12).any() or not np.allclose(weights.sum(axis=-1), 1):
        raise ValueError("Ensemble weights must be finite convex combinations")
    return np.einsum("nhm,nhm->nh", matrix, weights[np.asarray(group)]).astype(np.float32)


def library(dataset, base, cfg, output):
    v2 = ROOT / cfg["v2"]
    old = json.loads((v2 / "selection-frozen.json").read_text())
    positions, fold_ids = all_validation(dataset, base)
    specs, predictions, artifacts = [], [], {}
    for entry in old["selected"]:
        original = entry["spec"]
        spec = dict(original, origin="v2")
        name = spec["name"]
        path = v2 / "oof" / (name + ".npz")
        if name == "timestral-28m_glucose_24h":
            folder = v2 / "timestral-personal-cache"
            cached_positions = np.load(folder / "positions.npy", mmap_mode="r")
            ix = np.searchsorted(cached_positions, positions)
            np.testing.assert_array_equal(cached_positions[ix], positions)
            pred = np.asarray(np.load(folder / "prediction.npy", mmap_mode="r")[ix])
            artifacts[str((folder / "prediction.npy").relative_to(ROOT))] = sha256(folder / "prediction.npy")
        else:
            pred = load_forecast(path, positions)
            artifacts[str(path.relative_to(ROOT))] = sha256(path)
        specs.append(spec)
        predictions.append(pred)
    for phase in ["tabular", "trajectory", "finetune", "rolling"]:
        source = output / (phase + "-selection.json")
        for entry in json.loads(source.read_text())["candidates"]:
            spec = dict(entry["spec"], origin="v3", validation_epochs=[f["epochs"] for f in entry.get("folds", []) if f.get("epochs")])
            path = output / "oof" / (spec["name"] + ".npz")
            specs.append(spec)
            predictions.append(load_forecast(path, positions))
            artifacts[str(path.relative_to(ROOT))] = sha256(path)
    # The already frozen v2 ensemble is a reference and fallback candidate.
    lookup = {s["name"]: p for s, p in zip(specs, predictions)}
    old_prediction = np.empty((len(positions), 4), np.float32)
    for k, entry in enumerate(old["ensemble"]):
        old_prediction[:, k] = sum(w * lookup[n][:, k] for n, w in zip(entry["models"], entry["weights"]))
    specs.append({"name": "v2_selected", "family": "reference", "origin": "v2"})
    predictions.append(old_prediction)
    names = [s["name"] for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ensemble candidate names")
    return positions, fold_ids, specs, np.stack(predictions, axis=-1), artifacts


def run(dataset, base, cfg, output, device):
    stage(output, "ensemble", dataset, cfg, ["src/v3/ensemble.py", "src/v3/models.py", "src/v3/training.py", "src/v3/finetune.py", "src/v3/rolling.py"])
    if base != dataset.metadata["config"]:
        raise ValueError("Base configuration differs from the verified dataset")
    if (output / "ensemble-selection.json").exists():
        frozen = json.loads((output / "selection-frozen.json").read_text())
        for name, expected in frozen["input_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Ensemble input changed: {name}")
        print("Selezione ensemble già completa e ingressi verificati", flush=True)
        return
    positions, fold_ids, specs, matrix, artifacts = library(dataset, base, cfg, output)
    y = np.asarray(dataset.targets[positions])
    group = regimes(dataset, positions, cfg)
    reference_index = next(i for i,s in enumerate(specs) if s["name"] == "v2_selected")
    reference = matrix[:, :, reference_index]
    reference_score = score(y, reference)
    choices = []
    for mode in ["global", "gated"]:
        crossfit = np.empty_like(y)
        for fold in np.unique(fold_ids):
            train, valid = fold_ids != fold, fold_ids == fold
            weights = fit_policy(matrix[train], y[train], group[train], cfg, gated=mode == "gated")
            crossfit[valid] = apply_policy(matrix[valid], group[valid], weights)
        unrestricted_score = score(y, crossfit)
        fallback = [k for k in range(4) if unrestricted_score["mae"][k] >= reference_score["mae"][k] - cfg["ensemble"]["minimum_gain_mgdl"]]
        for k in fallback:
            crossfit[:,k] = reference[:,k]
        choice = {"mode": mode, "metrics": score(y, crossfit), "unrestricted_metrics": unrestricted_score,
                  "fallback_horizons": fallback}
        choices.append(choice)
        save_forecast(output / "oof" / f"ensemble_{mode}_crossfit.npz", positions, crossfit, folds=fold_ids)
        print(f"Ensemble {mode}: diagnostica MAE {np.round(choice['metrics']['mae'],2).tolist()}", flush=True)
    best = min(choices, key=lambda row: np.mean(row["metrics"]["mae"]))
    # Keep the simpler global policy unless regime gating clears a fixed margin.
    if best["mode"] == "gated" and np.mean(choices[0]["metrics"]["mae"]) - np.mean(best["metrics"]["mae"]) < cfg["ensemble"]["minimum_gain_mgdl"]:
        best = choices[0]
    policy_weights = {}
    for choice in choices:
        fitted = fit_policy(matrix,y,group,cfg,gated=choice["mode"] == "gated")
        for k in choice["fallback_horizons"]:
            fitted[:,k,:] = 0
            fitted[:,k,reference_index] = 1
        policy_weights[choice["mode"]] = fitted
    weights = policy_weights[best["mode"]]
    dense_choices = []
    dense_actual = dense_targets(dataset, positions, base)
    for spec in specs:
        if spec["origin"] != "v3" or spec["family"] not in {"trajectory", "finetune"}:
            continue
        curves = []
        for date, _ in base["folds"]:
            with np.load(output / "oof" / f"{spec['name']}_{date}.npz") as saved:
                curves.append(saved["curve"])
        dense_choices.append({"name": spec["name"], "validation_dense_mae": float(np.nanmean(np.abs(np.concatenate(curves)-dense_actual)))})
    selection = {"mode": best["mode"], "weights": weights.tolist(), "candidates": specs,
                 "policy_weights": {mode: value.tolist() for mode,value in policy_weights.items()},
                 "dense_choices": dense_choices, "dense_model": min(dense_choices, key=lambda v: v["validation_dense_mae"])["name"],
                 "crossfit_diagnostics": choices, "input_sha256": artifacts,
                 "note": "All candidates eligible, no top-three restriction. Crossfit is a development diagnostic; final holdout was previously inspected in v2."}
    frozen = output / "selection-frozen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != selection:
        raise ValueError("Previously frozen v3 ensemble differs")
    atomic_json(frozen, selection)
    crossfit = load_forecast(output / "oof" / f"ensemble_{best['mode']}_crossfit.npz", positions)
    save_forecast(output / "oof/selected_crossfit.npz", positions, crossfit, folds=fold_ids)
    save_forecast(output / "oof/selected_fitted.npz", positions, apply_policy(matrix, group, weights), folds=fold_ids)
    atomic_json(output / "ensemble-selection.json", {"selected_mode": best["mode"], "choices": choices})
