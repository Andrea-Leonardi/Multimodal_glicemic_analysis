"""Pinned Timestral benchmarks, longer contexts, physical covariates and readout.

No future meal/insulin is supplied. The only known-future inputs are calendar
variables. Learned personal readouts use cached frozen decoder representations.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data import ROOT, sha256
from src.forecasting import TimestralAdapter
from src.personal_data import historical_inputs, sample
from src.personal_experiment import all_validation, init_stage, metrics

PHYSICAL = ["iob_bolus_d360_p75", "activity_bolus_d360_p75", "carb_remaining_proxy_60", "activity_basal_d360_p75"]


def calendar_covariates(origins, length, horizon, timezone):
    queries = origins[:, None] + np.arange(-length + 1, horizon + 1)[None, :] * 300
    local = pd.to_datetime(queries.ravel(), unit="s", utc=True).tz_convert(timezone)
    hours = np.asarray(local.hour + local.minute / 60 + local.second / 3600).reshape(queries.shape)
    return np.stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)], axis=1).astype(np.float32)


class Predictor:
    def __init__(self, dataset, config, model):
        self.dataset, self.config, self.name = dataset, config, model
        self.adapter = TimestralAdapter(model, config["model_revisions"][model], ROOT / ".cache/huggingface", config["threads"])

    def predict(self, positions, spec, embeddings=False):
        context, indices = historical_inputs(self.dataset, positions, spec["context_hours"], self.config["history_tolerance_seconds"], block_size=32)
        past, known = None, None
        if spec["mode"] == "physical":
            columns = [self.dataset.feature_names.index(name) for name in PHYSICAL]
            past = np.stack([sample(self.dataset.features[:, column], indices) for column in columns], axis=1)
            known = calendar_covariates(self.dataset.times[positions], context.shape[-1], 36, self.config["timezone"])
        captured = []
        hook = None
        if embeddings:
            if spec["mode"] != "glucose":
                raise ValueError("Readout cache is defined for univariate glucose only")
            def capture(module, arguments):
                captured.append(arguments[0].detach().cpu().numpy().copy())
            hook = self.adapter.forecaster.model.head.register_forward_pre_hook(capture)
        try:
            result = self.adapter.forecaster.forecast(context[:, None, :], prediction_length=36,
                past_covariates=past, known_covariates=known)
        finally:
            if hook:
                hook.remove()
        quantiles = result.quantiles[:, 0].cpu().numpy()
        if not np.isfinite(quantiles).all():
            raise ValueError("Nonfinite Timestral predictions")
        steps = np.asarray(self.config["horizons_minutes"]) // 5 - 1
        prediction = quantiles[:, steps, 10].astype(np.float32)
        representation = None
        if embeddings:
            # Original and sign-mirrored passes; each contains two decoder tokens.
            if len(captured) != 2:
                raise ValueError("Unexpected Timestral symmetric decoding API")
            representation = np.concatenate([part.reshape(len(positions), -1) for part in captured], axis=1).astype(np.float32)
        return prediction, quantiles[:, steps], representation


def forecast_file(predictor, positions, spec, path, *, embeddings=False):
    path.parent.mkdir(exist_ok=True, parents=True)
    if path.exists():
        with np.load(path) as saved:
            if not np.array_equal(saved["positions"], positions):
                raise ValueError("Timestral cached origin mismatch")
            return saved["prediction"]
    started = time.perf_counter()
    predictions, quantiles = [], []
    batch_size = predictor.config["timestral_batch_size"]
    for first in range(0, len(positions), batch_size):
        pred, quantile, _ = predictor.predict(positions[first:first + batch_size], spec)
        predictions.append(pred)
        quantiles.append(quantile)
        if (first // batch_size + 1) % 100 == 0:
            print(f"  {spec['name']}: {min(first + batch_size, len(positions))}/{len(positions)}", flush=True)
    result = np.concatenate(predictions)
    np.savez_compressed(path, positions=positions, prediction=result, quantiles=np.concatenate(quantiles))
    path.with_suffix(".json").write_text(json.dumps({"spec": spec, "origins": len(positions), "seconds": time.perf_counter() - started}, indent=2))
    print(f"{spec['name']}: {len(positions)} origini, {time.perf_counter() - started:.1f}s", flush=True)
    return result


def cache_fingerprint(dataset, config):
    return {"config": config, "dataset": dataset.metadata["fingerprint"], "code": {str(p.relative_to(ROOT)): sha256(p) for p in [
        ROOT / "src/personal_timestral.py", ROOT / "src/personal_experiment.py", ROOT / "src/forecasting.py"]}}


def load_personal_cache(output):
    folder = Path(output) / "timestral-personal-cache"
    if not (folder / "complete.json").exists():
        raise ValueError("Complete the Timestral stage before personal readout training")
    return {name: np.load(folder / (name + ".npy"), mmap_mode="r") for name in ["positions", "prediction", "embedding"]}


def build_personal_cache(dataset, config, output):
    """Representation of known context only, usable for personal supervised fits."""
    folder = output / "timestral-personal-cache"
    folder.mkdir(exist_ok=True)
    positions = np.flatnonzero(dataset.eligible & (np.arange(len(dataset.times)) % config["train_stride_readings"] == 0))
    if (folder / "complete.json").exists():
        if not np.array_equal(np.load(folder / "positions.npy"), positions):
            raise ValueError("Representation cache positions differ")
        return
    predictor = Predictor(dataset, config, "timestral-28m")
    spec = {"name": "timestral28_readout24h", "model": "timestral-28m", "mode": "glucose", "context_hours": 24}
    batch_size = max(config["timestral_batch_size"], 16)
    predictions = np.lib.format.open_memmap(folder / "prediction.npy", mode="w+", dtype=np.float32, shape=(len(positions), 4))
    representation = None
    started = time.perf_counter()
    for first in range(0, len(positions), batch_size):
        batch = positions[first:first + batch_size]
        pred, _, hidden = predictor.predict(batch, spec, embeddings=True)
        if representation is None:
            representation = np.lib.format.open_memmap(folder / "embedding.npy", mode="w+", dtype=np.float32, shape=(len(positions), hidden.shape[1]))
        predictions[first:first + len(batch)] = pred
        representation[first:first + len(batch)] = hidden
        if (first // batch_size + 1) % 200 == 0:
            predictions.flush()
            representation.flush()
            print(f"  Timestral readout: {first + len(batch)}/{len(positions)}", flush=True)
    predictions.flush()
    representation.flush()
    np.save(folder / "positions.npy", positions, allow_pickle=False)
    (folder / "complete.json").write_text(json.dumps({"spec": spec, "positions": len(positions), "embedding_dimensions": representation.shape[1],
        "seconds": time.perf_counter() - started, "fingerprint": cache_fingerprint(dataset, config)}, indent=2))
    del predictor, predictions, representation
    gc.collect()


def run_timestral(dataset, config, output):
    output = Path(output)
    init_stage(output, "timestral", cache_fingerprint(dataset, config))
    valid, fold_labels = all_validation(dataset, config)
    sparse_mask = valid % config["timestral_selection_stride_readings"] == 0
    sparse = valid[sparse_mask]
    all_candidates, selected = [], []
    for model in config["timestral_models"]:
        predictor = Predictor(dataset, config, model)
        choices = []
        modes = [(hours, "glucose") for hours in config["timestral_context_hours"]]
        # The physical+calendar representation is an explicit group experiment;
        # it is not claimed to isolate the causal effect of any single covariate.
        modes += [(24, "physical"), (72, "physical")] if model == "timestral-28m" else [(24, "physical")]
        for hours, mode in modes:
            spec = {"name": f"{model}_{mode}_{hours}h", "family": "timestral", "model": model, "context_hours": hours, "mode": mode}
            path = output / "timestral" / ("sparse_" + spec["name"] + ".npz")
            prediction = forecast_file(predictor, sparse, spec, path)
            score = metrics(dataset.targets[sparse], prediction)
            candidate = {"spec": spec, "metrics": score}
            choices.append(candidate)
            all_candidates.append(candidate)
            print(f"  Selezione {spec['name']}: {np.round(score['mae'], 2).tolist()}", flush=True)
        chosen = min(choices, key=lambda c: np.mean(c["metrics"]["mae"]))
        spec = chosen["spec"]
        dense = forecast_file(predictor, valid, spec, output / "timestral" / ("dense_" + spec["name"] + ".npz"))
        np.savez_compressed(output / "oof" / (spec["name"] + ".npz"), positions=valid, prediction=dense, folds=fold_labels)
        selected.append({"spec": spec, "metrics": metrics(dataset.targets[valid], dense)})
        del predictor
        gc.collect()
    (output / "timestral-selection.json").write_text(json.dumps({"candidates_sparse": all_candidates, "candidates": selected,
        "selection_note": "Context/mode selected on sparse April/June/July validation; dense OOF shares all personal-model origins."}, indent=2))
    build_personal_cache(dataset, config, output)
