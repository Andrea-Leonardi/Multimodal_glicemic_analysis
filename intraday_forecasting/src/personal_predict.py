"""Reuse the fitted historical predictor, without reading target glucose values.

python -m src.personal_predict --at 2026-10-15T12:00:00+02:00
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT, sha256
from src.personal_data import configuration
from src.personal_experiment import load_dataset, utc_boundary
from src.personal_models import feature_columns
from src.personal_timestral import Predictor
from src.personal_evaluate import recent_meal, apply_intervals


def predict_at(dataset, config, output, at):
    output = Path(output)
    fingerprint = json.loads((output / "final-manifest.json").read_text())["fingerprint"]
    if fingerprint["config"] != config or fingerprint["dataset"] != dataset.metadata["fingerprint"]:
        raise ValueError("Saved experiment configuration or dataset differs")
    for name, expected in fingerprint["code"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Saved experiment numerical code changed: {name}")
    timestamp = pd.Timestamp(at)
    if timestamp.tzinfo is None:
        raise ValueError("--at requires an explicit timezone")
    position = np.searchsorted(dataset.times, timestamp.timestamp(), side="right") - 1
    if position < 0 or timestamp.timestamp() - dataset.times[position] > 300 or not dataset.eligible[position]:
        raise ValueError("No sufficiently recent eligible sensor reading for requested time")
    if dataset.times[position] < utc_boundary(config["test_start"], config).timestamp():
        raise ValueError("Saved model and August calibration were not yet available at this origin; use September or later")
    positions = np.array([position])
    selection = json.loads((output / "selection-frozen.json").read_text())
    specs = {c["spec"]["name"]: c["spec"] for c in selection["selected"]}
    names = set(name for item in selection["ensemble"] for name, w in zip(item["models"], item["weights"]) if w > 0)
    predictions, cached_base = {}, None
    for name in sorted(names):
        spec = specs[name]
        if spec["family"] == "timestral":
            predictions[name] = Predictor(dataset, config, spec["model"]).predict(positions, spec)[0]
            continue
        artifact = joblib.load(output / "models" / (name + ".joblib"))
        if artifact["fingerprint"] != fingerprint:
            raise ValueError("Model artifact belongs to a different experiment")
        if artifact["feature_names"] != dataset.feature_names:
            raise ValueError("Saved feature order differs from dataset")
        columns = feature_columns(dataset.feature_names, spec.get("features", "all"))
        x = np.asarray(dataset.features[positions][:, columns], dtype=np.float32)
        offset = np.asarray(dataset.glucose[positions])
        if spec.get("base") == "timestral24":
            if cached_base is None:
                cached_base = Predictor(dataset, config, "timestral-28m").predict(positions,
                    {"context_hours": 24, "mode": "glucose"}, embeddings=True)
            offset = cached_base[0]
            extra = [offset - np.asarray(dataset.glucose[positions])[:, None]]
            if spec.get("embedding"):
                extra.append(cached_base[2])
            x = np.column_stack([x, *extra])
        sequence = np.asarray(dataset.sequence[positions]) if spec.get("sequence") else None
        predictions[name] = artifact["model"].predict(x, offset, sequence=sequence)
    combined = np.array([[sum(w * predictions[n][0, k] for n, w in zip(item["models"], item["weights"]) if w > 0)
                          for k, item in enumerate(selection["ensemble"])]])
    calibration = json.loads((output / "predictions/intervals_selected_personal.json").read_text())
    meal = recent_meal(dataset, positions)
    lo, hi = apply_intervals(combined, meal, calibration)
    return {"origin": pd.Timestamp(float(dataset.times[position]), unit="s", tz="UTC").isoformat(),
            "glucose": float(dataset.glucose[position]), "horizons_minutes": config["horizons_minutes"],
            "prediction_mgdl": combined[0].tolist(), "lower90": lo[0].tolist(), "upper90": hi[0].tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--at", required=True)
    p.add_argument("--experiment", default=str(ROOT / "outputs/personal-v2"))
    p.add_argument("--dataset", default=str(ROOT / "outputs/memory-check/dataset"))
    args = p.parse_args()
    print(json.dumps(predict_at(load_dataset(args.dataset), configuration(), args.experiment, args.at), indent=2))


if __name__ == "__main__":
    main()
