"""Replay the frozen v3 historical policy without consulting outcome arrays."""

import argparse
import gc
import json

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT, sha256
from src.personal_models import feature_columns
from src.personal_experiment import Inputs as V2Inputs
from src.personal_predict import predict_at as v2_predict_at
from src.v3.cache import predictor
from src.v3.common import context, LongInputs, regimes, utc_boundary
from src.v3.ensemble import apply_policy
from src.v3.finetune import FineTuner
from src.v3.rolling import cutoffs


def member_prediction(dataset, base, cfg, output, device, spec, position, include_curve=False):
    positions = np.array([position])
    if spec["origin"] == "v2":
        old = ROOT / cfg["v2"]
        if spec["name"] == "v2_selected":
            at = pd.Timestamp(float(dataset.times[position]), unit="s", tz="UTC").isoformat()
            return np.array([v2_predict_at(dataset, base, old, at)["prediction_mgdl"]]), None
        if spec["family"] == "timestral":
            model = predictor(dataset, base, device, spec["model"])
            pred, _, _ = model.predict(positions, spec)
            return pred, None
        artifact = joblib.load(old / "models" / (spec["name"] + ".joblib"))
        old_fingerprint = json.loads((old / "final-manifest.json").read_text())["fingerprint"]
        if artifact["fingerprint"] != old_fingerprint or artifact["feature_names"] != dataset.feature_names:
            raise ValueError("V2 model artifact provenance differs")
        inputs = V2Inputs(dataset, old)
        x, offset = inputs.get(spec, positions)
        sequence = np.asarray(dataset.sequence[positions]) if spec.get("sequence") else None
        return artifact["model"].predict(x, offset, sequence=sequence), None
    if spec.get("policy") == "weekly":
        anchor = utc_boundary(base["calibration_start"], base).timestamp()
        cutoff = float(cutoffs(np.array([dataset.times[position]]), anchor, cfg["weekly"]["update_days"])[0])
        path = output / "weekly-models" / spec["name"] / "final" / f"{int(cutoff)}.joblib"
    else:
        path = output / "models" / (spec["name"] + ".joblib")
    artifact = joblib.load(path)
    if artifact["spec"] != spec:
        raise ValueError("Requested model specification differs from its artifact")
    if spec.get("policy") == "weekly":
        if artifact["training_cutoff"] != cutoff or not artifact["label_upper_bound"] < cutoff <= dataset.times[position]:
            raise ValueError("Weekly model uses an unavailable training state")
    else:
        expected = json.loads((output / "final-manifest.json").read_text())["fingerprint"]
        if artifact["fingerprint"] != expected:
            raise ValueError("V3 model artifact provenance differs")
    if spec["family"] == "finetune":
        model = FineTuner(dataset, base, cfg, spec, device)
        model.restore(artifact["state"])
        pred, curve, _ = model.predict(positions)
        return pred, curve
    model = artifact["model"]
    if hasattr(model, "to"):
        model.to(device)
    inputs = LongInputs(dataset, output)
    x, offset = inputs.get(positions, spec.get("residual", False))
    if spec["family"] == "trajectory":
        curve = model.predict_curve(x, offset, np.asarray(dataset.sequence[positions]))
        return curve[:, [5,11,23,35]], curve
    return model.predict(x, offset), None


def predict_at(dataset, base, cfg, output, at, device="cpu", include_dense=True):
    if base != dataset.metadata["config"]:
        raise ValueError("Base configuration differs from the verified dataset")
    timestamp = pd.Timestamp(at)
    if timestamp.tzinfo is None:
        raise ValueError("An explicit timezone is required")
    manifest = json.loads((output / "final-manifest.json").read_text())["fingerprint"]
    if manifest["data"] != dataset.metadata["fingerprint"] or manifest["settings"] != cfg:
        raise ValueError("Saved v3 policy and dataset/configuration differ")
    for name, expected in manifest["code"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Numerical v3 source changed: {name}")
    with np.load(output / "predictions/test_selected_v3.npz") as saved:
        available_positions = saved["positions"]
    row = np.searchsorted(dataset.times[available_positions], timestamp.timestamp(), side="right") - 1
    if row < 0:
        raise ValueError("Model/calibration were not yet available; use a test-period origin")
    position = int(available_positions[row])
    if timestamp.timestamp() - dataset.times[position] > 20 * 60:
        raise ValueError("No recent saved evaluation origin; this command replays historical ~15-minute origins")
    selection = json.loads((output / "selection-frozen.json").read_text())
    regime = regimes(dataset, np.array([position]), cfg)
    weights = np.asarray(selection["weights"])[regime[0]]
    required = np.flatnonzero(weights.sum(axis=0) > 0)
    matrix = np.zeros((1,4,len(selection["candidates"])), np.float64)
    dense = None
    for index in required:
        spec = selection["candidates"][index]
        pred, curve = member_prediction(dataset, base, cfg, output, device, spec, position)
        matrix[0,:,index] = pred[0]
        if include_dense and spec["name"] == selection["dense_model"]:
            dense = curve
        gc.collect()
    combined = apply_policy(matrix, regime, selection["weights"])[0]
    mode = json.loads((output / "interval-selection-frozen.json").read_text())["mode"]
    with np.load(output / "intervals" / f"test_{mode}.npz") as saved:
        np.testing.assert_array_equal(saved["positions"], available_positions)
        radius = (saved["0.9_upper"][row] - saved["0.9_lower"][row]) / 2
    if include_dense and dense is None:
        spec = next(s for s in selection["candidates"] if s["name"] == selection["dense_model"])
        _, dense = member_prediction(dataset, base, cfg, output, device, spec, position, include_curve=True)
    result = {"origin": pd.Timestamp(float(dataset.times[position]),unit="s",tz="UTC").isoformat(),
              "position":position,"glucose":float(dataset.glucose[position]),"horizons_minutes":base["horizons_minutes"],
              "prediction_mgdl":combined.tolist(),"lower90":(combined-radius).tolist(),"upper90":(combined+radius).tolist(),
              "point_policy":selection["mode"],"interval_policy":mode,
              "interval_note":"Replayed calibration state recorded causally for this historical origin"}
    if include_dense:
        result["dense_model"] = selection["dense_model"]
        result["dense_horizons_minutes"] = list(range(5,181,5))
        result["dense_prediction_mgdl"] = dense[0].tolist()
        result["dense_note"] = "Separate native 36-step model selected on validation; distinct from the four-horizon ensemble"
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at",required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=parser.parse_args()
    dataset,base,cfg,output=context()
    print(json.dumps(predict_at(dataset,base,cfg,output,args.at,args.device),indent=2))


if __name__=="__main__":main()
