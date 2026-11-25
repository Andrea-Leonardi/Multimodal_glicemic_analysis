"""Audit completed artifacts and replay saved models without future outcomes.

Run AFTER the final stage; this never fits/selects models or changes predictions.
Use the training memory profile because CPU torch and CatBoost load together.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.data import sha256
from src.personal_data import configuration, split_positions
from src.personal_experiment import load_dataset, metrics, utc_boundary
from src.personal_evaluate import apply_intervals, recent_meal
from src.personal_predict import predict_at


def verify(output, dataset_folder):
    started = time.perf_counter()
    output = Path(output)
    done = json.loads((output / "completed.json").read_text())
    manifest = json.loads((output / "final-manifest.json").read_text())
    config = configuration()
    dataset = load_dataset(dataset_folder)
    fingerprint = manifest["fingerprint"]
    assert fingerprint == done["fingerprint"]
    assert fingerprint["config"] == config
    assert fingerprint["dataset"] == dataset.metadata["fingerprint"]
    for name, expected in fingerprint["code"].items():
        assert sha256(ROOT / name) == expected, f"Source differs: {name}"
    for path in output.glob("*-manifest.json"):
        stage = json.loads(path.read_text())
        for name, expected in stage["fingerprint"].get("code", {}).items():
            assert sha256(ROOT / name) == expected, f"Stage source differs: {path.name}/{name}"

    selection = json.loads((output / "selection-frozen.json").read_text())
    proof_path = output / "selection-before-scoring.json"
    proof = json.loads(proof_path.read_text()) if proof_path.exists() else None
    if proof is not None:
        assert proof["selection_sha256"] == sha256(output / "selection-frozen.json")
        assert proof["test_metrics_exist"] is False
    calibration_start, test_start, test_end = [utc_boundary(config[k], config) for k in ["calibration_start", "test_start", "test_end"]]
    positions = {"calibration": split_positions(dataset, calibration_start, test_start, config["evaluation_stride_readings"], training=True),
                 "test": split_positions(dataset, test_start, test_end, config["evaluation_stride_readings"], training=True)}
    assert len(positions["test"]) == done["test_origins"]
    assert len(positions["calibration"]) == done["calibration_origins"]
    names = [c["spec"]["name"] for c in selection["selected"]] + ["persistence", "damped_trend", "selected_personal"]
    assert len(names) == len(set(names))
    saved = {"calibration": {}, "test": {}}
    artifacts = {"selection-frozen.json": sha256(output / "selection-frozen.json")}
    if proof is not None:
        artifacts[proof_path.name] = sha256(proof_path)
    for stage, pos in positions.items():
        for name in names:
            path = output / "predictions" / f"{stage}_{name}.npz"
            with np.load(path) as content:
                np.testing.assert_array_equal(content["positions"], pos)
                pred = content["prediction"].copy()
            assert pred.shape == (len(pos), len(config["horizons_minutes"]))
            assert np.isfinite(pred).all(), f"Nonfinite predictions: {path}"
            saved[stage][name] = pred
            artifacts[str(path.relative_to(output))] = sha256(path)
        combined = np.empty_like(saved[stage]["selected_personal"])
        for k, ensemble in enumerate(selection["ensemble"]):
            weights = np.asarray(ensemble["weights"])
            np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
            assert (weights >= 0).all()
            combined[:, k] = sum(w * saved[stage][n][:, k] for n, w in zip(ensemble["models"], ensemble["weights"]))
        np.testing.assert_array_equal(combined, saved[stage]["selected_personal"])

    model_count = 0
    for candidate in selection["selected"]:
        spec = candidate["spec"]
        if spec["family"] == "timestral":
            continue
        path = output / "models" / (spec["name"] + ".joblib")
        artifact = joblib.load(path)
        assert artifact["fingerprint"] == fingerprint
        assert artifact["spec"] == spec
        assert artifact["feature_names"] == dataset.feature_names
        assert artifact["training_last_origin_unix_s"] + max(config["horizons_minutes"]) * 60 + config["target_tolerance_seconds"] < calibration_start.timestamp()
        assert artifact["training_rows"] == done["training_rows_per_model"][spec["name"]]
        artifacts[str(path.relative_to(output))] = sha256(path)
        del artifact
        model_count += 1

    summary = pd.read_csv(output / "test-metrics.csv")
    actual = np.asarray(dataset.targets[positions["test"]])
    meal = recent_meal(dataset, positions["test"])
    maximum_metric_difference = {"mae": 0., "rmse": 0.}
    for name in names:
        # Use an independent float64 reduction: summing a strided float32
        # N-by-4 array accumulates more roundoff than pandas' 1-D reductions.
        score = metrics(actual.astype(np.float64), saved["test"][name].astype(np.float64))
        rows = summary.loc[summary.model.eq(name)].set_index("horizon_minutes").loc[config["horizons_minutes"]]
        np.testing.assert_allclose(rows.mae, score["mae"], atol=1e-4)
        np.testing.assert_allclose(rows.rmse, score["rmse"], atol=1e-4)
        for key in maximum_metric_difference:
            maximum_metric_difference[key] = max(maximum_metric_difference[key], float(np.max(np.abs(rows[key].to_numpy() - score[key]))))
        np.testing.assert_array_equal(rows.n, score["n"])
        path = output / "predictions" / f"intervals_{name}.json"
        calibrator = json.loads(path.read_text())
        artifacts[str(path.relative_to(output))] = sha256(path)
        for level in [.8, .9]:
            lo, hi = apply_intervals(saved["test"][name], meal, calibrator, level)
            for k in range(actual.shape[1]):
                valid = np.isfinite(actual[:, k])
                coverage = np.mean((actual[valid, k] >= lo[valid, k]) & (actual[valid, k] <= hi[valid, k]))
                np.testing.assert_allclose(coverage, rows[f"coverage{int(level * 100)}"].iloc[k], atol=1e-12)

    replay = []
    without_targets = replace(dataset, targets=None, target_indices=None)
    pos = positions["test"]
    for row in [0, len(pos) // 2, len(pos) - 1]:
        at = pd.Timestamp(float(dataset.times[pos[row]]), unit="s", tz="UTC").isoformat()
        ordinary = predict_at(dataset, config, output, at)
        blind = predict_at(without_targets, config, output, at)
        assert ordinary == blind, "Inference changed after removing future outcome arrays"
        difference = float(np.max(np.abs(np.asarray(blind["prediction_mgdl"]) - saved["test"]["selected_personal"][row])))
        assert difference <= .01, f"Replay differs from cached CPU/CUDA forecast: {difference} mg/dL"
        replay.append({"position": int(pos[row]), "max_difference_vs_saved_mgdl": difference, "targets_removed_identical": True, "forecast": blind})
        print(f"Replay {at}: target rimossi identico, differenza cache {difference:.6f} mg/dL", flush=True)

    required = ["report.md", "comparison.png", "comparison.pdf", "regimes.png", "first-origin.png", "test-predictions.csv.gz", "test-metrics.csv", "test-cohorts.csv", "validation-metrics.csv", "timestral-context-search.csv", "uncertainty-coverage.png", "uncertainty-coverage.pdf", "test-examples.png", "test-examples.pdf", "test-day.png", "test-day.pdf", "test-examples.json"]
    for name in required:
        path = output / name
        assert path.is_file() and path.stat().st_size > 0, f"Missing output: {name}"
        artifacts[name] = sha256(path)
    examples = json.loads((output / "test-examples.json").read_text())
    assert examples["prediction_sha256"] == artifacts["predictions/test_selected_personal.npz"]
    assert examples["intervals_sha256"] == artifacts["predictions/intervals_selected_personal.json"]
    assert examples["plot_code_sha256"] == sha256(ROOT / "src/personal_test_plots.py")
    result = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(), "audit_code_sha256": sha256(Path(__file__)),
              "report_code_sha256": sha256(ROOT / "src/personal_report.py"), "test_plot_code_sha256": sha256(ROOT / "src/personal_test_plots.py"),
              "seconds": time.perf_counter() - started, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "prediction_models_including_references": len(names), "fitted_model_artifacts": model_count,
              "calibration_origins": len(positions["calibration"]), "test_origins": len(pos),
              "selection_snapshot_before_scoring": proof,
              "float64_metric_reference_max_difference_mgdl": maximum_metric_difference,
              "checks": ["source/data provenance", "shared origins and finite forecasts", "frozen ensemble reconstruction", "training label purge", "recomputed test errors and coverage", "CPU replay without outcome arrays", "report and figures"],
              "replay": replay, "artifact_sha256": artifacts}
    (output / "verified.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ["artifact_sha256", "replay"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "outputs/personal-v2"))
    parser.add_argument("--dataset", default=str(ROOT / "outputs/memory-check/dataset"))
    args = parser.parse_args()
    verify(args.output, args.dataset)
