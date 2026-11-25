"""Refit the frozen v3 policies, archive predictions, then score exploratory test."""

from datetime import datetime, timezone
import gc
import json

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT, sha256
from src.v3.common import (LongInputs, atomic_joblib, atomic_json, load_forecast,
    periods, regimes, save_forecast, score, stage, training_positions, utc_boundary, dense_targets)
from src.v3.ensemble import apply_policy
from src.v3.finetune import FineTuner
from src.v3.rolling import predict_phase
from src.v3.training import fit, predict
from src.v3.uncertainty import scale_inputs, bands, describe


def candidate_forecasts(dataset, base, cfg, output, device, spec, groups, fingerprint):
    name = spec["name"]
    paths = {phase: output / "predictions" / f"{phase}_{name}.npz" for phase in groups}
    if all(path.exists() for path in paths.values()):
        return {phase: load_forecast(paths[phase], pos) for phase, pos in groups.items()}
    if spec["origin"] == "v2":
        old_name = "selected_personal" if name == "v2_selected" else name
        result = {phase: load_forecast(ROOT / cfg["v2"] / "predictions" / f"{phase}_{old_name}.npz", pos) for phase, pos in groups.items()}
        for phase, pos in groups.items():
            save_forecast(paths[phase], pos, result[phase])
        return result
    if spec.get("policy") == "weekly":
        anchor = utc_boundary(base["calibration_start"], base).timestamp()
        result = {phase: predict_phase(dataset, base, cfg, output, device, spec, pos, anchor, phase, "final") for phase, pos in groups.items()}
        for phase, pos in groups.items():
            # Preserve the availability trace alongside the prediction archive.
            source = output / "weekly" / f"{name}_{phase}.npz"
            with np.load(source) as saved:
                save_forecast(paths[phase], pos, result[phase], training_cutoff=saved["training_cutoff"])
        return result
    cutoff = utc_boundary(base["calibration_start"], base).timestamp()
    stride = cfg["finetune"]["train_stride_readings"] if spec["family"] == "finetune" else None
    train = training_positions(dataset, base, cutoff, stride)
    artifact_path = output / "models" / (name + ".joblib")
    epochs = int(round(np.mean(spec["validation_epochs"]))) if spec.get("validation_epochs") else None
    inputs = LongInputs(dataset, output)
    if spec["family"] == "finetune":
        learner = FineTuner(dataset, base, cfg, spec, device)
        if artifact_path.exists():
            artifact = joblib.load(artifact_path)
            if artifact["fingerprint"] != fingerprint:
                raise ValueError("Adapted model provenance differs")
            learner.restore(artifact["state"])
        else:
            equivalence = learner.equivalence(train[:2])
            learner.fit(train, epochs=epochs)
            atomic_joblib(artifact_path, {"spec": spec, "state": learner.state(), "epochs": learner.epochs_used,
                "train_cutoff": cutoff, "training_rows": len(train), "label_upper_bound": float(dataset.times[train[-1]] + 10800 + base["target_tolerance_seconds"]),
                "native_equivalence_max_mgdl": equivalence, "fingerprint": fingerprint})
        result = {}
        for phase, pos in groups.items():
            if paths[phase].exists():
                result[phase] = load_forecast(paths[phase], pos)
            else:
                pred, curve, q = learner.predict(pos)
                save_forecast(paths[phase], pos, pred, curve=curve, quantiles=q)
                result[phase] = pred
        del learner
    else:
        if artifact_path.exists():
            artifact = joblib.load(artifact_path)
            if artifact["fingerprint"] != fingerprint:
                raise ValueError("Personal model provenance differs")
            model = artifact["model"]
            if hasattr(model, "to"):
                model.to(device)
        else:
            model = fit(dataset, train, spec, inputs, cfg, device, epochs=epochs)
            if hasattr(model, "to"):
                model.to("cpu")
            atomic_joblib(artifact_path, {"spec": spec, "model": model, "epochs": model.epochs_used,
                "train_cutoff": cutoff, "training_rows": len(train), "label_upper_bound": float(dataset.times[train[-1]] + 10800 + base["target_tolerance_seconds"]), "fingerprint": fingerprint})
            if hasattr(model, "to"):
                model.to(device)
        result = {}
        for phase, pos in groups.items():
            if paths[phase].exists():
                result[phase] = load_forecast(paths[phase], pos)
            else:
                pred, curve = predict(dataset, pos, spec, inputs, model, curves=True)
                save_forecast(paths[phase], pos, pred, **({"curve": curve} if curve is not None else {}))
                result[phase] = pred
        del model
    gc.collect()
    return result


def summarize(dataset, base, cfg, positions, predictions):
    actual = np.asarray(dataset.targets[positions], np.float64)
    day = pd.to_datetime(dataset.times[positions], unit="s", utc=True).tz_convert(base["timezone"]).date
    days, day_index = np.unique(day, return_inverse=True)
    reference = predictions["v2_selected"]
    group = regimes(dataset, positions, cfg)
    rng = np.random.default_rng(cfg["seed"])
    draws = rng.integers(0, len(days), (cfg["bootstrap_repetitions"], len(days)))
    rows, cohorts = [], []
    for name, pred in predictions.items():
        for k, horizon in enumerate(base["horizons_minutes"]):
            valid = np.isfinite(actual[:, k])
            error = pred[:, k].astype(np.float64) - actual[:, k]
            ae = np.abs(error)
            gain = np.abs(reference[:, k]-actual[:, k])-ae
            counts = np.bincount(day_index[valid], minlength=len(days))
            sums = np.bincount(day_index[valid], weights=ae[valid], minlength=len(days))
            gains = np.bincount(day_index[valid], weights=gain[valid], minlength=len(days))
            denominator = counts[draws].sum(axis=1)
            low, high = np.quantile(sums[draws].sum(axis=1)/denominator, [.025,.975])
            gain_low, gain_high = np.quantile(gains[draws].sum(axis=1)/denominator, [.025,.975])
            rows.append({"model": name, "horizon_minutes": horizon, "n": int(valid.sum()), "mae": float(np.mean(ae[valid])),
                "rmse": float(np.sqrt(np.mean(error[valid]**2))), "bias": float(np.mean(error[valid])),
                "mae_low95": low, "mae_high95": high, "improvement_vs_v2": float(np.mean(gain[valid])),
                "improvement_low95": gain_low, "improvement_high95": gain_high})
            meal = dataset.features[positions,dataset.feature_names.index("carbs_sum_180m")]>0
            for label, condition in [("recent_meal",meal),("no_recent_meal",~meal),("rapid",group==2),("stable",group!=2)]:
                mask = valid & condition
                cohorts.append({"model": name, "horizon_minutes": horizon, "regime": label, "n": int(mask.sum()),
                                "mae": float(np.mean(ae[mask])) if mask.any() else None})
    return pd.DataFrame(rows), pd.DataFrame(cohorts)


def run(dataset, base, cfg, output, device):
    if base != dataset.metadata["config"]:
        raise ValueError("Base configuration differs from the verified dataset")
    sources = ["src/v3/"+p+".py" for p in ["evaluate","models","training","trajectory_network","cache","finetune","rolling","ensemble","uncertainty","predict"]]
    fingerprint = stage(output, "final", dataset, cfg, sources)
    if (output / "completed.json").exists() and (output / "report.md").exists():
        if json.loads((output / "completed.json").read_text())["fingerprint"] != fingerprint:
            raise ValueError("Completed evaluation provenance differs")
        print("Valutazione finale già completa; eseguire verify per controllare gli artefatti", flush=True)
        return
    selection = json.loads((output / "selection-frozen.json").read_text())
    interval_choice = json.loads((output / "interval-selection-frozen.json").read_text())
    proof = output / "selection-before-scoring.json"
    if not proof.exists():
        atomic_json(proof, {"utc": datetime.now(timezone.utc).isoformat(), "point_sha256": sha256(output / "selection-frozen.json"),
            "interval_sha256": sha256(output / "interval-selection-frozen.json"), "test_metrics_exist": (output / "test-metrics.csv").exists()})
    groups = periods(dataset, base)
    predictions = {phase: {} for phase in groups}
    for spec in selection["candidates"]:
        result = candidate_forecasts(dataset, base, cfg, output, device, spec, groups, fingerprint)
        for phase in groups:
            predictions[phase][spec["name"]] = result[phase]
        print(f"Previsioni finali salvate: {spec['name']}", flush=True)
    for phase, pos in groups.items():
        matrix = np.stack([predictions[phase][s["name"]] for s in selection["candidates"]], axis=-1)
        group = regimes(dataset, pos, cfg)
        for mode, weights in selection["policy_weights"].items():
            pred = apply_policy(matrix, group, weights)
            predictions[phase]["ensemble_"+mode] = pred
            save_forecast(output / "predictions" / f"{phase}_ensemble_{mode}.npz", pos, pred)
        chosen = predictions[phase]["ensemble_"+selection["mode"]]
        predictions[phase]["selected_v3"] = chosen
        save_forecast(output / "predictions" / f"{phase}_selected_v3.npz", pos, chosen)

    # Point and interval policies are frozen; outcomes now enter descriptive scoring.
    summary, cohorts = summarize(dataset, base, cfg, groups["test"], predictions["test"])
    summary.to_csv(output / "test-metrics.csv", index=False)
    cohorts.to_csv(output / "test-cohorts.csv", index=False)
    inputs = LongInputs(dataset, output)
    scale_model = joblib.load(output / "interval-scale.joblib")["model"]
    scales = {phase: scale_model.predict(scale_inputs(dataset, pos, predictions[phase]["selected_v3"], inputs)) for phase,pos in groups.items()}
    point = {phase: predictions[phase]["selected_v3"] for phase in groups}
    actual = {phase: np.asarray(dataset.targets[pos]) for phase,pos in groups.items()}
    group = {phase: regimes(dataset,pos,cfg) for phase,pos in groups.items()}
    meal = {phase: (dataset.features[pos,dataset.feature_names.index("carbs_sum_180m")]>0).astype(int) for phase,pos in groups.items()}
    interval_rows = []
    for mode in cfg["intervals"]["modes"]:
        ms = {phase: np.ones_like(point[phase]) if mode=="constant_meal" else scales[phase] for phase in groups}
        mg = meal if mode=="constant_meal" else group
        calibration = {"prediction": point["calibration"], "actual": actual["calibration"], "scale": ms["calibration"], "group": mg["calibration"]}
        ledger = {"prediction": np.concatenate([point[p] for p in groups]), "actual": np.concatenate([actual[p] for p in groups]),
            "scale": np.concatenate([ms[p] for p in groups]), "group": np.concatenate([mg[p] for p in groups]),
            "times": np.concatenate([dataset.times[pos] for pos in groups.values()])}
        interval, availability = bands(mode, point["test"], ms["test"], mg["test"], calibration, cfg, base,
            times=dataset.times[groups["test"]], ledger=ledger, anchor=utc_boundary(base["calibration_start"], base).timestamp())
        statistic = describe(actual["test"], point["test"], interval, group["test"])
        for k, row in enumerate(statistic):
            interval_rows.append({"mode": mode, "selected": mode==interval_choice["mode"], "horizon_minutes": base["horizons_minutes"][k], **row})
        arrays = {f"{key}_{field}": value for key, pair in interval.items() for field,value in pair.items()}
        save_forecast(output / "intervals" / f"test_{mode}.npz", groups["test"], point["test"], availability_upper=availability, **arrays)
    pd.DataFrame(interval_rows).to_csv(output / "test-intervals.csv", index=False)
    dense_rows = []
    dense_actual = dense_targets(dataset, groups["test"], base)
    for spec in selection["candidates"]:
        if spec["origin"]=="v3" and spec["family"] in {"trajectory","finetune"}:
            with np.load(output / "predictions" / f"test_{spec['name']}.npz") as saved:
                errors = np.abs(saved["curve"].astype(np.float64)-dense_actual)
            for k in range(cfg["dense_steps"]):
                dense_rows.append({"model":spec["name"],"horizon_minutes":(k+1)*5,"mae":float(np.nanmean(errors[:,k])),"n":int(np.isfinite(errors[:,k]).sum())})
    pd.DataFrame(dense_rows).to_csv(output / "test-dense-metrics.csv", index=False)
    atomic_json(output / "completed.json", {"fingerprint": fingerprint, "point_mode": selection["mode"], "interval_mode": interval_choice["mode"],
        "dense_model":selection["dense_model"],"calibration_origins":len(groups["calibration"]),"test_origins":len(groups["test"]),
        "note":cfg["evaluation_note"]})
    from src.v3.report import write_report
    write_report(dataset, base, cfg, output)
    print(summary.loc[summary.model.isin(["v2_selected","selected_v3","timestral-28m_physical_672h"])].round(3).to_string(index=False), flush=True)
