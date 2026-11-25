"""Resumable 28-day physical Timestral features; reuse complete v2 forecasts."""

import gc
import json
import time

import numpy as np

from src.data import ROOT, sha256
from src.personal_timestral import Predictor
from src.v3.common import all_positions, atomic_array, atomic_json, stage

SPEC = {"model": "timestral-28m", "mode": "physical", "context_hours": 672}


def predictor(dataset, base, device, model="timestral-28m"):
    result = Predictor(dataset, base, model)
    if device == "cuda":
        result.adapter.forecaster.model.to("cuda")
        result.adapter.forecaster.model.transformer.rope.rebuild_inv_freq()
    return result


def run(dataset, base, cfg, output, device):
    fingerprint = stage(output, "cache", dataset, cfg, ["src/v3/cache.py", "src/personal_timestral.py", "src/forecasting.py"])
    folder = output / "long-cache"
    folder.mkdir(exist_ok=True)
    positions = all_positions(dataset, base)
    if (folder / "complete.json").exists():
        if json.loads((folder / "complete.json").read_text())["fingerprint"] != fingerprint:
            raise ValueError("Long-cache fingerprint differs")
        print("Long-context cache already complete", flush=True)
        return
    atomic_array(folder / "positions.npy", positions)
    exists = (folder / "prediction.npy").exists() and (folder / "quantiles.npy").exists()
    prediction = np.lib.format.open_memmap(folder / "prediction.npy", mode="r+" if exists else "w+", dtype=np.float32, shape=(len(positions), 4))
    quantiles = np.lib.format.open_memmap(folder / "quantiles.npy", mode="r+" if exists else "w+", dtype=np.float32, shape=(len(positions), 4, 21))
    done = np.load(folder / "done.npy") if exists and (folder / "done.npy").exists() else np.zeros(len(positions), bool)
    assert prediction.shape == (len(positions), 4) and quantiles.shape == (len(positions), 4, 21) and done.shape == (len(positions),)
    reused = []
    source = ROOT / cfg["v2"]
    for path in [source / "timestral/dense_timestral-28m_physical_672h.npz", *[source / f"predictions/{p}_timestral-28m_physical_672h.npz" for p in ["calibration", "test"]]]:
        with np.load(path) as saved:
            ix = np.searchsorted(positions, saved["positions"])
            np.testing.assert_array_equal(positions[ix], saved["positions"])
            prediction[ix], quantiles[ix], done[ix] = saved["prediction"], saved["quantiles"], True
        reused.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    prediction.flush()
    quantiles.flush()
    atomic_array(folder / "done.npy", done)
    pending = np.flatnonzero(~done)
    print(f"Long cache: {done.sum()}/{len(done)} riusate; {len(pending)} da calcolare", flush=True)
    if len(pending):
        model = predictor(dataset, base, device)
        reused_rows = np.flatnonzero(done)[:4]
        check_reuse = model.predict(positions[reused_rows], SPEC)
        difference = float(np.max(np.abs(check_reuse[0] - prediction[reused_rows])))
        if difference > .01:
            raise ValueError(f"GPU/v2 cache equivalence failed: {difference} mg/dL")
        atomic_json(folder / "reuse-equivalence.json", {"origins": positions[reused_rows].tolist(), "max_difference_mgdl": difference})
        first = pending[:4]
        check = model.predict(positions[first], SPEC)
        assert np.isfinite(check[0]).all()
        started = time.perf_counter()
        batch = cfg["prediction_batch_size"]
        for start in range(0, len(pending), batch):
            ix = pending[start:start + batch]
            pred, q, _ = model.predict(positions[ix], SPEC)
            prediction[ix], quantiles[ix], done[ix] = pred, q, True
            if (start // batch + 1) % 100 == 0 or start + batch >= len(pending):
                prediction.flush()
                quantiles.flush()
                atomic_array(folder / "done.npy", done)
                print(f"  Long cache: {done.sum()}/{len(done)} ({time.perf_counter()-started:.1f}s)", flush=True)
        del model
        gc.collect()
    assert done.all() and np.isfinite(prediction).all() and np.isfinite(quantiles).all()
    atomic_json(folder / "complete.json", {"fingerprint": fingerprint, "reused": reused, "positions": len(positions), "spec": SPEC})
