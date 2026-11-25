"""Local journal -> forecast archive + latest JSON/PNG, without pump control.

Run the Dexcom collector separately. This worker refreshes on new readings or
manual events. IOB and unconfirmed plans are recorded but not conditioned on
by this first benchmarked adapter; see tasks.md for those separate experiments.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data import ROOT, read_config
from src.forecasting import FEATURES, QUANTILES, TimestralAdapter, intervals
from src.journal import Journal, timestamp


def live_context(events, now, hours, mode="glucose", logging_since=None):
    """Right-closed 5-min bins anchored at the ACTUAL issue time.

    Snapshot arrival is checked independently of occurrence time. Regridding
    phase differs from the fixed-hour historical benchmark and needs live
    evaluation; no reported offline metric is claimed to validate this phase.
    """
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("Live issue time requires a timezone")
    now = now.tz_convert("UTC")
    length = hours * 12
    x = np.full((1, length), np.nan, dtype=np.float32)
    used = []
    cov = np.full((1, 2, length), np.nan, dtype=np.float32) if mode == "manual" else None
    if mode not in {"glucose", "manual"}:
        raise ValueError("Live worker supports glucose/manual, not historical pump inputs")
    if mode == "manual":
        if logging_since is None:
            raise ValueError("Manual mode requires --manual-logging-since, marking the start of complete event logging")
        since = pd.Timestamp(timestamp(logging_since))
        grid_times = now - pd.to_timedelta(np.arange(length - 1, -1, -1) * 5, unit="min")
        # A whole bin must fall after the explicit complete-logging start.
        cov[:, :, grid_times - pd.Timedelta(minutes=5) >= since] = 0
    for event in sorted(events, key=lambda e: (e["event_time"], e["available_at"])):
        arrival, occurrence = pd.Timestamp(event["available_at"]), pd.Timestamp(event["event_time"])
        if arrival > now or occurrence > now:
            continue
        if event["kind"] == "glucose":
            steps = int((now - occurrence).total_seconds() // 300)
            if 0 <= steps < length:
                x[0, length - 1 - steps] = event["amount"]
                used.append(event["id"])
        elif mode == "manual" and event["kind"] in {"meal", "insulin_confirmed"}:
            # Same conservative availability-time pulse as historical adapter.
            steps = int((now - arrival).total_seconds() // 300)
            channel = 0 if event["kind"] == "meal" else 1
            if 0 <= steps < length and np.isfinite(cov[0, channel, length - 1 - steps]):
                cov[0, channel, length - 1 - steps] += event["amount"]
                used.append(event["id"])
    return x, cov, sorted(set(used))


def score_due(journal, grace_minutes=15):
    """Resolve future outcomes only after their target + arrival grace time."""
    journal.db.execute("""CREATE TABLE IF NOT EXISTS forecast_scores (
        forecast_id TEXT NOT NULL REFERENCES forecasts(id), step INTEGER NOT NULL,
        target_event_id TEXT NOT NULL REFERENCES events(id), scored_at TEXT NOT NULL,
        actual REAL NOT NULL, median REAL NOT NULL, absolute_error REAL NOT NULL,
        PRIMARY KEY(forecast_id,step))""")
    now = pd.Timestamp(journal.clock())
    inserted = 0
    for forecast in journal.db.execute("SELECT * FROM forecasts").fetchall():
        payload = json.loads(forecast["payload"])
        median_index = payload["quantile_levels"].index(.5)
        for k in [5, 11, 23, 35]:
            if k >= len(payload["target_times"]):
                continue
            target = pd.Timestamp(payload["target_times"][k])
            if target + pd.Timedelta(minutes=grace_minutes) > now:
                continue
            event = journal.db.execute("""SELECT * FROM events WHERE kind='glucose'
                AND event_time>? AND event_time<=? AND available_at<=?
                ORDER BY event_time DESC LIMIT 1""",
                (timestamp((target - pd.Timedelta(minutes=5)).to_pydatetime()), timestamp(target.to_pydatetime()), timestamp(journal.clock()))).fetchone()
            if event is None:
                continue
            median = payload["quantiles"][k][median_index]
            with journal.db:
                cursor = journal.db.execute("INSERT OR IGNORE INTO forecast_scores VALUES (?,?,?,?,?,?,?)",
                    (forecast["id"], k + 1, event["id"], timestamp(journal.clock()), event["amount"], median, abs(event["amount"] - median)))
            inserted += cursor.rowcount
    return inserted


def write_latest(payload, path):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / "latest.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path / "latest.json")
    curve = np.asarray(payload["quantiles"])
    times = pd.to_datetime(payload["target_times"], utc=True).tz_convert("Europe/Rome")
    bands = payload["interval90"]
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(times, curve[:, 10], label="Mediana prevista")
    ax.fill_between(times, bands["lower"], bands["upper"], alpha=.2, label="Fascia 90% calibrata sullo storico")
    ax.set(ylabel="mg/dL", title="Previsione sperimentale — calibrazione dal vivo da verificare")
    ax.legend()
    ax.grid(alpha=.2)
    fig.autofmt_xdate()
    fig.savefig(path / "latest.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db")
    parser.add_argument("--experiment", default=str(ROOT / "outputs/experiment-v1"))
    parser.add_argument("--output", default=str(ROOT / "outputs/live"))
    parser.add_argument("--model", choices=["timestral-28m", "timestral-209m"], default="timestral-28m")
    parser.add_argument("--mode", choices=["glucose", "manual"], default="glucose")
    parser.add_argument("--manual-logging-since")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.mode == "manual" and not args.manual_logging_since:
        parser.error("--mode manual requires --manual-logging-since with timezone")
    config = read_config()
    experiment = Path(args.experiment)
    manifest = json.loads((experiment / "manifest.json").read_text())
    if manifest["fingerprint"]["config"]["model_revisions"] != config["model_revisions"]:
        parser.error("Live model revisions differ from the calibrated experiment")
    selected = json.loads((experiment / "selection.json").read_text())["selected"]
    choice = next(c for c in selected if c["model"] == args.model and c["mode"] == args.mode)
    calibration = json.loads((experiment / f"calibration_{args.model}_{args.mode}.json").read_text())
    adapter = TimestralAdapter(args.model, config["model_revisions"][args.model], ROOT / ".cache/huggingface", config["torch_threads"])
    journal = Journal(args.db)
    last_signature = None
    try:
        while True:
            snapshot = journal.snapshot()
            signature = tuple(e["id"] for e in snapshot)
            if signature != last_signature:
                now = journal.clock()
                x, cov, used = live_context(snapshot, now, choice["context_hours"], args.mode, args.manual_logging_since)
                if not np.isfinite(x[0, -1]) or np.isnan(x).mean() > config["max_missing_context_fraction"]:
                    print(json.dumps({"status": "waiting_for_context", "context_hours": choice["context_hours"], "missing_fraction": float(np.isnan(x).mean())}), flush=True)
                else:
                    prediction = adapter.predict(x, 36, cov)
                    bands = intervals(prediction, calibration)
                    payload = {"model": args.model, "model_revision": config["model_revisions"][args.model],
                        "issue_time": timestamp(now), "mode": args.mode, "context_hours": choice["context_hours"],
                        "target_times": [timestamp(now + timedelta(minutes=5 * i)) for i in range(1, 37)],
                        "quantile_levels": QUANTILES.tolist(), "quantiles": prediction[0].tolist(), "input_event_ids": used,
                        "interval80": {"lower": bands[.8][0][0].tolist(), "upper": bands[.8][1][0].tolist()},
                        "interval90": {"lower": bands[.9][0][0].tolist(), "upper": bands[.9][1][0].tolist()},
                        "not_conditioned_on": ["manual_iob", "unconfirmed_insulin_plans"],
                        "validation_note": "Live delay, grid phase and manual-logging quality require prospective evaluation."}
                    # The SQLite archive is durable BEFORE generating displays.
                    forecast_id = journal.save_forecast(payload)
                    write_latest(payload | {"forecast_id": forecast_id}, args.output)
                    print(json.dumps({"forecast_id": forecast_id, "issue_time": timestamp(now)}), flush=True)
                last_signature = signature
            score_due(journal)
            if not args.watch:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        journal.close()


if __name__ == "__main__":
    main()
