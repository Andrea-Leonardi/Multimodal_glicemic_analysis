"""Append-only local event journal; no connection to or control of a pump.

Examples (timestamps MUST include an offset):
python -m src.journal add --kind meal --amount 50 --at 2026-10-23T20:00:00+02:00
python -m src.journal add --kind insulin_plan --amount 1 --at 2026-10-23T20:05:00+02:00
python -m src.journal confirm --plan-id UUID --amount 1 --at 2026-10-23T20:05:00+02:00
python -m src.journal list
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import uuid

from src.data import ROOT

KINDS = {"meal": "g", "insulin_plan": "U", "insulin_confirmed": "U", "iob": "U", "glucose": "mg/dL"}


def timestamp(value):
    d = datetime.fromisoformat(value) if isinstance(value, str) else value
    if d.tzinfo is None or d.utcoffset() is None:
        raise ValueError("Timestamp requires an explicit timezone offset")
    return d.astimezone(timezone.utc).isoformat()


class Journal:
    def __init__(self, path=None, clock=None):
        self.path = Path(path or ROOT / "data/live/events.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, amount REAL NOT NULL,
                unit TEXT NOT NULL, event_time TEXT NOT NULL, available_at TEXT NOT NULL,
                plan_id TEXT REFERENCES events(id), source TEXT NOT NULL,
                UNIQUE(plan_id)
            );
            CREATE INDEX IF NOT EXISTS events_arrival ON events(available_at);
            CREATE UNIQUE INDEX IF NOT EXISTS glucose_sample ON events(source,event_time) WHERE kind='glucose';
            CREATE TABLE IF NOT EXISTS forecasts (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL
            );
        """)

    def close(self):
        self.db.close()

    def add(self, kind, amount, event_time, plan_id=None, source="manual"):
        if kind not in KINDS:
            raise ValueError("Unknown event kind")
        amount = float(amount)
        if not math.isfinite(amount) or amount < 0 or (kind == "glucose" and amount == 0):
            raise ValueError("Amount must be finite and nonnegative; glucose must be positive")
        at, now = timestamp(event_time), timestamp(self.clock())
        if kind != "insulin_plan" and at > now:
            raise ValueError("An observed/confirmed event cannot be in the future")
        if plan_id is not None and kind != "insulin_confirmed":
            raise ValueError("Only insulin confirmations can reference a plan")
        event_id = str(uuid.uuid4())
        with self.db:
            if plan_id is not None:
                plan = self.db.execute("SELECT kind FROM events WHERE id=?", (plan_id,)).fetchone()
                if plan is None or plan["kind"] != "insulin_plan":
                    raise ValueError("Confirmation requires an existing insulin plan")
            try:
                self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
                                (event_id, kind, amount, KINDS[kind], at, now, plan_id, source))
            except sqlite3.IntegrityError as e:
                raise ValueError("This plan is already confirmed") from e
        return event_id

    def ingest_glucose(self, amount, observed_at, source="dexcom_share"):
        at = timestamp(observed_at)
        existing = self.db.execute("SELECT id,amount FROM events WHERE kind='glucose' AND source=? AND event_time=?", (source, at)).fetchone()
        if existing:
            if existing["amount"] != float(amount):
                raise ValueError("Conflicting glucose values at the same source timestamp")
            return None
        return self.add("glucose", amount, at, source=source)

    def snapshot(self, as_of=None):
        at = timestamp(as_of or self.clock())
        return [dict(row) for row in self.db.execute("SELECT * FROM events WHERE available_at<=? ORDER BY available_at,id", (at,))]

    def save_forecast(self, payload):
        """Reserve outcome-free forecast storage for the live collector.

        The forecast must contain future UTC/offset timestamps, an ordered
        quantile vector per step, and the exact input event IDs for audit.
        """
        now = timestamp(self.clock())
        required = {"model", "model_revision", "target_times", "quantile_levels", "quantiles", "input_event_ids"}
        if not required <= payload.keys() or "actual" in payload:
            raise ValueError("Incomplete forecast or outcome included")
        levels = payload["quantile_levels"]
        times, values = payload["target_times"], payload["quantiles"]
        if not levels or any(not math.isfinite(p) or not 0 < p < 1 for p in levels) or any(a >= b for a, b in zip(levels, levels[1:])):
            raise ValueError("Invalid quantile levels")
        if not times or len(times) != len(values):
            raise ValueError("Invalid forecast dimensions")
        normalized = [timestamp(t) for t in times]
        if any(t <= now for t in normalized) or any(a >= b for a, b in zip(normalized, normalized[1:])):
            raise ValueError("Forecast targets must be future and strictly increasing")
        for row in values:
            if len(row) != len(levels) or any(not math.isfinite(v) for v in row) or any(a > b for a, b in zip(row, row[1:])):
                raise ValueError("Invalid quantile values")
        known_ids = {event["id"] for event in self.snapshot(now)}
        if not set(payload["input_event_ids"]) <= known_ids:
            raise ValueError("Forecast references unavailable input events")
        forecast_id = str(uuid.uuid4())
        with self.db:
            self.db.execute("INSERT INTO forecasts VALUES (?,?,?)", (forecast_id, now, json.dumps(payload, allow_nan=False)))
        return forecast_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db")
    commands = p.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("--kind", choices=KINDS, required=True)
    add.add_argument("--amount", type=float, required=True)
    add.add_argument("--at", required=True)
    confirm = commands.add_parser("confirm")
    confirm.add_argument("--plan-id", required=True)
    confirm.add_argument("--amount", type=float, required=True)
    confirm.add_argument("--at", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--as-of")
    a = p.parse_args()
    journal = Journal(a.db)
    try:
        if a.command == "list":
            print(json.dumps(journal.snapshot(a.as_of), indent=2))
        else:
            print(journal.add(a.kind if a.command == "add" else "insulin_confirmed", a.amount, a.at,
                              getattr(a, "plan_id", None)))
    finally:
        journal.close()


if __name__ == "__main__":
    main()
