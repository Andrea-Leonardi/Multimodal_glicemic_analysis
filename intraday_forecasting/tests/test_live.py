from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.journal import Journal
from src.live import live_context, score_due


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)

    def event(self, kind, minutes_ago, amount=100, available=None):
        at = self.now - timedelta(minutes=minutes_ago)
        return {"id": str(minutes_ago) + kind, "kind": kind, "amount": amount, "event_time": at.isoformat(),
                "available_at": (available or at).isoformat()}

    def test_live_grid_uses_latest_reading_and_ignores_future_arrivals(self):
        events = [self.event("glucose", 1, 100), self.event("glucose", 6, 110),
                  self.event("glucose", 0, 999, self.now + timedelta(seconds=1))]
        x, _, ids = live_context(events, self.now, 1)
        self.assertEqual(x[0, -1], 100)
        self.assertEqual(x[0, -2], 110)
        self.assertEqual(len(ids), 2)

    def test_missing_manual_history_not_filled_as_zero(self):
        events = [self.event("meal", 1, 40), self.event("insulin_plan", -5, 1)]
        x, cov, ids = live_context(events, self.now, 1, "manual", (self.now - timedelta(minutes=20)).isoformat())
        self.assertTrue(np.isnan(cov[0, 0, 0]))
        self.assertEqual(cov[0, 0, -1], 40)
        self.assertEqual(cov[0, 1, -1], 0)
        self.assertEqual(ids, [events[0]["id"]])

    def test_plans_and_iob_are_explicitly_separate_from_confirmed_inputs(self):
        events = [self.event("iob", 1, 2), self.event("insulin_plan", 1, 1)]
        _, cov, ids = live_context(events, self.now, 1, "manual", (self.now - timedelta(hours=2)).isoformat())
        self.assertEqual(cov.sum(), 0)
        self.assertEqual(ids, [])

    def test_scoring_waits_for_outcome_and_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "events.sqlite", clock=lambda: self.now)
            targets = [(self.now + timedelta(minutes=5 * i)).isoformat() for i in range(1, 37)]
            forecast_id = journal.save_forecast({"model": "fixture", "model_revision": "fixture", "target_times": targets,
                                               "quantile_levels": [.1, .5, .9], "quantiles": [[90, 100, 110]] * 36, "input_event_ids": []})
            self.assertEqual(score_due(journal), 0)
            self.now += timedelta(minutes=31)
            journal.ingest_glucose(120, self.now - timedelta(minutes=1))
            self.assertEqual(score_due(journal), 0)  # 15-minute arrival grace
            self.now += timedelta(minutes=15)
            self.assertEqual(score_due(journal), 1)
            self.assertEqual(score_due(journal), 0)
            scored = journal.db.execute("SELECT * FROM forecast_scores").fetchone()
            self.assertEqual(scored["forecast_id"], forecast_id)
            self.assertEqual(scored["step"], 6)
            self.assertEqual(scored["absolute_error"], 20)
            journal.close()


if __name__ == "__main__":
    unittest.main()
