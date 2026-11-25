from datetime import datetime, timezone
import tempfile
from pathlib import Path
import unittest

from src.journal import Journal


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.now = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
        self.journal = Journal(Path(self.temp.name) / "events.sqlite", clock=lambda: self.now)

    def tearDown(self):
        self.journal.close()
        self.temp.cleanup()

    def test_plan_and_confirmation_are_separate(self):
        plan = self.journal.add("insulin_plan", 1., "2026-10-23T12:05:00+00:00")
        self.now = datetime(2026, 10, 23, 12, 6, tzinfo=timezone.utc)
        self.journal.add("insulin_confirmed", .8, "2026-10-23T12:05:00+00:00", plan)
        events = self.journal.snapshot()
        self.assertEqual([e["kind"] for e in events], ["insulin_plan", "insulin_confirmed"])
        self.assertEqual(events[1]["amount"], .8)
        self.assertEqual(events[1]["plan_id"], plan)
        self.assertEqual(events[0]["amount"], 1.)

    def test_plan_cannot_be_confirmed_twice(self):
        plan = self.journal.add("insulin_plan", 1., self.now)
        self.journal.add("insulin_confirmed", 1., self.now, plan)
        with self.assertRaises(ValueError):
            self.journal.add("insulin_confirmed", 1., self.now, plan)

    def test_late_report_is_not_available_earlier(self):
        self.journal.add("meal", 40., "2026-10-23T11:30:00+00:00")
        self.assertEqual(self.journal.snapshot("2026-10-23T11:45:00+00:00"), [])
        self.assertEqual(len(self.journal.snapshot()), 1)

    def test_timezone_and_future_observation_validation(self):
        with self.assertRaises(ValueError):
            self.journal.add("iob", 1., "2026-10-23T12:00:00")
        with self.assertRaises(ValueError):
            self.journal.add("glucose", 100., "2026-10-23T12:05:00+00:00")
        with self.assertRaises(ValueError):
            self.journal.add("meal", float("nan"), self.now)

    def payload(self):
        return {"model": "example", "model_revision": "test", "target_times": ["2026-10-23T12:30:00+00:00"],
                "quantile_levels": [.1, .5, .9], "quantiles": [[90., 110., 150.]], "input_event_ids": []}

    def test_forecast_saved_before_target(self):
        forecast_id = self.journal.save_forecast(self.payload())
        self.assertTrue(forecast_id)
        self.now = datetime(2026, 10, 23, 13, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            self.journal.save_forecast(self.payload())

    def test_no_outcomes_unavailable_events_or_crossing_quantiles(self):
        with self.assertRaises(ValueError):
            self.journal.save_forecast(self.payload() | {"actual": 120})
        with self.assertRaises(ValueError):
            self.journal.save_forecast(self.payload() | {"input_event_ids": ["unknown"]})
        with self.assertRaises(ValueError):
            self.journal.save_forecast(self.payload() | {"quantiles": [[120, 100, 90]]})


if __name__ == "__main__":
    unittest.main()
