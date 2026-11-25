"""Behavioral checks for causal data handling and honest evaluation."""

import unittest

import numpy as np
import pandas as pd

from src.benchmark import historical_context, origins
from src.data import ROOT, bolus_events, cgm_grid, classify_boluses, event_features, read_config, utc
from src.forecasting import baseline, calibrate, finite_sample_radius, intervals


def raw_bolus(**overrides):
    row = {
        "Data/Ora di completamento": "2026-01-01T03:00:00",
        "Orario/Data d'inizio del bolo esteso": None,
        "Completamento del bolo esteso": None,
        "Tipo di bolo": "Alimento/Correzione", "Modalità di erogazione del bolo": "Standard",
        "Quantità Carboidrati": 30, "Glicemia (mg/dL)": 290,
        "Insulina erogata": 1., "Insulina erogata con bolo esteso": None,
        "Durata (min)": 0, "Percentuale standard": 100,
        "Descrizione dello stato di completamento": "Completato", "Stato di completamento del bolo esteso": None,
    }
    row.update(overrides)
    return pd.DataFrame([row])


class PersonalRulesTests(unittest.TestCase):
    def test_fake_carbs_do_not_become_food(self):
        events = bolus_events(raw_bolus(), "Europe/Rome")
        self.assertEqual(events.carbs_recorded_g.iloc[0], 30)
        self.assertEqual(events.carbs_meal_inferred_g.iloc[0], 0)
        self.assertEqual(events.insulin_delivered_u.iloc[0], 1)

    def test_manual_zero_is_not_automatic(self):
        self.assertEqual(classify_boluses(raw_bolus(**{"Quantità Carboidrati": 0})).iloc[0], "Correzione")
        auto = raw_bolus(**{"Quantità Carboidrati": 0, "Modalità di erogazione del bolo": "Automatico"})
        self.assertEqual(classify_boluses(auto).iloc[0], "Automatico")

    def test_meal_rule_has_priority(self):
        row = raw_bolus(**{"Glicemia (mg/dL)": 170})
        self.assertEqual(classify_boluses(row).iloc[0], "Alimenti")

    def test_rules_match_existing_personal_labels(self):
        raw = pd.read_csv(ROOT / "data/tandemsource_data/merged/merged-Bolo.csv")
        raw = raw.loc[raw["Descrizione dello stato di completamento"].eq("Completato")].reset_index(drop=True)
        cleaned = pd.read_csv(ROOT / "data/cleaned/cleaned-Bolo.csv")
        expected = cleaned["Tipo di bolo"].copy()
        manual_zero = cleaned["Modalità di erogazione del bolo"].ne("Automatico") & cleaned["Quantità Carboidrati"].eq(0)
        expected.loc[manual_zero] = "Correzione"
        np.testing.assert_array_equal(classify_boluses(raw), expected)

    def test_extended_and_interrupted_delivery_preserved_without_early_total(self):
        row = raw_bolus(**{"Orario/Data d'inizio del bolo esteso": "2026-01-01T01:00:00",
                           "Completamento del bolo esteso": "2026-01-01T03:00:00",
                           "Modalità di erogazione del bolo": "Prolungato",
                           "Descrizione dello stato di completamento": "Arrestato dall'utente",
                           "Insulina erogata con bolo esteso": .6, "Durata (min)": 120})
        event = bolus_events(row, "Europe/Rome").iloc[0]
        self.assertEqual(event.insulin_delivered_u, 1.)
        self.assertEqual(event.extended_delivered_u, .6)
        self.assertEqual((event.available_at - event.event_time).total_seconds(), 7200)
        self.assertEqual(event.completion_status, "Arrestato dall'utente")


class CausalityTests(unittest.TestCase):
    def test_dst_is_not_an_hour_of_missing_data(self):
        dates = utc(pd.Series(["2026-03-29 01:55:00", "2026-03-29 03:00:00"]))
        self.assertEqual((dates.iloc[1] - dates.iloc[0]).total_seconds(), 300)

    def test_grid_never_uses_future_sample_or_interpolates(self):
        data = pd.DataFrame({"observed_at": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:15:00Z"]), "glucose": [100, 200, 250]})
        grid = cgm_grid(data)
        self.assertEqual(grid.glucose.iloc[0], 100)
        self.assertEqual(grid.glucose.iloc[1], 200)
        self.assertTrue(np.isnan(grid.glucose.iloc[2]))
        self.assertTrue((grid.observed_at.dropna().array <= grid.observed_at.dropna().index.array).all())

    def test_final_bolus_not_visible_before_completion(self):
        event = bolus_events(raw_bolus(**{"Orario/Data d'inizio del bolo esteso": "2026-01-01T01:00:00", "Modalità di erogazione del bolo": "Prolungato"}), "Europe/Rome")
        index = pd.date_range("2026-01-01T00:00:00Z", "2026-01-01T03:00:00Z", freq="5min")
        basal = pd.DataFrame({"observed_at": index, "basal_u": np.zeros(len(index))})
        features = event_features(event, basal, index, read_config())
        self.assertEqual(features.manual_bolus_confirmed_u.loc[:"2026-01-01T01:55:00Z"].sum(), 0)
        self.assertEqual(features.manual_bolus_confirmed_u.sum(), 1)

    def test_pump_missing_is_not_zero(self):
        event = bolus_events(raw_bolus(), "Europe/Rome")
        index = pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="5min")
        basal = pd.DataFrame({"observed_at": index[:1], "basal_u": [0.]})
        result = event_features(event, basal, index, read_config())
        self.assertEqual(result.basal_recorded_u.iloc[0], 0)
        self.assertTrue(np.isnan(result.basal_recorded_u.iloc[1]))

    def test_changing_future_never_changes_historical_inputs(self):
        index = pd.date_range("2026-01-01", periods=100, freq="5min", tz="UTC")
        grid = pd.DataFrame({"glucose": np.arange(100.), "observed_at": index,
                             "meal_carbs_g": 0., "manual_bolus_confirmed_u": 0.}, index=index)
        config = read_config()
        before = historical_context(grid, np.array([50]), 36, config, "manual")
        grid.loc[grid.index[51:], ["glucose", "meal_carbs_g", "manual_bolus_confirmed_u"]] = 999
        after = historical_context(grid, np.array([50]), 36, config, "manual")
        for x, y in zip(before, after):
            np.testing.assert_array_equal(x, y)

    def test_split_keeps_entire_horizon_and_ignores_missing_future_targets(self):
        index = pd.date_range("2026-01-01", periods=120, freq="5min", tz="UTC")
        grid = pd.DataFrame({"glucose": 100.}, index=index)
        config = read_config() | {"context_hours": [1], "origin_stride_minutes": 5}
        end = index[-1]
        first = origins(grid, index[0], end, config)
        self.assertTrue((index[first] + pd.Timedelta(minutes=180) < end).all())
        grid.loc[index[-10]:, "glucose"] = np.nan
        np.testing.assert_array_equal(first, origins(grid, index[0], end, config))

    def test_simulated_arrival_masks_unavailable_cgm(self):
        index = pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC")
        grid = pd.DataFrame({"glucose": 100., "observed_at": index}, index=index)
        config = read_config() | {"cgm_latency_minutes": 5}
        x, _ = historical_context(grid, np.array([10]), 5, config, "glucose")
        self.assertTrue(np.isnan(x[0, -1]))
        self.assertEqual(x[0, -2], 100)


class EvaluationTests(unittest.TestCase):
    def test_finite_sample_order_statistic(self):
        self.assertEqual(finite_sample_radius(np.arange(1, 10), .9), 9)
        with self.assertRaises(ValueError):
            finite_sample_radius([1], .9)

    def test_calibration_does_not_change_median(self):
        prediction = np.full((20, 36, 21), 100.)
        target = np.full((20, 36), 110.)
        calibration = calibrate(prediction, target, False)
        lo, hi = intervals(prediction, calibration)[.9]
        self.assertTrue(np.all(lo == 90))
        self.assertTrue(np.all(hi == 110))
        self.assertTrue(np.all(prediction[..., 10] == 100))

    def test_trend_is_damped_and_persistence_is_constant(self):
        context = np.array([[85., 90., 95., 100.]])
        flat = baseline(context, 36, "persistence")
        trend = baseline(context, 36, "damped_trend")
        self.assertTrue((flat == 100).all())
        self.assertGreater(trend[0, -1, 10], 120)
        self.assertLess(trend[0, -1, 10], 130)


if __name__ == "__main__":
    unittest.main()
