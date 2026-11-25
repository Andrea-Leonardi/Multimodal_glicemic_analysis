"""Model adapters: only historical context is passed to forecasters."""

from __future__ import annotations

import numpy as np

QUANTILES = np.array([.01, .05, .1, .15, .2, .25, .3, .35, .4, .45,
                      .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .99])
FEATURES = {
    "glucose": [],
    "manual": ["meal_carbs_g", "manual_bolus_confirmed_u"],
    "pump": ["meal_carbs_g", "manual_bolus_confirmed_u", "automatic_bolus_confirmed_u", "basal_recorded_u"],
}


def baseline(context, length, name):
    """Trend from last 15 min, exponentially damped with fixed 30 min time scale.

    Parameters fixed before validation; no physiological/therapy constraints.
    """
    x = np.asarray(context)
    last = x[:, -1]
    if not np.isfinite(last).all():
        raise ValueError("Baselines require an observed origin value")
    median = np.repeat(last[:, None], length, axis=1)
    if name == "damped_trend":
        slope = (last - x[:, -4]) / 15
        minutes = np.arange(1, length + 1) * 5
        median += slope[:, None] * 30 * (1 - np.exp(-minutes[None, :] / 30))
    elif name != "persistence":
        raise ValueError(name)
    return np.repeat(median[..., None], len(QUANTILES), axis=-1)


class TimestralAdapter:
    def __init__(self, name, revision, cache, threads=4, offline=True):
        import torch
        from timestral import TimestralForecaster
        torch.set_num_threads(threads)
        self.forecaster = TimestralForecaster.from_pretrained(
            "mistralai/" + {"timestral-28m": "Timestral-28M", "timestral-209m": "Timestral-209M"}[name],
            revision=revision, cache_dir=str(cache), local_files_only=offline, device="cpu",
        )
        self.forecaster.model.eval()
        self.levels = np.array(self.forecaster.quantile_levels)
        if not np.allclose(self.levels, QUANTILES):
            raise ValueError("Unexpected Timestral quantile levels")

    def predict(self, context, length, covariates=None):
        result = self.forecaster.forecast(
            np.ascontiguousarray(context[:, None, :], dtype=np.float32),
            prediction_length=length,
            past_covariates=None if covariates is None else np.ascontiguousarray(covariates, dtype=np.float32),
        )
        values = result.quantiles[:, 0].cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite Timestral forecasts")
        return values


def finite_sample_radius(scores, coverage):
    x = np.sort(np.asarray(scores)[np.isfinite(scores)])
    if not len(x):
        raise ValueError("No calibration observations")
    rank = int(np.ceil((len(x) + 1) * coverage))
    if rank > len(x):
        raise ValueError("Not enough calibration observations for requested coverage")
    return float(x[rank - 1])


def calibrate(prediction, target, probabilistic):
    """Widen-only CQR for Timestral; absolute residuals for point baselines.

    Dependence and distribution shift mean coverage is empirical, not a
    finite-sample guarantee for this glucose series.
    """
    out = {}
    for coverage, low, high in [(.8, .1, .9), (.9, .05, .95)]:
        lower = prediction[..., np.flatnonzero(np.isclose(QUANTILES, low))[0]]
        upper = prediction[..., np.flatnonzero(np.isclose(QUANTILES, high))[0]]
        scores = np.maximum(np.maximum(lower - target, target - upper), 0)
        if not probabilistic:
            scores = np.abs(prediction[..., 10] - target)
        out[str(coverage)] = {
            "radius": [finite_sample_radius(scores[:, k], coverage) for k in range(target.shape[1])],
            "counts": np.isfinite(scores).sum(axis=0).tolist(),
        }
    return out


def intervals(prediction, calibration=None):
    out = {}
    for coverage, low, high in [(.8, .1, .9), (.9, .05, .95)]:
        lo = prediction[..., np.flatnonzero(np.isclose(QUANTILES, low))[0]].copy()
        hi = prediction[..., np.flatnonzero(np.isclose(QUANTILES, high))[0]].copy()
        if calibration is not None:
            radius = np.asarray(calibration[str(coverage)]["radius"])
            lo -= radius
            hi += radius
        out[coverage] = (lo, hi)
    return out
