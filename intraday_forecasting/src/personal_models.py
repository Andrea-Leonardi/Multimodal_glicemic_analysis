"""Supervised personal predictors. Fitted transforms see training rows only."""

from __future__ import annotations

import copy
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


def feature_columns(names, mode="all"):
    if mode == "all":
        return np.arange(len(names))
    if mode == "glucose":
        return np.array([i for i, n in enumerate(names) if n.startswith(("glucose", "clock", "weekday"))])
    if mode == "raw":
        return np.array([i for i, n in enumerate(names) if not n.startswith(("iob_", "activity_", "past_insulin_effect_", "carb_remaining_", "carb_activity_", "past_carb_effect_"))])
    if mode == "humalog360":
        return np.array([i for i, n in enumerate(names) if not n.startswith(("iob_", "activity_", "past_insulin_effect_")) or "d360_p75" in n])
    raise ValueError(mode)


def candidate_specs():
    specs = [{"name": f"ridge_{a}", "family": "ridge", "alpha": a} for a in [1., 100., 10000.]]
    specs += [{"name": "extra_trees_leaf8", "family": "extra_trees", "leaf": 8},
              {"name": "extra_trees_leaf24", "family": "extra_trees", "leaf": 24}]
    specs += [{"name": "hist_boost_small", "family": "hist", "leaves": 15, "l2": 10.},
              {"name": "hist_boost_deep", "family": "hist", "leaves": 31, "l2": 20.}]
    specs += [{"name": "catboost_d4", "family": "catboost", "depth": 4, "iterations": 800, "l2": 10.},
              {"name": "catboost_d6", "family": "catboost", "depth": 6, "iterations": 1000, "l2": 15.},
              {"name": "catboost_d8", "family": "catboost", "depth": 8, "iterations": 800, "l2": 25.}]
    specs += [{"name": "mlp_mse", "family": "neural", "sequence": False, "loss": "mse", "epochs": 35},
              {"name": "mlp_huber", "family": "neural", "sequence": False, "loss": "huber", "epochs": 35},
              {"name": "tcn_huber", "family": "neural", "sequence": True, "loss": "huber", "epochs": 30}]
    return specs


class TabularModel:
    def __init__(self, spec, seed=42, threads=8):
        self.spec, self.seed, self.threads = spec, seed, threads
        self.models = []

    def fit(self, x, y, current_glucose, validation=None, **kwargs):
        offset = current_glucose[:, None] if current_glucose.ndim == 1 else current_glucose
        target = y - offset
        family = self.spec["family"]
        complete = np.isfinite(target).all(axis=1)
        if family == "ridge":
            model = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), Ridge(alpha=self.spec["alpha"]))
            model.fit(x[complete], target[complete])
            self.models = [model]
        elif family == "extra_trees":
            model = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=self.spec["leaf"], max_features=.85,
                                        n_jobs=self.threads, random_state=self.seed)
            # Explicit imputation also records missingness and supports model
            # persistence across sklearn variants with different NaN routing.
            model = make_pipeline(SimpleImputer(add_indicator=True), model)
            model.fit(x[complete], target[complete])
            self.models = [model]
        elif family == "catboost":
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(iterations=self.spec["iterations"], depth=self.spec["depth"],
                learning_rate=.04, l2_leaf_reg=self.spec["l2"], loss_function="MultiRMSE",
                thread_count=self.threads, random_seed=self.seed, verbose=False, allow_writing_files=False)
            model.fit(x[complete], target[complete])
            self.models = [model]
        elif family == "hist":
            self.models = []
            for k in range(y.shape[1]):
                valid = np.isfinite(target[:, k])
                model = HistGradientBoostingRegressor(max_iter=450, learning_rate=.05,
                    max_leaf_nodes=self.spec["leaves"], l2_regularization=self.spec["l2"],
                    min_samples_leaf=40, early_stopping=False, random_state=self.seed)
                model.fit(x[valid], target[valid, k])
                self.models.append(model)
        else:
            raise ValueError(family)
        self.epochs_used = None
        return self

    def predict(self, x, current_glucose, **kwargs):
        values = self.models[0].predict(x) if len(self.models) == 1 else np.column_stack([m.predict(x) for m in self.models])
        offset = current_glucose[:, None] if current_glucose.ndim == 1 else current_glucose
        return values + offset


class NeuralModel:
    """Small MLP, optionally with a dilated convolutional glucose/event encoder."""
    def __init__(self, spec, seed=42, threads=8):
        self.spec, self.seed, self.threads = spec, seed, threads

    def fit(self, x, y, current_glucose, validation=None, sequence=None, fixed_epochs=None, **kwargs):
        import torch
        from torch import nn
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        self.imputer = SimpleImputer(add_indicator=True)
        self.scaler = StandardScaler()
        inputs = self.scaler.fit_transform(self.imputer.fit_transform(x)).astype(np.float32)
        self.sequence_mode = bool(self.spec["sequence"])
        if self.sequence_mode:
            self.seq_mean = np.nanmean(sequence, axis=(0, 2), keepdims=True)
            self.seq_scale = np.nanstd(sequence, axis=(0, 2), keepdims=True).clip(.001)
            seq = np.nan_to_num((sequence - self.seq_mean) / self.seq_scale).astype(np.float32)
        else:
            seq = np.zeros((len(x), 1, 1), dtype=np.float32)
        self.network = make_network(inputs.shape[1], seq.shape[1] if self.sequence_mode else 0)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=.0008, weight_decay=.01)
        xt = torch.from_numpy(inputs)
        st = torch.from_numpy(seq)
        yt = torch.from_numpy(np.nan_to_num((y - current_glucose[:, None]) / 50).astype(np.float32))
        mask = torch.from_numpy(np.isfinite(y))
        criterion = nn.SmoothL1Loss(reduction="none", beta=.5) if self.spec["loss"] == "huber" else nn.MSELoss(reduction="none")
        epochs = fixed_epochs or self.spec["epochs"]
        best_score, best_state, best_epoch, patience = np.inf, None, epochs, 0
        for epoch in range(epochs):
            self.network.train()
            indices = torch.randperm(len(x))
            for batch in indices.split(256):
                optimizer.zero_grad(set_to_none=True)
                prediction = self.network(xt[batch], st[batch])
                loss = criterion(prediction, yt[batch])[mask[batch]].mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 2.)
                optimizer.step()
            if validation is not None and fixed_epochs is None:
                vx, vy, vg, vs = validation
                score = np.nanmean(np.abs(self.predict(vx, vg, sequence=vs) - vy))
                if score < best_score - .01:
                    best_score, best_state, best_epoch, patience = score, copy.deepcopy(self.network.state_dict()), epoch + 1, 0
                else:
                    patience += 1
                if patience >= 6:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.epochs_used = best_epoch
        return self

    def predict(self, x, current_glucose, sequence=None, **kwargs):
        import torch
        inputs = self.scaler.transform(self.imputer.transform(x)).astype(np.float32)
        if self.sequence_mode:
            seq = np.nan_to_num((sequence - self.seq_mean) / self.seq_scale).astype(np.float32)
        else:
            seq = np.zeros((len(x), 1, 1), dtype=np.float32)
        self.network.eval()
        outputs = []
        with torch.inference_mode():
            for first in range(0, len(x), 512):
                outputs.append(self.network(torch.from_numpy(inputs[first:first + 512]), torch.from_numpy(seq[first:first + 512])).numpy())
        return np.concatenate(outputs) * 50 + current_glucose[:, None]


def make_network(n_features, n_channels=0):
    # Top-level classes are defined lazily in the separate torch module, so
    # importing tabular training need not load torch and its allocator.
    from src.personal_network import PersonalNetwork
    return PersonalNetwork(n_features, n_channels)


def create(spec, seed=42, threads=8):
    return (NeuralModel if spec["family"] == "neural" else TabularModel)(spec, seed, threads)
