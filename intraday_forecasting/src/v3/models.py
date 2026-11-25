"""Weighted conditional-median boosting and dense trajectory learners."""

import copy

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class BoostModel:
    def __init__(self, parameters, seed=42, threads=8):
        self.parameters, self.seed, self.threads = parameters, seed, threads
        self.epochs_used = None

    def fit(self, x, target, offset, weights=None, **kwargs):
        from catboost import CatBoostRegressor
        delta = target - (offset[:, None] if offset.ndim == 1 else offset)
        self.models = []
        for k in range(target.shape[1]):
            valid = np.isfinite(delta[:, k])
            model = CatBoostRegressor(**self.parameters, loss_function="MAE", random_seed=self.seed,
                                      thread_count=self.threads, verbose=False, allow_writing_files=False)
            model.fit(x[valid], delta[valid, k], sample_weight=weights[valid] if weights is not None else None)
            self.models.append(model)
        return self

    def predict(self, x, offset, **kwargs):
        values = np.column_stack([model.predict(x) for model in self.models])
        return values + (offset[:, None] if offset.ndim == 1 else offset)


class TrajectoryModel:
    def __init__(self, parameters, weighted=False, seed=42, threads=8, device="cpu"):
        self.parameters, self.weighted, self.seed, self.threads = parameters, weighted, seed, threads
        self.device = device

    def to(self, device):
        self.device = str(device)
        if hasattr(self, "network"):
            self.network.to(device)
        return self

    def fit(self, x, target, offset, sequence, weights, validation=None, fixed_epochs=None, **kwargs):
        import torch
        from src.v3.trajectory_network import TrajectoryNetwork
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        self.imputer = SimpleImputer(add_indicator=True)
        self.scaler = StandardScaler()
        inputs = self.scaler.fit_transform(self.imputer.fit_transform(x)).astype(np.float32)
        self.seq_mean = np.nanmean(sequence, axis=(0, 2), keepdims=True)
        self.seq_scale = np.nanstd(sequence, axis=(0, 2), keepdims=True).clip(.001)
        seq = np.nan_to_num((sequence - self.seq_mean) / self.seq_scale).astype(np.float32)
        self.network = TrajectoryNetwork(inputs.shape[1], seq.shape[1], target.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.parameters["learning_rate"], weight_decay=.01)
        y = (target - offset[:, None]) / 50
        mask = np.isfinite(y)
        xt, st = torch.from_numpy(inputs), torch.from_numpy(seq)
        yt, mt, wt = torch.from_numpy(np.nan_to_num(y).astype(np.float32)), torch.from_numpy(mask), torch.from_numpy(weights)
        best_score, best_state, patience = np.inf, None, 0
        epochs = fixed_epochs or self.parameters["epochs"]
        self.epochs_used = epochs
        for epoch in range(epochs):
            self.network.train()
            for index in torch.randperm(len(x)).split(self.parameters["batch_size"]):
                valid = mt[index].to(self.device)
                if not valid.any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                predicted = self.network(xt[index].to(self.device), st[index].to(self.device))
                expected = yt[index].to(self.device)
                weight = wt[index].to(self.device)[:, None]
                error = torch.nn.functional.smooth_l1_loss(predicted, expected, reduction="none", beta=.5)
                loss = (error * weight)[valid].sum() / (weight.expand_as(error)[valid].sum())
                if self.weighted:
                    before_pred = torch.cat([torch.zeros_like(predicted[:, :1]), predicted[:, :-1]], dim=1)
                    before_true = torch.cat([torch.zeros_like(expected[:, :1]), expected[:, :-1]], dim=1)
                    before_mask = torch.cat([torch.ones_like(valid[:, :1]), valid[:, :-1]], dim=1)
                    pair_mask = valid & before_mask
                    slopes = torch.nn.functional.smooth_l1_loss((predicted - before_pred) * 5, (expected - before_true) * 5, beta=.5, reduction="none")
                    if pair_mask.any():
                        loss = loss + self.parameters["slope_penalty"] * (slopes * weight)[pair_mask].sum() / weight.expand_as(slopes)[pair_mask].sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 2.)
                optimizer.step()
            if validation is not None and fixed_epochs is None:
                vx, vy, vo, vs = validation
                predicted = self.predict_curve(vx, vo, vs)
                score = np.nanmean(np.abs(predicted[:, [5, 11, 23, 35]] - vy[:, [5, 11, 23, 35]]))
                if score < best_score - .01:
                    best_score, patience, self.epochs_used = score, 0, epoch + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                else:
                    patience += 1
                if patience >= self.parameters["patience"]:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        return self

    def predict_curve(self, x, offset, sequence):
        import torch
        inputs = self.scaler.transform(self.imputer.transform(x)).astype(np.float32)
        seq = np.nan_to_num((sequence - self.seq_mean) / self.seq_scale).astype(np.float32)
        self.network.eval()
        result = []
        device = next(self.network.parameters()).device
        with torch.inference_mode():
            for start in range(0, len(x), 512):
                result.append(self.network(torch.from_numpy(inputs[start:start+512]).to(device), torch.from_numpy(seq[start:start+512]).to(device)).cpu().numpy())
        return np.concatenate(result) * 50 + offset[:, None]

    def predict(self, x, offset, sequence=None):
        return self.predict_curve(x, offset, sequence)[:, [5, 11, 23, 35]]


def create(spec, cfg, device="cpu"):
    if spec["family"] == "trajectory":
        return TrajectoryModel(cfg["trajectory"], spec["weighted"], cfg["seed"], cfg["threads"], device)
    parameters = cfg["boosting"].copy()
    parameters["l2_leaf_reg"] = parameters.pop("l2")
    return BoostModel(parameters, cfg["seed"], cfg["threads"])
