"""Actual Timestral head / last-block adaptation through its native decoder.

The public forecast method is inference-only. This gradient path retains its
normalization, refinement, sign symmetry and nonnegative clipping, and is tested
against the public path before any optimization. Future inputs are calendar only.
"""

import gc
import json
import time

import numpy as np
import torch

from src.personal_data import historical_inputs, sample
from src.personal_timestral import PHYSICAL, calendar_covariates
from src.v3.cache import predictor
from src.v3.common import (atomic_joblib, atomic_json, dense_targets, folds,
    load_forecast, origin_weights, save_forecast, score, stage, training_positions,
    utc_boundary)


def differentiable_forecast(forecaster, context, past=None, known=None, horizon=36):
    from timestral.forecaster import nonnegative_context
    inputs, mask = forecaster._as_context(context, None)
    b, _, length = inputs.shape
    pc = forecaster._as_covariates(past, b, length, "past_covariates")
    kc = forecaster._as_covariates(known, b, length + horizon, "known_covariates")
    q = forecaster._forecast_raw(inputs, mask, horizon, pc, kc)
    reflected = forecaster._forecast_raw(-inputs, mask, horizon,
        None if pc is None else -pc, None if kc is None else -kc)
    q = .5 * (q - reflected.flip(-1))
    return torch.where(nonnegative_context(inputs, mask)[..., None, None], q.clamp_min(0), q)


def trainable_parameters(model, last_block):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    if last_block:
        for parameter in model.transformer.layers[-1].parameters():
            parameter.requires_grad_(True)
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


class FineTuner:
    def __init__(self, dataset, base, cfg, spec, device):
        torch.manual_seed(cfg["seed"])
        self.dataset, self.base, self.cfg, self.spec = dataset, base, cfg, spec
        self.predictor = predictor(dataset, base, device)
        self.forecaster = self.predictor.adapter.forecaster
        self.parameters = trainable_parameters(self.forecaster.model, spec["last_block"])
        self.anchor = {name: p.detach().clone() for name, p in self.parameters.items()}
        self.epochs_used = None

    def inputs(self, positions):
        values, index = historical_inputs(self.dataset, positions, self.spec["context_hours"], self.base["history_tolerance_seconds"], block_size=16)
        past, known = None, None
        if self.spec["mode"] == "physical":
            cols = [self.dataset.feature_names.index(n) for n in PHYSICAL]
            past = np.stack([sample(self.dataset.features[:, c], index) for c in cols], axis=1)
            known = calendar_covariates(self.dataset.times[positions], values.shape[-1], 36, self.base["timezone"])
        return values[:, None, :], past, known

    def state(self):
        return {name: parameter.detach().cpu().clone() for name, parameter in self.parameters.items()}

    def restore(self, state):
        missing, unexpected = self.forecaster.model.load_state_dict(state, strict=False)
        if unexpected or set(state) != set(self.parameters):
            raise ValueError("Partial Timestral checkpoint does not match trainable modules")

    def equivalence(self, positions):
        x, past, known = self.inputs(positions[:2])
        with torch.no_grad():
            expected = self.forecaster.forecast(x, prediction_length=36, past_covariates=past, known_covariates=known).quantiles
            actual = differentiable_forecast(self.forecaster, x, past, known)
        difference = float((actual - expected).abs().max().cpu())
        if difference > .001:
            raise ValueError(f"Differentiable/native Timestral paths disagree: {difference}")
        return difference

    def fit(self, positions, validation=None, epochs=None, checkpoint=None):
        settings = self.cfg["finetune"]
        head, block = [], []
        for name, p in self.parameters.items():
            (head if name.startswith("head.") else block).append(p)
        groups = [{"params": head, "lr": settings["head_learning_rate"]}]
        if block:
            groups.append({"params": block, "lr": settings["block_learning_rate"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=.001)
        levels = torch.tensor(self.forecaster.quantile_levels, device=self.forecaster.device)[None, None, :]
        target = dense_targets(self.dataset, positions, self.base)
        weights = origin_weights(self.dataset, positions, self.cfg)
        best, best_state, patience = np.inf, None, 0
        self.epochs_used = epochs or settings["epochs"]
        for epoch in range(epochs or settings["epochs"]):
            permutation = torch.randperm(len(positions)).numpy()
            losses = []
            for start in range(0, len(positions), self.spec["batch_size"]):
                ix = permutation[start:start+self.spec["batch_size"]]
                x, past, known = self.inputs(positions[ix])
                y = torch.as_tensor(np.nan_to_num(target[ix]), device=self.forecaster.device)
                valid = torch.as_tensor(np.isfinite(target[ix]), device=self.forecaster.device)
                if not valid.any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                q = differentiable_forecast(self.forecaster, x, past, known)[:, 0]
                error = (y[..., None] - q) / 50
                pinball = torch.maximum(levels * error, (levels - 1) * error).mean(dim=-1)
                weight = torch.as_tensor(weights[ix], device=q.device)[:, None]
                loss = (pinball * weight)[valid].sum() / weight.expand_as(pinball)[valid].sum()
                anchor = sum((p - self.anchor[n]).square().mean() for n, p in self.parameters.items())
                loss = loss + settings["anchor_penalty"] * anchor
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite Timestral training loss")
                loss.backward()
                for name, p in self.parameters.items():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise ValueError(f"Nonfinite Timestral gradient: {name}")
                torch.nn.utils.clip_grad_norm_(self.parameters.values(), 1.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            value = None
            if validation is not None and epochs is None:
                pred, _, _ = self.predict(validation)
                value = float(np.nanmean(np.abs(pred - self.dataset.targets[validation])))
                if value < best - .01:
                    best, best_state, patience, self.epochs_used = value, self.state(), 0, epoch + 1
                    if checkpoint is not None:
                        atomic_joblib(checkpoint, {"spec": self.spec, "state": best_state, "epochs": self.epochs_used})
                else:
                    patience += 1
            print(f"  {self.spec['name']} epoca {epoch+1}: loss={np.mean(losses):.4f}, MAE_validazione={value}", flush=True)
            if patience >= settings["patience"]:
                break
        if best_state is not None:
            self.restore(best_state)
        self.anchor.clear()
        return self

    def predict(self, positions):
        medians, curves, quantiles = [], [], []
        for start in range(0, len(positions), self.spec["batch_size"]):
            x, past, known = self.inputs(positions[start:start+self.spec["batch_size"]])
            q = self.forecaster.forecast(x, prediction_length=36, past_covariates=past, known_covariates=known).quantiles[:, 0].cpu().numpy()
            if not np.isfinite(q).all():
                raise ValueError("Nonfinite adapted Timestral forecast")
            curves.append(q[:, :, 10])
            medians.append(q[:, [5, 11, 23, 35], 10])
            quantiles.append(q[:, [5, 11, 23, 35]])
        return tuple(np.concatenate(parts).astype(np.float32) for parts in [medians, curves, quantiles])


def run(dataset, base, cfg, output, device):
    stage(output, "finetune", dataset, cfg, ["src/v3/finetune.py", "src/v3/cache.py", "src/personal_timestral.py", "src/forecasting.py"])
    candidates = []
    for parameters in cfg["finetune"]["specs"]:
        spec = dict(parameters, family="finetune", policy="fixed")
        positions, predictions, fold_ids, details = [], [], [], []
        for k, (date, _, valid) in enumerate(folds(dataset, base)):
            train = training_positions(dataset, base, utc_boundary(date, base).timestamp(), cfg["finetune"]["train_stride_readings"])
            selected = valid[valid % cfg["finetune"]["selection_stride_readings"] == 0]
            path = output / "oof" / f"{spec['name']}_{date}.npz"
            checkpoint = output / "finetune-models" / f"{spec['name']}_{date}.joblib"
            if path.exists():
                pred = load_forecast(path, valid)
                detail = json.loads(path.with_suffix(".json").read_text())
            else:
                started = time.perf_counter()
                learner = FineTuner(dataset, base, cfg, spec, device)
                equivalence = learner.equivalence(train[:2])
                learner.fit(train, validation=selected, checkpoint=checkpoint)
                pred, curve, q = learner.predict(valid)
                save_forecast(path, valid, pred, curve=curve, quantiles=q)
                detail = {"fold": date, "train_rows": len(train), "epochs": learner.epochs_used,
                          "native_equivalence_max_mgdl": equivalence, "trainable_parameters": sum(p.numel() for p in learner.parameters.values()),
                          "seconds": time.perf_counter()-started, "metrics": score(dataset.targets[valid], pred)}
                atomic_joblib(checkpoint, {"spec": spec, "state": learner.state(), "epochs": learner.epochs_used})
                atomic_json(path.with_suffix(".json"), detail)
                del learner
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            print(f"{spec['name']} / {date}: MAE {np.round(detail['metrics']['mae'],2).tolist()}", flush=True)
            positions.append(valid)
            predictions.append(pred)
            fold_ids.append(np.full(len(valid), k))
            details.append(detail)
        pos, pred = np.concatenate(positions), np.concatenate(predictions)
        save_forecast(output / "oof" / f"{spec['name']}.npz", pos, pred, folds=np.concatenate(fold_ids))
        candidates.append({"spec": spec, "metrics": score(dataset.targets[pos], pred), "folds": details})
    atomic_json(output / "finetune-selection.json", {"candidates": candidates})
