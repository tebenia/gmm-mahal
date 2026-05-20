"""Model loading, training, saving, and SHAP helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import load as joblib_load

from ..utils.paths import require_path


ACTIVE_MODEL_CONFIG: dict[str, Any] = {}


def configure(config: dict[str, Any]) -> None:
    ACTIVE_MODEL_CONFIG.clear()
    ACTIVE_MODEL_CONFIG.update(config)


def load_model(model_id, data_id, save_path=None, file_name=None):
    if model_id != "lightgbm":
        raise NotImplementedError(f"Only lightgbm is supported, got {model_id}")
    model_path = ACTIVE_MODEL_CONFIG.get("model_path")
    if model_path is None:
        if save_path is None or file_name is None:
            raise ValueError("model_path or save_path/file_name is required")
        model_path = Path(save_path) / file_name
    return load_lightgbm(model_path)


def load_lightgbm(model_path):
    path = require_path(model_path)
    if path.suffix.lower() in {".model", ".txt", ""}:
        return lgb.Booster(model_file=str(path))
    artifact = joblib_load(path)
    model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
    booster = getattr(model, "booster_", None)
    if booster is None and isinstance(model, lgb.Booster):
        booster = model
    if booster is None:
        raise ValueError(f"Unsupported LightGBM artifact: {path}")
    return booster


def train_model(model_id, x_train, y_train):
    if model_id != "lightgbm":
        raise NotImplementedError(f"Only lightgbm is supported, got {model_id}")
    return train_lightgbm(x_train, y_train)


def train_lightgbm(x_train, y_train):
    lgbm_dataset = lgb.Dataset(x_train, y_train)
    return lgb.train({"application": "binary"}, lgbm_dataset)


def save_model(model_id, model, save_path, file_name):
    if model_id != "lightgbm":
        raise NotImplementedError(f"Only lightgbm is supported, got {model_id}")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_model(str(save_path / file_name))


def explain_model(data_id, model_id, model, x_exp, x_back=None, perc=1.0, n_samples=100, load=False, save=False):
    if model_id != "lightgbm":
        raise NotImplementedError(f"Only lightgbm SHAP is supported, got {model_id}")
    contribs = model.predict(x_exp, pred_contrib=True)
    np_contribs = np.array(contribs)
    return pd.DataFrame(np_contribs[:, 0:-1])


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    predictions = np.array([1 if pred > 0.5 else 0 for pred in predictions])
    return float(np.mean(predictions == y_test))
