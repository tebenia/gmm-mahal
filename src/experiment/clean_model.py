"""Train and evaluate a seed-specific clean LightGBM model."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ..data import model_utils
from ..utils.paths import resolve_path
from .subsets import PreparedSubset, array_sha256


DEFAULT_LIGHTGBM_SPEC = {
    "params": {
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "num_threads": -1,
    },
    "num_boost_round": 400,
}


def train_clean_model(
    *,
    baseline_id: str,
    spec: dict[str, Any],
    training_profile: dict[str, Any],
    subset: PreparedSubset,
    overwrite: bool = False,
    retune: bool = False,
    batch_size: int = 8192,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_path = resolve_path(spec["model_path"])
    params_path = resolve_path(spec["lightgbm_params_path"])
    if model_path is None or params_path is None:
        raise ValueError("model_path and lightgbm_params_path are required")
    output_dir = model_path.parent
    metadata_path = output_dir / "training_metadata.json"
    metrics_json_path = output_dir / "clean_test_metrics.json"
    metrics_csv_path = output_dir / "clean_test_metrics.csv"
    existing = [path for path in (model_path, metadata_path, metrics_json_path, metrics_csv_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Clean-model outputs already exist: {}. Pass --overwrite to replace them.".format(
                ", ".join(str(path) for path in existing)
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    training_spec = resolve_or_tune_training_spec(
        params_path=params_path,
        profile=training_profile,
        subset=subset,
        retune=retune,
    )
    params, num_boost_round = model_utils.resolve_lightgbm_training_spec(
        training_spec,
        seed=subset.seed,
    )

    start = time.time()
    x_selected = np.asarray(subset.x_train[subset.indices], dtype=np.float32)
    y_selected = np.asarray(subset.y_train[subset.indices], dtype=np.int32)
    train_set = lgb.Dataset(x_selected, label=y_selected, free_raw_data=True)
    model = lgb.train(params, train_set, num_boost_round=num_boost_round)
    training_seconds = time.time() - start
    model.save_model(str(model_path))
    del x_selected, y_selected

    metrics = evaluate_binary_model(
        model,
        subset.x_test,
        subset.y_test,
        batch_size=batch_size,
    )
    metrics.update(
        {
            "baseline_id": baseline_id,
            "dataset_id": spec["dataset_id"],
            "seed": subset.seed,
            "train_rows": int(subset.indices.size),
            "feature_dim": subset.feature_dim,
            "training_seconds": training_seconds,
        }
    )
    metrics_json_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)

    metadata = {
        "baseline_id": baseline_id,
        "dataset_id": spec["dataset_id"],
        "kind": spec["kind"],
        "seed": subset.seed,
        "feature_dim": subset.feature_dim,
        "train_rows": int(subset.indices.size),
        "train_label_counts": label_count_dict(subset.y_train[subset.indices]),
        "test_rows": int(metrics["test_rows"]),
        "test_label_counts": {
            "0": int(metrics["test_benign_rows"]),
            "1": int(metrics["test_malware_rows"]),
        },
        "train_fraction": subset.train_fraction,
        "subset_mode": subset.subset_mode,
        "source_dir": str(subset.source_dir),
        "subset_indices_path": str(subset.indices_path),
        "subset_indices_sha256": array_sha256(subset.indices),
        "subset_metadata_path": str(subset.metadata_path),
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "lightgbm_params_path": str(params_path),
        "training_spec": training_spec,
        "resolved_training_params": params,
        "num_boost_round": num_boost_round,
        "metrics_path": str(metrics_json_path),
        "training_seconds": training_seconds,
        "package_versions": package_versions(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics_json_path": str(metrics_json_path),
        "metrics_csv_path": str(metrics_csv_path),
        "params_path": str(params_path),
        "metrics": metrics,
    }


def resolve_or_tune_training_spec(
    *,
    params_path: Path,
    profile: dict[str, Any],
    subset: PreparedSubset,
    retune: bool,
) -> dict[str, Any]:
    if params_path.exists() and not retune:
        return model_utils.load_lightgbm_training_spec(params_path)

    mode = str(profile.get("mode", "fixed"))
    if mode not in {"fixed", "optuna_if_missing"}:
        raise ValueError(f"Unsupported training mode: {mode}")
    if mode == "fixed":
        training_spec = {
            **DEFAULT_LIGHTGBM_SPEC,
            "selection": {"mode": "fixed"},
        }
    else:
        tuning_seed = int(profile.get("tuning_seed", 42))
        if subset.seed != tuning_seed:
            raise FileNotFoundError(
                f"Shared LightGBM parameters do not exist at {params_path}. "
                f"Prepare and train seed {tuning_seed} first; later seeds reuse its "
                "hyperparameters for a controlled comparison."
            )
        training_spec = tune_lightgbm(subset=subset, profile=profile)

    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(
        json.dumps(training_spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return training_spec


def tune_lightgbm(
    *,
    subset: PreparedSubset,
    profile: dict[str, Any],
) -> dict[str, Any]:
    n_trials = int(profile.get("n_trials", 30))
    metric = str(profile.get("metric", "roc_auc"))
    validation_fraction = float(profile.get("validation_fraction", 0.1))
    max_samples = int(profile.get("max_tuning_samples", 50_000))
    if n_trials <= 0 or max_samples <= 0:
        raise ValueError("n_trials and max_tuning_samples must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if metric != "roc_auc":
        raise ValueError(f"Unsupported tuning metric: {metric}")

    local_indices = np.arange(subset.indices.size)
    selected_labels = np.asarray(subset.y_train[subset.indices], dtype=np.int32)
    if local_indices.size > max_samples:
        local_indices, _ = train_test_split(
            local_indices,
            train_size=max_samples,
            stratify=selected_labels,
            random_state=subset.seed,
        )
    x_tuning = np.asarray(subset.x_train[subset.indices[local_indices]], dtype=np.float32)
    y_tuning = selected_labels[local_indices]
    train_idx, valid_idx = train_test_split(
        np.arange(y_tuning.size),
        test_size=validation_fraction,
        stratify=y_tuning,
        random_state=subset.seed,
    )

    def objective(trial: optuna.Trial) -> float:
        num_boost_round = trial.suggest_int("num_boost_round", 100, 700, step=50)
        candidate = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, step=16),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200, step=10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 20.0, log=True),
            "num_threads": -1,
        }
        params, _ = model_utils.resolve_lightgbm_training_spec(
            {"params": candidate, "num_boost_round": num_boost_round},
            seed=subset.seed,
        )
        model = lgb.train(
            params,
            lgb.Dataset(x_tuning[train_idx], label=y_tuning[train_idx]),
            num_boost_round=num_boost_round,
        )
        scores = model.predict(x_tuning[valid_idx])
        return float(roc_auc_score(y_tuning[valid_idx], scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=subset.seed),
        study_name=f"lightgbm_{subset.baseline_id}_roc_auc",
    )
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_trial.params)
    num_boost_round = int(best.pop("num_boost_round"))
    best["bagging_freq"] = 1
    best["num_threads"] = -1
    return {
        "params": best,
        "num_boost_round": num_boost_round,
        "selection": {
            "mode": "optuna",
            "metric": metric,
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
            "tuning_rows": int(y_tuning.size),
            "training_rows": int(train_idx.size),
            "validation_rows": int(valid_idx.size),
            "validation_fraction": validation_fraction,
            "seed": subset.seed,
        },
    }


def evaluate_binary_model(model, x_test, y_test, *, batch_size: int) -> dict[str, Any]:
    labels = np.asarray(y_test, dtype=np.int32)
    valid = labels != -1
    valid_positions = np.flatnonzero(valid)
    scores = np.empty(valid_positions.size, dtype=np.float64)
    for start in range(0, valid_positions.size, batch_size):
        end = min(start + batch_size, valid_positions.size)
        scores[start:end] = model.predict(x_test[valid_positions[start:end]])
    labels = labels[valid]
    predictions = (scores > 0.5).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "test_rows": int(labels.size),
        "test_benign_rows": int(np.count_nonzero(labels == 0)),
        "test_malware_rows": int(np.count_nonzero(labels == 1)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predictions)),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def label_count_dict(labels) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def package_versions() -> dict[str, str]:
    versions = {
        package: importlib.metadata.version(package)
        for package in ("lightgbm", "numpy", "optuna", "pandas", "scikit-learn")
    }
    versions["python"] = platform.python_version()
    return versions
