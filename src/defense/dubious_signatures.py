"""DUBIOUS-inspired test-time backdoored-input detector.

This module implements the perturbation-signature idea from DUBIOUS for the
saved malware backdoor artifacts in this project. It is a test-time rejection
detector: it does not remove poisoned training rows or retrain a model.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..attack.baseline import DEFAULT_BASELINES_CONFIG, build_context
from ..data import data_utils
from ..utils.paths import resolve_path
from .retrain_evaluate import find_backdoored_model_path


SUPPORTED_FEATURE_MODES = {"random", "shap_topk"}
SUPPORTED_REPLACEMENTS = {"benign_mean", "malware_mean", "benign_sample", "malware_sample"}
SUPPORTED_REFERENCE_SPLITS = {"train", "test"}
SUPPORTED_SCORE_MODES = {"dubious_l1", "apc_l1"}


@dataclass
class DubiousConfig:
    artifact_dir: str
    baseline: str
    output_dir: str
    config_path: str
    magnitudes: list[int]
    n_perturbations: int = 100
    feature_mode: str = "random"
    top_k: int = 50
    replacement: str = "benign_mean"
    reference_split: str = "train"
    max_reference_per_class: int = 50
    max_clean_eval_per_class: int = 500
    max_watermarked_eval: int = 1000
    threshold_scale: float = 1.5
    nearest_k: int = 3
    score_mode: str = "dubious_l1"
    random_state: int = 42


@dataclass
class DubiousResult:
    output_dir: str
    reference_signatures_path: str
    eval_signatures_path: str
    scores_path: str
    metrics_path: str
    metadata_path: str
    watermarked_detection_rate: float | None
    watermarked_predicted_benign_detection_rate: float | None
    clean_benign_fpr: float | None
    clean_malware_fpr: float | None
    max_clean_fpr: float | None
    detection_minus_max_fpr: float | None
    watermarked_asr_before_rejection: float | None
    watermarked_asr_after_rejection: float | None


def run_dubious_defense(
    artifact_dir: str | Path,
    baseline: str,
    output_dir: str | Path | None = None,
    config_path: str | Path = DEFAULT_BASELINES_CONFIG,
    magnitudes: list[int] | tuple[int, ...] = (10, 20, 30, 40, 50),
    n_perturbations: int = 100,
    feature_mode: str = "random",
    top_k: int = 50,
    replacement: str = "benign_mean",
    reference_split: str = "train",
    max_reference_per_class: int = 50,
    max_clean_eval_per_class: int = 500,
    max_watermarked_eval: int = 1000,
    threshold_scale: float = 1.5,
    nearest_k: int = 3,
    score_mode: str = "dubious_l1",
    random_state: int = 42,
    overwrite: bool = False,
    dry_run: bool = False,
) -> DubiousResult | dict:
    artifact_path = _resolve_existing_dir(artifact_dir)
    model_path = find_backdoored_model_path(artifact_path)
    if model_path is None:
        raise FileNotFoundError(f"Missing backdoored LightGBM model in {artifact_path}")
    watermarked_test_path = artifact_path / "watermarked_X_test.npy"
    if not watermarked_test_path.exists():
        raise FileNotFoundError(
            f"Missing {watermarked_test_path}. Re-run the attack with --save-attack-artifacts."
        )

    magnitudes = [int(value) for value in magnitudes]
    if not magnitudes:
        raise ValueError("--magnitudes must include at least one value")
    if any(value < 0 for value in magnitudes):
        raise ValueError("--magnitudes values must be non-negative")
    if n_perturbations <= 0:
        raise ValueError("--n-perturbations must be positive")
    if feature_mode not in SUPPORTED_FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode {feature_mode}. Available: {sorted(SUPPORTED_FEATURE_MODES)}")
    if replacement not in SUPPORTED_REPLACEMENTS:
        raise ValueError(f"Unsupported replacement {replacement}. Available: {sorted(SUPPORTED_REPLACEMENTS)}")
    if reference_split not in SUPPORTED_REFERENCE_SPLITS:
        raise ValueError(
            f"Unsupported reference_split {reference_split}. Available: {sorted(SUPPORTED_REFERENCE_SPLITS)}"
        )
    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(f"Unsupported score_mode {score_mode}. Available: {sorted(SUPPORTED_SCORE_MODES)}")
    if top_k <= 0:
        raise ValueError("--top-k must be positive")
    if max_reference_per_class <= 0:
        raise ValueError("--max-reference-per-class must be positive")
    if max_clean_eval_per_class <= 0:
        raise ValueError("--max-clean-eval-per-class must be positive")
    if max_watermarked_eval <= 0:
        raise ValueError("--max-watermarked-eval must be positive")
    if threshold_scale <= 0:
        raise ValueError("--threshold-scale must be positive")
    if nearest_k <= 0:
        raise ValueError("--nearest-k must be positive")

    output_path = resolve_output_dir(
        output_dir=output_dir,
        artifact_dir=artifact_path,
        feature_mode=feature_mode,
        replacement=replacement,
        score_mode=score_mode,
        magnitudes=magnitudes,
        n_perturbations=n_perturbations,
        max_watermarked_eval=max_watermarked_eval,
    )
    metadata_path = output_path / "dubious_metadata.json"
    reference_signatures_path = output_path / "reference_signatures.csv"
    eval_signatures_path = output_path / "eval_signatures.csv"
    scores_path = output_path / "dubious_scores.csv"
    metrics_path = output_path / "dubious_metrics.csv"

    if metadata_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"{metadata_path} already exists. Pass --overwrite to replace it.")

    config = DubiousConfig(
        artifact_dir=str(artifact_path),
        baseline=baseline,
        output_dir=str(output_path),
        config_path=str(config_path),
        magnitudes=magnitudes,
        n_perturbations=n_perturbations,
        feature_mode=feature_mode,
        top_k=top_k,
        replacement=replacement,
        reference_split=reference_split,
        max_reference_per_class=max_reference_per_class,
        max_clean_eval_per_class=max_clean_eval_per_class,
        max_watermarked_eval=max_watermarked_eval,
        threshold_scale=threshold_scale,
        nearest_k=nearest_k,
        score_mode=score_mode,
        random_state=random_state,
    )

    if dry_run:
        return {
            "artifact_dir": str(artifact_path),
            "backdoored_model_path": str(model_path),
            "watermarked_X_test_path": str(watermarked_test_path),
            "baseline": baseline,
            "config_path": str(config_path),
            "output_dir": str(output_path),
            "config": asdict(config),
        }

    start_time = time.time()
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    context = build_context(baseline, config_path=config_path)
    X_train, y_train, X_test, y_test = data_utils.load_dataset(dataset=context.dataset_id, selected=True)
    y_train = np.asarray(y_train).astype(np.int8)
    y_test = np.asarray(y_test).astype(np.int8)
    X_watermarked = _unwrap_saved_array(np.load(watermarked_test_path, mmap_mode="r", allow_pickle=True))
    model = lgb.Booster(model_file=str(model_path))

    X_replacement_source, y_replacement_source = X_train, y_train
    replacement_values = build_replacement_values(X_replacement_source, y_replacement_source, replacement, rng)

    X_ref_source, y_ref_source = (X_train, y_train) if reference_split == "train" else (X_test, y_test)
    reference_selection = select_clean_reference_rows(
        model=model,
        X=X_ref_source,
        y=y_ref_source,
        max_per_class=max_reference_per_class,
        rng=rng,
    )
    clean_eval_selection = select_clean_eval_rows(
        model=model,
        X=X_test,
        y=y_test,
        max_per_class=max_clean_eval_per_class,
        rng=rng,
    )
    watermarked_selection = select_watermarked_rows(X_watermarked, max_rows=max_watermarked_eval, rng=rng)

    reference_df = build_signature_frame(
        model=model,
        X=X_ref_source[reference_selection["indices"]],
        true_labels=y_ref_source[reference_selection["indices"]],
        source_row_ids=reference_selection["indices"],
        group_labels=reference_selection["groups"],
        magnitudes=magnitudes,
        n_perturbations=n_perturbations,
        feature_mode=feature_mode,
        top_k=top_k,
        replacement_values=replacement_values,
        rng=rng,
    )
    eval_clean_df = build_signature_frame(
        model=model,
        X=X_test[clean_eval_selection["indices"]],
        true_labels=y_test[clean_eval_selection["indices"]],
        source_row_ids=clean_eval_selection["indices"],
        group_labels=clean_eval_selection["groups"],
        magnitudes=magnitudes,
        n_perturbations=n_perturbations,
        feature_mode=feature_mode,
        top_k=top_k,
        replacement_values=replacement_values,
        rng=rng,
    )
    watermarked_df = build_signature_frame(
        model=model,
        X=X_watermarked[watermarked_selection],
        true_labels=np.ones(watermarked_selection.shape[0], dtype=np.int8),
        source_row_ids=watermarked_selection,
        group_labels=np.full(watermarked_selection.shape[0], "watermarked_malware", dtype=object),
        magnitudes=magnitudes,
        n_perturbations=n_perturbations,
        feature_mode=feature_mode,
        top_k=top_k,
        replacement_values=replacement_values,
        rng=rng,
    )
    eval_df = pd.concat([eval_clean_df, watermarked_df], ignore_index=True)

    signature_columns = signature_feature_columns(magnitudes, score_mode=score_mode)
    scores_df, threshold_metadata = score_signatures(
        reference_df=reference_df,
        eval_df=eval_df,
        signature_columns=signature_columns,
        nearest_k=nearest_k,
        threshold_scale=threshold_scale,
    )
    metrics = compute_metrics(scores_df)

    reference_df.to_csv(reference_signatures_path, index=False)
    eval_df.to_csv(eval_signatures_path, index=False)
    scores_df.to_csv(scores_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    metadata = {
        "config": asdict(config),
        "artifact_files": {
            "artifact_dir": str(artifact_path),
            "backdoored_model": str(model_path),
            "watermarked_X_test": str(watermarked_test_path),
        },
        "dataset": {
            "baseline": baseline,
            "dataset_id": context.dataset_id,
            "reference_split": reference_split,
            "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
            "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
            "watermarked_test_shape": [int(X_watermarked.shape[0]), int(X_watermarked.shape[1])],
        },
        "selection": {
            "reference_rows": int(reference_df.shape[0]),
            "clean_eval_rows": int(eval_clean_df.shape[0]),
            "watermarked_eval_rows": int(watermarked_df.shape[0]),
            "reference_groups": reference_df["group"].value_counts().to_dict(),
            "eval_groups": eval_df["group"].value_counts().to_dict(),
        },
        "thresholds": threshold_metadata,
        "metrics": metrics,
        "outputs": {
            "reference_signatures": str(reference_signatures_path),
            "eval_signatures": str(eval_signatures_path),
            "scores": str(scores_path),
            "metrics": str(metrics_path),
        },
        "runtime_seconds": time.time() - start_time,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return DubiousResult(
        output_dir=str(output_path),
        reference_signatures_path=str(reference_signatures_path),
        eval_signatures_path=str(eval_signatures_path),
        scores_path=str(scores_path),
        metrics_path=str(metrics_path),
        metadata_path=str(metadata_path),
        watermarked_detection_rate=metrics.get("watermarked_detection_rate"),
        watermarked_predicted_benign_detection_rate=metrics.get("watermarked_predicted_benign_detection_rate"),
        clean_benign_fpr=metrics.get("clean_benign_fpr"),
        clean_malware_fpr=metrics.get("clean_malware_fpr"),
        max_clean_fpr=metrics.get("max_clean_fpr"),
        detection_minus_max_fpr=metrics.get("detection_minus_max_fpr"),
        watermarked_asr_before_rejection=metrics.get("watermarked_asr_before_rejection"),
        watermarked_asr_after_rejection=metrics.get("watermarked_asr_after_rejection"),
    )


def build_signature_frame(
    model: lgb.Booster,
    X,
    true_labels: np.ndarray,
    source_row_ids: np.ndarray,
    group_labels: np.ndarray,
    magnitudes: list[int],
    n_perturbations: int,
    feature_mode: str,
    top_k: int,
    replacement_values: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    X_array = np.asarray(X)
    true_labels = np.asarray(true_labels).astype(np.int8)
    source_row_ids = np.asarray(source_row_ids, dtype=np.int64)
    group_labels = np.asarray(group_labels, dtype=object)
    base_raw = raw_scores(model, X_array)
    base_prob = sigmoid(base_raw)
    base_pred = (base_prob > 0.5).astype(np.int8)
    base_class_logit = predicted_class_logit(base_raw, base_pred)
    feature_candidates = build_feature_candidates(model, X_array, feature_mode=feature_mode, top_k=top_k)

    rows: list[dict[str, Any]] = []
    for row_pos in range(X_array.shape[0]):
        row = {
            "source_row_id": int(source_row_ids[row_pos]),
            "group": str(group_labels[row_pos]),
            "true_label": int(true_labels[row_pos]),
            "base_raw_score": float(base_raw[row_pos]),
            "base_probability": float(base_prob[row_pos]),
            "base_pred": int(base_pred[row_pos]),
            "base_correct": bool(base_pred[row_pos] == true_labels[row_pos]),
        }
        x_row = np.asarray(X_array[row_pos], dtype=np.float32)
        for magnitude in magnitudes:
            if magnitude == 0:
                pert_raw = np.repeat(base_raw[row_pos], n_perturbations)
            else:
                perturbed = perturb_repeated(
                    x=x_row,
                    candidates=feature_candidates[row_pos],
                    magnitude=magnitude,
                    n_perturbations=n_perturbations,
                    replacement_values=replacement_values,
                    rng=rng,
                )
                pert_raw = raw_scores(model, perturbed)
            pert_prob = sigmoid(pert_raw)
            pert_pred = (pert_prob > 0.5).astype(np.int8)
            class_logits = predicted_class_logit(pert_raw, np.repeat(base_pred[row_pos], pert_raw.shape[0]))
            mean_logit = float(np.mean(class_logits))
            std_logit = float(np.std(class_logits))
            accuracy = float(np.mean(pert_pred == base_pred[row_pos]))
            abs_delta = float(abs(mean_logit - base_class_logit[row_pos]))
            denom = max(abs(float(base_class_logit[row_pos])), 1e-12)
            apc = float(abs_delta / denom * 100.0)
            tag = magnitude_tag(magnitude)
            row[f"mean_logit_m{tag}"] = mean_logit
            row[f"std_logit_m{tag}"] = std_logit
            row[f"stability_m{tag}"] = accuracy
            row[f"abs_delta_m{tag}"] = abs_delta
            row[f"apc_m{tag}"] = apc
        rows.append(row)
    return pd.DataFrame(rows)


def perturb_repeated(
    x: np.ndarray,
    candidates: np.ndarray,
    magnitude: int,
    n_perturbations: int,
    replacement_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("No feature candidates available for perturbation")
    perturb_count = min(int(magnitude), candidates.shape[0])
    X_rep = np.repeat(x.reshape(1, -1), n_perturbations, axis=0).astype(np.float32, copy=True)
    for row_idx in range(n_perturbations):
        chosen = rng.choice(candidates, size=perturb_count, replace=False)
        if replacement_values.ndim == 1:
            X_rep[row_idx, chosen] = replacement_values[chosen]
        else:
            source_row = int(rng.integers(0, replacement_values.shape[0]))
            X_rep[row_idx, chosen] = replacement_values[source_row, chosen]
    return X_rep


def score_signatures(
    reference_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    signature_columns: list[str],
    nearest_k: int,
    threshold_scale: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if reference_df.empty:
        raise ValueError("No clean reference signatures were selected")
    missing = [column for column in signature_columns if column not in reference_df.columns]
    if missing:
        raise KeyError(f"Missing signature columns in reference signatures: {missing}")
    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(reference_df[signature_columns].to_numpy(dtype=np.float64))
    eval_scaled = scaler.transform(eval_df[signature_columns].to_numpy(dtype=np.float64))

    ref_by_class: dict[int, np.ndarray] = {}
    thresholds: dict[int, float] = {}
    threshold_meta: dict[str, Any] = {}
    ref_preds = reference_df["base_pred"].to_numpy(dtype=np.int8)
    for pred_class in sorted(np.unique(ref_preds).tolist()):
        class_mask = ref_preds == pred_class
        class_ref = ref_scaled[class_mask]
        ref_by_class[int(pred_class)] = class_ref
        threshold, details = clean_nn_threshold(class_ref, nearest_k=nearest_k, threshold_scale=threshold_scale)
        thresholds[int(pred_class)] = threshold
        threshold_meta[str(int(pred_class))] = details

    scores = []
    for row_idx, row in eval_df.reset_index(drop=True).iterrows():
        pred_class = int(row["base_pred"])
        class_ref = ref_by_class.get(pred_class)
        threshold = thresholds.get(pred_class)
        if class_ref is None or threshold is None or class_ref.shape[0] == 0:
            distance = np.nan
            suspicious = False
        else:
            distances = np.sum(np.abs(class_ref - eval_scaled[row_idx]), axis=1)
            distance = float(np.min(distances))
            suspicious = bool(distance > threshold)
        scores.append(
            {
                **row.to_dict(),
                "dubious_l1_score": distance,
                "dubious_threshold": float(threshold) if threshold is not None else np.nan,
                "is_suspicious": suspicious,
            }
        )
    return pd.DataFrame(scores), {"classes": threshold_meta, "signature_columns": signature_columns}


def clean_nn_threshold(class_ref: np.ndarray, nearest_k: int, threshold_scale: float) -> tuple[float, dict[str, Any]]:
    n_rows = int(class_ref.shape[0])
    if n_rows <= 1:
        return float("inf"), {
            "reference_rows": n_rows,
            "nearest_k_used": 0,
            "mean_clean_nn_distance": None,
            "threshold": float("inf"),
            "threshold_scale": threshold_scale,
        }
    k = min(int(nearest_k), n_rows - 1)
    distances = np.sum(np.abs(class_ref[:, None, :] - class_ref[None, :, :]), axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    per_row_mean = np.mean(nearest, axis=1)
    mean_distance = float(np.mean(per_row_mean))
    threshold = mean_distance * threshold_scale
    return threshold, {
        "reference_rows": n_rows,
        "nearest_k_used": k,
        "mean_clean_nn_distance": mean_distance,
        "threshold": threshold,
        "threshold_scale": threshold_scale,
    }


def compute_metrics(scores_df: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for group, metric_name in [
        ("clean_benign", "clean_benign_fpr"),
        ("clean_malware", "clean_malware_fpr"),
        ("watermarked_malware", "watermarked_detection_rate"),
    ]:
        group_df = scores_df[scores_df["group"] == group]
        metrics[f"{group}_rows"] = int(group_df.shape[0])
        metrics[metric_name] = float(group_df["is_suspicious"].mean()) if not group_df.empty else None

    clean_fprs = [
        value
        for value in [metrics.get("clean_benign_fpr"), metrics.get("clean_malware_fpr")]
        if value is not None
    ]
    metrics["max_clean_fpr"] = float(max(clean_fprs)) if clean_fprs else None
    if metrics.get("watermarked_detection_rate") is not None and metrics.get("max_clean_fpr") is not None:
        metrics["detection_minus_max_fpr"] = (
            metrics["watermarked_detection_rate"] - metrics["max_clean_fpr"]
        )
    else:
        metrics["detection_minus_max_fpr"] = None

    watermarked = scores_df[scores_df["group"] == "watermarked_malware"]
    if not watermarked.empty:
        predicted_benign = watermarked["base_pred"].to_numpy(dtype=np.int8) == 0
        suspicious = watermarked["is_suspicious"].to_numpy(dtype=bool)
        metrics["watermarked_asr_before_rejection"] = float(np.mean(predicted_benign))
        metrics["watermarked_asr_after_rejection"] = float(np.mean(predicted_benign & ~suspicious))
        if np.any(predicted_benign):
            metrics["watermarked_predicted_benign_detection_rate"] = float(np.mean(suspicious[predicted_benign]))
        else:
            metrics["watermarked_predicted_benign_detection_rate"] = None
    else:
        metrics["watermarked_asr_before_rejection"] = None
        metrics["watermarked_asr_after_rejection"] = None
        metrics["watermarked_predicted_benign_detection_rate"] = None
    return metrics


def select_clean_reference_rows(
    model: lgb.Booster,
    X,
    y: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    y = np.asarray(y).astype(np.int8)
    probs = predict_probability(model, X)
    preds = (probs > 0.5).astype(np.int8)
    selected_indices: list[np.ndarray] = []
    selected_groups: list[np.ndarray] = []
    for class_id, group_name in [(0, "reference_benign"), (1, "reference_malware")]:
        candidate = np.flatnonzero((y == class_id) & (preds == class_id))
        if candidate.size == 0:
            candidate = np.flatnonzero(y == class_id)
        chosen = sample_indices(candidate, max_per_class, rng)
        selected_indices.append(chosen)
        selected_groups.append(np.full(chosen.shape[0], group_name, dtype=object))
    return {
        "indices": np.concatenate(selected_indices).astype(np.int64),
        "groups": np.concatenate(selected_groups).astype(object),
    }


def select_clean_eval_rows(
    model: lgb.Booster,
    X,
    y: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    y = np.asarray(y).astype(np.int8)
    selected_indices: list[np.ndarray] = []
    selected_groups: list[np.ndarray] = []
    for class_id, group_name in [(0, "clean_benign"), (1, "clean_malware")]:
        candidate = np.flatnonzero(y == class_id)
        chosen = sample_indices(candidate, max_per_class, rng)
        selected_indices.append(chosen)
        selected_groups.append(np.full(chosen.shape[0], group_name, dtype=object))
    return {
        "indices": np.concatenate(selected_indices).astype(np.int64),
        "groups": np.concatenate(selected_groups).astype(object),
    }


def select_watermarked_rows(X_watermarked, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    return sample_indices(np.arange(X_watermarked.shape[0], dtype=np.int64), max_rows, rng)


def sample_indices(indices: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= max_rows:
        return np.sort(indices)
    return np.sort(rng.choice(indices, size=max_rows, replace=False).astype(np.int64))


def build_replacement_values(X, y: np.ndarray, replacement: str, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y).astype(np.int8)
    if replacement.startswith("benign"):
        class_id = 0
    elif replacement.startswith("malware"):
        class_id = 1
    else:
        raise ValueError(f"Unsupported replacement {replacement}")
    class_indices = np.flatnonzero(y == class_id)
    if class_indices.size == 0:
        raise ValueError(f"No class {class_id} rows available for replacement={replacement}")

    if replacement.endswith("_mean"):
        return class_feature_mean(X, class_indices)

    max_rows = min(class_indices.size, 10000)
    sampled = sample_indices(class_indices, max_rows, rng)
    return np.asarray(X[sampled], dtype=np.float32)


def class_feature_mean(X, indices: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("Cannot compute a class mean from zero rows")
    total = None
    count = 0
    for start in range(0, indices.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        batch = np.asarray(X[batch_idx], dtype=np.float64)
        batch_sum = np.sum(batch, axis=0)
        if total is None:
            total = batch_sum
        else:
            total += batch_sum
        count += int(batch.shape[0])
    return np.asarray(total / count, dtype=np.float32)


def build_feature_candidates(model: lgb.Booster, X: np.ndarray, feature_mode: str, top_k: int) -> list[np.ndarray]:
    n_features = X.shape[1]
    if feature_mode == "random":
        all_features = np.arange(n_features, dtype=np.int64)
        return [all_features for _ in range(X.shape[0])]
    if feature_mode == "shap_topk":
        contribs = np.asarray(model.predict(X, pred_contrib=True))
        shap_values = contribs[:, :-1]
        k = min(int(top_k), n_features)
        top = np.argpartition(np.abs(shap_values), kth=n_features - k, axis=1)[:, -k:]
        return [np.asarray(row, dtype=np.int64) for row in top]
    raise ValueError(f"Unsupported feature_mode {feature_mode}")


def signature_feature_columns(magnitudes: list[int], score_mode: str) -> list[str]:
    if score_mode == "dubious_l1":
        columns = []
        for magnitude in magnitudes:
            tag = magnitude_tag(magnitude)
            columns.extend([f"mean_logit_m{tag}", f"std_logit_m{tag}", f"stability_m{tag}"])
        return columns
    if score_mode == "apc_l1":
        return [f"apc_m{magnitude_tag(magnitude)}" for magnitude in magnitudes]
    raise ValueError(f"Unsupported score_mode {score_mode}")


def raw_scores(model: lgb.Booster, X) -> np.ndarray:
    raw = np.asarray(model.predict(X, raw_score=True))
    if raw.ndim > 1:
        raw = raw[:, -1]
    return raw.astype(np.float64, copy=False)


def predict_probability(model: lgb.Booster, X) -> np.ndarray:
    pred = np.asarray(model.predict(X))
    if pred.ndim > 1:
        pred = pred[:, -1]
    return pred.astype(np.float64, copy=False)


def sigmoid(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-raw))


def predicted_class_logit(raw: np.ndarray, pred: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.int8)
    return np.where(pred == 1, raw, -raw)


def magnitude_tag(value: int) -> str:
    return str(int(value))


def resolve_output_dir(
    output_dir: str | Path | None,
    artifact_dir: Path,
    feature_mode: str,
    replacement: str,
    score_mode: str,
    magnitudes: list[int],
    n_perturbations: int,
    max_watermarked_eval: int,
) -> Path:
    if output_dir is not None:
        resolved = resolve_path(output_dir)
        return resolved or Path(output_dir)
    mag_tag = "m" + "-".join(magnitude_tag(value) for value in magnitudes)
    return (
        artifact_dir
        / "dubious_signatures"
        / f"{feature_mode}_{replacement}_{score_mode}_{mag_tag}_n{n_perturbations}_wm{max_watermarked_eval}"
    )


def _resolve_existing_dir(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("Expected a directory path")
    resolved = resolve_path(path)
    path_obj = resolved or Path(path)
    if not path_obj.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")
    return path_obj


def _unwrap_saved_array(value):
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value
