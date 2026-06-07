"""Isolation Forest, Spectral Signature, and HDBSCAN defenses.

These are detector stages: they score benign-labeled rows in a saved attack
artifact, write ``remove_watermarked_idx.npy``, and leave retraining to
``run_defense_retrain``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import MinMaxScaler

from ..features import ember2024_feature_utils, ember_feature_utils
from ..utils.paths import resolve_path


SUPPORTED_METHODS = {"hdbscan", "isolation_forest", "spectral_signature"}
SUPPORTED_FEATURE_MODES = {"watermark", "shap", "hybrid"}


@dataclass
class SeveriDetectorConfig:
    artifact_dir: str
    output_dir: str
    method: str
    feature_mode: str = "hybrid"
    top_k: int = 32
    contamination: str | float = "auto"
    removal_percent: float | None = None
    spectral_oracle_poison_count: bool = False
    standardize: bool = True
    batch_size: int = 8192
    max_benign_rows: int | None = None
    random_state: int = 42
    hdbscan_min_cluster_size: int | None = None
    hdbscan_min_cluster_percent: float = 0.5
    hdbscan_min_samples: int | None = None
    hdbscan_min_samples_percent: float = 0.1
    hdbscan_threshold_max_percent: float = 10.0
    hdbscan_min_keep: float = 0.2


@dataclass
class SeveriDetectorResult:
    output_dir: str
    method: str
    suspicious_scores_path: str
    selected_features_path: str
    remove_watermarked_idx_path: str
    metadata_path: str
    removed_rows: int
    removed_poisoned_rows: int | None
    removed_clean_rows: int | None
    poison_recall: float | None
    clean_false_positive_rate: float | None


def run_severi_detector_defense(
    artifact_dir: str | Path,
    method: str,
    output_dir: str | Path | None = None,
    feature_mode: str = "hybrid",
    top_k: int = 32,
    contamination: str | float = "auto",
    removal_percent: float | None = None,
    spectral_oracle_poison_count: bool = False,
    standardize: bool = True,
    batch_size: int = 8192,
    max_benign_rows: int | None = None,
    random_state: int = 42,
    hdbscan_min_cluster_size: int | None = None,
    hdbscan_min_cluster_percent: float = 0.5,
    hdbscan_min_samples: int | None = None,
    hdbscan_min_samples_percent: float = 0.1,
    hdbscan_threshold_max_percent: float = 10.0,
    hdbscan_min_keep: float = 0.2,
    overwrite: bool = False,
    dry_run: bool = False,
) -> SeveriDetectorResult | dict:
    artifact_path = _resolve_existing_dir(artifact_dir)
    validate_config(
        method=method,
        feature_mode=feature_mode,
        top_k=top_k,
        contamination=contamination,
        removal_percent=removal_percent,
        batch_size=batch_size,
        max_benign_rows=max_benign_rows,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_cluster_percent=hdbscan_min_cluster_percent,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_min_samples_percent=hdbscan_min_samples_percent,
        hdbscan_threshold_max_percent=hdbscan_threshold_max_percent,
        hdbscan_min_keep=hdbscan_min_keep,
    )
    output_path = resolve_output_dir(
        artifact_path=artifact_path,
        output_dir=output_dir,
        method=method,
        feature_mode=feature_mode,
        top_k=top_k,
        contamination=contamination,
        removal_percent=removal_percent,
        spectral_oracle_poison_count=spectral_oracle_poison_count,
        standardize=standardize,
        max_benign_rows=max_benign_rows,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_cluster_percent=hdbscan_min_cluster_percent,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_min_samples_percent=hdbscan_min_samples_percent,
        hdbscan_threshold_max_percent=hdbscan_threshold_max_percent,
        hdbscan_min_keep=hdbscan_min_keep,
    )

    required = {
        "watermarked_X": artifact_path / "watermarked_X.npy",
        "watermarked_y": artifact_path / "watermarked_y.npy",
        "wm_config": artifact_path / "wm_config.npy",
        "defense_metadata_npz": artifact_path / "defense_metadata.npz",
        "defense_metadata_json": artifact_path / "defense_metadata.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Severi-detector artifact(s): {', '.join(missing)}")

    if dry_run:
        return {
            "artifact_dir": str(artifact_path),
            "output_dir": str(output_path),
            "method": method,
            "feature_mode": feature_mode,
            "top_k": top_k,
            "contamination": contamination,
            "removal_percent": removal_percent,
            "spectral_oracle_poison_count": spectral_oracle_poison_count,
            "standardize": standardize,
            "batch_size": batch_size,
            "max_benign_rows": max_benign_rows,
            "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
            "hdbscan_min_cluster_percent": hdbscan_min_cluster_percent,
            "hdbscan_min_samples": hdbscan_min_samples,
            "hdbscan_min_samples_percent": hdbscan_min_samples_percent,
            "hdbscan_threshold_max_percent": hdbscan_threshold_max_percent,
            "hdbscan_min_keep": hdbscan_min_keep,
            "required_paths": {key: str(path) for key, path in required.items()},
        }

    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "severi_detector_metadata.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_path} already exists. Pass --overwrite to replace it.")

    start_time = time.time()
    X_all = load_saved_array(required["watermarked_X"], mmap_mode="r")
    y_all = np.asarray(load_saved_array(required["watermarked_y"], mmap_mode="r")).reshape(-1)
    wm_config = np.load(required["wm_config"], allow_pickle=True).item()
    meta = np.load(required["defense_metadata_npz"])
    metadata_json = json.loads(required["defense_metadata_json"].read_text(encoding="utf-8"))

    if X_all.shape[0] != y_all.shape[0]:
        raise ValueError(f"watermarked_X rows {X_all.shape[0]} do not match watermarked_y rows {y_all.shape[0]}")

    benign_idx, poison_mask_benign = load_benign_alignment(meta, y_all)
    if max_benign_rows is not None:
        row_limit = min(int(max_benign_rows), benign_idx.shape[0])
        benign_idx = benign_idx[:row_limit]
        poison_mask_benign = poison_mask_benign[:row_limit]

    feature_names = feature_names_for_width(X_all.shape[1])
    selected_features = select_defense_feature_indices(
        artifact_path=artifact_path,
        wm_config=wm_config,
        feature_mode=feature_mode,
        top_k=top_k,
        n_features=X_all.shape[1],
    )
    X_selected = selected_dense_rows(
        X_all,
        row_indices=benign_idx,
        feature_indices=selected_features,
        batch_size=batch_size,
    )
    if standardize:
        X_detector = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X_selected)
    else:
        X_detector = X_selected

    model_path = None
    extra_score_columns: dict[str, np.ndarray] = {}
    extra_output_files: dict[str, str | None] = {}
    detector_details: dict[str, Any] = {}
    hdbscan_result: dict[str, Any] | None = None
    hdbscan_cluster_summary_path: Path | None = None
    if method == "isolation_forest":
        scores, default_remove_mask, fitted_model = isolation_forest_scores(
            X_detector,
            contamination=contamination,
            random_state=random_state,
        )
        model_path = output_path / "isolation_forest.joblib"
        joblib.dump(fitted_model, model_path)
        detector_details = {"contamination": contamination}
    elif method == "spectral_signature":
        scores = spectral_signature_scores(X_detector)
        default_remove_mask = spectral_default_remove_mask(
            scores=scores,
            poison_mask_benign=poison_mask_benign,
            spectral_oracle_poison_count=spectral_oracle_poison_count,
            removal_percent=removal_percent,
        )
        detector_details = {
            "spectral_oracle_poison_count": spectral_oracle_poison_count,
            "default_removal_percent": None if spectral_oracle_poison_count else removal_percent or 1.0,
        }
    elif method == "hdbscan":
        hdbscan_result = hdbscan_severi_scores_and_mask(
            X_detector,
            poison_mask_benign=poison_mask_benign,
            total_train_rows=int(X_all.shape[0]),
            min_cluster_size=hdbscan_min_cluster_size,
            min_cluster_percent=hdbscan_min_cluster_percent,
            min_samples=hdbscan_min_samples,
            min_samples_percent=hdbscan_min_samples_percent,
            threshold_max_percent=hdbscan_threshold_max_percent,
            min_keep=hdbscan_min_keep,
            random_state=random_state,
        )
        scores = hdbscan_result["scores"]
        default_remove_mask = hdbscan_result["remove_mask"]
        fitted_model = hdbscan_result["model"]
        model_path = output_path / "hdbscan.joblib"
        joblib.dump(fitted_model, model_path)
        labels_path = output_path / "hdbscan_labels.npy"
        hdbscan_cluster_summary_path = output_path / "hdbscan_cluster_summary.csv"
        np.save(labels_path, hdbscan_result["labels"])
        extra_score_columns = {
            "hdbscan_label": hdbscan_result["labels"].astype(np.int64, copy=False),
            "cluster_avg_silhouette": hdbscan_result["cluster_avg_silhouette"],
            "keep_probability": hdbscan_result["keep_probability"],
            "hdbscan_probability": hdbscan_result["probabilities"],
            "hdbscan_outlier_score": hdbscan_result["outlier_scores"],
        }
        extra_output_files = {
            "hdbscan_labels": str(labels_path),
            "hdbscan_cluster_summary": str(hdbscan_cluster_summary_path),
        }
        detector_details = hdbscan_result["details"]
    else:
        raise ValueError(f"Unsupported method: {method}")

    if removal_percent is not None and not (method == "spectral_signature" and spectral_oracle_poison_count):
        remove_mask = top_percent_mask(scores, removal_percent)
        threshold_mode = "top_percent"
    else:
        remove_mask = default_remove_mask
        threshold_mode = "detector_default"

    remove_watermarked_idx = benign_idx[remove_mask].astype(np.int64, copy=False)
    remove_positions = np.flatnonzero(remove_mask).astype(np.int64, copy=False)
    metrics = removal_metrics(poison_mask_benign=poison_mask_benign, remove_mask=remove_mask)

    if method == "hdbscan" and hdbscan_result is not None and hdbscan_cluster_summary_path is not None:
        build_hdbscan_cluster_summary(
            labels=hdbscan_result["labels"],
            scores=scores,
            remove_mask=remove_mask,
            poison_mask_benign=poison_mask_benign,
            cluster_avg_silhouette=hdbscan_result["cluster_avg_silhouette_by_label"],
            probabilities=hdbscan_result["probabilities"],
            outlier_scores=hdbscan_result["outlier_scores"],
        ).to_csv(hdbscan_cluster_summary_path, index=False)

    scores_path = output_path / "suspicious_scores.csv"
    selected_features_path = output_path / "selected_features.csv"
    remove_path = output_path / "remove_watermarked_idx.npy"
    remove_positions_path = output_path / "remove_benign_positions.npy"

    build_scores_df(
        benign_idx=benign_idx,
        poison_mask_benign=poison_mask_benign,
        scores=scores,
        remove_mask=remove_mask,
        extra_columns=extra_score_columns,
    ).to_csv(scores_path, index=False)
    build_selected_features_df(
        selected_features=selected_features,
        feature_names=feature_names,
        wm_config=wm_config,
    ).to_csv(selected_features_path, index=False)
    np.save(remove_path, remove_watermarked_idx)
    np.save(remove_positions_path, remove_positions)

    config = SeveriDetectorConfig(
        artifact_dir=str(artifact_path),
        output_dir=str(output_path),
        method=method,
        feature_mode=feature_mode,
        top_k=top_k,
        contamination=contamination,
        removal_percent=removal_percent,
        spectral_oracle_poison_count=spectral_oracle_poison_count,
        standardize=standardize,
        batch_size=batch_size,
        max_benign_rows=max_benign_rows,
        random_state=random_state,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_cluster_percent=hdbscan_min_cluster_percent,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_min_samples_percent=hdbscan_min_samples_percent,
        hdbscan_threshold_max_percent=hdbscan_threshold_max_percent,
        hdbscan_min_keep=hdbscan_min_keep,
    )
    metadata = {
        "method_note": (
            "Notebook-style Severi detector port. Isolation Forest and Spectral "
            "Signature follow the notebook helpers; HDBSCAN follows the original "
            "defense_filtering.py clustering filter with selected benign features, "
            "MinMax scaling, cluster silhouettes, and a min_keep sampling rule. "
            "Ground-truth poison labels are used only for diagnostics unless "
            "spectral_oracle_poison_count is true."
        ),
        "config": asdict(config),
        "detector_details": detector_details,
        "threshold_mode": threshold_mode,
        "input_shape": {
            "watermarked_X": [int(X_all.shape[0]), int(X_all.shape[1])],
            "benign_rows_scored": int(benign_idx.shape[0]),
            "selected_features": int(selected_features.shape[0]),
        },
        "artifact_metadata": {
            "dataset": metadata_json.get("dataset"),
            "model_id": metadata_json.get("model_id"),
            "num_poisoned_rows": metadata_json.get("num_poisoned_rows"),
            "row_order": metadata_json.get("row_order"),
        },
        "removal_metrics": metrics,
        "output_files": {
            "suspicious_scores": str(scores_path),
            "selected_features": str(selected_features_path),
            "remove_watermarked_idx": str(remove_path),
            "remove_benign_positions": str(remove_positions_path),
            "model": str(model_path) if model_path is not None else None,
            **extra_output_files,
        },
        "runtime_seconds": time.time() - start_time,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return SeveriDetectorResult(
        output_dir=str(output_path),
        method=method,
        suspicious_scores_path=str(scores_path),
        selected_features_path=str(selected_features_path),
        remove_watermarked_idx_path=str(remove_path),
        metadata_path=str(metadata_path),
        removed_rows=int(metrics["removed_rows"]),
        removed_poisoned_rows=metrics.get("removed_poisoned_rows"),
        removed_clean_rows=metrics.get("removed_clean_rows"),
        poison_recall=metrics.get("poison_recall"),
        clean_false_positive_rate=metrics.get("clean_false_positive_rate"),
    )


def validate_config(
    *,
    method: str,
    feature_mode: str,
    top_k: int,
    contamination: str | float,
    removal_percent: float | None,
    batch_size: int,
    max_benign_rows: int | None,
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_cluster_percent: float,
    hdbscan_min_samples: int | None,
    hdbscan_min_samples_percent: float,
    hdbscan_threshold_max_percent: float,
    hdbscan_min_keep: float,
) -> None:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {sorted(SUPPORTED_METHODS)}, got {method!r}")
    if feature_mode not in SUPPORTED_FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {sorted(SUPPORTED_FEATURE_MODES)}, got {feature_mode!r}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if isinstance(contamination, float) and not 0 < contamination <= 0.5:
        raise ValueError("numeric contamination must be in (0, 0.5]")
    if removal_percent is not None and not 0 < removal_percent <= 100:
        raise ValueError("removal_percent must be in (0, 100]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_benign_rows is not None and max_benign_rows <= 0:
        raise ValueError("max_benign_rows must be positive")
    if hdbscan_min_cluster_size is not None and hdbscan_min_cluster_size < 2:
        raise ValueError("hdbscan_min_cluster_size must be at least 2")
    if not 0 < hdbscan_min_cluster_percent <= 100:
        raise ValueError("hdbscan_min_cluster_percent must be in (0, 100]")
    if hdbscan_min_samples is not None and hdbscan_min_samples < 1:
        raise ValueError("hdbscan_min_samples must be positive")
    if not 0 < hdbscan_min_samples_percent <= 100:
        raise ValueError("hdbscan_min_samples_percent must be in (0, 100]")
    if not 0 < hdbscan_threshold_max_percent <= 100:
        raise ValueError("hdbscan_threshold_max_percent must be in (0, 100]")
    if not 0 <= hdbscan_min_keep <= 1:
        raise ValueError("hdbscan_min_keep must be in [0, 1]")


def select_defense_feature_indices(
    *,
    artifact_path: Path,
    wm_config: dict[str, Any],
    feature_mode: str,
    top_k: int,
    n_features: int,
) -> np.ndarray:
    watermark_ids = valid_feature_ids(wm_config.get("wm_feat_ids", []), n_features=n_features)
    if feature_mode == "watermark":
        if watermark_ids.size == 0:
            raise ValueError("feature_mode='watermark' requires wm_feat_ids in wm_config.npy")
        return unique_limited(watermark_ids, limit=top_k)
    if feature_mode == "hybrid" and watermark_ids.size >= top_k:
        return unique_limited(watermark_ids, limit=top_k)

    shap_ranked = np.asarray([], dtype=np.int64)
    shap_path = artifact_path / "backdoored_model_benign_shap.npy"
    if shap_path.exists():
        shap_ranked = top_mean_abs_shap_features(shap_path, n_features=n_features)
    elif feature_mode == "shap":
        raise FileNotFoundError(f"feature_mode='shap' requires {shap_path}")

    if feature_mode == "shap":
        return unique_limited(shap_ranked, limit=top_k)
    if watermark_ids.size == 0 and shap_ranked.size == 0:
        raise ValueError("feature_mode='hybrid' requires wm_feat_ids or backdoored_model_benign_shap.npy")
    return unique_limited(np.concatenate([watermark_ids, shap_ranked]), limit=top_k)


def top_mean_abs_shap_features(shap_path: Path, n_features: int, batch_size: int = 8192) -> np.ndarray:
    X = np.load(shap_path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D SHAP matrix at {shap_path}, got {X.shape}")
    if X.shape[1] != n_features:
        raise ValueError(f"SHAP width {X.shape[1]} does not match watermarked_X width {n_features}")
    sums = np.zeros(X.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, X.shape[0], batch_size):
        batch = np.asarray(X[start : start + batch_size], dtype=np.float32)
        sums += np.sum(np.abs(batch), axis=0)
        count += batch.shape[0]
    if count == 0:
        raise ValueError(f"SHAP matrix at {shap_path} is empty")
    return np.argsort(sums / count)[::-1].astype(np.int64, copy=False)


def valid_feature_ids(values: Any, n_features: int) -> np.ndarray:
    ids = []
    if values is None:
        values = []
    for value in values:
        feature_id = int(value)
        if 0 <= feature_id < n_features:
            ids.append(feature_id)
    return np.asarray(ids, dtype=np.int64)


def unique_limited(values: np.ndarray, limit: int) -> np.ndarray:
    selected: list[int] = []
    seen: set[int] = set()
    for raw_value in values:
        value = int(raw_value)
        if value not in seen:
            selected.append(value)
            seen.add(value)
        if len(selected) >= limit:
            break
    return np.asarray(selected, dtype=np.int64)


def selected_dense_rows(
    X: Any,
    *,
    row_indices: np.ndarray,
    feature_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    try:
        import scipy.sparse
    except ImportError:
        scipy = None
    else:
        scipy = scipy.sparse

    if scipy is not None and scipy.issparse(X):
        return X[row_indices][:, feature_indices].toarray().astype(np.float32, copy=False)

    output = np.empty((row_indices.shape[0], feature_indices.shape[0]), dtype=np.float32)
    for start in range(0, row_indices.shape[0], batch_size):
        end = min(start + batch_size, row_indices.shape[0])
        rows = row_indices[start:end]
        output[start:end] = np.asarray(X[np.ix_(rows, feature_indices)], dtype=np.float32)
    return output


def isolation_forest_scores(
    X: np.ndarray,
    *,
    contamination: str | float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, IsolationForest]:
    model = IsolationForest(
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    predictions = model.fit_predict(X)
    scores = -np.asarray(model.score_samples(X), dtype=np.float64)
    remove_mask = predictions == -1
    return scores, remove_mask, model


def spectral_signature_scores(X: np.ndarray) -> np.ndarray:
    centered = X - np.mean(X, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    top_vector = vh[0]
    return np.square(centered @ top_vector).astype(np.float64, copy=False)


def spectral_default_remove_mask(
    *,
    scores: np.ndarray,
    poison_mask_benign: np.ndarray,
    spectral_oracle_poison_count: bool,
    removal_percent: float | None,
) -> np.ndarray:
    if spectral_oracle_poison_count:
        remove_count = max(1, int(np.sum(poison_mask_benign)))
    else:
        percent = 1.0 if removal_percent is None else removal_percent
        remove_count = max(1, int(np.ceil(scores.shape[0] * percent / 100.0)))
    remove_count = min(remove_count, scores.shape[0])
    remove_idx = np.argsort(scores)[-remove_count:]
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[remove_idx] = True
    return mask


def hdbscan_severi_scores_and_mask(
    X: np.ndarray,
    *,
    poison_mask_benign: np.ndarray,
    total_train_rows: int,
    min_cluster_size: int | None,
    min_cluster_percent: float,
    min_samples: int | None,
    min_samples_percent: float,
    threshold_max_percent: float,
    min_keep: float,
    random_state: int,
) -> dict[str, Any]:
    """Run the HDBSCAN clustering filter used by Severi's EMBER defense.

    The original ``defense_filtering.py`` flow clusters selected goodware
    features, computes average silhouette per cluster, and keeps each point with
    probability ``(1 - normalized_cluster_silhouette) + min_keep`` for clusters
    smaller than ``t_max``. Here we expose the complement of that keep
    probability as a suspicious score and save the generated removal mask.
    """
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN defense requires the optional 'hdbscan' package. Install it "
            "in this environment, for example with `python3 -m pip install hdbscan`, "
            "then rerun this command."
        ) from exc

    n_rows = int(X.shape[0])
    count_base_rows = int(total_train_rows)
    resolved_min_cluster_size = resolve_count(
        explicit=min_cluster_size,
        percent=min_cluster_percent,
        total=count_base_rows,
        minimum=2,
    )
    resolved_min_samples = resolve_count(
        explicit=min_samples,
        percent=min_samples_percent,
        total=count_base_rows,
        minimum=1,
    )
    threshold_max_size = resolve_count(
        explicit=None,
        percent=threshold_max_percent,
        total=count_base_rows,
        minimum=1,
    )

    model = hdbscan.HDBSCAN(
        metric="euclidean",
        core_dist_n_jobs=-1,
        min_cluster_size=resolved_min_cluster_size,
        min_samples=resolved_min_samples,
    )
    labels = model.fit_predict(X)
    cluster_sizes = label_size_map(labels)

    silhouettes, cluster_avg_silhouette, silhouette_status = safe_cluster_silhouettes(X, labels)
    expanded_silhouette = np.asarray(
        [
            cluster_avg_silhouette.get(int(label), np.nan)
            if cluster_sizes[int(label)] <= threshold_max_size
            else -1.0
            for label in labels
        ],
        dtype=np.float64,
    )

    if silhouette_status == "computed":
        scaled_silhouette = minmax_vector(expanded_silhouette, feature_range=(0.0, 1.0))
        keep_probability = np.clip((1.0 - scaled_silhouette) + float(min_keep), 0.0, 1.0)
        scores = 1.0 - keep_probability
        rng = np.random.RandomState(random_state)
        remove_mask = keep_probability < rng.random_sample(n_rows)
        score_mode = "severi_silhouette_min_keep"
    else:
        scores = hdbscan_fallback_scores(
            labels=labels,
            cluster_sizes=cluster_sizes,
            threshold_max_size=threshold_max_size,
            probabilities=getattr(model, "probabilities_", None),
            outlier_scores=getattr(model, "outlier_scores_", None),
        )
        remove_mask = labels == -1
        keep_probability = 1.0 - scores
        score_mode = f"fallback_{silhouette_status}"

    probabilities = np.asarray(getattr(model, "probabilities_", np.ones(n_rows)), dtype=np.float64)
    outlier_scores = np.asarray(getattr(model, "outlier_scores_", np.zeros(n_rows)), dtype=np.float64)

    return {
        "model": model,
        "labels": labels.astype(np.int64, copy=False),
        "scores": scores.astype(np.float64, copy=False),
        "remove_mask": remove_mask.astype(bool, copy=False),
        "cluster_avg_silhouette": expanded_silhouette,
        "cluster_avg_silhouette_by_label": cluster_avg_silhouette,
        "keep_probability": keep_probability.astype(np.float64, copy=False),
        "probabilities": probabilities,
        "outlier_scores": outlier_scores,
        "details": {
            "score_mode": score_mode,
            "silhouette_status": silhouette_status,
            "min_cluster_size": int(resolved_min_cluster_size),
            "min_cluster_percent": float(min_cluster_percent),
            "min_samples": int(resolved_min_samples),
            "min_samples_percent": float(min_samples_percent),
            "count_base_rows": int(count_base_rows),
            "threshold_max_size": int(threshold_max_size),
            "threshold_max_percent": float(threshold_max_percent),
            "min_keep": float(min_keep),
            "cluster_count_including_noise": int(len(cluster_sizes)),
            "noise_rows": int(np.sum(labels == -1)),
        },
    }


def resolve_count(
    *,
    explicit: int | None,
    percent: float,
    total: int,
    minimum: int,
) -> int:
    if explicit is not None:
        return max(int(explicit), minimum)
    return max(int(np.ceil(total * float(percent) / 100.0)), minimum)


def label_size_map(labels: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(labels.astype(np.int64, copy=False), return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def safe_cluster_silhouettes(
    X: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[int, float], str]:
    labels = labels.astype(np.int64, copy=False)
    unique_labels = np.unique(labels)
    if unique_labels.shape[0] < 2:
        empty = np.full(labels.shape[0], np.nan, dtype=np.float64)
        return empty, {int(unique_labels[0]): np.nan} if unique_labels.size else {}, "one_cluster"
    if unique_labels.shape[0] >= labels.shape[0]:
        empty = np.full(labels.shape[0], np.nan, dtype=np.float64)
        return empty, {int(label): np.nan for label in unique_labels}, "too_many_clusters"

    silhouettes = silhouette_samples(X, labels, metric="euclidean")
    cluster_avg = {
        int(label): float(np.mean(silhouettes[labels == label]))
        for label in unique_labels
    }
    return silhouettes.astype(np.float64, copy=False), cluster_avg, "computed"


def minmax_vector(values: np.ndarray, feature_range: tuple[float, float]) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float64)
    clean = values.copy()
    clean[~finite] = np.nanmin(clean[finite])
    min_value = float(np.min(clean))
    max_value = float(np.max(clean))
    low, high = feature_range
    if max_value == min_value:
        return np.full(clean.shape, low, dtype=np.float64)
    scaled = (clean - min_value) / (max_value - min_value)
    return scaled * (high - low) + low


def hdbscan_fallback_scores(
    *,
    labels: np.ndarray,
    cluster_sizes: dict[int, int],
    threshold_max_size: int,
    probabilities: np.ndarray | None,
    outlier_scores: np.ndarray | None,
) -> np.ndarray:
    labels = labels.astype(np.int64, copy=False)
    scores = np.zeros(labels.shape[0], dtype=np.float64)
    for label, size in cluster_sizes.items():
        mask = labels == label
        if label == -1:
            scores[mask] = 1.0
        elif size <= threshold_max_size:
            scores[mask] = max(0.0, 1.0 - (size / max(threshold_max_size, 1)))
    if probabilities is not None:
        scores = np.maximum(scores, 1.0 - np.asarray(probabilities, dtype=np.float64))
    if outlier_scores is not None:
        scores = np.maximum(scores, minmax_vector(np.asarray(outlier_scores, dtype=np.float64), (0.0, 1.0)))
    return scores


def top_percent_mask(scores: np.ndarray, percent: float) -> np.ndarray:
    remove_count = max(1, int(np.ceil(scores.shape[0] * percent / 100.0)))
    remove_count = min(remove_count, scores.shape[0])
    idx = np.argsort(scores)[-remove_count:]
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[idx] = True
    return mask


def build_scores_df(
    *,
    benign_idx: np.ndarray,
    poison_mask_benign: np.ndarray,
    scores: np.ndarray,
    remove_mask: np.ndarray,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    data = {
        "benign_position": np.arange(benign_idx.shape[0], dtype=np.int64),
        "watermarked_idx": benign_idx.astype(np.int64, copy=False),
        "is_poisoned": poison_mask_benign.astype(bool, copy=False),
        "suspicious_score": scores,
        "remove": remove_mask.astype(bool, copy=False),
    }
    if extra_columns:
        for name, values in extra_columns.items():
            data[name] = values
    df = pd.DataFrame(data)
    return df.sort_values("suspicious_score", ascending=False).reset_index(drop=True)


def build_hdbscan_cluster_summary(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    remove_mask: np.ndarray,
    poison_mask_benign: np.ndarray,
    cluster_avg_silhouette: dict[int, float],
    probabilities: np.ndarray,
    outlier_scores: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for label in sorted(np.unique(labels).astype(np.int64).tolist()):
        mask = labels == label
        removed = remove_mask & mask
        poisoned = poison_mask_benign & mask
        rows.append(
            {
                "cluster_label": int(label),
                "is_noise": bool(label == -1),
                "rows": int(np.sum(mask)),
                "row_fraction": float(np.mean(mask)),
                "removed_rows": int(np.sum(removed)),
                "removed_poisoned_rows": int(np.sum(removed & poison_mask_benign)),
                "removed_clean_rows": int(np.sum(removed & ~poison_mask_benign)),
                "poisoned_rows": int(np.sum(poisoned)),
                "poison_fraction_within_cluster": float(np.mean(poison_mask_benign[mask])) if np.any(mask) else np.nan,
                "mean_suspicious_score": float(np.mean(scores[mask])) if np.any(mask) else np.nan,
                "mean_hdbscan_probability": float(np.mean(probabilities[mask])) if np.any(mask) else np.nan,
                "mean_hdbscan_outlier_score": float(np.mean(outlier_scores[mask])) if np.any(mask) else np.nan,
                "avg_silhouette": cluster_avg_silhouette.get(int(label), np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["removed_poisoned_rows", "mean_suspicious_score", "rows"],
        ascending=[False, False, False],
    )


def build_selected_features_df(
    *,
    selected_features: np.ndarray,
    feature_names: list[str],
    wm_config: dict[str, Any],
) -> pd.DataFrame:
    watermark_feature_ids = {int(v) for v in wm_config.get("wm_feat_ids", [])}
    watermark_values = wm_config.get("watermark_features", {}) or {}
    name_to_value = {str(name): value for name, value in watermark_values.items()}
    rows = []
    for rank, feature_id in enumerate(selected_features, start=1):
        feature_name = feature_names[feature_id] if feature_id < len(feature_names) else f"feature_{feature_id}"
        rows.append(
            {
                "rank": rank,
                "feature_id": int(feature_id),
                "feature_name": feature_name,
                "is_watermark_feature": int(feature_id) in watermark_feature_ids,
                "watermark_value": name_to_value.get(feature_name, np.nan),
            }
        )
    return pd.DataFrame(rows)


def load_benign_alignment(meta: np.lib.npyio.NpzFile, y_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "benign_watermarked_idx" in meta.files:
        benign_idx = np.asarray(meta["benign_watermarked_idx"], dtype=np.int64).reshape(-1)
    else:
        benign_idx = np.flatnonzero(np.asarray(y_all) == 0).astype(np.int64, copy=False)

    if "poison_mask_benign" in meta.files:
        poison_mask = np.asarray(meta["poison_mask_benign"], dtype=bool).reshape(-1)
    elif "poison_mask_full" in meta.files:
        poison_mask = np.asarray(meta["poison_mask_full"], dtype=bool).reshape(-1)[benign_idx]
    else:
        poison_mask = np.zeros(benign_idx.shape[0], dtype=bool)

    if poison_mask.shape[0] != benign_idx.shape[0]:
        raise ValueError(
            f"poison_mask_benign length {poison_mask.shape[0]} does not match benign rows {benign_idx.shape[0]}"
        )
    return benign_idx, poison_mask


def removal_metrics(poison_mask_benign: np.ndarray, remove_mask: np.ndarray) -> dict[str, int | float | None]:
    removed_poisoned = int(np.sum(remove_mask & poison_mask_benign))
    removed_clean = int(np.sum(remove_mask & ~poison_mask_benign))
    total_poisoned = int(np.sum(poison_mask_benign))
    total_clean = int(poison_mask_benign.shape[0] - total_poisoned)
    return {
        "total_poisoned_rows": total_poisoned,
        "total_clean_rows": total_clean,
        "removed_rows": int(np.sum(remove_mask)),
        "removed_poisoned_rows": removed_poisoned,
        "removed_clean_rows": removed_clean,
        "poison_recall": float(removed_poisoned / total_poisoned) if total_poisoned else None,
        "clean_false_positive_rate": float(removed_clean / total_clean) if total_clean else None,
    }


def feature_names_for_width(n_features: int) -> list[str]:
    if n_features == ember_feature_utils.NUM_EMBER_FEATURES:
        return ember_feature_utils.build_feature_names()
    if ember_feature_utils.NUM_EMBER_FEATURES < n_features < ember2024_feature_utils.NUM_EMBER2024_FEATURES:
        names = ember_feature_utils.build_feature_names()
        names.extend(f"feature_{i}" for i in range(len(names), n_features))
        return names
    if n_features == ember2024_feature_utils.NUM_EMBER2024_FEATURES:
        return ember2024_feature_utils.build_feature_names()
    return [f"feature_{i}" for i in range(n_features)]


def load_saved_array(path: Path, mmap_mode: str | None = None) -> Any:
    value = np.load(path, mmap_mode=mmap_mode, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value


def resolve_output_dir(
    *,
    artifact_path: Path,
    output_dir: str | Path | None,
    method: str,
    feature_mode: str,
    top_k: int,
    contamination: str | float,
    removal_percent: float | None,
    spectral_oracle_poison_count: bool,
    standardize: bool,
    max_benign_rows: int | None,
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_cluster_percent: float,
    hdbscan_min_samples: int | None,
    hdbscan_min_samples_percent: float,
    hdbscan_threshold_max_percent: float,
    hdbscan_min_keep: float,
) -> Path:
    if output_dir is not None:
        resolved = resolve_path(output_dir)
        return resolved or Path(output_dir)
    scale_tag = "scaled" if standardize else "raw"
    row_tag = f"_rows{max_benign_rows}" if max_benign_rows is not None else ""
    if method == "isolation_forest":
        if removal_percent is None:
            threshold_tag = f"contam{format_tag(contamination)}"
        else:
            threshold_tag = f"remove{format_tag(removal_percent)}p"
    elif method == "spectral_signature":
        threshold_tag = "oracle_poison_count" if spectral_oracle_poison_count else f"remove{format_tag(removal_percent or 1.0)}p"
    elif method == "hdbscan":
        mcs_tag = count_or_percent_tag(hdbscan_min_cluster_size, hdbscan_min_cluster_percent)
        ms_tag = count_or_percent_tag(hdbscan_min_samples, hdbscan_min_samples_percent)
        base_tag = (
            f"mcs{mcs_tag}_ms{ms_tag}_"
            f"tmax{format_tag(hdbscan_threshold_max_percent)}pct_"
            f"keep{format_tag(hdbscan_min_keep)}"
        )
        threshold_tag = f"{base_tag}_remove{format_tag(removal_percent)}p" if removal_percent is not None else base_tag
    else:
        raise ValueError(f"Unsupported method: {method}")
    dirname = f"{method}_{feature_mode}_top{top_k}_{scale_tag}_{threshold_tag}{row_tag}"
    return artifact_path / "severi_detectors" / dirname


def count_or_percent_tag(count: int | None, percent: float) -> str:
    if count is not None:
        return str(count)
    return f"{format_tag(percent)}pct"


def format_tag(value: str | float) -> str:
    if isinstance(value, str):
        return value.replace(".", "p")
    return f"{value:g}".replace(".", "p")


def _resolve_existing_dir(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("Expected a directory path")
    resolved = resolve_path(path)
    path_obj = resolved or Path(path)
    if not path_obj.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")
    return path_obj
