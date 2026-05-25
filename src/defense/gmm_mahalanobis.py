"""GMM-BIC and Mahalanobis scoring for preprocessed benign SHAP rows."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from ..utils.paths import resolve_path


@dataclass
class GmmDefenseConfig:
    preprocess_dir: str
    artifact_dir: str
    output_dir: str
    k_min: int = 1
    k_max: int = 20
    covariance_types: tuple[str, ...] = ("diag",)
    reg_covar: float = 1e-6
    removal_percent: float = 1.0
    score_name: str = "local_z"
    random_state: int = 42
    n_init: int = 3
    max_iter: int = 200
    max_rows: int | None = None


@dataclass
class GmmDefenseResult:
    output_dir: str
    bic_scores_path: str
    suspicious_scores_path: str
    component_summary_path: str
    component_geometry_path: str
    metadata_path: str
    best_gmm_path: str
    global_gmm_path: str
    best_n_components: int
    best_covariance_type: str
    best_bic: float
    removed_rows: int
    poison_recall: float | None
    clean_false_positive_rate: float | None


def run_gmm_defense(
    preprocess_dir: str | Path,
    artifact_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    k_min: int = 1,
    k_max: int = 10,
    covariance_types: Iterable[str] = ("diag",),
    reg_covar: float = 1e-6,
    removal_percent: float = 1.0,
    score_name: str = "local_z",
    random_state: int = 42,
    n_init: int = 3,
    max_iter: int = 200,
    max_rows: int | None = None,
    overwrite: bool = False,
) -> GmmDefenseResult:
    preprocess_path = _resolve_existing_dir(preprocess_dir)
    preprocess_metadata_path = preprocess_path / "preprocessing_metadata.json"
    X_path = preprocess_path / "X_shap_reduced.npy"
    if not X_path.exists():
        raise FileNotFoundError(f"Missing preprocessed SHAP matrix: {X_path}")
    preprocess_metadata = load_json(preprocess_metadata_path)

    if artifact_dir is None:
        artifact_dir = preprocess_metadata.get("config", {}).get("artifact_dir")
    artifact_path = _resolve_existing_dir(artifact_dir)
    defense_metadata_path = artifact_path / "defense_metadata.npz"
    if not defense_metadata_path.exists():
        raise FileNotFoundError(f"Missing defense metadata: {defense_metadata_path}")

    covariance_types = tuple(covariance_types)
    validate_config(k_min, k_max, covariance_types, reg_covar, removal_percent, score_name)

    output_path = resolve_output_dir(
        preprocess_path,
        output_dir=output_dir,
        covariance_types=covariance_types,
        k_min=k_min,
        k_max=k_max,
        reg_covar=reg_covar,
        removal_percent=removal_percent,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_out_path = output_path / "gmm_defense_metadata.json"
    if metadata_out_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_out_path} already exists. Pass --overwrite to replace it.")

    start_time = time.time()
    X = np.load(X_path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D X_shap_reduced, got shape {X.shape}")
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        X = X[: min(max_rows, X.shape[0])]
    X_fit = np.asarray(X, dtype=np.float64)

    meta = np.load(defense_metadata_path)
    aligned = load_aligned_metadata(meta, n_rows=X_fit.shape[0])

    global_gmm = fit_gmm(
        X_fit,
        n_components=1,
        covariance_type="diag",
        reg_covar=reg_covar,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    global_labels = global_gmm.predict(X_fit)
    global_scores = component_mahalanobis_scores(X_fit, global_gmm, global_labels)
    global_z = zscore(global_scores)

    best_gmm, bic_df = select_best_gmm(
        X_fit,
        k_min=k_min,
        k_max=k_max,
        covariance_types=covariance_types,
        reg_covar=reg_covar,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    local_labels = best_gmm.predict(X_fit)
    local_scores = component_mahalanobis_scores(X_fit, best_gmm, local_labels)
    local_z = clusterwise_zscore(local_scores, local_labels)
    local_global_z = zscore(local_scores)
    diagnostics = compute_gmm_row_diagnostics(X_fit, best_gmm, local_labels)

    score_values = {
        "local_z": local_z,
        "local_mahalanobis": local_scores,
        "local_global_z": local_global_z,
        "global_z": global_z,
        "global_mahalanobis": global_scores,
    }[score_name]
    removed_positions = top_percent_indices(score_values, removal_percent)
    remove_mask = np.zeros(X_fit.shape[0], dtype=bool)
    remove_mask[removed_positions] = True

    scores_df = build_scores_df(
        aligned=aligned,
        local_labels=local_labels,
        local_scores=local_scores,
        local_z=local_z,
        local_global_z=local_global_z,
        global_scores=global_scores,
        global_z=global_z,
        log_likelihood=diagnostics["log_likelihood"],
        responsibility_assigned=diagnostics["responsibility_assigned"],
        responsibility_confidence=diagnostics["responsibility_confidence"],
        responsibility_entropy=diagnostics["responsibility_entropy"],
        remove_mask=remove_mask,
    )
    component_geometry_df = build_component_geometry(
        scores_df=scores_df,
        gmm=best_gmm,
        global_gmm=global_gmm,
    )
    component_df = build_component_summary(scores_df).merge(
        component_geometry_df.drop(columns=["rows"], errors="ignore"),
        on="component",
        how="left",
    )
    metrics = removal_metrics(scores_df)

    bic_scores_path = output_path / "bic_scores.csv"
    suspicious_scores_path = output_path / "suspicious_scores.csv"
    component_summary_path = output_path / "component_summary.csv"
    component_geometry_path = output_path / "component_geometry.csv"
    best_gmm_path = output_path / "best_local_gmm.joblib"
    global_gmm_path = output_path / "global_gmm.joblib"

    bic_df.to_csv(bic_scores_path, index=False)
    scores_df.to_csv(suspicious_scores_path, index=False)
    component_df.to_csv(component_summary_path, index=False)
    component_geometry_df.to_csv(component_geometry_path, index=False)
    np.save(output_path / "local_component_labels.npy", local_labels)
    np.save(output_path / "local_mahalanobis_scores.npy", local_scores)
    np.save(output_path / "local_z_scores.npy", local_z)
    np.save(output_path / "global_mahalanobis_scores.npy", global_scores)
    np.save(output_path / "global_z_scores.npy", global_z)
    np.save(output_path / "remove_benign_positions.npy", removed_positions)
    np.save(output_path / "remove_watermarked_idx.npy", scores_df.loc[remove_mask, "watermarked_idx"].to_numpy())
    joblib.dump(best_gmm, best_gmm_path)
    joblib.dump(global_gmm, global_gmm_path)

    config = GmmDefenseConfig(
        preprocess_dir=str(preprocess_path),
        artifact_dir=str(artifact_path),
        output_dir=str(output_path),
        k_min=k_min,
        k_max=k_max,
        covariance_types=covariance_types,
        reg_covar=reg_covar,
        removal_percent=removal_percent,
        score_name=score_name,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        max_rows=max_rows,
    )
    best_row = bic_df.loc[bic_df["bic"].idxmin()].to_dict()
    metadata = {
        "config": asdict(config),
        "input_shape": [int(X_fit.shape[0]), int(X_fit.shape[1])],
        "preprocessing_metadata_path": str(preprocess_metadata_path),
        "defense_metadata_path": str(defense_metadata_path),
        "best_model": best_row,
        "removal_metrics": metrics,
        "output_files": {
            "bic_scores": str(bic_scores_path),
            "suspicious_scores": str(suspicious_scores_path),
            "component_summary": str(component_summary_path),
            "component_geometry": str(component_geometry_path),
            "best_local_gmm": str(best_gmm_path),
            "global_gmm": str(global_gmm_path),
        },
        "runtime_seconds": time.time() - start_time,
    }
    metadata_out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return GmmDefenseResult(
        output_dir=str(output_path),
        bic_scores_path=str(bic_scores_path),
        suspicious_scores_path=str(suspicious_scores_path),
        component_summary_path=str(component_summary_path),
        component_geometry_path=str(component_geometry_path),
        metadata_path=str(metadata_out_path),
        best_gmm_path=str(best_gmm_path),
        global_gmm_path=str(global_gmm_path),
        best_n_components=int(best_gmm.n_components),
        best_covariance_type=str(best_gmm.covariance_type),
        best_bic=float(best_row["bic"]),
        removed_rows=int(metrics["removed_rows"]),
        poison_recall=metrics.get("poison_recall"),
        clean_false_positive_rate=metrics.get("clean_false_positive_rate"),
    )


def fit_gmm(
    X: np.ndarray,
    n_components: int,
    covariance_type: str,
    reg_covar: float,
    random_state: int,
    n_init: int,
    max_iter: int,
) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    gmm.fit(X)
    return gmm


def select_best_gmm(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    covariance_types: tuple[str, ...],
    reg_covar: float,
    random_state: int,
    n_init: int,
    max_iter: int,
) -> tuple[GaussianMixture, pd.DataFrame]:
    rows = []
    best_gmm = None
    best_bic = np.inf
    for covariance_type in covariance_types:
        for n_components in range(k_min, k_max + 1):
            fit_start = time.time()
            gmm = fit_gmm(
                X,
                n_components=n_components,
                covariance_type=covariance_type,
                reg_covar=reg_covar,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
            )
            bic = float(gmm.bic(X))
            rows.append(
                {
                    "n_components": int(n_components),
                    "covariance_type": covariance_type,
                    "bic": bic,
                    "converged": bool(gmm.converged_),
                    "n_iter": int(gmm.n_iter_),
                    "lower_bound": float(gmm.lower_bound_),
                    "fit_seconds": time.time() - fit_start,
                }
            )
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
    if best_gmm is None:
        raise RuntimeError("No GMM candidates were fit")
    return best_gmm, pd.DataFrame(rows)


def component_mahalanobis_scores(X: np.ndarray, gmm: GaussianMixture, labels: np.ndarray) -> np.ndarray:
    scores = np.zeros(X.shape[0], dtype=np.float64)
    for component_id in range(gmm.n_components):
        idx = np.flatnonzero(labels == component_id)
        if idx.size == 0:
            continue
        diff = X[idx] - gmm.means_[component_id]
        if gmm.covariance_type == "diag":
            var = gmm.covariances_[component_id] + 1e-12
            scores[idx] = np.sum((diff * diff) / var, axis=1)
        elif gmm.covariance_type == "spherical":
            var = float(gmm.covariances_[component_id]) + 1e-12
            scores[idx] = np.sum(diff * diff, axis=1) / var
        elif gmm.covariance_type == "tied":
            inv_cov = np.linalg.pinv(gmm.covariances_)
            scores[idx] = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        elif gmm.covariance_type == "full":
            inv_cov = np.linalg.pinv(gmm.covariances_[component_id])
            scores[idx] = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        else:
            raise ValueError(f"Unsupported covariance_type: {gmm.covariance_type}")
    return scores


def compute_gmm_row_diagnostics(
    X: np.ndarray,
    gmm: GaussianMixture,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    responsibilities = gmm.predict_proba(X)
    row_ids = np.arange(X.shape[0])
    assigned = responsibilities[row_ids, labels]
    confidence = responsibilities.max(axis=1)
    entropy = responsibility_entropy(responsibilities)
    return {
        "log_likelihood": gmm.score_samples(X),
        "responsibility_assigned": assigned,
        "responsibility_confidence": confidence,
        "responsibility_entropy": entropy,
    }


def responsibility_entropy(responsibilities: np.ndarray) -> np.ndarray:
    if responsibilities.shape[1] <= 1:
        return np.zeros(responsibilities.shape[0], dtype=np.float64)
    safe = np.clip(responsibilities, 1e-300, 1.0)
    entropy = -np.sum(safe * np.log(safe), axis=1)
    return entropy / np.log(responsibilities.shape[1])


def zscore(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean()) / (scores.std() + 1e-12)


def clusterwise_zscore(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    z = np.zeros_like(scores, dtype=np.float64)
    for component_id in np.unique(labels):
        idx = labels == component_id
        z[idx] = zscore(scores[idx])
    return z


def top_percent_indices(scores: np.ndarray, percent: float) -> np.ndarray:
    n_remove = max(1, int(np.ceil(scores.shape[0] * (percent / 100.0))))
    n_remove = min(n_remove, scores.shape[0])
    if n_remove == scores.shape[0]:
        return np.argsort(scores)[::-1]
    candidate = np.argpartition(scores, -n_remove)[-n_remove:]
    return candidate[np.argsort(scores[candidate])[::-1]]


def load_aligned_metadata(meta: np.lib.npyio.NpzFile, n_rows: int) -> dict[str, np.ndarray]:
    keys = [
        "benign_watermarked_idx",
        "benign_original_idx",
        "benign_source_idx",
        "poison_mask_benign",
    ]
    aligned = {}
    for key in keys:
        if key not in meta.files:
            raise KeyError(f"Missing {key} in defense_metadata.npz")
        arr = np.asarray(meta[key])
        if arr.shape[0] < n_rows:
            raise ValueError(f"{key} has {arr.shape[0]} rows, but X has {n_rows}")
        aligned[key] = arr[:n_rows]
    return aligned


def build_scores_df(
    aligned: dict[str, np.ndarray],
    local_labels: np.ndarray,
    local_scores: np.ndarray,
    local_z: np.ndarray,
    local_global_z: np.ndarray,
    global_scores: np.ndarray,
    global_z: np.ndarray,
    log_likelihood: np.ndarray,
    responsibility_assigned: np.ndarray,
    responsibility_confidence: np.ndarray,
    responsibility_entropy: np.ndarray,
    remove_mask: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "benign_position": np.arange(local_labels.shape[0], dtype=np.int64),
            "watermarked_idx": aligned["benign_watermarked_idx"].astype(np.int64, copy=False),
            "original_idx": aligned["benign_original_idx"].astype(np.int64, copy=False),
            "source_idx": aligned["benign_source_idx"].astype(np.int64, copy=False),
            "is_poisoned": aligned["poison_mask_benign"].astype(bool, copy=False),
            "component": local_labels.astype(np.int64, copy=False),
            "local_mahalanobis": local_scores,
            "local_z": local_z,
            "local_global_z": local_global_z,
            "global_mahalanobis": global_scores,
            "global_z": global_z,
            "gmm_log_likelihood": log_likelihood,
            "responsibility_assigned": responsibility_assigned,
            "responsibility_confidence": responsibility_confidence,
            "responsibility_entropy": responsibility_entropy,
            "removed": remove_mask,
        }
    )


def build_component_summary(scores_df: pd.DataFrame) -> pd.DataFrame:
    grouped = scores_df.groupby("component", sort=True)
    summary = grouped.agg(
        rows=("component", "size"),
        poisoned_rows=("is_poisoned", "sum"),
        mean_local_mahalanobis=("local_mahalanobis", "mean"),
        std_local_mahalanobis=("local_mahalanobis", "std"),
        max_local_z=("local_z", "max"),
        removed_rows=("removed", "sum"),
    ).reset_index()
    summary["poison_rate"] = summary["poisoned_rows"] / summary["rows"].clip(lower=1)
    return summary


def build_component_geometry(
    scores_df: pd.DataFrame,
    gmm: GaussianMixture,
    global_gmm: GaussianMixture | None = None,
) -> pd.DataFrame:
    n_features = int(gmm.means_.shape[1])
    labels = scores_df["component"].to_numpy(dtype=np.int64)
    total_rows = max(int(scores_df.shape[0]), 1)
    global_mean = global_gmm.means_[0] if global_gmm is not None else np.average(gmm.means_, axis=0, weights=gmm.weights_)
    global_var = (
        covariance_diag_for_component(global_gmm, 0, n_features)
        if global_gmm is not None
        else np.average([covariance_diag_for_component(gmm, i, n_features) for i in range(gmm.n_components)], axis=0)
    )

    rows = []
    grouped = scores_df.groupby("component", sort=True)
    for component_id in range(gmm.n_components):
        component_mask = labels == component_id
        assigned_rows = int(component_mask.sum())
        empirical_weight = float(assigned_rows / total_rows)
        gmm_weight = float(gmm.weights_[component_id])
        mean = gmm.means_[component_id]
        diff = mean - global_mean
        diag = covariance_diag_for_component(gmm, component_id, n_features)
        cov_logdet = covariance_logdet_for_component(gmm, component_id, n_features)
        density_proxy_log = float(np.log(max(gmm_weight, 1e-300)) - 0.5 * cov_logdet)
        empirical_density_proxy_log = float(np.log(max(empirical_weight, 1e-300)) - 0.5 * cov_logdet)
        row = {
            "component": int(component_id),
            "rows": assigned_rows,
            "empirical_weight": empirical_weight,
            "gmm_weight": gmm_weight,
            "mean_l2_from_global": float(np.linalg.norm(diff)),
            "mean_global_mahalanobis": float(np.sum((diff * diff) / (global_var + 1e-12))),
            "cov_trace": float(np.sum(diag)),
            "cov_mean_var": float(np.mean(diag)),
            "cov_min_var": float(np.min(diag)),
            "cov_max_var": float(np.max(diag)),
            "cov_logdet": float(cov_logdet),
            "cov_volume_log": float(0.5 * cov_logdet),
            "density_proxy_log": density_proxy_log,
            "empirical_density_proxy_log": empirical_density_proxy_log,
            "density_proxy": float(np.exp(np.clip(density_proxy_log, -700, 700))),
            "empirical_density_proxy": float(np.exp(np.clip(empirical_density_proxy_log, -700, 700))),
        }
        if component_id in grouped.groups:
            component_rows = grouped.get_group(component_id)
            for source_col, out_col in [
                ("gmm_log_likelihood", "avg_log_likelihood"),
                ("responsibility_assigned", "responsibility_assigned_mean"),
                ("responsibility_confidence", "responsibility_confidence_mean"),
                ("responsibility_entropy", "responsibility_entropy_mean"),
            ]:
                if source_col in component_rows:
                    row[out_col] = float(component_rows[source_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def covariance_diag_for_component(gmm: GaussianMixture, component_id: int, n_features: int) -> np.ndarray:
    if gmm.covariance_type == "diag":
        return np.asarray(gmm.covariances_[component_id], dtype=np.float64) + 1e-12
    if gmm.covariance_type == "spherical":
        return np.full(n_features, float(gmm.covariances_[component_id]) + 1e-12, dtype=np.float64)
    if gmm.covariance_type == "tied":
        return np.diag(np.asarray(gmm.covariances_, dtype=np.float64)) + 1e-12
    if gmm.covariance_type == "full":
        return np.diag(np.asarray(gmm.covariances_[component_id], dtype=np.float64)) + 1e-12
    raise ValueError(f"Unsupported covariance_type: {gmm.covariance_type}")


def covariance_logdet_for_component(gmm: GaussianMixture, component_id: int, n_features: int) -> float:
    if gmm.covariance_type in {"diag", "spherical"}:
        return float(np.sum(np.log(covariance_diag_for_component(gmm, component_id, n_features))))
    if gmm.covariance_type == "tied":
        cov = np.asarray(gmm.covariances_, dtype=np.float64)
    elif gmm.covariance_type == "full":
        cov = np.asarray(gmm.covariances_[component_id], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported covariance_type: {gmm.covariance_type}")
    sign, logdet = np.linalg.slogdet(cov + np.eye(n_features) * 1e-12)
    if sign <= 0:
        return float("nan")
    return float(logdet)


def removal_metrics(scores_df: pd.DataFrame) -> dict:
    removed = scores_df["removed"].to_numpy(dtype=bool)
    poisoned = scores_df["is_poisoned"].to_numpy(dtype=bool)
    clean = ~poisoned
    removed_poison = int(np.sum(removed & poisoned))
    removed_clean = int(np.sum(removed & clean))
    total_poison = int(np.sum(poisoned))
    total_clean = int(np.sum(clean))
    return {
        "rows": int(scores_df.shape[0]),
        "removed_rows": int(np.sum(removed)),
        "total_poisoned": total_poison,
        "total_clean": total_clean,
        "removed_poisoned": removed_poison,
        "removed_clean": removed_clean,
        "poison_recall": float(removed_poison / total_poison) if total_poison else None,
        "clean_false_positive_rate": float(removed_clean / total_clean) if total_clean else None,
    }


def validate_config(
    k_min: int,
    k_max: int,
    covariance_types: tuple[str, ...],
    reg_covar: float,
    removal_percent: float,
    score_name: str,
) -> None:
    if k_min < 1 or k_max < k_min:
        raise ValueError("Expected 1 <= k_min <= k_max")
    supported_covariance_types = {"diag", "tied", "full", "spherical"}
    invalid = sorted(set(covariance_types) - supported_covariance_types)
    if invalid:
        raise ValueError(f"Unsupported covariance type(s): {invalid}")
    if reg_covar <= 0:
        raise ValueError("--reg-covar must be positive")
    if not (0 < removal_percent <= 100):
        raise ValueError("--removal-percent must be in (0, 100]")
    supported_scores = {"local_z", "local_mahalanobis", "local_global_z", "global_z", "global_mahalanobis"}
    if score_name not in supported_scores:
        raise ValueError(f"Unsupported score_name {score_name}. Valid choices: {sorted(supported_scores)}")


def resolve_output_dir(
    preprocess_dir: Path,
    output_dir: str | Path | None,
    covariance_types: tuple[str, ...],
    k_min: int,
    k_max: int,
    reg_covar: float,
    removal_percent: float,
) -> Path:
    if output_dir is not None:
        resolved = resolve_path(output_dir)
        return resolved or Path(output_dir)
    cov_tag = "-".join(covariance_types)
    reg_tag = f"{reg_covar:g}".replace("-", "m").replace(".", "p")
    remove_tag = f"{removal_percent:g}".replace(".", "p")
    return preprocess_dir / "gmm_defense" / f"cov_{cov_tag}_k{k_min}-{k_max}_reg{reg_tag}_remove{remove_tag}p"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_existing_dir(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("Directory path is required")
    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Missing directory: {path}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved}")
    return resolved
