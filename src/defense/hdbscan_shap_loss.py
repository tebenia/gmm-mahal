"""HDBSCAN SHAP-space clustering plus loss-ranked sanitization.

This is an HDBSCAN-based instantiation of the fixed-threshold idea from the
Severi model-agnostic mitigation paper: cluster benign-labeled samples in a
SHAP-reduced representation, score clusters by surrogate-model benign loss, keep
the lowest-loss clusters, and remove the remaining benign rows before retraining.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..utils.paths import resolve_path


SUPPORTED_COVERAGE_UNITS = {"clusters", "rows"}
SUPPORTED_NOISE_POLICIES = {"as_cluster", "split"}


@dataclass
class HdbscanShapLossConfig:
    artifact_dir: str
    preprocess_dir: str
    output_dir: str
    clean_fraction: float = 0.80
    coverage_unit: str = "clusters"
    min_cluster_size: int | None = None
    min_cluster_percent: float = 0.5
    min_samples: int | None = None
    min_samples_percent: float = 0.1
    noise_policy: str = "split"
    noise_chunk_size: int = 1000
    standardize_reduced: bool = False
    surrogate_num_boost_round: int = 50
    random_state: int = 42
    max_benign_rows: int | None = None


@dataclass
class HdbscanShapLossResult:
    output_dir: str
    metadata_path: str
    cluster_assignments_path: str
    cluster_summary_path: str
    remove_watermarked_idx_path: str
    clusters: int
    clean_clusters: int
    suspicious_clusters: int
    removed_rows: int
    removed_poisoned_rows: int | None
    poison_recall: float | None


def run_hdbscan_shap_loss_defense(
    artifact_dir: str | Path,
    preprocess_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    clean_fraction: float = 0.80,
    coverage_unit: str = "clusters",
    min_cluster_size: int | None = None,
    min_cluster_percent: float = 0.5,
    min_samples: int | None = None,
    min_samples_percent: float = 0.1,
    noise_policy: str = "split",
    noise_chunk_size: int = 1000,
    standardize_reduced: bool = False,
    surrogate_num_boost_round: int = 50,
    random_state: int = 42,
    max_benign_rows: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> HdbscanShapLossResult | dict[str, Any]:
    validate_config(
        clean_fraction=clean_fraction,
        coverage_unit=coverage_unit,
        min_cluster_size=min_cluster_size,
        min_cluster_percent=min_cluster_percent,
        min_samples=min_samples,
        min_samples_percent=min_samples_percent,
        noise_policy=noise_policy,
        noise_chunk_size=noise_chunk_size,
        surrogate_num_boost_round=surrogate_num_boost_round,
        max_benign_rows=max_benign_rows,
    )

    artifact_path = _resolve_existing_dir(artifact_dir)
    preprocess_path = resolve_preprocess_dir(artifact_path, preprocess_dir)
    output_path = resolve_output_dir(
        artifact_path=artifact_path,
        output_dir=output_dir,
        clean_fraction=clean_fraction,
        coverage_unit=coverage_unit,
        min_cluster_size=min_cluster_size,
        min_cluster_percent=min_cluster_percent,
        min_samples=min_samples,
        min_samples_percent=min_samples_percent,
        noise_policy=noise_policy,
        max_benign_rows=max_benign_rows,
    )

    required = {
        "watermarked_X": artifact_path / "watermarked_X.npy",
        "watermarked_y": artifact_path / "watermarked_y.npy",
        "defense_metadata_npz": artifact_path / "defense_metadata.npz",
        "defense_metadata_json": artifact_path / "defense_metadata.json",
        "X_shap_reduced": preprocess_path / "X_shap_reduced.npy",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required HDBSCAN SHAP-loss artifact(s): {', '.join(missing)}")

    if dry_run:
        return {
            "artifact_dir": str(artifact_path),
            "preprocess_dir": str(preprocess_path),
            "output_dir": str(output_path),
            "required_paths": {key: str(path) for key, path in required.items()},
            "clean_fraction": clean_fraction,
            "coverage_unit": coverage_unit,
        }

    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN SHAP-loss defense requires the optional 'hdbscan' package. "
            "Install it in this environment, then rerun this command."
        ) from exc

    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "hdbscan_shap_loss_metadata.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_path} already exists. Pass --overwrite to replace it.")

    start_time = time.time()
    X_all = _unwrap_saved_array(np.load(required["watermarked_X"], mmap_mode="r", allow_pickle=True))
    y_all = np.asarray(_unwrap_saved_array(np.load(required["watermarked_y"], mmap_mode="r", allow_pickle=True)))
    X_shap = np.load(required["X_shap_reduced"], mmap_mode="r")
    meta = np.load(required["defense_metadata_npz"])
    metadata_json = json.loads(required["defense_metadata_json"].read_text(encoding="utf-8"))

    benign_idx = np.asarray(meta["benign_watermarked_idx"], dtype=np.int64).reshape(-1)
    malware_idx = np.flatnonzero(y_all.astype(int) == 1).astype(np.int64)
    row_count = min(benign_idx.shape[0], X_shap.shape[0])
    if max_benign_rows is not None:
        row_count = min(row_count, int(max_benign_rows))
    benign_idx = benign_idx[:row_count]
    X_shap_used = np.asarray(X_shap[:row_count], dtype=np.float32)
    if benign_idx.size == 0 or malware_idx.size == 0:
        raise ValueError("Expected both benign-labeled and malware-labeled training rows")
    if standardize_reduced:
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(X_shap_used)
    else:
        scaler = None
        X_cluster = X_shap_used

    resolved_min_cluster_size = resolve_count(
        explicit=min_cluster_size,
        percent=min_cluster_percent,
        total=int(X_all.shape[0]),
        minimum=2,
    )
    resolved_min_samples = resolve_count(
        explicit=min_samples,
        percent=min_samples_percent,
        total=int(X_all.shape[0]),
        minimum=1,
    )
    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        core_dist_n_jobs=-1,
        min_cluster_size=resolved_min_cluster_size,
        min_samples=resolved_min_samples,
    )
    raw_labels = np.asarray(clusterer.fit_predict(X_cluster), dtype=np.int64)
    labels = normalize_noise_labels(raw_labels, noise_policy=noise_policy, noise_chunk_size=noise_chunk_size)
    clusters = build_clusters(labels)
    if not clusters:
        raise ValueError("HDBSCAN did not produce any usable clusters")

    seed_cluster = max(clusters, key=lambda label: clusters[label].shape[0])
    surrogate_model = train_surrogate(
        X_all=X_all,
        y_all=y_all,
        train_idx=train_rows_for_clusters(benign_idx, malware_idx, clusters, {seed_cluster}),
        num_boost_round=surrogate_num_boost_round,
        random_state=random_state,
    )
    cluster_summary = score_and_select_clusters(
        X_all=X_all,
        benign_idx=benign_idx,
        clusters=clusters,
        seed_cluster=seed_cluster,
        model=surrogate_model,
        clean_fraction=clean_fraction,
        coverage_unit=coverage_unit,
    )
    clean_clusters = set(cluster_summary.loc[cluster_summary["kept_clean"], "cluster"].astype(int).tolist())
    suspicious_clusters = sorted(set(clusters) - clean_clusters)
    suspicious_positions = positions_for_clusters(clusters, suspicious_clusters)
    remove_watermarked_idx = benign_idx[suspicious_positions].astype(np.int64, copy=False)
    removal_stats = compute_removal_stats(meta, remove_watermarked_idx, n_rows=X_all.shape[0])

    cluster_assignments_path = output_path / "cluster_assignments.csv"
    cluster_summary_path = output_path / "cluster_summary.csv"
    remove_path = output_path / "remove_watermarked_idx.npy"
    model_path = output_path / "surrogate_model.txt"
    clusterer_path = output_path / "hdbscan.joblib"
    scaler_path = output_path / "standard_scaler_reduced.joblib" if scaler is not None else None

    cluster_summary = add_oracle_columns(cluster_summary, labels=labels, benign_idx=benign_idx, meta=meta)
    build_assignment_frame(
        labels=labels,
        benign_idx=benign_idx,
        meta=meta,
        suspicious_clusters=suspicious_clusters,
        seed_cluster=seed_cluster,
        cluster_summary=cluster_summary,
    ).to_csv(cluster_assignments_path, index=False)
    cluster_summary.to_csv(cluster_summary_path, index=False)
    np.save(remove_path, remove_watermarked_idx)
    surrogate_model.save_model(str(model_path))
    joblib.dump(clusterer, clusterer_path)
    if scaler is not None and scaler_path is not None:
        joblib.dump(scaler, scaler_path)

    config = HdbscanShapLossConfig(
        artifact_dir=str(artifact_path),
        preprocess_dir=str(preprocess_path),
        output_dir=str(output_path),
        clean_fraction=clean_fraction,
        coverage_unit=coverage_unit,
        min_cluster_size=min_cluster_size,
        min_cluster_percent=min_cluster_percent,
        min_samples=min_samples,
        min_samples_percent=min_samples_percent,
        noise_policy=noise_policy,
        noise_chunk_size=noise_chunk_size,
        standardize_reduced=standardize_reduced,
        surrogate_num_boost_round=surrogate_num_boost_round,
        random_state=random_state,
        max_benign_rows=max_benign_rows,
    )
    metadata = {
        "method": "HDBSCAN SHAP-space clustering plus fixed-threshold loss-ranked sanitization",
        "method_note": (
            "HDBSCAN is used for SHAP-space clusters ranked by surrogate benign loss. "
            "Poison labels are used only for diagnostics."
        ),
        "paper": "Model-agnostic clean-label backdoor mitigation in cybersecurity environments",
        "config": asdict(config),
        "dataset": metadata_json.get("dataset"),
        "input_shape": {
            "watermarked_X": list(X_all.shape),
            "X_shap_reduced": list(X_shap.shape),
            "benign_rows_used": int(benign_idx.shape[0]),
            "malware_rows_used": int(malware_idx.shape[0]),
        },
        "hdbscan": {
            "min_cluster_size": int(resolved_min_cluster_size),
            "min_samples": int(resolved_min_samples),
            "raw_clusters": int(len(set(int(v) for v in raw_labels if int(v) != -1))),
            "raw_noise_rows": int(np.sum(raw_labels == -1)),
            "noise_policy": noise_policy,
            "normalized_clusters": int(len(clusters)),
            "seed_cluster": int(seed_cluster),
        },
        "selection": {
            "clean_fraction": float(clean_fraction),
            "coverage_unit": coverage_unit,
            "clean_clusters": int(len(clean_clusters)),
            "suspicious_clusters": int(len(suspicious_clusters)),
        },
        "removal_stats": removal_stats,
        "output_files": {
            "cluster_assignments": str(cluster_assignments_path),
            "cluster_summary": str(cluster_summary_path),
            "remove_watermarked_idx": str(remove_path),
            "surrogate_model": str(model_path),
            "hdbscan_model": str(clusterer_path),
            "scaler": str(scaler_path) if scaler_path is not None else None,
        },
        "runtime_seconds": time.time() - start_time,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return HdbscanShapLossResult(
        output_dir=str(output_path),
        metadata_path=str(metadata_path),
        cluster_assignments_path=str(cluster_assignments_path),
        cluster_summary_path=str(cluster_summary_path),
        remove_watermarked_idx_path=str(remove_path),
        clusters=int(len(clusters)),
        clean_clusters=int(len(clean_clusters)),
        suspicious_clusters=int(len(suspicious_clusters)),
        removed_rows=int(remove_watermarked_idx.shape[0]),
        removed_poisoned_rows=removal_stats.get("removed_poisoned_rows"),
        poison_recall=removal_stats.get("poison_recall"),
    )


def score_and_select_clusters(
    *,
    X_all,
    benign_idx: np.ndarray,
    clusters: dict[int, np.ndarray],
    seed_cluster: int,
    model,
    clean_fraction: float,
    coverage_unit: str,
) -> pd.DataFrame:
    rows = []
    for cluster, positions in sorted(clusters.items()):
        row_idx = benign_idx[positions]
        rows.append(
            {
                "cluster": int(cluster),
                "rows": int(positions.shape[0]),
                "is_seed_cluster": bool(cluster == seed_cluster),
                "mean_benign_loss": mean_benign_log_loss(model, X_all, row_idx),
            }
        )
    df = pd.DataFrame(rows).sort_values(["mean_benign_loss", "rows", "cluster"], ascending=[True, False, True])
    if coverage_unit == "clusters":
        keep_count = max(1, int(math.ceil(len(df) * clean_fraction)))
        df["kept_clean"] = False
        df.iloc[:keep_count, df.columns.get_loc("kept_clean")] = True
        threshold_value = keep_count
    else:
        target_rows = max(1, int(math.ceil(df["rows"].sum() * clean_fraction)))
        cumulative = df["rows"].cumsum()
        df["kept_clean"] = (cumulative - df["rows"]) < target_rows
        if not df["kept_clean"].any():
            df.iloc[0, df.columns.get_loc("kept_clean")] = True
        threshold_value = target_rows
    df["loss_rank"] = np.arange(1, len(df) + 1, dtype=np.int64)
    df["is_suspicious_cluster"] = ~df["kept_clean"]
    df["coverage_unit"] = coverage_unit
    df["clean_threshold"] = int(threshold_value)
    return df.sort_values(["is_suspicious_cluster", "mean_benign_loss", "cluster"])


def build_assignment_frame(
    *,
    labels: np.ndarray,
    benign_idx: np.ndarray,
    meta: np.lib.npyio.NpzFile,
    suspicious_clusters: list[int],
    seed_cluster: int,
    cluster_summary: pd.DataFrame,
) -> pd.DataFrame:
    suspicious = set(int(v) for v in suspicious_clusters)
    poison_mask = aligned_poison_mask(meta, benign_idx)
    cluster_loss = cluster_summary.set_index("cluster")["mean_benign_loss"].to_dict()
    return pd.DataFrame(
        {
            "benign_position": np.arange(benign_idx.shape[0], dtype=np.int64),
            "watermarked_idx": benign_idx,
            "cluster": labels.astype(np.int64),
            "mean_benign_loss": [cluster_loss.get(int(label), np.nan) for label in labels],
            "is_seed_cluster": labels == int(seed_cluster),
            "is_suspicious_cluster": np.array([int(label) in suspicious for label in labels], dtype=bool),
            "is_poisoned": poison_mask,
        }
    )


def add_oracle_columns(
    cluster_summary: pd.DataFrame,
    *,
    labels: np.ndarray,
    benign_idx: np.ndarray,
    meta: np.lib.npyio.NpzFile,
) -> pd.DataFrame:
    assignments = pd.DataFrame(
        {
            "cluster": labels.astype(np.int64),
            "is_poisoned": aligned_poison_mask(meta, benign_idx),
        }
    )
    total_poison = max(int(assignments["is_poisoned"].sum()), 1)
    oracle = (
        assignments.groupby("cluster", sort=True)["is_poisoned"]
        .agg(oracle_poisoned="sum", rows_check="size")
        .reset_index()
    )
    oracle["oracle_poison_rate"] = oracle["oracle_poisoned"] / oracle["rows_check"]
    oracle["oracle_poison_share"] = oracle["oracle_poisoned"] / total_poison
    out = cluster_summary.merge(oracle.drop(columns=["rows_check"]), on="cluster", how="left")
    out["oracle_poisoned"] = out["oracle_poisoned"].fillna(0).astype(int)
    out["oracle_poison_rate"] = out["oracle_poison_rate"].fillna(0.0)
    out["oracle_poison_share"] = out["oracle_poison_share"].fillna(0.0)
    return out.sort_values(["kept_clean", "mean_benign_loss", "rows"], ascending=[True, False, False])


def resolve_count(*, explicit: int | None, percent: float, total: int, minimum: int) -> int:
    if explicit is not None:
        return max(int(explicit), minimum)
    return max(int(np.ceil(total * float(percent) / 100.0)), minimum)


def resolve_preprocess_dir(artifact_path: Path, preprocess_dir: str | Path | None) -> Path:
    if preprocess_dir is not None:
        resolved = resolve_path(preprocess_dir)
        path = resolved or Path(preprocess_dir)
    else:
        path = artifact_path / "defense_preprocessing" / "standardized_pca50"
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing preprocessing directory: {path}. Run `python3 -m run_defense_preprocess --artifact-dir {artifact_path}` first."
        )
    return path


def resolve_output_dir(
    *,
    artifact_path: Path,
    output_dir: str | Path | None,
    clean_fraction: float,
    coverage_unit: str,
    min_cluster_size: int | None,
    min_cluster_percent: float,
    min_samples: int | None,
    min_samples_percent: float,
    noise_policy: str,
    max_benign_rows: int | None,
) -> Path:
    if output_dir is not None:
        resolved = resolve_path(output_dir)
        return resolved or Path(output_dir)
    mcs_tag = str(min_cluster_size) if min_cluster_size is not None else f"{tag_float(min_cluster_percent)}pct"
    ms_tag = str(min_samples) if min_samples is not None else f"{tag_float(min_samples_percent)}pct"
    row_tag = f"_rows{max_benign_rows}" if max_benign_rows is not None else ""
    dirname = (
        f"{coverage_unit}_clean{fraction_tag(clean_fraction)}_"
        f"mcs{mcs_tag}_ms{ms_tag}_noise{noise_policy}{row_tag}"
    )
    return artifact_path / "hdbscan_shap_loss" / dirname


def normalize_noise_labels(raw_labels: np.ndarray, noise_policy: str, noise_chunk_size: int) -> np.ndarray:
    labels = np.asarray(raw_labels, dtype=np.int64).copy()
    if noise_policy == "as_cluster" or not np.any(labels == -1):
        return labels
    next_label = int(labels.max()) + 1 if labels.size else 0
    noise_positions = np.flatnonzero(labels == -1)
    for start in range(0, noise_positions.shape[0], noise_chunk_size):
        end = min(start + noise_chunk_size, noise_positions.shape[0])
        labels[noise_positions[start:end]] = next_label
        next_label += 1
    return labels


def build_clusters(labels: np.ndarray) -> dict[int, np.ndarray]:
    clusters: dict[int, np.ndarray] = {}
    for label in sorted(set(int(v) for v in labels)):
        positions = np.flatnonzero(labels == label).astype(np.int64)
        if positions.size:
            clusters[int(label)] = positions
    return clusters


def train_surrogate(
    X_all,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    num_boost_round: int,
    random_state: int,
) -> lgb.Booster:
    dataset = lgb.Dataset(X_all[train_idx], label=np.asarray(y_all[train_idx]).astype(np.int8), free_raw_data=False)
    params = {
        "objective": "binary",
        "verbosity": -1,
        "seed": int(random_state),
        "feature_fraction_seed": int(random_state),
        "bagging_seed": int(random_state),
    }
    return lgb.train(params, dataset, num_boost_round=num_boost_round)


def mean_benign_log_loss(model: lgb.Booster, X_all, row_idx: np.ndarray) -> float:
    if row_idx.size == 0:
        return float("inf")
    preds = np.asarray(model.predict(X_all[row_idx]), dtype=np.float64)
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds = preds[:, 1]
    preds = np.clip(preds, 1e-9, 1.0 - 1e-9)
    return float(np.mean(-np.log1p(-preds)))


def train_rows_for_clusters(
    benign_idx: np.ndarray,
    malware_idx: np.ndarray,
    clusters: dict[int, np.ndarray],
    clean_clusters: set[int],
) -> np.ndarray:
    benign_parts = [benign_idx[clusters[int(label)]] for label in clean_clusters]
    if benign_parts:
        clean_benign_idx = np.concatenate(benign_parts).astype(np.int64, copy=False)
        return np.concatenate([clean_benign_idx, malware_idx]).astype(np.int64, copy=False)
    return malware_idx.astype(np.int64, copy=False)


def positions_for_clusters(clusters: dict[int, np.ndarray], selected_clusters: list[int]) -> np.ndarray:
    parts = [clusters[int(label)] for label in selected_clusters if int(label) in clusters]
    if not parts:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(parts).astype(np.int64, copy=False))


def aligned_poison_mask(meta: np.lib.npyio.NpzFile, benign_watermarked_idx: np.ndarray) -> np.ndarray:
    if "poison_mask_benign" in meta.files and meta["poison_mask_benign"].shape[0] >= benign_watermarked_idx.shape[0]:
        return np.asarray(meta["poison_mask_benign"][: benign_watermarked_idx.shape[0]], dtype=bool)
    if "poison_mask_full" in meta.files:
        full = np.asarray(meta["poison_mask_full"], dtype=bool)
        return full[benign_watermarked_idx]
    return np.zeros(benign_watermarked_idx.shape[0], dtype=bool)


def compute_removal_stats(meta: np.lib.npyio.NpzFile, remove_idx: np.ndarray, n_rows: int) -> dict[str, Any]:
    if "poison_mask_full" not in meta.files:
        return {}
    poison_mask = np.asarray(meta["poison_mask_full"], dtype=bool)
    if poison_mask.shape[0] < n_rows:
        n_rows = poison_mask.shape[0]
    remove_idx = remove_idx[remove_idx < n_rows]
    removed_mask = np.zeros(n_rows, dtype=bool)
    removed_mask[remove_idx] = True
    total_poison = int(poison_mask[:n_rows].sum())
    total_clean = int(n_rows - total_poison)
    removed_poison = int(np.sum(removed_mask & poison_mask[:n_rows]))
    removed_clean = int(np.sum(removed_mask & ~poison_mask[:n_rows]))
    return {
        "total_poisoned_rows": total_poison,
        "total_clean_rows": total_clean,
        "removed_rows": int(remove_idx.shape[0]),
        "removed_poisoned_rows": removed_poison,
        "removed_clean_rows": removed_clean,
        "poison_recall": float(removed_poison / total_poison) if total_poison else None,
        "clean_false_positive_rate": float(removed_clean / total_clean) if total_clean else None,
    }


def fraction_tag(value: float) -> str:
    return f"{int(round(value * 100))}p" if 0 < value <= 1 else tag_float(value)


def tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def validate_config(
    *,
    clean_fraction: float,
    coverage_unit: str,
    min_cluster_size: int | None,
    min_cluster_percent: float,
    min_samples: int | None,
    min_samples_percent: float,
    noise_policy: str,
    noise_chunk_size: int,
    surrogate_num_boost_round: int,
    max_benign_rows: int | None,
) -> None:
    if not 0 < clean_fraction <= 1:
        raise ValueError("clean_fraction must be in (0, 1]")
    if coverage_unit not in SUPPORTED_COVERAGE_UNITS:
        raise ValueError(f"coverage_unit must be one of {sorted(SUPPORTED_COVERAGE_UNITS)}")
    if min_cluster_size is not None and min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    if not 0 < min_cluster_percent <= 100:
        raise ValueError("min_cluster_percent must be in (0, 100]")
    if min_samples is not None and min_samples < 1:
        raise ValueError("min_samples must be positive")
    if not 0 < min_samples_percent <= 100:
        raise ValueError("min_samples_percent must be in (0, 100]")
    if noise_policy not in SUPPORTED_NOISE_POLICIES:
        raise ValueError(f"noise_policy must be one of {sorted(SUPPORTED_NOISE_POLICIES)}")
    if noise_chunk_size <= 0:
        raise ValueError("noise_chunk_size must be positive")
    if surrogate_num_boost_round <= 0:
        raise ValueError("surrogate_num_boost_round must be positive")
    if max_benign_rows is not None and max_benign_rows <= 0:
        raise ValueError("max_benign_rows must be positive")


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
