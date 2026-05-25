"""OPTICS plus iterative loss filtering for clean-label backdoor defense.

This is a paper-described reimplementation of "Model-agnostic clean-label
backdoor mitigation in cybersecurity environments" for this repo's saved
attack artifacts. It implements the filtering path: select important features,
cluster benign-labeled rows with OPTICS, iteratively add low-loss clusters to a
clean set, and save the remaining suspicious training row ids for retraining.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .component_trigger_matching import candidate_feature_ids, feature_names_for_width, load_watermark
from .mdr_inspired import aligned_poison_mask, compute_removal_stats, dense_feature_matrix
from ..utils.paths import resolve_path


SUPPORTED_CANDIDATE_FEATURE_SOURCES = {"all", "nonhashed", "oracle_watermark"}
SUPPORTED_SELECTION_MODES = {"fixed_threshold", "loss_delta_z", "fixed_or_delta"}
SUPPORTED_DELTA_TAILS = {"lower", "upper", "absolute"}
SUPPORTED_NOISE_POLICIES = {"as_cluster", "split"}


@dataclass
class OpticsIterativeConfig:
    artifact_dir: str
    output_dir: str
    candidate_feature_source: str = "all"
    top_features: int = 16
    standardize: bool = True
    importance_max_rows: int | None = 50_000
    decision_tree_max_depth: int | None = None
    min_samples: int = 50
    max_eps: float = np.inf
    xi: float = 0.05
    min_cluster_size: int | float | None = None
    noise_policy: str = "split"
    noise_chunk_size: int = 1000
    window_fraction: float = 0.05
    clean_cluster_fraction: float = 0.80
    selection_mode: str = "fixed_threshold"
    delta_z_threshold: float = 2.0
    delta_tail: str = "lower"
    surrogate_num_boost_round: int = 50
    random_state: int = 42
    max_benign_rows: int | None = None


@dataclass
class OpticsIterativeResult:
    output_dir: str
    metadata_path: str
    selected_features_path: str
    cluster_assignments_path: str
    cluster_summary_path: str
    iteration_scores_path: str
    cluster_deltas_path: str
    remove_watermarked_idx_path: str
    selected_feature_count: int
    clusters: int
    suspicious_clusters: int
    removed_rows: int
    removed_poisoned_rows: int | None
    poison_recall: float | None


def run_optics_iterative_defense(
    artifact_dir: str | Path,
    output_dir: str | Path | None = None,
    candidate_feature_source: str = "all",
    top_features: int = 16,
    standardize: bool = True,
    importance_max_rows: int | None = 50_000,
    decision_tree_max_depth: int | None = None,
    min_samples: int = 50,
    max_eps: float = np.inf,
    xi: float = 0.05,
    min_cluster_size: int | float | None = None,
    noise_policy: str = "split",
    noise_chunk_size: int = 1000,
    window_fraction: float = 0.05,
    clean_cluster_fraction: float = 0.80,
    selection_mode: str = "fixed_threshold",
    delta_z_threshold: float = 2.0,
    delta_tail: str = "lower",
    surrogate_num_boost_round: int = 50,
    random_state: int = 42,
    max_benign_rows: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> OpticsIterativeResult | dict[str, Any]:
    validate_config(
        candidate_feature_source=candidate_feature_source,
        top_features=top_features,
        importance_max_rows=importance_max_rows,
        min_samples=min_samples,
        max_eps=max_eps,
        xi=xi,
        min_cluster_size=min_cluster_size,
        noise_policy=noise_policy,
        noise_chunk_size=noise_chunk_size,
        window_fraction=window_fraction,
        clean_cluster_fraction=clean_cluster_fraction,
        selection_mode=selection_mode,
        delta_z_threshold=delta_z_threshold,
        delta_tail=delta_tail,
        surrogate_num_boost_round=surrogate_num_boost_round,
        max_benign_rows=max_benign_rows,
    )

    artifact_path = _resolve_existing_dir(artifact_dir)
    output_path = resolve_output_dir(
        output_dir=output_dir,
        artifact_dir=artifact_path,
        candidate_feature_source=candidate_feature_source,
        top_features=top_features,
        selection_mode=selection_mode,
        clean_cluster_fraction=clean_cluster_fraction,
        window_fraction=window_fraction,
    )
    required = {
        "watermarked_X": artifact_path / "watermarked_X.npy",
        "watermarked_y": artifact_path / "watermarked_y.npy",
        "defense_metadata_npz": artifact_path / "defense_metadata.npz",
        "defense_metadata_json": artifact_path / "defense_metadata.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required OPTICS defense artifact(s): {', '.join(missing)}")

    if dry_run:
        return {
            "artifact_dir": str(artifact_path),
            "output_dir": str(output_path),
            "required_paths": {key: str(path) for key, path in required.items()},
            "candidate_feature_source": candidate_feature_source,
            "top_features": top_features,
            "selection_mode": selection_mode,
        }

    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "optics_iterative_metadata.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_path} already exists. Pass --overwrite to replace it.")

    start_time = time.time()
    X_all = _unwrap_saved_array(np.load(required["watermarked_X"], mmap_mode="r", allow_pickle=True))
    y_all = np.asarray(_unwrap_saved_array(np.load(required["watermarked_y"], mmap_mode="r", allow_pickle=True)))
    meta = np.load(required["defense_metadata_npz"])
    metadata_json = json.loads(required["defense_metadata_json"].read_text(encoding="utf-8"))

    benign_idx = np.flatnonzero(y_all.astype(int) == 0).astype(np.int64)
    malware_idx = np.flatnonzero(y_all.astype(int) == 1).astype(np.int64)
    if max_benign_rows is not None:
        benign_idx = benign_idx[: min(max_benign_rows, benign_idx.shape[0])]
    if benign_idx.size == 0 or malware_idx.size == 0:
        raise ValueError("Expected both benign-labeled and malware-labeled training rows")

    n_features = int(X_all.shape[1])
    feature_names = feature_names_for_width(n_features)
    watermark = load_watermark(required["defense_metadata_json"], artifact_path / "wm_config.npy", feature_names)
    candidate_features = candidate_feature_ids(candidate_feature_source, n_features, watermark)

    selected_features_df = select_important_features(
        X_all=X_all,
        y_all=y_all,
        candidate_features=candidate_features,
        feature_names=feature_names,
        top_features=top_features,
        max_rows=importance_max_rows,
        decision_tree_max_depth=decision_tree_max_depth,
        random_state=random_state,
    )
    selected_feature_ids = selected_features_df.loc[selected_features_df["selected"], "feature_idx"].to_list()
    if not selected_feature_ids:
        raise ValueError("No important features were selected")

    X_reduced = dense_feature_matrix(X_all, benign_idx, selected_feature_ids)
    if standardize:
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(X_reduced)
    else:
        scaler = None
        X_cluster = X_reduced

    raw_labels = fit_optics_labels(
        X_cluster=X_cluster,
        min_samples=min_samples,
        max_eps=max_eps,
        xi=xi,
        min_cluster_size=min_cluster_size,
    )
    labels = normalize_noise_labels(raw_labels, noise_policy=noise_policy, noise_chunk_size=noise_chunk_size)
    clusters = build_clusters(labels)
    if not clusters:
        raise ValueError("OPTICS did not produce any usable clusters")

    seed_cluster = max(clusters, key=lambda label: clusters[label].shape[0])
    scoring = iterative_cluster_scoring(
        X_all=X_all,
        y_all=y_all,
        benign_idx=benign_idx,
        malware_idx=malware_idx,
        clusters=clusters,
        seed_cluster=seed_cluster,
        window_fraction=window_fraction,
        clean_cluster_fraction=clean_cluster_fraction,
        selection_mode=selection_mode,
        delta_z_threshold=delta_z_threshold,
        delta_tail=delta_tail,
        surrogate_num_boost_round=surrogate_num_boost_round,
        random_state=random_state,
    )

    suspicious_clusters = scoring["suspicious_clusters"]
    suspicious_positions = positions_for_clusters(clusters, suspicious_clusters)
    remove_watermarked_idx = benign_idx[suspicious_positions].astype(np.int64, copy=False)
    removal_stats = compute_removal_stats(meta, remove_watermarked_idx, n_rows=X_all.shape[0])

    selected_features_path = output_path / "selected_features.csv"
    cluster_assignments_path = output_path / "cluster_assignments.csv"
    cluster_summary_path = output_path / "cluster_summary.csv"
    iteration_scores_path = output_path / "iteration_scores.csv"
    cluster_deltas_path = output_path / "cluster_deltas.csv"
    remove_watermarked_idx_path = output_path / "remove_watermarked_idx.npy"

    selected_features_df.to_csv(selected_features_path, index=False)
    cluster_assignments = build_assignment_frame(
        labels=labels,
        benign_idx=benign_idx,
        meta=meta,
        suspicious_clusters=suspicious_clusters,
        seed_cluster=seed_cluster,
    )
    cluster_summary = build_cluster_summary(
        labels=labels,
        meta=meta,
        benign_idx=benign_idx,
        seed_cluster=seed_cluster,
        suspicious_clusters=suspicious_clusters,
        cluster_order=scoring["cluster_order"],
        threshold_cluster_count=scoring["summary"]["threshold_cluster_count"],
    )
    cluster_deltas = pd.DataFrame(scoring["cluster_deltas"])
    iteration_scores = pd.DataFrame(scoring["iteration_scores"])
    cluster_assignments.to_csv(cluster_assignments_path, index=False)
    cluster_summary.to_csv(cluster_summary_path, index=False)
    iteration_scores.to_csv(iteration_scores_path, index=False)
    cluster_deltas.to_csv(cluster_deltas_path, index=False)
    np.save(remove_watermarked_idx_path, remove_watermarked_idx)

    config = OpticsIterativeConfig(
        artifact_dir=str(artifact_path),
        output_dir=str(output_path),
        candidate_feature_source=candidate_feature_source,
        top_features=top_features,
        standardize=standardize,
        importance_max_rows=importance_max_rows,
        decision_tree_max_depth=decision_tree_max_depth,
        min_samples=min_samples,
        max_eps=max_eps,
        xi=xi,
        min_cluster_size=min_cluster_size,
        noise_policy=noise_policy,
        noise_chunk_size=noise_chunk_size,
        window_fraction=window_fraction,
        clean_cluster_fraction=clean_cluster_fraction,
        selection_mode=selection_mode,
        delta_z_threshold=delta_z_threshold,
        delta_tail=delta_tail,
        surrogate_num_boost_round=surrogate_num_boost_round,
        random_state=random_state,
        max_benign_rows=max_benign_rows,
    )
    metadata = {
        "method": "OPTICS iterative clean-label defense reimplementation from paper description",
        "paper": "Model-agnostic clean-label backdoor mitigation in cybersecurity environments",
        "config": asdict(config),
        "dataset": metadata_json.get("dataset"),
        "input_shape": {
            "watermarked_X": list(X_all.shape),
            "benign_rows_used": int(benign_idx.shape[0]),
            "malware_rows_used": int(malware_idx.shape[0]),
            "candidate_features": int(len(candidate_features)),
            "selected_features": int(len(selected_feature_ids)),
            "clusters": int(len(clusters)),
        },
        "optics": {
            "raw_clusters": int(len(set(int(v) for v in raw_labels if int(v) != -1))),
            "raw_noise_rows": int(np.sum(raw_labels == -1)),
            "noise_policy": noise_policy,
            "normalized_clusters": int(len(clusters)),
            "seed_cluster": int(seed_cluster),
        },
        "scoring": scoring["summary"],
        "removal_stats": removal_stats,
        "output_files": {
            "selected_features": str(selected_features_path),
            "cluster_assignments": str(cluster_assignments_path),
            "cluster_summary": str(cluster_summary_path),
            "iteration_scores": str(iteration_scores_path),
            "cluster_deltas": str(cluster_deltas_path),
            "remove_watermarked_idx": str(remove_watermarked_idx_path),
        },
        "runtime_seconds": time.time() - start_time,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return OpticsIterativeResult(
        output_dir=str(output_path),
        metadata_path=str(metadata_path),
        selected_features_path=str(selected_features_path),
        cluster_assignments_path=str(cluster_assignments_path),
        cluster_summary_path=str(cluster_summary_path),
        iteration_scores_path=str(iteration_scores_path),
        cluster_deltas_path=str(cluster_deltas_path),
        remove_watermarked_idx_path=str(remove_watermarked_idx_path),
        selected_feature_count=int(len(selected_feature_ids)),
        clusters=int(len(clusters)),
        suspicious_clusters=int(len(suspicious_clusters)),
        removed_rows=int(remove_watermarked_idx.shape[0]),
        removed_poisoned_rows=removal_stats.get("removed_poisoned_rows"),
        poison_recall=removal_stats.get("poison_recall"),
    )


def select_important_features(
    X_all,
    y_all: np.ndarray,
    candidate_features: list[int],
    feature_names: list[str],
    top_features: int,
    max_rows: int | None,
    decision_tree_max_depth: int | None,
    random_state: int,
) -> pd.DataFrame:
    rows = sample_rows(y_all.shape[0], max_rows=max_rows, random_state=random_state)
    X_tree = dense_feature_matrix(X_all, rows, candidate_features)
    y_tree = np.asarray(y_all[rows]).astype(np.int8)
    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=decision_tree_max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    tree.fit(X_tree, y_tree)
    importances = np.asarray(tree.feature_importances_, dtype=np.float64)
    if not np.any(importances > 0):
        importances = np.var(X_tree, axis=0)
    order = np.argsort(importances)[::-1]
    selected_positions = set(int(pos) for pos in order[: min(top_features, len(order))])
    rows_out = []
    for rank, pos in enumerate(order, start=1):
        feature_idx = int(candidate_features[int(pos)])
        rows_out.append(
            {
                "importance_rank": int(rank),
                "feature_idx": feature_idx,
                "feature_name": feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}",
                "importance": float(importances[int(pos)]),
                "selected": bool(int(pos) in selected_positions),
            }
        )
    return pd.DataFrame(rows_out)


def fit_optics_labels(
    X_cluster: np.ndarray,
    min_samples: int,
    max_eps: float,
    xi: float,
    min_cluster_size: int | float | None,
) -> np.ndarray:
    optics = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        metric="minkowski",
        cluster_method="xi",
        xi=xi,
        min_cluster_size=min_cluster_size,
        n_jobs=-1,
    )
    return np.asarray(optics.fit_predict(X_cluster), dtype=np.int64)


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


def iterative_cluster_scoring(
    X_all,
    y_all: np.ndarray,
    benign_idx: np.ndarray,
    malware_idx: np.ndarray,
    clusters: dict[int, np.ndarray],
    seed_cluster: int,
    window_fraction: float,
    clean_cluster_fraction: float,
    selection_mode: str,
    delta_z_threshold: float,
    delta_tail: str,
    surrogate_num_boost_round: int,
    random_state: int,
) -> dict[str, Any]:
    all_clusters = sorted(clusters)
    total_clusters = len(all_clusters)
    window_size = max(1, int(math.ceil(total_clusters * window_fraction)))
    threshold_cluster_count = max(1, int(math.ceil(total_clusters * clean_cluster_fraction)))

    clean_clusters: set[int] = {int(seed_cluster)}
    remaining: set[int] = set(all_clusters) - clean_clusters
    current_model = train_surrogate(
        X_all=X_all,
        y_all=y_all,
        train_idx=train_rows_for_clusters(benign_idx, malware_idx, clusters, clean_clusters),
        num_boost_round=surrogate_num_boost_round,
        random_state=random_state,
    )

    iteration_scores: list[dict[str, Any]] = []
    cluster_deltas: list[dict[str, Any]] = []
    cluster_order: dict[int, int] = {int(seed_cluster): 0}
    threshold_iteration: int | None = None
    iteration = 0

    while remaining:
        iteration += 1
        losses = []
        for cluster_label in sorted(remaining):
            row_idx = benign_idx[clusters[cluster_label]]
            mean_loss = mean_benign_log_loss(current_model, X_all, row_idx)
            losses.append((int(cluster_label), float(mean_loss)))
            iteration_scores.append(
                {
                    "iteration": int(iteration),
                    "cluster": int(cluster_label),
                    "rows": int(clusters[cluster_label].shape[0]),
                    "mean_loss": float(mean_loss),
                    "selected": False,
                    "clean_cluster_count_before": int(len(clean_clusters)),
                }
            )
        losses.sort(key=lambda item: item[1])
        selected = [cluster_label for cluster_label, _ in losses[: min(window_size, len(losses))]]
        before_loss = {cluster_label: loss for cluster_label, loss in losses if cluster_label in selected}
        for row in iteration_scores[-len(losses) :]:
            if row["cluster"] in selected:
                row["selected"] = True

        clean_clusters.update(selected)
        remaining.difference_update(selected)
        for cluster_label in selected:
            cluster_order.setdefault(int(cluster_label), len(cluster_order))

        next_model = train_surrogate(
            X_all=X_all,
            y_all=y_all,
            train_idx=train_rows_for_clusters(benign_idx, malware_idx, clusters, clean_clusters),
            num_boost_round=surrogate_num_boost_round,
            random_state=random_state + iteration,
        )
        for cluster_label in selected:
            row_idx = benign_idx[clusters[cluster_label]]
            loss_after = mean_benign_log_loss(next_model, X_all, row_idx)
            delta = float(loss_after - before_loss[cluster_label])
            cluster_deltas.append(
                {
                    "cluster": int(cluster_label),
                    "iteration": int(iteration),
                    "selection_order": int(cluster_order[cluster_label]),
                    "rows": int(clusters[cluster_label].shape[0]),
                    "loss_before": float(before_loss[cluster_label]),
                    "loss_after": float(loss_after),
                    "loss_delta": delta,
                    "included_before_fixed_threshold": bool(len(clean_clusters) <= threshold_cluster_count),
                }
            )
        current_model = next_model

        if threshold_iteration is None and len(clean_clusters) >= threshold_cluster_count:
            threshold_iteration = iteration
            if selection_mode == "fixed_threshold":
                break

    fixed_suspicious = set(remaining) if selection_mode == "fixed_threshold" else {
        label for label in all_clusters if cluster_order.get(label, total_clusters + 1) >= threshold_cluster_count
    }
    delta_suspicious = delta_suspicious_clusters(cluster_deltas, delta_z_threshold, delta_tail)
    for row in cluster_deltas:
        row["loss_delta_z"] = float(row.get("loss_delta_z", np.nan))
        row["suspicious_by_delta"] = bool(row["cluster"] in delta_suspicious)

    if selection_mode == "fixed_threshold":
        suspicious_clusters = fixed_suspicious
    elif selection_mode == "loss_delta_z":
        suspicious_clusters = delta_suspicious
    else:
        suspicious_clusters = fixed_suspicious | delta_suspicious

    summary = {
        "selection_mode": selection_mode,
        "seed_cluster": int(seed_cluster),
        "window_size": int(window_size),
        "threshold_cluster_count": int(threshold_cluster_count),
        "threshold_iteration": int(threshold_iteration) if threshold_iteration is not None else None,
        "clusters_in_clean_set": int(len(clean_clusters)),
        "remaining_clusters_after_scoring": int(len(remaining)),
        "fixed_suspicious_clusters": int(len(fixed_suspicious)),
        "delta_suspicious_clusters": int(len(delta_suspicious)),
        "suspicious_clusters": int(len(suspicious_clusters)),
    }
    return {
        "iteration_scores": iteration_scores,
        "cluster_deltas": cluster_deltas,
        "cluster_order": cluster_order,
        "threshold_iteration": threshold_iteration,
        "fixed_suspicious": sorted(int(v) for v in fixed_suspicious),
        "delta_suspicious": sorted(int(v) for v in delta_suspicious),
        "suspicious_clusters": sorted(int(v) for v in suspicious_clusters),
        "summary": summary,
    }


def delta_suspicious_clusters(
    cluster_deltas: list[dict[str, Any]],
    delta_z_threshold: float,
    delta_tail: str,
) -> set[int]:
    if not cluster_deltas:
        return set()
    values = np.asarray([row["loss_delta"] for row in cluster_deltas], dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 0:
        z_values = np.zeros_like(values)
    else:
        z_values = (values - mean) / std
    suspicious: set[int] = set()
    for row, z_value in zip(cluster_deltas, z_values, strict=True):
        row["loss_delta_z"] = float(z_value)
        if delta_tail == "lower":
            is_suspicious = z_value <= -delta_z_threshold
        elif delta_tail == "upper":
            is_suspicious = z_value >= delta_z_threshold
        else:
            is_suspicious = abs(z_value) >= delta_z_threshold
        if is_suspicious:
            suspicious.add(int(row["cluster"]))
    return suspicious


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
    clean_clusters: Iterable[int],
) -> np.ndarray:
    benign_parts = [benign_idx[clusters[int(label)]] for label in clean_clusters]
    if benign_parts:
        clean_benign_idx = np.concatenate(benign_parts).astype(np.int64, copy=False)
        return np.concatenate([clean_benign_idx, malware_idx]).astype(np.int64, copy=False)
    return malware_idx.astype(np.int64, copy=False)


def positions_for_clusters(clusters: dict[int, np.ndarray], selected_clusters: Iterable[int]) -> np.ndarray:
    parts = [clusters[int(label)] for label in selected_clusters if int(label) in clusters]
    if not parts:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(parts).astype(np.int64, copy=False))


def build_assignment_frame(
    labels: np.ndarray,
    benign_idx: np.ndarray,
    meta: np.lib.npyio.NpzFile,
    suspicious_clusters: Iterable[int],
    seed_cluster: int,
) -> pd.DataFrame:
    suspicious = set(int(v) for v in suspicious_clusters)
    poison_mask = aligned_poison_mask(meta, benign_idx)
    return pd.DataFrame(
        {
            "benign_position": np.arange(benign_idx.shape[0], dtype=np.int64),
            "watermarked_idx": benign_idx,
            "cluster": labels.astype(np.int64),
            "is_seed_cluster": labels == int(seed_cluster),
            "is_suspicious_cluster": np.array([int(label) in suspicious for label in labels], dtype=bool),
            "is_poisoned": poison_mask,
        }
    )


def build_cluster_summary(
    labels: np.ndarray,
    meta: np.lib.npyio.NpzFile,
    benign_idx: np.ndarray,
    seed_cluster: int,
    suspicious_clusters: Iterable[int],
    cluster_order: dict[int, int],
    threshold_cluster_count: int,
) -> pd.DataFrame:
    assignments = build_assignment_frame(labels, benign_idx, meta, suspicious_clusters, seed_cluster)
    total_poison = max(int(assignments["is_poisoned"].sum()), 1)
    rows = []
    for cluster, group in assignments.groupby("cluster", sort=True):
        poisoned = int(group["is_poisoned"].sum())
        rows.append(
            {
                "cluster": int(cluster),
                "rows": int(len(group)),
                "is_seed_cluster": bool(cluster == seed_cluster),
                "selection_order": int(cluster_order[cluster]) if int(cluster) in cluster_order else None,
                "included_before_fixed_threshold": included_before_threshold(
                    int(cluster), cluster_order, threshold_cluster_count
                ),
                "is_suspicious_cluster": bool(group["is_suspicious_cluster"].iloc[0]),
                "oracle_poisoned": poisoned,
                "oracle_poison_rate": float(poisoned / len(group)) if len(group) else 0.0,
                "oracle_poison_share": float(poisoned / total_poison),
            }
        )
    return pd.DataFrame(rows).sort_values(["rows", "cluster"], ascending=[False, True])


def included_before_threshold(
    cluster: int,
    cluster_order: dict[int, int],
    threshold_cluster_count: int,
) -> bool | None:
    if cluster not in cluster_order:
        return False
    return int(cluster_order[cluster]) < int(threshold_cluster_count)


def sample_rows(n_rows: int, max_rows: int | None, random_state: int) -> np.ndarray:
    if max_rows is None or n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_rows, size=int(max_rows), replace=False)).astype(np.int64)


def resolve_output_dir(
    output_dir: str | Path | None,
    artifact_dir: Path,
    candidate_feature_source: str,
    top_features: int,
    selection_mode: str,
    clean_cluster_fraction: float,
    window_fraction: float,
) -> Path:
    if output_dir is not None:
        resolved = resolve_path(output_dir)
        return resolved or Path(output_dir)
    clean_tag = fraction_tag(clean_cluster_fraction)
    window_tag = fraction_tag(window_fraction)
    dirname = f"{candidate_feature_source}_top{top_features}_{selection_mode}_clean{clean_tag}_w{window_tag}"
    return artifact_dir / "optics_iterative" / dirname


def fraction_tag(value: float) -> str:
    return f"{value * 100:g}".replace(".", "p") + "p"


def validate_config(
    candidate_feature_source: str,
    top_features: int,
    importance_max_rows: int | None,
    min_samples: int,
    max_eps: float,
    xi: float,
    min_cluster_size: int | float | None,
    noise_policy: str,
    noise_chunk_size: int,
    window_fraction: float,
    clean_cluster_fraction: float,
    selection_mode: str,
    delta_z_threshold: float,
    delta_tail: str,
    surrogate_num_boost_round: int,
    max_benign_rows: int | None,
) -> None:
    if candidate_feature_source not in SUPPORTED_CANDIDATE_FEATURE_SOURCES:
        raise ValueError(f"candidate_feature_source must be one of {sorted(SUPPORTED_CANDIDATE_FEATURE_SOURCES)}")
    if top_features <= 0:
        raise ValueError("top_features must be positive")
    if importance_max_rows is not None and importance_max_rows <= 0:
        raise ValueError("importance_max_rows must be positive")
    if min_samples <= 1:
        raise ValueError("min_samples must be greater than 1")
    if max_eps <= 0:
        raise ValueError("max_eps must be positive")
    if not 0 < xi < 1:
        raise ValueError("xi must be in (0, 1)")
    if min_cluster_size is not None:
        if isinstance(min_cluster_size, float) and not 0 < min_cluster_size <= 1:
            raise ValueError("float min_cluster_size must be in (0, 1]")
        if isinstance(min_cluster_size, int) and min_cluster_size <= 1:
            raise ValueError("integer min_cluster_size must be greater than 1")
    if noise_policy not in SUPPORTED_NOISE_POLICIES:
        raise ValueError(f"noise_policy must be one of {sorted(SUPPORTED_NOISE_POLICIES)}")
    if noise_chunk_size <= 0:
        raise ValueError("noise_chunk_size must be positive")
    if not 0 < window_fraction <= 1:
        raise ValueError("window_fraction must be in (0, 1]")
    if not 0 < clean_cluster_fraction <= 1:
        raise ValueError("clean_cluster_fraction must be in (0, 1]")
    if selection_mode not in SUPPORTED_SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {sorted(SUPPORTED_SELECTION_MODES)}")
    if delta_z_threshold < 0:
        raise ValueError("delta_z_threshold must be non-negative")
    if delta_tail not in SUPPORTED_DELTA_TAILS:
        raise ValueError(f"delta_tail must be one of {sorted(SUPPORTED_DELTA_TAILS)}")
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
