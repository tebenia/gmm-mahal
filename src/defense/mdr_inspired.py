"""MDR-inspired feature-value cleaning for malware backdoor artifacts.

This reimplements the core ideas from "Make Data Reliable" for this repo's
saved attack artifacts:

1. build SHAP-guided goodware-oriented feature-value dictionaries,
2. form a thresholded similarity graph and Louvain communities,
3. choose the community whose common feature-values most reduce malware scores,
4. identify watermark-like feature-value elements enriched in that community,
5. save training row indices matching the identified watermark.

It is intentionally named MDR-inspired because it is adapted from the paper
description rather than imported from the authors' implementation.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from .retrain_evaluate import find_backdoored_model_path
from .component_trigger_matching import candidate_feature_ids, feature_names_for_width, load_watermark
from ..utils.paths import resolve_path


SUPPORTED_REMOVE_SCOPES = {"all", "benign"}


@dataclass
class MdrInspiredConfig:
    artifact_dir: str
    output_dir: str
    candidate_feature_source: str = "nonhashed"
    thresholds: tuple[int, ...] = tuple(range(3, 13))
    variance_threshold: float = 0.0
    dict_size: int = 40
    only_negative_shap: bool = True
    community_tolerance: float = 0.8
    watermark_tolerance_start: float = 0.8
    watermark_tolerance_step: float = 0.1
    window_size: int = 8
    max_combination_size: int = 3
    max_bucket_size: int = 2000
    max_edges: int = 2_000_000
    max_anti_elements: int = 100
    max_malware_probes: int = 100
    remove_scope: str = "all"
    value_round_decimals: int | None = None
    random_state: int = 42
    max_benign_rows: int | None = None
    max_features: int | None = None


@dataclass
class MdrInspiredResult:
    output_dir: str
    metadata_path: str
    threshold_diagnostics_path: str
    suspicious_community_path: str
    watermark_path: str
    remove_watermarked_idx_path: str
    best_threshold: int | None
    suspicious_community_size: int
    identified_watermark_size: int
    removed_rows: int
    removed_poisoned_rows: int | None
    poison_recall: float | None


def run_mdr_inspired(
    artifact_dir: str | Path,
    output_dir: str | Path | None = None,
    candidate_feature_source: str = "nonhashed",
    thresholds: Iterable[int] = range(3, 13),
    variance_threshold: float = 0.0,
    dict_size: int = 40,
    only_negative_shap: bool = True,
    community_tolerance: float = 0.8,
    watermark_tolerance_start: float = 0.8,
    watermark_tolerance_step: float = 0.1,
    window_size: int = 8,
    max_combination_size: int = 3,
    max_bucket_size: int = 2000,
    max_edges: int = 2_000_000,
    max_anti_elements: int = 100,
    max_malware_probes: int = 100,
    remove_scope: str = "all",
    value_round_decimals: int | None = None,
    random_state: int = 42,
    max_benign_rows: int | None = None,
    max_features: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> MdrInspiredResult | dict[str, Any]:
    thresholds = tuple(int(v) for v in thresholds)
    validate_config(
        thresholds=thresholds,
        variance_threshold=variance_threshold,
        dict_size=dict_size,
        community_tolerance=community_tolerance,
        watermark_tolerance_start=watermark_tolerance_start,
        watermark_tolerance_step=watermark_tolerance_step,
        window_size=window_size,
        max_combination_size=max_combination_size,
        max_bucket_size=max_bucket_size,
        max_edges=max_edges,
        max_anti_elements=max_anti_elements,
        max_malware_probes=max_malware_probes,
        remove_scope=remove_scope,
        value_round_decimals=value_round_decimals,
        max_benign_rows=max_benign_rows,
        max_features=max_features,
    )

    artifact_path = _resolve_existing_dir(artifact_dir)
    output_path = resolve_output_dir(output_dir, artifact_path)
    required = {
        "watermarked_X": artifact_path / "watermarked_X.npy",
        "watermarked_y": artifact_path / "watermarked_y.npy",
        "defense_metadata_npz": artifact_path / "defense_metadata.npz",
        "defense_metadata_json": artifact_path / "defense_metadata.json",
        "benign_shap": artifact_path / "backdoored_model_benign_shap.npy",
    }
    model_path = find_backdoored_model_path(artifact_path)
    if model_path is None:
        required["backdoored_model"] = artifact_path / "<missing *_backdoored model>"
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required MDR-inspired artifact(s): {', '.join(missing)}")

    if dry_run:
        return {
            "artifact_dir": str(artifact_path),
            "output_dir": str(output_path),
            "required_paths": {key: str(path) for key, path in required.items()},
            "backdoored_model_path": str(model_path),
            "thresholds": list(thresholds),
            "candidate_feature_source": candidate_feature_source,
            "remove_scope": remove_scope,
        }

    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "mdr_inspired_metadata.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_path} already exists. Pass --overwrite to replace it.")

    start_time = time.time()
    X_all = _unwrap_saved_array(np.load(required["watermarked_X"], mmap_mode="r", allow_pickle=True))
    y_all = np.asarray(_unwrap_saved_array(np.load(required["watermarked_y"], mmap_mode="r", allow_pickle=True)))
    meta = np.load(required["defense_metadata_npz"])
    defense_metadata = json.loads(required["defense_metadata_json"].read_text(encoding="utf-8"))
    benign_watermarked_idx = np.asarray(meta["benign_watermarked_idx"], dtype=np.int64)
    if max_benign_rows is not None:
        benign_watermarked_idx = benign_watermarked_idx[: min(max_benign_rows, benign_watermarked_idx.shape[0])]

    n_features = int(X_all.shape[1])
    feature_names = feature_names_for_width(n_features)
    watermark_truth = load_watermark(required["defense_metadata_json"], artifact_path / "wm_config.npy", feature_names)
    candidate_features = candidate_feature_ids(candidate_feature_source, n_features, watermark_truth)
    if max_features is not None:
        candidate_features = candidate_features[:max_features]

    shap = np.load(required["benign_shap"], mmap_mode="r")
    if shap.shape[0] < benign_watermarked_idx.shape[0]:
        raise ValueError(
            f"{required['benign_shap']} has {shap.shape[0]} rows but needs {benign_watermarked_idx.shape[0]}"
        )
    shap_benign = np.asarray(shap[: benign_watermarked_idx.shape[0]][:, candidate_features], dtype=np.float32)
    X_benign_candidates = dense_feature_matrix(X_all, benign_watermarked_idx, candidate_features)

    selected_feature_mask = np.var(X_benign_candidates, axis=0) > variance_threshold
    if not np.any(selected_feature_mask):
        raise ValueError("No candidate features survived variance filtering")
    selected_features = [fid for fid, keep in zip(candidate_features, selected_feature_mask, strict=False) if keep]
    X_benign_selected = X_benign_candidates[:, selected_feature_mask]
    shap_selected = shap_benign[:, selected_feature_mask]

    row_dicts, element_lookup = build_feature_dictionaries(
        values=X_benign_selected,
        shap_values=shap_selected,
        feature_ids=selected_features,
        dict_size=dict_size,
        only_negative_shap=only_negative_shap,
        value_round_decimals=value_round_decimals,
    )
    edge_counts, graph_stats = build_intersection_edge_counts(
        row_dicts,
        max_bucket_size=max_bucket_size,
        max_edges=max_edges,
    )

    model = lgb.Booster(model_file=str(model_path))
    malware_idx = select_malware_probe_indices(y_all, max_malware_probes, random_state=random_state)
    malware_probe = dense_rows(X_all, malware_idx)

    diagnostics, best = select_suspicious_community(
        row_dicts=row_dicts,
        edge_counts=edge_counts,
        thresholds=thresholds,
        element_lookup=element_lookup,
        model=model,
        malware_probe=malware_probe,
        tolerance=community_tolerance,
        max_anti_elements=max_anti_elements,
        random_state=random_state,
    )
    if best is None:
        suspicious_positions = np.array([], dtype=np.int64)
        suspicious_elements: list[int] = []
        best_threshold = None
    else:
        suspicious_positions = np.asarray(best["community_positions"], dtype=np.int64)
        suspicious_elements = [int(v) for v in best["anti_elements"]]
        best_threshold = int(best["threshold"])

    if suspicious_positions.size:
        watermark_elements, watermark_steps = identify_watermark(
            X_benign=X_benign_selected,
            shap_selected=shap_selected,
            selected_features=selected_features,
            suspicious_positions=suspicious_positions,
            window_size=window_size,
            tolerance_start=watermark_tolerance_start,
            tolerance_step=watermark_tolerance_step,
            max_combination_size=max_combination_size,
            value_round_decimals=value_round_decimals,
        )
    else:
        watermark_elements = []
        watermark_steps = []

    remove_watermarked_idx = select_rows_matching_watermark(
        X_all=X_all,
        watermark_elements=watermark_elements,
        candidate_rows=benign_watermarked_idx if remove_scope == "benign" else None,
        value_round_decimals=value_round_decimals,
    )
    removal_stats = compute_removal_stats(meta, remove_watermarked_idx, n_rows=X_all.shape[0])

    threshold_diagnostics_path = output_path / "threshold_diagnostics.csv"
    suspicious_community_path = output_path / "suspicious_community_rows.csv"
    watermark_path = output_path / "identified_watermark.csv"
    watermark_steps_path = output_path / "watermark_steps.csv"
    remove_watermarked_idx_path = output_path / "remove_watermarked_idx.npy"

    pd.DataFrame(diagnostics).to_csv(threshold_diagnostics_path, index=False)
    pd.DataFrame(
        {
            "benign_position": suspicious_positions,
            "watermarked_idx": benign_watermarked_idx[suspicious_positions] if suspicious_positions.size else [],
            "is_poisoned": aligned_poison_mask(meta, benign_watermarked_idx)[suspicious_positions]
            if suspicious_positions.size
            else [],
        }
    ).to_csv(suspicious_community_path, index=False)
    watermark_df = watermark_to_frame(watermark_elements, feature_names)
    watermark_df.to_csv(watermark_path, index=False)
    pd.DataFrame(watermark_steps).to_csv(watermark_steps_path, index=False)
    np.save(remove_watermarked_idx_path, remove_watermarked_idx)

    config = MdrInspiredConfig(
        artifact_dir=str(artifact_path),
        output_dir=str(output_path),
        candidate_feature_source=candidate_feature_source,
        thresholds=thresholds,
        variance_threshold=variance_threshold,
        dict_size=dict_size,
        only_negative_shap=only_negative_shap,
        community_tolerance=community_tolerance,
        watermark_tolerance_start=watermark_tolerance_start,
        watermark_tolerance_step=watermark_tolerance_step,
        window_size=window_size,
        max_combination_size=max_combination_size,
        max_bucket_size=max_bucket_size,
        max_edges=max_edges,
        max_anti_elements=max_anti_elements,
        max_malware_probes=max_malware_probes,
        remove_scope=remove_scope,
        value_round_decimals=value_round_decimals,
        random_state=random_state,
        max_benign_rows=max_benign_rows,
        max_features=max_features,
    )
    metadata = {
        "method": "MDR-inspired reimplementation from paper description",
        "config": asdict(config),
        "dataset": defense_metadata.get("dataset"),
        "input_shape": {
            "watermarked_X": list(X_all.shape),
            "benign_rows_used": int(benign_watermarked_idx.shape[0]),
            "candidate_features": int(len(candidate_features)),
            "selected_features_after_variance": int(len(selected_features)),
            "row_dictionary_count": int(len(row_dicts)),
            "row_dictionary_mean_size": float(np.mean([len(v) for v in row_dicts])) if row_dicts else 0.0,
        },
        "graph_stats": graph_stats,
        "best_threshold": best_threshold,
        "suspicious_community": summarize_best_community(best),
        "identified_watermark_size": int(len(watermark_elements)),
        "identified_watermark": watermark_df.to_dict(orient="records"),
        "removal_stats": removal_stats,
        "known_watermark": watermark_truth,
        "output_files": {
            "threshold_diagnostics": str(threshold_diagnostics_path),
            "suspicious_community": str(suspicious_community_path),
            "identified_watermark": str(watermark_path),
            "watermark_steps": str(watermark_steps_path),
            "remove_watermarked_idx": str(remove_watermarked_idx_path),
        },
        "runtime_seconds": time.time() - start_time,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return MdrInspiredResult(
        output_dir=str(output_path),
        metadata_path=str(metadata_path),
        threshold_diagnostics_path=str(threshold_diagnostics_path),
        suspicious_community_path=str(suspicious_community_path),
        watermark_path=str(watermark_path),
        remove_watermarked_idx_path=str(remove_watermarked_idx_path),
        best_threshold=best_threshold,
        suspicious_community_size=int(suspicious_positions.size),
        identified_watermark_size=int(len(watermark_elements)),
        removed_rows=int(remove_watermarked_idx.shape[0]),
        removed_poisoned_rows=removal_stats.get("removed_poisoned_rows"),
        poison_recall=removal_stats.get("poison_recall"),
    )


def build_feature_dictionaries(
    values: np.ndarray,
    shap_values: np.ndarray,
    feature_ids: list[int],
    dict_size: int,
    only_negative_shap: bool,
    value_round_decimals: int | None,
) -> tuple[list[tuple[int, ...]], dict[int, tuple[int, float]]]:
    element_to_id: dict[tuple[int, float], int] = {}
    element_lookup: dict[int, tuple[int, float]] = {}
    row_dicts: list[tuple[int, ...]] = []
    for row_values, row_shap in zip(values, shap_values, strict=True):
        if only_negative_shap:
            candidate_positions = np.flatnonzero(row_shap < 0)
            if candidate_positions.size == 0:
                candidate_positions = np.arange(row_shap.shape[0])
        else:
            candidate_positions = np.arange(row_shap.shape[0])
        order = candidate_positions[np.argsort(row_shap[candidate_positions])[:dict_size]]
        row_elements = []
        for pos in order:
            feature_id = int(feature_ids[pos])
            value = canonical_value(row_values[pos], value_round_decimals)
            key = (feature_id, value)
            element_id = element_to_id.get(key)
            if element_id is None:
                element_id = len(element_to_id)
                element_to_id[key] = element_id
                element_lookup[element_id] = key
            row_elements.append(element_id)
        row_dicts.append(tuple(sorted(set(row_elements))))
    return row_dicts, element_lookup


def build_intersection_edge_counts(
    row_dicts: list[tuple[int, ...]],
    max_bucket_size: int,
    max_edges: int,
) -> tuple[dict[tuple[int, int], int], dict[str, int]]:
    inverted: dict[int, list[int]] = defaultdict(list)
    for row_id, elements in enumerate(row_dicts):
        for element in elements:
            inverted[element].append(row_id)

    edge_counts: dict[tuple[int, int], int] = {}
    skipped_large_buckets = 0
    used_buckets = 0
    for row_ids in inverted.values():
        if len(row_ids) < 2:
            continue
        if len(row_ids) > max_bucket_size:
            skipped_large_buckets += 1
            continue
        used_buckets += 1
        for i, j in itertools.combinations(row_ids, 2):
            key = (i, j) if i < j else (j, i)
            edge_counts[key] = edge_counts.get(key, 0) + 1
            if len(edge_counts) > max_edges:
                raise RuntimeError(
                    f"MDR graph exceeded max_edges={max_edges}. "
                    "Increase --thresholds, lower --dict-size, lower --max-bucket-size, "
                    "or use --max-benign-rows for a smaller diagnostic run."
                )
    return edge_counts, {
        "unique_elements": int(len(inverted)),
        "used_buckets": int(used_buckets),
        "skipped_large_buckets": int(skipped_large_buckets),
        "candidate_edges": int(len(edge_counts)),
    }


def select_suspicious_community(
    row_dicts: list[tuple[int, ...]],
    edge_counts: dict[tuple[int, int], int],
    thresholds: tuple[int, ...],
    element_lookup: dict[int, tuple[int, float]],
    model: lgb.Booster,
    malware_probe: np.ndarray,
    tolerance: float,
    max_anti_elements: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    diagnostics: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        graph = nx.Graph()
        graph.add_weighted_edges_from((i, j, count) for (i, j), count in edge_counts.items() if count >= threshold)
        if graph.number_of_edges() == 0:
            diagnostics.append(
                {
                    "threshold": int(threshold),
                    "communities": 0,
                    "graph_nodes": int(graph.number_of_nodes()),
                    "graph_edges": 0,
                    "best_apv": np.nan,
                    "best_community_size": 0,
                    "best_anti_elements": 0,
                }
            )
            continue
        communities = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=random_state)
        threshold_best = None
        for community_id, community in enumerate(communities):
            positions = np.array(sorted(community), dtype=np.int64)
            anti_elements = extract_common_elements(row_dicts, positions, tolerance=tolerance, max_elements=max_anti_elements)
            if not anti_elements:
                continue
            apv = average_prediction_with_elements(
                model=model,
                malware_probe=malware_probe,
                elements=[element_lookup[element] for element in anti_elements],
            )
            item = {
                "threshold": int(threshold),
                "community_id": int(community_id),
                "community_positions": positions,
                "community_size": int(positions.shape[0]),
                "anti_elements": anti_elements,
                "anti_elements_count": int(len(anti_elements)),
                "apv": float(apv),
            }
            if threshold_best is None or item["apv"] < threshold_best["apv"]:
                threshold_best = item
            if best is None or item["apv"] < best["apv"]:
                best = item
        diagnostics.append(
            {
                "threshold": int(threshold),
                "communities": int(len(communities)),
                "graph_nodes": int(graph.number_of_nodes()),
                "graph_edges": int(graph.number_of_edges()),
                "best_apv": threshold_best["apv"] if threshold_best is not None else np.nan,
                "best_community_id": threshold_best["community_id"] if threshold_best is not None else None,
                "best_community_size": threshold_best["community_size"] if threshold_best is not None else 0,
                "best_anti_elements": threshold_best["anti_elements_count"] if threshold_best is not None else 0,
            }
        )
    return diagnostics, best


def extract_common_elements(
    row_dicts: list[tuple[int, ...]],
    positions: np.ndarray,
    tolerance: float,
    max_elements: int,
) -> list[int]:
    min_count = max(1, int(math.ceil(float(tolerance) * positions.shape[0])))
    counts: Counter[int] = Counter()
    for position in positions:
        counts.update(row_dicts[int(position)])
    elements = [element for element, count in counts.items() if count >= min_count]
    elements.sort(key=lambda element: counts[element], reverse=True)
    return elements[:max_elements]


def identify_watermark(
    X_benign: np.ndarray,
    shap_selected: np.ndarray,
    selected_features: list[int],
    suspicious_positions: np.ndarray,
    window_size: int,
    tolerance_start: float,
    tolerance_step: float,
    max_combination_size: int,
    value_round_decimals: int | None,
) -> tuple[list[tuple[int, float]], list[dict[str, Any]]]:
    cm_positions = np.asarray(suspicious_positions, dtype=np.int64)
    all_positions = np.arange(X_benign.shape[0], dtype=np.int64)
    cb_mask = np.ones(X_benign.shape[0], dtype=bool)
    cb_mask[cm_positions] = False
    cb_positions = all_positions[cb_mask]
    feature_scores = shap_selected[cm_positions].sum(axis=0)
    feature_order = np.argsort(feature_scores)
    feature_id_to_local = {int(feature_id): pos for pos, feature_id in enumerate(selected_features)}
    watermark: list[tuple[int, float]] = []
    best_score = 0.0
    tolerance = float(tolerance_start)
    steps: list[dict[str, Any]] = []

    for start in range(0, max(0, len(feature_order) - window_size + 1), window_size):
        if cm_positions.size == 0:
            break
        window = feature_order[start : start + window_size]
        candidate_elements = element_select(
            X_benign=X_benign,
            positions=cm_positions,
            feature_positions=window,
            feature_ids=selected_features,
            tolerance=tolerance,
            value_round_decimals=value_round_decimals,
        )
        combinations = generate_element_combinations(candidate_elements, max_combination_size=max_combination_size)
        best_combo = None
        best_combo_score = -np.inf
        best_combo_counts = (0, 0)
        for combo in combinations:
            merged = merge_elements(watermark, combo)
            if not merged:
                continue
            cm_count = count_matching_elements(
                X_benign,
                cm_positions,
                merged,
                feature_id_to_local,
                value_round_decimals,
            )
            cb_count = count_matching_elements(
                X_benign,
                cb_positions,
                merged,
                feature_id_to_local,
                value_round_decimals,
            )
            score = float(cm_count / max(cb_count, 1))
            if score > best_combo_score:
                best_combo = combo
                best_combo_score = score
                best_combo_counts = (cm_count, cb_count)
        accepted = best_combo is not None and best_combo_score >= best_score
        if accepted:
            watermark = merge_elements(watermark, best_combo)
            best_score = float(best_combo_score)
            keep = rows_match_elements(
                X_benign,
                cm_positions,
                watermark,
                feature_id_to_local,
                value_round_decimals,
            )
            cm_positions = cm_positions[keep]
        steps.append(
            {
                "window_start": int(start),
                "tolerance": float(tolerance),
                "candidate_elements": int(len(candidate_elements)),
                "candidate_combinations": int(len(combinations)),
                "accepted": bool(accepted),
                "best_score": float(best_combo_score) if np.isfinite(best_combo_score) else None,
                "best_cm_count": int(best_combo_counts[0]),
                "best_cb_count": int(best_combo_counts[1]),
                "watermark_size": int(len(watermark)),
                "remaining_suspicious_rows": int(cm_positions.size),
            }
        )
        tolerance = min(1.0, tolerance + tolerance_step)
    return watermark, steps


def element_select(
    X_benign: np.ndarray,
    positions: np.ndarray,
    feature_positions: np.ndarray,
    feature_ids: list[int],
    tolerance: float,
    value_round_decimals: int | None,
) -> list[tuple[int, float]]:
    min_count = max(1, int(math.ceil(float(tolerance) * positions.shape[0])))
    elements = []
    for pos in feature_positions:
        values = canonical_array(X_benign[positions, pos], value_round_decimals)
        unique_values, counts = np.unique(values, return_counts=True)
        keep = np.flatnonzero(counts >= min_count)
        for idx in keep:
            elements.append((int(feature_ids[pos]), float(unique_values[idx])))
    return elements


def generate_element_combinations(
    elements: list[tuple[int, float]],
    max_combination_size: int,
) -> list[list[tuple[int, float]]]:
    combos: list[list[tuple[int, float]]] = []
    max_size = min(max_combination_size, len(elements))
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(elements, size):
            features = [feature_id for feature_id, _ in combo]
            if len(set(features)) == len(features):
                combos.append(list(combo))
    return combos


def select_rows_matching_watermark(
    X_all,
    watermark_elements: list[tuple[int, float]],
    candidate_rows: np.ndarray | None,
    value_round_decimals: int | None,
) -> np.ndarray:
    if not watermark_elements:
        return np.array([], dtype=np.int64)
    rows = np.asarray(candidate_rows, dtype=np.int64) if candidate_rows is not None else np.arange(X_all.shape[0])
    feature_ids = [feature_id for feature_id, _ in watermark_elements]
    values = [canonical_value(value, value_round_decimals) for _, value in watermark_elements]
    matched: list[np.ndarray] = []
    batch_size = 16384
    for start in range(0, rows.shape[0], batch_size):
        end = min(start + batch_size, rows.shape[0])
        batch_rows = rows[start:end]
        X_batch = dense_feature_matrix(X_all, batch_rows, feature_ids)
        mask = np.ones(batch_rows.shape[0], dtype=bool)
        for pos, expected in enumerate(values):
            mask &= canonical_array(X_batch[:, pos], value_round_decimals) == expected
        if np.any(mask):
            matched.append(batch_rows[mask])
    if not matched:
        return np.array([], dtype=np.int64)
    return np.concatenate(matched).astype(np.int64, copy=False)


def count_matching_elements(
    X_benign: np.ndarray,
    positions: np.ndarray,
    elements: list[tuple[int, float]],
    feature_id_to_local: dict[int, int],
    value_round_decimals: int | None,
) -> int:
    if positions.size == 0 or not elements:
        return 0
    mask = rows_match_elements(X_benign, positions, elements, feature_id_to_local, value_round_decimals)
    return int(mask.sum())


def rows_match_elements(
    X_benign: np.ndarray,
    positions: np.ndarray,
    elements: list[tuple[int, float]],
    feature_id_to_local: dict[int, int],
    value_round_decimals: int | None,
) -> np.ndarray:
    if positions.size == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(positions.shape[0], dtype=bool)
    for feature_id, value in elements:
        local_pos = feature_id_to_local[feature_id]
        col = canonical_array(X_benign[positions, local_pos], value_round_decimals)
        mask &= col == canonical_value(value, value_round_decimals)
    return mask


def merge_elements(
    base: list[tuple[int, float]],
    extra: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    merged: dict[int, float] = {feature_id: value for feature_id, value in base}
    for feature_id, value in extra:
        if feature_id in merged and merged[feature_id] != value:
            return []
        merged[int(feature_id)] = float(value)
    return sorted(merged.items())


def average_prediction_with_elements(
    model: lgb.Booster,
    malware_probe: np.ndarray,
    elements: list[tuple[int, float]],
) -> float:
    if malware_probe.shape[0] == 0:
        raise ValueError("No malware probe rows are available")
    probe = np.array(malware_probe, copy=True)
    for feature_id, value in elements:
        if feature_id < probe.shape[1]:
            probe[:, feature_id] = value
    preds = np.asarray(model.predict(probe))
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds = preds[:, 1]
    return float(np.mean(preds))


def dense_feature_matrix(X, rows: np.ndarray, feature_ids: list[int]) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X[rows][:, feature_ids].toarray(), dtype=np.float32)
    return np.asarray(X[np.ix_(rows, feature_ids)], dtype=np.float32)


def dense_rows(X, rows: np.ndarray) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X[rows].toarray(), dtype=np.float32)
    return np.asarray(X[rows], dtype=np.float32)


def select_malware_probe_indices(y_all: np.ndarray, max_malware_probes: int, random_state: int) -> np.ndarray:
    malware_idx = np.flatnonzero(np.asarray(y_all).astype(int) == 1)
    if malware_idx.shape[0] <= max_malware_probes:
        return malware_idx.astype(np.int64)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(malware_idx, size=max_malware_probes, replace=False)).astype(np.int64)


def canonical_value(value: Any, value_round_decimals: int | None) -> float:
    value = float(value)
    if value_round_decimals is not None:
        value = round(value, value_round_decimals)
    return value


def canonical_array(values: np.ndarray, value_round_decimals: int | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if value_round_decimals is not None:
        arr = np.round(arr, value_round_decimals)
    return arr


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


def watermark_to_frame(watermark_elements: list[tuple[int, float]], feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_idx": int(feature_id),
                "feature_name": feature_names[feature_id] if feature_id < len(feature_names) else f"feature_{feature_id}",
                "value": float(value),
            }
            for feature_id, value in watermark_elements
        ]
    )


def summarize_best_community(best: dict[str, Any] | None) -> dict[str, Any] | None:
    if best is None:
        return None
    return {
        "threshold": int(best["threshold"]),
        "community_id": int(best["community_id"]),
        "community_size": int(best["community_size"]),
        "anti_elements_count": int(best["anti_elements_count"]),
        "apv": float(best["apv"]),
    }


def resolve_output_dir(output_dir: str | Path | None, artifact_dir: Path) -> Path:
    if output_dir is not None:
        resolved = resolve_path(output_dir)
        return resolved or Path(output_dir)
    return artifact_dir / "mdr_inspired"


def validate_config(
    thresholds: tuple[int, ...],
    variance_threshold: float,
    dict_size: int,
    community_tolerance: float,
    watermark_tolerance_start: float,
    watermark_tolerance_step: float,
    window_size: int,
    max_combination_size: int,
    max_bucket_size: int,
    max_edges: int,
    max_anti_elements: int,
    max_malware_probes: int,
    remove_scope: str,
    value_round_decimals: int | None,
    max_benign_rows: int | None,
    max_features: int | None,
) -> None:
    if not thresholds or any(v <= 0 for v in thresholds):
        raise ValueError("thresholds must contain positive integers")
    if variance_threshold < 0:
        raise ValueError("variance_threshold must be non-negative")
    if dict_size <= 0:
        raise ValueError("dict_size must be positive")
    for name, value in [
        ("community_tolerance", community_tolerance),
        ("watermark_tolerance_start", watermark_tolerance_start),
    ]:
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if watermark_tolerance_step < 0:
        raise ValueError("watermark_tolerance_step must be non-negative")
    if window_size <= 0 or max_combination_size <= 0:
        raise ValueError("window_size and max_combination_size must be positive")
    if max_bucket_size <= 1 or max_edges <= 0 or max_anti_elements <= 0 or max_malware_probes <= 0:
        raise ValueError("max bucket/edge/anti/probe limits must be positive")
    if remove_scope not in SUPPORTED_REMOVE_SCOPES:
        raise ValueError(f"remove_scope must be one of {sorted(SUPPORTED_REMOVE_SCOPES)}")
    if value_round_decimals is not None and value_round_decimals < 0:
        raise ValueError("value_round_decimals must be non-negative")
    if max_benign_rows is not None and max_benign_rows <= 0:
        raise ValueError("max_benign_rows must be positive")
    if max_features is not None and max_features <= 0:
        raise ValueError("max_features must be positive")


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
