"""Artifact-level detectability diagnostics for clean-label backdoor runs.

These diagnostics are not defenses by themselves. They measure how visible a
saved trigger is in the poisoned training artifact: marginal trigger-value
rarity, joint trigger co-occurrence, trigger-SHAP concentration, and a simple
nearest-neighbor density proxy in the trigger feature subspace.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from ..defense.severi_detectors import (
    feature_names_for_width,
    load_benign_alignment,
    load_saved_array,
    selected_dense_rows,
    valid_feature_ids,
)
from ..utils.paths import resolve_path


DEFAULT_RECALL_PERCENTS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
DEFAULT_JOINT_THRESHOLDS = (0.25, 0.5, 0.75, 1.0)
EPSILON = 1e-12


@dataclass
class DetectabilityDiagnosticsConfig:
    artifact_dir: str
    output_dir: str
    exact_atol: float = 1e-6
    batch_size: int = 8192
    max_benign_rows: int | None = None
    skip_shap: bool = False
    skip_density: bool = False
    knn_neighbors: int = 10
    max_knn_reference_rows: int = 10000
    recall_percents: list[float] | None = None
    joint_thresholds: list[float] | None = None
    random_state: int = 42
    save_row_scores: bool = True


@dataclass
class DetectabilityDiagnosticsResult:
    output_dir: str
    summary_path: str
    footprint_summary_path: str
    marginal_rarity_path: str
    value_frequency_path: str
    joint_rarity_path: str
    cooccurrence_support_path: str
    shap_footprint_path: str | None
    score_metrics_path: str
    row_scores_path: str | None
    metadata_path: str
    benign_rows_scored: int
    poisoned_benign_rows: int
    watermark_feature_count: int
    joint_all_clean_frequency: float | None
    trigger_match_fraction_auroc: float | None
    trigger_rarity_score_auroc: float | None
    shap_trigger_abs_ratio_auroc: float | None
    knn_trigger_mean_distance_auroc: float | None


def run_detectability_diagnostics(
    artifact_dir: str | Path,
    output_dir: str | Path | None = None,
    exact_atol: float = 1e-6,
    batch_size: int = 8192,
    max_benign_rows: int | None = None,
    skip_shap: bool = False,
    skip_density: bool = False,
    knn_neighbors: int = 10,
    max_knn_reference_rows: int = 10000,
    recall_percents: list[float] | tuple[float, ...] = DEFAULT_RECALL_PERCENTS,
    joint_thresholds: list[float] | tuple[float, ...] = DEFAULT_JOINT_THRESHOLDS,
    random_state: int = 42,
    save_row_scores: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> DetectabilityDiagnosticsResult | dict[str, Any]:
    artifact_path = _resolve_existing_dir(artifact_dir)
    output_path = resolve_output_dir(artifact_path=artifact_path, output_dir=output_dir, max_benign_rows=max_benign_rows)
    recall_values = [float(value) for value in recall_percents]
    threshold_values = [float(value) for value in joint_thresholds]
    validate_config(
        exact_atol=exact_atol,
        batch_size=batch_size,
        max_benign_rows=max_benign_rows,
        knn_neighbors=knn_neighbors,
        max_knn_reference_rows=max_knn_reference_rows,
        recall_percents=recall_values,
        joint_thresholds=threshold_values,
    )

    required = {
        "watermarked_X": artifact_path / "watermarked_X.npy",
        "watermarked_y": artifact_path / "watermarked_y.npy",
        "wm_config": artifact_path / "wm_config.npy",
        "defense_metadata_npz": artifact_path / "defense_metadata.npz",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required detectability artifact(s): {}. Re-run the attack with "
            "--save-attack-artifacts --save-defense-inputs.".format(", ".join(missing))
        )

    summary_path = output_path / "detectability_summary.csv"
    footprint_summary_path = output_path / "trigger_footprint_summary.csv"
    marginal_path = output_path / "trigger_marginal_rarity.csv"
    value_frequency_path = output_path / "trigger_value_frequency.csv"
    joint_path = output_path / "trigger_joint_rarity.csv"
    cooccurrence_support_path = output_path / "trigger_cooccurrence_support.csv"
    shap_footprint_path = output_path / "trigger_shap_footprint.csv"
    metrics_path = output_path / "detectability_score_metrics.csv"
    row_scores_path = output_path / "detectability_row_scores.csv"
    metadata_path = output_path / "detectability_metadata.json"
    knn_reference_path = output_path / "knn_reference_benign_positions.npy"

    if dry_run:
        return {
            "artifact_dir": str(artifact_path),
            "output_dir": str(output_path),
            "required_paths": {key: str(path) for key, path in required.items()},
            "optional_paths": {
                "backdoored_model_benign_shap": str(artifact_path / "backdoored_model_benign_shap.npy"),
                "defense_metadata_json": str(artifact_path / "defense_metadata.json"),
            },
            "output_files": {
                "trigger_footprint_summary": str(footprint_summary_path),
                "trigger_value_frequency": str(value_frequency_path),
                "trigger_cooccurrence_support": str(cooccurrence_support_path),
                "trigger_shap_footprint": str(shap_footprint_path),
            },
            "config": asdict(
                DetectabilityDiagnosticsConfig(
                    artifact_dir=str(artifact_path),
                    output_dir=str(output_path),
                    exact_atol=exact_atol,
                    batch_size=batch_size,
                    max_benign_rows=max_benign_rows,
                    skip_shap=skip_shap,
                    skip_density=skip_density,
                    knn_neighbors=knn_neighbors,
                    max_knn_reference_rows=max_knn_reference_rows,
                    recall_percents=recall_values,
                    joint_thresholds=threshold_values,
                    random_state=random_state,
                    save_row_scores=save_row_scores,
                )
            ),
        }

    output_path.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_path} already exists. Pass --overwrite to replace it.")

    start_time = time.time()
    rng = np.random.default_rng(random_state)
    X_all = load_saved_array(required["watermarked_X"], mmap_mode="r")
    y_all = np.asarray(load_saved_array(required["watermarked_y"], mmap_mode="r")).reshape(-1)
    wm_config = np.load(required["wm_config"], allow_pickle=True).item()
    meta = np.load(required["defense_metadata_npz"])

    if X_all.shape[0] != y_all.shape[0]:
        raise ValueError(f"watermarked_X rows {X_all.shape[0]} do not match watermarked_y rows {y_all.shape[0]}")

    benign_idx_full, poison_mask_full = load_benign_alignment(meta, y_all)
    benign_positions = choose_benign_positions(
        poison_mask=poison_mask_full,
        max_benign_rows=max_benign_rows,
        rng=rng,
    )
    benign_idx = benign_idx_full[benign_positions]
    poison_mask_benign = poison_mask_full[benign_positions]

    n_features = int(X_all.shape[1])
    feature_names = feature_names_for_width(n_features)
    watermark_feature_ids = valid_feature_ids(wm_config.get("wm_feat_ids", []), n_features=n_features)
    if watermark_feature_ids.size == 0:
        raise ValueError("wm_config.npy does not contain valid wm_feat_ids")

    X_watermark = selected_dense_rows(
        X_all,
        row_indices=benign_idx,
        feature_indices=watermark_feature_ids,
        batch_size=batch_size,
    )
    watermark_values, value_sources = resolve_watermark_values(
        wm_config=wm_config,
        feature_ids=watermark_feature_ids,
        feature_names=feature_names,
        X_watermark=X_watermark,
        poison_mask=poison_mask_benign,
    )
    exact_matches = np.isclose(X_watermark, watermark_values.reshape(1, -1), rtol=0.0, atol=exact_atol)

    marginal_df = build_marginal_rarity_df(
        X_watermark=X_watermark,
        exact_matches=exact_matches,
        poison_mask=poison_mask_benign,
        feature_ids=watermark_feature_ids,
        feature_names=feature_names,
        watermark_values=watermark_values,
        value_sources=value_sources,
    )
    clean_freq = marginal_df["clean_exact_frequency"].to_numpy(dtype=np.float64)
    rarity_weights = -np.log10(np.clip(clean_freq, EPSILON, 1.0))
    match_count = np.sum(exact_matches, axis=1).astype(np.int16, copy=False)
    match_fraction = match_count / float(watermark_feature_ids.shape[0])
    rarity_score = exact_matches.astype(np.float64) @ rarity_weights

    row_df = pd.DataFrame(
        {
            "benign_position": benign_positions.astype(np.int64, copy=False),
            "watermarked_idx": benign_idx.astype(np.int64, copy=False),
            "is_poisoned": poison_mask_benign.astype(bool, copy=False),
            "trigger_match_count": match_count.astype(np.int64, copy=False),
            "trigger_match_fraction": match_fraction,
            "trigger_all_match": match_count == watermark_feature_ids.shape[0],
            "trigger_rarity_score": rarity_score,
        }
    )
    joint_df, joint_summary = build_joint_rarity_df(
        match_count=match_count,
        poison_mask=poison_mask_benign,
        watermark_feature_count=int(watermark_feature_ids.shape[0]),
        clean_frequencies=clean_freq,
        thresholds=threshold_values,
    )

    metric_rows: list[dict[str, Any]] = []
    summary_scores: dict[str, Any] = {}
    add_score_diagnostics(
        score_name="trigger_match_fraction",
        scores=row_df["trigger_match_fraction"].to_numpy(dtype=np.float64),
        labels=poison_mask_benign,
        recall_percents=recall_values,
        metric_rows=metric_rows,
        summary_scores=summary_scores,
    )
    add_score_diagnostics(
        score_name="trigger_rarity_score",
        scores=row_df["trigger_rarity_score"].to_numpy(dtype=np.float64),
        labels=poison_mask_benign,
        recall_percents=recall_values,
        metric_rows=metric_rows,
        summary_scores=summary_scores,
    )

    shap_path = artifact_path / "backdoored_model_benign_shap.npy"
    shap_status = "skipped" if skip_shap else "missing"
    shap_feature_df = pd.DataFrame()
    if not skip_shap and shap_path.exists():
        shap_scores = compute_shap_trigger_footprint(
            shap_path=shap_path,
            benign_positions=benign_positions,
            poison_mask=poison_mask_benign,
            watermark_feature_ids=watermark_feature_ids,
            feature_names=feature_names,
            watermark_values=watermark_values,
            n_features=n_features,
            batch_size=batch_size,
        )
        shap_feature_df = shap_scores["feature_df"]
        row_df["shap_trigger_signed_sum"] = shap_scores["trigger_signed_sum"]
        row_df["shap_trigger_signed_mean"] = shap_scores["trigger_signed_mean"]
        row_df["shap_trigger_abs_sum"] = shap_scores["trigger_abs_sum"]
        row_df["shap_trigger_abs_mean"] = shap_scores["trigger_abs_mean"]
        row_df["shap_total_abs_sum"] = shap_scores["total_abs_sum"]
        row_df["shap_trigger_abs_ratio"] = shap_scores["trigger_abs_ratio"]
        add_score_diagnostics(
            score_name="shap_trigger_abs_sum",
            scores=shap_scores["trigger_abs_sum"],
            labels=poison_mask_benign,
            recall_percents=recall_values,
            metric_rows=metric_rows,
            summary_scores=summary_scores,
        )
        add_score_diagnostics(
            score_name="shap_trigger_abs_ratio",
            scores=shap_scores["trigger_abs_ratio"],
            labels=poison_mask_benign,
            recall_percents=recall_values,
            metric_rows=metric_rows,
            summary_scores=summary_scores,
        )
        summary_scores.update(group_difference_summary("shap_trigger_signed_sum", shap_scores["trigger_signed_sum"], poison_mask_benign))
        summary_scores.update(group_difference_summary("shap_trigger_abs_sum", shap_scores["trigger_abs_sum"], poison_mask_benign))
        summary_scores.update(group_difference_summary("shap_trigger_abs_ratio", shap_scores["trigger_abs_ratio"], poison_mask_benign))
        shap_status = "computed"

    density_status = "skipped" if skip_density else "computed"
    knn_reference_positions = np.asarray([], dtype=np.int64)
    if not skip_density:
        knn_scores, knn_reference_positions = compute_knn_trigger_density_scores(
            X_watermark=X_watermark,
            poison_mask=poison_mask_benign,
            knn_neighbors=knn_neighbors,
            max_reference_rows=max_knn_reference_rows,
            rng=rng,
        )
        row_df["knn_trigger_mean_distance"] = knn_scores
        np.save(knn_reference_path, benign_positions[knn_reference_positions].astype(np.int64, copy=False))
        add_score_diagnostics(
            score_name="knn_trigger_mean_distance",
            scores=knn_scores,
            labels=poison_mask_benign,
            recall_percents=recall_values,
            metric_rows=metric_rows,
            summary_scores=summary_scores,
        )
        summary_scores.update(group_difference_summary("knn_trigger_mean_distance", knn_scores, poison_mask_benign))

    context = parse_artifact_context(artifact_path)
    clean_mask = ~poison_mask_benign
    summary = {
        **context,
        "artifact_dir": str(artifact_path),
        "diagnostic_output_dir": str(output_path),
        "total_train_rows": int(X_all.shape[0]),
        "total_features": n_features,
        "benign_rows_full": int(benign_idx_full.shape[0]),
        "benign_rows_scored": int(benign_idx.shape[0]),
        "clean_benign_rows_scored": int(np.sum(clean_mask)),
        "poisoned_benign_rows_scored": int(np.sum(poison_mask_benign)),
        "watermark_feature_count": int(watermark_feature_ids.shape[0]),
        "exact_atol": float(exact_atol),
        "marginal_clean_frequency_min": safe_float(np.nanmin(clean_freq)) if clean_freq.size else np.nan,
        "marginal_clean_frequency_median": safe_float(np.nanmedian(clean_freq)) if clean_freq.size else np.nan,
        "marginal_clean_frequency_mean": safe_float(np.nanmean(clean_freq)) if clean_freq.size else np.nan,
        "marginal_clean_frequency_max": safe_float(np.nanmax(clean_freq)) if clean_freq.size else np.nan,
        "marginal_neg_log10_frequency_sum": safe_float(np.sum(rarity_weights)),
        "marginal_neg_log10_frequency_mean": safe_float(np.mean(rarity_weights)),
        "clean_trigger_match_fraction_mean": group_mean(match_fraction, clean_mask),
        "poison_trigger_match_fraction_mean": group_mean(match_fraction, poison_mask_benign),
        "poison_minus_clean_trigger_match_fraction_mean": group_mean(match_fraction, poison_mask_benign)
        - group_mean(match_fraction, clean_mask),
        "shap_status": shap_status,
        "density_status": density_status,
        "knn_neighbors": int(knn_neighbors),
        "knn_reference_rows": int(knn_reference_positions.shape[0]),
        **joint_summary,
        **summary_scores,
    }
    footprint_summary = build_trigger_footprint_summary(summary)

    marginal_df.to_csv(marginal_path, index=False)
    marginal_df.to_csv(value_frequency_path, index=False)
    joint_df.to_csv(joint_path, index=False)
    joint_df.to_csv(cooccurrence_support_path, index=False)
    shap_feature_df.to_csv(shap_footprint_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame([footprint_summary]).to_csv(footprint_summary_path, index=False)
    saved_row_scores_path: Path | None = row_scores_path if save_row_scores else None
    if saved_row_scores_path is not None:
        row_df.to_csv(saved_row_scores_path, index=False)

    metadata_json_path = artifact_path / "defense_metadata.json"
    artifact_metadata = {}
    if metadata_json_path.exists():
        artifact_metadata = json.loads(metadata_json_path.read_text(encoding="utf-8"))

    metadata = {
        "method_note": (
            "Detectability diagnostics for saved attack artifacts. Ground-truth poison labels "
            "are used only to evaluate the observable scores; they are not used to construct "
            "trigger rarity, joint co-occurrence, SHAP concentration, or kNN density scores."
        ),
        "config": asdict(
            DetectabilityDiagnosticsConfig(
                artifact_dir=str(artifact_path),
                output_dir=str(output_path),
                exact_atol=exact_atol,
                batch_size=batch_size,
                max_benign_rows=max_benign_rows,
                skip_shap=skip_shap,
                skip_density=skip_density,
                knn_neighbors=knn_neighbors,
                max_knn_reference_rows=max_knn_reference_rows,
                recall_percents=recall_values,
                joint_thresholds=threshold_values,
                random_state=random_state,
                save_row_scores=save_row_scores,
            )
        ),
        "artifact_context": context,
        "artifact_metadata": artifact_metadata,
        "output_files": {
            "detectability_summary": str(summary_path),
            "trigger_footprint_summary": str(footprint_summary_path),
            "trigger_marginal_rarity": str(marginal_path),
            "trigger_value_frequency": str(value_frequency_path),
            "trigger_joint_rarity": str(joint_path),
            "trigger_cooccurrence_support": str(cooccurrence_support_path),
            "trigger_shap_footprint": str(shap_footprint_path) if shap_status == "computed" else None,
            "detectability_score_metrics": str(metrics_path),
            "detectability_row_scores": str(saved_row_scores_path) if saved_row_scores_path is not None else None,
            "knn_reference_benign_positions": str(knn_reference_path) if not skip_density else None,
        },
        "runtime_seconds": time.time() - start_time,
    }
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")

    return DetectabilityDiagnosticsResult(
        output_dir=str(output_path),
        summary_path=str(summary_path),
        footprint_summary_path=str(footprint_summary_path),
        marginal_rarity_path=str(marginal_path),
        value_frequency_path=str(value_frequency_path),
        joint_rarity_path=str(joint_path),
        cooccurrence_support_path=str(cooccurrence_support_path),
        shap_footprint_path=str(shap_footprint_path) if shap_status == "computed" else None,
        score_metrics_path=str(metrics_path),
        row_scores_path=str(saved_row_scores_path) if saved_row_scores_path is not None else None,
        metadata_path=str(metadata_path),
        benign_rows_scored=int(benign_idx.shape[0]),
        poisoned_benign_rows=int(np.sum(poison_mask_benign)),
        watermark_feature_count=int(watermark_feature_ids.shape[0]),
        joint_all_clean_frequency=summary.get("joint_threshold_100p_clean_frequency"),
        trigger_match_fraction_auroc=summary.get("trigger_match_fraction_auroc"),
        trigger_rarity_score_auroc=summary.get("trigger_rarity_score_auroc"),
        shap_trigger_abs_ratio_auroc=summary.get("shap_trigger_abs_ratio_auroc"),
        knn_trigger_mean_distance_auroc=summary.get("knn_trigger_mean_distance_auroc"),
    )


def validate_config(
    *,
    exact_atol: float,
    batch_size: int,
    max_benign_rows: int | None,
    knn_neighbors: int,
    max_knn_reference_rows: int,
    recall_percents: list[float],
    joint_thresholds: list[float],
) -> None:
    if exact_atol < 0:
        raise ValueError("--exact-atol must be non-negative")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if max_benign_rows is not None and max_benign_rows <= 1:
        raise ValueError("--max-benign-rows must be greater than 1")
    if knn_neighbors <= 0:
        raise ValueError("--knn-neighbors must be positive")
    if max_knn_reference_rows <= 1:
        raise ValueError("--max-knn-reference-rows must be greater than 1")
    if not recall_percents or any(value <= 0 or value > 100 for value in recall_percents):
        raise ValueError("--recall-percents values must be in (0, 100]")
    if not joint_thresholds or any(value <= 0 or value > 1 for value in joint_thresholds):
        raise ValueError("--joint-thresholds values must be in (0, 1]")


def choose_benign_positions(
    *,
    poison_mask: np.ndarray,
    max_benign_rows: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    all_positions = np.arange(poison_mask.shape[0], dtype=np.int64)
    if max_benign_rows is None or max_benign_rows >= all_positions.shape[0]:
        return all_positions

    poison_positions = np.flatnonzero(poison_mask).astype(np.int64, copy=False)
    clean_positions = np.flatnonzero(~poison_mask).astype(np.int64, copy=False)
    max_rows = int(max_benign_rows)
    if poison_positions.shape[0] >= max_rows:
        chosen = rng.choice(poison_positions, size=max_rows, replace=False)
        return np.sort(chosen).astype(np.int64, copy=False)

    clean_needed = max_rows - poison_positions.shape[0]
    clean_needed = min(clean_needed, clean_positions.shape[0])
    chosen_clean = rng.choice(clean_positions, size=clean_needed, replace=False)
    return np.sort(np.concatenate([chosen_clean, poison_positions])).astype(np.int64, copy=False)


def resolve_watermark_values(
    *,
    wm_config: dict[str, Any],
    feature_ids: np.ndarray,
    feature_names: list[str],
    X_watermark: np.ndarray,
    poison_mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    config_values = wm_config.get("watermark_features", {}) or {}
    by_name: dict[str, Any] = {}
    by_id: dict[int, Any] = {}
    if isinstance(config_values, dict):
        for raw_key, raw_value in config_values.items():
            key = str(raw_key)
            by_name[key] = raw_value
            try:
                by_id[int(key)] = raw_value
            except ValueError:
                pass
    elif isinstance(config_values, (list, tuple, np.ndarray)) and len(config_values) == len(feature_ids):
        by_id = {int(fid): value for fid, value in zip(feature_ids, config_values)}

    values = np.full(feature_ids.shape[0], np.nan, dtype=np.float64)
    sources: list[str] = []
    for i, feature_id in enumerate(feature_ids):
        fid = int(feature_id)
        feature_name = feature_names[fid] if fid < len(feature_names) else f"feature_{fid}"
        if fid in by_id:
            values[i] = float(by_id[fid])
            sources.append("wm_config_id")
        elif feature_name in by_name:
            values[i] = float(by_name[feature_name])
            sources.append("wm_config_name")
        elif np.any(poison_mask):
            values[i] = float(X_watermark[poison_mask, i][0])
            sources.append("poisoned_row_fallback")
        else:
            raise ValueError(f"Could not resolve watermark value for feature {fid} ({feature_name})")
    return values, sources


def build_marginal_rarity_df(
    *,
    X_watermark: np.ndarray,
    exact_matches: np.ndarray,
    poison_mask: np.ndarray,
    feature_ids: np.ndarray,
    feature_names: list[str],
    watermark_values: np.ndarray,
    value_sources: list[str],
) -> pd.DataFrame:
    rows = []
    clean_mask = ~poison_mask
    clean_total = int(np.sum(clean_mask))
    poison_total = int(np.sum(poison_mask))
    for col, feature_id in enumerate(feature_ids):
        clean_values = X_watermark[clean_mask, col]
        poison_values = X_watermark[poison_mask, col]
        clean_exact = exact_matches[clean_mask, col]
        poison_exact = exact_matches[poison_mask, col]
        clean_count = int(np.sum(clean_exact))
        poison_count = int(np.sum(poison_exact))
        trigger_value = float(watermark_values[col])
        clean_std = float(np.std(clean_values)) if clean_values.size else np.nan
        rows.append(
            {
                "watermark_rank": col + 1,
                "feature_id": int(feature_id),
                "feature_name": feature_names[int(feature_id)] if int(feature_id) < len(feature_names) else f"feature_{int(feature_id)}",
                "watermark_value": trigger_value,
                "value_source": value_sources[col],
                "clean_exact_count": clean_count,
                "clean_exact_frequency": clean_count / clean_total if clean_total else np.nan,
                "poisoned_exact_count": poison_count,
                "poisoned_exact_frequency": poison_count / poison_total if poison_total else np.nan,
                "clean_percentile_leq_value": float(np.mean(clean_values <= trigger_value)) if clean_values.size else np.nan,
                "clean_nearest_abs_distance": float(np.min(np.abs(clean_values - trigger_value))) if clean_values.size else np.nan,
                "clean_unique_values": int(np.unique(clean_values).shape[0]) if clean_values.size else 0,
                "clean_mean": float(np.mean(clean_values)) if clean_values.size else np.nan,
                "clean_std": clean_std,
                "clean_min": float(np.min(clean_values)) if clean_values.size else np.nan,
                "clean_max": float(np.max(clean_values)) if clean_values.size else np.nan,
                "trigger_z_from_clean": (trigger_value - float(np.mean(clean_values))) / clean_std
                if clean_values.size and clean_std > 0
                else np.nan,
                "poisoned_unique_values": int(np.unique(poison_values).shape[0]) if poison_values.size else 0,
            }
        )
    out = pd.DataFrame(rows)
    out["neg_log10_clean_frequency"] = -np.log10(np.clip(out["clean_exact_frequency"].to_numpy(dtype=np.float64), EPSILON, 1.0))
    return out


def build_joint_rarity_df(
    *,
    match_count: np.ndarray,
    poison_mask: np.ndarray,
    watermark_feature_count: int,
    clean_frequencies: np.ndarray,
    thresholds: list[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean_mask = ~poison_mask
    clean_total = int(np.sum(clean_mask))
    poison_total = int(np.sum(poison_mask))
    rows = []
    summary: dict[str, Any] = {}
    for threshold in thresholds:
        required = int(np.ceil(watermark_feature_count * threshold))
        clean_count = int(np.sum(match_count[clean_mask] >= required))
        poison_count = int(np.sum(match_count[poison_mask] >= required))
        threshold_tag = f"{int(round(threshold * 100)):03d}p"
        clean_freq = clean_count / clean_total if clean_total else np.nan
        poison_freq = poison_count / poison_total if poison_total else np.nan
        rows.append(
            {
                "threshold_fraction": threshold,
                "required_matching_features": required,
                "clean_count": clean_count,
                "clean_frequency": clean_freq,
                "clean_neg_log10_frequency": -np.log10(np.clip(clean_freq, EPSILON, 1.0))
                if np.isfinite(clean_freq)
                else np.nan,
                "poisoned_count": poison_count,
                "poisoned_frequency": poison_freq,
                "poison_only_support_ratio": poison_freq / max(clean_freq, EPSILON)
                if np.isfinite(clean_freq) and np.isfinite(poison_freq)
                else np.nan,
            }
        )
        summary[f"joint_threshold_{threshold_tag}_required_features"] = required
        summary[f"joint_threshold_{threshold_tag}_clean_count"] = clean_count
        summary[f"joint_threshold_{threshold_tag}_clean_frequency"] = clean_freq
        summary[f"joint_threshold_{threshold_tag}_clean_neg_log10_frequency"] = (
            -np.log10(np.clip(clean_freq, EPSILON, 1.0)) if np.isfinite(clean_freq) else np.nan
        )
        summary[f"joint_threshold_{threshold_tag}_poisoned_count"] = poison_count
        summary[f"joint_threshold_{threshold_tag}_poisoned_frequency"] = poison_freq
        summary[f"joint_threshold_{threshold_tag}_poison_only_support_ratio"] = (
            poison_freq / max(clean_freq, EPSILON)
            if np.isfinite(clean_freq) and np.isfinite(poison_freq)
            else np.nan
        )

    all_clean_frequency = summary.get("joint_threshold_100p_clean_frequency", np.nan)
    expected_independent = float(np.prod(np.clip(clean_frequencies, EPSILON, 1.0)))
    summary["joint_expected_independent_frequency"] = expected_independent
    summary["joint_all_lift_vs_independent"] = (
        float(all_clean_frequency / expected_independent)
        if np.isfinite(all_clean_frequency) and expected_independent > 0
        else np.nan
    )
    return pd.DataFrame(rows), summary


def compute_shap_trigger_footprint(
    *,
    shap_path: Path,
    benign_positions: np.ndarray,
    poison_mask: np.ndarray,
    watermark_feature_ids: np.ndarray,
    feature_names: list[str],
    watermark_values: np.ndarray,
    n_features: int,
    batch_size: int,
) -> dict[str, Any]:
    shap = np.load(shap_path, mmap_mode="r")
    if shap.ndim != 2:
        raise ValueError(f"Expected 2D SHAP matrix at {shap_path}, got {shap.shape}")
    if shap.shape[1] == n_features + 1:
        shap_feature_width = n_features
    elif shap.shape[1] == n_features:
        shap_feature_width = shap.shape[1]
    else:
        raise ValueError(f"SHAP width {shap.shape[1]} does not match feature width {n_features}")
    if np.max(benign_positions) >= shap.shape[0]:
        raise ValueError(
            f"Requested benign SHAP position {int(np.max(benign_positions))}, but {shap_path} has {shap.shape[0]} rows"
        )

    trigger_shap = np.empty((benign_positions.shape[0], watermark_feature_ids.shape[0]), dtype=np.float32)
    trigger_abs = np.empty(benign_positions.shape[0], dtype=np.float64)
    trigger_signed = np.empty(benign_positions.shape[0], dtype=np.float64)
    total_abs = np.empty(benign_positions.shape[0], dtype=np.float64)
    for start in range(0, benign_positions.shape[0], batch_size):
        end = min(start + batch_size, benign_positions.shape[0])
        rows = benign_positions[start:end]
        batch = np.asarray(shap[rows, :shap_feature_width], dtype=np.float32)
        trigger_batch = batch[:, watermark_feature_ids]
        trigger_shap[start:end] = trigger_batch
        abs_batch = np.abs(batch).astype(np.float32, copy=False)
        trigger_abs[start:end] = np.sum(np.abs(trigger_batch), axis=1, dtype=np.float64)
        trigger_signed[start:end] = np.sum(trigger_batch, axis=1, dtype=np.float64)
        total_abs[start:end] = np.sum(abs_batch, axis=1, dtype=np.float64)
    ratio = np.divide(trigger_abs, total_abs, out=np.zeros_like(trigger_abs), where=total_abs > 0)
    clean_mask = ~poison_mask.astype(bool, copy=False)
    poison_mask = poison_mask.astype(bool, copy=False)
    feature_rows = []
    for col, feature_id in enumerate(watermark_feature_ids):
        clean_values = trigger_shap[clean_mask, col]
        poison_values = trigger_shap[poison_mask, col]
        clean_abs = np.abs(clean_values)
        poison_abs = np.abs(poison_values)
        feature_rows.append(
            {
                "watermark_rank": col + 1,
                "feature_id": int(feature_id),
                "feature_name": feature_names[int(feature_id)] if int(feature_id) < len(feature_names) else f"feature_{int(feature_id)}",
                "watermark_value": float(watermark_values[col]),
                "clean_shap_signed_mean": float(np.mean(clean_values)) if clean_values.size else np.nan,
                "poisoned_shap_signed_mean": float(np.mean(poison_values)) if poison_values.size else np.nan,
                "poison_minus_clean_shap_signed_mean": (
                    float(np.mean(poison_values)) - float(np.mean(clean_values))
                    if clean_values.size and poison_values.size
                    else np.nan
                ),
                "clean_shap_abs_mean": float(np.mean(clean_abs)) if clean_abs.size else np.nan,
                "poisoned_shap_abs_mean": float(np.mean(poison_abs)) if poison_abs.size else np.nan,
                "poison_minus_clean_shap_abs_mean": (
                    float(np.mean(poison_abs)) - float(np.mean(clean_abs))
                    if clean_abs.size and poison_abs.size
                    else np.nan
                ),
                "clean_shap_signed_median": float(np.median(clean_values)) if clean_values.size else np.nan,
                "poisoned_shap_signed_median": float(np.median(poison_values)) if poison_values.size else np.nan,
                "clean_shap_abs_median": float(np.median(clean_abs)) if clean_abs.size else np.nan,
                "poisoned_shap_abs_median": float(np.median(poison_abs)) if poison_abs.size else np.nan,
            }
        )
    return {
        "trigger_signed_sum": trigger_signed,
        "trigger_signed_mean": trigger_signed / float(max(1, watermark_feature_ids.shape[0])),
        "trigger_abs_sum": trigger_abs,
        "trigger_abs_mean": trigger_abs / float(max(1, watermark_feature_ids.shape[0])),
        "total_abs_sum": total_abs,
        "trigger_abs_ratio": ratio,
        "feature_df": pd.DataFrame(feature_rows),
    }


def build_trigger_footprint_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Small paper-facing subset of the full detectability summary."""
    preferred_keys = [
        "dataset_label",
        "sampling_strategy",
        "baseline_tag",
        "model_family",
        "feature_selector",
        "value_selector",
        "target_features",
        "artifact_dir",
        "diagnostic_output_dir",
        "benign_rows_scored",
        "clean_benign_rows_scored",
        "poisoned_benign_rows_scored",
        "watermark_feature_count",
        "marginal_clean_frequency_min",
        "marginal_clean_frequency_median",
        "marginal_clean_frequency_mean",
        "marginal_clean_frequency_max",
        "marginal_neg_log10_frequency_sum",
        "marginal_neg_log10_frequency_mean",
        "joint_threshold_075p_required_features",
        "joint_threshold_075p_clean_frequency",
        "joint_threshold_075p_clean_neg_log10_frequency",
        "joint_threshold_075p_poison_only_support_ratio",
        "joint_threshold_100p_required_features",
        "joint_threshold_100p_clean_frequency",
        "joint_threshold_100p_clean_neg_log10_frequency",
        "joint_threshold_100p_poison_only_support_ratio",
        "joint_expected_independent_frequency",
        "joint_all_lift_vs_independent",
        "trigger_match_fraction_auroc",
        "trigger_rarity_score_auroc",
        "shap_status",
        "shap_trigger_signed_sum_clean_mean",
        "shap_trigger_signed_sum_poison_mean",
        "shap_trigger_signed_sum_poison_minus_clean_mean",
        "shap_trigger_abs_sum_clean_mean",
        "shap_trigger_abs_sum_poison_mean",
        "shap_trigger_abs_sum_poison_minus_clean_mean",
        "shap_trigger_abs_sum_auroc",
        "shap_trigger_abs_ratio_clean_mean",
        "shap_trigger_abs_ratio_poison_mean",
        "shap_trigger_abs_ratio_poison_minus_clean_mean",
        "shap_trigger_abs_ratio_auroc",
    ]
    return {key: summary.get(key, np.nan) for key in preferred_keys}


def compute_knn_trigger_density_scores(
    *,
    X_watermark: np.ndarray,
    poison_mask: np.ndarray,
    knn_neighbors: int,
    max_reference_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    clean_positions = np.flatnonzero(~poison_mask).astype(np.int64, copy=False)
    if clean_positions.shape[0] < 2:
        raise ValueError("kNN density diagnostics require at least two clean benign rows")

    reference_count = min(int(max_reference_rows), clean_positions.shape[0])
    reference_positions = np.sort(rng.choice(clean_positions, size=reference_count, replace=False)).astype(np.int64, copy=False)
    X_clean = X_watermark[clean_positions]
    mean = np.mean(X_clean, axis=0, dtype=np.float64)
    std = np.std(X_clean, axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    X_scaled = (X_watermark.astype(np.float64, copy=False) - mean.reshape(1, -1)) / std.reshape(1, -1)
    X_ref = X_scaled[reference_positions]

    query_neighbors = min(knn_neighbors + 1, reference_count)
    model = NearestNeighbors(n_neighbors=query_neighbors, algorithm="auto", metric="euclidean")
    model.fit(X_ref)
    distances, _ = model.kneighbors(X_scaled, return_distance=True)

    is_reference = np.zeros(X_scaled.shape[0], dtype=bool)
    is_reference[reference_positions] = True
    scores = np.empty(X_scaled.shape[0], dtype=np.float64)
    for i in range(X_scaled.shape[0]):
        row_distances = distances[i]
        if is_reference[i] and row_distances.shape[0] > 1 and row_distances[0] <= 1e-10:
            selected = row_distances[1 : min(knn_neighbors + 1, row_distances.shape[0])]
        else:
            selected = row_distances[: min(knn_neighbors, row_distances.shape[0])]
        scores[i] = float(np.mean(selected)) if selected.size else float(row_distances[0])
    return scores, reference_positions


def add_score_diagnostics(
    *,
    score_name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    recall_percents: list[float],
    metric_rows: list[dict[str, Any]],
    summary_scores: dict[str, Any],
) -> None:
    labels = labels.astype(bool, copy=False)
    finite_mask = np.isfinite(scores)
    valid_scores = scores[finite_mask]
    valid_labels = labels[finite_mask]
    auroc = safe_auroc(valid_labels, valid_scores)
    average_precision = safe_average_precision(valid_labels, valid_scores)
    total_poisoned = int(np.sum(valid_labels))
    summary_scores[f"{score_name}_auroc"] = auroc
    summary_scores[f"{score_name}_average_precision"] = average_precision
    summary_scores[f"{score_name}_valid_rows"] = int(valid_scores.shape[0])

    order = np.argsort(valid_scores)[::-1]
    for percent in recall_percents:
        rows_flagged = max(1, int(np.ceil(valid_scores.shape[0] * percent / 100.0)))
        rows_flagged = min(rows_flagged, valid_scores.shape[0])
        top_labels = valid_labels[order[:rows_flagged]]
        poisoned_caught = int(np.sum(top_labels))
        recall = poisoned_caught / total_poisoned if total_poisoned else np.nan
        precision = poisoned_caught / rows_flagged if rows_flagged else np.nan
        tag = f"recall_at_{format_percent_tag(percent)}"
        if percent in {0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0}:
            summary_scores[f"{score_name}_{tag}"] = recall
        metric_rows.append(
            {
                "score_name": score_name,
                "auroc": auroc,
                "average_precision": average_precision,
                "top_percent": percent,
                "rows_flagged": rows_flagged,
                "poisoned_caught": poisoned_caught,
                "poison_recall": recall,
                "removal_precision": precision,
            }
        )


def group_difference_summary(score_name: str, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    poison_mask = labels.astype(bool, copy=False)
    clean_mask = ~poison_mask
    clean_mean = group_mean(scores, clean_mask)
    poison_mean = group_mean(scores, poison_mask)
    clean_median = group_median(scores, clean_mask)
    poison_median = group_median(scores, poison_mask)
    return {
        f"{score_name}_clean_mean": clean_mean,
        f"{score_name}_poison_mean": poison_mean,
        f"{score_name}_poison_minus_clean_mean": poison_mean - clean_mean,
        f"{score_name}_clean_median": clean_median,
        f"{score_name}_poison_median": poison_median,
        f"{score_name}_poison_minus_clean_median": poison_median - clean_median,
    }


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or np.unique(labels).shape[0] < 2:
        return None
    return float(roc_auc_score(labels.astype(int), scores))


def safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or np.unique(labels).shape[0] < 2:
        return None
    return float(average_precision_score(labels.astype(int), scores))


def group_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else np.nan


def group_median(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.median(values[mask])) if np.any(mask) else np.nan


def format_percent_tag(value: float) -> str:
    return f"{value:g}p".replace(".", "p")


def parse_artifact_context(artifact_path: Path) -> dict[str, Any]:
    resolved = artifact_path.resolve()
    parts = list(resolved.parts)
    context: dict[str, Any] = {
        "artifact_name": resolved.name,
        "dataset_path": None,
        "dataset_label": None,
        "sampling_strategy": None,
        "baseline_tag": None,
        "model_family": None,
        "feature_selector": None,
        "value_selector": None,
        "target_features": None,
    }

    if "results" in parts and "attack_artifacts" in parts:
        results_idx = parts.index("results")
        artifact_idx = parts.index("attack_artifacts")
        before_attack = parts[results_idx + 1 : artifact_idx]
        sampling_idx = None
        for i, part in enumerate(before_attack):
            if part.endswith("-defense"):
                sampling_idx = i
                context["sampling_strategy"] = part[: -len("-defense")]
                break
        dataset_parts = before_attack[:sampling_idx] if sampling_idx is not None else before_attack
        context["dataset_path"] = "/".join(dataset_parts) if dataset_parts else None
        context["dataset_label"] = dataset_label_from_parts(dataset_parts)

    name_parts = resolved.name.split("__")
    if len(name_parts) >= 5:
        context["baseline_tag"] = name_parts[0]
        context["model_family"] = name_parts[1]
        context["feature_selector"] = name_parts[2]
        context["value_selector"] = name_parts[3]
        context["target_features"] = "__".join(name_parts[4:])
    return context


def dataset_label_from_parts(parts: list[str]) -> str | None:
    if not parts:
        return None
    if parts[0] == "ember":
        return "EMBER2018"
    if parts[0] == "ember2024":
        if len(parts) > 1:
            return f"EMBER2024 {parts[1].upper()}"
        return "EMBER2024"
    return "/".join(parts)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def safe_float(value: Any) -> float:
    return float(value) if value is not None else np.nan


def resolve_output_dir(
    *,
    artifact_path: Path,
    output_dir: str | Path | None,
    max_benign_rows: int | None,
) -> Path:
    if output_dir is not None:
        resolved = resolve_path(output_dir)
        return resolved or Path(output_dir)
    row_tag = "full" if max_benign_rows is None else f"rows{max_benign_rows}"
    return artifact_path / "detectability_diagnostics" / f"default_{row_tag}"


def _resolve_existing_dir(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("Expected a directory path")
    resolved = resolve_path(path)
    path_obj = resolved or Path(path)
    if not path_obj.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")
    return path_obj
