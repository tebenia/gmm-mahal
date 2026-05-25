"""Component-guided trigger-like feature-value mining.

This is an attack-family-aware diagnostic/removal-index generator. It uses
GMM component assignments only to decide where to search, then mines repeated
feature-value pairs inside those components.
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

from .gmm_mahalanobis import build_component_geometry, compute_gmm_row_diagnostics
from ..features import ember2024_feature_utils, ember_feature_utils
from ..utils.paths import resolve_path


SUPPORTED_FEATURE_SOURCES = {"nonhashed", "all", "oracle_watermark"}
TRIGGER_ENRICHMENT_RULES = {
    "trigger_pair_count",
    "trigger_weighted_lift_sum",
    "trigger_weighted_lift_max",
    "trigger_lift_max",
}
GEOMETRY_COMPONENT_RULES = {
    "gmm_weight",
    "density_proxy_log",
    "empirical_density_proxy_log",
    "mean_l2_from_global",
    "mean_global_mahalanobis",
    "smallest_cov_volume",
    "smallest_cov_trace",
    "avg_log_likelihood",
    "responsibility_confidence_mean",
    "responsibility_entropy_mean",
}
SUPPORTED_COMPONENT_RULES = {
    "all",
    "largest",
    "global_z_mean",
    "global_z_q90",
    "global_z_max",
    "frac_top1p_global_z",
    "frac_top5p_global_z",
    "frac_top10p_global_z",
    "local_z_q90",
    "removed_rate",
} | GEOMETRY_COMPONENT_RULES | TRIGGER_ENRICHMENT_RULES
COMPONENT_RULE_COLUMNS = {
    "largest": "rows",
    "smallest_cov_volume": "cov_logdet",
    "smallest_cov_trace": "cov_trace",
}
ASCENDING_COMPONENT_RULES = {"smallest_cov_volume", "smallest_cov_trace"}
SUPPORTED_PAIR_RANKS = {"lift", "component_count", "weighted_lift", "oracle_precision", "oracle_recall"}
SUPPORTED_PAIR_APPLY_SCOPES = {"component", "global"}
SUPPORTED_ROW_RANKS = {"trigger_score", "matched_pairs", "matched_pairs_then_score"}


@dataclass
class ComponentTriggerConfig:
    artifact_dir: str
    gmm_dir: str
    output_dir: str
    component_rule: str = "global_z_max"
    top_components: int = 3
    components: list[int] | None = None
    candidate_feature_source: str = "nonhashed"
    min_component_count: int = 20
    min_lift: float = 2.0
    max_global_frequency: float = 0.10
    top_values_per_feature: int = 3
    top_pairs_per_component: int = 50
    pair_rank: str = "weighted_lift"
    pair_apply_scope: str = "global"
    row_rank: str = "matched_pairs"
    removal_percent: float = 1.0
    min_matched_pairs: int = 1
    max_features: int | None = None
    max_rows: int | None = None


@dataclass
class ComponentTriggerResult:
    output_dir: str
    metadata_path: str
    component_profile_path: str
    pair_candidates_path: str
    row_scores_path: str
    remove_watermarked_idx_path: str
    selected_components: list[int]
    candidate_features: int
    mined_pairs: int
    selected_pairs: int
    removed_rows: int
    removed_poisoned_rows: int | None
    poison_recall: float | None


def run_component_trigger_matching(
    artifact_dir: str | Path,
    gmm_dir: str | Path,
    output_dir: str | Path | None = None,
    component_rule: str = "global_z_max",
    top_components: int = 3,
    components: list[int] | None = None,
    candidate_feature_source: str = "nonhashed",
    min_component_count: int = 20,
    min_lift: float = 2.0,
    max_global_frequency: float = 0.10,
    top_values_per_feature: int = 3,
    top_pairs_per_component: int = 50,
    pair_rank: str = "weighted_lift",
    pair_apply_scope: str = "global",
    row_rank: str = "matched_pairs",
    removal_percent: float = 1.0,
    min_matched_pairs: int = 1,
    max_features: int | None = None,
    max_rows: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> ComponentTriggerResult | dict[str, Any]:
    validate_config(
        component_rule=component_rule,
        top_components=top_components,
        components=components,
        candidate_feature_source=candidate_feature_source,
        min_component_count=min_component_count,
        min_lift=min_lift,
        max_global_frequency=max_global_frequency,
        top_values_per_feature=top_values_per_feature,
        top_pairs_per_component=top_pairs_per_component,
        pair_rank=pair_rank,
        pair_apply_scope=pair_apply_scope,
        row_rank=row_rank,
        removal_percent=removal_percent,
        min_matched_pairs=min_matched_pairs,
        max_features=max_features,
        max_rows=max_rows,
    )

    artifact_path = _resolve_existing_dir(artifact_dir)
    gmm_path = _resolve_existing_dir(gmm_dir)
    output_path = resolve_output_dir(output_dir, gmm_path, component_rule, removal_percent)

    required = {
        "watermarked_X": artifact_path / "watermarked_X.npy",
        "defense_metadata_npz": artifact_path / "defense_metadata.npz",
        "defense_metadata_json": artifact_path / "defense_metadata.json",
        "suspicious_scores": gmm_path / "suspicious_scores.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifact(s): {', '.join(missing)}")

    if dry_run:
        return {
            "artifact_dir": str(artifact_path),
            "gmm_dir": str(gmm_path),
            "output_dir": str(output_path),
            "required_paths": {key: str(path) for key, path in required.items()},
            "component_rule": component_rule,
            "top_components": top_components,
            "components": components,
            "candidate_feature_source": candidate_feature_source,
            "removal_percent": removal_percent,
        }

    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "component_trigger_metadata.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_path} already exists. Pass --overwrite to replace it.")

    start_time = time.time()
    scores = pd.read_csv(required["suspicious_scores"])
    if max_rows is not None:
        scores = scores.iloc[: min(max_rows, len(scores))].copy()
    validate_scores(scores)

    X_all = _unwrap_saved_array(np.load(required["watermarked_X"], mmap_mode="r", allow_pickle=True))
    benign_idx = scores["watermarked_idx"].to_numpy(dtype=np.int64)
    if np.any(benign_idx < 0) or np.any(benign_idx >= X_all.shape[0]):
        raise ValueError("suspicious_scores.csv contains watermarked_idx outside watermarked_X.npy rows")
    n_features = int(X_all.shape[1])

    metadata_json = json.loads(required["defense_metadata_json"].read_text(encoding="utf-8"))
    feature_names = feature_names_for_width(n_features)
    watermark = load_watermark(required["defense_metadata_json"], artifact_path / "wm_config.npy", feature_names)
    candidate_features = candidate_feature_ids(
        source=candidate_feature_source,
        n_features=n_features,
        watermark=watermark,
    )
    if max_features is not None:
        candidate_features = candidate_features[:max_features]

    component_profile = build_component_profile(scores, gmm_path)
    preselection_pairs = None
    if component_rule in TRIGGER_ENRICHMENT_RULES:
        preselection_pairs = mine_component_pairs(
            X_all=X_all,
            benign_idx=benign_idx,
            scores=scores,
            feature_names=feature_names,
            feature_ids=candidate_features,
            components=[int(v) for v in sorted(component_profile["component"].unique())],
            min_component_count=min_component_count,
            min_lift=min_lift,
            max_global_frequency=max_global_frequency,
            top_values_per_feature=top_values_per_feature,
        )
        component_profile = add_trigger_enrichment_profile(component_profile, preselection_pairs)
    selected_components = select_components(component_profile, component_rule, top_components, components)
    component_profile["selected_for_mining"] = component_profile["component"].isin(selected_components)

    if preselection_pairs is not None:
        pairs = preselection_pairs[preselection_pairs["component"].isin(selected_components)].copy()
    else:
        pairs = mine_component_pairs(
            X_all=X_all,
            benign_idx=benign_idx,
            scores=scores,
            feature_names=feature_names,
            feature_ids=candidate_features,
            components=selected_components,
            min_component_count=min_component_count,
            min_lift=min_lift,
            max_global_frequency=max_global_frequency,
            top_values_per_feature=top_values_per_feature,
        )
    selected_pairs = select_top_pairs(
        pairs,
        top_pairs_per_component=top_pairs_per_component,
        pair_rank=pair_rank,
    )
    row_scores = score_rows(
        X_all=X_all,
        benign_idx=benign_idx,
        scores=scores,
        selected_pairs=selected_pairs,
        watermark=watermark,
        pair_apply_scope=pair_apply_scope,
        min_matched_pairs=min_matched_pairs,
    )
    remove_positions = select_rows_for_removal(
        row_scores=row_scores,
        removal_percent=removal_percent,
        min_matched_pairs=min_matched_pairs,
        row_rank=row_rank,
    )
    remove_watermarked_idx = row_scores.loc[remove_positions, "watermarked_idx"].to_numpy(dtype=np.int64)
    row_scores["removed"] = False
    row_scores.loc[remove_positions, "removed"] = True

    component_profile_path = output_path / "component_profile.csv"
    pair_candidates_path = output_path / "component_trigger_pairs.csv"
    preselection_pairs_path = output_path / "component_trigger_preselection_pairs.csv"
    selected_pairs_path = output_path / "selected_component_trigger_pairs.csv"
    row_scores_path = output_path / "component_trigger_row_scores.csv"
    remove_watermarked_idx_path = output_path / "remove_watermarked_idx.npy"

    component_profile.to_csv(component_profile_path, index=False)
    if preselection_pairs is not None:
        preselection_pairs.to_csv(preselection_pairs_path, index=False)
    pairs.to_csv(pair_candidates_path, index=False)
    selected_pairs.to_csv(selected_pairs_path, index=False)
    row_scores.to_csv(row_scores_path, index=False)
    np.save(remove_watermarked_idx_path, remove_watermarked_idx)

    removed_stats = removal_stats(row_scores)
    known_wm_summary = known_watermark_summary(row_scores)
    metadata = {
        "config": asdict(
            ComponentTriggerConfig(
                artifact_dir=str(artifact_path),
                gmm_dir=str(gmm_path),
                output_dir=str(output_path),
                component_rule=component_rule,
                top_components=top_components,
                components=components,
                candidate_feature_source=candidate_feature_source,
                min_component_count=min_component_count,
                min_lift=min_lift,
                max_global_frequency=max_global_frequency,
                top_values_per_feature=top_values_per_feature,
                top_pairs_per_component=top_pairs_per_component,
                pair_rank=pair_rank,
                pair_apply_scope=pair_apply_scope,
                row_rank=row_rank,
                removal_percent=removal_percent,
                min_matched_pairs=min_matched_pairs,
                max_features=max_features,
                max_rows=max_rows,
            )
        ),
        "dataset": metadata_json.get("dataset"),
        "input_shape": {
            "watermarked_X": list(X_all.shape),
            "benign_rows": int(benign_idx.shape[0]),
            "features": n_features,
        },
        "selected_components": [int(v) for v in selected_components],
        "candidate_features": int(len(candidate_features)),
        "mined_pairs": int(len(pairs)),
        "selected_pairs": int(len(selected_pairs)),
        "removed_stats": removed_stats,
        "known_watermark_summary": known_wm_summary,
        "output_files": {
            "component_profile": str(component_profile_path),
            "pair_candidates": str(pair_candidates_path),
            "preselection_pairs": str(preselection_pairs_path) if preselection_pairs is not None else None,
            "selected_pairs": str(selected_pairs_path),
            "row_scores": str(row_scores_path),
            "remove_watermarked_idx": str(remove_watermarked_idx_path),
        },
        "runtime_seconds": time.time() - start_time,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return ComponentTriggerResult(
        output_dir=str(output_path),
        metadata_path=str(metadata_path),
        component_profile_path=str(component_profile_path),
        pair_candidates_path=str(pair_candidates_path),
        row_scores_path=str(row_scores_path),
        remove_watermarked_idx_path=str(remove_watermarked_idx_path),
        selected_components=[int(v) for v in selected_components],
        candidate_features=int(len(candidate_features)),
        mined_pairs=int(len(pairs)),
        selected_pairs=int(len(selected_pairs)),
        removed_rows=int(removed_stats["removed_rows"]),
        removed_poisoned_rows=removed_stats.get("removed_poisoned_rows"),
        poison_recall=removed_stats.get("poison_recall"),
    )


def validate_config(
    component_rule: str,
    top_components: int,
    components: list[int] | None,
    candidate_feature_source: str,
    min_component_count: int,
    min_lift: float,
    max_global_frequency: float,
    top_values_per_feature: int,
    top_pairs_per_component: int,
    pair_rank: str,
    pair_apply_scope: str,
    row_rank: str,
    removal_percent: float,
    min_matched_pairs: int,
    max_features: int | None,
    max_rows: int | None,
) -> None:
    if component_rule not in SUPPORTED_COMPONENT_RULES:
        raise ValueError(f"Unsupported component_rule {component_rule}. Valid: {sorted(SUPPORTED_COMPONENT_RULES)}")
    if candidate_feature_source not in SUPPORTED_FEATURE_SOURCES:
        raise ValueError(
            f"Unsupported candidate_feature_source {candidate_feature_source}. "
            f"Valid: {sorted(SUPPORTED_FEATURE_SOURCES)}"
        )
    if pair_rank not in SUPPORTED_PAIR_RANKS:
        raise ValueError(f"Unsupported pair_rank {pair_rank}. Valid: {sorted(SUPPORTED_PAIR_RANKS)}")
    if pair_apply_scope not in SUPPORTED_PAIR_APPLY_SCOPES:
        raise ValueError(
            f"Unsupported pair_apply_scope {pair_apply_scope}. Valid: {sorted(SUPPORTED_PAIR_APPLY_SCOPES)}"
        )
    if row_rank not in SUPPORTED_ROW_RANKS:
        raise ValueError(f"Unsupported row_rank {row_rank}. Valid: {sorted(SUPPORTED_ROW_RANKS)}")
    if top_components <= 0:
        raise ValueError("top_components must be positive")
    if components is not None and not components:
        raise ValueError("components must not be empty when provided")
    if min_component_count <= 0:
        raise ValueError("min_component_count must be positive")
    if min_lift < 0:
        raise ValueError("min_lift must be non-negative")
    if not 0 < max_global_frequency <= 1:
        raise ValueError("max_global_frequency must be in (0, 1]")
    if top_values_per_feature <= 0:
        raise ValueError("top_values_per_feature must be positive")
    if top_pairs_per_component <= 0:
        raise ValueError("top_pairs_per_component must be positive")
    if removal_percent <= 0 or removal_percent > 100:
        raise ValueError("removal_percent must be in (0, 100]")
    if min_matched_pairs <= 0:
        raise ValueError("min_matched_pairs must be positive")
    if max_features is not None and max_features <= 0:
        raise ValueError("max_features must be positive")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")


def validate_scores(scores: pd.DataFrame) -> None:
    required_cols = {"watermarked_idx", "component", "is_poisoned"}
    missing = sorted(required_cols - set(scores.columns))
    if missing:
        raise KeyError(f"suspicious_scores.csv is missing columns: {missing}")


def resolve_output_dir(
    output_dir: str | Path | None,
    gmm_dir: Path,
    component_rule: str,
    removal_percent: float,
) -> Path:
    if output_dir is not None:
        resolved = resolve_path(output_dir)
        return resolved or Path(output_dir)
    remove_tag = f"{removal_percent:g}".replace(".", "p")
    return gmm_dir / "component_trigger_matching" / f"rule_{component_rule}_remove{remove_tag}p"


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


def candidate_feature_ids(source: str, n_features: int, watermark: dict[str, Any]) -> list[int]:
    if source == "all":
        return list(range(n_features))
    if source == "oracle_watermark":
        ids = [int(v) for v in watermark.get("feature_ids", [])]
        if not ids:
            raise ValueError("oracle_watermark source requested, but no watermark feature ids were found")
        return ids
    if n_features == ember_feature_utils.NUM_EMBER_FEATURES:
        return ember_feature_utils.get_non_hashed_features()
    if ember_feature_utils.NUM_EMBER_FEATURES < n_features < ember2024_feature_utils.NUM_EMBER2024_FEATURES:
        return ember_feature_utils.get_non_hashed_features()
    if n_features == ember2024_feature_utils.NUM_EMBER2024_FEATURES:
        return ember2024_feature_utils.get_non_hashed_features()
    return list(range(n_features))


def load_watermark(metadata_json_path: Path, wm_config_path: Path, feature_names: list[str]) -> dict[str, Any]:
    feature_ids: list[int] = []
    feature_values: list[float] = []
    feature_names_out: list[str] = []
    metadata = json.loads(metadata_json_path.read_text(encoding="utf-8"))

    if wm_config_path.exists():
        wm = np.load(wm_config_path, allow_pickle=True)
        if isinstance(wm, np.ndarray) and wm.shape == () and wm.dtype == object:
            wm = wm.item()
        if isinstance(wm, dict):
            feature_ids = [int(v) for v in wm.get("wm_feat_ids", [])]
            wm_features = wm.get("watermark_features", {})
            metadata_values = metadata.get("watermark_values", [])
            metadata_names = metadata.get("watermark_features", [])
            if feature_ids and len(metadata_values) == len(feature_ids):
                feature_values = [float(v) for v in metadata_values]
                if len(metadata_names) == len(feature_ids):
                    feature_names_out = [str(v) for v in metadata_names]
                else:
                    feature_names_out = [
                        feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in feature_ids
                    ]
            elif feature_ids and isinstance(wm_features, dict):
                feature_values = [float(wm_features.get(feature_names[i], np.nan)) for i in feature_ids]
                feature_names_out = [feature_names[i] for i in feature_ids]

    if not feature_ids:
        names = metadata.get("watermark_features", [])
        values = metadata.get("watermark_values", [])
        name_to_id = {name: idx for idx, name in enumerate(feature_names)}
        for name, value in zip(names, values, strict=False):
            if name in name_to_id:
                feature_names_out.append(str(name))
                feature_ids.append(int(name_to_id[name]))
                feature_values.append(float(value))

    return {
        "feature_ids": feature_ids,
        "feature_names": feature_names_out,
        "feature_values": feature_values,
        "feature_value_map": dict(zip(feature_ids, feature_values, strict=False)),
    }


def build_component_profile(scores: pd.DataFrame, gmm_dir: Path | None = None) -> pd.DataFrame:
    total_poison = max(int(scores["is_poisoned"].sum()), 1)
    global_poison_rate = float(scores["is_poisoned"].mean())
    profile = scores.groupby("component", sort=True).agg(
        rows=("component", "size"),
        oracle_poisoned=("is_poisoned", "sum"),
    )
    profile["oracle_poison_rate"] = profile["oracle_poisoned"] / profile["rows"]
    profile["oracle_poison_share"] = profile["oracle_poisoned"] / total_poison
    profile["oracle_lift_vs_uniform"] = profile["oracle_poison_rate"] / (global_poison_rate + 1e-12)
    for score_col in ["global_z", "local_z", "local_global_z", "global_mahalanobis", "local_mahalanobis"]:
        if score_col in scores.columns:
            grouped = scores.groupby("component", sort=True)[score_col]
            profile[f"{score_col}_mean"] = grouped.mean()
            profile[f"{score_col}_q90"] = grouped.quantile(0.90)
            profile[f"{score_col}_max"] = grouped.max()
            for pct in (1, 5, 10):
                n = max(1, int(np.ceil(len(scores) * pct / 100.0)))
                threshold = scores[score_col].nlargest(n).min()
                top_mask = scores[score_col] >= threshold
                profile[f"frac_top{pct}p_{score_col}"] = top_mask.groupby(scores["component"]).mean()
    if "removed" in scores.columns:
        profile["removed_rate"] = scores["removed"].groupby(scores["component"]).mean()
    else:
        profile["removed_rate"] = 0.0
    profile = profile.reset_index()
    geometry = load_component_geometry(gmm_dir, scores) if gmm_dir is not None else None
    if geometry is None:
        return profile
    geometry_for_merge = geometry.drop(columns=["rows"], errors="ignore")
    return profile.merge(geometry_for_merge, on="component", how="left")


def load_component_geometry(gmm_dir: Path, scores: pd.DataFrame) -> pd.DataFrame | None:
    geometry_path = gmm_dir / "component_geometry.csv"
    if geometry_path.exists():
        return pd.read_csv(geometry_path)

    best_gmm_path = gmm_dir / "best_local_gmm.joblib"
    metadata_path = gmm_dir / "gmm_defense_metadata.json"
    if not best_gmm_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    preprocess_dir = metadata.get("config", {}).get("preprocess_dir")
    if not preprocess_dir:
        return None
    preprocess_path = resolve_path(preprocess_dir) or Path(preprocess_dir)
    X_path = preprocess_path / "X_shap_reduced.npy"
    if not X_path.exists():
        return None

    best_gmm = joblib.load(best_gmm_path)
    global_gmm_path = gmm_dir / "global_gmm.joblib"
    global_gmm = joblib.load(global_gmm_path) if global_gmm_path.exists() else None
    X = np.asarray(np.load(X_path, mmap_mode="r")[: len(scores)], dtype=np.float64)
    labels = scores["component"].to_numpy(dtype=np.int64)
    diagnostics = compute_gmm_row_diagnostics(X, best_gmm, labels)
    scores_with_diagnostics = scores.copy()
    scores_with_diagnostics["gmm_log_likelihood"] = diagnostics["log_likelihood"]
    scores_with_diagnostics["responsibility_assigned"] = diagnostics["responsibility_assigned"]
    scores_with_diagnostics["responsibility_confidence"] = diagnostics["responsibility_confidence"]
    scores_with_diagnostics["responsibility_entropy"] = diagnostics["responsibility_entropy"]
    return build_component_geometry(scores_with_diagnostics, best_gmm, global_gmm)


def add_trigger_enrichment_profile(component_profile: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    profile = component_profile.copy()
    trigger_cols = {
        "trigger_pair_count": 0.0,
        "trigger_weighted_lift_sum": 0.0,
        "trigger_weighted_lift_max": 0.0,
        "trigger_lift_max": 0.0,
    }
    for col, default in trigger_cols.items():
        profile[col] = default
    if pairs.empty:
        return profile

    grouped = pairs.groupby("component", sort=True).agg(
        trigger_pair_count=("feature_idx", "size"),
        trigger_weighted_lift_sum=("weighted_lift", "sum"),
        trigger_weighted_lift_max=("weighted_lift", "max"),
        trigger_lift_max=("lift", "max"),
    )
    profile = profile.set_index("component")
    for col in trigger_cols:
        profile[col] = grouped[col].reindex(profile.index).fillna(0.0)
    return profile.reset_index()


def select_components(
    component_profile: pd.DataFrame,
    component_rule: str,
    top_components: int,
    components: list[int] | None,
) -> list[int]:
    if components is not None:
        known = set(component_profile["component"].astype(int))
        unknown = sorted(set(int(v) for v in components) - known)
        if unknown:
            raise ValueError(f"Requested components not present in suspicious_scores.csv: {unknown}")
        return [int(v) for v in components]
    if component_rule == "all":
        return [int(v) for v in sorted(component_profile["component"].unique())]
    order_col = COMPONENT_RULE_COLUMNS.get(component_rule, component_rule)
    if order_col not in component_profile.columns:
        raise ValueError(f"Component rule {component_rule} is unavailable in component profile")
    ranked = component_profile.sort_values(
        order_col,
        ascending=component_rule in ASCENDING_COMPONENT_RULES,
    ).head(top_components)
    return [int(v) for v in ranked["component"]]


def mine_component_pairs(
    X_all: np.ndarray,
    benign_idx: np.ndarray,
    scores: pd.DataFrame,
    feature_names: list[str],
    feature_ids: list[int],
    components: list[int],
    min_component_count: int,
    min_lift: float,
    max_global_frequency: float,
    top_values_per_feature: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    component_values = scores["component"].to_numpy()
    poison_mask = scores["is_poisoned"].to_numpy(dtype=bool)
    n_rows = int(benign_idx.shape[0])
    total_poison = max(int(poison_mask.sum()), 1)

    for component in components:
        comp_mask = component_values == component
        comp_positions = np.flatnonzero(comp_mask)
        comp_size = int(comp_positions.shape[0])
        if comp_size == 0:
            continue
        comp_poison = int(poison_mask[comp_positions].sum())
        for feature_id in feature_ids:
            feature_values = np.asarray(X_all[benign_idx, feature_id])
            global_values, global_counts = np.unique(feature_values, return_counts=True)
            comp_values_unique, comp_counts = np.unique(feature_values[comp_positions], return_counts=True)
            if comp_values_unique.size == 0:
                continue
            global_lookup = np.searchsorted(global_values, comp_values_unique)
            valid = (global_lookup >= 0) & (global_lookup < global_values.size)
            if np.any(valid):
                matched = np.zeros_like(valid, dtype=bool)
                matched[valid] = global_values[global_lookup[valid]] == comp_values_unique[valid]
                valid &= matched
            if not np.any(valid):
                continue
            comp_values_unique = comp_values_unique[valid]
            comp_counts = comp_counts[valid]
            global_counts_for_comp = global_counts[global_lookup[valid]]
            component_freq = comp_counts / comp_size
            global_freq = global_counts_for_comp / n_rows
            lift = component_freq / (global_freq + 1e-12)
            keep = (comp_counts >= min_component_count) & (lift >= min_lift) & (global_freq <= max_global_frequency)
            if not np.any(keep):
                continue
            candidate_idx = np.flatnonzero(keep)
            rank = comp_counts[candidate_idx] * np.log2(lift[candidate_idx] + 1.0)
            order = np.argsort(rank)[::-1][:top_values_per_feature]
            candidate_idx = candidate_idx[order]
            selected_rank = rank[order]
            for rank_value, idx in zip(selected_rank, candidate_idx, strict=False):
                value = comp_values_unique[idx]
                match_mask = comp_mask & (feature_values == value)
                matched_rows = int(match_mask.sum())
                matched_poison = int(np.sum(match_mask & poison_mask))
                rows.append(
                    {
                        "component": int(component),
                        "feature_idx": int(feature_id),
                        "feature_name": feature_names[feature_id] if feature_id < len(feature_names) else f"feature_{feature_id}",
                        "value": float(value),
                        "component_rows": comp_size,
                        "component_poisoned": comp_poison,
                        "component_count": int(comp_counts[idx]),
                        "global_count": int(global_counts_for_comp[idx]),
                        "component_frequency": float(component_freq[idx]),
                        "global_frequency": float(global_freq[idx]),
                        "lift": float(lift[idx]),
                        "weighted_lift": float(rank_value),
                        "oracle_matched_poisoned": matched_poison,
                        "oracle_match_precision": float(matched_poison / matched_rows) if matched_rows else 0.0,
                        "oracle_match_recall": float(matched_poison / total_poison),
                        "oracle_component_poison_coverage": float(matched_poison / comp_poison) if comp_poison else 0.0,
                    }
                )
    if not rows:
        return empty_pairs_frame()
    pairs = pd.DataFrame(rows)
    return pairs.sort_values(["component", "weighted_lift"], ascending=[True, False]).reset_index(drop=True)


def empty_pairs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "component",
            "feature_idx",
            "feature_name",
            "value",
            "component_rows",
            "component_poisoned",
            "component_count",
            "global_count",
            "component_frequency",
            "global_frequency",
            "lift",
            "weighted_lift",
            "oracle_matched_poisoned",
            "oracle_match_precision",
            "oracle_match_recall",
            "oracle_component_poison_coverage",
        ]
    )


def select_top_pairs(pairs: pd.DataFrame, top_pairs_per_component: int, pair_rank: str) -> pd.DataFrame:
    if pairs.empty:
        return pairs.copy()
    rank_col = {
        "lift": "lift",
        "component_count": "component_count",
        "weighted_lift": "weighted_lift",
        "oracle_precision": "oracle_match_precision",
        "oracle_recall": "oracle_match_recall",
    }[pair_rank]
    return (
        pairs.sort_values(["component", rank_col, "weighted_lift"], ascending=[True, False, False])
        .groupby("component", group_keys=False)
        .head(top_pairs_per_component)
        .reset_index(drop=True)
    )


def score_rows(
    X_all: np.ndarray,
    benign_idx: np.ndarray,
    scores: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    watermark: dict[str, Any],
    pair_apply_scope: str,
    min_matched_pairs: int,
) -> pd.DataFrame:
    out = scores[
        [
            "benign_position",
            "watermarked_idx",
            "original_idx",
            "source_idx",
            "is_poisoned",
            "component",
        ]
    ].copy()
    out["trigger_score"] = 0.0
    out["matched_pairs"] = 0
    out["matched_feature_names"] = ""

    component_values = scores["component"].to_numpy()
    matched_names: list[list[str]] = [[] for _ in range(len(out))]
    feature_cache: dict[int, np.ndarray] = {}

    def feature_vector(feature_idx: int) -> np.ndarray:
        if feature_idx not in feature_cache:
            feature_cache[feature_idx] = np.asarray(X_all[benign_idx, feature_idx])
        return feature_cache[feature_idx]

    for pair in selected_pairs.itertuples(index=False):
        component = int(pair.component)
        feature_idx = int(pair.feature_idx)
        value = float(pair.value)
        weight = float(getattr(pair, "weighted_lift"))
        match = feature_vector(feature_idx) == value
        if pair_apply_scope == "component":
            match = (component_values == component) & match
        if not np.any(match):
            continue
        out.loc[match, "trigger_score"] += weight
        out.loc[match, "matched_pairs"] += 1
        feature_name = str(pair.feature_name)
        for pos in np.flatnonzero(match):
            matched_names[pos].append(feature_name)

    if watermark.get("feature_ids"):
        wm_ids = np.asarray(watermark["feature_ids"], dtype=np.int64)
        wm_values = np.asarray(watermark["feature_values"], dtype=float)
        wm_counts = np.zeros(len(out), dtype=np.int64)
        for feature_idx, value in zip(wm_ids, wm_values, strict=False):
            wm_counts += feature_vector(int(feature_idx)) == float(value)
        out["known_watermark_match_count"] = wm_counts
        out["known_watermark_match_fraction"] = out["known_watermark_match_count"] / max(len(wm_ids), 1)
    else:
        out["known_watermark_match_count"] = np.nan
        out["known_watermark_match_fraction"] = np.nan

    out["eligible_for_removal"] = out["matched_pairs"] >= min_matched_pairs
    out["matched_feature_names"] = [";".join(names) for names in matched_names]
    return out


def select_rows_for_removal(
    row_scores: pd.DataFrame,
    removal_percent: float,
    min_matched_pairs: int,
    row_rank: str,
) -> np.ndarray:
    n_remove = max(1, int(np.ceil(len(row_scores) * removal_percent / 100.0)))
    eligible = row_scores[row_scores["matched_pairs"] >= min_matched_pairs].copy()
    eligible = eligible[eligible["trigger_score"] > 0]
    if eligible.empty:
        return np.array([], dtype=np.int64)
    if row_rank == "trigger_score":
        sort_cols = ["trigger_score", "matched_pairs"]
    elif row_rank == "matched_pairs":
        sort_cols = ["matched_pairs", "trigger_score"]
    elif row_rank == "matched_pairs_then_score":
        sort_cols = ["matched_pairs", "trigger_score"]
    else:
        raise ValueError(f"Unsupported row_rank: {row_rank}")
    selected = eligible.sort_values(sort_cols, ascending=False).head(n_remove)
    return selected.index.to_numpy(dtype=np.int64)


def removal_stats(row_scores: pd.DataFrame) -> dict[str, Any]:
    removed = row_scores["removed"].to_numpy(dtype=bool) if "removed" in row_scores.columns else np.zeros(len(row_scores), dtype=bool)
    poisoned = row_scores["is_poisoned"].to_numpy(dtype=bool)
    total_poisoned = int(poisoned.sum())
    total_clean = int(len(poisoned) - total_poisoned)
    removed_poisoned = int(np.sum(removed & poisoned))
    removed_clean = int(np.sum(removed & ~poisoned))
    return {
        "removed_rows": int(removed.sum()),
        "total_poisoned_rows": total_poisoned,
        "total_clean_rows": total_clean,
        "removed_poisoned_rows": removed_poisoned,
        "removed_clean_rows": removed_clean,
        "poison_recall": float(removed_poisoned / total_poisoned) if total_poisoned else None,
        "clean_false_positive_rate": float(removed_clean / total_clean) if total_clean else None,
        "removal_precision": float(removed_poisoned / max(int(removed.sum()), 1)),
    }


def known_watermark_summary(row_scores: pd.DataFrame) -> dict[str, Any]:
    if "known_watermark_match_count" not in row_scores.columns:
        return {}
    poisoned = row_scores["is_poisoned"].to_numpy(dtype=bool)
    counts = row_scores["known_watermark_match_count"]
    if counts.isna().all():
        return {}
    max_count = int(counts.max())
    rows = {}
    for threshold in sorted({1, max(1, max_count // 2), max_count}):
        mask = counts >= threshold
        rows[f"match_at_least_{threshold}"] = {
            "rows": int(mask.sum()),
            "poisoned": int(np.sum(mask & poisoned)),
            "precision": float(np.sum(mask & poisoned) / max(int(mask.sum()), 1)),
            "poison_recall": float(np.sum(mask & poisoned) / max(int(poisoned.sum()), 1)),
        }
    return rows


def _resolve_existing_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    path_obj = resolved or Path(path)
    if not path_obj.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")
    return path_obj


def _unwrap_saved_array(value):
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value
