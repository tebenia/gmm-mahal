from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.result_catalog import (
    DEFENSE_LABELS,
    load_attack_summaries,
    load_detector_metrics,
)


SELECTORS = [
    "min_population_new",
    "argmin_Nv_sum_abs_shap",
    "combined_shap",
    "low_shap_signed",
    "frequency_bounded_signed_shap",
    "corr_count_abs_shap",
    "benign_prototype",
    "quantile_95",
]
SAMPLING = ["random", "distribution_based_distance"]
SANITIZERS = ["isolation_forest", "spectral_signature", "hdbscan"]
SEED = 43
LABELS = {
    "min_population_new": "MinPopulation",
    "argmin_Nv_sum_abs_shap": "CountAbsSHAP",
    "combined_shap": "CombinedSHAP",
    "low_shap_signed": "LowSHAPSigned",
    "frequency_bounded_signed_shap": "FrequencyBoundedSignedSHAP",
    "corr_count_abs_shap": "CorrCountAbsSHAP",
    "benign_prototype": "BenignPrototype",
    "quantile_95": "Quantile95",
}


def _source(selector: str) -> str:
    return "Severi-derived" if selector in {
        "min_population_new",
        "argmin_Nv_sum_abs_shap",
        "combined_shap",
    } else "This work"


def _label(selector: str) -> str:
    suffix = "*" if _source(selector) == "Severi-derived" else ""
    return LABELS[selector] + suffix


def _settings_match(frame: pd.DataFrame) -> pd.Series:
    method = frame["defense_method"]
    contamination = pd.to_numeric(frame["detector_contamination"], errors="coerce")
    removal = pd.to_numeric(frame["detector_removal_percent"], errors="coerce")
    return (
        ((method == "isolation_forest") & np.isclose(contamination, 0.02))
        | ((method == "spectral_signature") & np.isclose(removal, 2.0))
        | (method == "hdbscan")
    )


def build_tables(results_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    results_root = Path(results_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector_metrics(results_root)
    if detector.empty:
        raise FileNotFoundError(f"No detector metadata found below {results_root}")
    poison_rate = pd.to_numeric(detector["poison_rate_train"], errors="coerce")
    frame = detector[
        (detector["dataset_label"] == "EMBER2024 WIN64")
        & detector["sampling_strategy"].isin(SAMPLING)
        & np.isclose(poison_rate, 0.01)
        & (detector["seed"] == SEED)
        & detector["value_selector"].isin(SELECTORS)
        & detector["defense_method"].isin(SANITIZERS)
        & (detector["target_features"] == "problem_space_conservative")
        & (detector["detector_feature_mode"] == "hybrid")
        & (pd.to_numeric(detector["detector_top_k"], errors="coerce") == 32)
        & _settings_match(detector)
    ].copy()
    keys = ["sampling_strategy", "value_selector", "defense_method"]
    frame = frame.sort_values("source_mtime").drop_duplicates(keys, keep="last")
    frame["selector"] = frame["value_selector"].map(_label)
    frame["source"] = frame["value_selector"].map(_source)
    frame["sanitizer"] = frame["defense_method"].map(DEFENSE_LABELS)

    attacks = load_attack_summaries(results_root)
    attacks = attacks[
        (attacks["dataset_label"] == "EMBER2024 WIN64")
        & attacks["sampling_strategy"].isin(SAMPLING)
        & np.isclose(pd.to_numeric(attacks["poison_rate_train"], errors="coerce"), 0.01)
        & (attacks["seed"] == SEED)
        & attacks["value_selector"].isin(SELECTORS)
        & (attacks["target_features"] == "problem_space_conservative")
    ].copy()
    attacks = attacks.sort_values("source_mtime").drop_duplicates(
        ["sampling_strategy", "value_selector"], keep="last"
    )
    attacks = attacks[[
        "sampling_strategy", "value_selector", "attack_effectiveness_percent",
        "backdoored_clean_accuracy_percent", "clean_model_original_test_accuracy_percent",
    ]]
    frame = frame.merge(attacks, on=["sampling_strategy", "value_selector"], how="left")

    columns = [
        "dataset_label", "seed", "sampling_strategy", "poison_rate_percent",
        "feature_selector", "value_selector", "selector", "source", "target_features",
        "defense_method", "sanitizer", "defense_setting", "detector_feature_mode",
        "detector_top_k", "detector_contamination", "detector_removal_percent",
        "benign_rows_scored", "selected_features", "total_poisoned_rows",
        "total_clean_rows", "removed_poisoned_rows", "removed_clean_rows", "removed_rows",
        "poison_recall", "poison_recall_percent", "clean_false_positive_rate",
        "clean_false_positive_rate_percent", "poison_removal_precision_percent",
        "detector_removal_budget_percent", "attack_effectiveness_percent",
        "backdoored_clean_accuracy_percent", "clean_model_original_test_accuracy_percent",
        "detector_metadata_path", "source_mtime",
    ]
    seed_level = frame[[c for c in columns if c in frame]].sort_values(keys).reset_index(drop=True)

    expected = pd.DataFrame(
        product(SAMPLING, SELECTORS, SANITIZERS),
        columns=keys,
    )
    coverage = expected.merge(seed_level[keys + ["detector_metadata_path"]], on=keys, how="left")
    coverage["selector"] = coverage["value_selector"].map(_label)
    coverage["sanitizer"] = coverage["defense_method"].map(DEFENSE_LABELS)
    coverage["status"] = np.where(coverage["detector_metadata_path"].notna(), "complete", "missing")

    metric_columns = [
        "poison_recall_percent", "removed_poisoned_rows", "removed_clean_rows",
        "removed_rows", "clean_false_positive_rate_percent", "poison_removal_precision_percent",
        "attack_effectiveness_percent", "backdoored_clean_accuracy_percent",
    ]
    aggregate = seed_level.groupby(
        ["sampling_strategy", "value_selector", "selector", "source", "defense_method", "sanitizer"],
        sort=False,
    )[metric_columns].agg(["count", "mean", "std", "min", "max"]).reset_index()
    aggregate.columns = [
        "_".join(str(part) for part in c if part).rstrip("_") if isinstance(c, tuple) else c
        for c in aggregate.columns
    ]

    combined = seed_level.groupby(
        ["sampling_strategy", "value_selector", "selector", "source"], sort=False
    ).agg(
        poison_recall_mean_percent=("poison_recall_percent", "mean"),
        poison_recall_sd_percent=("poison_recall_percent", "std"),
        asr_percent=("attack_effectiveness_percent", "first"),
        clean_accuracy_percent=("backdoored_clean_accuracy_percent", "first"),
    ).reset_index()

    paths = {
        "seed_level": output_dir / "sanitizer_metrics_seed43_win64_seed_level.csv",
        "aggregate": output_dir / "sanitizer_metrics_seed43_win64_summary.csv",
        "coverage": output_dir / "sanitizer_metrics_seed43_win64_coverage.csv",
        "effectiveness": output_dir / "sanitizer_effectiveness_seed43_win64.csv",
        "manifest": output_dir / "sanitizer_metrics_seed43_win64_manifest.json",
    }
    seed_level.to_csv(paths["seed_level"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    combined.to_csv(paths["effectiveness"], index=False)
    manifest = {
        "scope": {
            "dataset": "EMBER2024 Win64", "seed": SEED,
            "sampling": SAMPLING, "poison_rate": 0.01,
            "selectors": SELECTORS, "sanitizers": SANITIZERS,
            "target_features": "problem_space_conservative",
        },
        "settings": {
            "feature_mode": "hybrid", "top_k": 32, "standardize": True,
            "defense_random_state": 42, "isolation_contamination": 0.02,
            "spectral_removal_percent": 2.0,
            "hdbscan_min_cluster_percent": 0.5,
            "hdbscan_min_samples_percent": 0.1,
            "hdbscan_threshold_max_percent": 10.0,
            "hdbscan_min_keep": 0.2,
        },
        "expected_rows": int(len(coverage)),
        "completed_rows": int((coverage["status"] == "complete").sum()),
        "missing_rows": int((coverage["status"] != "complete").sum()),
        "deletion_note": "Scalar sanitizer results are preserved; row-level scores and indices are not.",
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths

