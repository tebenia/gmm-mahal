"""Export compact Experiment 1 EMBER2018 sanitizer results."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd

from src.analysis.result_catalog import load_detector_metrics


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
SAMPLINGS = ["random", "distribution_based_distance"]
SANITIZERS = ["isolation_forest", "spectral_signature", "hdbscan"]
LABELS = {
    "min_population_new": "MinPopulation*",
    "argmin_Nv_sum_abs_shap": "CountAbsSHAP*",
    "combined_shap": "CombinedSHAP*",
    "low_shap_signed": "LowSHAPSigned",
    "frequency_bounded_signed_shap": "FrequencyBoundedSignedSHAP",
    "corr_count_abs_shap": "CorrCountAbsSHAP",
    "benign_prototype": "BenignPrototype",
    "quantile_95": "Quantile95",
}


def export_summary(results_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    results_root = Path(results_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector_metrics(results_root)
    selected = detector[
        detector["sampling_strategy"].isin(SAMPLINGS)
        & detector["value_selector"].isin(SELECTORS)
        & detector["feature_selector"].isin(["shap_largest_abs", "combined_shap"])
        & detector["defense_method"].isin(SANITIZERS)
        & detector["target_features"].eq("problem_space_conservative")
        & detector["detector_feature_mode"].eq("hybrid")
        & detector["detector_top_k"].eq(32)
        & detector["poison_rate_train"].eq(0.01)
    ].copy()
    if selected.empty:
        raise FileNotFoundError(f"No matching sanitizer metadata found below {results_root}")

    selected["dataset"] = "EMBER2018"
    selected["selector"] = selected["value_selector"].map(LABELS)
    columns = [
        "dataset", "sampling_strategy", "selector", "feature_selector", "value_selector",
        "defense_method", "defense_setting", "poison_rate_train", "detector_feature_mode",
        "detector_top_k", "benign_rows_scored", "total_poisoned_rows", "total_clean_rows",
        "removed_poisoned_rows", "removed_clean_rows", "removed_rows", "poison_recall",
        "poison_recall_percent", "clean_false_positive_rate", "clean_false_positive_rate_percent",
        "poison_removal_precision_percent", "detector_removal_budget_percent",
        "detector_metadata_path", "detector_dir",
    ]
    seed_level = selected[[c for c in columns if c in selected]].sort_values(
        ["sampling_strategy", "value_selector", "defense_method"]
    ).reset_index(drop=True)

    expected = pd.DataFrame(
        product(SAMPLINGS, SELECTORS, SANITIZERS),
        columns=["sampling_strategy", "value_selector", "defense_method"],
    )
    coverage = expected.merge(
        seed_level[["sampling_strategy", "value_selector", "defense_method", "detector_metadata_path"]],
        on=["sampling_strategy", "value_selector", "defense_method"],
        how="left",
    )
    coverage["selector"] = coverage["value_selector"].map(LABELS)
    coverage["status"] = coverage["detector_metadata_path"].notna().map(
        {True: "complete", False: "missing"}
    )

    recall_pivot = seed_level.pivot_table(
        index=["sampling_strategy", "selector"],
        columns="defense_method",
        values="poison_recall_percent",
        aggfunc="first",
    ).reset_index()

    paths = {
        "seed_level": output_dir / "sanitizer_metrics_ember2018_seed43.csv",
        "coverage": output_dir / "sanitizer_metrics_ember2018_coverage.csv",
        "poison_recall_pivot": output_dir / "sanitizer_poison_recall_ember2018.csv",
        "manifest": output_dir / "sanitizer_metrics_ember2018_manifest.json",
    }
    seed_level.to_csv(paths["seed_level"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    recall_pivot.to_csv(paths["poison_recall_pivot"], index=False)
    manifest = {
        "scope": {
            "dataset": "EMBER2018",
            "experiment": "experiment 1",
            "seed": 43,
            "poison_rate": 0.01,
            "selectors": SELECTORS,
            "sampling_strategies": SAMPLINGS,
            "sanitizers": SANITIZERS,
            "target_features": "problem_space_conservative",
        },
        "settings": {
            "feature_mode": "hybrid",
            "top_k": 32,
            "isolation_forest_contamination": 0.02,
            "spectral_signature_removal_percent": 2.0,
            "hdbscan_min_cluster_percent": 0.5,
            "hdbscan_min_samples_percent": 0.1,
            "hdbscan_threshold_max_percent": 10.0,
            "hdbscan_min_keep": 0.2,
            "random_state": 42,
        },
        "rows": int(len(seed_level)),
        "complete_rows": int((coverage["status"] == "complete").sum()),
        "source_root": str(results_root),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths
