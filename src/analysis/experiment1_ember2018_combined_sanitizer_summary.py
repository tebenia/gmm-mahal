"""Export compact sanitizer results for Experiment 1 EMBER2018 CombinedSHAP."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.result_catalog import load_detector_metrics


SANITIZERS = ["isolation_forest", "spectral_signature", "hdbscan"]
SAMPLINGS = ["random", "distribution_based_distance"]


def export_summary(results_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    results_root = Path(results_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector_metrics(results_root)
    if detector.empty:
        raise FileNotFoundError(f"No sanitizer metadata found below {results_root}")

    selected = detector[
        detector["sampling_strategy"].isin(SAMPLINGS)
        & detector["value_selector"].eq("combined_shap")
        & detector["feature_selector"].eq("combined_shap")
        & detector["defense_method"].isin(SANITIZERS)
        & detector["target_features"].eq("problem_space_conservative")
        & detector["detector_feature_mode"].eq("hybrid")
        & detector["detector_top_k"].eq(32)
        & detector["poison_rate_train"].eq(0.01)
    ].copy()

    if selected.empty:
        raise FileNotFoundError("No matching EMBER2018 CombinedSHAP sanitizer rows found")

    selected["sampling"] = selected["sampling_strategy"]
    selected["selector"] = "CombinedSHAP*"
    selected["dataset"] = "EMBER2018"

    columns = [
        "dataset", "sampling", "selector", "defense_method", "defense_setting",
        "poison_rate_train", "detector_feature_mode", "detector_top_k",
        "benign_rows_scored", "total_poisoned_rows", "total_clean_rows",
        "removed_poisoned_rows", "removed_clean_rows", "removed_rows",
        "poison_recall", "poison_recall_percent", "clean_false_positive_rate",
        "clean_false_positive_rate_percent", "poison_removal_precision_percent",
        "detector_removal_budget_percent", "detector_metadata_path", "detector_dir",
    ]
    seed_level = selected[[c for c in columns if c in selected]].sort_values(
        ["sampling", "defense_method"]
    ).reset_index(drop=True)

    expected = pd.MultiIndex.from_product(
        [SAMPLINGS, SANITIZERS], names=["sampling", "defense_method"]
    ).to_frame(index=False)
    coverage = expected.merge(
        seed_level[["sampling", "defense_method", "detector_metadata_path"]],
        on=["sampling", "defense_method"],
        how="left",
    )
    coverage["status"] = coverage["detector_metadata_path"].notna().map(
        {True: "complete", False: "missing"}
    )

    pivot = seed_level.pivot(
        index="sampling", columns="defense_method", values="poison_recall_percent"
    ).reset_index()

    paths = {
        "seed_level": output_dir / "sanitizer_metrics_ember2018_combined_seed43.csv",
        "coverage": output_dir / "sanitizer_metrics_ember2018_combined_coverage.csv",
        "poison_recall_pivot": output_dir / "sanitizer_poison_recall_ember2018_combined.csv",
        "manifest": output_dir / "sanitizer_metrics_ember2018_combined_manifest.json",
    }
    seed_level.to_csv(paths["seed_level"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    pivot.to_csv(paths["poison_recall_pivot"], index=False)

    manifest = {
        "scope": {
            "dataset": "EMBER2018",
            "experiment": "experiment 1",
            "seed": 43,
            "poison_rate": 0.01,
            "feature_selector": "combined_shap",
            "value_selector": "combined_shap",
            "target_features": "problem_space_conservative",
            "sampling_strategies": SAMPLINGS,
            "sanitizers": SANITIZERS,
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
