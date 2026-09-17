from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.result_catalog import (
    DEFENSE_LABELS,
    VALUE_LABELS,
    load_detector_metrics,
)


EXPERIMENT2_SELECTORS = [
    "min_population_new",
    "argmin_Nv_sum_abs_shap",
    "combined_shap",
    "low_shap_signed",
    "frequency_bounded_signed_shap",
    "corr_count_abs_shap",
    "benign_prototype",
    "quantile_95",
]
EXPERIMENT2_SANITIZERS = [
    "isolation_forest",
    "spectral_signature",
    "hdbscan",
]
EXPERIMENT2_SEEDS = [42, 43]
EXPERIMENT2_SELECTOR_LABELS = {
    "min_population_new": "MinPopulation",
    "argmin_Nv_sum_abs_shap": "CountAbsSHAP",
    "combined_shap": "CombinedSHAP",
    "low_shap_signed": "LowSHAPSigned",
    "frequency_bounded_signed_shap": "FrequencyBoundedSignedSHAP",
    "corr_count_abs_shap": "CorrCountAbsSHAP",
    "benign_prototype": "BenignPrototype",
    "quantile_95": "Quantile95",
}


def _selector_source(selector: str) -> str:
    if selector in {"min_population_new", "argmin_Nv_sum_abs_shap", "combined_shap"}:
        return "Severi-derived"
    return "This work"


def _selector_label(selector: str) -> str:
    label = EXPERIMENT2_SELECTOR_LABELS.get(selector, VALUE_LABELS.get(selector, selector))
    return f"{label}*" if _selector_source(selector) == "Severi-derived" else label


def _matches_experiment2_setting(frame: pd.DataFrame) -> pd.Series:
    method = frame["defense_method"]
    contamination = pd.to_numeric(frame["detector_contamination"], errors="coerce")
    removal = pd.to_numeric(frame["detector_removal_percent"], errors="coerce")
    return (
        ((method == "isolation_forest") & np.isclose(contamination, 0.02))
        | ((method == "spectral_signature") & np.isclose(removal, 2.0))
        | (method == "hdbscan")
    )


def build_experiment2_sanitizer_tables(
    results_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    results_root = Path(results_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector_metrics(results_root)
    if detector.empty:
        raise FileNotFoundError(f"No sanitizer metadata found below {results_root}")

    poison_rate = pd.to_numeric(detector["poison_rate_train"], errors="coerce")
    selected = detector[
        (detector["dataset_label"] == "EMBER2018")
        & (detector["sampling_strategy"] == "random")
        & poison_rate.notna()
        & np.isclose(poison_rate, 0.01)
        & detector["seed"].isin(EXPERIMENT2_SEEDS)
        & detector["value_selector"].isin(EXPERIMENT2_SELECTORS)
        & detector["defense_method"].isin(EXPERIMENT2_SANITIZERS)
        & (detector["target_features"] == "problem_space_conservative")
        & (detector["detector_feature_mode"] == "hybrid")
        & (pd.to_numeric(detector["detector_top_k"], errors="coerce") == 32)
        & _matches_experiment2_setting(detector)
    ].copy()

    keys = ["seed", "value_selector", "defense_method"]
    selected = (
        selected.sort_values("source_mtime")
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )
    selected["selector"] = selected["value_selector"].map(_selector_label)
    selected["source"] = selected["value_selector"].map(_selector_source)
    selected["sanitizer"] = selected["defense_method"].map(DEFENSE_LABELS)

    keep_columns = [
        "dataset_label",
        "seed",
        "sampling_strategy",
        "poison_rate_train",
        "poison_rate_percent",
        "feature_selector",
        "value_selector",
        "selector",
        "source",
        "target_features",
        "defense_method",
        "sanitizer",
        "defense_setting",
        "detector_feature_mode",
        "detector_top_k",
        "detector_contamination",
        "detector_removal_percent",
        "benign_rows_scored",
        "selected_features",
        "total_poisoned_rows",
        "total_clean_rows",
        "removed_poisoned_rows",
        "removed_clean_rows",
        "removed_rows",
        "poison_recall",
        "poison_recall_percent",
        "clean_false_positive_rate",
        "clean_false_positive_rate_percent",
        "poison_removal_precision_percent",
        "detector_removal_budget_percent",
        "detector_metadata_path",
        "source_mtime",
    ]
    seed_level = selected[[column for column in keep_columns if column in selected]].copy()
    seed_level = seed_level.sort_values(
        ["seed", "defense_method", "value_selector"]
    ).reset_index(drop=True)

    expected = pd.DataFrame(
        product(EXPERIMENT2_SEEDS, EXPERIMENT2_SELECTORS, EXPERIMENT2_SANITIZERS),
        columns=["seed", "value_selector", "defense_method"],
    )
    coverage = expected.merge(
        seed_level[keys + ["detector_metadata_path"]],
        on=keys,
        how="left",
    )
    coverage["selector"] = coverage["value_selector"].map(_selector_label)
    coverage["sanitizer"] = coverage["defense_method"].map(DEFENSE_LABELS)
    coverage["status"] = np.where(
        coverage["detector_metadata_path"].notna(), "complete", "missing"
    )

    metric_columns = [
        "poison_recall_percent",
        "removed_poisoned_rows",
        "removed_clean_rows",
        "removed_rows",
        "clean_false_positive_rate_percent",
        "poison_removal_precision_percent",
    ]
    aggregate = (
        seed_level.groupby(
            ["value_selector", "selector", "source", "defense_method", "sanitizer"],
            sort=False,
        )[metric_columns]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in aggregate.columns
    ]

    recall_pivot = seed_level.pivot_table(
        index=["value_selector", "selector", "source"],
        columns=["defense_method", "seed"],
        values="poison_recall_percent",
        aggfunc="first",
    ).reset_index()
    recall_pivot.columns = [
        column
        if isinstance(column, str)
        else "_".join(str(part) for part in column if str(part) not in {"", "None"})
        for column in recall_pivot.columns
    ]

    paths = {
        "seed_level": output_dir / "sanitizer_metrics_seed_level.csv",
        "aggregate": output_dir / "sanitizer_metrics_mean_std.csv",
        "recall_pivot": output_dir / "sanitizer_poison_recall_pivot.csv",
        "coverage": output_dir / "sanitizer_coverage.csv",
        "manifest": output_dir / "sanitizer_export_manifest.json",
    }
    seed_level.to_csv(paths["seed_level"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    recall_pivot.to_csv(paths["recall_pivot"], index=False)
    coverage.to_csv(paths["coverage"], index=False)

    completed = int((coverage["status"] == "complete").sum())
    manifest = {
        "scope": {
            "dataset": "EMBER2018",
            "sampling_strategy": "random",
            "poison_rate_train": 0.01,
            "seeds": EXPERIMENT2_SEEDS,
            "selectors": EXPERIMENT2_SELECTORS,
            "sanitizers": EXPERIMENT2_SANITIZERS,
            "target_features": "problem_space_conservative",
        },
        "sanitizer_settings": {
            "common": {
                "feature_mode": "hybrid",
                "top_k": 32,
                "standardize": True,
                "random_state": 42,
            },
            "isolation_forest": {"contamination": 0.02},
            "spectral_signature": {"removal_percent": 2.0},
            "hdbscan": {
                "min_cluster_percent": 0.5,
                "min_samples_percent": 0.1,
                "threshold_max_percent": 10.0,
                "min_keep": 0.2,
            },
        },
        "expected_rows": int(len(coverage)),
        "completed_rows": completed,
        "missing_rows": int(len(coverage) - completed),
        "deletion_note": (
            "These compact files preserve scalar sanitizer results. They do not "
            "preserve row-level scores, removal indices, or arrays required to "
            "rerun a sanitizer or retrain a defended model."
        ),
        "files": {name: str(path) for name, path in paths.items() if name != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
