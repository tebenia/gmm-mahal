from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(path: Path, cells: list):
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    nbf.write(notebook, path)
    print("Wrote", path)


COMMON_SETUP = r"""
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = PROJECT_ROOT / "results"
MPLCONFIGDIR = PROJECT_ROOT / "build" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.result_catalog import (
    DEFENSE_LABELS,
    DEFENSE_METHOD_ORDER,
    PMSB_SAMPLING_STRATEGIES,
    SAMPLING_ORDER,
    VALUE_LABELS,
    VALUE_ORDER,
    add_display_columns,
    comparison_coverage,
    discover_attack_artifacts,
    load_attack_summaries,
    load_detector_metrics,
    load_defense_metrics,
    load_detectability_score_metrics,
    load_detectability_summaries,
    merge_effectiveness,
    preferred_detector_metrics,
    preferred_defense_metrics,
)

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 220)
sns.set_theme(style="whitegrid", context="notebook")

PAPER_VALUE_SELECTORS = [
    "argmin_Nv_sum_abs_shap",
    "min_population_new",
    "low_shap_signed",
    "frequency_bounded",
    "frequency_bounded_signed_shap",
    "quantile_05",
    "quantile_50",
    "quantile_95",
    "benign_prototype",
    "corr_count_abs_shap",
]
PAPER_SAMPLINGS = ["random", "distribution_based_distance"]
PAPER_TARGET_FEATURES = ["problem_space_conservative"]
FOCUS_DATASETS = ["EMBER2018", "EMBER2024 WIN64"]
FOCUS_POISON_RATES = [0.01, 0.03, 0.05]

print("Project root:", PROJECT_ROOT)
print("Results root:", RESULTS_ROOT)
"""


FILTER_HELPERS = r"""
def filter_paper_rows(df):
    if df.empty:
        return df.copy()
    out = df[
        df["dataset_label"].isin(FOCUS_DATASETS)
        & df["sampling_strategy"].isin(PAPER_SAMPLINGS)
        & df["value_selector"].isin(PAPER_VALUE_SELECTORS)
        & df["target_features"].isin(PAPER_TARGET_FEATURES)
    ].copy()
    if FOCUS_POISON_RATES is not None:
        wanted = np.asarray(FOCUS_POISON_RATES, dtype=float)
        out = out[
            out["poison_rate_train"].map(
                lambda value: np.isclose(value, wanted, atol=1e-12).any()
                if pd.notna(value)
                else False
            )
        ]
    return add_display_columns(out)


def ordered_pivot(df, *, index, columns, values, aggfunc="mean"):
    pivot = df.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        observed=True,
    )
    if columns in {"value_selector", "selector_label"}:
        labels = [VALUE_LABELS[value] for value in PAPER_VALUE_SELECTORS]
        pivot = pivot.reindex(columns=[label for label in labels if label in pivot.columns])
    return pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)


def show_heatmap(pivot, title, *, cmap="viridis", fmt=".1f", center=None, vmin=None, vmax=None):
    if pivot.empty:
        print("No rows for", title)
        return
    width = max(9, 0.82 * pivot.shape[1] + 2.5)
    height = max(3.2, 0.55 * pivot.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()
"""


def notebook_06():
    cells = [
        md(
            """
            # Effectiveness And Defense-Based Detectability

            This notebook compares the paper's value selectors across:

            - attack effectiveness,
            - poison rate,
            - random and distribution-based sampling,
            - Isolation Forest, Spectral Signature, and HDBSCAN.

            Poison rate is calculated from saved row counts. Folder text such as
            `Poison Rate 0.3` is retained for auditing but is not treated as the
            scientific rate. For example, 3,600 / 120,000 is reported as 3%.

            Defense-based detectability and attack effectiveness are different
            quantities. Poison recall measures whether a defense ranks poisoned
            training rows as suspicious; ASR reduction measures the effect after
            removal and retraining.
            """
        ),
        code(COMMON_SETUP),
        md(
            """
            ## Analysis Controls

            Edit these lists to narrow the notebook. Leaving
            The current paper matrix is fixed to 1%, 3%, and 5% poisoning.
            """
        ),
        code(FILTER_HELPERS),
        md("## Load Current Results"),
        code(
            """
            attack_df_all = load_attack_summaries(RESULTS_ROOT)
            detector_df_all = load_detector_metrics(RESULTS_ROOT)
            detector_df_preferred = preferred_detector_metrics(detector_df_all)
            retrain_df_all = load_defense_metrics(RESULTS_ROOT)
            retrain_df_preferred = preferred_defense_metrics(retrain_df_all)

            attack_df = filter_paper_rows(attack_df_all)
            detector_df = filter_paper_rows(detector_df_preferred)
            detector_df = merge_effectiveness(detector_df, attack_df)
            retrain_df = filter_paper_rows(retrain_df_preferred)

            print("Attack summary rows:", len(attack_df_all), "paper rows:", len(attack_df))
            print("Detector rows:", len(detector_df_all), "matched/current rows:", len(detector_df))
            print("Defense retrain rows:", len(retrain_df_all), "matched/current rows:", len(retrain_df))
            print("Datasets:", sorted(attack_df["dataset_label"].dropna().unique()))
            print("Computed poison rates:", sorted(attack_df["poison_rate_label"].dropna().unique()))
            print("Sampling strategies:", sorted(attack_df["sampling_strategy"].dropna().unique()))
            """
        ),
        md(
            """
            ## Coverage And Missing Combinations

            This is the first table to check after adding results. A row count of
            10 means every paper value selector is present for that dataset,
            poison rate, and sampling strategy.
            """
        ),
        code(
            """
            attack_coverage = comparison_coverage(
                attack_df,
                datasets=FOCUS_DATASETS,
                samplings=PAPER_SAMPLINGS,
                target_features=PAPER_TARGET_FEATURES,
            )
            display(attack_coverage)

            dataset_rates = attack_df[
                ["dataset_label", "poison_rate_label"]
            ].drop_duplicates()
            expected = (
                dataset_rates.assign(_key=1)
                .merge(
                    pd.DataFrame({"sampling_strategy": PAPER_SAMPLINGS, "_key": 1}),
                    on="_key",
                )
                .merge(
                    pd.DataFrame({"value_selector": PAPER_VALUE_SELECTORS, "_key": 1}),
                    on="_key",
                )
                .drop(columns="_key")
            )
            observed = attack_df[
                ["dataset_label", "poison_rate_label", "sampling_strategy", "value_selector"]
            ].drop_duplicates()
            missing_attack = expected.merge(observed, how="left", indicator=True)
            missing_attack = missing_attack[missing_attack["_merge"] == "left_only"].drop(columns="_merge")
            print("Cross-product gaps (some dataset/rate pairs were never intended for every sampling):", len(missing_attack))
            display(missing_attack.head(80))

            detector_coverage = (
                detector_df.groupby(
                    ["dataset_label", "poison_rate_label", "sampling_strategy", "defense_method"],
                    dropna=False,
                )
                .agg(
                    selectors=("value_selector", "nunique"),
                    mean_detector_budget_percent=("detector_removal_budget_percent", "mean"),
                    mean_poison_recall_percent=("poison_recall_percent", "mean"),
                )
                .reset_index()
                .sort_values(["dataset_label", "poison_rate_label", "defense_method"])
            )
            display(detector_coverage.round(3))
            """
        ),
        md(
            """
            ## Attack Effectiveness Across Sampling Strategies

            These heatmaps use attack `evasions_success_percent`. They include
            random and distribution-based sampling.
            """
        ),
        code(
            """
            for (dataset, poison_label), group in attack_df.groupby(
                ["dataset_label", "poison_rate_label"], dropna=False
            ):
                pivot = ordered_pivot(
                    group,
                    index="sampling_label",
                    columns="selector_label",
                    values="attack_effectiveness_percent",
                )
                show_heatmap(
                    pivot,
                    f"{dataset}, poison {poison_label}: attack effectiveness / evasion (%)",
                    cmap="magma",
                    vmin=0,
                    vmax=100,
                )
            """
        ),
        md(
            """
            ## Poison-Rate Comparison

            A single-rate experiment appears as one point. Additional rates will
            extend the same plot automatically.
            """
        ),
        code(
            """
            for dataset, group in attack_df.groupby("dataset_label"):
                fig, ax = plt.subplots(figsize=(11, 6))
                sns.lineplot(
                    data=group,
                    x="poison_rate_percent",
                    y="attack_effectiveness_percent",
                    hue="selector_label",
                    style="sampling_label",
                    markers=True,
                    dashes=False,
                    estimator="mean",
                    errorbar=None,
                    ax=ax,
                )
                ax.set_title(f"{dataset}: effectiveness across poison rate and sampling")
                ax.set_xlabel("Poisoned rows / training rows (%)")
                ax.set_ylabel("Attack effectiveness / evasion (%)")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## Detector Results At Comparable Budgets

            For Isolation Forest and Spectral Signature, the loader keeps the run
            whose removal budget matches the poison prevalence in the benign-labeled
            detector pool. HDBSCAN has no fixed contamination budget and is retained
            as its own clustering policy. The budget columns should therefore remain
            visible in paper tables.
            """
        ),
        code(
            """
            detector_columns = [
                "dataset_label",
                "poison_rate_label",
                "sampling_strategy",
                "selector_label",
                "defense_label",
                "defense_setting",
                "attack_effectiveness_percent",
                "poison_recall_percent",
                "poison_removal_precision_percent",
                "clean_false_positive_rate_percent",
                "detector_removal_budget_percent",
                "removed_poisoned_rows",
                "removed_clean_rows",
            ]
            display(
                detector_df
                .sort_values(["dataset_label", "poison_rate_percent", "selector_label", "defense_label"])
                [detector_columns]
                .round(3)
            )

            for (dataset, poison_label, sampling), group in detector_df.groupby(
                ["dataset_label", "poison_rate_label", "sampling_strategy"], dropna=False
            ):
                for metric, title, cmap, center, vmin, vmax in [
                    ("poison_recall_percent", "Poison recall (%)", "YlGnBu", None, 0, 100),
                    ("poison_removal_precision_percent", "Removal precision (%)", "YlGnBu", None, 0, 100),
                    ("clean_false_positive_rate_percent", "Clean false-positive rate (%)", "rocket_r", None, 0, 100),
                ]:
                    pivot = ordered_pivot(
                        group,
                        index="defense_label",
                        columns="selector_label",
                        values=metric,
                    )
                    show_heatmap(
                        pivot,
                        f"{dataset}, poison {poison_label}, {sampling}: {title}",
                        cmap=cmap,
                        center=center,
                        vmin=vmin,
                        vmax=vmax,
                    )
            """
        ),
        md("## Effectiveness Versus Defense Detectability"),
        code(
            """
            for (dataset, poison_label), group in detector_df.groupby(
                ["dataset_label", "poison_rate_label"], dropna=False
            ):
                fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
                sns.scatterplot(
                    data=group,
                    x="attack_effectiveness_percent",
                    y="poison_recall_percent",
                    hue="defense_label",
                    style="selector_label",
                    size="detector_removal_budget_percent",
                    sizes=(50, 260),
                    alpha=0.85,
                    ax=axes[0],
                )
                axes[0].set_title(f"{dataset}, poison {poison_label}: effectiveness vs detection")
                axes[0].set_xlabel("Attack effectiveness / evasion (%)")
                axes[0].set_ylabel("Poison recall (%)")
                axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

                sns.scatterplot(
                    data=group,
                    x="clean_false_positive_rate_percent",
                    y="poison_recall_percent",
                    hue="defense_label",
                    style="selector_label",
                    size="detector_removal_budget_percent",
                    sizes=(50, 260),
                    alpha=0.85,
                    ax=axes[1],
                )
                axes[1].plot([0, 100], [0, 100], linestyle="--", color="black", alpha=0.35)
                axes[1].set_title("Poison recall versus clean false positives")
                axes[1].set_xlabel("Clean false-positive rate (%)")
                axes[1].set_ylabel("Poison recall (%)")
                axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## Optional Mitigation After Retraining

            Most recent experiments intentionally stop after detection. This section
            shows ASR reduction and clean-accuracy cost only for configurations that
            already have defended retraining outputs; missing rows are not treated as
            defense failures.
            """
        ),
        code(
            """
            if retrain_df.empty:
                print("No matching retraining outputs are available.")
            else:
                retrain_columns = [
                    "dataset_label",
                    "poison_rate_label",
                    "sampling_strategy",
                    "selector_label",
                    "defense_label",
                    "backdoored_asr_percent",
                    "defended_asr_percent",
                    "asr_reduction_pp",
                    "clean_accuracy_delta_pp",
                ]
                display(
                    retrain_df
                    .sort_values(["dataset_label", "poison_rate_percent", "selector_label", "defense_label"])
                    [retrain_columns]
                    .round(3)
                )
            """
        ),
        md("## Selector-Level Paper Table"),
        code(
            """
            selector_summary = (
                detector_df.groupby(
                    ["dataset_label", "poison_rate_label", "sampling_strategy", "value_selector", "selector_label"],
                    observed=True,
                )
                .agg(
                    attack_effectiveness_percent=("attack_effectiveness_percent", "mean"),
                    mean_poison_recall_percent=("poison_recall_percent", "mean"),
                    mean_removal_precision_percent=("poison_removal_precision_percent", "mean"),
                    mean_clean_false_positive_rate_percent=("clean_false_positive_rate_percent", "mean"),
                    max_clean_rows_removed=("removed_clean_rows", "max"),
                    defense_methods=("defense_method", "nunique"),
                )
                .reset_index()
                .sort_values(["dataset_label", "poison_rate_label", "attack_effectiveness_percent"], ascending=[True, True, False])
            )
            display(selector_summary.round(3))
            """
        ),
        md("## Optional Export"),
        code(
            """
            SAVE_TABLES = False
            if SAVE_TABLES:
                out_dir = RESULTS_ROOT / "tables"
                out_dir.mkdir(parents=True, exist_ok=True)
                attack_df.to_csv(out_dir / "paper_attack_effectiveness_catalog.csv", index=False)
                detector_df.to_csv(out_dir / "paper_defense_detectability_catalog.csv", index=False)
                retrain_df.to_csv(out_dir / "paper_defense_retraining_catalog.csv", index=False)
                selector_summary.to_csv(out_dir / "paper_selector_defense_summary.csv", index=False)
                print("Saved tables to", out_dir)
            else:
                print("SAVE_TABLES is False; no files written.")
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "06_compare_defense_metrics.ipynb", cells)


def notebook_07():
    cells = [
        md(
            """
            # Detectability Diagnostics

            This notebook compares non-defense detectability signals against attack
            effectiveness for every paper value selector. It distinguishes dataset,
            poison rate, sampling strategy, and target-feature policy.

            Current artifact diagnostics are preferred over historical aggregate
            tables. The current experiment matrix contains 1%, 3%, and 5% poisoning
            under random and distribution-based sampling.
            """
        ),
        code(COMMON_SETUP),
        md("## Analysis Controls"),
        code(FILTER_HELPERS),
        md("## Load Effectiveness, Artifacts, And Diagnostics"),
        code(
            """
            attack_df_all = load_attack_summaries(RESULTS_ROOT)
            artifact_df_all = discover_attack_artifacts(RESULTS_ROOT)
            diagnostic_df_all = load_detectability_summaries(
                RESULTS_ROOT, include_legacy_aggregate=True
            )
            score_df_all = load_detectability_score_metrics(
                RESULTS_ROOT, include_legacy_aggregate=True
            )

            attack_df = filter_paper_rows(attack_df_all)
            artifact_df = filter_paper_rows(artifact_df_all)
            diagnostic_df = filter_paper_rows(diagnostic_df_all)
            score_df = filter_paper_rows(score_df_all)
            merged_df = merge_effectiveness(diagnostic_df, attack_df)

            print("Attack rows:", len(attack_df))
            print("Attack artifact records:", len(artifact_df))
            print("Diagnostics-ready artifacts:", int(artifact_df["diagnostics_ready"].sum()))
            print("Diagnostic summaries:", len(diagnostic_df))
            print("Diagnostic score rows:", len(score_df))
            if not diagnostic_df.empty:
                display(
                    diagnostic_df.groupby(
                        ["dataset_label", "poison_rate_label", "sampling_strategy", "diagnostic_source"],
                        dropna=False,
                    )
                    .size()
                    .rename("rows")
                    .reset_index()
                )
            """
        ),
        md(
            """
            ## Coverage And Missing Diagnostics

            `artifact_available=True` means diagnostics can be generated directly
            with the existing files. Summary-only runs still need
            `--save-attack-artifacts --save-defense-inputs` before diagnostics can be
            computed.
            """
        ),
        code(
            """
            attack_coverage = comparison_coverage(
                attack_df,
                datasets=FOCUS_DATASETS,
                samplings=PAPER_SAMPLINGS,
                target_features=PAPER_TARGET_FEATURES,
            )
            display(attack_coverage)

            keys = [
                "dataset_label",
                "sampling_strategy",
                "poison_rate_train",
                "feature_selector",
                "value_selector",
                "target_features",
            ]
            available = artifact_df[
                artifact_df["diagnostics_ready"]
            ][keys + ["artifact_dir"]].drop_duplicates(keys)
            artifact_status = artifact_df[
                keys + ["diagnostics_ready", "missing_diagnostic_files"]
            ].drop_duplicates(keys)
            completed = diagnostic_df[keys + ["diagnostic_source"]].drop_duplicates(keys)
            missing_diagnostics = (
                attack_df[keys + ["poison_rate_label", "summary_df_path"]]
                .drop_duplicates(keys)
                .merge(artifact_status, on=keys, how="left")
                .merge(available, on=keys, how="left")
                .merge(completed, on=keys, how="left")
            )
            missing_diagnostics["artifact_available"] = missing_diagnostics["artifact_dir"].notna()
            missing_diagnostics = missing_diagnostics[
                missing_diagnostics["diagnostic_source"].isna()
            ].sort_values(["dataset_label", "poison_rate_train", "sampling_strategy", "value_selector"])

            print("Attack configurations without matching diagnostics:", len(missing_diagnostics))
            display(
                missing_diagnostics[
                    [
                        "dataset_label",
                        "poison_rate_label",
                        "sampling_strategy",
                        "value_selector",
                        "artifact_available",
                        "missing_diagnostic_files",
                        "artifact_dir",
                    ]
                ].head(100)
            )
            """
        ),
        md("## Commands For Available Missing Artifacts"),
        code(
            """
            runnable = missing_diagnostics[missing_diagnostics["artifact_available"]].copy()
            if runnable.empty:
                print("No complete artifacts are waiting for diagnostics.")
            else:
                for artifact_dir in runnable["artifact_dir"].drop_duplicates():
                    print(
                        "python3 -m run_detectability_diagnostics "
                        f'--artifact-dir "{artifact_dir}" --overwrite'
                    )
            """
        ),
        md(
            """
            ## Effectiveness Across Poison Rate And Sampling

            This section remains useful even before every detectability diagnostic
            is generated.
            """
        ),
        code(
            """
            for (dataset, poison_label), group in attack_df.groupby(
                ["dataset_label", "poison_rate_label"], dropna=False
            ):
                pivot = ordered_pivot(
                    group,
                    index="sampling_label",
                    columns="selector_label",
                    values="attack_effectiveness_percent",
                )
                show_heatmap(
                    pivot,
                    f"{dataset}, poison {poison_label}: attack effectiveness / evasion (%)",
                    cmap="magma",
                    vmin=0,
                    vmax=100,
                )

            for dataset, group in attack_df.groupby("dataset_label"):
                fig, ax = plt.subplots(figsize=(11, 6))
                sns.lineplot(
                    data=group,
                    x="poison_rate_percent",
                    y="attack_effectiveness_percent",
                    hue="selector_label",
                    style="sampling_label",
                    markers=True,
                    dashes=False,
                    estimator="mean",
                    errorbar=None,
                    ax=ax,
                )
                ax.set_title(f"{dataset}: attack effectiveness across poison rates")
                ax.set_xlabel("Poisoned rows / training rows (%)")
                ax.set_ylabel("Attack effectiveness / evasion (%)")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## Detectability Heatmaps

            Higher AUROC means stronger separation between poisoned and clean benign
            rows. Trigger rarity and exact trigger matching assume knowledge of the
            trigger and should be described as diagnostic or oracle-informed signals.
            SHAP concentration and kNN trigger-subspace separation are less direct.
            """
        ),
        code(
            """
            detectability_metrics = [
                ("marginal_neg_log10_frequency_sum", "Marginal rarity sum", "mako", ".1f"),
                ("trigger_rarity_score_auroc", "Trigger rarity AUROC", "crest", ".2f"),
                ("shap_trigger_abs_ratio_auroc", "SHAP concentration AUROC", "crest", ".2f"),
                ("knn_trigger_mean_distance_auroc", "kNN trigger-distance AUROC", "crest", ".2f"),
            ]

            for (dataset, poison_label, sampling), group in diagnostic_df.groupby(
                ["dataset_label", "poison_rate_label", "sampling_strategy"], dropna=False
            ):
                for metric, title, cmap, fmt in detectability_metrics:
                    if metric not in group or group[metric].notna().sum() == 0:
                        continue
                    pivot = ordered_pivot(
                        group,
                        index="feature_selector",
                        columns="selector_label",
                        values=metric,
                    )
                    show_heatmap(
                        pivot,
                        f"{dataset}, poison {poison_label}, {sampling}: {title}",
                        cmap=cmap,
                        fmt=fmt,
                        vmin=0 if "auroc" in metric else None,
                        vmax=1 if "auroc" in metric else None,
                    )
            """
        ),
        md("## Effectiveness Versus Non-Defense Detectability"),
        code(
            """
            for metric, metric_title, _, _ in detectability_metrics:
                if merged_df.empty or metric not in merged_df:
                    continue
                focus = merged_df.dropna(subset=["attack_effectiveness_percent", metric]).copy()
                if focus.empty:
                    continue
                for (dataset, poison_label), group in focus.groupby(
                    ["dataset_label", "poison_rate_label"], dropna=False
                ):
                    fig, ax = plt.subplots(figsize=(9, 5.5))
                    sns.scatterplot(
                        data=group,
                        x="attack_effectiveness_percent",
                        y=metric,
                        hue="selector_label",
                        style="sampling_label",
                        size="watermark_size",
                        sizes=(60, 180),
                        s=90,
                        ax=ax,
                    )
                    ax.set_title(
                        f"{dataset}, poison {poison_label}: effectiveness vs {metric_title}"
                    )
                    ax.set_xlabel("Attack effectiveness / evasion (%)")
                    ax.set_ylabel(metric_title)
                    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                    plt.tight_layout()
                    plt.show()
            """
        ),
        md("## Poison-Rate Detectability Comparison"),
        code(
            """
            if diagnostic_df.empty:
                print("No diagnostics available for poison-rate comparison.")
            else:
                metric = "shap_trigger_abs_ratio_auroc"
                focus = diagnostic_df.dropna(subset=[metric]).copy()
                for dataset, group in focus.groupby("dataset_label"):
                    fig, ax = plt.subplots(figsize=(11, 6))
                    sns.lineplot(
                        data=group,
                        x="poison_rate_percent",
                        y=metric,
                        hue="selector_label",
                        style="sampling_label",
                        markers=True,
                        dashes=False,
                        estimator="mean",
                        errorbar=None,
                        ax=ax,
                    )
                    ax.set_title(f"{dataset}: SHAP detectability across poison rates")
                    ax.set_xlabel("Poisoned rows / training rows (%)")
                    ax.set_ylabel("SHAP concentration AUROC")
                    ax.set_ylim(0, 1.02)
                    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                    plt.tight_layout()
                    plt.show()
            """
        ),
        md("## Recall At A Suspicion Budget"),
        code(
            """
            if score_df.empty:
                print("No detectability score metrics available.")
            else:
                for (dataset, poison_label, sampling), group in score_df.groupby(
                    ["dataset_label", "poison_rate_label", "sampling_strategy"], dropna=False
                ):
                    g = sns.relplot(
                        data=group,
                        x="top_percent",
                        y="poison_recall",
                        hue="selector_label",
                        col="score_name",
                        col_wrap=2,
                        kind="line",
                        marker="o",
                        facet_kws={"sharex": True, "sharey": True},
                        height=4,
                        aspect=1.15,
                    )
                    g.set_axis_labels("Top suspicious rows removed (%)", "Poison recall")
                    g.set_titles("{col_name}")
                    g.fig.suptitle(
                        f"{dataset}, poison {poison_label}, {sampling}: recall at budget",
                        y=1.02,
                    )
                    for ax in g.axes.flat:
                        ax.set_ylim(0, 1.02)
                    plt.show()
            """
        ),
        md("## Optional Export"),
        code(
            """
            SAVE_TABLES = False
            if SAVE_TABLES:
                out_dir = RESULTS_ROOT / "tables"
                out_dir.mkdir(parents=True, exist_ok=True)
                attack_df.to_csv(out_dir / "paper_attack_effectiveness_for_detectability.csv", index=False)
                diagnostic_df.to_csv(out_dir / "paper_detectability_catalog.csv", index=False)
                merged_df.to_csv(out_dir / "paper_effectiveness_vs_detectability.csv", index=False)
                missing_diagnostics.to_csv(out_dir / "paper_missing_detectability_runs.csv", index=False)
                print("Saved tables to", out_dir)
            else:
                print("SAVE_TABLES is False; no files written.")
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "07_detectability_diagnostics.ipynb", cells)


if __name__ == "__main__":
    notebook_06()
    notebook_07()
