"""Generate the paper-format ASR versus Isolation Forest poison-recall figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "results/experiment 1/plots/final_result/poison_1p_seed_comparison/two_seed_mean_std_effectiveness_detectability.csv"
OUTPUT_PDF = PROJECT_ROOT / "paper/figure/asr_vs_isolation_forest_recall_1p_mean.pdf"
OUTPUT_PNG = PROJECT_ROOT / "paper/figure/asr_vs_isolation_forest_recall_1p_mean.png"

SELECTOR_ORDER = [
    "min_population_new",
    "argmin_Nv_sum_abs_shap",
    "combined_shap",
    "low_shap_signed",
    "frequency_bounded_signed_shap",
    "corr_count_abs_shap",
    "benign_prototype",
    "quantile_95",
]
SELECTOR_LABELS = {
    "min_population_new": "MinPopulation",
    "argmin_Nv_sum_abs_shap": "CountAbsSHAP",
    "combined_shap": "CombinedSHAP",
    "low_shap_signed": "LowSHAPSigned",
    "frequency_bounded_signed_shap": "FreqBoundedSignedSHAP",
    "corr_count_abs_shap": "CorrCountAbsSHAP",
    "benign_prototype": "BenignPrototype",
    "quantile_95": "Quantile95",
}
DATASET_ORDER = ["EMBER2018", "EMBER2024 Win64"]
DATASET_LABELS = {
    "EMBER2018": "EMBER 2018",
    "EMBER2024 Win64": "EMBER 2024 Win64",
}
SAMPLING_ORDER = ["Random", "Distribution-based"]
SAMPLING_MARKERS = {"Random": "o", "Distribution-based": "^"}


def main() -> None:
    frame = pd.read_csv(INPUT)
    frame = frame[
        frame["dataset"].isin(DATASET_ORDER)
        & frame["sampling"].isin(SAMPLING_ORDER)
        & frame["value_selector"].isin(SELECTOR_ORDER)
    ].copy()
    expected = len(DATASET_ORDER) * len(SAMPLING_ORDER) * len(SELECTOR_ORDER)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} mean rows, found {len(frame)}")

    colors = dict(zip(SELECTOR_ORDER, plt.get_cmap("tab10").colors[:len(SELECTOR_ORDER)]))
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 8.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = frame[frame["dataset"] == dataset]
        for selector_index, selector in enumerate(SELECTOR_ORDER):
            for sampling_index, sampling in enumerate(SAMPLING_ORDER):
                row = subset[
                    (subset["value_selector"] == selector)
                    & (subset["sampling"] == sampling)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]
                # Small deterministic offsets preserve the original figure's readability
                # when two mean points are nearly identical.
                offset = (sampling_index - 0.5) * 0.55
                ax.scatter(
                    float(row["poison_recall_mean_isolation_forest"]) + offset,
                    float(row["asr_mean"]) + offset * 0.15,
                    s=150,
                    marker=SAMPLING_MARKERS[sampling],
                    color=colors[selector],
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.9,
                    label=SELECTOR_LABELS[selector] if dataset == DATASET_ORDER[0] and sampling == SAMPLING_ORDER[0] else "_nolegend_",
                )
        ax.set_title(DATASET_LABELS[dataset], fontsize=14, pad=8)
        ax.set_ylabel("ASR / evasion (%)", fontsize=11)
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(True, color="#bdbdbd", linewidth=1.0, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)

    axes[-1].set_xlabel("Isolation Forest Poison Recall (%)", fontsize=11)
    axes[-1].set_xticks(np.arange(0, 101, 20))
    fig.suptitle("ASR vs Isolation Forest Poison Recall (1% poison)", fontsize=15, y=0.975)

    selector_handles, selector_labels = axes[0].get_legend_handles_labels()
    selector_legend = fig.legend(
        selector_handles,
        selector_labels,
        title="Selector",
        loc="lower center",
        bbox_to_anchor=(0.31, 0.015),
        ncol=2,
        fontsize=7.2,
        title_fontsize=8.2,
        frameon=True,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.add_artist(selector_legend)
    sampling_handles = [
        plt.Line2D(
            [0], [0], marker=SAMPLING_MARKERS[sampling], color="white",
            markerfacecolor="#555555", markeredgecolor="black", markersize=7,
            linestyle="None", label=sampling,
        )
        for sampling in SAMPLING_ORDER
    ]
    fig.legend(
        sampling_handles,
        SAMPLING_ORDER,
        title="Sampling",
        loc="lower center",
        bbox_to_anchor=(0.78, 0.055),
        ncol=1,
        fontsize=7.2,
        title_fontsize=8.2,
        frameon=True,
        handletextpad=0.45,
    )

    fig.subplots_adjust(left=0.14, right=0.97, top=0.875, bottom=0.205, hspace=0.28)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
