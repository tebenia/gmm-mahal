"""Export new three-seed paper assets without replacing the two-seed originals."""

from pathlib import Path
import os
import re

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import generate_mean_isolation_forest_figure as figure


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/experiment 1/tables/final_result_seed42_seed43_seed44/poison_1p/three_seed_mean_std_effectiveness_detectability.csv"


def export_table(frame, dataset, stem):
    template = ROOT / "paper/table" / f"{stem}.tex"
    text = template.read_text()
    text = text.replace(f"tab:{stem}", f"tab:{stem}_3seeds")
    text = text.replace("across 3 runs.", "across three runs (seeds 42, 43, and 44).")
    sampling = None
    row_count = 0
    result = []
    for line in text.splitlines():
        if "\\textit{Random sampling}" in line:
            sampling = "Random"
        elif "\\textit{Distribution-based sampling}" in line:
            sampling = "Distribution-based"
        match = re.match(r"(\s*)([A-Za-z0-9*]+)\s+&\s+[0-9]", line)
        if match:
            label = match.group(2)
            code = next(code for code, name in figure.SELECTOR_LABELS.items() if name == label.rstrip("*"))
            rows = frame[(frame.dataset == dataset) & (frame.sampling == sampling) & (frame.value_selector == code)]
            if len(rows) != 1:
                raise ValueError(f"Expected one row for {dataset}/{sampling}/{code}")
            row = rows.iloc[0]
            metrics = [("asr_mean", "asr_std"), ("clean_accuracy_mean", "clean_accuracy_std"),
                       ("poison_recall_mean_isolation_forest", "poison_recall_std_isolation_forest")]
            values = [f"{row[mean]:.2f} $\\pm$ {row[std]:.2f}" for mean, std in metrics]
            line = match.group(1) + f"{label:<20}" + " & " + " & ".join(values) + r" \\"
            row_count += 1
        result.append(line)
    if row_count != 16:
        raise ValueError(f"Expected 16 table rows, found {row_count}")
    output = template.with_name(f"{stem}_3seeds.tex")
    output.write_text("\n".join(result) + "\n")
    print(output)


def main():
    frame = pd.read_csv(INPUT)
    keys = ["dataset", "sampling", "value_selector"]
    if len(frame) != 32 or frame.duplicated(keys).any() or not frame.seed_count.eq(3).all():
        raise ValueError("Expected 32 unique, complete three-seed conditions")
    metrics = ["asr_mean", "asr_std", "clean_accuracy_mean", "clean_accuracy_std",
               "poison_recall_mean_isolation_forest", "poison_recall_std_isolation_forest"]
    if not np.isfinite(frame[metrics].to_numpy()).all():
        raise ValueError("Missing or nonfinite paper metrics")
    export_table(frame, "EMBER 2018", "ember2018_1p_if")
    export_table(frame, "EMBER 2024 Win64", "ember2024_1p_if")
    figure.INPUT = INPUT
    figure.OUTPUT_PDF = ROOT / "paper/figure/asr_vs_isolation_forest_recall_1p_mean_3seeds.pdf"
    figure.OUTPUT_PNG = figure.OUTPUT_PDF.with_suffix(".png")
    figure.DATASET_ORDER = ["EMBER 2018", "EMBER 2024 Win64"]
    figure.DATASET_LABELS = {name: name for name in figure.DATASET_ORDER}
    figure.main()


if __name__ == "__main__":
    main()
