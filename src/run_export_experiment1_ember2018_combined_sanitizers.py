from __future__ import annotations

import argparse

from src.analysis.experiment1_ember2018_combined_sanitizer_summary import export_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results/experiment 1/ember2018/seed_43/poison_rate_0p01",
    )
    parser.add_argument(
        "--output-dir",
        default="results/experiment 1/tables/sanitizer_metrics_ember2018_combined_seed43",
    )
    args = parser.parse_args()
    for name, path in export_summary(args.results_root, args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
