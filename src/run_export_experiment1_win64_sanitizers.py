from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.experiment1_win64_sanitizer_summary import build_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Experiment 1 Win64 seed-43 sanitizer metrics.")
    parser.add_argument("--results-root", type=Path, default=Path("results/experiment 1"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiment 1/tables/sanitizer_metrics_seed43_win64"),
    )
    args = parser.parse_args()
    print(json.dumps({k: str(v) for k, v in build_tables(args.results_root, args.output_dir).items()}, indent=2))


if __name__ == "__main__":
    main()

