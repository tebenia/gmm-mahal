from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.experiment2_sanitizer_summary import (
    build_experiment2_sanitizer_tables,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export compact Experiment 2 sanitizer metrics before removing arrays."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/experiment 2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiment 2/tables/sanitizer_metrics"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = build_experiment2_sanitizer_tables(args.results_root, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()

