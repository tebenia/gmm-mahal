"""Run notebook-style Isolation Forest or Spectral Signature defenses."""

from __future__ import annotations

import argparse
import json

from .defense.severi_detectors import (
    SUPPORTED_FEATURE_MODES,
    SUPPORTED_METHODS,
    run_severi_detector_defense,
)


def parse_contamination(value: str) -> str | float:
    if value.lower() == "auto":
        return "auto"
    parsed = float(value)
    if not 0 < parsed <= 0.5:
        raise argparse.ArgumentTypeError("contamination must be 'auto' or a float in (0, 0.5]")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="Attack artifact directory.")
    parser.add_argument("--method", required=True, choices=sorted(SUPPORTED_METHODS))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <artifact-dir>/severi_detectors/<settings>.",
    )
    parser.add_argument(
        "--feature-mode",
        default="hybrid",
        choices=sorted(SUPPORTED_FEATURE_MODES),
        help="Feature subspace: watermark ids, top mean-abs SHAP ids, or watermark padded with SHAP.",
    )
    parser.add_argument("--top-k", type=int, default=32, help="Maximum selected feature count.")
    parser.add_argument(
        "--contamination",
        type=parse_contamination,
        default="auto",
        help="IsolationForest contamination. Use 'auto' or a float, for example 0.005.",
    )
    parser.add_argument(
        "--removal-percent",
        type=float,
        default=None,
        help=(
            "Remove the top suspicious percent by score. For IsolationForest this overrides "
            "the model threshold. For Spectral Signature the default is 1%% when omitted."
        ),
    )
    parser.add_argument(
        "--spectral-oracle-poison-count",
        action="store_true",
        help=(
            "Spectral Signature diagnostic matching the old notebook default: remove the known "
            "number of poisoned benign rows. This uses ground-truth metadata as a removal budget."
        ),
    )
    parser.add_argument("--no-standardize", action="store_true", help="Disable MinMax scaling to [-1, 1].")
    parser.add_argument("--batch-size", type=int, default=8192, help="Rows per selected-feature extraction batch.")
    parser.add_argument("--max-benign-rows", type=int, default=None, help="Smoke-test limit on benign rows.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_severi_detector_defense(
        artifact_dir=args.artifact_dir,
        method=args.method,
        output_dir=args.output_dir,
        feature_mode=args.feature_mode,
        top_k=args.top_k,
        contamination=args.contamination,
        removal_percent=args.removal_percent,
        spectral_oracle_poison_count=args.spectral_oracle_poison_count,
        standardize=not args.no_standardize,
        batch_size=args.batch_size,
        max_benign_rows=args.max_benign_rows,
        random_state=args.random_state,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if hasattr(result, "__dict__"):
        result = result.__dict__
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
