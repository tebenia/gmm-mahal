"""Run HDBSCAN SHAP-space loss-ranked clean-label defense."""

from __future__ import annotations

import argparse
import json

from .defense.hdbscan_shap_loss import (
    SUPPORTED_COVERAGE_UNITS,
    SUPPORTED_NOISE_POLICIES,
    run_hdbscan_shap_loss_defense,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="Attack artifact directory.")
    parser.add_argument(
        "--preprocess-dir",
        default=None,
        help="Defense preprocessing directory containing X_shap_reduced.npy. Defaults to <artifact-dir>/defense_preprocessing/standardized_pca50.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <artifact-dir>/hdbscan_shap_loss/<settings>.",
    )
    parser.add_argument(
        "--clean-fraction",
        type=float,
        default=0.80,
        help="Fraction of lowest-loss clusters or rows to keep as clean.",
    )
    parser.add_argument(
        "--coverage-unit",
        default="clusters",
        choices=sorted(SUPPORTED_COVERAGE_UNITS),
        help="Apply --clean-fraction to cluster count or benign-row count.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Explicit HDBSCAN min_cluster_size. Overrides --min-cluster-percent.",
    )
    parser.add_argument(
        "--min-cluster-percent",
        type=float,
        default=0.5,
        help="HDBSCAN min_cluster_size as percent of total watermarked training rows.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Explicit HDBSCAN min_samples. Overrides --min-samples-percent.",
    )
    parser.add_argument(
        "--min-samples-percent",
        type=float,
        default=0.1,
        help="HDBSCAN min_samples as percent of total watermarked training rows.",
    )
    parser.add_argument("--noise-policy", default="split", choices=sorted(SUPPORTED_NOISE_POLICIES))
    parser.add_argument("--noise-chunk-size", type=int, default=1000)
    parser.add_argument(
        "--standardize-reduced",
        action="store_true",
        help="Apply a second StandardScaler to X_shap_reduced before HDBSCAN.",
    )
    parser.add_argument("--surrogate-num-boost-round", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-benign-rows", type=int, default=None, help="Smoke-test limit on benign rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_hdbscan_shap_loss_defense(
        artifact_dir=args.artifact_dir,
        preprocess_dir=args.preprocess_dir,
        output_dir=args.output_dir,
        clean_fraction=args.clean_fraction,
        coverage_unit=args.coverage_unit,
        min_cluster_size=args.min_cluster_size,
        min_cluster_percent=args.min_cluster_percent,
        min_samples=args.min_samples,
        min_samples_percent=args.min_samples_percent,
        noise_policy=args.noise_policy,
        noise_chunk_size=args.noise_chunk_size,
        standardize_reduced=args.standardize_reduced,
        surrogate_num_boost_round=args.surrogate_num_boost_round,
        random_state=args.random_state,
        max_benign_rows=args.max_benign_rows,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if hasattr(result, "__dict__"):
        result = result.__dict__
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
