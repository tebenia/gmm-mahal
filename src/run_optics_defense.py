"""Run the OPTICS iterative clean-label defense baseline."""

from __future__ import annotations

import argparse
import json

from .defense.optics_iterative import (
    SUPPORTED_CANDIDATE_FEATURE_SOURCES,
    SUPPORTED_DELTA_TAILS,
    SUPPORTED_NOISE_POLICIES,
    SUPPORTED_SELECTION_MODES,
    run_optics_iterative_defense,
)


def parse_min_cluster_size(value: str) -> int | float | None:
    if value.lower() in {"none", "null"}:
        return None
    if "." in value:
        return float(value)
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="Attack artifact directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <artifact-dir>/optics_iterative.",
    )
    parser.add_argument(
        "--candidate-feature-source",
        default="all",
        choices=sorted(SUPPORTED_CANDIDATE_FEATURE_SOURCES),
        help="Feature pool before entropy/decision-tree top-feature selection.",
    )
    parser.add_argument("--top-features", type=int, default=16)
    parser.add_argument("--no-standardize", action="store_true", help="Disable StandardScaler before OPTICS.")
    parser.add_argument(
        "--importance-max-rows",
        type=int,
        default=50_000,
        help="Maximum rows used to fit the entropy decision tree. Use 0 for all rows.",
    )
    parser.add_argument("--decision-tree-max-depth", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--max-eps", type=float, default=float("inf"))
    parser.add_argument("--xi", type=float, default=0.05)
    parser.add_argument(
        "--min-cluster-size",
        type=parse_min_cluster_size,
        default=None,
        help="OPTICS min_cluster_size. Accepts int, float fraction, or none.",
    )
    parser.add_argument("--noise-policy", default="split", choices=sorted(SUPPORTED_NOISE_POLICIES))
    parser.add_argument("--noise-chunk-size", type=int, default=1000)
    parser.add_argument("--window-fraction", type=float, default=0.05)
    parser.add_argument("--clean-cluster-fraction", type=float, default=0.80)
    parser.add_argument("--selection-mode", default="fixed_threshold", choices=sorted(SUPPORTED_SELECTION_MODES))
    parser.add_argument("--delta-z-threshold", type=float, default=2.0)
    parser.add_argument("--delta-tail", default="lower", choices=sorted(SUPPORTED_DELTA_TAILS))
    parser.add_argument("--surrogate-num-boost-round", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-benign-rows", type=int, default=None, help="Smoke-test limit on benign rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_optics_iterative_defense(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        candidate_feature_source=args.candidate_feature_source,
        top_features=args.top_features,
        standardize=not args.no_standardize,
        importance_max_rows=None if args.importance_max_rows == 0 else args.importance_max_rows,
        decision_tree_max_depth=args.decision_tree_max_depth,
        min_samples=args.min_samples,
        max_eps=args.max_eps,
        xi=args.xi,
        min_cluster_size=args.min_cluster_size,
        noise_policy=args.noise_policy,
        noise_chunk_size=args.noise_chunk_size,
        window_fraction=args.window_fraction,
        clean_cluster_fraction=args.clean_cluster_fraction,
        selection_mode=args.selection_mode,
        delta_z_threshold=args.delta_z_threshold,
        delta_tail=args.delta_tail,
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
