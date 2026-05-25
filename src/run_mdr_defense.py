"""Run an MDR-inspired feature-value cleaning baseline."""

from __future__ import annotations

import argparse
import json

from .defense.mdr_inspired import SUPPORTED_REMOVE_SCOPES, run_mdr_inspired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="Attack artifact directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <artifact-dir>/mdr_inspired.",
    )
    parser.add_argument(
        "--candidate-feature-source",
        default="nonhashed",
        choices=["nonhashed", "all", "oracle_watermark"],
        help="Feature pool used before variance and SHAP dictionary filtering.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(range(3, 13)),
        help="Intersection-index thresholds used for graph construction.",
    )
    parser.add_argument("--variance-threshold", type=float, default=0.0)
    parser.add_argument("--dict-size", type=int, default=40)
    parser.add_argument(
        "--include-positive-shap",
        action="store_true",
        help="Do not restrict per-row dictionaries to negative/goodware-oriented SHAP values.",
    )
    parser.add_argument("--community-tolerance", type=float, default=0.8)
    parser.add_argument("--watermark-tolerance-start", type=float, default=0.8)
    parser.add_argument("--watermark-tolerance-step", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--max-combination-size", type=int, default=3)
    parser.add_argument("--max-bucket-size", type=int, default=2000)
    parser.add_argument("--max-edges", type=int, default=2_000_000)
    parser.add_argument("--max-anti-elements", type=int, default=100)
    parser.add_argument("--max-malware-probes", type=int, default=100)
    parser.add_argument("--remove-scope", default="all", choices=sorted(SUPPORTED_REMOVE_SCOPES))
    parser.add_argument(
        "--value-round-decimals",
        type=int,
        default=None,
        help="Optional rounding before exact feature-value comparisons.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-benign-rows", type=int, default=None, help="Smoke-test limit on benign rows.")
    parser.add_argument("--max-features", type=int, default=None, help="Smoke-test limit on candidate features.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_mdr_inspired(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        candidate_feature_source=args.candidate_feature_source,
        thresholds=args.thresholds,
        variance_threshold=args.variance_threshold,
        dict_size=args.dict_size,
        only_negative_shap=not args.include_positive_shap,
        community_tolerance=args.community_tolerance,
        watermark_tolerance_start=args.watermark_tolerance_start,
        watermark_tolerance_step=args.watermark_tolerance_step,
        window_size=args.window_size,
        max_combination_size=args.max_combination_size,
        max_bucket_size=args.max_bucket_size,
        max_edges=args.max_edges,
        max_anti_elements=args.max_anti_elements,
        max_malware_probes=args.max_malware_probes,
        remove_scope=args.remove_scope,
        value_round_decimals=args.value_round_decimals,
        random_state=args.random_state,
        max_benign_rows=args.max_benign_rows,
        max_features=args.max_features,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if hasattr(result, "__dict__"):
        result = result.__dict__
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
