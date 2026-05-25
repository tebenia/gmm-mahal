"""Run component-guided trigger-like feature-value mining."""

from __future__ import annotations

import argparse
import json

from .defense.component_trigger_matching import (
    SUPPORTED_COMPONENT_RULES,
    SUPPORTED_FEATURE_SOURCES,
    SUPPORTED_PAIR_APPLY_SCOPES,
    SUPPORTED_PAIR_RANKS,
    SUPPORTED_ROW_RANKS,
    run_component_trigger_matching,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="Attack artifact directory.")
    parser.add_argument("--gmm-dir", required=True, help="GMM defense output directory with suspicious_scores.csv.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <gmm-dir>/component_trigger_matching/<settings>.",
    )
    parser.add_argument(
        "--component-rule",
        default="global_z_max",
        choices=sorted(SUPPORTED_COMPONENT_RULES),
        help="Observable rule for selecting components to mine.",
    )
    parser.add_argument("--top-components", type=int, default=3, help="Number of components to use with a rule.")
    parser.add_argument(
        "--components",
        type=int,
        nargs="+",
        default=None,
        help="Explicit component ids to mine. Overrides --component-rule.",
    )
    parser.add_argument(
        "--candidate-feature-source",
        default="nonhashed",
        choices=sorted(SUPPORTED_FEATURE_SOURCES),
        help="Feature pool used for feature-value mining.",
    )
    parser.add_argument("--min-component-count", type=int, default=20)
    parser.add_argument("--min-lift", type=float, default=2.0)
    parser.add_argument("--max-global-frequency", type=float, default=0.10)
    parser.add_argument("--top-values-per-feature", type=int, default=3)
    parser.add_argument("--top-pairs-per-component", type=int, default=50)
    parser.add_argument(
        "--pair-rank",
        default="weighted_lift",
        choices=sorted(SUPPORTED_PAIR_RANKS),
        help="Ranking used when selecting top mined pairs per component.",
    )
    parser.add_argument(
        "--pair-apply-scope",
        default="global",
        choices=sorted(SUPPORTED_PAIR_APPLY_SCOPES),
        help="Apply mined pairs only in their source component, or globally to every benign row.",
    )
    parser.add_argument(
        "--row-rank",
        default="matched_pairs",
        choices=sorted(SUPPORTED_ROW_RANKS),
        help="Ranking used when selecting rows for removal.",
    )
    parser.add_argument("--removal-percent", type=float, default=1.0)
    parser.add_argument("--min-matched-pairs", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=None, help="Debug/smoke limit on candidate features.")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug/smoke limit on benign score rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_component_trigger_matching(
        artifact_dir=args.artifact_dir,
        gmm_dir=args.gmm_dir,
        output_dir=args.output_dir,
        component_rule=args.component_rule,
        top_components=args.top_components,
        components=args.components,
        candidate_feature_source=args.candidate_feature_source,
        min_component_count=args.min_component_count,
        min_lift=args.min_lift,
        max_global_frequency=args.max_global_frequency,
        top_values_per_feature=args.top_values_per_feature,
        top_pairs_per_component=args.top_pairs_per_component,
        pair_rank=args.pair_rank,
        pair_apply_scope=args.pair_apply_scope,
        row_rank=args.row_rank,
        removal_percent=args.removal_percent,
        min_matched_pairs=args.min_matched_pairs,
        max_features=args.max_features,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if hasattr(result, "__dict__"):
        result = result.__dict__
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
