"""Compare component-selection rules for component-guided trigger matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .defense.component_trigger_matching import (
    SUPPORTED_COMPONENT_RULES,
    SUPPORTED_FEATURE_SOURCES,
    SUPPORTED_PAIR_APPLY_SCOPES,
    SUPPORTED_PAIR_RANKS,
    SUPPORTED_ROW_RANKS,
    run_component_trigger_matching,
)
from .utils.paths import resolve_path


DEFAULT_COMPONENT_RULES = [
    "largest",
    "density_proxy_log",
    "mean_global_mahalanobis",
    "smallest_cov_volume",
    "avg_log_likelihood",
    "responsibility_entropy_mean",
    "trigger_weighted_lift_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="Attack artifact directory.")
    parser.add_argument("--gmm-dir", required=True, help="GMM defense output directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Sweep output directory. Defaults to <gmm-dir>/component_trigger_rule_sweep.",
    )
    parser.add_argument(
        "--component-rules",
        nargs="+",
        default=DEFAULT_COMPONENT_RULES,
        choices=sorted(SUPPORTED_COMPONENT_RULES),
        help="Component-selection rules to compare.",
    )
    parser.add_argument("--top-components", type=int, default=3)
    parser.add_argument(
        "--candidate-feature-source",
        default="nonhashed",
        choices=sorted(SUPPORTED_FEATURE_SOURCES),
    )
    parser.add_argument("--min-component-count", type=int, default=20)
    parser.add_argument("--min-lift", type=float, default=2.0)
    parser.add_argument("--max-global-frequency", type=float, default=0.10)
    parser.add_argument("--top-values-per-feature", type=int, default=3)
    parser.add_argument("--top-pairs-per-component", type=int, default=50)
    parser.add_argument("--pair-rank", default="weighted_lift", choices=sorted(SUPPORTED_PAIR_RANKS))
    parser.add_argument("--pair-apply-scope", default="global", choices=sorted(SUPPORTED_PAIR_APPLY_SCOPES))
    parser.add_argument("--row-rank", default="matched_pairs", choices=sorted(SUPPORTED_ROW_RANKS))
    parser.add_argument("--removal-percent", type=float, default=1.0)
    parser.add_argument("--min-matched-pairs", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gmm_dir = resolve_path(args.gmm_dir) or Path(args.gmm_dir)
    if args.output_dir is None:
        output_dir = gmm_dir / "component_trigger_rule_sweep"
    else:
        output_dir = resolve_path(args.output_dir) or Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rule in args.component_rules:
        rule_output_dir = output_dir / f"rule_{rule}"
        result = run_component_trigger_matching(
            artifact_dir=args.artifact_dir,
            gmm_dir=args.gmm_dir,
            output_dir=rule_output_dir,
            component_rule=rule,
            top_components=args.top_components,
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
        result_dict = result if isinstance(result, dict) else result.__dict__
        rows.append(
            {
                "component_rule": rule,
                "selected_components": json.dumps(result_dict.get("selected_components")),
                "candidate_features": result_dict.get("candidate_features"),
                "mined_pairs": result_dict.get("mined_pairs"),
                "selected_pairs": result_dict.get("selected_pairs"),
                "removed_rows": result_dict.get("removed_rows"),
                "removed_poisoned_rows": result_dict.get("removed_poisoned_rows"),
                "poison_recall": result_dict.get("poison_recall"),
                "output_dir": result_dict.get("output_dir"),
                "metadata_path": result_dict.get("metadata_path"),
            }
        )

    summary = pd.DataFrame(rows)
    summary_csv = output_dir / "component_rule_sweep_summary.csv"
    summary_json = output_dir / "component_rule_sweep_summary.json"
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_csv": str(summary_csv), "summary_json": str(summary_json), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
