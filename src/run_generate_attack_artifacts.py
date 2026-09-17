"""Generate defense-ready attack artifacts without repeating attack evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attack.baseline import (
    DEFAULT_BASELINES_CONFIG,
    build_contexts,
    describe_detector_artifact_context,
    generate_detector_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_BASELINES_CONFIG),
        help="YAML or JSON baseline configuration file.",
    )
    parser.add_argument("--baseline", required=True, help="Baseline preset from --config.")
    parser.add_argument("--sampling", required=True, help="One train-benign sampling strategy.")
    parser.add_argument("--feature-selection", required=True, help="One feature selector.")
    parser.add_argument("--value-selection", required=True, help="One value selector.")
    parser.add_argument(
        "--target-features",
        default=None,
        choices=[
            "all",
            "non_hashed",
            "feature_space_feasible",
            "problem_space_conservative",
            "severi_exact_overlap",
            "feasible",
        ],
    )
    parser.add_argument("--poison-rate", type=float, required=True, help="One poison rate, e.g. 0.01.")
    parser.add_argument("--watermark-size", type=int, required=True, help="One trigger feature count.")
    parser.add_argument(
        "--defense-shap-batch-size",
        type=int,
        default=8192,
        help="Rows per LightGBM pred_contrib batch.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing artifact package.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and settings without loading data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "sampling_strategies": [args.sampling],
        "feature_selection": [args.feature_selection],
        "value_selection": [args.value_selection],
        "target_features": args.target_features,
        "poison_rates": [args.poison_rate],
        "watermark_sizes": [args.watermark_size],
        "iterations": 1,
    }
    contexts = build_contexts(
        args.baseline,
        overrides=overrides,
        config_path=Path(args.config),
    )
    if len(contexts) != 1:
        raise ValueError(f"Expected exactly one resolved context, received {len(contexts)}")
    context = contexts[0]
    if args.dry_run:
        print(json.dumps(describe_detector_artifact_context(context), indent=2, sort_keys=True))
        return
    result = generate_detector_artifacts(
        context,
        defense_shap_batch_size=args.defense_shap_batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
