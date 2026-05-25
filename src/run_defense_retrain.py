"""Retrain and evaluate a defended model after GMM suspicious-row removal."""

from __future__ import annotations

import argparse
import json

from .attack.baseline import DEFAULT_BASELINES_CONFIG
from .defense.retrain_evaluate import run_defense_retrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Attack artifact directory containing watermarked_X.npy, watermarked_y.npy, and metadata.",
    )
    parser.add_argument(
        "--gmm-dir",
        default=None,
        help="GMM defense output directory containing remove_watermarked_idx.npy.",
    )
    parser.add_argument(
        "--remove-watermarked-idx",
        default=None,
        help="Optional explicit path to remove_watermarked_idx.npy. Use when --gmm-dir is not supplied.",
    )
    parser.add_argument(
        "--oracle-remove-poisoned",
        action="store_true",
        help=(
            "Oracle sanity check: remove the true poisoned training rows from defense_metadata.npz. "
            "This is not a real defense; it tests whether perfect detection would reduce ASR."
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline id used to load the clean test set for clean-accuracy evaluation.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_BASELINES_CONFIG),
        help="YAML or JSON baseline config used with --baseline.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <gmm-dir>/defended_retrain.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Use only the first N watermarked training rows for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=None,
        help="Use only the first N evaluation rows for smoke tests.",
    )
    parser.add_argument("--no-save-model", action="store_true", help="Do not save the defended model file.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing retrain outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print resolved inputs without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_defense_retrain(
        artifact_dir=args.artifact_dir,
        gmm_dir=args.gmm_dir,
        remove_idx_path=args.remove_watermarked_idx,
        output_dir=args.output_dir,
        baseline=args.baseline,
        config_path=args.config,
        oracle_remove_poisoned=args.oracle_remove_poisoned,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        save_model=not args.no_save_model,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if hasattr(result, "__dict__"):
        result = result.__dict__
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
