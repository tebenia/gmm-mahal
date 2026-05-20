"""Run notebook-equivalent EMBER attack baselines from Python files."""

from __future__ import annotations

import argparse
import json

from .attack_baseline import build_context, describe_context, run_attack_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="ember2024_win64_20pct",
        choices=["ember2018_20pct", "ember2024_win64_20pct", "ember2024_win32_0p0667"],
        help="Baseline preset to run.",
    )
    parser.add_argument(
        "--sampling",
        default=None,
        help="Override train goodware sampling strategy, e.g. random, wasserstein_distance, shap_contribution_distance.",
    )
    parser.add_argument(
        "--feature-selection",
        action="append",
        default=None,
        help="Override feature selector. Can be passed multiple times.",
    )
    parser.add_argument(
        "--value-selection",
        action="append",
        default=None,
        help="Override value selector. Can be passed multiple times.",
    )
    parser.add_argument(
        "--poison-rate",
        action="append",
        type=float,
        default=None,
        help="Override local poison rate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--watermark-size",
        action="append",
        type=int,
        default=None,
        help="Override watermark size. Can be passed multiple times.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override number of attack iterations.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="Override EMBER2024 test fraction. Useful for quick smoke runs.",
    )
    parser.add_argument(
        "--save-attack-artifacts",
        action="store_true",
        help="Also save watermarked train/test arrays, wm_config, and the backdoored model for defense work.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show resolved settings and paths without loading arrays or running the attack.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "sampling_strategy": args.sampling,
        "feature_selection": args.feature_selection,
        "value_selection": args.value_selection,
        "poison_rates": args.poison_rate,
        "watermark_sizes": args.watermark_size,
        "iterations": args.iterations,
        "test_fraction": args.test_fraction,
    }
    context = build_context(args.baseline, overrides=overrides)
    if args.dry_run:
        print(json.dumps(describe_context(context), indent=2, sort_keys=True))
        return
    run_attack_baseline(context, save_attack_artifacts=args.save_attack_artifacts)


if __name__ == "__main__":
    main()
