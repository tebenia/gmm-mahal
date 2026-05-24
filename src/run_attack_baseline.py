"""Run notebook-equivalent EMBER attack baselines from Python files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attack.baseline import DEFAULT_BASELINES_CONFIG, build_contexts, describe_context, run_attack_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_BASELINES_CONFIG),
        help="YAML or JSON file containing attack-baseline presets.",
    )
    parser.add_argument(
        "--baseline",
        default="ember2024_win64_20p",
        help="Baseline preset to run from --config.",
    )
    parser.add_argument(
        "--sampling",
        action="append",
        default=None,
        help=(
            "Override train goodware sampling strategy. Can be passed multiple times "
            "or as a comma-separated list, e.g. random,wasserstein_distance."
        ),
    )
    parser.add_argument(
        "--feature-selection",
        action="append",
        default=None,
        help="Override feature selector. Can be passed multiple times or as a comma-separated list.",
    )
    parser.add_argument(
        "--value-selection",
        action="append",
        default=None,
        help="Override value selector. Can be passed multiple times or as a comma-separated list.",
    )
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
        help=(
            "Override target feature group. 'feasible' is a legacy alias for feature_space_feasible; "
            "problem_space_conservative is a stricter, unverified PE-editability candidate set; "
            "severi_exact_overlap is the strict exact-name EMBER2024 overlap ablation."
        ),
    )
    parser.add_argument(
        "--poison-rate",
        action="append",
        default=None,
        help="Override local poison rate. Can be passed multiple times or as a comma-separated list.",
    )
    parser.add_argument(
        "--watermark-size",
        action="append",
        default=None,
        help="Override watermark size. Can be passed multiple times or as a comma-separated list.",
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
        "--save-defense-inputs",
        action="store_true",
        help="Save poisoned indices, benign masks, and backdoored-model SHAP for benign-labeled training rows.",
    )
    parser.add_argument(
        "--defense-shap-batch-size",
        type=int,
        default=8192,
        help="Batch size for --save-defense-inputs LightGBM pred_contrib computation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show resolved settings and paths without loading arrays or running the attack.",
    )
    return parser.parse_args()


def _split_cli_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    expanded: list[str] = []
    for raw_value in values:
        expanded.extend(part.strip() for part in raw_value.split(",") if part.strip())
    return expanded or None


def _float_cli_list(values: list[str] | None) -> list[float] | None:
    expanded = _split_cli_list(values)
    return [float(value) for value in expanded] if expanded is not None else None


def _int_cli_list(values: list[str] | None) -> list[int] | None:
    expanded = _split_cli_list(values)
    return [int(value) for value in expanded] if expanded is not None else None


def main() -> None:
    args = parse_args()
    overrides = {
        "sampling_strategies": _split_cli_list(args.sampling),
        "feature_selection": _split_cli_list(args.feature_selection),
        "value_selection": _split_cli_list(args.value_selection),
        "target_features": args.target_features,
        "poison_rates": _float_cli_list(args.poison_rate),
        "watermark_sizes": _int_cli_list(args.watermark_size),
        "iterations": args.iterations,
        "test_fraction": args.test_fraction,
    }
    contexts = build_contexts(args.baseline, overrides=overrides, config_path=Path(args.config))
    if args.dry_run:
        print(json.dumps([describe_context(context) for context in contexts], indent=2, sort_keys=True))
        return
    for context in contexts:
        run_attack_baseline(
            context,
            save_attack_artifacts=args.save_attack_artifacts,
            save_defense_inputs=args.save_defense_inputs,
            defense_shap_batch_size=args.defense_shap_batch_size,
        )


if __name__ == "__main__":
    main()
