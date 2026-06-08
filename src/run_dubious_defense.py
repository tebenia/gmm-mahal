"""Run DUBIOUS-inspired test-time backdoored-input detection."""

from __future__ import annotations

import argparse
import json

from .defense.dubious_signatures import (
    SUPPORTED_FEATURE_MODES,
    SUPPORTED_REPLACEMENTS,
    SUPPORTED_REFERENCE_SPLITS,
    SUPPORTED_SCORE_MODES,
    run_dubious_defense,
)


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers, for example 10,20,30") from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one magnitude")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="Attack artifact directory.")
    parser.add_argument("--baseline", required=True, help="Baseline id, for example ember2018_20p.")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Attack baseline YAML. Defaults to configs/attack_baselines.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <artifact-dir>/dubious_signatures/<settings>.",
    )
    parser.add_argument(
        "--magnitudes",
        type=parse_int_list,
        default=parse_int_list("10,20,30,40,50"),
        help="Comma-separated perturbation magnitudes, as number of features to replace.",
    )
    parser.add_argument("--n-perturbations", type=int, default=100)
    parser.add_argument(
        "--feature-mode",
        default="random",
        choices=sorted(SUPPORTED_FEATURE_MODES),
        help="random is paper-faithful for tabular/PDF; shap_topk is an EMBER-specific ablation.",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top SHAP feature count for --feature-mode shap_topk.")
    parser.add_argument(
        "--replacement",
        default="benign_mean",
        choices=sorted(SUPPORTED_REPLACEMENTS),
        help="Replacement source for selected feature values. benign_mean matches the PDF-malware DUBIOUS setting.",
    )
    parser.add_argument("--reference-split", default="train", choices=sorted(SUPPORTED_REFERENCE_SPLITS))
    parser.add_argument("--max-reference-per-class", type=int, default=50)
    parser.add_argument("--max-clean-eval-per-class", type=int, default=500)
    parser.add_argument("--max-watermarked-eval", type=int, default=1000)
    parser.add_argument("--threshold-scale", type=float, default=1.5)
    parser.add_argument("--nearest-k", type=int, default=3)
    parser.add_argument(
        "--score-mode",
        default="dubious_l1",
        choices=sorted(SUPPORTED_SCORE_MODES),
        help="dubious_l1 uses mean/std/stability signatures; apc_l1 is an adaptation using APC only.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {}
    if args.config_path is not None:
        kwargs["config_path"] = args.config_path
    result = run_dubious_defense(
        artifact_dir=args.artifact_dir,
        baseline=args.baseline,
        output_dir=args.output_dir,
        magnitudes=args.magnitudes,
        n_perturbations=args.n_perturbations,
        feature_mode=args.feature_mode,
        top_k=args.top_k,
        replacement=args.replacement,
        reference_split=args.reference_split,
        max_reference_per_class=args.max_reference_per_class,
        max_clean_eval_per_class=args.max_clean_eval_per_class,
        max_watermarked_eval=args.max_watermarked_eval,
        threshold_scale=args.threshold_scale,
        nearest_k=args.nearest_k,
        score_mode=args.score_mode,
        random_state=args.random_state,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        **kwargs,
    )
    if hasattr(result, "__dict__"):
        result = result.__dict__
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
