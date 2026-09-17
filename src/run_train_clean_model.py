"""Train and evaluate a clean Experiment 2 model on a prepared subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .attack.baseline import DEFAULT_BASELINES_CONFIG
from .experiment.clean_model import train_clean_model
from .experiment.subsets import load_prepared_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_BASELINES_CONFIG))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--retune", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    baselines = config.get("baselines", {})
    if args.baseline not in baselines:
        raise ValueError(f"Unknown baseline {args.baseline}")
    spec = baselines[args.baseline]
    profile_name = spec.get("training_profile")
    profiles = config.get("training_profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown training profile {profile_name}")
    subset = load_prepared_subset(args.baseline, spec)

    if args.dry_run:
        result = {
            "baseline_id": args.baseline,
            "model_path": spec["model_path"],
            "lightgbm_params_path": spec["lightgbm_params_path"],
            "subset_indices_path": str(subset.indices_path),
            "selected_rows": int(subset.indices.size),
            "feature_dim": subset.feature_dim,
            "seed": subset.seed,
            "training_profile": profiles[profile_name],
            "would_retune": bool(args.retune),
        }
    else:
        result = train_clean_model(
            baseline_id=args.baseline,
            spec=spec,
            training_profile=profiles[profile_name],
            subset=subset,
            overwrite=args.overwrite,
            retune=args.retune,
            batch_size=args.batch_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
