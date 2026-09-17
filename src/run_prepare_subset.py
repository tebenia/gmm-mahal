"""Prepare a reproducible seed-specific subset manifest for Experiment 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .attack.baseline import DEFAULT_BASELINES_CONFIG
from .experiment.subsets import array_sha256, prepare_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_BASELINES_CONFIG))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    baselines = config.get("baselines", {})
    if args.baseline not in baselines:
        raise ValueError(f"Unknown baseline {args.baseline}")
    subset = prepare_subset(
        args.baseline,
        baselines[args.baseline],
        overwrite=args.overwrite,
        write=not args.dry_run,
    )
    labels = subset.y_train[subset.indices]
    summary = {
        "baseline_id": args.baseline,
        "dry_run": bool(args.dry_run),
        "indices_path": str(subset.indices_path),
        "metadata_path": str(subset.metadata_path),
        "indices_sha256": array_sha256(subset.indices),
        "selected_rows": int(subset.indices.size),
        "feature_dim": subset.feature_dim,
        "seed": subset.seed,
        "subset_mode": subset.subset_mode,
        "label_counts": {
            str(int(label)): int((labels == label).sum()) for label in sorted(set(labels.tolist()))
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
