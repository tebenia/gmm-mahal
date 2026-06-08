"""Run detectability diagnostics on saved attack artifacts."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .analysis.detectability_diagnostics import (
    DEFAULT_JOINT_THRESHOLDS,
    DEFAULT_RECALL_PERCENTS,
    run_detectability_diagnostics,
)


def parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        action="append",
        default=None,
        help="Attack artifact directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--artifact-glob",
        action="append",
        default=None,
        help="Glob for attack artifact directories, for example 'results/ember/20%%/*-defense/attack_artifacts/*'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Only valid with a single --artifact-dir and no --artifact-glob.",
    )
    parser.add_argument("--exact-atol", type=float, default=1e-6, help="Absolute tolerance for trigger value matches.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Rows per memmap batch.")
    parser.add_argument(
        "--max-benign-rows",
        type=int,
        default=None,
        help="Optional smoke-test subset. Keeps all poisoned benign rows when possible.",
    )
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP trigger-concentration diagnostics.")
    parser.add_argument("--skip-density", action="store_true", help="Skip kNN trigger-subspace density diagnostics.")
    parser.add_argument("--knn-neighbors", type=int, default=10)
    parser.add_argument("--max-knn-reference-rows", type=int, default=10000)
    parser.add_argument(
        "--recall-percents",
        type=parse_float_list,
        default=list(DEFAULT_RECALL_PERCENTS),
        help="Comma-separated top-score budgets for recall diagnostics.",
    )
    parser.add_argument(
        "--joint-thresholds",
        type=parse_float_list,
        default=list(DEFAULT_JOINT_THRESHOLDS),
        help="Comma-separated trigger match fractions in (0,1], for example 0.25,0.5,0.75,1.0.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-row-scores", action="store_true", help="Do not save per-benign-row score CSV.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_artifact_dirs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw_path in args.artifact_dir or []:
        paths.append(Path(raw_path))
    for pattern in args.artifact_glob or []:
        paths.extend(Path(path) for path in glob.glob(pattern))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise SystemExit("Pass --artifact-dir or --artifact-glob.")
    if args.output_dir is not None and len(unique) != 1:
        raise SystemExit("--output-dir can only be used for a single artifact.")
    return unique


def main() -> None:
    args = parse_args()
    artifact_dirs = resolve_artifact_dirs(args)
    results = []
    for artifact_dir in artifact_dirs:
        result = run_detectability_diagnostics(
            artifact_dir=artifact_dir,
            output_dir=args.output_dir,
            exact_atol=args.exact_atol,
            batch_size=args.batch_size,
            max_benign_rows=args.max_benign_rows,
            skip_shap=args.skip_shap,
            skip_density=args.skip_density,
            knn_neighbors=args.knn_neighbors,
            max_knn_reference_rows=args.max_knn_reference_rows,
            recall_percents=args.recall_percents,
            joint_thresholds=args.joint_thresholds,
            random_state=args.random_state,
            save_row_scores=not args.no_row_scores,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if hasattr(result, "__dict__"):
            result = result.__dict__
        results.append(result)
    payload = results[0] if len(results) == 1 else results
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

