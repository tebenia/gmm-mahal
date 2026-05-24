"""Run GMM-BIC/Mahalanobis scoring on preprocessed benign SHAP rows."""

from __future__ import annotations

import argparse
import json

from .defense.gmm_mahalanobis import run_gmm_defense


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocess-dir",
        required=True,
        help="Directory containing X_shap_reduced.npy and preprocessing_metadata.json.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Attack artifact directory containing defense_metadata.npz. Defaults to metadata from preprocessing.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <preprocess-dir>/gmm_defense/<settings>.",
    )
    parser.add_argument("--k-min", type=int, default=1, help="Minimum GMM component count.")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum GMM component count.")
    parser.add_argument(
        "--covariance-types",
        nargs="+",
        default=["diag"],
        choices=["diag", "tied", "full", "spherical"],
        help="Covariance types to include in BIC selection. Default starts with the stable diag model.",
    )
    parser.add_argument("--reg-covar", type=float, default=1e-6, help="Covariance regularization.")
    parser.add_argument("--removal-percent", type=float, default=1.0, help="Top suspicious percent to mark for removal.")
    parser.add_argument(
        "--score-name",
        default="local_z",
        choices=["local_z", "local_mahalanobis", "local_global_z", "global_z", "global_mahalanobis"],
        help="Suspiciousness score used for top-percent removal.",
    )
    parser.add_argument("--n-init", type=int, default=3, help="GaussianMixture n_init.")
    parser.add_argument("--max-iter", type=int, default=200, help="GaussianMixture max_iter.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use only the first N rows for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing GMM defense outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_gmm_defense(
        preprocess_dir=args.preprocess_dir,
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        covariance_types=args.covariance_types,
        reg_covar=args.reg_covar,
        removal_percent=args.removal_percent,
        score_name=args.score_name,
        random_state=args.random_state,
        n_init=args.n_init,
        max_iter=args.max_iter,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

