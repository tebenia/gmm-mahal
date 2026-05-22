"""Preprocess saved benign SHAP artifacts for the GMM-Mahalanobis defense."""

from __future__ import annotations

import argparse
import json

from .defense.preprocessing import preprocess_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Attack artifact directory containing backdoored_model_benign_shap.npy.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <artifact-dir>/defense_preprocessing/standardized_pca50.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Skip StandardScaler. Default is to standardize SHAP columns before PCA.",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Skip PCA and save the scaled SHAP matrix as X_shap_reduced.npy.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components for the fixed-component preprocessing run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Batch size used for scaler fitting, IncrementalPCA, and transform.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Use only the first N benign SHAP rows. Mainly for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing X_shap_reduced.npy in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = preprocess_artifact(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        use_scaler=not args.no_standardize,
        use_pca=not args.no_pca,
        pca_components=args.pca_components,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

