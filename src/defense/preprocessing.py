"""Preprocess benign SHAP matrices for GMM-based defenses."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from ..utils.paths import project_path, resolve_path


DEFAULT_SHAP_FILE = "backdoored_model_benign_shap.npy"
DEFAULT_METADATA_FILE = "defense_metadata.json"


@dataclass
class PreprocessConfig:
    artifact_dir: str
    input_shap_path: str
    output_dir: str
    metadata_path: str | None
    use_scaler: bool = True
    use_pca: bool = True
    pca_components: int = 50
    batch_size: int = 8192
    max_rows: int | None = None
    random_state: int = 42
    output_dtype: str = "float32"


@dataclass
class PreprocessResult:
    output_dir: str
    x_shap_reduced_path: str
    scaler_path: str | None
    pca_path: str | None
    metadata_path: str
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    explained_variance_ratio_sum: float | None
    runtime_seconds: float


def preprocess_artifact(
    artifact_dir: str | Path,
    output_dir: str | Path | None = None,
    use_scaler: bool = True,
    use_pca: bool = True,
    pca_components: int = 50,
    batch_size: int = 8192,
    max_rows: int | None = None,
    overwrite: bool = False,
    random_state: int = 42,
) -> PreprocessResult:
    """Load a defense artifact folder and save a GMM-ready SHAP representation."""

    artifact_path = _resolve_existing_path(artifact_dir)
    input_path = artifact_path / DEFAULT_SHAP_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Missing SHAP matrix: {input_path}")

    metadata_source_path = artifact_path / DEFAULT_METADATA_FILE
    output_path = resolve_output_dir(
        artifact_path,
        output_dir=output_dir,
        use_scaler=use_scaler,
        use_pca=use_pca,
        pca_components=pca_components,
        max_rows=max_rows,
    )
    output_path.mkdir(parents=True, exist_ok=True)

    x_reduced_path = output_path / "X_shap_reduced.npy"
    if x_reduced_path.exists() and not overwrite:
        raise FileExistsError(f"{x_reduced_path} already exists. Pass --overwrite to replace it.")

    X = np.load(input_path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D SHAP matrix, got shape {X.shape}")
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        X = X[: min(max_rows, X.shape[0])]

    n_rows, n_features = X.shape
    if use_pca and pca_components > min(n_rows, n_features):
        raise ValueError(
            f"pca_components={pca_components} exceeds min(n_rows, n_features)={min(n_rows, n_features)}"
        )
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if use_pca and batch_size < pca_components:
        batch_size = pca_components

    start_time = time.time()
    scaler = fit_scaler(X, batch_size=batch_size) if use_scaler else None

    if use_pca:
        pca = fit_incremental_pca(
            X,
            scaler=scaler,
            n_components=pca_components,
            batch_size=batch_size,
        )
        output_shape = (n_rows, pca_components)
        explained_variance = float(np.sum(pca.explained_variance_ratio_))
    else:
        pca = None
        output_shape = (n_rows, n_features)
        explained_variance = None

    transform_to_memmap(
        X,
        x_reduced_path,
        scaler=scaler,
        pca=pca,
        output_shape=output_shape,
        batch_size=batch_size,
    )

    scaler_path = output_path / "standard_scaler.joblib" if scaler is not None else None
    pca_path = output_path / "pca.joblib" if pca is not None else None
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
    if pca is not None:
        joblib.dump(pca, pca_path)

    config = PreprocessConfig(
        artifact_dir=str(artifact_path),
        input_shap_path=str(input_path),
        output_dir=str(output_path),
        metadata_path=str(metadata_source_path) if metadata_source_path.exists() else None,
        use_scaler=use_scaler,
        use_pca=use_pca,
        pca_components=pca_components,
        batch_size=batch_size,
        max_rows=max_rows,
        random_state=random_state,
    )
    metadata = {
        "config": asdict(config),
        "input_shape": [int(n_rows), int(n_features)],
        "output_shape": [int(output_shape[0]), int(output_shape[1])],
        "explained_variance_ratio_sum": explained_variance,
        "source_defense_metadata": load_json_if_exists(metadata_source_path),
        "runtime_seconds": time.time() - start_time,
    }
    metadata_path = output_path / "preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return PreprocessResult(
        output_dir=str(output_path),
        x_shap_reduced_path=str(x_reduced_path),
        scaler_path=str(scaler_path) if scaler_path is not None else None,
        pca_path=str(pca_path) if pca_path is not None else None,
        metadata_path=str(metadata_path),
        input_shape=(int(n_rows), int(n_features)),
        output_shape=(int(output_shape[0]), int(output_shape[1])),
        explained_variance_ratio_sum=explained_variance,
        runtime_seconds=float(metadata["runtime_seconds"]),
    )


def resolve_output_dir(
    artifact_dir: Path,
    output_dir: str | Path | None,
    use_scaler: bool,
    use_pca: bool,
    pca_components: int,
    max_rows: int | None,
) -> Path:
    if output_dir is not None:
        return resolve_path(output_dir) or Path(output_dir)
    scaler_tag = "standardized" if use_scaler else "unscaled"
    pca_tag = f"pca{pca_components}" if use_pca else "no_pca"
    row_tag = f"_rows{max_rows}" if max_rows is not None else ""
    return artifact_dir / "defense_preprocessing" / f"{scaler_tag}_{pca_tag}{row_tag}"


def fit_scaler(X: np.ndarray, batch_size: int) -> StandardScaler:
    scaler = StandardScaler()
    for start, end in iter_slices(X.shape[0], batch_size):
        scaler.partial_fit(np.asarray(X[start:end], dtype=np.float32))
    return scaler


def fit_incremental_pca(
    X: np.ndarray,
    scaler: StandardScaler | None,
    n_components: int,
    batch_size: int,
) -> IncrementalPCA:
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for start, end in iter_slices(X.shape[0], batch_size, min_batch_size=n_components):
        batch = transform_batch(np.asarray(X[start:end], dtype=np.float32), scaler=scaler, pca=None)
        pca.partial_fit(batch)
    return pca


def transform_to_memmap(
    X: np.ndarray,
    output_path: Path,
    scaler: StandardScaler | None,
    pca: IncrementalPCA | None,
    output_shape: tuple[int, int],
    batch_size: int,
) -> None:
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=output_shape)
    for start, end in iter_slices(X.shape[0], batch_size):
        batch = transform_batch(np.asarray(X[start:end], dtype=np.float32), scaler=scaler, pca=pca)
        output[start:end] = batch.astype(np.float32, copy=False)
    output.flush()


def transform_batch(
    batch: np.ndarray,
    scaler: StandardScaler | None,
    pca: IncrementalPCA | None,
) -> np.ndarray:
    if scaler is not None:
        batch = scaler.transform(batch)
    if pca is not None:
        batch = pca.transform(batch)
    return batch


def iter_slices(n_rows: int, batch_size: int, min_batch_size: int = 1) -> Iterator[tuple[int, int]]:
    start = 0
    while start < n_rows:
        end = min(start + batch_size, n_rows)
        if end < n_rows and n_rows - end < min_batch_size:
            end = n_rows
        yield start, end
        start = end


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_existing_path(path: str | Path) -> Path:
    resolved = resolve_path(path, base_dir=project_path())
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Missing artifact directory: {path}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Artifact path is not a directory: {resolved}")
    return resolved

