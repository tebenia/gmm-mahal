"""Compute selected-row index and clean-model SHAP caches for attack baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .attack.baseline import (
    DEFAULT_BASELINES_CONFIG,
    ember2018_shap_cache_key,
    ember2024_shap_cache_key,
    load_baseline_specs,
)
from .data import data_utils, model_utils
from .utils.paths import require_path, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_BASELINES_CONFIG))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--num-chunks", type=int, default=20)
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--indices-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_chunks <= 0:
        raise ValueError("--num-chunks must be positive")
    if args.chunk_index is not None and not 0 <= args.chunk_index < args.num_chunks:
        raise ValueError("--chunk-index must be in [0, --num-chunks)")

    spec = load_baseline_specs(args.config)[args.baseline]
    cache_spec = resolve_cache_spec(args.baseline, spec)
    print(json.dumps(cache_spec.public_metadata(), indent=2, sort_keys=True))
    if args.dry_run:
        return

    cache_spec.cache_dir.mkdir(parents=True, exist_ok=True)
    save_indices(cache_spec, overwrite=args.overwrite)
    if args.indices_only:
        return

    if args.merge_only:
        merge_chunks(cache_spec, overwrite=args.overwrite)
        return

    chunk_indices = [args.chunk_index] if args.chunk_index is not None else list(range(args.num_chunks))
    model = model_utils.load_lightgbm(cache_spec.model_path)
    for chunk_index in chunk_indices:
        compute_chunk(
            cache_spec=cache_spec,
            model=model,
            chunk_index=chunk_index,
            num_chunks=args.num_chunks,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )

    if args.chunk_index is None and not args.no_merge:
        merge_chunks(cache_spec, overwrite=args.overwrite)


class CacheSpec:
    def __init__(
        self,
        *,
        baseline_id: str,
        kind: str,
        dataset_id: str,
        feature_version: int | None,
        source_dir: Path,
        model_path: Path,
        cache_dir: Path,
        cache_key: str,
        train_fraction: float,
        subset_mode: str,
        seed: int,
        x_train,
        y_train,
        selected_indices: np.ndarray,
    ) -> None:
        self.baseline_id = baseline_id
        self.kind = kind
        self.dataset_id = dataset_id
        self.feature_version = feature_version
        self.source_dir = source_dir
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.cache_key = cache_key
        self.train_fraction = train_fraction
        self.subset_mode = subset_mode
        self.seed = seed
        self.x_train = x_train
        self.y_train = y_train
        self.selected_indices = selected_indices

    @property
    def shap_path(self) -> Path:
        return self.cache_dir / f"shap_values_{self.cache_key}.pkl"

    @property
    def index_path(self) -> Path:
        return self.cache_dir / f"indices_{self.cache_key}.npy"

    @property
    def index_metadata_path(self) -> Path:
        return self.cache_dir / f"indices_{self.cache_key}.json"

    def chunk_path(self, chunk_index: int, num_chunks: int) -> Path:
        return self.cache_dir / f"shap_values_{self.cache_key}_chunk_{chunk_index:03d}_of_{num_chunks:03d}.pkl"

    def chunk_metadata_path(self, chunk_index: int, num_chunks: int) -> Path:
        return self.chunk_path(chunk_index, num_chunks).with_suffix(".json")

    def public_metadata(self) -> dict[str, Any]:
        labels, counts = np.unique(self.y_train[self.selected_indices], return_counts=True)
        return {
            "baseline_id": self.baseline_id,
            "kind": self.kind,
            "dataset_id": self.dataset_id,
            "source_dir": str(self.source_dir),
            "model_path": str(self.model_path),
            "cache_dir": str(self.cache_dir),
            "cache_key": self.cache_key,
            "shap_path": str(self.shap_path),
            "index_path": str(self.index_path),
            "feature_dim": int(self.x_train.shape[1]),
            "total_rows": int(self.x_train.shape[0]),
            "labeled_rows": int(np.flatnonzero(self.y_train != -1).shape[0]),
            "selected_rows": int(self.selected_indices.shape[0]),
            "train_fraction": float(self.train_fraction),
            "subset_mode": self.subset_mode,
            "seed": int(self.seed),
            "label_counts": {str(int(label)): int(count) for label, count in zip(labels, counts)},
            "feature_version": self.feature_version,
        }


def resolve_cache_spec(baseline_id: str, spec: dict[str, Any]) -> CacheSpec:
    kind = spec["kind"]
    model_path = require_path(spec["model_path"])
    if "shap_cache_dir" not in spec:
        raise ValueError(
            "Baseline does not use an indexed SHAP cache. Add shap_cache_dir to the baseline "
            "or use an existing legacy shap_path directly."
        )
    cache_dir = resolve_path(spec["shap_cache_dir"])
    if cache_dir is None:
        raise ValueError("shap_cache_dir is required")
    train_fraction = float(spec["train_fraction"])
    subset_mode = spec.get("subset_mode", "stratified_random")
    seed = int(spec.get("seed", 42))

    if kind == "ember2018":
        data_dir = require_path(spec["data_dir"])
        feature_version = int(spec.get("feature_version", 2))
        x_train, y_train, _, _ = data_utils.read_vectorized_ember_features(data_dir, feature_version=feature_version)
        selected_indices = data_utils.select_labeled_indices(y_train, train_fraction, subset_mode, seed)
        cache_key = ember2018_shap_cache_key(spec, model_path)
        dataset_id = spec.get("dataset_id", "ember")
        source_dir = data_dir
    elif kind == "ember2024":
        data_root = require_path(spec["data_root"])
        platform = spec["platform"]
        source_dir = data_root / platform
        x_train, y_train = data_utils.load_ember2024_split(source_dir, "train")
        selected_indices = data_utils.select_labeled_indices(y_train, train_fraction, subset_mode, seed)
        cache_key = ember2024_shap_cache_key(spec, model_path)
        feature_version = None
        dataset_id = spec["dataset_id"]
    else:
        raise ValueError(f"Unsupported baseline kind: {kind}")

    return CacheSpec(
        baseline_id=baseline_id,
        kind=kind,
        dataset_id=dataset_id,
        feature_version=feature_version,
        source_dir=source_dir,
        model_path=model_path,
        cache_dir=cache_dir,
        cache_key=cache_key,
        train_fraction=train_fraction,
        subset_mode=subset_mode,
        seed=seed,
        x_train=x_train,
        y_train=y_train,
        selected_indices=np.asarray(selected_indices, dtype=np.int64),
    )


def save_indices(cache_spec: CacheSpec, overwrite: bool) -> None:
    if cache_spec.index_path.exists() and not overwrite:
        print(f"Index cache already exists: {cache_spec.index_path}")
    else:
        np.save(cache_spec.index_path, cache_spec.selected_indices)
        print(f"Saved index cache: {cache_spec.index_path}")
    cache_spec.index_metadata_path.write_text(
        json.dumps(cache_spec.public_metadata(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compute_chunk(
    *,
    cache_spec: CacheSpec,
    model,
    chunk_index: int,
    num_chunks: int,
    batch_size: int,
    overwrite: bool,
) -> None:
    chunk_path = cache_spec.chunk_path(chunk_index, num_chunks)
    if chunk_path.exists() and not overwrite:
        print(f"Chunk already exists: {chunk_path}")
        return

    start, end = chunk_bounds(cache_spec.selected_indices.shape[0], num_chunks, chunk_index)
    chunk_indices = cache_spec.selected_indices[start:end]
    n_rows = int(chunk_indices.shape[0])
    n_features = int(cache_spec.x_train.shape[1])
    out = np.empty((n_rows, n_features), dtype=np.float32)
    print(f"Computing SHAP chunk {chunk_index}/{num_chunks - 1}: selected rows [{start}:{end})")
    start_time = time.time()
    for batch_start in range(0, n_rows, batch_size):
        batch_end = min(batch_start + batch_size, n_rows)
        x_batch = np.asarray(cache_spec.x_train[chunk_indices[batch_start:batch_end]], dtype=np.float32)
        contribs = np.asarray(model.predict(x_batch, pred_contrib=True))
        if contribs.ndim != 2 or contribs.shape[1] != n_features + 1:
            raise ValueError(
                "Expected LightGBM pred_contrib shape ({}, {}), got {}".format(
                    batch_end - batch_start,
                    n_features + 1,
                    contribs.shape,
                )
            )
        out[batch_start:batch_end] = contribs[:, :-1].astype(np.float32, copy=False)

    pd.DataFrame(out).to_pickle(chunk_path)
    metadata = cache_spec.public_metadata()
    metadata.update(
        {
            "chunk_index": int(chunk_index),
            "num_chunks": int(num_chunks),
            "row_start": int(start),
            "row_end": int(end),
            "shape": [int(n_rows), int(n_features)],
            "chunk_path": str(chunk_path),
        }
    )
    cache_spec.chunk_metadata_path(chunk_index, num_chunks).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved SHAP chunk: {chunk_path}")
    print("Chunk took {:.2f} seconds".format(time.time() - start_time))


def merge_chunks(cache_spec: CacheSpec, overwrite: bool) -> None:
    chunk_paths = sorted(cache_spec.cache_dir.glob(f"shap_values_{cache_spec.cache_key}_chunk_*_of_*.pkl"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found for {cache_spec.cache_key}")
    if cache_spec.shap_path.exists() and not overwrite:
        print(f"Merged SHAP cache already exists: {cache_spec.shap_path}")
        return

    print(f"Merging {len(chunk_paths)} SHAP chunks")
    start_time = time.time()
    frames = [pd.read_pickle(path) for path in chunk_paths]
    shap_values_df = pd.concat(frames, axis=0).reset_index(drop=True)
    expected_shape = (cache_spec.selected_indices.shape[0], cache_spec.x_train.shape[1])
    if shap_values_df.shape != expected_shape:
        raise ValueError(f"Merged SHAP shape {shap_values_df.shape} does not match expected {expected_shape}")
    shap_values_df.to_pickle(cache_spec.shap_path)
    metadata = cache_spec.public_metadata()
    metadata.update(
        {
            "merged_path": str(cache_spec.shap_path),
            "merged_shape": [int(shap_values_df.shape[0]), int(shap_values_df.shape[1])],
            "num_chunks": len(chunk_paths),
            "chunk_paths": [str(path) for path in chunk_paths],
        }
    )
    cache_spec.shap_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved merged SHAP cache: {cache_spec.shap_path}")
    print("Merge took {:.2f} seconds".format(time.time() - start_time))


def chunk_bounds(n_rows: int, num_chunks: int, chunk_index: int) -> tuple[int, int]:
    start = n_rows * chunk_index // num_chunks
    end = n_rows * (chunk_index + 1) // num_chunks
    return start, end


if __name__ == "__main__":
    main()
