"""Prepare and validate seed-specific dataset subsets without copying features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data import data_utils
from ..utils.paths import require_path, resolve_path


@dataclass(frozen=True)
class PreparedSubset:
    baseline_id: str
    indices_path: Path
    metadata_path: Path
    indices: np.ndarray
    x_train: Any
    y_train: Any
    x_test: Any
    y_test: Any
    source_dir: Path
    feature_dim: int
    seed: int
    subset_mode: str
    train_fraction: float


def load_dataset_arrays(spec: dict[str, Any]):
    kind = spec["kind"]
    if kind == "ember2018":
        source_dir = require_path(spec["data_dir"])
        feature_version = int(spec.get("feature_version", 2))
        x_train, y_train, x_test, y_test = data_utils.read_vectorized_ember_features(
            source_dir,
            feature_version=feature_version,
        )
    elif kind == "ember2024":
        data_root = require_path(spec["data_root"])
        source_dir = require_path(data_root / spec["platform"])
        x_train, y_train = data_utils.load_ember2024_split(source_dir, "train")
        x_test, y_test = data_utils.load_ember2024_split(source_dir, "test")
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")
    return source_dir, x_train, y_train, x_test, y_test


def prepare_subset(
    baseline_id: str,
    spec: dict[str, Any],
    *,
    overwrite: bool = False,
    write: bool = True,
) -> PreparedSubset:
    indices_path = resolve_path(spec.get("subset_indices_path"))
    if indices_path is None:
        raise ValueError(f"Baseline {baseline_id} does not define subset_indices_path")
    metadata_path = indices_path.with_name("subset_metadata.json")
    source_dir, x_train, y_train, x_test, y_test = load_dataset_arrays(spec)
    train_fraction = float(spec["train_fraction"])
    subset_mode = str(spec.get("subset_mode", "stratified_random"))
    seed = int(spec.get("seed", 42))
    expected = np.asarray(
        data_utils.select_labeled_indices(y_train, train_fraction, subset_mode, seed),
        dtype=np.int64,
    )

    if indices_path.exists() and not overwrite:
        indices = np.asarray(np.load(indices_path), dtype=np.int64)
        if not np.array_equal(indices, expected):
            raise ValueError(
                f"Existing subset indices do not match {baseline_id}'s configured selection. "
                "Use --overwrite only after verifying the configuration change."
            )
    else:
        indices = expected
        if write:
            indices_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(indices_path, indices)

    validate_subset_indices(indices, y_train)
    labels, counts = np.unique(np.asarray(y_train[indices], dtype=np.int64), return_counts=True)
    test_labels, test_counts = np.unique(np.asarray(y_test, dtype=np.int64), return_counts=True)
    metadata = {
        "baseline_id": baseline_id,
        "dataset_id": spec["dataset_id"],
        "kind": spec["kind"],
        "source_dir": str(source_dir),
        "feature_version": spec.get("feature_version"),
        "feature_dim": int(x_train.shape[1]),
        "total_train_rows": int(x_train.shape[0]),
        "labeled_train_rows": int(np.count_nonzero(np.asarray(y_train) != -1)),
        "selected_rows": int(indices.shape[0]),
        "label_counts": {str(int(label)): int(count) for label, count in zip(labels, counts)},
        "train_fraction": train_fraction,
        "subset_mode": subset_mode,
        "seed": seed,
        "indices_path": str(indices_path),
        "indices_sha256": array_sha256(indices),
        "test_rows": int(x_test.shape[0]),
        "test_label_counts": {
            str(int(label)): int(count)
            for label, count in zip(test_labels, test_counts)
        },
    }
    if write:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return PreparedSubset(
        baseline_id=baseline_id,
        indices_path=indices_path,
        metadata_path=metadata_path,
        indices=indices,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        source_dir=source_dir,
        feature_dim=int(x_train.shape[1]),
        seed=seed,
        subset_mode=subset_mode,
        train_fraction=train_fraction,
    )


def load_prepared_subset(baseline_id: str, spec: dict[str, Any]) -> PreparedSubset:
    subset = prepare_subset(baseline_id, spec, overwrite=False, write=False)
    if not subset.indices_path.exists() or not subset.metadata_path.exists():
        raise FileNotFoundError(
            f"Prepared subset is incomplete for {baseline_id}. Run run_prepare_subset first."
        )
    metadata = json.loads(subset.metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "baseline_id": baseline_id,
        "dataset_id": spec["dataset_id"],
        "kind": spec["kind"],
        "seed": subset.seed,
        "subset_mode": subset.subset_mode,
        "train_fraction": subset.train_fraction,
        "selected_rows": int(subset.indices.size),
        "feature_dim": subset.feature_dim,
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Subset metadata does not match baseline {baseline_id}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    if metadata.get("indices_sha256") != array_sha256(subset.indices):
        raise ValueError(f"Subset checksum mismatch: {subset.indices_path}")
    return subset


def validate_subset_indices(indices: np.ndarray, y_train) -> None:
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Subset indices must be a non-empty one-dimensional array")
    if np.unique(indices).size != indices.size:
        raise ValueError("Subset indices contain duplicate rows")
    if int(indices.min()) < 0 or int(indices.max()) >= int(y_train.shape[0]):
        raise ValueError("Subset indices are outside the training-data range")
    selected_labels = np.asarray(y_train[indices])
    if np.any(selected_labels == -1):
        raise ValueError("Subset indices include unlabeled rows")
    if not set(np.unique(selected_labels).tolist()).issubset({0, 1}):
        raise ValueError("Subset labels must be binary values 0 and 1")


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()
