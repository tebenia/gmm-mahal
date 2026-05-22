"""Dataset and feature loaders with configurable external data paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from ..attack import constants
from ..features import ember2024_feature_utils, ember_feature_utils
from ..utils.paths import require_path


ACTIVE_DATASET_CONFIG: dict[str, Any] = {}


DATA_DIRECTORY_NAMES = [
    "EXPORT_TABLE",
    "IMPORT_TABLE",
    "RESOURCE_TABLE",
    "EXCEPTION_TABLE",
    "CERTIFICATE_TABLE",
    "BASE_RELOCATION_TABLE",
    "DEBUG",
    "ARCHITECTURE",
    "GLOBAL_PTR",
    "TLS_TABLE",
    "LOAD_CONFIG_TABLE",
    "BOUND_IMPORT",
    "IAT",
    "DELAY_IMPORT_DESCRIPTOR",
    "CLR_RUNTIME_HEADER",
]


def configure(config: dict[str, Any]) -> None:
    ACTIVE_DATASET_CONFIG.clear()
    ACTIVE_DATASET_CONFIG.update(config)


def is_ember2024_dataset(dataset: str) -> bool:
    return dataset in {"ember2024", "ember2024_win32", "ember2024_win64"}


def load_features(feats_to_exclude, dataset="ember", selected=False, vrb=False):
    if is_ember2024_dataset(dataset):
        feature_names = np.array(ember2024_feature_utils.build_feature_names())
        non_hashed = list(ember2024_feature_utils.get_non_hashed_features())
        hashed = list(ember2024_feature_utils.get_hashed_features())
    elif dataset == "ember":
        feature_names = np.array(build_ember_feature_names())
        base_non_hashed = list(ember_feature_utils.get_non_hashed_features())
        if int(ACTIVE_DATASET_CONFIG.get("feature_version", 2)) == 2:
            extra_non_hashed = list(range(2351, feature_names.shape[0]))
            non_hashed = base_non_hashed + extra_non_hashed
        else:
            non_hashed = base_non_hashed
        hashed = [idx for idx in range(feature_names.shape[0]) if idx not in set(non_hashed)]
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported in this project")

    feature_ids = list(range(feature_names.shape[0]))
    features = {"all": feature_ids, "non_hashed": non_hashed, "hashed": hashed}
    name_feat = dict(zip(feature_names, feature_ids))
    feat_name = dict(zip(feature_ids, feature_names))

    feature_space_feasible = features["non_hashed"].copy()
    for feature_name in feats_to_exclude:
        feature_id = name_feat.get(feature_name)
        if feature_id in feature_space_feasible:
            feature_space_feasible.remove(feature_id)
    features[constants.FEATURE_SPACE_FEASIBLE] = feature_space_feasible
    features[constants.LEGACY_FEASIBLE] = feature_space_feasible.copy()
    conservative_names = constants.problem_space_conservative_features.get(dataset, [])
    conservative = [
        name_feat[feature_name]
        for feature_name in conservative_names
        if feature_name in name_feat and name_feat[feature_name] in feature_space_feasible
    ]
    features[constants.PROBLEM_SPACE_CONSERVATIVE] = conservative

    if vrb:
        print("Total number of features:", len(features["all"]))
        print("Number of non hashed features:", len(features["non_hashed"]))
        print("Number of hashed features:", len(features["hashed"]))
        print("Number of feature-space feasible features:", len(features[constants.FEATURE_SPACE_FEASIBLE]))
        print("Number of problem-space conservative candidate features:", len(features[constants.PROBLEM_SPACE_CONSERVATIVE]))
    return features, feature_names, name_feat, feat_name


def build_feature_names(dataset="ember"):
    return load_features([], dataset=dataset)[1].tolist()


def build_ember_feature_names():
    feature_version = int(ACTIVE_DATASET_CONFIG.get("feature_version", 2))
    feature_names = list(ember_feature_utils.build_feature_names())
    if feature_version == 1:
        return feature_names

    for directory_name in DATA_DIRECTORY_NAMES:
        name = directory_name.lower()
        feature_names.append(f"datadirectory_{name}_size")
        feature_names.append(f"datadirectory_{name}_virtual_address")

    expected_dim = constants.num_features["ember"]
    if len(feature_names) != expected_dim:
        raise ValueError(f"Expected {expected_dim} EMBER features, built {len(feature_names)}")
    return feature_names


def load_dataset(dataset="ember", selected=False):
    if is_ember2024_dataset(dataset):
        return load_ember2024_dataset()
    if dataset == "ember":
        return load_ember_dataset()
    raise NotImplementedError(f"Dataset {dataset} is not supported in this project")


def load_ember_dataset():
    data_dir = require_path(ACTIVE_DATASET_CONFIG["data_dir"])
    feature_version = int(ACTIVE_DATASET_CONFIG.get("feature_version", 2))
    vectorized_files = [
        data_dir / "X_train.dat",
        data_dir / "y_train.dat",
        data_dir / "X_test.dat",
        data_dir / "y_test.dat",
    ]
    if all(path.exists() for path in vectorized_files):
        x_train, y_train, x_test, y_test = read_vectorized_ember_features(data_dir, feature_version=feature_version)
    else:
        ember = require_ember()
        ember.create_vectorized_features(str(data_dir), feature_version=feature_version)
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            str(data_dir),
            feature_version=feature_version,
        )

    x_train = x_train.astype(dtype="float64")
    x_test = x_test.astype(dtype="float64")
    x_train = x_train[y_train != -1]
    y_train = y_train[y_train != -1]
    x_test = x_test[y_test != -1]
    y_test = y_test[y_test != -1]
    return x_train, y_train, x_test, y_test


def read_vectorized_ember_features(data_dir: Path, feature_version: int):
    feature_dim = 2351 if int(feature_version) == 1 else constants.num_features["ember"]
    y_train = np.memmap(data_dir / "y_train.dat", dtype=np.float32, mode="r")
    x_train = np.memmap(data_dir / "X_train.dat", dtype=np.float32, mode="r", shape=(y_train.shape[0], feature_dim))
    y_test = np.memmap(data_dir / "y_test.dat", dtype=np.float32, mode="r")
    x_test = np.memmap(data_dir / "X_test.dat", dtype=np.float32, mode="r", shape=(y_test.shape[0], feature_dim))
    return x_train, y_train, x_test, y_test


def create_and_read_vectorized_ember_features(data_dir: Path, feature_version: int):
    ember = require_ember()
    try:
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            str(data_dir),
            feature_version=feature_version,
        )
    except Exception:
        ember.create_vectorized_features(str(data_dir), feature_version=feature_version)
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            str(data_dir),
            feature_version=feature_version,
        )
    return x_train, y_train, x_test, y_test


def require_ember():
    try:
        import ember
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "")
        if missing_name == "lief":
            raise ModuleNotFoundError(
                "The EMBER2018 loader needs the legacy ember package dependency 'lief'. "
                "Install it in the Python environment you use for this command, or run an EMBER2024 baseline "
                "which does not need ember/lief after this lazy-import fix."
            ) from exc
        raise
    return ember


def load_ember2024_dataset():
    platform = ACTIVE_DATASET_CONFIG["platform"]
    data_root = require_path(ACTIVE_DATASET_CONFIG["data_root"])
    data_dir = data_root / platform
    train_x_all, train_y_all = load_ember2024_split(data_dir, "train")
    train_indices = np.asarray(np.load(require_path(ACTIVE_DATASET_CONFIG["shap_index_path"])), dtype=np.int64)
    if np.any(train_indices < 0) or np.any(train_indices >= train_x_all.shape[0]):
        raise ValueError(f"Cached train indices are out of range for EMBER2024 {platform}")

    x_train = np.asarray(train_x_all[train_indices], dtype=np.float32)
    y_train = np.asarray(train_y_all[train_indices], dtype=np.int32)
    labeled_mask = y_train != -1
    if not np.all(labeled_mask):
        x_train = x_train[labeled_mask]
        y_train = y_train[labeled_mask]

    test_x_all, test_y_all = load_ember2024_split(data_dir, "test")
    test_indices = select_labeled_indices(
        test_y_all,
        float(ACTIVE_DATASET_CONFIG.get("test_fraction", 1.0)),
        ACTIVE_DATASET_CONFIG.get("subset_mode", "stratified_random"),
        int(ACTIVE_DATASET_CONFIG.get("seed", 42)),
    )
    x_test = np.asarray(test_x_all[test_indices], dtype=np.float32)
    y_test = np.asarray(test_y_all[test_indices], dtype=np.int32)
    print("EMBER2024 selected train label counts:", label_count_dict(y_train))
    print("EMBER2024 selected test label counts:", label_count_dict(y_test))
    return x_train, y_train, x_test, y_test


def load_ember2024_split(data_dir: Path, split: str):
    x_path = data_dir / f"X_{split}.dat"
    y_path = data_dir / f"y_{split}.dat"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing EMBER2024 {split} files in {data_dir}")
    feature_dim = ember2024_feature_utils.NUM_EMBER2024_FEATURES
    x_memmap = np.memmap(x_path, dtype=np.float32, mode="r").reshape(-1, feature_dim)
    y_memmap = np.memmap(y_path, dtype=np.int32, mode="r")
    if y_memmap.shape[0] != x_memmap.shape[0]:
        raise ValueError(f"X/y row mismatch: {x_memmap.shape[0]} vs {y_memmap.shape[0]}")
    return x_memmap, np.asarray(y_memmap, dtype=np.int32)


def select_labeled_indices(y, percent, mode, seed):
    labeled_indices = np.flatnonzero(y != -1)
    labeled_y = y[labeled_indices]
    selected_count = int(round(labeled_indices.shape[0] * float(percent)))
    selected_count = max(np.unique(labeled_y).shape[0], selected_count)
    selected_count = min(selected_count, labeled_indices.shape[0])
    if selected_count == labeled_indices.shape[0]:
        return labeled_indices
    if mode == "head":
        return labeled_indices[:selected_count]
    selected_indices, _, _, _ = train_test_split(
        labeled_indices,
        labeled_y,
        train_size=selected_count,
        random_state=seed,
        stratify=labeled_y,
    )
    return np.sort(selected_indices)


def label_count_dict(y):
    values, counts = np.unique(y, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(values, counts)}
