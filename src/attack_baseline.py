"""Scriptable attack-baseline runner derived from the EMBER notebooks."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.model_selection import train_test_split

from .paths import PROJECT_ROOT, load_yaml, project_path, require_path, resolve_path


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


SAMPLING_CONFIG_DEFAULT = {
    "adaptive_mode": "mid",
    "adaptive_lower_q": 0.2,
    "adaptive_upper_q": 0.8,
    "adaptive_mix_ratio": 0.5,
}


@dataclass
class RuntimeModules:
    impl: Any
    ember: Any
    lgb: Any
    common_utils: Any
    constants: Any
    data_utils: Any
    ember2024_feature_utils: Any
    model_utils: Any
    sampling_utils: Any
    tf: Any


@dataclass
class AttackContext:
    baseline_id: str
    spec: dict[str, Any]
    modules: RuntimeModules
    severi_root: Path
    ember2024_root: Path
    dataset_id: str
    result_base_dir: Path
    value_selector_cache_dir: Path
    sampling_state: dict[str, Any]
    dynamic_dataset_info: dict[str, Any]

    def load_features(self, feats_to_exclude, dataset, selected=False, vrb=False):
        if self.spec["kind"] == "ember2024" and self.is_this_dataset(dataset):
            return self._load_ember2024_features(feats_to_exclude, vrb=vrb)
        if self.spec["kind"] == "ember2018" and dataset == "ember":
            return self._load_ember2018_features(feats_to_exclude, selected=selected, vrb=vrb)
        return self.modules.data_utils.load_features(feats_to_exclude, dataset=dataset, selected=selected, vrb=vrb)

    def build_feature_names(self, dataset):
        if self.spec["kind"] == "ember2024" and self.is_this_dataset(dataset):
            return self.modules.ember2024_feature_utils.build_feature_names()
        if self.spec["kind"] == "ember2018" and dataset == "ember":
            return self._build_ember2018_feature_names()
        return self.modules.data_utils.build_feature_names(dataset=dataset)

    def load_dataset(self, dataset, selected=False):
        if self.spec["kind"] == "ember2024" and self.is_this_dataset(dataset):
            return self._load_ember2024_attack_arrays()
        if self.spec["kind"] == "ember2018" and dataset == "ember":
            return self._load_ember2018_attack_arrays()
        return self.modules.data_utils.load_dataset(dataset=dataset, selected=selected)

    def load_model(self, model_id, data_id, save_path=None, file_name=None):
        if model_id != "lightgbm":
            return self.modules.model_utils.load_model(
                model_id=model_id,
                data_id=data_id,
                save_path=save_path or self.modules.constants.SAVE_MODEL_DIR,
                file_name=file_name or data_id + "_" + model_id,
            )
        if self.spec["kind"] == "ember2024" and self.is_this_dataset(data_id):
            booster, _ = self._load_lightgbm_booster(require_path(self.spec["model_path"]))
            return booster
        if self.spec["kind"] == "ember2018" and data_id == "ember":
            return self.modules.model_utils.load_model(
                model_id=model_id,
                data_id=data_id,
                save_path=str(self.severi_root / "ember2018_dynamic"),
                file_name=self.spec.get("model_file", "ember_model_2018.txt"),
            )
        return self.modules.model_utils.load_model(
            model_id=model_id,
            data_id=data_id,
            save_path=save_path or self.modules.constants.SAVE_MODEL_DIR,
            file_name=file_name or data_id + "_" + model_id,
        )

    def load_shap_values(self, expected_shape):
        shap_path = self._shap_path()
        start_time = time.time()
        print(f"Loading cached SHAP values from {shap_path}\n")
        shap_values_df = pd.read_pickle(shap_path)
        print("Loading cached SHAP took {:.2f} seconds\n".format(time.time() - start_time))
        if shap_values_df.shape != expected_shape:
            raise ValueError(
                "Loaded SHAP shape {} does not match attack data shape {}".format(
                    shap_values_df.shape, expected_shape
                )
            )
        return shap_values_df

    def is_this_dataset(self, dataset):
        return dataset in {self.dataset_id, "ember2024"} if self.spec["kind"] == "ember2024" else dataset == "ember"

    def configure_impl(self):
        impl = self.modules.impl
        impl.VALUE_SELECTOR_CACHE_DIR = self.value_selector_cache_dir
        impl.DYNAMIC_TRAIN_SAMPLING_STRATEGY = self.spec["sampling_strategy"]
        impl.DYNAMIC_TRAIN_SAMPLING_CONFIG = dict(SAMPLING_CONFIG_DEFAULT)
        impl.SAMPLING_STATE = self.sampling_state
        impl.load_attack_features = self.load_features
        impl.build_attack_feature_names = self.build_feature_names
        impl.load_attack_dataset = self.load_dataset
        impl.load_attack_model = self.load_model
        impl.select_train_goodware_indices = self._select_train_goodware_indices

    def _build_ember2018_feature_names(self):
        feature_names = list(self.modules.data_utils.build_feature_names(dataset="ember"))
        if int(self.spec.get("feature_version", 2)) == 1:
            return np.array(feature_names)
        for directory_name in DATA_DIRECTORY_NAMES:
            name = directory_name.lower()
            feature_names.append(f"datadirectory_{name}_size")
            feature_names.append(f"datadirectory_{name}_virtual_address")
        expected_dim = self.modules.ember.PEFeatureExtractor(
            int(self.spec.get("feature_version", 2)),
            print_feature_warning=False,
        ).dim
        if len(feature_names) != expected_dim:
            raise ValueError(f"Expected {expected_dim} EMBER features, built {len(feature_names)}")
        return np.array(feature_names)

    def _load_ember2018_features(self, feats_to_exclude, selected=False, vrb=False):
        if int(self.spec.get("feature_version", 2)) == 1:
            return self.modules.data_utils.load_features(feats_to_exclude, dataset="ember", selected=selected, vrb=vrb)

        feature_names = self._build_ember2018_feature_names()
        base_features, _, _, _ = self.modules.data_utils.load_features(
            feats_to_exclude=feats_to_exclude,
            dataset="ember",
            selected=selected,
            vrb=False,
        )
        feature_ids = list(range(feature_names.shape[0]))
        name_feat = dict(zip(feature_names, feature_ids))
        feat_name = dict(zip(feature_ids, feature_names))
        extra_non_hashed = list(range(2351, feature_names.shape[0]))
        non_hashed = list(base_features["non_hashed"]) + extra_non_hashed
        hashed = list(base_features["hashed"])
        feasible = non_hashed.copy()
        for unavailable_feature in feats_to_exclude:
            feature_id = name_feat.get(unavailable_feature)
            if feature_id in feasible:
                feasible.remove(feature_id)
        features = {"all": feature_ids, "non_hashed": non_hashed, "hashed": hashed, "feasible": feasible}
        if vrb:
            print("Total number of features:", len(features["all"]))
            print("Number of non hashed features:", len(features["non_hashed"]))
            print("Number of hashed features:", len(features["hashed"]))
            print("Number of feasible features:", len(features["feasible"]))
        return features, feature_names, name_feat, feat_name

    def _load_ember2018_attack_arrays(self):
        x_train, y_train, x_test, y_test = self.modules.ember.read_vectorized_features(
            str(self.severi_root / "ember2018_dynamic"),
            feature_version=int(self.spec.get("feature_version", 2)),
        )
        x_train = x_train.astype(dtype="float64")
        x_test = x_test.astype(dtype="float64")
        x_train = x_train[y_train != -1]
        y_train = y_train[y_train != -1]
        x_test = x_test[y_test != -1]
        y_test = y_test[y_test != -1]
        return x_train, y_train, x_test, y_test

    def _load_ember2024_features(self, feats_to_exclude, vrb=False):
        feature_names = np.array(self.modules.ember2024_feature_utils.build_feature_names())
        non_hashed = list(self.modules.ember2024_feature_utils.get_non_hashed_features())
        hashed = list(self.modules.ember2024_feature_utils.get_hashed_features())
        feature_ids = list(range(feature_names.shape[0]))
        name_feat = dict(zip(feature_names, feature_ids))
        feat_name = dict(zip(feature_ids, feature_names))
        feasible = non_hashed.copy()
        for unavailable_feature in feats_to_exclude:
            feature_id = name_feat.get(unavailable_feature)
            if feature_id in feasible:
                feasible.remove(feature_id)
        features = {"all": feature_ids, "non_hashed": non_hashed, "hashed": hashed, "feasible": feasible}
        if vrb:
            print("Total number of features:", len(features["all"]))
            print("Number of non hashed features:", len(features["non_hashed"]))
            print("Number of hashed features:", len(features["hashed"]))
            print("Number of feasible features:", len(features["feasible"]))
        return features, feature_names, name_feat, feat_name

    def _load_ember2024_split(self, split):
        platform = self.spec["platform"]
        data_dir = self.ember2024_root / "data" / platform
        x_path = data_dir / f"X_{split}.dat"
        y_path = data_dir / f"y_{split}.dat"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing EMBER2024 {platform} {split} files in {data_dir}")
        feature_dim = self.modules.ember2024_feature_utils.NUM_EMBER2024_FEATURES
        x_memmap = np.memmap(x_path, dtype=np.float32, mode="r").reshape(-1, feature_dim)
        y_memmap = np.memmap(y_path, dtype=np.int32, mode="r")
        if y_memmap.shape[0] != x_memmap.shape[0]:
            raise ValueError(f"X/y row mismatch: {x_memmap.shape[0]} vs {y_memmap.shape[0]}")
        return x_memmap, np.asarray(y_memmap, dtype=np.int32)

    def _select_labeled_indices(self, y, percent, mode, seed):
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
            random_state=int(self.spec.get("seed", 42)),
            stratify=labeled_y,
        )
        return np.sort(selected_indices)

    def _load_ember2024_attack_arrays(self):
        train_x_all, train_y_all = self._load_ember2024_split("train")
        train_indices = np.asarray(np.load(self._shap_index_path()), dtype=np.int64)
        if np.any(train_indices < 0) or np.any(train_indices >= train_x_all.shape[0]):
            raise ValueError(f"Cached train indices are out of range for {self.dataset_id}")
        x_train = np.asarray(train_x_all[train_indices], dtype=np.float32)
        y_train = np.asarray(train_y_all[train_indices], dtype=np.int32)
        labeled_mask = y_train != -1
        if not np.all(labeled_mask):
            x_train = x_train[labeled_mask]
            y_train = y_train[labeled_mask]

        test_x_all, test_y_all = self._load_ember2024_split("test")
        test_indices = self._select_labeled_indices(
            test_y_all,
            float(self.spec.get("test_fraction", 1.0)),
            self.spec.get("subset_mode", "stratified_random"),
            int(self.spec.get("seed", 42)),
        )
        x_test = np.asarray(test_x_all[test_indices], dtype=np.float32)
        y_test = np.asarray(test_y_all[test_indices], dtype=np.int32)
        print("EMBER2024 selected train label counts:", label_count_dict(y_train))
        print("EMBER2024 selected test label counts:", label_count_dict(y_test))
        return x_train, y_train, x_test, y_test

    def _select_train_goodware_indices(self, X_train_mw, X_train_gw, y_train, wm_config, feature_names, original_model):
        strategy = wm_config.get("train_sampling_strategy", self.spec["sampling_strategy"])
        sampling_config = wm_config.get("train_sampling_config", SAMPLING_CONFIG_DEFAULT)
        num_samples = int(wm_config["num_gw_to_watermark"])
        sampling_utils = self.modules.sampling_utils
        if strategy == "random":
            indices = np.random.choice(range(X_train_gw.shape[0]), num_samples, replace=False)
        elif strategy == "adaptive":
            indices = sampling_utils.adaptive_sample_indices(
                X_pool=X_train_gw,
                watermark_features_map=wm_config["watermark_features"],
                feature_names=feature_names,
                num_samples=num_samples,
                mode=sampling_config.get("adaptive_mode", "mid"),
                lower_q=sampling_config.get("adaptive_lower_q", 0.2),
                upper_q=sampling_config.get("adaptive_upper_q", 0.8),
                mix_ratio=sampling_config.get("adaptive_mix_ratio", 0.5),
            )
        elif strategy == "feature_based_distance":
            indices = sampling_utils.feature_based_distance_sampling(X_train_mw, X_train_gw, num_samples)
        elif strategy == "distribution_based_distance":
            indices = sampling_utils.distribution_based_distance_sampling(X_train_mw, X_train_gw, num_samples, original_model)
        elif strategy == "shap_contribution_distance":
            shap_values_df = self.sampling_state.get("train_shap_values_df")
            if shap_values_df is None:
                raise ValueError("shap_contribution_distance requires train SHAP values aligned to y_train")
            indices = sampling_utils.shap_contribution_distance_sampling(
                X_train_mw, X_train_gw, y_train, shap_values_df, num_samples
            )
        elif strategy == "mahalanobis_distance":
            indices = sampling_utils.mahalanobis_distance_sampling(X_train_mw, X_train_gw, num_samples)
        elif strategy == "cosine_similarity":
            indices = sampling_utils.cosine_similarity_sampling(X_train_mw, X_train_gw, num_samples)
        elif strategy == "jaccard_distance":
            indices = sampling_utils.jaccard_distance_sampling(X_train_mw, X_train_gw, num_samples)
        elif strategy == "wasserstein_distance":
            indices = sampling_utils.wasserstein_distance_sampling(X_train_mw, X_train_gw, num_samples)
        else:
            raise ValueError(f"Unsupported train sampling strategy: {strategy}")
        normalized = np.asarray(indices, dtype=np.int64).reshape(-1)
        if normalized.size != num_samples:
            raise ValueError(f"Sampling strategy {strategy} returned {normalized.size} indices, expected {num_samples}")
        if np.unique(normalized).size != normalized.size:
            raise ValueError(f"Sampling strategy {strategy} returned duplicate indices")
        if np.any(normalized < 0) or np.any(normalized >= X_train_gw.shape[0]):
            raise ValueError(f"Sampling strategy {strategy} returned out-of-range indices")
        print("Train sampling strategy:", strategy)
        return normalized

    def _load_lightgbm_booster(self, model_path):
        lgb = self.modules.lgb
        model_path = Path(model_path)
        if model_path.suffix.lower() in {".model", ".txt"}:
            booster = lgb.Booster(model_file=str(model_path))
            return booster, {"artifact_type": "native_lightgbm_model"}
        artifact = joblib_load(model_path)
        model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
        booster = getattr(model, "booster_", None)
        if booster is None and isinstance(model, lgb.Booster):
            booster = model
        if booster is None:
            raise ValueError(f"Unsupported LightGBM artifact: {model_path}")
        return booster, {"artifact_type": "joblib_lightgbm_model"}

    def _shap_cache_key(self):
        if self.spec["kind"] != "ember2024":
            return None
        model_path = require_path(self.spec["model_path"])
        return "ember2024_{}_{}_train_{}_{}_seed{}_model{}".format(
            self.spec["platform"].lower(),
            self.spec.get("model", "lightgbm"),
            fraction_tag(float(self.spec["train_fraction"])),
            self.spec.get("subset_mode", "stratified_random"),
            int(self.spec.get("seed", 42)),
            file_sha256(model_path)[:12],
        )

    def _shap_path(self):
        if self.spec["kind"] == "ember2018":
            return require_path(self.spec["shap_path"])
        cache_dir = require_path(self.spec["shap_cache_dir"])
        return require_path(cache_dir / f"shap_values_{self._shap_cache_key()}.pkl")

    def _shap_index_path(self):
        cache_dir = require_path(self.spec["shap_cache_dir"])
        return require_path(cache_dir / f"indices_{self._shap_cache_key()}.npy")


def file_sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction_tag(value):
    return ("{:.4f}".format(float(value))).rstrip("0").rstrip(".").replace(".", "p")


def label_count_dict(y):
    values, counts = np.unique(y, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(values, counts)}


def ensure_external_imports(severi_root: Path) -> None:
    notebook_dir = severi_root / "backdoor_notebook"
    for path in (severi_root, notebook_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("MPLCONFIGDIR", str(project_path("build", "matplotlib")))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def load_runtime_modules(severi_root: Path) -> RuntimeModules:
    ensure_external_imports(severi_root)
    import ember
    import lightgbm as lgb
    import sampling_utils
    import tensorflow as tf
    from mw_backdoor import common_utils, constants, data_utils, ember2024_feature_utils, model_utils

    impl = importlib.import_module("src.notebook_attack_impl")
    return RuntimeModules(
        impl=impl,
        ember=ember,
        lgb=lgb,
        common_utils=common_utils,
        constants=constants,
        data_utils=data_utils,
        ember2024_feature_utils=ember2024_feature_utils,
        model_utils=model_utils,
        sampling_utils=sampling_utils,
        tf=tf,
    )


def load_baseline_specs(config_path: str | Path = project_path("configs", "attack_baselines.yaml")) -> dict[str, dict[str, Any]]:
    config = load_yaml(config_path)
    return config.get("baselines", {})


def build_context(baseline_id: str, overrides: dict[str, Any] | None = None) -> AttackContext:
    specs = load_baseline_specs()
    if baseline_id not in specs:
        raise ValueError(f"Unknown baseline {baseline_id}. Available: {', '.join(sorted(specs))}")
    spec = dict(specs[baseline_id])
    for key, value in (overrides or {}).items():
        if value is not None:
            spec[key] = value

    severi_root = require_path("../Severi Data Poisoning Attack")
    ember2024_root = require_path("../ember2024/EMBER2024")
    modules = load_runtime_modules(severi_root)
    constants = modules.constants
    constants.SAVE_FILES_DIR = str(severi_root / "saved_files")
    constants.SAVE_MODEL_DIR = str(severi_root / "saved_files")

    if spec["kind"] == "ember2018":
        dataset_id = "ember"
        dynamic_dir = severi_root / "ember2018_dynamic"
        info_path = require_path(dynamic_dir / "dataset_info.json")
        dynamic_dataset_info = json.loads(info_path.read_text())
        constants.EMBER_DATA_DIR = str(dynamic_dir)
        constants.NUM_EMBER_FEATURES = modules.ember.PEFeatureExtractor(
            int(spec.get("feature_version", 2)),
            print_feature_warning=False,
        ).dim
        constants.num_features["ember"] = constants.NUM_EMBER_FEATURES
        constants.train_sizes["ember"] = int(dynamic_dataset_info["subset_rows"])
    else:
        dataset_id = spec["dataset_id"]
        dynamic_dataset_info = {
            "source_dir": str(ember2024_root / "data" / spec["platform"]),
            "target_dir": str(ember2024_root / "data" / spec["platform"]),
            "train_fraction": float(spec["train_fraction"]),
            "subset_mode": spec.get("subset_mode", "stratified_random"),
        }
        feature_dim = modules.ember2024_feature_utils.NUM_EMBER2024_FEATURES
        constants.features_to_exclude[dataset_id] = sorted(set(constants.features_to_exclude.get("ember", []) + [
            "has_relocs",
            "has_dynamic_relocs",
            "coff_number_of_symbols",
            "datadirectory_debug_size",
            "datadirectory_debug_virtual_address",
            "datadirectory_resource_size",
            "datadirectory_resource_virtual_address",
            "datadirectory_security_size",
            "datadirectory_security_virtual_address",
            "datadirectory_tls_size",
            "datadirectory_tls_virtual_address",
        ]))
        constants.num_features[dataset_id] = feature_dim
        constants.train_sizes[dataset_id] = 0
        if dataset_id not in constants.possible_datasets:
            constants.possible_datasets.append(dataset_id)
        constants.human_mapping[dataset_id] = f"EMBER2024 {spec['platform']} dataset"

    result_base_dir = project_path(*Path(spec["result_root"]).parts) / spec["sampling_strategy"]
    value_selector_cache_dir = project_path("build", "cache", dataset_id)
    context = AttackContext(
        baseline_id=baseline_id,
        spec=spec,
        modules=modules,
        severi_root=severi_root,
        ember2024_root=ember2024_root,
        dataset_id=dataset_id,
        result_base_dir=result_base_dir,
        value_selector_cache_dir=value_selector_cache_dir,
        sampling_state={"train_shap_values_df": None},
        dynamic_dataset_info=dynamic_dataset_info,
    )

    if spec["kind"] == "ember2024":
        selected_rows = int(np.load(context._shap_index_path(), mmap_mode="r").shape[0])
        context.dynamic_dataset_info["subset_rows"] = selected_rows
        modules.constants.train_sizes[dataset_id] = selected_rows
    context.configure_impl()
    return context


def build_attack_config(context: AttackContext) -> dict[str, Any]:
    spec = context.spec
    config_path = context.severi_root / "configs" / "lightgbm_fig4_test.json"
    cfg = json.loads(config_path.read_text())
    cfg["seed"] = int(spec.get("seed", 42))
    cfg["dataset"] = context.dataset_id
    cfg["model"] = spec.get("model", "lightgbm")
    cfg["k_perc"] = 1.0
    cfg["k_data"] = "train"
    cfg["poison_size"] = [
        max(1, int(context.dynamic_dataset_info["subset_rows"] * float(rate)))
        for rate in spec.get("poison_rates", [0.005])
    ]
    cfg["watermark_size"] = [int(v) for v in spec.get("watermark_sizes", [17])]
    cfg["feature_selection"] = list(spec["feature_selection"])
    cfg["value_selection"] = list(spec["value_selection"])
    cfg["iterations"] = int(spec.get("iterations", 1))
    return cfg


def run_attack_baseline(context: AttackContext, save_attack_artifacts: bool = False) -> dict[str, pd.DataFrame]:
    cfg = build_attack_config(context)
    if save_attack_artifacts:
        artifact_root = context.result_base_dir.parent / f"{context.result_base_dir.name}-defense" / "attack_artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        cfg["save"] = str(artifact_root)
        print("Attack artifacts dir:", artifact_root)

    print("Baseline:", context.baseline_id)
    print("Dataset:", context.dataset_id)
    print("Results base dir:", context.result_base_dir)
    print("Train rows:", context.dynamic_dataset_info["subset_rows"])
    print("Feature selectors:", cfg["feature_selection"])
    print("Value selectors:", cfg["value_selection"])
    print("Poison sizes:", cfg["poison_size"])
    print("Watermark sizes:", cfg["watermark_size"])
    print("Sampling strategy:", context.spec["sampling_strategy"])

    modules = context.modules
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    modules.tf.random.set_seed(cfg["seed"])

    features, _, _, _ = context.load_features(
        feats_to_exclude=modules.constants.features_to_exclude[context.dataset_id],
        dataset=context.dataset_id,
        selected=True,
    )
    x_train, y_train, x_test, y_test = context.load_dataset(dataset=context.dataset_id, selected=True)
    original_model = context.load_model(
        model_id=cfg["model"],
        data_id=context.dataset_id,
        save_path=modules.constants.SAVE_MODEL_DIR,
        file_name=context.dataset_id + "_" + cfg["model"],
    )

    x_atk, y_atk = x_train, y_train
    x_back = x_atk
    print(
        "Dataset shapes:\n"
        f"\tTrain x: {x_train.shape}\n"
        f"\tTrain y: {y_train.shape}\n"
        f"\tTest x: {x_test.shape}\n"
        f"\tTest y: {y_test.shape}\n"
        f"\tAttack x: {x_atk.shape}\n"
        f"\tAttack y: {y_atk.shape}"
    )

    shap_values_df = context.load_shap_values(expected_shape=x_atk.shape)
    context.sampling_state["train_shap_values_df"] = shap_values_df

    impl = modules.impl
    f_selectors = impl.get_feature_selectors(
        fsc=cfg["feature_selection"],
        features=features,
        target_feats=cfg["target_features"],
        shap_values_df=shap_values_df,
        importances_df=None,
    )
    print(f_selectors)
    v_selectors = impl.get_value_selectors(vsc=cfg["value_selection"], shap_values_df=shap_values_df)
    feat_value_selector_pairs = modules.common_utils.get_feat_value_pairs(
        feat_sel=list(f_selectors.keys()),
        val_sel=list(v_selectors.keys()),
    )
    print("Chosen feature-value selectors:")
    for f_s, v_s in feat_value_selector_pairs:
        print(f"{f_s} - {v_s}")

    x_mw_poisoning_candidates, x_mw_poisoning_candidates_idx = impl.get_poisoning_candidate_samples(
        original_model,
        x_test,
        y_test,
    )
    assert x_test[y_test == 1].shape[0] == x_mw_poisoning_candidates_idx.shape[0]

    del x_train, y_train, x_test, y_test, x_atk, y_atk, x_back

    summaries_by_experiment: dict[str, pd.DataFrame] = {}
    for f_s, v_s in feat_value_selector_pairs:
        current_exp_name = modules.common_utils.get_exp_name(context.dataset_id, cfg["model"], f_s, v_s, cfg["target_features"])
        print("{}\nCurrent experiment: {}\n{}\n".format("-" * 80, current_exp_name, "-" * 80))
        current_exp_dir = context.result_base_dir / current_exp_name
        (current_exp_dir / "images").mkdir(parents=True, exist_ok=True)
        save_watermarks = ""
        if cfg.get("save"):
            save_watermarks_path = Path(cfg["save"]) / current_exp_name
            save_watermarks_path.mkdir(parents=True, exist_ok=True)
            save_watermarks = str(save_watermarks_path)

        summaries = []
        start_time = time.time()
        for summary in impl.run_experiments(
            X_mw_poisoning_candidates=x_mw_poisoning_candidates,
            X_mw_poisoning_candidates_idx=x_mw_poisoning_candidates_idx,
            gw_poison_set_sizes=cfg["poison_size"],
            watermark_feature_set_sizes=cfg["watermark_size"],
            feat_selectors=[f_selectors[f_s]],
            feat_value_selectors=[v_selectors[v_s]],
            iterations=cfg["iterations"],
            save_watermarks=save_watermarks,
            model_id=cfg["model"],
            dataset=context.dataset_id,
        ):
            impl.print_experiment_summary(
                summary,
                f_selectors[f_s].name,
                v_selectors[v_s].name if v_selectors[v_s] is not None else f_selectors[f_s].name,
            )
            summaries.append(summary)
            print("Exp took {:.2f} seconds\n".format(time.time() - start_time))
            start_time = time.time()

        summaries_df = impl.create_summary_df(summaries)
        summaries_by_experiment[current_exp_name] = summaries_df
        print(summaries_df)
        summaries_df.to_csv(current_exp_dir / f"{current_exp_name}__summary_df.csv")

    return summaries_by_experiment


def describe_context(context: AttackContext) -> dict[str, Any]:
    return {
        "baseline_id": context.baseline_id,
        "kind": context.spec["kind"],
        "dataset_id": context.dataset_id,
        "sampling_strategy": context.spec["sampling_strategy"],
        "result_base_dir": str(context.result_base_dir),
        "dynamic_dataset_info": context.dynamic_dataset_info,
        "shap_path": str(context._shap_path()),
        "feature_selection": context.spec["feature_selection"],
        "value_selection": context.spec["value_selection"],
        "poison_rates": context.spec["poison_rates"],
        "watermark_sizes": context.spec["watermark_sizes"],
    }
