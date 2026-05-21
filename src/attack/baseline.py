"""Run notebook-derived poisoning attack baselines from local Python modules."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from difflib import get_close_matches
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import attack_utils, common_utils, constants
from ..data import data_utils, model_utils
from ..utils.paths import load_yaml, project_path, require_path


SAMPLING_CONFIG_DEFAULT = {
    "adaptive_mode": "mid",
    "adaptive_lower_q": 0.2,
    "adaptive_upper_q": 0.8,
    "adaptive_mix_ratio": 0.5,
}


@dataclass
class AttackContext:
    baseline_id: str
    spec: dict[str, Any]
    dataset_id: str
    result_base_dir: Path
    value_selector_cache_dir: Path
    shap_path: Path
    model_path: Path
    dataset_info: dict[str, Any]
    shap_index_path: Path | None = None


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
    validate_baseline_spec(spec)

    os.environ.setdefault("MPLCONFIGDIR", str(project_path("build", "matplotlib")))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    dataset_id = spec["dataset_id"]
    model_path = require_path(spec["model_path"])
    shap_index_path = None

    if spec["kind"] == "ember2018":
        data_dir = require_path(spec["data_dir"])
        info_path = data_dir / spec.get("dataset_info_file", "dataset_info.json")
        dataset_info = _load_dataset_info(info_path, fallback_rows=constants.train_sizes["ember"])
        shap_path = require_path(spec["shap_path"])
        data_utils.configure(
            {
                "kind": "ember2018",
                "dataset_id": dataset_id,
                "data_dir": str(data_dir),
                "feature_version": int(spec.get("feature_version", 2)),
            }
        )
        constants.num_features["ember"] = int(spec.get("num_features", constants.num_features["ember"]))

    elif spec["kind"] == "ember2024":
        data_root = require_path(spec["data_root"])
        cache_key = ember2024_shap_cache_key(spec, model_path)
        shap_cache_dir = require_path(spec["shap_cache_dir"])
        shap_path = require_path(shap_cache_dir / f"shap_values_{cache_key}.pkl")
        shap_index_path = require_path(shap_cache_dir / f"indices_{cache_key}.npy")
        subset_rows = int(np.load(shap_index_path, mmap_mode="r").shape[0])
        dataset_info = {
            "source_dir": str(data_root / spec["platform"]),
            "target_dir": str(data_root / spec["platform"]),
            "subset_rows": subset_rows,
            "train_fraction": float(spec["train_fraction"]),
            "test_fraction": float(spec.get("test_fraction", 1.0)),
            "subset_mode": spec.get("subset_mode", "stratified_random"),
            "shap_cache_key": cache_key,
        }
        data_utils.configure(
            {
                "kind": "ember2024",
                "dataset_id": dataset_id,
                "data_root": str(data_root),
                "platform": spec["platform"],
                "shap_index_path": str(shap_index_path),
                "test_fraction": float(spec.get("test_fraction", 1.0)),
                "subset_mode": spec.get("subset_mode", "stratified_random"),
                "seed": int(spec.get("seed", 42)),
            }
        )
        constants.num_features[dataset_id] = int(spec.get("num_features", constants.num_features[dataset_id]))

    else:
        raise ValueError(f"Unsupported baseline kind: {spec['kind']}")

    constants.train_sizes[dataset_id] = int(dataset_info["subset_rows"])
    constants.SAVE_FILES_DIR = str(project_path("artifacts", "attack"))
    constants.SAVE_MODEL_DIR = str(project_path("artifacts", "models"))
    constants.CONTAGIO_DATA_DIR = ""
    if dataset_id not in constants.possible_datasets:
        constants.possible_datasets.append(dataset_id)

    model_utils.configure({"model_path": str(model_path)})

    result_base_dir = project_path(*Path(spec["result_root"]).parts) / spec["sampling_strategy"]
    value_selector_cache_dir = project_path("build", "cache", dataset_id)
    attack_utils.VALUE_SELECTOR_CACHE_DIR = str(value_selector_cache_dir)
    attack_utils.DYNAMIC_TRAIN_SAMPLING_STRATEGY = spec["sampling_strategy"]
    attack_utils.DYNAMIC_TRAIN_SAMPLING_CONFIG = dict(SAMPLING_CONFIG_DEFAULT)
    attack_utils.SAMPLING_STATE = {"train_shap_values_df": None}

    return AttackContext(
        baseline_id=baseline_id,
        spec=spec,
        dataset_id=dataset_id,
        result_base_dir=result_base_dir,
        value_selector_cache_dir=value_selector_cache_dir,
        shap_path=shap_path,
        shap_index_path=shap_index_path,
        model_path=model_path,
        dataset_info=dataset_info,
    )


def _load_dataset_info(info_path: Path, fallback_rows: int) -> dict[str, Any]:
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if "subset_rows" not in info:
            info["subset_rows"] = fallback_rows
        return info
    return {"subset_rows": fallback_rows, "dataset_info_file": str(info_path)}


def validate_baseline_spec(spec: dict[str, Any]) -> None:
    invalid_features = sorted(set(spec.get("feature_selection", [])) - constants.feature_selection_criteria)
    invalid_values = sorted(set(spec.get("value_selection", [])) - constants.value_selection_criteria)
    errors = []
    if invalid_features:
        errors.append(_invalid_choice_message("feature selector", invalid_features, constants.feature_selection_criteria))
    if invalid_values:
        errors.append(_invalid_choice_message("value selector", invalid_values, constants.value_selection_criteria))
    target_features = spec.get("target_features", "feasible")
    if target_features not in constants.possible_features_targets:
        errors.append(_invalid_choice_message("target feature group", [target_features], constants.possible_features_targets))
    if errors:
        raise ValueError("\n".join(errors))


def _invalid_choice_message(label: str, invalid_values: list[str], valid_values: set[str]) -> str:
    valid_sorted = sorted(valid_values)
    parts = []
    for value in invalid_values:
        suggestions = get_close_matches(value, valid_sorted, n=1)
        suggestion_text = f" Did you mean '{suggestions[0]}'?" if suggestions else ""
        parts.append(f"Unsupported {label}: '{value}'.{suggestion_text}")
    return "{} Valid choices: {}".format(" ".join(parts), ", ".join(valid_sorted))


def ember2024_shap_cache_key(spec: dict[str, Any], model_path: Path) -> str:
    return "ember2024_{}_{}_train_{}_{}_seed{}_model{}".format(
        spec["platform"].lower(),
        spec.get("model", "lightgbm"),
        fraction_tag(float(spec["train_fraction"])),
        spec.get("subset_mode", "stratified_random"),
        int(spec.get("seed", 42)),
        file_sha256(model_path)[:12],
    )


def file_sha256(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction_tag(value: float) -> str:
    return ("{:.4f}".format(float(value))).rstrip("0").rstrip(".").replace(".", "p")


def build_attack_config(context: AttackContext) -> dict[str, Any]:
    spec = context.spec
    return {
        "seed": int(spec.get("seed", 42)),
        "dataset": context.dataset_id,
        "model": spec.get("model", "lightgbm"),
        "target_features": spec.get("target_features", "feasible"),
        "k_perc": 1.0,
        "k_data": "train",
        "poison_size": [
            max(1, int(context.dataset_info["subset_rows"] * float(rate)))
            for rate in spec.get("poison_rates", [0.005])
        ],
        "watermark_size": [int(v) for v in spec.get("watermark_sizes", [17])],
        "feature_selection": list(spec["feature_selection"]),
        "value_selection": list(spec["value_selection"]),
        "iterations": int(spec.get("iterations", 1)),
    }


def run_attack_baseline(
    context: AttackContext,
    save_attack_artifacts: bool = False,
    save_defense_inputs: bool = False,
    defense_shap_batch_size: int = 8192,
) -> dict[str, pd.DataFrame]:
    cfg = build_attack_config(context)
    if save_attack_artifacts or save_defense_inputs:
        artifact_root = context.result_base_dir.parent / f"{context.result_base_dir.name}-defense" / "attack_artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        cfg["save"] = str(artifact_root)
        print("Attack/defense artifacts dir:", artifact_root)

    _print_run_header(context, cfg)
    _set_random_seeds(cfg["seed"])

    features, _, _, _ = data_utils.load_features(
        feats_to_exclude=constants.features_to_exclude[context.dataset_id],
        dataset=context.dataset_id,
        selected=True,
    )
    x_train, y_train, x_test, y_test = data_utils.load_dataset(dataset=context.dataset_id, selected=True)
    original_model = model_utils.load_model(
        model_id=cfg["model"],
        data_id=context.dataset_id,
        save_path=constants.SAVE_MODEL_DIR,
        file_name=context.dataset_id + "_" + cfg["model"],
    )

    print(
        "Dataset shapes:\n"
        f"\tTrain x: {x_train.shape}\n"
        f"\tTrain y: {y_train.shape}\n"
        f"\tTest x: {x_test.shape}\n"
        f"\tTest y: {y_test.shape}"
    )
    shap_values_df = load_shap_values(context.shap_path, expected_shape=x_train.shape)
    attack_utils.SAMPLING_STATE["train_shap_values_df"] = shap_values_df
    source_train_indices = load_source_train_indices(context, train_rows=x_train.shape[0])

    f_selectors = attack_utils.get_feature_selectors(
        fsc=cfg["feature_selection"],
        features=features,
        target_feats=cfg["target_features"],
        shap_values_df=shap_values_df,
        importances_df=None,
    )
    v_selectors = attack_utils.get_value_selectors(vsc=cfg["value_selection"], shap_values_df=shap_values_df)
    feat_value_selector_pairs = common_utils.get_feat_value_pairs(
        feat_sel=list(f_selectors.keys()),
        val_sel=list(v_selectors.keys()),
    )

    print("Chosen feature-value selectors:")
    for feature_selector, value_selector in feat_value_selector_pairs:
        print(f"{feature_selector} - {value_selector}")

    x_mw_poisoning_candidates, x_mw_poisoning_candidates_idx = attack_utils.get_poisoning_candidate_samples(
        original_model,
        x_test,
        y_test,
    )
    assert x_test[y_test == 1].shape[0] == x_mw_poisoning_candidates_idx.shape[0]

    del x_train, y_train, x_test, y_test

    summaries_by_experiment: dict[str, pd.DataFrame] = {}
    for feature_selector_name, value_selector_name in feat_value_selector_pairs:
        current_exp_name = common_utils.get_exp_name(
            context.dataset_id,
            cfg["model"],
            feature_selector_name,
            value_selector_name,
            cfg["target_features"],
        )
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
        for summary in attack_utils.run_experiments(
            X_mw_poisoning_candidates=x_mw_poisoning_candidates,
            X_mw_poisoning_candidates_idx=x_mw_poisoning_candidates_idx,
            gw_poison_set_sizes=cfg["poison_size"],
            watermark_feature_set_sizes=cfg["watermark_size"],
            feat_selectors=[f_selectors[feature_selector_name]],
            feat_value_selectors=[v_selectors[value_selector_name]],
            iterations=cfg["iterations"],
            save_watermarks=save_watermarks,
            model_id=cfg["model"],
            dataset=context.dataset_id,
            save_full_artifacts=save_attack_artifacts,
            save_defense_inputs=save_defense_inputs,
            defense_shap_batch_size=defense_shap_batch_size,
            source_train_indices=source_train_indices,
        ):
            attack_utils.print_experiment_summary(
                summary,
                f_selectors[feature_selector_name].name,
                v_selectors[value_selector_name].name
                if v_selectors[value_selector_name] is not None
                else f_selectors[feature_selector_name].name,
            )
            summaries.append(summary)
            print("Exp took {:.2f} seconds\n".format(time.time() - start_time))
            start_time = time.time()

        summaries_df = attack_utils.create_summary_df(summaries)
        summaries_by_experiment[current_exp_name] = summaries_df
        print(summaries_df)
        summaries_df.to_csv(current_exp_dir / f"{current_exp_name}__summary_df.csv")

    return summaries_by_experiment


def load_shap_values(shap_path: Path, expected_shape: tuple[int, int]) -> pd.DataFrame:
    start_time = time.time()
    print(f"Loading cached SHAP values from {shap_path}\n")
    shap_values_df = pd.read_pickle(shap_path)
    print("Loading cached SHAP took {:.2f} seconds\n".format(time.time() - start_time))
    if shap_values_df.shape != expected_shape:
        raise ValueError(f"Loaded SHAP shape {shap_values_df.shape} does not match attack data shape {expected_shape}")
    return shap_values_df


def load_source_train_indices(context: AttackContext, train_rows: int) -> np.ndarray:
    if context.shap_index_path is None:
        return np.arange(train_rows, dtype=np.int64)
    source_indices = np.asarray(np.load(context.shap_index_path), dtype=np.int64)
    if source_indices.shape[0] != train_rows:
        print(
            "Source train index count {} does not match loaded train rows {}; using local row ids.".format(
                source_indices.shape[0],
                train_rows,
            )
        )
        return np.arange(train_rows, dtype=np.int64)
    return source_indices


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def _print_run_header(context: AttackContext, cfg: dict[str, Any]) -> None:
    print("Baseline:", context.baseline_id)
    print("Dataset:", context.dataset_id)
    print("Data info:", context.dataset_info)
    print("Model path:", context.model_path)
    print("SHAP path:", context.shap_path)
    print("Results base dir:", context.result_base_dir)
    print("Train rows:", context.dataset_info["subset_rows"])
    print("Feature selectors:", cfg["feature_selection"])
    print("Value selectors:", cfg["value_selection"])
    print("Poison sizes:", cfg["poison_size"])
    print("Watermark sizes:", cfg["watermark_size"])
    print("Sampling strategy:", context.spec["sampling_strategy"])
    if cfg.get("save"):
        print("Artifact root:", cfg["save"])


def describe_context(context: AttackContext) -> dict[str, Any]:
    return {
        "baseline_id": context.baseline_id,
        "kind": context.spec["kind"],
        "dataset_id": context.dataset_id,
        "sampling_strategy": context.spec["sampling_strategy"],
        "result_base_dir": str(context.result_base_dir),
        "value_selector_cache_dir": str(context.value_selector_cache_dir),
        "dataset_info": context.dataset_info,
        "model_path": str(context.model_path),
        "shap_path": str(context.shap_path),
        "shap_index_path": str(context.shap_index_path) if context.shap_index_path else None,
        "feature_selection": context.spec["feature_selection"],
        "value_selection": context.spec["value_selection"],
        "poison_rates": context.spec["poison_rates"],
        "watermark_sizes": context.spec["watermark_sizes"],
    }
