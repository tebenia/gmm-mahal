"""Common naming and configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path

from . import constants


def read_config(cfg_path, atk_def=True):
    cfg_path = Path(cfg_path)
    if not cfg_path.is_file():
        raise ValueError(f"Provided configuration file does not exist: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    for value in cfg["poison_size"]:
        if type(value) is not float:
            raise ValueError("Poison sizes must be all floats in [0, 1]")
    for value in cfg["watermark_size"]:
        if type(value) is not int:
            raise ValueError("Watermark sizes must be all integers")
    if cfg["target_features"] not in constants.possible_features_targets:
        raise ValueError(f"Invalid feature target {cfg['target_features']}")
    cfg["target_features"] = constants.canonical_feature_target(cfg["target_features"])
    for value in cfg["feature_selection"]:
        if value not in constants.feature_selection_criteria:
            raise ValueError(f"Invalid feature selection criterion {value}")
    for value in cfg["value_selection"]:
        if value not in constants.value_selection_criteria:
            raise ValueError(f"Invalid value selection criterion {value}")
    if cfg["dataset"] not in constants.possible_datasets:
        raise ValueError(f"Invalid dataset {cfg['dataset']}")

    train_size = constants.train_sizes[cfg["dataset"]]
    cfg["poison_size"] = [int(train_size * value) for value in cfg["poison_size"]]
    if atk_def and type(cfg["iterations"]) is not int:
        raise ValueError(f"Iterations must be an integer {cfg['iterations']}")
    return cfg


def get_exp_name(data, mod, f_s, v_s, target):
    return data + "__" + mod + "__" + f_s + "__" + v_s + "__" + target


def get_feat_value_pairs(feat_sel, val_sel):
    pairs = []
    combined_feature_selectors = {
        constants.feature_selection_criterion_combined,
        constants.feature_selection_criterion_combined_additive,
        constants.feature_selection_criterion_fix,
    }
    for feat in feat_sel:
        if feat in combined_feature_selectors:
            pairs.append((feat, feat))
        else:
            for val in val_sel:
                pairs.append((feat, val))
    return pairs
