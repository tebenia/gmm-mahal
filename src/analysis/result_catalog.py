from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PMSB_SAMPLING_STRATEGIES = [
    "feature_based_distance",
    "distribution_based_distance",
    "shap_contribution_distance",
]

SAMPLING_ORDER = [
    "random",
    *PMSB_SAMPLING_STRATEGIES,
]

VALUE_ORDER = [
    "combined_shap",
    "argmin_Nv_sum_abs_shap",
    "min_population_new",
    "low_shap_signed",
    "frequency_bounded",
    "frequency_bounded_signed_shap",
    "quantile_05",
    "quantile_50",
    "quantile_95",
    "benign_prototype",
    "corr_count_abs_shap",
]

VALUE_LABELS = {
    "combined_shap": "CombinedSHAP",
    "argmin_Nv_sum_abs_shap": "CountAbsSHAP",
    "min_population_new": "MinPopulation",
    "low_shap_signed": "LowSignedSHAP",
    "frequency_bounded": "FreqBounded",
    "frequency_bounded_signed_shap": "FreqSignedSHAP",
    "quantile_05": "Q05",
    "quantile_50": "Q50",
    "quantile_95": "Q95",
    "benign_prototype": "BenignProto",
    "corr_count_abs_shap": "CorrCountAbs",
}

DEFENSE_METHOD_ORDER = [
    "isolation_forest",
    "spectral_signature",
    "hdbscan",
]

DEFENSE_LABELS = {
    "isolation_forest": "Isolation Forest",
    "spectral_signature": "Spectral Signature",
    "hdbscan": "HDBSCAN",
}

FALLBACK_TRAIN_ROWS = {
    "EMBER2018": 120_000,
    "EMBER2024 WIN32": 208_000,
    "EMBER2024 WIN64": 208_000,
}

_POISON_FOLDER_RE = re.compile(
    r"poison(?:ing)?(?:\s+rate)?[\s_-]*([0-9]+(?:[p.][0-9]+)?)",
    flags=re.IGNORECASE,
)


def project_root_from_cwd(cwd: str | Path | None = None) -> Path:
    root = Path.cwd() if cwd is None else Path(cwd)
    return root.parent if root.name == "notebooks" else root


def parse_experiment_name(name: str) -> dict[str, Any]:
    parts = name.split("__")
    if len(parts) < 5:
        return {
            "baseline_tag": None,
            "model_family": None,
            "feature_selector": None,
            "value_selector": None,
            "target_features": None,
        }
    return {
        "baseline_tag": parts[0],
        "model_family": parts[1],
        "feature_selector": parts[2],
        "value_selector": parts[3],
        "target_features": "__".join(parts[4:]),
    }


def _is_experiment_name(name: str) -> bool:
    return len(name.split("__")) >= 5


def _experiment_dir_below_attack_artifacts(path: Path) -> Path | None:
    parts = list(path.parts)
    if "attack_artifacts" not in parts:
        return None
    start = parts.index("attack_artifacts") + 1
    for index in range(start, len(parts)):
        if _is_experiment_name(parts[index]):
            return Path(*parts[: index + 1])
    return None


def _relative_result_parts(path: Path, results_root: Path) -> list[str]:
    resolved = path.resolve()
    try:
        return list(resolved.relative_to(results_root.resolve()).parts)
    except ValueError:
        parts = list(resolved.parts)
        if "results" not in parts:
            return []
        return parts[parts.index("results") + 1 :]


def _dataset_context(parts: list[str]) -> dict[str, Any]:
    if not parts:
        return {
            "dataset_path": None,
            "dataset_label": "unknown",
            "dataset_family": "unknown",
            "platform": None,
        }
    if parts[0] == "ember":
        dataset_path = "/".join(parts[:2]) if len(parts) > 1 else "ember"
        return {
            "dataset_path": dataset_path,
            "dataset_label": "EMBER2018",
            "dataset_family": "ember2018",
            "platform": None,
        }
    if parts[0] == "ember2024":
        platform = parts[1].lower() if len(parts) > 1 else None
        label = f"EMBER2024 {platform.upper()}" if platform else "EMBER2024"
        dataset_path = "/".join(parts[:2]) if len(parts) > 1 else "ember2024"
        return {
            "dataset_path": dataset_path,
            "dataset_label": label,
            "dataset_family": "ember2024",
            "platform": platform,
        }
    return {
        "dataset_path": parts[0],
        "dataset_label": parts[0],
        "dataset_family": parts[0],
        "platform": None,
    }


def _sampling_from_parts(parts: Iterable[str]) -> str:
    for part in parts:
        candidate = part.removesuffix("-defense")
        if candidate in SAMPLING_ORDER:
            return candidate
    return "unknown"


def _poison_folder_label(parts: Iterable[str]) -> tuple[str | None, float | None]:
    for part in parts:
        match = _POISON_FOLDER_RE.search(part)
        if not match:
            continue
        raw = match.group(1)
        try:
            value = float(raw.replace("p", "."))
        except ValueError:
            value = None
        return part, value
    return None, None


def parse_result_context(path: str | Path, results_root: str | Path) -> dict[str, Any]:
    path = Path(path)
    results_root = Path(results_root)
    parts = _relative_result_parts(path, results_root)
    dataset = _dataset_context(parts)
    folder_label, folder_value = _poison_folder_label(parts)
    return {
        **dataset,
        "sampling_strategy": _sampling_from_parts(parts),
        "poison_folder_label": folder_label,
        "poison_folder_value": folder_value,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_train_rows(results_root: str | Path) -> dict[str, int]:
    results_root = Path(results_root)
    candidates: dict[str, list[int]] = {}
    for path in results_root.glob("**/attack_artifacts/**/defense_metadata.json"):
        if _experiment_dir_below_attack_artifacts(path) != path.parent:
            continue
        metadata = _load_json(path)
        if not metadata:
            continue
        context = parse_result_context(path, results_root)
        rows = metadata.get("num_train_rows")
        if rows is None:
            continue
        candidates.setdefault(context["dataset_label"], []).append(int(rows))
    out = dict(FALLBACK_TRAIN_ROWS)
    for label, values in candidates.items():
        out[label] = int(pd.Series(values).mode().iloc[0])
    return out


def _rate_columns(
    poison_size: pd.Series,
    train_rows: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    size = pd.to_numeric(poison_size, errors="coerce")
    rows = pd.to_numeric(train_rows, errors="coerce")
    rate = size / rows.replace(0, np.nan)
    percent = rate * 100
    label = percent.map(lambda value: f"{value:g}%" if pd.notna(value) else "unknown")
    return rate, percent, label


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "value_selector" in out:
        values = out["value_selector"].astype("string")
        out["selector_label"] = values.map(VALUE_LABELS).fillna(values)
    if "defense_method" in out:
        methods = out["defense_method"].astype("string")
        out["defense_label"] = methods.map(DEFENSE_LABELS).fillna(methods)
    if "sampling_strategy" in out:
        out["sampling_label"] = (
            out["sampling_strategy"]
            .astype("string")
            .str.replace("_", " ", regex=False)
            .str.title()
        )
    return out


def load_attack_summaries(results_root: str | Path) -> pd.DataFrame:
    results_root = Path(results_root)
    train_rows_by_dataset = discover_train_rows(results_root)
    frames = []
    for path in sorted(results_root.glob("**/*__summary_df.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"Could not read {path}: {exc}")
            continue
        context = parse_result_context(path, results_root)
        parsed = parse_experiment_name(path.parent.name)
        for key, value in {**context, **parsed}.items():
            frame[key] = value
        frame["summary_df_path"] = str(path)
        frame["source_mtime"] = path.stat().st_mtime
        frame["train_rows"] = context["dataset_label"] and train_rows_by_dataset.get(
            context["dataset_label"]
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    frames = [frame.dropna(axis=1, how="all") for frame in frames]
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["poison_size"] = pd.to_numeric(out.get("num_gw_to_watermark"), errors="coerce")
    out["poison_rate_train"], out["poison_rate_percent"], out["poison_rate_label"] = (
        _rate_columns(out["poison_size"], out["train_rows"])
    )
    out["attack_effectiveness_percent"] = pd.to_numeric(
        out.get("evasions_success_percent"), errors="coerce"
    )
    out["clean_model_original_test_accuracy_percent"] = pd.to_numeric(
        out.get("orig_model_orig_test_set_accuracy"), errors="coerce"
    )
    out["clean_model_watermarked_test_accuracy_percent"] = pd.to_numeric(
        out.get("orig_model_mw_test_set_accuracy"), errors="coerce"
    )
    out["clean_model_attack_test_accuracy_percent"] = (
        out["clean_model_watermarked_test_accuracy_percent"]
    )
    out["backdoored_clean_accuracy_percent"] = pd.to_numeric(
        out.get("new_model_orig_test_set_accuracy"), errors="coerce"
    )
    out["watermark_size"] = pd.to_numeric(
        out.get("num_watermark_features"), errors="coerce"
    )
    return add_display_columns(out)


def _parse_defense_method(setting: str) -> str:
    if setting.startswith("isolation_forest"):
        return "isolation_forest"
    if setting.startswith("spectral_signature"):
        return "spectral_signature"
    if setting.startswith("hdbscan"):
        return "hdbscan"
    return setting.split("_")[0]


def _detector_budget_columns(out: pd.DataFrame) -> pd.DataFrame:
    benign_scored = pd.to_numeric(out["benign_rows_scored"], errors="coerce")
    out["detector_pool_poison_rate"] = out["poison_size"] / benign_scored.replace(
        0, np.nan
    )
    out["detector_pool_poison_percent"] = out["detector_pool_poison_rate"] * 100
    out["detector_removal_budget_percent"] = (
        pd.to_numeric(out["removed_rows"], errors="coerce")
        / benign_scored.replace(0, np.nan)
        * 100
    )
    out["poison_recall_percent"] = out["poison_recall"] * 100
    out["clean_false_positive_rate_percent"] = (
        out["clean_false_positive_rate"] * 100
    )
    out["poison_removal_precision_percent"] = np.where(
        out["removed_rows"] > 0,
        out["removed_poisoned_rows"] / out["removed_rows"] * 100,
        np.nan,
    )
    out["setting_matches_poison_budget"] = False
    isolation = out["defense_method"] == "isolation_forest"
    out.loc[isolation, "setting_matches_poison_budget"] = np.isclose(
        pd.to_numeric(
            out.loc[isolation, "detector_contamination"], errors="coerce"
        ),
        out.loc[isolation, "detector_pool_poison_rate"],
        atol=1e-9,
        equal_nan=False,
    )
    spectral = out["defense_method"] == "spectral_signature"
    out.loc[spectral, "setting_matches_poison_budget"] = np.isclose(
        pd.to_numeric(
            out.loc[spectral, "detector_removal_percent"], errors="coerce"
        )
        / 100,
        out.loc[spectral, "detector_pool_poison_rate"],
        atol=1e-9,
        equal_nan=False,
    )
    out.loc[out["defense_method"] == "hdbscan", "setting_matches_poison_budget"] = True
    return out


def load_detector_metrics(results_root: str | Path) -> pd.DataFrame:
    results_root = Path(results_root)
    rows = []
    for path in sorted(
        results_root.glob("**/severi_detectors/*/severi_detector_metadata.json")
    ):
        artifact_dir = _experiment_dir_below_attack_artifacts(path)
        if artifact_dir is None:
            continue
        metadata = _load_json(path) or {}
        config = metadata.get("config", {})
        metrics = metadata.get("removal_metrics", {})
        input_shape = metadata.get("input_shape", {})
        watermarked_shape = input_shape.get("watermarked_X") or []
        train_rows = watermarked_shape[0] if watermarked_shape else np.nan
        setting = path.parent.name
        context = parse_result_context(path, results_root)
        parsed = parse_experiment_name(artifact_dir.name)
        rows.append(
            {
                **context,
                **parsed,
                **metrics,
                "artifact_dir": str(artifact_dir),
                "detector_dir": str(path.parent),
                "detector_metadata_path": str(path),
                "defense_method": config.get("method")
                or _parse_defense_method(setting),
                "defense_setting": setting,
                "detector_feature_mode": config.get("feature_mode"),
                "detector_top_k": config.get("top_k"),
                "detector_contamination": config.get("contamination"),
                "detector_removal_percent": config.get("removal_percent"),
                "detector_max_benign_rows": config.get("max_benign_rows"),
                "benign_rows_scored": input_shape.get("benign_rows_scored"),
                "selected_features": input_shape.get("selected_features"),
                "train_rows": train_rows,
                "source_mtime": path.stat().st_mtime,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["poison_size"] = pd.to_numeric(out["total_poisoned_rows"], errors="coerce")
    out["train_rows"] = pd.to_numeric(out["train_rows"], errors="coerce")
    out["poison_rate_train"], out["poison_rate_percent"], out["poison_rate_label"] = (
        _rate_columns(out["poison_size"], out["train_rows"])
    )
    out = _detector_budget_columns(out)
    return add_display_columns(out)


def preferred_detector_metrics(detector_df: pd.DataFrame) -> pd.DataFrame:
    if detector_df.empty:
        return detector_df.copy()
    out = detector_df[
        detector_df["defense_method"].isin(DEFENSE_METHOD_ORDER)
        & detector_df["detector_max_benign_rows"].isna()
        & detector_df["setting_matches_poison_budget"]
    ].copy()
    keys = [
        "dataset_label",
        "sampling_strategy",
        "poison_rate_train",
        "feature_selector",
        "value_selector",
        "target_features",
        "defense_method",
    ]
    return (
        out.sort_values("source_mtime")
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )


def load_defense_metrics(results_root: str | Path) -> pd.DataFrame:
    results_root = Path(results_root)
    rows = []
    for path in sorted(results_root.glob("**/defense_retrain_metrics.csv")):
        if "severi_detectors" not in path.parts:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        artifact_dir = _experiment_dir_below_attack_artifacts(path)
        if artifact_dir is None:
            continue
        experiment_name = artifact_dir.name
        severi_idx = path.parts.index("severi_detectors")
        setting = path.parts[severi_idx + 1]
        detector_dir = Path(*path.parts[: severi_idx + 2])
        detector_meta = _load_json(detector_dir / "severi_detector_metadata.json") or {}
        detector_config = detector_meta.get("config", {})
        input_shape = detector_meta.get("input_shape", {})
        context = parse_result_context(path, results_root)
        parsed = parse_experiment_name(experiment_name)
        row = frame.iloc[0].to_dict()
        row.update(
            {
                **context,
                **parsed,
                "artifact_dir": str(artifact_dir),
                "metrics_path": str(path),
                "detector_dir": str(detector_dir),
                "defense_method": _parse_defense_method(setting),
                "defense_setting": setting,
                "detector_feature_mode": detector_config.get("feature_mode"),
                "detector_top_k": detector_config.get("top_k"),
                "detector_contamination": detector_config.get("contamination"),
                "detector_removal_percent": detector_config.get("removal_percent"),
                "detector_max_benign_rows": detector_config.get("max_benign_rows"),
                "benign_rows_scored": input_shape.get("benign_rows_scored"),
                "source_mtime": path.stat().st_mtime,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["poison_size"] = pd.to_numeric(out["total_poisoned_rows"], errors="coerce")
    out["train_rows"] = pd.to_numeric(out["train_rows_before"], errors="coerce")
    out["poison_rate_train"], out["poison_rate_percent"], out["poison_rate_label"] = (
        _rate_columns(out["poison_size"], out["train_rows"])
    )
    out["backdoored_asr_percent"] = out["backdoored_asr"] * 100
    out["defended_asr_percent"] = out["defended_asr"] * 100
    out["asr_reduction_pp"] = (
        out["backdoored_asr"] - out["defended_asr"]
    ) * 100
    out["backdoored_clean_accuracy_percent"] = (
        out["backdoored_clean_accuracy"] * 100
    )
    out["defended_clean_accuracy_percent"] = out["defended_clean_accuracy"] * 100
    out["clean_accuracy_delta_pp"] = (
        out["defended_clean_accuracy"] - out["backdoored_clean_accuracy"]
    ) * 100
    out["removed_row_percent"] = out["removed_rows"] / out["train_rows_before"] * 100
    out = _detector_budget_columns(out)
    return add_display_columns(out)


def preferred_defense_metrics(defense_df: pd.DataFrame) -> pd.DataFrame:
    if defense_df.empty:
        return defense_df.copy()
    out = defense_df[
        defense_df["defense_method"].isin(DEFENSE_METHOD_ORDER)
        & defense_df["detector_max_benign_rows"].isna()
        & defense_df["setting_matches_poison_budget"]
    ].copy()
    keys = [
        "dataset_label",
        "sampling_strategy",
        "poison_rate_train",
        "feature_selector",
        "value_selector",
        "target_features",
        "defense_method",
    ]
    return (
        out.sort_values("source_mtime")
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )


def discover_attack_artifacts(results_root: str | Path) -> pd.DataFrame:
    results_root = Path(results_root)
    rows = []
    diagnostic_required = [
        "watermarked_X.npy",
        "watermarked_y.npy",
        "wm_config.npy",
        "defense_metadata.npz",
    ]
    for metadata_path in sorted(
        results_root.glob("**/attack_artifacts/**/defense_metadata.json")
    ):
        path = metadata_path.parent
        if _experiment_dir_below_attack_artifacts(metadata_path) != path:
            continue
        metadata = _load_json(metadata_path) or {}
        context = parse_result_context(path, results_root)
        parsed = parse_experiment_name(path.name)
        poison_size = metadata.get("num_poisoned_rows")
        train_rows = metadata.get("num_train_rows")
        missing_diagnostic_files = [
            name for name in diagnostic_required if not (path / name).exists()
        ]
        rate = (
            float(poison_size) / float(train_rows)
            if poison_size is not None and train_rows
            else np.nan
        )
        rows.append(
            {
                **context,
                **parsed,
                "artifact_name": path.name,
                "artifact_dir": str(path),
                "poison_size": poison_size,
                "train_rows": train_rows,
                "poison_rate_train": rate,
                "poison_rate_percent": rate * 100,
                "poison_rate_label": f"{rate * 100:g}%",
                "watermark_size": len(metadata.get("watermark_features", [])),
                "defense_shap_saved": metadata.get("defense_shap_saved"),
                "diagnostics_ready": not missing_diagnostic_files,
                "missing_diagnostic_files": ", ".join(missing_diagnostic_files),
            }
        )
    return add_display_columns(pd.DataFrame(rows))


def _diagnostic_rate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    poisoned = pd.to_numeric(out.get("poisoned_benign_rows_scored"), errors="coerce")
    total = pd.to_numeric(out.get("total_train_rows"), errors="coerce")
    out["poison_size"] = poisoned
    out["train_rows"] = total
    out["poison_rate_train"], out["poison_rate_percent"], out["poison_rate_label"] = (
        _rate_columns(poisoned, total)
    )
    return out


def load_detectability_summaries(
    results_root: str | Path,
    *,
    include_legacy_aggregate: bool = True,
) -> pd.DataFrame:
    results_root = Path(results_root)
    frames = []
    direct_paths = sorted(
        results_root.glob("**/detectability_diagnostics/*/detectability_summary.csv")
    )
    for path in direct_paths:
        frame = pd.read_csv(path)
        artifact_dir = path.parents[2]
        context = parse_result_context(artifact_dir, results_root)
        parsed = parse_experiment_name(artifact_dir.name)
        for key, value in {**context, **parsed}.items():
            if key not in frame:
                frame[key] = value
        frame["artifact_dir"] = str(artifact_dir)
        frame["diagnostic_output_dir"] = str(path.parent)
        frame["summary_path"] = str(path)
        frame["diagnostic_source"] = "current_artifact"
        frame["source_mtime"] = path.stat().st_mtime
        frames.append(frame)
    if include_legacy_aggregate:
        aggregate = results_root / "plots" / "detectability_diagnostics_summary.csv"
        if aggregate.exists():
            frame = pd.read_csv(aggregate)
            frame["diagnostic_source"] = "legacy_aggregate"
            frame["source_mtime"] = aggregate.stat().st_mtime
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = _diagnostic_rate_columns(
        pd.concat(frames, ignore_index=True, sort=False)
    )
    keys = [
        "dataset_label",
        "sampling_strategy",
        "poison_rate_train",
        "feature_selector",
        "value_selector",
        "target_features",
    ]
    out["source_priority"] = out["diagnostic_source"].map(
        {"legacy_aggregate": 0, "current_artifact": 1}
    )
    out = (
        out.sort_values(["source_priority", "source_mtime"])
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )
    return add_display_columns(out)


def load_detectability_score_metrics(
    results_root: str | Path,
    *,
    include_legacy_aggregate: bool = True,
) -> pd.DataFrame:
    results_root = Path(results_root)
    frames = []
    direct_paths = sorted(
        results_root.glob(
            "**/detectability_diagnostics/*/detectability_score_metrics.csv"
        )
    )
    for path in direct_paths:
        frame = pd.read_csv(path)
        artifact_dir = path.parents[2]
        context = parse_result_context(artifact_dir, results_root)
        parsed = parse_experiment_name(artifact_dir.name)
        metadata = _load_json(artifact_dir / "defense_metadata.json") or {}
        for key, value in {**context, **parsed}.items():
            frame[key] = value
        frame["artifact_dir"] = str(artifact_dir)
        frame["score_metrics_path"] = str(path)
        frame["diagnostic_source"] = "current_artifact"
        frame["poison_size"] = metadata.get("num_poisoned_rows")
        frame["train_rows"] = metadata.get("num_train_rows")
        frames.append(frame)
    if include_legacy_aggregate:
        aggregate = results_root / "plots" / "detectability_score_metrics.csv"
        if aggregate.exists():
            frame = pd.read_csv(aggregate)
            frame["diagnostic_source"] = "legacy_aggregate"
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "poison_size" not in out:
        out["poison_size"] = np.nan
    if "train_rows" not in out:
        out["train_rows"] = np.nan
    missing_rate = out["poison_size"].isna() | out["train_rows"].isna()
    if missing_rate.any() and "artifact_dir" in out:
        summary = load_detectability_summaries(
            results_root, include_legacy_aggregate=include_legacy_aggregate
        )
        lookup = summary.set_index("artifact_dir")[
            ["poison_size", "train_rows"]
        ].to_dict("index")
        for idx in out.index[missing_rate]:
            values = lookup.get(str(out.at[idx, "artifact_dir"]))
            if values:
                out.at[idx, "poison_size"] = values["poison_size"]
                out.at[idx, "train_rows"] = values["train_rows"]
    out["poison_rate_train"], out["poison_rate_percent"], out["poison_rate_label"] = (
        _rate_columns(out["poison_size"], out["train_rows"])
    )
    return add_display_columns(out)


def merge_effectiveness(
    left: pd.DataFrame,
    attack_df: pd.DataFrame,
) -> pd.DataFrame:
    if left.empty or attack_df.empty:
        return left.copy()
    keys = [
        "dataset_label",
        "sampling_strategy",
        "poison_rate_train",
        "feature_selector",
        "value_selector",
        "target_features",
    ]
    attack_cols = keys + [
        "attack_effectiveness_percent",
        "clean_model_original_test_accuracy_percent",
        "clean_model_watermarked_test_accuracy_percent",
        "clean_model_attack_test_accuracy_percent",
        "backdoored_clean_accuracy_percent",
        "poison_size",
        "watermark_size",
        "summary_df_path",
    ]
    attack_unique = (
        attack_df.sort_values("source_mtime")
        .drop_duplicates(keys, keep="last")[attack_cols]
    )
    return left.merge(
        attack_unique,
        on=keys,
        how="left",
        suffixes=("", "_attack"),
    )


def comparison_coverage(
    attack_df: pd.DataFrame,
    *,
    datasets: Iterable[str] | None = None,
    samplings: Iterable[str] | None = None,
    target_features: Iterable[str] | None = None,
) -> pd.DataFrame:
    out = attack_df.copy()
    if datasets is not None:
        out = out[out["dataset_label"].isin(datasets)]
    if samplings is not None:
        out = out[out["sampling_strategy"].isin(samplings)]
    if target_features is not None:
        out = out[out["target_features"].isin(target_features)]
    if out.empty:
        return out
    return (
        out.groupby(
            [
                "dataset_label",
                "poison_rate_label",
                "sampling_strategy",
                "target_features",
            ],
            dropna=False,
        )
        .agg(
            selectors=("value_selector", "nunique"),
            rows=("value_selector", "size"),
            watermark_sizes=("watermark_size", lambda values: sorted(set(values.dropna()))),
        )
        .reset_index()
        .sort_values(["dataset_label", "poison_rate_label", "sampling_strategy"])
    )
