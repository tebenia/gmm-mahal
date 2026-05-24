"""Retrain a defended model after suspicious-row removal and evaluate it."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..attack.baseline import DEFAULT_BASELINES_CONFIG, build_context
from ..data import data_utils, model_utils
from ..utils.paths import resolve_path


@dataclass
class DefenseRetrainConfig:
    artifact_dir: str
    gmm_dir: str | None
    remove_idx_path: str
    output_dir: str
    baseline: str | None = None
    config_path: str | None = None
    max_train_rows: int | None = None
    max_eval_rows: int | None = None
    save_model: bool = True


@dataclass
class DefenseRetrainResult:
    output_dir: str
    metrics_path: str
    metadata_path: str
    defended_model_path: str | None
    train_rows_before: int
    train_rows_after: int
    removed_rows: int
    removed_poisoned_rows: int | None
    removed_clean_rows: int | None
    defended_clean_accuracy: float | None
    defended_asr: float
    backdoored_clean_accuracy: float | None
    backdoored_asr: float | None


def run_defense_retrain(
    artifact_dir: str | Path,
    gmm_dir: str | Path | None = None,
    remove_idx_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    baseline: str | None = None,
    config_path: str | Path = DEFAULT_BASELINES_CONFIG,
    max_train_rows: int | None = None,
    max_eval_rows: int | None = None,
    save_model: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> DefenseRetrainResult | dict:
    artifact_path = _resolve_existing_dir(artifact_dir)
    gmm_path = _resolve_existing_dir(gmm_dir) if gmm_dir is not None else None
    remove_path = resolve_remove_idx_path(remove_idx_path=remove_idx_path, gmm_dir=gmm_path)
    output_path = resolve_output_dir(output_dir=output_dir, gmm_dir=gmm_path, artifact_dir=artifact_path)

    required_paths = {
        "watermarked_X": artifact_path / "watermarked_X.npy",
        "watermarked_y": artifact_path / "watermarked_y.npy",
        "watermarked_X_test": artifact_path / "watermarked_X_test.npy",
        "defense_metadata": artifact_path / "defense_metadata.npz",
        "remove_watermarked_idx": remove_path,
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required retrain artifact(s): {}. "
            "Attack artifacts must be produced with --save-attack-artifacts, and GMM must produce "
            "remove_watermarked_idx.npy.".format(", ".join(missing))
        )

    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "defense_retrain_metadata.json"
    metrics_path = output_path / "defense_retrain_metrics.csv"
    defended_model_path = output_path / "defended_model.txt" if save_model else None
    if metadata_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"{metadata_path} already exists. Pass --overwrite to replace it.")

    if dry_run:
        return {
            "artifact_dir": str(artifact_path),
            "gmm_dir": str(gmm_path) if gmm_path is not None else None,
            "remove_idx_path": str(remove_path),
            "output_dir": str(output_path),
            "required_paths": {key: str(path) for key, path in required_paths.items()},
            "baseline": baseline,
            "config_path": str(config_path) if config_path is not None else None,
            "max_train_rows": max_train_rows,
            "max_eval_rows": max_eval_rows,
        }

    start_time = time.time()
    X_train = np.load(required_paths["watermarked_X"], mmap_mode="r", allow_pickle=True)
    y_train = np.load(required_paths["watermarked_y"], mmap_mode="r", allow_pickle=True)
    X_watermarked_mw = np.load(required_paths["watermarked_X_test"], mmap_mode="r", allow_pickle=True)
    X_train = _unwrap_saved_array(X_train)
    y_train = _unwrap_saved_array(y_train)
    X_watermarked_mw = _unwrap_saved_array(X_watermarked_mw)

    if max_train_rows is not None:
        if max_train_rows <= 0:
            raise ValueError("--max-train-rows must be positive")
        row_limit = min(int(max_train_rows), X_train.shape[0])
    else:
        row_limit = X_train.shape[0]

    y_train = np.asarray(y_train[:row_limit])
    remove_idx = load_remove_indices(remove_path, n_rows=X_train.shape[0], row_limit=row_limit)
    keep_mask = np.ones(row_limit, dtype=bool)
    keep_mask[remove_idx] = False
    X_defended = X_train[:row_limit][keep_mask]
    y_defended = y_train[keep_mask]

    defended_model = model_utils.train_model("lightgbm", X_defended, y_defended)
    if save_model and defended_model_path is not None:
        defended_model.save_model(str(defended_model_path))

    if max_eval_rows is not None:
        if max_eval_rows <= 0:
            raise ValueError("--max-eval-rows must be positive")
        X_watermarked_mw_eval = X_watermarked_mw[: min(int(max_eval_rows), X_watermarked_mw.shape[0])]
    else:
        X_watermarked_mw_eval = X_watermarked_mw

    defended_watermarked_preds = predict_binary(defended_model, X_watermarked_mw_eval)
    defended_asr = float(np.mean(defended_watermarked_preds == 0))
    defended_detection_rate = float(np.mean(defended_watermarked_preds == 1))

    context = None
    clean_eval = None
    defended_clean_accuracy = None
    backdoored_clean_accuracy = None
    if baseline is not None:
        context = build_context(baseline, config_path=config_path)
        _, _, X_clean_test, y_clean_test = data_utils.load_dataset(dataset=context.dataset_id, selected=True)
        if max_eval_rows is not None:
            eval_rows = min(int(max_eval_rows), X_clean_test.shape[0])
            X_clean_test = X_clean_test[:eval_rows]
            y_clean_test = y_clean_test[:eval_rows]
        defended_clean_accuracy = accuracy(defended_model, X_clean_test, y_clean_test)
        clean_eval = {
            "rows": int(y_clean_test.shape[0]),
            "dataset_id": context.dataset_id,
        }

    backdoored_model = load_backdoored_model(artifact_path)
    backdoored_asr = None
    backdoored_detection_rate = None
    if backdoored_model is not None:
        backdoored_watermarked_preds = predict_binary(backdoored_model, X_watermarked_mw_eval)
        backdoored_asr = float(np.mean(backdoored_watermarked_preds == 0))
        backdoored_detection_rate = float(np.mean(backdoored_watermarked_preds == 1))
        if baseline is not None and clean_eval is not None:
            # Reload clean test only if it was not kept above; local variable remains in scope when baseline is set.
            backdoored_clean_accuracy = accuracy(backdoored_model, X_clean_test, y_clean_test)

    removal_stats = removal_statistics(required_paths["defense_metadata"], remove_idx, row_limit=row_limit)
    metrics = {
        "train_rows_before": int(row_limit),
        "train_rows_after": int(y_defended.shape[0]),
        "removed_rows": int(remove_idx.shape[0]),
        **removal_stats,
        "watermarked_malware_eval_rows": int(X_watermarked_mw_eval.shape[0]),
        "defended_watermarked_malware_detection_rate": defended_detection_rate,
        "defended_asr": defended_asr,
        "defended_clean_accuracy": defended_clean_accuracy,
        "backdoored_watermarked_malware_detection_rate": backdoored_detection_rate,
        "backdoored_asr": backdoored_asr,
        "backdoored_clean_accuracy": backdoored_clean_accuracy,
    }
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    metadata = {
        "config": asdict(
            DefenseRetrainConfig(
                artifact_dir=str(artifact_path),
                gmm_dir=str(gmm_path) if gmm_path is not None else None,
                remove_idx_path=str(remove_path),
                output_dir=str(output_path),
                baseline=baseline,
                config_path=str(config_path) if config_path is not None else None,
                max_train_rows=max_train_rows,
                max_eval_rows=max_eval_rows,
                save_model=save_model,
            )
        ),
        "metrics": metrics,
        "clean_eval": clean_eval,
        "artifact_files": {key: str(path) for key, path in required_paths.items()},
        "defended_model_path": str(defended_model_path) if defended_model_path is not None else None,
        "backdoored_model_path": str(find_backdoored_model_path(artifact_path) or ""),
        "runtime_seconds": time.time() - start_time,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return DefenseRetrainResult(
        output_dir=str(output_path),
        metrics_path=str(metrics_path),
        metadata_path=str(metadata_path),
        defended_model_path=str(defended_model_path) if defended_model_path is not None else None,
        train_rows_before=int(row_limit),
        train_rows_after=int(y_defended.shape[0]),
        removed_rows=int(remove_idx.shape[0]),
        removed_poisoned_rows=metrics.get("removed_poisoned_rows"),
        removed_clean_rows=metrics.get("removed_clean_rows"),
        defended_clean_accuracy=defended_clean_accuracy,
        defended_asr=defended_asr,
        backdoored_clean_accuracy=backdoored_clean_accuracy,
        backdoored_asr=backdoored_asr,
    )


def resolve_remove_idx_path(remove_idx_path: str | Path | None, gmm_dir: Path | None) -> Path:
    if remove_idx_path is not None:
        resolved = resolve_path(remove_idx_path)
        return resolved or Path(remove_idx_path)
    if gmm_dir is None:
        raise ValueError("Either --gmm-dir or --remove-watermarked-idx must be provided")
    return gmm_dir / "remove_watermarked_idx.npy"


def resolve_output_dir(output_dir: str | Path | None, gmm_dir: Path | None, artifact_dir: Path) -> Path:
    if output_dir is not None:
        resolved = resolve_path(output_dir)
        return resolved or Path(output_dir)
    if gmm_dir is not None:
        return gmm_dir / "defended_retrain"
    return artifact_dir / "defended_retrain"


def load_remove_indices(remove_path: Path, n_rows: int, row_limit: int) -> np.ndarray:
    remove_idx = np.asarray(np.load(remove_path), dtype=np.int64).reshape(-1)
    remove_idx = np.unique(remove_idx)
    if np.any(remove_idx < 0) or np.any(remove_idx >= n_rows):
        raise ValueError(f"{remove_path} contains row ids outside watermarked_X rows [0, {n_rows})")
    remove_idx = remove_idx[remove_idx < row_limit]
    return remove_idx


def removal_statistics(defense_metadata_path: Path, remove_idx: np.ndarray, row_limit: int) -> dict:
    meta = np.load(defense_metadata_path)
    stats = {}
    if "poison_mask_full" in meta.files:
        poison_mask = np.asarray(meta["poison_mask_full"][:row_limit], dtype=bool)
        removed_mask = np.zeros(row_limit, dtype=bool)
        removed_mask[remove_idx] = True
        removed_poisoned = int(np.sum(removed_mask & poison_mask))
        removed_clean = int(np.sum(removed_mask & ~poison_mask))
        total_poisoned = int(np.sum(poison_mask))
        total_clean = int(poison_mask.shape[0] - total_poisoned)
        stats.update(
            {
                "total_poisoned_rows": total_poisoned,
                "total_clean_rows": total_clean,
                "removed_poisoned_rows": removed_poisoned,
                "removed_clean_rows": removed_clean,
                "poison_recall": float(removed_poisoned / total_poisoned) if total_poisoned else None,
                "clean_false_positive_rate": float(removed_clean / total_clean) if total_clean else None,
            }
        )
    return stats


def load_backdoored_model(artifact_dir: Path) -> lgb.Booster | None:
    model_path = find_backdoored_model_path(artifact_dir)
    if model_path is None:
        return None
    return lgb.Booster(model_file=str(model_path))


def find_backdoored_model_path(artifact_dir: Path) -> Path | None:
    candidates = sorted(path for path in artifact_dir.iterdir() if path.is_file() and path.name.endswith("_backdoored"))
    return candidates[0] if candidates else None


def predict_binary(model: lgb.Booster, X) -> np.ndarray:
    preds = np.asarray(model.predict(X))
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds = preds[:, 1]
    return (preds > 0.5).astype(np.int8)


def accuracy(model: lgb.Booster, X, y) -> float:
    return float(np.mean(predict_binary(model, X) == np.asarray(y).astype(np.int8)))


def _resolve_existing_dir(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("Expected a directory path")
    resolved = resolve_path(path)
    path_obj = resolved or Path(path)
    if not path_obj.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")
    return path_obj


def _unwrap_saved_array(value):
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value
