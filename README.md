# Defense GMM-Maha

Experiment harness for SHAP-space GMM-Mahalanobis defense work connected to the EMBER2018 and EMBER2024 poisoning experiments.

The attack and defense code lives in this repository. Large datasets, saved models, and cached SHAP values are referenced by configurable paths instead of being duplicated.

## Attack Baseline Runner

Run the notebook-equivalent data poisoning attack from Python files:

```bash
python3 -m run_attack_baseline --baseline ember2024_win64_20p
```

Available baselines:

```text
ember2018_20p
ember2024_win64_20p
ember2024_win32_06p
```

Examples:

```bash
python3 -m run_attack_baseline --baseline ember2024_win32_06p --sampling random
python3 -m run_attack_baseline --baseline ember2018_20p --sampling wasserstein_distance
python3 -m run_attack_baseline --baseline ember2024_win64_20p --dry-run
```

By default, `configs/attack_baselines.yaml` points at the current local EMBER2018/EMBER2024 dataset, model, and SHAP cache locations. You can move those assets and edit the YAML paths without changing the code. New attack summary CSVs are written under this repository's `results/` tree. Use `--save-attack-artifacts` when you also need the large watermarked arrays and backdoored model for defense experiments.

The default target feature group is `feature_space_feasible`. This is a
Severi-style feature-vector candidate set: non-hashed features minus configured
exclusions. It does not by itself prove that the same trigger can be edited into
a real PE binary without changing functionality. The old name `feasible` is kept
as a compatibility alias for older configs and artifact paths.

For stricter comparison runs, use `--target-features problem_space_conservative`.
This is a smaller heuristic candidate set that excludes hashed bins, histograms,
byte-entropy bins, PE warning flags, Authenticode fields, Rich-header hashes,
checksums, and data-directory fields. It is intentionally conservative, but it is
still not a proof of PE editability unless the binaries are actually modified and
features are re-extracted.

To prepare defense inputs from the backdoored model, add:

```bash
python3 -m run_attack_baseline --baseline ember2024_win64_20p --save-defense-inputs
```

This saves poisoned-row indices, benign-row indices, poison masks, and LightGBM `pred_contrib=True` SHAP values for benign-labeled poisoned-training rows under the experiment's attack-artifact directory. Add `--save-attack-artifacts` as well if you also want full `watermarked_X.npy`, `watermarked_y.npy`, the watermarked test set, and the backdoored model file.

## Source Layout

```text
run_attack_baseline.py          CLI entry point
src/
  run_attack_baseline.py        CLI implementation
  attack/                       poisoning attack pipeline
  data/                         dataset and model loaders
  features/                     EMBER feature names and selector classes
  utils/                        path/config helpers and shared utilities
```
