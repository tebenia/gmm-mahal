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

The attack runner can expand a small experiment grid from either YAML/JSON config
or CLI overrides. These list fields are iterable:

```yaml
feature_selection: [combined_shap, shap_largest_abs]
value_selection: [combined_shap, min_population_new, argmin_Nv_sum_abs_shap]
sampling_strategies: [random, cosine_similarity]
poison_rates: [0.005, 0.01]
watermark_sizes: [17, 25]
```

The same grid can be overridden from the terminal:

```bash
python3 -m run_attack_baseline \
  --baseline ember2024_win64_20p \
  --sampling random,cosine_similarity \
  --feature-selection combined_shap,shap_largest_abs \
  --value-selection combined_shap,min_population_new,argmin_Nv_sum_abs_shap \
  --poison-rate 0.005,0.01 \
  --watermark-size 17,25 \
  --dry-run
```

Selector pairing follows the original notebook logic: combined selectors such as
`combined_shap` run as `combined_shap + combined_shap`, while non-combined
feature selectors such as `shap_largest_abs` are paired with each listed value
selector. Feature-only value names such as `combined_shap`, `combined_additive_shap`,
and `fixed` are skipped for non-combined feature selectors.

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

For an even stricter ablation, use `--target-features severi_exact_overlap`.
This keeps only the Severi 17 on EMBER2018 and only the exact-name subset that
survives in EMBER2024. See `docs/severi_ember2024_feature_mapping.md` for the
semantic mapping.

To prepare defense inputs from the backdoored model, add:

```bash
python3 -m run_attack_baseline --baseline ember2024_win64_20p --save-defense-inputs
```

This saves poisoned-row indices, benign-row indices, poison masks, and LightGBM `pred_contrib=True` SHAP values for benign-labeled poisoned-training rows under the experiment's attack-artifact directory. Add `--save-attack-artifacts` as well if you also want full `watermarked_X.npy`, `watermarked_y.npy`, the watermarked test set, and the backdoored model file.

## Defense Preprocessing

Preprocess an attack artifact's benign SHAP matrix before fitting GMMs:

```bash
python3 -m run_defense_preprocess \
  --artifact-dir results/ember2024/win64/random-defense/attack_artifacts/ember2024_win64__lightgbm__combined_shap__combined_shap__problem_space_conservative
```

The default defense representation is StandardScaler plus fixed 50-component
IncrementalPCA. It writes `X_shap_reduced.npy`, `standard_scaler.joblib`,
`pca.joblib`, and `preprocessing_metadata.json` under
`<artifact-dir>/defense_preprocessing/standardized_pca50/`. Use
`--pca-components 100` for a larger fixed representation, or `--no-pca` /
`--no-standardize` for ablations.

Run GMM-BIC/Mahalanobis scoring on the preprocessed representation:

```bash
python3 -m run_gmm_defense \
  --preprocess-dir results/ember2024/win64/random-defense/attack_artifacts/ember2024_win64__lightgbm__combined_shap__combined_shap__problem_space_conservative/defense_preprocessing/standardized_pca50
```

The default GMM grid uses `covariance_type=diag`, `K=1..10`, `reg_covar=1e-6`,
and removes the top 1% by cluster-wise local Mahalanobis z-score. It also fits a
global `K=1` Mahalanobis baseline. Outputs include BIC scores, component
summaries, per-row suspiciousness scores, model files, and the benign-row /
watermarked-row ids selected for removal.

## Source Layout

```text
run_attack_baseline.py          CLI entry point
run_defense_preprocess.py       SHAP scaler/PCA preprocessing entry point
run_gmm_defense.py              GMM-BIC/Mahalanobis scoring entry point
src/
  run_attack_baseline.py        CLI implementation
  attack/                       poisoning attack pipeline
  data/                         dataset and model loaders
  defense/                      defense preprocessing and scoring utilities
  features/                     EMBER feature names and selector classes
  utils/                        path/config helpers and shared utilities
```
