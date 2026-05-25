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
value_selection: [combined_shap, min_population_new, argmin_Nv_sum_abs_shap, quantile_10]
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
  --value-selection combined_shap,min_population_new,argmin_Nv_sum_abs_shap,quantile_10 \
  --poison-rate 0.005,0.01 \
  --watermark-size 17,25 \
  --dry-run
```

Selector pairing follows the original notebook logic: combined selectors such as
`combined_shap` run as `combined_shap + combined_shap`, while non-combined
feature selectors such as `shap_largest_abs` are paired with each listed value
selector. Feature-only value names such as `combined_shap`, `combined_additive_shap`,
and `fixed` are skipped for non-combined feature selectors.

Quantile value selectors choose an observed training-set value at a fixed
empirical quantile for each selected feature. Available options are
`quantile_05`, `quantile_10`, `quantile_25`, `quantile_50`, `quantile_75`,
`quantile_90`, and `quantile_95`.

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
summaries, component geometry, per-row suspiciousness scores, model files, and
the benign-row / watermarked-row ids selected for removal. The geometry table
adds GMM weight, mean distance, covariance size, density proxies, log
likelihood, and responsibility confidence/entropy per component.

Component-guided trigger matching can use either score summaries, GMM geometry,
or feature-value enrichment to choose which components to mine:

```bash
python3 -m run_component_trigger_matching \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --gmm-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative/defense_preprocessing/standardized_pca50/gmm_defense/cov_diag_k1-20_reg1em06_remove1p \
  --component-rule density_proxy_log \
  --top-components 3 \
  --pair-apply-scope global \
  --row-rank matched_pairs \
  --removal-percent 1
```

Useful `--component-rule` ablations include `largest`, `density_proxy_log`,
`mean_global_mahalanobis`, `smallest_cov_volume`, `avg_log_likelihood`,
`responsibility_entropy_mean`, and `trigger_weighted_lift_sum`.

To run those ablations as one sweep:

```bash
python3 -m run_component_rule_sweep \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --gmm-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative/defense_preprocessing/standardized_pca50/gmm_defense/cov_diag_k1-20_reg1em06_remove1p \
  --top-components 3 \
  --pair-apply-scope global \
  --row-rank matched_pairs \
  --removal-percent 1
```

## OPTICS Iterative Defense

Run a paper-described reimplementation of "Model-agnostic clean-label backdoor
mitigation in cybersecurity environments" on a saved attack artifact:

```bash
python3 -m run_optics_defense \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --top-features 16 \
  --min-samples 50 \
  --window-fraction 0.05 \
  --clean-cluster-fraction 0.80 \
  --selection-mode fixed_threshold \
  --overwrite
```

This defense selects the top entropy/decision-tree features, clusters
benign-labeled rows with OPTICS, initializes the clean set with the largest
cluster plus all malware-labeled rows, iteratively adds the lowest-loss 5% of
clusters, and writes the suspicious cluster rows to `remove_watermarked_idx.npy`.
It is based on the paper description, not the authors' original code.
By default, outputs are written under
`<artifact-dir>/optics_iterative/all_top16_fixed_threshold_clean80p_w5p/`.

The filtering mode from the paper is the default. A loss-delta z-score variant
is also available:

```bash
python3 -m run_optics_defense \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --selection-mode loss_delta_z \
  --delta-z-threshold 2 \
  --delta-tail lower \
  --overwrite
```

Retrain from the OPTICS removal indices:

```bash
python3 -m run_defense_retrain \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --remove-watermarked-idx results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative/optics_iterative/all_top16_fixed_threshold_clean80p_w5p/remove_watermarked_idx.npy \
  --baseline ember2018_20p \
  --output-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative/optics_iterative/all_top16_fixed_threshold_clean80p_w5p/defended_retrain \
  --overwrite
```

## MDR-Inspired Defense

Run the MDR-inspired feature-value cleaning baseline on a saved attack artifact:

```bash
python3 -m run_mdr_defense \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__combined_shap__combined_shap__problem_space_conservative \
  --thresholds 3 4 5 6 7 8 9 10 11 12 \
  --dict-size 40 \
  --community-tolerance 0.8 \
  --remove-scope all \
  --overwrite
```

This is an MDR-inspired reimplementation from the paper description, not the
authors' original implementation. It builds SHAP-guided goodware-oriented
feature-value dictionaries, constructs thresholded intersection graphs, uses
Louvain communities to choose a suspicious community by malware-score reduction,
identifies enriched watermark-like feature-value elements, and writes
`remove_watermarked_idx.npy` for retraining.

Retrain from the MDR-inspired removal indices:

```bash
python3 -m run_defense_retrain \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__combined_shap__combined_shap__problem_space_conservative \
  --remove-watermarked-idx results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__combined_shap__combined_shap__problem_space_conservative/mdr_inspired/remove_watermarked_idx.npy \
  --baseline ember2018_20p \
  --output-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__combined_shap__combined_shap__problem_space_conservative/mdr_inspired/defended_retrain \
  --overwrite
```

Retrain a defended model after removing the suspicious benign rows selected by
GMM:

```bash
python3 -m run_defense_retrain \
  --artifact-dir results/ember2024/win64/random-defense/attack_artifacts/ember2024_win64__lightgbm__combined_shap__combined_shap__problem_space_conservative \
  --gmm-dir results/ember2024/win64/random-defense/attack_artifacts/ember2024_win64__lightgbm__combined_shap__combined_shap__problem_space_conservative/defense_preprocessing/standardized_pca50/gmm_defense/cov_diag_k1-10_reg1em06_remove1p \
  --baseline ember2024_win64_20p
```

This stage loads `watermarked_X.npy` / `watermarked_y.npy`, removes
`remove_watermarked_idx.npy`, retrains LightGBM, and evaluates clean accuracy
when `--baseline` is provided. It also evaluates ASR on `watermarked_X_test.npy`.
The outputs are written to `<gmm-dir>/defended_retrain/`.

For an oracle sanity check, remove the known poisoned rows instead of the GMM
selection:

```bash
python3 -m run_defense_retrain \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --gmm-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative/defense_preprocessing/standardized_pca50/gmm_defense/cov_diag_k1-10_reg1em06_remove1p \
  --baseline ember2018_20p \
  --oracle-remove-poisoned
```

This is not a deployable defense because it uses ground-truth poison labels.
It tells us whether ASR would drop if suspicious-row detection were perfect.

## Source Layout

```text
run_attack_baseline.py          CLI entry point
run_defense_preprocess.py       SHAP scaler/PCA preprocessing entry point
run_gmm_defense.py              GMM-BIC/Mahalanobis scoring entry point
run_optics_defense.py           OPTICS iterative filtering entry point
run_defense_retrain.py          defended retraining/evaluation entry point
src/
  run_attack_baseline.py        CLI implementation
  attack/                       poisoning attack pipeline
  data/                         dataset and model loaders
  defense/                      defense preprocessing and scoring utilities
  features/                     EMBER feature names and selector classes
  utils/                        path/config helpers and shared utilities
```
