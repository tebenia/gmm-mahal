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
value_selection: [combined_shap, min_population_new, argmin_Nv_sum_abs_shap, quantile_10, benign_prototype, low_shap_signed, frequency_bounded, frequency_bounded_signed_shap, corr_count_abs_shap]
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
  --value-selection combined_shap,min_population_new,argmin_Nv_sum_abs_shap,quantile_10,benign_prototype,low_shap_signed,frequency_bounded,frequency_bounded_signed_shap,corr_count_abs_shap \
  --poison-rate 0.005,0.01 \
  --watermark-size 17,25 \
  --dry-run
```

Selector pairing follows the original notebook logic: combined selectors such as
`combined_shap` run as `combined_shap + combined_shap`, while non-combined
feature selectors such as `shap_largest_abs` are paired with each listed value
selector. Feature-only value names such as `combined_shap`, `combined_additive_shap`,
and `fixed` are skipped for non-combined feature selectors.

Accepted `--value-selection` values:

```text
min_population_new
argmin_Nv_sum_abs_shap

combined_shap
combined_additive_shap
fixed

quantile_05
quantile_10
quantile_25
quantile_50
quantile_75
quantile_90
quantile_95

benign_prototype
benign_prototype_median

low_shap_signed
signed_shap_min
signed_shap_min_mean
signed_shap_min_sum

frequency_bounded
freq_0p1_1p
freq_0p1_5p
freq_0p5_5p
freq_1p_10p

frequency_bounded_signed_shap
freq_signed_0p1_1p
freq_signed_0p1_5p
freq_signed_0p5_5p
freq_signed_1p_10p

corr_count_abs_shap
corr_count_abs_shap_min10
corr_count_abs_shap_min50
corr_count_abs_shap_min100
```

Quantile value selectors choose an observed training-set value at a fixed
empirical quantile for each selected feature. Available options are
`quantile_05`, `quantile_10`, `quantile_25`, `quantile_50`, `quantile_75`,
`quantile_90`, and `quantile_95`.

`benign_prototype` copies all selected feature values from one real benign
training row. The current rule chooses the benign row closest to the
coordinate-wise benign median in the selected feature subspace. This preserves
an observed benign feature-value combination instead of combining each feature's
value independently. `benign_prototype_median` is an equivalent explicit alias.

`low_shap_signed` chooses observed values whose signed SHAP contribution is most
negative on average, meaning most benign-directional for the current binary
malware model. `signed_shap_min` and `signed_shap_min_mean` are aliases.
`signed_shap_min_sum` is a frequency-weighted variant that uses the total signed
SHAP over rows with each value.

`frequency_bounded` chooses the least frequent observed value whose count is
within a configured frequency band. The default band is 0.1%-5% of training
rows. Additional bands are `freq_0p1_1p`, `freq_0p1_5p`, `freq_0p5_5p`, and
`freq_1p_10p`. If no value falls inside the band, the selector uses the observed
value whose count is closest to the band.

`frequency_bounded_signed_shap` first restricts candidate values to the same
frequency band idea, then chooses the value with the most negative mean signed
SHAP. Additional bands are `freq_signed_0p1_1p`, `freq_signed_0p1_5p`,
`freq_signed_0p5_5p`, and `freq_signed_1p_10p`.

`corr_count_abs_shap` is a correlation-preserving CountAbsSHAP variant. It
selects values greedily on benign rows that still match previously selected
trigger values, preferring CountAbsSHAP-style low `1/count + sum(abs(SHAP))`
values only when the partial trigger keeps at least 10 benign rows. Variants
`corr_count_abs_shap_min50` and `corr_count_abs_shap_min100` require more
co-occurrence support.

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
