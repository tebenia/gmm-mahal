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
ember2018_20p_legacy
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

`ember2018_20p` now follows the EMBER2024-style cache mechanism: the source
dataset remains in the original EMBER2018 folder, while the selected train rows
are stored in an `indices_*.npy` file and clean-model SHAP values are stored in
a matching `shap_values_*.pkl` file. The default EMBER2018 subset is balanced
and reproducible with `subset_mode: balanced_stratified_random` and `seed: 42`.
The old chunk-prefix materialized dataset is still available as
`ember2018_20p_legacy`.

Build or inspect a SHAP cache for a baseline:

```bash
python3 -m run_compute_shap_cache --baseline ember2018_20p --dry-run
python3 -m run_compute_shap_cache --baseline ember2018_20p --num-chunks 20 --chunk-index 0
python3 -m run_compute_shap_cache --baseline ember2018_20p --num-chunks 20 --merge-only
```

Run one chunk index at a time for large SHAP jobs, then merge after all chunks
exist. The generated cache is ignored by Git under `artifacts/shap_cache/`.

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

Accepted `--sampling` values:

```text
random
adaptive

feature_based_distance
distribution_based_distance
score_wasserstein_distance
score_kde_density_ratio
shap_contribution_distance

mahalanobis_distance
cosine_similarity
jaccard_distance
wasserstein_distance
```

`distribution_based_distance` is the original score-space sampler: it computes
KDE overlap for diagnostics, then selects benign rows nearest to the malware
mean score. `score_wasserstein_distance` is also score-space, but compares each
benign score to the empirical malware score distribution with 1D Wasserstein
distance, equivalent to `mean(abs(benign_score - malware_scores))`.
`score_kde_density_ratio` selects benign rows whose clean-model scores have high
malware-score density relative to benign-score density.

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

To run the same trigger-like feature-value mining without GMM, omit `--gmm-dir`.
This treats all benign-labeled rows as one pseudo-component, so use
`--component-rule all` or `--component-rule largest`:

```bash
python3 -m run_component_trigger_matching \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --component-rule all \
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

## HDBSCAN SHAP-Loss Defense

Run the paper-style SHAP-space clustering and loss-ranked filtering defense with
HDBSCAN. It uses the saved SHAP/PCA representation from `run_defense_preprocess`
as the clustering space. This command requires the optional Python package
`hdbscan` for a full run; `--dry-run` can still validate paths without it:

```bash
python3 -m run_defense_preprocess \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative

python3 -m run_hdbscan_shap_loss_defense \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --clean-fraction 0.80 \
  --coverage-unit clusters \
  --min-cluster-percent 0.5 \
  --min-samples-percent 0.1 \
  --overwrite
```

This defense clusters benign-labeled rows in `X_shap_reduced.npy`, trains a
surrogate model on the largest benign cluster plus malware rows, scores clusters
by benign log-loss, keeps the lowest-loss 80%, and writes the remaining benign
rows to `remove_watermarked_idx.npy`. Poison labels are used only in the saved
diagnostic columns, not to choose the removed rows.

Retrain from the HDBSCAN SHAP-loss removal indices:

```bash
python3 -m run_defense_retrain \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --remove-watermarked-idx results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative/hdbscan_shap_loss/clusters_clean80p_mcs0p5pct_ms0p1pct_noisesplit/remove_watermarked_idx.npy \
  --baseline ember2018_20p \
  --output-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative/hdbscan_shap_loss/clusters_clean80p_mcs0p5pct_ms0p1pct_noisesplit/defended_retrain \
  --overwrite
```

Use `--coverage-unit rows` if you want the 80% threshold to mean retained
benign rows instead of retained clusters. Use `--dry-run` to validate paths
without importing or running HDBSCAN.

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

## Notebook-Style Isolation Forest, Spectral Signature, and HDBSCAN

Run the Isolation Forest, Spectral Signature, or HDBSCAN detectors ported from
the Severi defense code and the `backdoor_codex_*` notebook defense cells:

```bash
python3 -m run_severi_defense \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --method isolation_forest \
  --feature-mode hybrid \
  --top-k 32 \
  --contamination 0.005 \
  --overwrite
```

```bash
python3 -m run_severi_defense \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --method spectral_signature \
  --feature-mode hybrid \
  --top-k 32 \
  --removal-percent 1 \
  --overwrite
```

```bash
python3 -m run_severi_defense \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --method hdbscan \
  --feature-mode hybrid \
  --top-k 32 \
  --hdbscan-min-cluster-percent 0.5 \
  --hdbscan-min-samples-percent 0.1 \
  --hdbscan-threshold-max-percent 10 \
  --hdbscan-min-keep 0.2 \
  --overwrite
```

The feature subspace follows the old notebook idea: `watermark` uses the actual
watermark feature ids from `wm_config.npy`, `shap` uses the highest mean-absolute
backdoored-model SHAP features, and `hybrid` starts with watermark features then
pads with SHAP features up to `--top-k`. The detector writes
`suspicious_scores.csv`, `selected_features.csv`, and `remove_watermarked_idx.npy`
under `<artifact-dir>/severi_detectors/<settings>/`.

`--spectral-oracle-poison-count` matches the old notebook's Spectral Signature
default by removing the known number of poisoned benign rows. Treat that as a
diagnostic budget because it uses ground-truth poison metadata. For a non-oracle
run, prefer an explicit `--removal-percent`.

The HDBSCAN path follows Severi's original `defense_filtering.py` idea more than
a simple "remove density noise" rule: it clusters selected benign-labeled rows,
computes cluster-average silhouettes, and removes rows from small high-silhouette
clusters according to the `--hdbscan-min-keep` sampling rule. It also saves
`hdbscan_labels.npy` and `hdbscan_cluster_summary.csv` so the clusters can be
inspected. If `hdbscan` is not installed, install the optional package before
running this method.

Retrain from either detector's removal indices:

```bash
python3 -m run_defense_retrain \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --remove-watermarked-idx results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative/severi_detectors/spectral_signature_hybrid_top32_scaled_remove1p/remove_watermarked_idx.npy \
  --baseline ember2018_20p \
  --output-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative/severi_detectors/spectral_signature_hybrid_top32_scaled_remove1p/defended_retrain \
  --overwrite
```

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

## DUBIOUS-Inspired Test-Time Detection

Run a DUBIOUS-style input rejection diagnostic on saved attack artifacts:

```bash
python3 -m run_dubious_defense \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --baseline ember2018_20p \
  --magnitudes 10,20,30,40,50 \
  --n-perturbations 100 \
  --replacement benign_mean \
  --feature-mode random \
  --overwrite
```

This is a test-time detector, not a training-row sanitizer. It perturbs clean
reference samples and watermarked malware samples, builds signatures from
LightGBM raw-score mean, raw-score standard deviation, and prediction stability,
then compares each test signature with clean signatures of the same predicted
class using scaled L1 nearest-neighbor distance. The default `random` feature
mode and `benign_mean` replacement follow the DUBIOUS PDF-malware experiment
most closely: selected features are random, and their values are replaced with
average benign values.

Useful ablations:

```bash
python3 -m run_dubious_defense \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --baseline ember2018_20p \
  --feature-mode shap_topk \
  --top-k 50 \
  --score-mode apc_l1 \
  --replacement benign_mean \
  --overwrite
```

Outputs are written under `<artifact-dir>/dubious_signatures/<settings>/`:
`reference_signatures.csv`, `eval_signatures.csv`, `dubious_scores.csv`,
`dubious_metrics.csv`, and `dubious_metadata.json`. The main metrics are
watermarked detection rate, false positive rate on clean benign/malware,
`detection_minus_max_fpr`, and ASR before/after rejection.

## Detectability Diagnostics

Run artifact-level detectability diagnostics on a saved attack artifact:

```bash
python3 -m run_detectability_diagnostics \
  --artifact-dir results/ember/20%/random-defense/attack_artifacts/ember__lightgbm__shap_largest_abs__min_population_new__problem_space_conservative \
  --overwrite
```

Batch example for every saved EMBER2018 random artifact:

```bash
python3 -m run_detectability_diagnostics \
  --artifact-glob "results/ember/20%/random-defense/attack_artifacts/*" \
  --overwrite
```

This stage does not remove rows or retrain a model. It measures non-defense
detectability signals from existing artifacts: marginal trigger-value rarity,
joint trigger co-occurrence rarity, trigger-SHAP concentration, and a kNN
trigger-subspace density proxy. Ground-truth poison labels from
`defense_metadata.npz` are used only for evaluation metrics such as AUROC,
average precision, and recall-at-budget.

Outputs are written under `<artifact-dir>/detectability_diagnostics/<settings>/`:
`detectability_summary.csv`, `trigger_marginal_rarity.csv`,
`trigger_joint_rarity.csv`, `detectability_score_metrics.csv`,
`detectability_row_scores.csv`, and `detectability_metadata.json`.

Use `notebooks/07_detectability_diagnostics.ipynb` to aggregate these outputs,
list artifact folders that still need diagnostics, and compare detectability
against attack effectiveness from the saved `summary_df.csv` files.

## Source Layout

```text
run_attack_baseline.py          CLI entry point
run_defense_preprocess.py       SHAP scaler/PCA preprocessing entry point
run_gmm_defense.py              GMM-BIC/Mahalanobis scoring entry point
run_hdbscan_shap_loss_defense.py HDBSCAN SHAP-space loss-ranked filtering entry point
run_severi_defense.py           Isolation Forest / Spectral Signature detector entry point
run_dubious_defense.py          DUBIOUS-style test-time detector entry point
run_detectability_diagnostics.py Trigger rarity / co-occurrence / SHAP diagnostics entry point
run_defense_retrain.py          defended retraining/evaluation entry point
src/
  run_attack_baseline.py        CLI implementation
  run_detectability_diagnostics.py CLI implementation for detectability diagnostics
  attack/                       poisoning attack pipeline
  analysis/                      artifact-level comparison and detectability utilities
  data/                         dataset and model loaders
  defense/                      defense preprocessing and scoring utilities
  features/                     EMBER feature names and selector classes
  utils/                        path/config helpers and shared utilities
```
