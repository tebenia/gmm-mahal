# Experiment 2 Layout

Experiment 2 trains one local clean LightGBM model for each dataset and master
seed. The external datasets are read in place and are never copied into this
repository.

## External datasets

- EMBER2018: `../malwarebackdoors/ember2018`
- EMBER2024 Win64: `../ember2024/EMBER2024/data/Win64`

## Local generated artifacts

```text
artifacts/experiment_2/
├── subsets/
│   ├── ember2018/seed_{42,43,44}/train_indices.npy
│   └── ember2024_win64/seed_{42,43,44}/train_indices.npy
├── models/
│   ├── ember2018/seed_{42,43,44}/clean_model.txt
│   └── ember2024_win64/seed_{42,43,44}/clean_model.txt
├── shap_cache/
│   ├── ember2018/seed_{42,43,44}/
│   └── ember2024_win64/seed_{42,43,44}/
└── value_selector_cache/
    ├── ember2018/seed_{42,43,44}/
    └── ember2024_win64/seed_{42,43,44}/
```

Subset indices, model files, and SHAP caches are generated only by their
respective preparation, training, and SHAP commands. Value-selector caches are separated by seed
so one subset cannot silently reuse CountAbsSHAP values from another subset.

## Results

```text
results/experiment 2/
├── ember2018/
│   └── seed_{42,43,44}/
│       └── poison_rate_{0p01,0p03,0p05}/
│           ├── random/
│           └── distribution_based_distance/
└── ember2024_win64/
    └── seed_{42,43,44}/
        └── poison_rate_{0p01,0p03,0p05}/
            ├── random/
            └── distribution_based_distance/
```

Defense artifacts are created beside a sampling result directory using the
existing `<sampling>-defense/attack_artifacts/<experiment-name>` convention.
Sanitizer and defended-retraining outputs remain inside that attack-artifact
directory, preserving the complete provenance chain for one dataset, seed,
poison rate, sampling strategy, and selector.

## Configuration

`configs/experiment_2.yaml` contains the six dataset/seed baselines. Its
`partition_results_by_poison_rate: true` setting makes the runner expand 1%, 3%,
and 5% into separate output contexts before executing an attack. This prevents
one rate from overwriting another rate's large arrays and metadata.

The Python versions used for subset preparation and clean-model training are
pinned in `requirements-experiment2.txt`.

The EMBER2018 seed-42, seed-43, and seed-44 subset manifests and clean LightGBM
models have been generated. Their full-test accuracy values are 0.96483,
0.964695, and 0.96503; their ROC-AUC values are 0.993584006, 0.99351821305,
and 0.9939751776, respectively. Across the three seeds, clean accuracy is
0.96485167 +/- 0.00016855 and ROC-AUC is 0.99369247 +/- 0.00024704 using sample
standard deviation. The complete 20-chunk EMBER2018 seed-42, seed-43, and
seed-44 SHAP caches have been computed and merged into separate
120000 x 2381 float32 matrices.

The EMBER2024 Win64 seed-42, seed-43, and seed-44 subset manifests and clean
LightGBM models have also been generated. Each subset contains 208000 balanced
rows with 2568 features. Their full-test accuracy values are 0.98355,
0.98375833333, and 0.98335; their ROC-AUC values are 0.99847965736,
0.99844290444, and 0.99837634694, respectively. Across the three seeds, clean
accuracy is 0.98355278 +/- 0.00020418 and ROC-AUC is
0.99843297 +/- 0.00005237 using sample standard deviation. The complete
20-chunk Win64 seed-42, seed-43, and seed-44 SHAP caches have been computed and
merged into separate 208000 x 2568 float32 matrices.

For example, the EMBER2018 seed-42 stages before SHAP are:

```bash
python3 -m run_prepare_subset \
  --config configs/experiment_2.yaml \
  --baseline ember2018_20p_seed42

python3 -m run_train_clean_model \
  --config configs/experiment_2.yaml \
  --baseline ember2018_20p_seed42
```

The first command stores only source-row indices and metadata; it does not copy
the feature matrix. The second command tunes the dataset-level LightGBM profile
when it is absent, trains on the prepared subset, evaluates the full test split,
and stores the model, metrics, and provenance metadata.

Seed 42 is the designated Optuna tuning run for each dataset. Seeds 43 and 44
reuse the resulting dataset-level hyperparameters so seed comparisons do not
also change the model specification. Consequently, train seed 42 before the
other two seeds.

The ignored empty directories can be recreated at any time without training or
copying data:

```bash
python3 tools/prepare_experiment_2_layout.py
```
