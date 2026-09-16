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

Model files and SHAP caches are intentionally absent until their respective
training and SHAP commands are run. Value-selector caches are separated by seed
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

No Experiment 2 model has been trained and no Experiment 2 SHAP cache has been
computed yet.

The ignored empty directories can be recreated at any time without training or
copying data:

```bash
python3 tools/prepare_experiment_2_layout.py
```
