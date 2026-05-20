# Defense GMM-Maha

Experiment harness for SHAP-space GMM-Mahalanobis defense work connected to the EMBER2018 and EMBER2024 poisoning experiments.

The attack and defense code lives in this repository. Large datasets, saved models, and cached SHAP values are referenced by configurable paths instead of being duplicated.

## Attack Baseline Runner

Run the notebook-equivalent data poisoning attack from Python files:

```bash
python3 -m src.run_attack_baseline --baseline ember2024_win64_20pct
```

Available baselines:

```text
ember2018_20pct
ember2024_win64_20pct
ember2024_win32_0p0667
```

Examples:

```bash
python3 -m src.run_attack_baseline --baseline ember2024_win32_0p0667 --sampling random
python3 -m src.run_attack_baseline --baseline ember2018_20pct --sampling wasserstein_distance
python3 -m src.run_attack_baseline --baseline ember2024_win64_20pct --dry-run
```

By default, `configs/attack_baselines.yaml` points at the current local EMBER2018/EMBER2024 dataset, model, and SHAP cache locations. You can move those assets and edit the YAML paths without changing the code. New attack summary CSVs are written under this repository's `results/` tree. Use `--save-attack-artifacts` when you also need the large watermarked arrays and backdoored model for defense experiments.

## Source Layout

```text
src/
  run_attack_baseline.py        CLI entry point
  attack/                       poisoning attack pipeline
  data/                         dataset and model loaders
  features/                     EMBER feature names and selector classes
  utils/                        path/config helpers and shared utilities
```
