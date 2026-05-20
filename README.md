# Defense GMM-Maha

Experiment harness for SHAP-space GMM-Mahalanobis defense work connected to the existing EMBER2024 and Severi data poisoning attack projects.

This repository is intended to reference existing datasets, models, poisoned artifacts, SHAP caches, and defense results rather than duplicate large files.

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

By default, source datasets, saved models, and cached SHAP values stay in the sibling Severi/EMBER folders. New attack summary CSVs are written under this repository's `results/` tree. Use `--save-attack-artifacts` when you also need the large watermarked arrays and backdoored model for defense experiments.
