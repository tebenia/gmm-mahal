# Artifacts

Generated intermediate files go here, such as trained models, poisoned-set manifests, SHAP matrices, GMM fits, defense scores, and per-run metadata.

Large generated artifacts are ignored by Git by default.

Experiment 1's surviving EMBER2018 SHAP cache remains under
`shap_cache/ember2018`. Experiment 2 uses the seed-isolated layout documented in
`../docs/experiment_2_layout.md`:

- `experiment_2/models/<dataset>/seed_<seed>/clean_model.txt`
- `experiment_2/subsets/<dataset>/seed_<seed>/train_indices.npy`
- `experiment_2/shap_cache/<dataset>/seed_<seed>/`
- `experiment_2/value_selector_cache/<dataset>/seed_<seed>/`

The Experiment 2 directories are placeholders only. No model or SHAP data is
created by preparing the directory structure.
