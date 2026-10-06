"""Create the separate Experiment 1 three-seed notebook, leaving older notebooks intact."""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source):
    item = {"id": f"three-seed-{len(cells):02d}", "cell_type": kind, "metadata": {}, "source": dedent(source).strip() + "\n"}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """
# Final Results: Seeds 42, 43, and 44

Experiment 1, original clean models, 1% poison rate, conservative trigger features,
eight selectors, Random and Distribution-based sampling, EMBER 2018 and EMBER 2024 Win64.
This notebook is independent of the previous notebooks. It exports deletion-safe scalar
snapshots and compares individual runs with three-run means and sample standard deviations
(`ddof=1`). The ASR/evasion column retains the existing result catalog definition.
Poison recall is reported separately for each sanitizer, not averaged across sanitizers.
""")
cell("code", """
from pathlib import Path
import hashlib
import json
import os
import sys
import itertools
import numpy as np
import pandas as pd
from IPython.display import display, FileLink

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == 'notebooks':
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'build' / 'matplotlib'))
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from src.analysis.result_catalog import load_attack_summaries, load_detector_metrics

RESULTS_ROOT = PROJECT_ROOT / 'results' / 'experiment 1'
OUT = RESULTS_ROOT / 'tables' / 'final_result_seed42_seed43_seed44' / 'poison_1p'
PLOTS = RESULTS_ROOT / 'plots' / 'final_result_seed42_seed43_seed44' / 'poison_1p'
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44]
SAMPLINGS = ['random', 'distribution_based_distance']
SELECTORS = ['min_population_new', 'argmin_Nv_sum_abs_shap', 'combined_shap',
             'low_shap_signed', 'frequency_bounded_signed_shap', 'corr_count_abs_shap',
             'benign_prototype', 'quantile_95']
LABELS = dict(zip(SELECTORS, ['MinPopulation*', 'CountAbsSHAP*', 'CombinedSHAP*',
                            'LowSignedSHAP', 'FreqBoundedSigned', 'CorrCountAbsSHAP',
                            'BenignPrototype', 'Quantile95']))
DATASETS = ['EMBER 2018', 'EMBER 2024 Win64']
SAMPLING_LABELS = dict(zip(SAMPLINGS, ['Random', 'Distribution-based']))
METHODS = ['hdbscan', 'spectral_signature', 'isolation_forest']
METHOD_LABELS = dict(zip(METHODS, ['HDBSCAN', 'Spectral Signature', 'Isolation Forest']))
SETTINGS = ['hdbscan_hybrid_top32_scaled_mcs0p5pct_ms0p1pct_tmax10pct_keep0p2',
            'spectral_signature_hybrid_top32_scaled_remove2p',
            'isolation_forest_hybrid_top32_scaled_contam0p02']
COLORS = dict(zip(SELECTORS, sns.color_palette('tab10', 8)))
sns.set_theme(style='whitegrid', context='notebook')
pd.set_option('display.max_rows', 120)
pd.set_option('display.max_columns', 40)
""")
cell("markdown", """
## Load and Preserve Matched Results

Seeds 43 and 44 are identified by the explicit seed-folder metadata; legacy seed-42
results are identified by their original dataset roots. Only the intended feature-selector
pair and matching full-pool sanitizer settings are retained. If live files are deleted,
the notebook falls back to the saved scalar CSV snapshots. No large NPY inputs are read.
""")
cell("code", """
ATTACK_KEYS = ['experiment_seed', 'dataset', 'sampling_strategy', 'value_selector']
DETECTOR_KEYS = ATTACK_KEYS + ['defense_method']

def prepare(frame, detector=False):
    out = frame.copy()
    if out.empty:
        return out
    out['experiment_seed'] = pd.to_numeric(out['seed'], errors='coerce')
    legacy = {'ember/20p_balanced': 'EMBER 2018', 'ember2024/win64': 'EMBER 2024 Win64'}
    out['dataset'] = out['dataset_path'].map({**legacy, 'ember2018': 'EMBER 2018',
                                           'ember2024_win64': 'EMBER 2024 Win64'})
    out.loc[out['dataset_path'].isin(legacy), 'experiment_seed'] = 42
    expected_feature = np.where(out['value_selector'].eq('combined_shap'),
                                'combined_shap', 'shap_largest_abs')
    mask = (out['experiment_seed'].isin(SEEDS) & out['dataset'].isin(DATASETS)
            & np.isclose(out['poison_rate_train'], 0.01, atol=1e-12)
            & out['sampling_strategy'].isin(SAMPLINGS)
            & out['target_features'].eq('problem_space_conservative')
            & out['value_selector'].isin(SELECTORS)
            & out['feature_selector'].eq(expected_feature))
    if detector:
        mask &= (out['defense_method'].isin(METHODS)
                 & out['defense_setting'].isin(SETTINGS)
                 & out['detector_max_benign_rows'].isna())
    out = out[mask].copy()
    out['selector'] = out['value_selector'].map(LABELS)
    out['sampling'] = out['sampling_strategy'].map(SAMPLING_LABELS)
    out['experiment_seed'] = out['experiment_seed'].astype(int)
    keys = DETECTOR_KEYS if detector else ATTACK_KEYS
    return out.sort_values('source_mtime').drop_duplicates(keys, keep='last').reset_index(drop=True)

def load_with_snapshot(loader, filename, detector=False):
    live = prepare(loader(RESULTS_ROOT), detector)
    snapshot = OUT / filename
    if snapshot.exists():
        saved = pd.read_csv(snapshot)
        live = pd.concat([saved, live], ignore_index=True)
        keys = DETECTOR_KEYS if detector else ATTACK_KEYS
        live = live.drop_duplicates(keys, keep='last').reset_index(drop=True)
    return live

seed_attack = load_with_snapshot(load_attack_summaries, 'seed_level_attack.csv')
seed_detector = load_with_snapshot(load_detector_metrics, 'seed_level_sanitizers.csv', True)
expected_attack = set(itertools.product(SEEDS, DATASETS, SAMPLINGS, SELECTORS))
expected_detector = set(itertools.product(SEEDS, DATASETS, SAMPLINGS, SELECTORS, METHODS))
actual_attack = set(seed_attack[ATTACK_KEYS].itertuples(index=False, name=None))
actual_detector = set(seed_detector[DETECTOR_KEYS].itertuples(index=False, name=None))
missing_attack = sorted(expected_attack - actual_attack)
missing_detector = sorted(expected_detector - actual_detector)
assert not missing_attack, f'Missing attack conditions: {missing_attack}'
assert not missing_detector, f'Missing sanitizer conditions: {missing_detector}'
assert np.isfinite(seed_attack[['attack_effectiveness_percent', 'backdoored_clean_accuracy_percent']].to_numpy()).all()
assert np.isfinite(seed_detector['poison_recall_percent']).all()
seed_attack.to_csv(OUT / 'seed_level_attack.csv', index=False)
seed_detector.to_csv(OUT / 'seed_level_sanitizers.csv', index=False)

# Preserve settings and provenance outside the folders containing large inputs.
metadata_dir = OUT / 'sanitizer_metadata'
metadata_dir.mkdir(exist_ok=True)
for _, row in seed_detector.iterrows():
    source = Path(row['detector_metadata_path'])
    if source.exists():
        name = f"{row['experiment_seed']}__{row['dataset']}__{row['sampling_strategy']}__{row['value_selector']}__{row['defense_method']}.json"
        (metadata_dir / name).write_text(source.read_text())
manifest = {'seeds': SEEDS, 'poison_rate': 0.01, 'attack_conditions': len(seed_attack),
            'sanitizer_conditions': len(seed_detector), 'std_ddof': 1,
            'sanitizer_settings': SETTINGS, 'large_inputs_required': False}
manifest['sha256'] = {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                      for name in ['seed_level_attack.csv', 'seed_level_sanitizers.csv']}
(OUT / 'snapshot_manifest.json').write_text(json.dumps(manifest, indent=2))
display(seed_attack.groupby(['experiment_seed', 'dataset', 'sampling']).size().rename('attack_conditions').reset_index())
display(seed_detector.groupby(['experiment_seed', 'dataset', 'sampling', 'defense_method']).size().rename('sanitizer_conditions').reset_index())
print('Validated:', len(seed_attack), 'attack conditions;', len(seed_detector), 'sanitizer conditions')
""")
cell("markdown", "## Individual Seed Results\n\nASR/evasion and backdoored clean-test accuracy are percentages. Each recall column is a separate sanitizer.\n")
cell("code", """
GROUP_KEYS = ['dataset', 'sampling_strategy', 'sampling', 'selector', 'value_selector']
wide_recall = seed_detector.pivot(index=ATTACK_KEYS, columns='defense_method', values='poison_recall_percent').reset_index()
seed_wide = seed_attack[ATTACK_KEYS + ['selector', 'sampling', 'attack_effectiveness_percent', 'backdoored_clean_accuracy_percent']].merge(wide_recall, on=ATTACK_KEYS, validate='one_to_one')
seed_wide.to_csv(OUT / 'seed_level_effectiveness_detectability.csv', index=False)
for dataset in DATASETS:
    display(seed_wide[seed_wide.dataset.eq(dataset)].sort_values(['sampling', 'selector', 'experiment_seed']).round(3))
""")
cell("markdown", """
## Three-Seed Mean and Standard Deviation

Only matched conditions with all three seeds contribute. Standard deviations describe
variation across these three runs, not confidence intervals. Units are percentage points.
* indicates Severi-derived selectors. Clean accuracy is the backdoored model's accuracy
on the unchanged test set, not its accuracy on triggered malware.
""")
cell("code", """
attack_stats = seed_attack.groupby(GROUP_KEYS).agg(
    seed_count=('experiment_seed', 'nunique'),
    asr_mean=('attack_effectiveness_percent', 'mean'), asr_std=('attack_effectiveness_percent', 'std'),
    clean_accuracy_mean=('backdoored_clean_accuracy_percent', 'mean'),
    clean_accuracy_std=('backdoored_clean_accuracy_percent', 'std')).reset_index()
detector_stats = seed_detector.groupby(GROUP_KEYS + ['defense_method']).agg(
    seed_count=('experiment_seed', 'nunique'),
    poison_recall_mean=('poison_recall_percent', 'mean'), poison_recall_std=('poison_recall_percent', 'std'),
    clean_fpr_mean=('clean_false_positive_rate_percent', 'mean'), clean_fpr_std=('clean_false_positive_rate_percent', 'std'),
    removed_poisoned_mean=('removed_poisoned_rows', 'mean'), removed_clean_mean=('removed_clean_rows', 'mean')).reset_index()
assert attack_stats.seed_count.eq(3).all() and detector_stats.seed_count.eq(3).all()
wide_stats = detector_stats.pivot(index=GROUP_KEYS, columns='defense_method', values=['poison_recall_mean', 'poison_recall_std']).reset_index()
wide_stats.columns = ['_'.join(filter(None, col)) if isinstance(col, tuple) else col for col in wide_stats.columns]
mean_std = attack_stats.merge(wide_stats, on=GROUP_KEYS, validate='one_to_one')
mean_std.to_csv(OUT / 'three_seed_mean_std_effectiveness_detectability.csv', index=False)
detector_stats.to_csv(OUT / 'three_seed_mean_std_sanitizers_long.csv', index=False)
for dataset in DATASETS:
    table = mean_std[mean_std.dataset.eq(dataset)].copy()
    table.to_csv(OUT / f"mean_std_{dataset.replace(' ', '_').lower()}.csv", index=False)
    display(table.round(3))
""")
cell("markdown", "## Figure: Mean Effectiveness Versus Detectability\n\nOne figure per sanitizer, two dataset panels, with sample standard-deviation error bars. Companion mean-only exports omit error bars for manuscript use. Markers remain at the actual means.\n")
cell("code", """
def save_figure(fig, stem):
    fig.savefig(PLOTS / f'{stem}.png', dpi=220, bbox_inches='tight')
    fig.savefig(PLOTS / f'{stem}.pdf', bbox_inches='tight')

legend_handles = [Line2D([], [], color=COLORS[s], marker='o', linestyle='', label=LABELS[s]) for s in SELECTORS]
legend_handles += [Line2D([], [], color='black', marker=m, linestyle='', label=s)
                   for s, m in [('Random', 'o'), ('Distribution-based', '^')]]
for method in METHODS:
    for show_sd in [True, False]:
        fig, axes = plt.subplots(2, 1, figsize=(10, 13), sharex=True, sharey=True)
        for ax, dataset in zip(axes, DATASETS):
            sub = mean_std[mean_std.dataset.eq(dataset)]
            for _, row in sub.iterrows():
                ax.errorbar(row[f'poison_recall_mean_{method}'], row.asr_mean,
                            xerr=row[f'poison_recall_std_{method}'] if show_sd else None,
                            yerr=row.asr_std if show_sd else None,
                            fmt='o' if row.sampling == 'Random' else '^',
                            color=COLORS[row.value_selector], markersize=10,
                            markeredgecolor='black', markeredgewidth=0.6, capsize=3, alpha=0.85)
            ax.set_title(f'{dataset} - {METHOD_LABELS[method]}', fontsize=17)
            ax.set_xlim(-4, 104); ax.set_ylim(-4, 104)
            ax.set_xlabel('Poison Recall (%)', fontsize=15)
            ax.set_ylabel('ASR / evasion (%)', fontsize=15)
            ax.tick_params(labelsize=13)
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=11)
        fig.text(0.5, 0.001, '* indicates Severi-derived selectors', ha='center', fontsize=10)
        fig.tight_layout(rect=[0, 0.115, 1, 1])
        save_figure(fig, f'asr_vs_{method}_recall_1p_three_seed_' + ('mean_std' if show_sd else 'mean'))
        if show_sd:
            plt.show()
        plt.close(fig)
""")
cell("markdown", "## Figure: Direct ASR and Poison-Recall Comparisons\n\nTwo sampling panels per dataset and sanitizer. Selectors are ranked by mean ASR within each panel. Black numbers show means; error bars show sample standard deviation.\n")
cell("code", """
for method in METHODS:
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharex=True)
        for ax, sampling in zip(axes, ['Random', 'Distribution-based']):
            sub = mean_std[mean_std.dataset.eq(dataset) & mean_std.sampling.eq(sampling)].sort_values('asr_mean', ascending=False)
            y = np.arange(len(sub))
            for shift, metric, sd, color, label in [
                (-0.18, 'asr_mean', 'asr_std', '#4C78A8', 'ASR / evasion'),
                (0.18, f'poison_recall_mean_{method}', f'poison_recall_std_{method}', '#F58518', 'Poison recall')]:
                ax.barh(y + shift, sub[metric], height=0.34, xerr=sub[sd], capsize=3, color=color, label=label)
                for pos, value, deviation in zip(y + shift, sub[metric], sub[sd]):
                    ax.text(value + deviation + 1, pos, f'{value:.1f}', va='center', fontsize=10, color='black')
            ax.set_yticks(y, sub.selector, fontsize=12)
            ax.invert_yaxis(); ax.set_xlim(0, 120)
            ax.set_title(sampling, fontsize=15)
            ax.set_xlabel('')
            ax.legend(loc='lower right', fontsize=11)
        fig.suptitle(f'{dataset} - {METHOD_LABELS[method]}: three-seed mean +/- SD', fontsize=17)
        fig.text(0.5, 0.01, '* indicates Severi-derived selectors; values in percent', ha='center', fontsize=11)
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        save_figure(fig, f"three_seed_bars_{dataset.replace(' ', '_').lower()}_{method}")
        plt.show(); plt.close(fig)
""")
cell("markdown", "## Heatmap: Three-Seed Mean ASR / Evasion\n")
cell("code", """
for dataset in DATASETS:
    sub = mean_std[mean_std.dataset.eq(dataset)]
    pivot = sub.pivot(index='sampling', columns='value_selector', values='asr_mean').reindex(index=['Random', 'Distribution-based'], columns=SELECTORS)
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu', vmin=0, vmax=100,
                annot_kws={'color': 'black'}, linewidths=0.5, ax=ax)
    ax.set_title(f'{dataset}: three-seed mean ASR / evasion at 1% poisoning', fontsize=15)
    ax.set_xticklabels([LABELS[s] for s in SELECTORS], rotation=35, ha='right')
    ax.set_xlabel(''); ax.set_ylabel('')
    save_figure(fig, f"three_seed_asr_heatmap_{dataset.replace(' ', '_').lower()}")
    plt.show(); plt.close(fig)
""")
cell("markdown", "## Downloads and Artifact Cleanup\n\nAll scalar snapshots and copied sanitizer settings below are outside `attack_artifacts`. After successful execution and coverage validation, large NPY inputs can be removed without breaking this notebook. Entire artifact folders can also be removed if no row-level analysis or retraining is needed: the snapshot fallback preserves these tables and figures. Keep the merged clean-model SHAP caches for future runs. No files are deleted by this notebook.\n")
cell("code", """
for path in sorted(OUT.glob('*.csv')):
    display(FileLink(str(path)))
display(FileLink(str(OUT / 'snapshot_manifest.json')))
print('Scalar results:', OUT)
print('PDF/PNG figures:', PLOTS)
""")

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12.0"}}, "nbformat": 4, "nbformat_minor": 5}
path = ROOT / "notebooks" / "08_final_result_seed42_seed43_seed44.ipynb"
path.write_text(json.dumps(notebook, indent=1) + "\n")
print(path)
