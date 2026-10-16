# Ground-dwelling tiger beetle classification — revised pipeline

Re-run of the machine-learning classification study, revised to address the
reviewer comments (location leakage, SMOTE on categoricals, scaling,
reproducibility, numeric consistency, feature-group ablation, per-class loss
reporting).

## What it does

Two evaluation tracks, both reported side by side:

- **Track A — specimen-level** (reproduces the manuscript): stratified 75:25
  split + stratified 5-fold CV for tuning, scored once on the held-out test set.
- **Track B — location-independent**: a *site-disjoint* 75:25 split (the test
  set contains only sites absent from training) + StratifiedGroupKFold for
  tuning, scored on unseen sites. This is the honest "unseen-location" estimate.

The two best models are selected by **kappa on the Track A held-out test**, and
**both** models then run through **all** downstream analyses on **both** tracks.

Leakage-safe by construction: scaling, **SMOTE-NC** (categorical-aware
oversampling), and one-hot encoding are all fit inside each training fold only.

## Files

| file | purpose |
|------|---------|
| `config.py` | all settings: paths, feature roles, site definition, model grids, repeats, plotting. **Start here to change anything.** |
| `data_prep.py` | load, de-duplicate, per-class loss report, build `site_id` |
| `samplers.py` | adaptive SMOTE-NC / SMOTE (safe for tiny minority classes) |
| `pipelines.py` | leakage-safe model pipelines |
| `evaluation.py` | both tracks, repeats, timing, top-2 selection |
| `feature_analysis.py` | per-class table, permutation importance, ablation |
| `cooccurrence.py` | co-occurrence, Shapiro/Pearson, variance decomposition |
| `plotting.py` | all figures (600 dpi, two-panel, uppercase panel letters) |
| `run_all.py` | orchestrator — run this |

## How to run

```bash
pip install -r requirements.txt

# full run (10 repeats)
TB_DATA=/path/to/Ground-dwelling_tiger_beetle_dataset.csv python run_all.py

# fast smoke test (tiny grids, 2 repeats)
TB_QUICK=1 TB_DATA=/path/to/dataset.csv python run_all.py
```

Outputs land in `outputs/figures/` (PNG @ 600 dpi) and `outputs/tables/` (CSV).

## Output map (to manuscript tables/figures)

| output | manuscript |
|--------|-----------|
| `class_distribution_dedup.csv` | per-class records lost in de-duplication (reviewer comment 5) |
| `table3_trackA_specimen_test.csv` | Table 3 (held-out test) |
| `table3B_trackB_unseen_site_test.csv` | **new** — unseen-location performance |
| `model_comparison_full.csv` | all models × both tracks, mean ± std, + modal best hyperparameters |
| `best_hyperparameters_per_repeat.csv` | the hyperparameters chosen in every repeat (reproducibility) |
| `best_hyperparameters_downstream.csv` | hyperparameters of the two models used in the figures |
| `table7_perclass_<track>_<model>.csv` | Table 7 (per-class, from OOF CV) |
| `table5_shapiro_*`, `table6_pearson_*` | Tables 5 & 6 |
| `table8_cooccurrence.csv` | Table 8 |
| `fig3_single_importance_*` | Fig 3 |
| `fig4_perm_schemes_*` | Fig 4 |
| `fig5_block_importance_*` | Fig 5 |
| `fig6_f1_vs_cooccurrence_*` | Fig 6 |
| `fig7_variance_decomposition.png` | Fig 7 |
| `ablation_*` | **new** — feature-group ablation |

## Key defaults (all in `config.py`)

- Top-2 selected on **Track A** held-out kappa.
- `env-without-GPS` ablation drops latitude & longitude (keeps elevation).
- Timing = **classifier only** (consistent with manuscript Table 3).
- 10 repeats, split re-drawn each repeat; std = spread across repeats.
- Pearson (not Spearman) for the co-occurrence correlation, justified by the
  Shapiro-Wilk normality test the code reports.

## Note on results

Absolute timings depend on the machine. The **Track A vs Track B gap** is the
important finding: scores are high within sampled sites (Track A) but drop
sharply on unseen sites (Track B), confirming that the environmental variables
act largely as a site fingerprint. Report both tracks and interpret feature
importance as association, not causation.
