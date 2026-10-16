"""
cooccurrence.py
===============
Co-occurrence analysis and the within/between-site variance decomposition.

  * co_occurrence_table         -> Table 8
  * correlation(per_class_f1)   -> Shapiro normality (Table 5) + Pearson (Table 6)
                                    + scatter data (Fig 6). Uses per-class F1 from
                                    the chosen model/track (cross-validated OOF).
  * variance_decomposition      -> Fig 7 (model-agnostic; computed once).

Pearson is used (not Spearman) because both variables pass the Shapiro-Wilk
normality test; the code reports the test so the choice is justified in the
manuscript (reviewer comment 4). Correlations are associations, not causation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import config as C


def co_occurrence(df: pd.DataFrame, site_id: np.ndarray) -> pd.Series:
    """Number of OTHER species sharing the sites occupied by each species."""
    tmp = pd.DataFrame({"species": df[C.TARGET].to_numpy(), "site": site_id})
    species_by_site = tmp.groupby("site")["species"].apply(set)
    out = {}
    for sp in sorted(tmp["species"].unique()):
        sites = tmp.loc[tmp["species"] == sp, "site"].unique()
        others = set()
        for s in sites:
            others |= species_by_site[s]
        others.discard(sp)
        out[sp] = len(others)
    return pd.Series(out, name="co_occurrence").sort_values()


def correlation(per_class_f1: pd.Series, cooc: pd.Series):
    """Return (shapiro_df, pearson_df, scatter_df)."""
    joined = pd.DataFrame({
        "f1": per_class_f1, "co_occurrence": cooc}).dropna()

    sh_f1 = stats.shapiro(joined["f1"])
    sh_co = stats.shapiro(joined["co_occurrence"])
    shapiro_df = pd.DataFrame({
        "variable": ["Per class F1 score", "Co-occurrence"],
        "statistic": [sh_f1.statistic, sh_co.statistic],
        "p_value": [sh_f1.pvalue, sh_co.pvalue],
    })

    r, p = stats.pearsonr(joined["co_occurrence"], joined["f1"])
    pearson_df = pd.DataFrame({
        "variables": ["Per class F1 (dependent), Co-occurrence (independent)"],
        "pearson_r": [r], "p_value": [p],
    })
    return shapiro_df, pearson_df, joined.reset_index().rename(columns={"index": "species"})


def variance_decomposition(df: pd.DataFrame, site_id: np.ndarray):
    """Fig 7 data.

    Returns (within_frac, var_partition, meta) where:
      within_frac   : for each feature, fraction of multi-specimen sites in
                      which it varies (categorical + continuous).
      var_partition : for each CONTINUOUS feature, within-site share of total
                      variance.
      meta          : dict with n multi-specimen sites / specimens.
    """
    d = df.copy()
    d["_site"] = site_id
    counts = d["_site"].value_counts()
    multi = counts[counts > 1].index
    dm = d[d["_site"].isin(multi)]
    meta = {"n_multi_sites": int(len(multi)), "n_specimens": int(len(dm))}

    feats = C.CATEGORICAL + C.CONTINUOUS
    within = {}
    for f in feats:
        varies = dm.groupby("_site")[f].nunique() > 1
        within[f] = float(varies.mean())
    within_frac = pd.Series(within, name="within_site_fraction")

    # variance partition for continuous features (between vs within site)
    part = {}
    for f in C.CONTINUOUS:
        grand = dm[f].var(ddof=0)
        if grand == 0:
            part[f] = 0.0
            continue
        site_means = dm.groupby("_site")[f].transform("mean")
        within_var = (dm[f] - site_means).var(ddof=0)
        part[f] = float(within_var / grand) if grand > 0 else 0.0
    var_partition = pd.Series(part, name="within_site_share")
    return within_frac, var_partition, meta
