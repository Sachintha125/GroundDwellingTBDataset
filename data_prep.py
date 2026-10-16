"""
data_prep.py
============
Load the raw dataset, clean it, and expose the pieces the rest of the
pipeline needs: feature matrix X, target y, and a site_id vector used only
for location-independent / grouped evaluation.

Reviewer-relevant outputs saved here:
  * class_distribution_dedup.csv  -- per-class counts before/after de-duplication
    and the number of records lost from each class (reviewer comment 5).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C


def load_raw() -> pd.DataFrame:
    """Read the CSV and rename columns strictly by position to clean names."""
    df = pd.read_csv(C.DATA_PATH, header=0)
    if df.shape[1] != len(C.CLEAN_COLUMNS):
        raise ValueError(
            f"Expected {len(C.CLEAN_COLUMNS)} columns, found {df.shape[1]}: "
            f"{list(df.columns)}")
    df.columns = C.CLEAN_COLUMNS
    return df


def clean(df: pd.DataFrame):
    """Remove exact duplicate rows and records with missing values.

    Returns (clean_df, report_df) where report_df documents the per-class
    reduction caused by de-duplication and missing-value removal.
    """
    before = df[C.TARGET].value_counts().sort_index()
    n_before = len(df)

    # 1) missing values
    n_missing = int(df.isna().any(axis=1).sum())
    df_nm = df.dropna(axis=0, how="any")

    # 2) exact duplicate rows
    n_dup = int(df_nm.duplicated().sum())
    df_clean = df_nm.drop_duplicates().reset_index(drop=True)

    after = df_clean[C.TARGET].value_counts().sort_index()

    report = (
        pd.DataFrame({"count_before": before, "count_after": after})
        .fillna(0).astype(int))
    report["lost"] = report["count_before"] - report["count_after"]
    report.index.name = "species"

    summary = {
        "records_before": n_before,
        "records_after": len(df_clean),
        "removed_total": n_before - len(df_clean),
        "removed_missing": n_missing,
        "removed_duplicates": n_dup,
    }
    return df_clean, report, summary


def make_site_id(df: pd.DataFrame) -> np.ndarray:
    """Integer site id = unique combination of the 12 environmental variables.

    GPS is already part of this fingerprint. Raw coordinates are left as-is
    (some are slightly jittered) because the full environmental tuple, not
    the coordinate alone, defines a site -- this reproduces the manuscript's
    66 multi-specimen sites.
    """
    return df.groupby(C.SITE_KEY, sort=False).ngroup().to_numpy()


def site_report(df: pd.DataFrame, site_id: np.ndarray) -> dict:
    vc = pd.Series(site_id).value_counts()
    return {
        "n_sites": int(vc.size),
        "n_multi_specimen_sites": int((vc > 1).sum()),
        "n_specimens_in_multi_sites": int(vc[vc > 1].sum()),
        "n_singleton_sites": int((vc == 1).sum()),
    }


def get_data():
    """Full prep entry point.

    Returns a dict with X (DataFrame of FEATURES), y (int array of species),
    site_id, the cleaned dataframe, and the saved report tables.
    """
    raw = load_raw()
    df, dedup_report, summary = clean(raw)

    # target: keep original integer species labels (1..14)
    y = df[C.TARGET].to_numpy().astype(int)
    X = df[C.FEATURES].copy()
    site_id = make_site_id(df)

    # persist reviewer-facing tables
    dedup_report.to_csv(C.TABLE_DIR / "class_distribution_dedup.csv")
    pd.DataFrame([summary]).to_csv(
        C.TABLE_DIR / "cleaning_summary.csv", index=False)

    return {
        "df": df,
        "X": X,
        "y": y,
        "site_id": site_id,
        "dedup_report": dedup_report,
        "cleaning_summary": summary,
        "site_summary": site_report(df, site_id),
    }


if __name__ == "__main__":
    d = get_data()
    print("Cleaning summary:", d["cleaning_summary"])
    print("Site summary:", d["site_summary"])
    print("\nPer-class de-duplication report:")
    print(d["dedup_report"])
