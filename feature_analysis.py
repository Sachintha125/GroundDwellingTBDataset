"""
feature_analysis.py
====================
Downstream, model-specific analyses for the selected models, run per track:

  * out-of-fold (OOF) predictions  -> per-class precision/recall/F1/support (Table 7)
  * single-feature permutation importance across folds        (Fig 3)
  * permutation schemes: env block jointly vs summed vs body length (Fig 4)
  * grouped (block) permutation importance                    (Fig 5)
  * feature-group ablation                                    (reviewer comment 4)

All "mean decrease in macro-F1" quantities are averaged across CV folds and
reported with their standard deviation (the error bars in the figures).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, GridSearchCV, cross_validate)
from sklearn.metrics import f1_score, precision_recall_fscore_support, make_scorer, cohen_kappa_score
from sklearn.base import clone

import config as C
from pipelines import build_pipeline, build_pipeline_cols

KAPPA = make_scorer(cohen_kappa_score)


# --------------------------------------------------------------------------
# CV splitter per track
# --------------------------------------------------------------------------
def _cv(track):
    """Reported CV (OOF predictions, ablation) -- full N_SPLITS."""
    if track == "A":
        return StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=C.BASE_SEED)
    return StratifiedGroupKFold(n_splits=C.N_SPLITS)


def _tune_cv(track):
    """Cheaper CV used only inside grid-search tuning -- GRID_CV_SPLITS."""
    if track == "A":
        return StratifiedKFold(C.GRID_CV_SPLITS, shuffle=True, random_state=C.BASE_SEED)
    return StratifiedGroupKFold(n_splits=C.GRID_CV_SPLITS)


def _splits(track, X, y, site_id):
    cv = _cv(track)
    if track == "A":
        return list(cv.split(X, y))
    return list(cv.split(X, y, groups=site_id))


# --------------------------------------------------------------------------
# representative tuned model (fix hyperparameters once per model/track)
# --------------------------------------------------------------------------
def tune_once(model_name, X, y, site_id, track):
    """Representative hyperparameters via the same sequential coordinate search
    used in model comparison (not full grid search)."""
    from evaluation import sequential_search
    cv = _tune_cv(track)
    groups = None if track == "A" else site_id
    return sequential_search(model_name, X, y, cv, groups=groups).best_params_


def fit_folds(model_name, best_params, X, y, site_id, track):
    """Fit the pipeline on each training fold; return (oof_pred, folds).

    folds = list of (fitted_pipe, val_idx) for permutation importance.
    """
    factory, _ = C.model_grids()[model_name]
    oof = np.full(len(y), -1)
    folds = []
    for tr, va in _splits(track, X, y, site_id):
        pipe = build_pipeline(model_name, factory(C.BASE_SEED))
        pipe.set_params(**best_params)
        pipe.fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict(X.iloc[va])
        folds.append((pipe, va))
    return oof, folds


# --------------------------------------------------------------------------
# Table 7 : per-class precision / recall / F1 / support from OOF predictions
# --------------------------------------------------------------------------
def per_class_table(y, oof):
    mask = oof >= 0                      # some classes may be unpredicted in grouped CV
    labels = np.unique(y)
    p, r, f, s = precision_recall_fscore_support(
        y[mask], oof[mask], labels=labels, zero_division=0)
    return pd.DataFrame({
        "species": labels, "precision": p, "recall": r,
        "f1": f, "support": s}).set_index("species")


# --------------------------------------------------------------------------
# permutation importance across folds (base - permuted macro-F1)
# --------------------------------------------------------------------------
def _macro_f1(pipe, Xv, yv):
    return f1_score(yv, pipe.predict(Xv), average="macro")


def _permute_cols(Xv, cols, rng):
    """Return a copy of Xv with `cols` jointly permuted (shared row order)."""
    Xp = Xv.copy()
    order = rng.permutation(len(Xp))
    for c in cols:
        Xp[c] = Xp[c].to_numpy()[order]
    return Xp


def _importance_of(folds, X, y, col_sets):
    """For each named col-set, mean/std decrease in macro-F1 across folds."""
    rng = np.random.default_rng(C.BASE_SEED)
    out = {name: [] for name in col_sets}
    for pipe, va in folds:
        Xv, yv = X.iloc[va], y[va]
        base = _macro_f1(pipe, Xv, yv)
        for name, cols in col_sets.items():
            drops = []
            for _ in range(C.PERM_REPEATS):
                Xp = _permute_cols(Xv, cols, rng)
                drops.append(base - _macro_f1(pipe, Xp, yv))
            out[name].append(np.mean(drops))
    rows = [{"feature": n, "mean": np.mean(v), "std": np.std(v, ddof=1) if len(v) > 1 else 0.0}
            for n, v in out.items()]
    return pd.DataFrame(rows)


def single_feature_importance(folds, X, y):
    """Fig 3 : permute each predictor individually."""
    col_sets = {C.PRETTY[c]: [c] for c in C.FEATURES}
    df = _importance_of(folds, X, y, col_sets)
    return df.sort_values("mean", ascending=False).reset_index(drop=True)


def permutation_schemes(folds, X, y):
    """Fig 4 : env block jointly vs summed-individual vs body length alone."""
    # joint env block
    joint = _importance_of(folds, X, y, {"env_block": C.ENVIRONMENTAL})
    # summed individual env
    indiv = _importance_of(folds, X, y, {c: [c] for c in C.ENVIRONMENTAL})
    summed_mean = indiv["mean"].sum()
    # per-fold sum for std: recompute
    body = _importance_of(folds, X, y, {"body": ["body_length"]})
    rows = [
        {"scheme": "Environmental block\n(permuted jointly)",
         "mean": joint.loc[0, "mean"], "std": joint.loc[0, "std"]},
        {"scheme": "Environmental features\n(individually, summed)",
         "mean": summed_mean, "std": float(np.sqrt((indiv["std"]**2).sum()))},
        {"scheme": "Body length\n(permuted alone)",
         "mean": body.loc[0, "mean"], "std": body.loc[0, "std"]},
    ]
    return pd.DataFrame(rows)


def block_importance(folds, X, y):
    """Fig 5 : grouped (block) permutation importance."""
    df = _importance_of(folds, X, y, C.BLOCKS)
    return df.sort_values("mean", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# feature-group ablation
# --------------------------------------------------------------------------
def ablation(model_name, best_params, X, y, site_id, track):
    """Macro-F1 (mean/std across folds) for each feature subset."""
    factory, _ = C.model_grids()[model_name]
    cv = _cv(track)
    rows = []
    for set_name, cols in C.ABLATION_SETS.items():
        cat_cols = [c for c in cols if c in C.CATEGORICAL]
        cont_cols = [c for c in cols if c in C.CONTINUOUS]
        pipe = build_pipeline_cols(model_name, factory(C.BASE_SEED), cat_cols, cont_cols)
        # reuse tuned clf hyperparameters
        clf_params = {k: v for k, v in best_params.items() if k.startswith("clf__")}
        pipe.set_params(**clf_params)
        kw = {} if track == "A" else {"groups": site_id}
        res = cross_validate(pipe, X[cols], y, cv=cv, scoring="f1_macro",
                             n_jobs=-1, **kw)
        rows.append({"feature_set": set_name,
                     "f1_macro_mean": res["test_score"].mean(),
                     "f1_macro_std": res["test_score"].std(ddof=1)})
    return pd.DataFrame(rows)
