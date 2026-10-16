"""
evaluation.py
=============
Model comparison across the two evaluation tracks, with repeats, and the
selection of the top-2 models.

TRACK A  -- Specimen-level (reproduces the manuscript)
    * stratified 75:25 split (seed re-drawn each repeat)
    * hyperparameters tuned by stratified 5-fold CV on the training part
    * scored ONCE on the held-out 25% test set
    * supporting stratified 5-fold CV metrics (Table 4 equivalent)

TRACK B  -- Location-independent (answers the reviewer)
    * site-disjoint 75:25 split (GroupShuffleSplit on site_id): the test set
      contains ONLY sites absent from training -> performance on unseen locations
    * hyperparameters tuned by StratifiedGroupKFold on the training sites
    * scored ONCE on the unseen-site test set
    * supporting location-grouped 5-fold CV metrics (stability check)

All metric values are reported as mean +/- std across the repeats.
"""
from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, StratifiedGroupKFold,
    GroupShuffleSplit, GridSearchCV, cross_validate)
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, make_scorer)

import config as C
from pipelines import build_pipeline, set_seed

warnings.filterwarnings("ignore")

KAPPA = make_scorer(cohen_kappa_score)
SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro", "kappa": KAPPA}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class _SearchResult:
    """Lightweight stand-in for a fitted GridSearchCV."""
    def __init__(self, best_estimator_, best_params_):
        self.best_estimator_ = best_estimator_
        self.best_params_ = best_params_


def sequential_search(model_name, X, y, cv, groups=None):
    """One-parameter-at-a-time (coordinate) hyperparameter search.

    For each hyperparameter in turn, hold the others at their current best and
    pick the value with the highest mean CV kappa. Costs the SUM of the grid
    sizes rather than the PRODUCT, so it is much faster than full grid search.
    Trade-off: it can miss parameter interactions (e.g. SVM C depends on the
    kernel), so describe tuning as 'sequential one-parameter-at-a-time' in the
    manuscript, not 'grid search'. Each candidate is still scored by CV, and
    the final model is refit on all of (X, y) with the chosen values.
    """
    factory, grid = C.model_grids()[model_name]
    best = {k: v[0] for k, v in grid.items()}        # start at first value

    def _score(params):
        pipe = build_pipeline(model_name, factory(C.BASE_SEED))
        pipe.set_params(**params)
        res = cross_validate(pipe, X, y, cv=cv, scoring=KAPPA,
                             groups=groups, n_jobs=-1)
        return res["test_score"].mean()

    for k, values in grid.items():
        scored = [(_score({**best, k: v}), v) for v in values]
        best[k] = max(scored, key=lambda t: t[0])[1]

    final = build_pipeline(model_name, factory(C.BASE_SEED))
    final.set_params(**best)
    final.fit(X, y)
    return _SearchResult(final, best)


def _grid_search(model_name, X, y, cv, groups=None):
    """Tune one model (now via sequential coordinate search)."""
    return sequential_search(model_name, X, y, cv, groups=groups)


def _time_classifier(best_pipe, X_tr, y_tr, X_te):
    """Time ONLY the classifier (fit on preprocessed train, predict on test).

    Preprocessing (scale/SMOTE/encode) is applied with the already-fitted
    transformers so the timing isolates the classifier, matching the
    manuscript's Table 3 train/inference-time comparison.
    """
    scale = best_pipe.named_steps["scale"]
    smote = best_pipe.named_steps["smote"]
    encode = best_pipe.named_steps["encode"]
    clf = best_pipe.named_steps["clf"]

    Xs = scale.transform(X_tr)
    Xr, yr = smote.fit_resample(Xs, y_tr)
    Xe = encode.transform(Xr)
    Xe_te = encode.transform(scale.transform(X_te))

    from sklearn.base import clone
    c = clone(clf)
    c.fit(Xe, yr)                       # warm-up / build
    t0 = time.perf_counter(); c.fit(Xe, yr); t1 = time.perf_counter()
    c.predict(Xe_te)                    # warm-up predict
    t2 = time.perf_counter(); c.predict(Xe_te); t3 = time.perf_counter()

    train_ms = (t1 - t0) * 1e3
    infer_ms_per_sample = (t3 - t2) * 1e3 / max(1, len(X_te))
    return train_ms, infer_ms_per_sample


def _eval_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def _agg(records: list[dict]) -> dict:
    """mean/std across repeats for each numeric key."""
    df = pd.DataFrame(records)
    out = {}
    for k in df.columns:
        out[f"{k}_mean"] = df[k].mean()
        out[f"{k}_std"] = df[k].std(ddof=1) if len(df) > 1 else 0.0
    return out


# --------------------------------------------------------------------------
# per-model, per-track evaluation across repeats
# --------------------------------------------------------------------------
def evaluate_model(model_name, X, y, site_id, track):
    """Return (aggregated metrics dict, per-repeat best-params list) for one
    model/track."""
    test_rows, cv_rows, time_rows, param_rows = [], [], [], []

    for r in range(C.N_REPEATS):
        seed = C.BASE_SEED + r

        if track == "A":
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=C.TEST_SIZE, stratify=y, random_state=seed)
            tune_cv = StratifiedKFold(C.GRID_CV_SPLITS, shuffle=True, random_state=seed)
            report_cv = StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=seed)
            gs = _grid_search(model_name, X_tr, y_tr, tune_cv)
            groups_tr = None
        else:  # Track B -- site-disjoint split
            gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE,
                                    random_state=seed)
            tr_idx, te_idx = next(gss.split(X, y, groups=site_id))
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            groups_tr = site_id[tr_idx]
            tune_cv = StratifiedGroupKFold(n_splits=C.GRID_CV_SPLITS)
            report_cv = StratifiedGroupKFold(n_splits=C.N_SPLITS)
            gs = _grid_search(model_name, X_tr, y_tr, tune_cv, groups=groups_tr)

        best = gs.best_estimator_
        y_pred = best.predict(X_te)
        test_rows.append(_eval_metrics(y_te, y_pred))

        # record the best hyperparameters chosen in this repeat
        bp = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        param_rows.append({"model": model_name, "track": track,
                           "repeat": r, "test_kappa": test_rows[-1]["kappa"],
                           **bp})

        # supporting CV metrics (re-use best params on the training part);
        # reported at the full N_SPLITS, independent of the grid-search folds.
        if track == "A":
            cv_res = cross_validate(best, X_tr, y_tr, cv=report_cv,
                                    scoring=SCORING, n_jobs=-1)
        else:
            cv_res = cross_validate(best, X_tr, y_tr, cv=report_cv,
                                    scoring=SCORING, n_jobs=-1,
                                    groups=groups_tr)
        cv_rows.append({
            "accuracy": cv_res["test_accuracy"].mean(),
            "f1_macro": cv_res["test_f1_macro"].mean(),
            "kappa": cv_res["test_kappa"].mean(),
        })

        try:
            tr_ms, inf_ms = _time_classifier(best, X_tr, y_tr, X_te)
            time_rows.append({"train_ms": tr_ms, "infer_ms_per_sample": inf_ms})
        except Exception:
            pass

    out = {"model": model_name, "track": track}
    out.update({f"test_{k}": v for k, v in _agg(test_rows).items()})
    out.update({f"cv_{k}": v for k, v in _agg(cv_rows).items()})
    if time_rows:
        out.update(_agg(time_rows))

    # most frequently selected hyperparameters across repeats (modal config)
    pdf = pd.DataFrame(param_rows)
    hp_cols = [c for c in pdf.columns
               if c not in ("model", "track", "repeat", "test_kappa")]
    modal = {c: pdf[c].mode().iloc[0] for c in hp_cols} if len(pdf) else {}
    out["best_hyperparameters"] = "; ".join(f"{k}={v}" for k, v in modal.items())

    return out, param_rows


def compare_all_models(X, y, site_id):
    """Run every model on both tracks.

    Returns (results_df, best_params_df) where best_params_df holds the
    hyperparameters selected in every repeat.
    """
    rows, all_params = [], []
    for name in C.model_grids().keys():
        for track in ("A", "B"):
            print(f"  [{track}] {name} ...", flush=True)
            out, params = evaluate_model(name, X, y, site_id, track)
            rows.append(out)
            all_params.extend(params)
    return pd.DataFrame(rows), pd.DataFrame(all_params)


def select_top2(results: pd.DataFrame, track="A", metric="test_kappa_mean"):
    """Pick the two best models by kappa on the chosen track's held-out set."""
    col = metric if metric in results.columns else "test_kappa_mean"
    sub = results[results["track"] == track].sort_values(col, ascending=False)
    return sub["model"].head(2).tolist()
