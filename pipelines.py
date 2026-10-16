"""
pipelines.py
============
Build a leakage-safe, categorical-aware pipeline for any model.

Order of operations (ALL fit inside each training fold only):
    1. scale   : StandardScaler on continuous features (passthrough for trees),
                 categoricals passed through unchanged.
    2. smote   : AdaptiveSMOTENC on the TRAINING rows only (never on val/test).
    3. encode  : one-hot encode the (now valid) categorical columns.
    4. clf     : the classifier.

Because scaling / SMOTE / encoding all live inside the pipeline, they are refit
on the training part of every CV fold and of every train/test split, so no
information leaks from validation or test data. (Reviewer comment 3.)
"""
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline

import config as C
from samplers import AdaptiveSMOTENC, AdaptiveSMOTE

# Column layout inside the model input matrix X[FEATURES]:
#   FEATURES = CATEGORICAL + CONTINUOUS
# After the 'scale' ColumnTransformer below we reorder to [CONTINUOUS, CATEGORICAL],
# so the categorical columns become the LAST len(CATEGORICAL) columns.
_N_CONT = len(C.CONTINUOUS)
_N_CAT = len(C.CATEGORICAL)
_CAT_IDX_AFTER_SCALE = list(range(_N_CONT, _N_CONT + _N_CAT))
_CONT_IDX_AFTER_SCALE = list(range(_N_CONT))


def build_pipeline(model_name: str, estimator):
    """Return an imblearn Pipeline for the given model."""
    num_tf = "passthrough" if model_name in C.TREE_MODELS else StandardScaler()

    scale = ColumnTransformer(
        transformers=[
            ("num", num_tf, C.CONTINUOUS),
            ("cat", "passthrough", C.CATEGORICAL),
        ],
        remainder="drop",
    )

    smote = AdaptiveSMOTENC(
        categorical_features=_CAT_IDX_AFTER_SCALE,
        k_neighbors=5,
    )

    encode = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             _CAT_IDX_AFTER_SCALE),
            ("rest", "passthrough", _CONT_IDX_AFTER_SCALE),
        ],
        remainder="drop",
    )

    pipe = Pipeline(steps=[
        ("scale", scale),
        ("smote", smote),
        ("encode", encode),
        ("clf", estimator),
    ])
    return pipe


def build_pipeline_cols(model_name, estimator, cat_cols, cont_cols):
    """Flexible pipeline for an arbitrary subset of columns (used in the
    feature-group ablation). Handles the all-continuous case (no categorical)
    by switching SMOTE-NC -> plain SMOTE."""
    num_tf = "passthrough" if model_name in C.TREE_MODELS else StandardScaler()
    n_cont = len(cont_cols)
    n_cat = len(cat_cols)

    if n_cat > 0:
        scale = ColumnTransformer([
            ("num", num_tf, cont_cols),
            ("cat", "passthrough", cat_cols),
        ])
        cat_idx = list(range(n_cont, n_cont + n_cat))
        cont_idx = list(range(n_cont))
        smote = AdaptiveSMOTENC(categorical_features=cat_idx, k_neighbors=5)
        encode = ColumnTransformer([
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx),
            ("rest", "passthrough", cont_idx),
        ])
        steps = [("scale", scale), ("smote", smote), ("encode", encode),
                 ("clf", estimator)]
    else:
        scale = ColumnTransformer([("num", num_tf, cont_cols)])
        smote = AdaptiveSMOTE(k_neighbors=5)
        steps = [("scale", scale), ("smote", smote), ("clf", estimator)]
    return Pipeline(steps=steps)


def set_seed(pipe, seed):
    """Set the random_state of the SMOTE step and the classifier, if present."""
    pipe.set_params(smote__random_state=seed)
    clf = pipe.named_steps["clf"]
    if "random_state" in clf.get_params():
        pipe.set_params(clf__random_state=seed)
    return pipe
