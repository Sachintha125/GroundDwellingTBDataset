"""
config.py
=========
Central configuration for the ground-dwelling tiger beetle classification re-run.

Everything that a reviewer might ask you to justify or change lives here:
feature roles, the site definition used for location-independent evaluation,
model hyperparameter grids, the number of repeats, and plotting settings.

Nothing in this file runs analysis; it only declares constants that the other
scripts import.
"""
from __future__ import annotations
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Point DATA_PATH at the dataset CSV. Override with the TB_DATA env var.
DATA_PATH = Path(
    os.environ.get("TB_DATA", "dataset.csv")
)

OUTPUT_DIR = Path(os.environ.get("TB_OUT", "outputs"))
FIG_DIR = OUTPUT_DIR / "figures"      # all .png figures (600 dpi)
TABLE_DIR = OUTPUT_DIR / "tables"     # all .csv result tables

for _d in (OUTPUT_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# QUICK mode  --  set TB_QUICK=1 for a fast end-to-end smoke test
# (shrinks grids and repeats; NOT for final results)
# --------------------------------------------------------------------------
QUICK = os.environ.get("TB_QUICK", "0") == "1"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
BASE_SEED = 42
N_REPEATS = 2 if QUICK else 10        # independent repeats (split re-drawn each time)
N_SPLITS = 5                          # folds for the REPORTED cross-validation (Table 4, OOF, ablation)
GRID_CV_SPLITS = 3                    # folds used ONLY inside grid-search tuning (lower = faster)
TEST_SIZE = 0.25                      # held-out fraction (both tracks)

# --------------------------------------------------------------------------
# Raw -> clean column names
# The CSV header contains stray symbols (c, EUR, (c)) and units, so we rename
# strictly by column ORDER (14 columns) to avoid any encoding mismatch.
# --------------------------------------------------------------------------
CLEAN_COLUMNS = [
    "habitat_type",        # 1  categorical
    "climatic_zone",       # 2  categorical
    "latitude",            # 3  continuous (geographic coordinate)
    "longitude",           # 4  continuous (geographic coordinate)
    "elevation",           # 5  continuous
    "temperature",         # 6  continuous (weather)
    "solar_radiation",     # 7  continuous (weather)
    "relative_humidity",   # 8  continuous (weather)
    "wind_speed",          # 9  continuous (weather)
    "soil_moisture",       # 10 continuous (soil)
    "soil_salinity",       # 11 continuous (soil)
    "soil_ph",             # 12 continuous (soil)
    "body_length",         # 13 continuous (morphometric)
    "species",             # 14 target
]

TARGET = "species"

# --------------------------------------------------------------------------
# Feature roles
# --------------------------------------------------------------------------
CATEGORICAL = ["habitat_type", "climatic_zone"]        # one-hot + SMOTE-NC categorical
MORPHOMETRIC = ["body_length"]
# all non-target, non-categorical numeric predictors that get standardized:
CONTINUOUS = [
    "latitude", "longitude", "elevation",
    "temperature", "solar_radiation", "relative_humidity", "wind_speed",
    "soil_moisture", "soil_salinity", "soil_ph",
    "body_length",
]
# All predictors (order defines the model input matrix)
FEATURES = CATEGORICAL + CONTINUOUS

# The 12 environmental variables = everything except body_length and species.
# Their unique combination defines a collection "site" (the site fingerprint).
ENVIRONMENTAL = [c for c in FEATURES if c not in MORPHOMETRIC]
SITE_KEY = ENVIRONMENTAL  # columns whose unique combo identifies a site

# Human-readable labels for figures (match the manuscript's axis labels)
PRETTY = {
    "habitat_type": "Habitat type",
    "climatic_zone": "Climatic zone",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "elevation": "Elevation",
    "temperature": "Temperature",
    "solar_radiation": "Solar radiation",
    "relative_humidity": "Relative humidity",
    "wind_speed": "Wind speed",
    "soil_moisture": "Soil moisture",
    "soil_salinity": "Soil salinity",
    "soil_ph": "Soil pH",
    "body_length": "Body length",
}

# --------------------------------------------------------------------------
# Thematic blocks for grouped (block) permutation importance  (Fig 5)
# --------------------------------------------------------------------------
BLOCKS = {
    "Site fingerprint (All 12 environmental)": ENVIRONMENTAL,
    "Habitat and climatic zone": ["habitat_type", "climatic_zone"],
    "Location (Latitude / Longitude / Elevation)": ["latitude", "longitude", "elevation"],
    "Body length (morphology)": ["body_length"],
    "Weather (Temp / RH / Solar / Wind)": [
        "temperature", "relative_humidity", "solar_radiation", "wind_speed"],
    "Soil (Moisture / Salinity / pH)": ["soil_moisture", "soil_salinity", "soil_ph"],
}

# --------------------------------------------------------------------------
# Feature-group ablation sets  (reviewer comment 4)
# --------------------------------------------------------------------------
ABLATION_SETS = {
    "All features": FEATURES,
    "Body length only": ["body_length"],
    "Environmental only": ENVIRONMENTAL,
    "Environmental w/o GPS": [c for c in ENVIRONMENTAL if c not in ("latitude", "longitude")],
}

# --------------------------------------------------------------------------
# Models + hyperparameter grids  (from manuscript Table 1)
# Keys must match the 'clf' step name in the pipeline.
# --------------------------------------------------------------------------
def model_grids():
    """Return {model_name: (estimator_factory, param_grid)}.

    estimator_factory(seed) -> a fresh sklearn estimator.
    param_grid keys are prefixed 'clf__' for use inside the Pipeline.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.ensemble import (
        GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier)

    if QUICK:
        grids = {
            "KNN": (lambda s: KNeighborsClassifier(),
                    {"clf__n_neighbors": [3, 5], "clf__weights": ["distance"]}),
            "Naive Bayes": (lambda s: GaussianNB(),
                            {"clf__var_smoothing": [1e-9, 1e-6]}),
            "SVM": (lambda s: SVC(random_state=s),
                    {"clf__C": [1, 10], "clf__kernel": ["rbf", "poly"]}),
            "Gradient Boosting": (lambda s: GradientBoostingClassifier(random_state=s),
                                  {"clf__n_estimators": [50], "clf__max_depth": [3]}),
            "Extra Trees": (lambda s: ExtraTreesClassifier(random_state=s),
                            {"clf__n_estimators": [100], "clf__max_depth": [10]}),
            "RF": (lambda s: RandomForestClassifier(random_state=s),
                   {"clf__n_estimators": [100], "clf__max_depth": [10]}),
        }
        return grids

    grids = {
        "KNN": (
            lambda s: KNeighborsClassifier(),
            {"clf__n_neighbors": [1, 3, 5, 7],
             "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            lambda s: GaussianNB(),
            {"clf__var_smoothing": [1e-11, 1.58e-8, 2.51e-5, 1e-6]},
        ),
        "SVM": (
            lambda s: SVC(random_state=s, cache_size=500),
            # C grid trimmed from the manuscript's 9 values: dropped 205/500/1000,
            # which are the slowest to fit and rarely win on scaled data.
            # Restore the full list here if you want to search it exactly:
            #   [0.1, 1, 10, 50, 100, 200, 205, 500, 1000]
            {"clf__C": [10, 50, 100, 200],
             "clf__kernel": ["rbf", "poly"]},
        ),
        "Gradient Boosting": (
            lambda s: GradientBoostingClassifier(random_state=s),
            {"clf__n_estimators": [50, 100, 200],
             "clf__max_depth": [3, 4, 6],
             "clf__learning_rate": [0.05, 0.1]},
        ),
        "Extra Trees": (
            lambda s: ExtraTreesClassifier(random_state=s),
            {"clf__n_estimators": [50, 100, 200, 300],
             "clf__max_depth": [None, 10, 20]},
        ),
        "RF": (
            lambda s: RandomForestClassifier(random_state=s),
            {"clf__n_estimators": [50, 100, 200, 300],
             "clf__max_depth": [None, 10, 20]},
        ),
    }
    return grids

# Tree-based models do not need feature scaling (harmless, but we skip to be clean)
TREE_MODELS = {"Gradient Boosting", "Extra Trees", "RF"}

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
DPI = 600
# Consistent two-colour scheme for "Model 1 / Model 2" across every figure.
MODEL_COLORS = ["#2B6CB0", "#C05621"]   # blue, orange
PANEL_LETTERS = ["A", "B", "C", "D", "E", "F"]

# Number of permutation repeats for importance
PERM_REPEATS = 3 if QUICK else 10
