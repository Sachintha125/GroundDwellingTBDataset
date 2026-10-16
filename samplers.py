"""
samplers.py
===========
A categorical-aware oversampler that is safe for very small minority classes.

Why not plain SMOTE?
  Habitat type and climatic zone are categorical. Standard SMOTE interpolates
  between neighbours and would create fractional / invalid category values.
  SMOTE-NC (Nominal-Continuous) keeps categorical columns valid by taking the
  majority category among neighbours. (Reviewer comment 3.)

Why a wrapper?
  Some rare species have very few specimens, and inside a cross-validation fold
  (especially the location-grouped folds) a class can have fewer than
  k_neighbors+1 samples -- or be missing entirely. This wrapper:
    * caps k_neighbors at (smallest present class count - 1) per fit, and
    * falls back to RandomOverSampler when any class has < 2 samples.
  It never touches the validation/test fold because it lives inside the
  training pipeline.
"""
from __future__ import annotations
import numpy as np
from collections import Counter

from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.base import BaseSampler


class AdaptiveSMOTENC(BaseSampler):
    """SMOTE-NC with k_neighbors auto-capped; RandomOverSampler fallback."""

    _sampling_type = "over-sampling"
    _parameter_constraints: dict = {}   # skip sklearn param validation

    def __init__(self, categorical_features, k_neighbors=5, random_state=None,
                 sampling_strategy="auto"):
        super().__init__()
        self.categorical_features = categorical_features
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy

    def _fit_resample(self, X, y):
        counts = Counter(y)
        min_count = min(counts.values())

        # If a class is a singleton, SMOTE cannot build neighbours -> duplicate.
        if min_count < 2:
            ros = RandomOverSampler(random_state=self.random_state)
            return ros.fit_resample(X, y)

        k = min(self.k_neighbors, min_count - 1)
        k = max(1, k)
        sm = SMOTENC(
            categorical_features=self.categorical_features,
            k_neighbors=k,
            random_state=self.random_state,
        )
        return sm.fit_resample(X, y)


class AdaptiveSMOTE(BaseSampler):
    """Plain SMOTE with k auto-capped; for all-continuous feature subsets
    (e.g. the 'body length only' ablation where there is no categorical)."""

    _sampling_type = "over-sampling"
    _parameter_constraints: dict = {}

    def __init__(self, k_neighbors=5, random_state=None, sampling_strategy="auto"):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy

    def _fit_resample(self, X, y):
        counts = Counter(y)
        min_count = min(counts.values())
        if min_count < 2:
            return RandomOverSampler(
                random_state=self.random_state).fit_resample(X, y)
        k = max(1, min(self.k_neighbors, min_count - 1))
        return SMOTE(k_neighbors=k,
                     random_state=self.random_state).fit_resample(X, y)
