"""
M3 — Bias Correction
====================
Adjusts under-reported accident counts using Poisson Inverse Probability
Weighting (IPW).

Core novelty of the ABC-HP system.

Pipeline
--------
1. Estimate the probability of an accident being *reported* via a logistic
   regression on observable covariates (population density, road type,
   weather conditions, time of day).
2. Compute per-record IPW weights  ``w(i) = 1 / p(i)``.
3. Aggregate bias-corrected weighted counts per spatial grid cell.

Types of bias addressed
-----------------------
* Geographic — rural areas under-reported
* Severity   — minor crashes rarely recorded
* Temporal   — night-time incidents missing
* Socioeconomic — low-income zones missed
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Default feature columns used to model reporting probability
DEFAULT_COVARIATE_COLS: List[str] = [
    "pop_density",
    "road_type_encoded",
    "hour",
    "is_weekend",
    "weather_encoded",
]

# Clip weights to avoid extreme values when p(i) ≈ 0
_WEIGHT_CLIP_MAX: float = 100.0
_MIN_REPORTING_PROB: float = 0.01


class BiasCorrector:
    """Estimate reporting probabilities and compute IPW bias-correction weights.

    Parameters
    ----------
    covariate_cols : list of str, optional
        Feature columns used to model P(reported | covariates).
        Defaults to :data:`DEFAULT_COVARIATE_COLS`.
    weight_clip_max : float
        Upper bound applied to computed weights to limit leverage of extreme
        observations.  Default: 100.
    random_state : int, optional
        Random state for the logistic regression solver.
    """

    def __init__(
        self,
        covariate_cols: Optional[List[str]] = None,
        weight_clip_max: float = _WEIGHT_CLIP_MAX,
        random_state: Optional[int] = 42,
    ) -> None:
        self.covariate_cols = covariate_cols or DEFAULT_COVARIATE_COLS
        self.weight_clip_max = weight_clip_max
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        reported_col: str = "is_reported",
    ) -> "BiasCorrector":
        """Train the reporting-probability model.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset that contains both reported and estimated-unreported
            accidents.  Must include the columns listed in
            ``self.covariate_cols`` and a binary ``reported_col``.
        reported_col : str
            Column name for the binary reporting indicator (1 = reported,
            0 = estimated unreported).

        Returns
        -------
        self
        """
        X, y = self._prepare_features(df, reported_col=reported_col)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._fitted = True
        logger.info(
            "BiasCorrector fitted on %d samples (%.1f%% reported)",
            len(df),
            100.0 * y.mean(),
        )
        return self

    def predict_reporting_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(reported) for each row in *df*.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        ndarray of shape (n,)
            Estimated probability that each accident was reported.
        """
        self._assert_fitted()
        X, _ = self._prepare_features(df, reported_col=None)
        X_scaled = self._scaler.transform(X)
        probs = self._model.predict_proba(X_scaled)[:, 1]
        probs = np.clip(probs, _MIN_REPORTING_PROB, 1.0)
        return probs

    def compute_weights(self, df: pd.DataFrame) -> pd.Series:
        """Compute IPW bias-correction weights ``w(i) = 1 / p(i)``.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.Series
            Per-row weights indexed like *df*.
        """
        probs = self.predict_reporting_probability(df)
        weights = 1.0 / probs
        weights = np.clip(weights, 1.0, self.weight_clip_max)
        return pd.Series(weights, index=df.index, name="bias_weight")

    def apply_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach ``bias_weight`` column to *df* and return the result.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Copy of *df* with an additional ``bias_weight`` column.
        """
        df = df.copy()
        df["bias_weight"] = self.compute_weights(df)
        logger.info(
            "Bias weights computed — mean: %.2f, max: %.2f",
            df["bias_weight"].mean(),
            df["bias_weight"].max(),
        )
        return df

    def aggregate_weighted_counts(
        self,
        df: pd.DataFrame,
        grid_col: str = "grid_id",
    ) -> pd.Series:
        """Sum bias-corrected weights per spatial grid cell.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``bias_weight`` and *grid_col* columns.
        grid_col : str
            Column identifying the spatial grid cell.

        Returns
        -------
        pd.Series
            Bias-corrected accident counts indexed by grid cell ID.
        """
        if "bias_weight" not in df.columns:
            raise ValueError("Call apply_weights() before aggregate_weighted_counts().")
        if grid_col not in df.columns:
            raise ValueError(f"Grid column '{grid_col}' not found in DataFrame.")

        counts = df.groupby(grid_col)["bias_weight"].sum()
        counts.name = "corrected_count"
        return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_features(
        self,
        df: pd.DataFrame,
        reported_col: Optional[str],
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        available = [c for c in self.covariate_cols if c in df.columns]
        if not available:
            raise ValueError(
                f"None of the covariate columns {self.covariate_cols} found in DataFrame. "
                "Encode categorical columns (road_type_encoded, weather_encoded) before "
                "calling BiasCorrector."
            )
        X = df[available].fillna(0).to_numpy(dtype=float)
        y = None
        if reported_col and reported_col in df.columns:
            y = df[reported_col].to_numpy(dtype=int)
        return X, y

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("BiasCorrector must be fitted before predicting.")

    # ------------------------------------------------------------------
    # Convenience: encode categorical covariates
    # ------------------------------------------------------------------

    @staticmethod
    def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode ``road_type`` and ``weather`` columns for use as
        numeric covariates.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Copy with ``road_type_encoded`` and ``weather_encoded`` columns
            added.
        """
        df = df.copy()
        if "road_type" in df.columns:
            df["road_type_encoded"] = df["road_type"].astype("category").cat.codes
        if "weather" in df.columns:
            df["weather_encoded"] = df["weather"].astype("category").cat.codes
        return df
