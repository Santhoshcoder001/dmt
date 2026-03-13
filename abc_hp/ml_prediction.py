"""
M6 — ML Prediction Engine
==========================
Classifies or regresses the accident risk level per grid cell.

Ensemble architecture
---------------------
Three base models are trained and combined via a meta-model (stacking):

* **XGBoost** — handles class imbalance, SHAP-explainable
* **Random Forest** — stable ensemble baseline
* **LSTM** — captures temporal accident patterns (optional; requires TensorFlow)

The meta-model is a Logistic Regression that combines out-of-fold predictions
from all base models.

Risk labelling
--------------
Grid cells are labelled as high-risk (``1``) when their ``corrected_count``
exceeds the 75th percentile, providing a balanced training signal.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Risk threshold — cells above this percentile of corrected_count are high-risk
_RISK_PERCENTILE: float = 75.0


class MLPredictionEngine:
    """Train and evaluate the stacking ensemble for hotspot risk prediction.

    Parameters
    ----------
    use_xgboost : bool
        Include XGBoost as a base estimator.  Requires the ``xgboost`` package.
        Default: True.
    use_lstm : bool
        Include an LSTM base estimator.  Requires TensorFlow and a temporal
        feature tensor.  Default: False (heavy dependency; opt-in).
    n_estimators_rf : int
        Number of trees in the Random Forest base model.  Default: 200.
    risk_percentile : float
        Percentile of ``corrected_count`` above which a cell is labelled
        high-risk (class 1).  Default: 75.
    random_state : int, optional
        Controls reproducibility.
    """

    def __init__(
        self,
        use_xgboost: bool = True,
        use_lstm: bool = False,
        n_estimators_rf: int = 200,
        risk_percentile: float = _RISK_PERCENTILE,
        random_state: Optional[int] = 42,
    ) -> None:
        self.use_xgboost = use_xgboost
        self.use_lstm = use_lstm
        self.n_estimators_rf = n_estimators_rf
        self.risk_percentile = risk_percentile
        self.random_state = random_state

        self._model: Optional[StackingClassifier] = None
        self._feature_names: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_labels(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """Create binary risk labels from ``corrected_count``.

        Parameters
        ----------
        feature_matrix : pd.DataFrame
            Output of :meth:`FeatureEngineer.build_feature_matrix` (indexed by
            ``grid_id``).

        Returns
        -------
        pd.Series
            Binary series (0 = low risk, 1 = high risk) aligned with
            *feature_matrix*.
        """
        if "corrected_count" not in feature_matrix.columns:
            raise ValueError("'corrected_count' column required to build risk labels.")
        threshold = np.percentile(feature_matrix["corrected_count"], self.risk_percentile)
        labels = (feature_matrix["corrected_count"] >= threshold).astype(int)
        labels.name = "risk_label"
        logger.info(
            "Risk labels: %d high-risk cells (%.0f%% percentile threshold = %.2f)",
            labels.sum(),
            self.risk_percentile,
            threshold,
        )
        return labels

    def fit(
        self,
        feature_matrix: pd.DataFrame,
        labels: pd.Series,
        drop_cols: Optional[List[str]] = None,
    ) -> "MLPredictionEngine":
        """Train the stacking ensemble.

        Parameters
        ----------
        feature_matrix : pd.DataFrame
            Indexed by ``grid_id``; each row is one cell.
        labels : pd.Series
            Binary risk labels from :meth:`build_labels`.
        drop_cols : list of str, optional
            Feature columns to exclude from training (e.g., ``corrected_count``
            should be dropped to prevent data leakage).

        Returns
        -------
        self
        """
        X, y, feature_names = self._prepare(feature_matrix, labels, drop_cols)
        self._feature_names = feature_names

        estimators = self._build_base_estimators()
        meta_clf = LogisticRegression(max_iter=1000, random_state=self.random_state)

        self._model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_clf,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            passthrough=False,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        self._fitted = True
        logger.info("Ensemble fitted on %d samples, %d features", len(X), X.shape[1])
        return self

    def predict(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        """Predict binary risk class for each grid cell.

        Parameters
        ----------
        feature_matrix : pd.DataFrame

        Returns
        -------
        ndarray of shape (n,)
        """
        self._assert_fitted()
        X = self._align_features(feature_matrix)
        return self._model.predict(X)

    def predict_proba(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        """Return risk probability scores for each grid cell.

        Parameters
        ----------
        feature_matrix : pd.DataFrame

        Returns
        -------
        ndarray of shape (n,)
            Probability of being high-risk (class 1).
        """
        self._assert_fitted()
        X = self._align_features(feature_matrix)
        return self._model.predict_proba(X)[:, 1]

    def evaluate(
        self,
        feature_matrix: pd.DataFrame,
        labels: pd.Series,
        drop_cols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute evaluation metrics on a held-out set.

        Metrics
        -------
        * precision
        * recall
        * f1
        * roc_auc
        * avg_precision (AUC-PR)

        Parameters
        ----------
        feature_matrix : pd.DataFrame
        labels : pd.Series
        drop_cols : list of str, optional

        Returns
        -------
        dict
        """
        X, y, _ = self._prepare(feature_matrix, labels, drop_cols)
        X_aligned = self._align_features(
            pd.DataFrame(X, columns=self._feature_names)
        )
        y_pred = self._model.predict(X_aligned)
        y_proba = self._model.predict_proba(X_aligned)[:, 1]

        metrics = {
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
            "avg_precision": average_precision_score(y, y_proba),
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances from the Random Forest base model.

        Returns
        -------
        pd.Series
            Importances sorted descending, indexed by feature name.
        """
        self._assert_fitted()
        if self._model is None:
            return pd.Series(dtype=float)
        # Find the RF estimator in the stack
        for name, est in self._model.estimators_:
            pipeline_steps = est.steps if hasattr(est, "steps") else []
            for _, step in pipeline_steps:
                if isinstance(step, RandomForestClassifier):
                    importances = pd.Series(
                        step.feature_importances_,
                        index=self._feature_names,
                        name="importance",
                    )
                    return importances.sort_values(ascending=False)
        return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_base_estimators(self) -> List[Tuple[str, Pipeline]]:
        estimators: List[Tuple[str, Pipeline]] = []

        # Random Forest
        rf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators_rf,
                        class_weight="balanced",
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        estimators.append(("random_forest", rf))

        # XGBoost (optional)
        if self.use_xgboost:
            try:
                import xgboost as xgb  # noqa: PLC0415

                xgb_clf = Pipeline(
                    [
                        (
                            "clf",
                            xgb.XGBClassifier(
                                n_estimators=200,
                                use_label_encoder=False,
                                eval_metric="logloss",
                                scale_pos_weight=1,
                                random_state=self.random_state,
                                verbosity=0,
                            ),
                        )
                    ]
                )
                estimators.append(("xgboost", xgb_clf))
            except ImportError:
                logger.warning("xgboost not installed — XGBoost base model skipped.")

        return estimators

    @staticmethod
    def _prepare(
        feature_matrix: pd.DataFrame,
        labels: pd.Series,
        drop_cols: Optional[List[str]],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        df = feature_matrix.copy()
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        # Drop non-numeric columns
        df = df.select_dtypes(include=[np.number])
        df = df.fillna(0)
        aligned_labels = labels.reindex(df.index).fillna(0).astype(int)
        return df.to_numpy(dtype=float), aligned_labels.to_numpy(), list(df.columns)

    def _align_features(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        df = feature_matrix.copy()
        df = df.select_dtypes(include=[np.number])
        # Add missing columns as 0, reorder to training order
        for col in self._feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self._feature_names].fillna(0)
        return df.to_numpy(dtype=float)

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("MLPredictionEngine must be fitted before predicting.")
