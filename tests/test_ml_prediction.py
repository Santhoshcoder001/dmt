"""
Tests for M6 — MLPredictionEngine
"""

import numpy as np
import pandas as pd
import pytest

from abc_hp.ml_prediction import MLPredictionEngine
from abc_hp.spatial_grid import SpatialGridEngine
from abc_hp.feature_engineering import FeatureEngineer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_feature_matrix(n_cells=80, seed=42):
    rng = np.random.default_rng(seed)
    grid_ids = [f"cell_{i}" for i in range(n_cells)]
    data = {
        "corrected_count": rng.uniform(1, 50, n_cells),
        "accident_density": rng.uniform(0, 1, n_cells),
        "peak_hour_ratio": rng.uniform(0, 1, n_cells),
        "night_ratio": rng.uniform(0, 1, n_cells),
        "weekend_ratio": rng.uniform(0, 1, n_cells),
        "rain_accident_ratio": rng.uniform(0, 1, n_cells),
        "fog_ratio": rng.uniform(0, 0.5, n_cells),
        "severe_weather_ratio": rng.uniform(0, 0.3, n_cells),
        "pop_density": rng.uniform(100, 10000, n_cells),
        "junction_count": rng.integers(0, 20, n_cells).astype(float),
    }
    return pd.DataFrame(data, index=pd.Index(grid_ids, name="grid_id"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildLabels:
    def test_binary_labels(self):
        fm = make_feature_matrix()
        engine = MLPredictionEngine(use_xgboost=False)
        labels = engine.build_labels(fm)
        assert set(labels.unique()).issubset({0, 1})

    def test_labels_length(self):
        fm = make_feature_matrix()
        engine = MLPredictionEngine(use_xgboost=False)
        labels = engine.build_labels(fm)
        assert len(labels) == len(fm)

    def test_labels_percentile_75(self):
        fm = make_feature_matrix()
        engine = MLPredictionEngine(use_xgboost=False, risk_percentile=75)
        labels = engine.build_labels(fm)
        # Approximately 25% should be high-risk
        ratio = labels.mean()
        assert 0.1 <= ratio <= 0.5  # generous bounds

    def test_missing_corrected_count_raises(self):
        fm = make_feature_matrix().drop(columns=["corrected_count"])
        engine = MLPredictionEngine(use_xgboost=False)
        with pytest.raises(ValueError, match="corrected_count"):
            engine.build_labels(fm)


class TestFitPredict:
    def test_fit_returns_self(self):
        fm = make_feature_matrix()
        engine = MLPredictionEngine(use_xgboost=False)
        labels = engine.build_labels(fm)
        result = engine.fit(fm, labels, drop_cols=["corrected_count"])
        assert result is engine

    def test_predict_shape(self):
        fm = make_feature_matrix()
        engine = MLPredictionEngine(use_xgboost=False)
        labels = engine.build_labels(fm)
        engine.fit(fm, labels, drop_cols=["corrected_count"])
        preds = engine.predict(fm)
        assert preds.shape == (len(fm),)

    def test_predict_binary(self):
        fm = make_feature_matrix()
        engine = MLPredictionEngine(use_xgboost=False)
        labels = engine.build_labels(fm)
        engine.fit(fm, labels, drop_cols=["corrected_count"])
        preds = engine.predict(fm)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_range(self):
        fm = make_feature_matrix()
        engine = MLPredictionEngine(use_xgboost=False)
        labels = engine.build_labels(fm)
        engine.fit(fm, labels, drop_cols=["corrected_count"])
        proba = engine.predict_proba(fm)
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()

    def test_predict_before_fit_raises(self):
        fm = make_feature_matrix()
        engine = MLPredictionEngine(use_xgboost=False)
        with pytest.raises(RuntimeError, match="fitted"):
            engine.predict(fm)

    def test_predict_proba_before_fit_raises(self):
        fm = make_feature_matrix()
        engine = MLPredictionEngine(use_xgboost=False)
        with pytest.raises(RuntimeError, match="fitted"):
            engine.predict_proba(fm)


class TestEvaluate:
    def test_returns_expected_metrics(self):
        fm = make_feature_matrix()
        engine = MLPredictionEngine(use_xgboost=False)
        labels = engine.build_labels(fm)
        engine.fit(fm, labels, drop_cols=["corrected_count"])
        metrics = engine.evaluate(fm, labels, drop_cols=["corrected_count"])
        for key in ("precision", "recall", "f1", "roc_auc", "avg_precision"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0


class TestFeatureImportance:
    def test_returns_series(self):
        fm = make_feature_matrix()
        engine = MLPredictionEngine(use_xgboost=False)
        labels = engine.build_labels(fm)
        engine.fit(fm, labels, drop_cols=["corrected_count"])
        imp = engine.get_feature_importance()
        assert isinstance(imp, pd.Series)

    def test_importance_before_fit_returns_empty(self):
        engine = MLPredictionEngine(use_xgboost=False)
        # _model is None — should return empty series without raising
        engine._fitted = True
        engine._model = None
        imp = engine.get_feature_importance()
        assert len(imp) == 0
