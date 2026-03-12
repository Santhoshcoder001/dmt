"""
Tests for M3 — BiasCorrector
"""

import numpy as np
import pandas as pd
import pytest

from abc_hp.bias_correction import BiasCorrector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "pop_density": rng.uniform(100, 10000, n),
            "road_type_encoded": rng.integers(0, 5, n),
            "hour": rng.integers(0, 24, n),
            "is_weekend": rng.integers(0, 2, n),
            "weather_encoded": rng.integers(0, 8, n),
            "grid_id": [f"cell_{i % 20}" for i in range(n)],
        }
    )
    # Simulate is_reported: higher pop_density → higher reporting probability
    prob = 0.3 + 0.5 * (df["pop_density"] / df["pop_density"].max())
    df["is_reported"] = (rng.uniform(size=n) < prob).astype(int)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBiasCorrectorFit:
    def test_fit_returns_self(self):
        df = make_df()
        bc = BiasCorrector()
        result = bc.fit(df)
        assert result is bc

    def test_fitted_flag_set(self):
        df = make_df()
        bc = BiasCorrector()
        bc.fit(df)
        assert bc._fitted is True

    def test_predict_before_fit_raises(self):
        df = make_df()
        bc = BiasCorrector()
        with pytest.raises(RuntimeError, match="fitted"):
            bc.predict_reporting_probability(df)


class TestReportingProbability:
    def test_probabilities_in_range(self):
        df = make_df()
        bc = BiasCorrector()
        bc.fit(df)
        probs = bc.predict_reporting_probability(df)
        assert (probs >= 0.01).all()
        assert (probs <= 1.0).all()

    def test_probabilities_shape(self):
        df = make_df()
        bc = BiasCorrector()
        bc.fit(df)
        probs = bc.predict_reporting_probability(df)
        assert len(probs) == len(df)


class TestWeightComputation:
    def test_weights_at_least_one(self):
        df = make_df()
        bc = BiasCorrector()
        bc.fit(df)
        weights = bc.compute_weights(df)
        assert (weights >= 1.0).all()

    def test_weights_clipped_at_max(self):
        df = make_df()
        bc = BiasCorrector(weight_clip_max=50.0)
        bc.fit(df)
        weights = bc.compute_weights(df)
        assert (weights <= 50.0).all()

    def test_apply_weights_adds_column(self):
        df = make_df()
        bc = BiasCorrector()
        bc.fit(df)
        result = bc.apply_weights(df)
        assert "bias_weight" in result.columns
        assert len(result) == len(df)


class TestAggregatedCounts:
    def test_corrected_counts_indexed_by_grid(self):
        df = make_df()
        bc = BiasCorrector()
        bc.fit(df)
        df = bc.apply_weights(df)
        counts = bc.aggregate_weighted_counts(df, grid_col="grid_id")
        assert counts.index.name == "grid_id"
        assert (counts > 0).all()

    def test_aggregate_without_weight_col_raises(self):
        df = make_df()
        bc = BiasCorrector()
        bc.fit(df)
        with pytest.raises(ValueError, match="apply_weights"):
            bc.aggregate_weighted_counts(df)

    def test_aggregate_missing_grid_col_raises(self):
        df = make_df()
        bc = BiasCorrector()
        bc.fit(df)
        df = bc.apply_weights(df)
        with pytest.raises(ValueError, match="nonexistent"):
            bc.aggregate_weighted_counts(df, grid_col="nonexistent")


class TestEncodeCategoricals:
    def test_road_type_encoded_added(self):
        df = pd.DataFrame({"road_type": ["primary", "secondary", "residential"]})
        result = BiasCorrector.encode_categoricals(df)
        assert "road_type_encoded" in result.columns
        assert result["road_type_encoded"].dtype in (int, "int8", "int16", "int32", "int64")

    def test_weather_encoded_added(self):
        df = pd.DataFrame({"weather": ["Clear", "Rain", "Fog", "Clear"]})
        result = BiasCorrector.encode_categoricals(df)
        assert "weather_encoded" in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"road_type": ["primary"]})
        _ = BiasCorrector.encode_categoricals(df)
        assert "road_type_encoded" not in df.columns
