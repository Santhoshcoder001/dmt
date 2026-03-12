"""
Tests for M5 — FeatureEngineer
"""

import numpy as np
import pandas as pd
import pytest

from abc_hp.feature_engineering import FeatureEngineer, _PEAK_HOURS, _NIGHT_HOURS
from abc_hp.spatial_grid import SpatialGridEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_accident_df(n=60, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "accident_id": range(n),
            "latitude": rng.uniform(37.3, 37.5, n),
            "longitude": rng.uniform(-122.1, -122.0, n),
            "hour": rng.integers(0, 24, n),
            "is_weekend": rng.integers(0, 2, n),
            "weather": rng.choice(["Clear", "Rain", "Fog", "Storm", None], n),
            "bias_weight": rng.uniform(1.0, 3.0, n),
        }
    )


def make_grid_and_accidents():
    df = make_accident_df()
    engine = SpatialGridEngine(grid_type="square", cell_size_deg=0.05)
    df = engine.assign_cells(df)
    summary = engine.build_grid_summary(df, weight_col="bias_weight")
    return df, summary, engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_returns_dataframe(self):
        df, summary, engine = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert isinstance(fm, pd.DataFrame)

    def test_index_is_grid_id(self):
        df, summary, engine = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert fm.index.name == "grid_id"

    def test_one_row_per_cell(self):
        df, summary, engine = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert len(fm) == summary["grid_id"].nunique()

    def test_no_nan_in_output(self):
        df, summary, engine = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert fm.isna().sum().sum() == 0

    def test_accident_density_in_range(self):
        df, summary, engine = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert "accident_density" in fm.columns
        assert fm["accident_density"].between(0, 1).all()


class TestTemporalFeatures:
    def test_peak_hour_ratio_present(self):
        df, summary, _ = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert "peak_hour_ratio" in fm.columns

    def test_peak_hour_ratio_in_range(self):
        df, summary, _ = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert fm["peak_hour_ratio"].between(0, 1).all()

    def test_night_ratio_present(self):
        df, summary, _ = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert "night_ratio" in fm.columns

    def test_weekend_ratio_in_range(self):
        df, summary, _ = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert "weekend_ratio" in fm.columns
        assert fm["weekend_ratio"].between(0, 1).all()


class TestWeatherFeatures:
    def test_rain_ratio_present(self):
        df, summary, _ = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert "rain_accident_ratio" in fm.columns

    def test_fog_ratio_in_range(self):
        df, summary, _ = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert "fog_ratio" in fm.columns
        assert fm["fog_ratio"].between(0, 1).all()

    def test_severe_weather_ratio_present(self):
        df, summary, _ = make_grid_and_accidents()
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary)
        assert "severe_weather_ratio" in fm.columns


class TestContextualFeatures:
    def test_contextual_merged(self):
        df, summary, _ = make_grid_and_accidents()
        contextual = FeatureEngineer.make_contextual_template(
            summary["grid_id"].tolist()
        )
        contextual["pop_density"] = 1000.0
        fe = FeatureEngineer()
        fm = fe.build_feature_matrix(df, summary, contextual=contextual)
        assert "pop_density" in fm.columns

    def test_make_contextual_template_correct_columns(self):
        template = FeatureEngineer.make_contextual_template(["cell_1", "cell_2"])
        assert "grid_id" in template.columns
        assert "pop_density" in template.columns
        assert len(template) == 2


class TestPeakHourConstants:
    def test_peak_hours_non_empty(self):
        assert len(_PEAK_HOURS) > 0

    def test_night_hours_non_empty(self):
        assert len(_NIGHT_HOURS) > 0

    def test_peak_and_night_disjoint(self):
        assert _PEAK_HOURS.isdisjoint(_NIGHT_HOURS)
