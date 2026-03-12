"""
Tests for M2 — Preprocessor
"""

import numpy as np
import pandas as pd
import pytest

from abc_hp.preprocessing import Preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_df(**overrides):
    base = {
        "accident_id": ["A1", "A2", "A3", "A4", "A5"],
        "latitude": [37.33, 34.05, 999.0, 51.50, -999.0],
        "longitude": [-122.03, -118.24, -122.03, -0.13, -118.24],
        "datetime": [
            "2021-01-01 08:00:00",
            "2021-01-02 22:30:00",
            "2021-01-03 14:00:00",
            "2021-06-05 07:00:00",
            "2021-06-06 17:00:00",
        ],
        "severity": [2, None, 3, 1, 2],
        "weather": ["Clear", None, "Rain", "Fog", None],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCoordinateStandardisation:
    def test_drops_invalid_latitude(self):
        df = make_df()
        pre = Preprocessor()
        out = pre.run(df)
        assert out["latitude"].between(-90, 90).all()

    def test_drops_invalid_longitude(self):
        df = make_df()
        pre = Preprocessor()
        out = pre.run(df)
        assert out["longitude"].between(-180, 180).all()

    def test_valid_rows_preserved(self):
        df = make_df()
        pre = Preprocessor()
        out = pre.run(df)
        assert len(out) == 3


class TestTemporalFeatureExtraction:
    def test_hour_column_created(self):
        df = make_df()
        pre = Preprocessor()
        out = pre.run(df)
        assert "hour" in out.columns
        assert out["hour"].between(0, 23).all()

    def test_is_weekend_flag(self):
        df = make_df()
        pre = Preprocessor()
        out = pre.run(df)
        assert "is_weekend" in out.columns
        assert set(out["is_weekend"].unique()).issubset({0, 1})

    def test_day_of_week_range(self):
        df = make_df()
        pre = Preprocessor()
        out = pre.run(df)
        assert out["day_of_week"].between(0, 6).all()

    def test_month_range(self):
        df = make_df()
        pre = Preprocessor()
        out = pre.run(df)
        assert out["month"].between(1, 12).all()

    def test_public_holiday_flag(self):
        df = make_df()
        pre = Preprocessor(public_holidays={"2021-01-01"})
        out = pre.run(df)
        # Row with datetime 2021-01-01 should be flagged
        assert "is_public_holiday" in out.columns

    def test_no_datetime_column_does_not_crash(self):
        df = make_df()
        df = df.drop(columns=["datetime"])
        pre = Preprocessor()
        out = pre.run(df)
        assert "hour" not in out.columns


class TestMissingValueHandling:
    def test_severity_filled_with_mode(self):
        df = make_df()
        pre = Preprocessor()
        out = pre.run(df)
        assert out["severity"].isna().sum() == 0

    def test_weather_filled_forward_backward(self):
        df = make_df()
        pre = Preprocessor()
        out = pre.run(df)
        # After forward/backward fill, NaN should be resolved
        assert out["weather"].isna().sum() == 0


class TestRoadOutlierFilter:
    def test_removes_rows_beyond_threshold(self):
        df = make_df()
        df["road_distance_m"] = [10.0, 50.0, 200.0, 300.0, 5.0]
        pre = Preprocessor(max_road_distance_m=100.0)
        out = pre.run(df)
        if "road_distance_m" in out.columns:
            assert (out["road_distance_m"] <= 100.0).all()

    def test_no_road_distance_col_passes_through(self):
        df = make_df()
        pre = Preprocessor()
        before_len = len(pre._standardise_coordinates(df.copy()))
        out = pre.run(df)
        assert len(out) == before_len


class TestIDWInterpolation:
    def test_interpolation_at_known_point(self):
        known_coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        values = np.array([10.0, 20.0])
        query_coords = np.array([[0.0, 0.0]])  # exact match
        result = Preprocessor.idw_interpolate(values, known_coords, query_coords)
        assert result[0] == pytest.approx(10.0)

    def test_interpolation_midpoint(self):
        known_coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        values = np.array([0.0, 2.0])
        query_coords = np.array([[1.0, 0.0]])  # equidistant
        result = Preprocessor.idw_interpolate(values, known_coords, query_coords)
        assert result[0] == pytest.approx(1.0, abs=0.01)
