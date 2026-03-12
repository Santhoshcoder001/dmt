"""
Tests for M4 — SpatialGridEngine
"""

import pandas as pd
import pytest

from abc_hp.spatial_grid import SpatialGridEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_accident_df():
    return pd.DataFrame(
        {
            "accident_id": list(range(10)),
            "latitude": [37.33, 37.34, 37.35, 37.36, 37.37, 34.05, 34.06, 51.50, 51.51, 51.52],
            "longitude": [-122.03, -122.04, -122.05, -122.06, -122.07,
                          -118.24, -118.25, -0.13, -0.14, -0.15],
            "bias_weight": [1.2, 1.5, 2.0, 1.1, 1.3, 1.8, 1.4, 1.0, 1.6, 1.7],
        }
    )


# ---------------------------------------------------------------------------
# Tests — Square grid (no h3 dependency)
# ---------------------------------------------------------------------------

class TestSquareGrid:
    def test_assigns_grid_id_column(self):
        df = make_accident_df()
        engine = SpatialGridEngine(grid_type="square", cell_size_deg=0.1)
        result = engine.assign_cells(df)
        assert "grid_id" in result.columns

    def test_grid_id_format(self):
        df = make_accident_df()
        engine = SpatialGridEngine(grid_type="square", cell_size_deg=0.1)
        result = engine.assign_cells(df)
        for gid in result["grid_id"]:
            parts = gid.split(":")
            assert len(parts) == 2, f"Unexpected grid_id format: {gid}"

    def test_nearby_points_same_cell(self):
        """Two very close points should fall in the same square cell."""
        df = pd.DataFrame(
            {
                "accident_id": [1, 2],
                "latitude": [37.331, 37.332],
                "longitude": [-122.031, -122.032],
            }
        )
        engine = SpatialGridEngine(grid_type="square", cell_size_deg=0.5)
        result = engine.assign_cells(df)
        assert result["grid_id"].iloc[0] == result["grid_id"].iloc[1]

    def test_distant_points_different_cells(self):
        """Points in different cities should fall in different cells."""
        df = pd.DataFrame(
            {
                "accident_id": [1, 2],
                "latitude": [37.33, 51.50],
                "longitude": [-122.03, -0.13],
            }
        )
        engine = SpatialGridEngine(grid_type="square", cell_size_deg=0.5)
        result = engine.assign_cells(df)
        assert result["grid_id"].iloc[0] != result["grid_id"].iloc[1]

    def test_row_count_preserved(self):
        df = make_accident_df()
        engine = SpatialGridEngine(grid_type="square")
        result = engine.assign_cells(df)
        assert len(result) == len(df)


class TestGridSummary:
    def test_accident_count_column(self):
        df = make_accident_df()
        engine = SpatialGridEngine(grid_type="square", cell_size_deg=0.5)
        df = engine.assign_cells(df)
        summary = engine.build_grid_summary(df)
        assert "accident_count" in summary.columns
        assert summary["accident_count"].sum() == len(df)

    def test_corrected_count_when_weight_present(self):
        df = make_accident_df()
        engine = SpatialGridEngine(grid_type="square", cell_size_deg=0.5)
        df = engine.assign_cells(df)
        summary = engine.build_grid_summary(df, weight_col="bias_weight")
        assert "corrected_count" in summary.columns
        assert summary["corrected_count"].sum() == pytest.approx(
            df["bias_weight"].sum(), rel=1e-6
        )

    def test_no_grid_id_raises(self):
        df = make_accident_df()
        engine = SpatialGridEngine(grid_type="square")
        with pytest.raises(ValueError, match="assign_cells"):
            engine.build_grid_summary(df)


class TestInvalidGridType:
    def test_invalid_grid_type_raises(self):
        with pytest.raises(ValueError):
            SpatialGridEngine(grid_type="triangle")


class TestH3GridOptional:
    def test_h3_assign_raises_import_error_when_not_installed(self):
        """Gracefully fail when h3 is not installed."""
        import importlib
        import sys

        # Temporarily hide the h3 module
        h3_mod = sys.modules.pop("h3", None)
        try:
            engine = SpatialGridEngine(grid_type="h3", resolution=8)
            df = make_accident_df()
            with pytest.raises(ImportError, match="h3"):
                engine.assign_cells(df)
        finally:
            if h3_mod is not None:
                sys.modules["h3"] = h3_mod

    def test_get_neighbours_raises_for_square_grid(self):
        engine = SpatialGridEngine(grid_type="square")
        with pytest.raises(NotImplementedError):
            engine.get_neighbours("123:456")
