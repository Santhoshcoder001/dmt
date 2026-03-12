"""
Tests for M7 — HotspotVisualizer
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abc_hp.visualization import HotspotVisualizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_risk_df(n=30, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "grid_id": [f"cell_{i}" for i in range(n)],
            "latitude": rng.uniform(37.3, 37.5, n),
            "longitude": rng.uniform(-122.1, -122.0, n),
            "risk_score": rng.uniform(0, 1, n),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateSummaryReport:
    def test_report_contains_stats(self):
        df = make_risk_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = HotspotVisualizer(output_dir=tmpdir)
            report = viz.generate_summary_report(df)
        assert "ABC-HP" in report
        assert "Total grid cells" in report

    def test_high_risk_count_correct(self):
        df = make_risk_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = HotspotVisualizer(output_dir=tmpdir)
            report = viz.generate_summary_report(df)
        expected_high = int((df["risk_score"] >= 0.75).sum())
        assert str(expected_high) in report

    def test_report_file_written(self):
        df = make_risk_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = HotspotVisualizer(output_dir=tmpdir)
            viz.generate_summary_report(df, output_filename="test_report.txt")
            assert (Path(tmpdir) / "test_report.txt").exists()


class TestExportRiskScores:
    def test_csv_export_creates_file(self):
        df = make_risk_df().set_index("grid_id")
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = HotspotVisualizer(output_dir=tmpdir)
            out_path = viz.export_risk_scores(df, fmt="csv", output_filename="scores.csv")
            assert out_path.exists()

    def test_csv_content_matches(self):
        df = make_risk_df().set_index("grid_id")
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = HotspotVisualizer(output_dir=tmpdir)
            out_path = viz.export_risk_scores(df, fmt="csv")
            loaded = pd.read_csv(str(out_path), index_col=0)
        assert len(loaded) == len(df)

    def test_invalid_format_raises(self):
        df = make_risk_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = HotspotVisualizer(output_dir=tmpdir)
            with pytest.raises(ValueError, match="fmt"):
                viz.export_risk_scores(df, fmt="xlsx")


class TestOutputDirCreation:
    def test_output_dir_created_automatically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "nested" / "output"
            viz = HotspotVisualizer(output_dir=new_dir)
            assert new_dir.exists()


class TestRiskColour:
    def test_low_score_is_green(self):
        colour = HotspotVisualizer._risk_colour(0.0)
        assert colour == "#2ecc71"

    def test_high_score_is_dark_red(self):
        colour = HotspotVisualizer._risk_colour(1.0)
        assert colour == "#8e1a0e"

    def test_midrange_not_green_or_dark_red(self):
        colour = HotspotVisualizer._risk_colour(0.5)
        assert colour not in ("#2ecc71", "#8e1a0e")


class TestPlotRiskDistribution:
    def test_creates_figure(self):
        pytest.importorskip("matplotlib")
        df = make_risk_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = HotspotVisualizer(output_dir=tmpdir)
            fig = viz.plot_risk_distribution(df, output_filename="dist.png")
        import matplotlib.pyplot as plt
        assert fig is not None
        plt.close("all")

    def test_saves_png_file(self):
        pytest.importorskip("matplotlib")
        df = make_risk_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = HotspotVisualizer(output_dir=tmpdir)
            viz.plot_risk_distribution(df, output_filename="dist.png")
            assert (Path(tmpdir) / "dist.png").exists()
        import matplotlib.pyplot as plt
        plt.close("all")


class TestPlotFeatureImportance:
    def test_creates_figure(self):
        pytest.importorskip("matplotlib")
        importances = pd.Series(
            {"corrected_count": 0.4, "pop_density": 0.3, "peak_hour_ratio": 0.2, "night_ratio": 0.1}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = HotspotVisualizer(output_dir=tmpdir)
            fig = viz.plot_feature_importance(importances, output_filename="imp.png")
        import matplotlib.pyplot as plt
        assert fig is not None
        plt.close("all")
