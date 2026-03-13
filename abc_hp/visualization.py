"""
M7 — Visualization & Reporting
================================
Generates interactive hotspot risk maps and summary dashboards.

Output types
------------
* **Folium map** — interactive choropleth / circle-marker hotspot map
* **Plotly Dash app** — web dashboard with time-sliced views
* **Matplotlib figures** — static figures for research papers
* **GeoJSON / CSV export** — machine-readable risk scores

All heavy plotting dependencies (folium, plotly, dash) are imported lazily
so that the rest of the ABC-HP system remains importable in environments
where only the core science stack is installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default colour ramp from low risk (green) to high risk (red)
_RISK_COLOURS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e1a0e"]


class HotspotVisualizer:
    """Generate risk maps and reports for the ABC-HP prediction output.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory where output files are saved.  Defaults to ``./output``.
    """

    def __init__(self, output_dir: Optional[str | Path] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Folium — interactive hotspot map
    # ------------------------------------------------------------------

    def folium_risk_map(
        self,
        risk_df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        risk_col: str = "risk_score",
        title: str = "ABC-HP Hotspot Risk Map",
        output_filename: str = "hotspot_map.html",
    ):
        """Create an interactive Folium choropleth map.

        Parameters
        ----------
        risk_df : pd.DataFrame
            Must contain latitude/longitude centroid columns and a numeric
            risk score column.
        lat_col, lon_col : str
            Column names for cell centroids.
        risk_col : str
            Column holding the risk probability (0–1).
        title : str
            Map title shown in a top panel.
        output_filename : str
            HTML file name written to ``self.output_dir``.

        Returns
        -------
        folium.Map
        """
        try:
            import folium  # noqa: PLC0415
            from folium.plugins import HeatMap  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "folium is required for interactive maps. "
                "Install it with: pip install folium"
            ) from exc

        df = risk_df.dropna(subset=[lat_col, lon_col, risk_col])
        centre_lat = df[lat_col].mean()
        centre_lon = df[lon_col].mean()

        m = folium.Map(location=[centre_lat, centre_lon], zoom_start=11, tiles="CartoDB dark_matter")

        # Heat map layer
        heat_data = df[[lat_col, lon_col, risk_col]].values.tolist()
        HeatMap(heat_data, radius=15, blur=20, max_zoom=13).add_to(m)

        # Circle markers for top hotspots
        top_hotspots = df.nlargest(min(50, len(df)), risk_col)
        for _, row in top_hotspots.iterrows():
            colour = self._risk_colour(row[risk_col])
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=6 + row[risk_col] * 10,
                color=colour,
                fill=True,
                fill_opacity=0.7,
                tooltip=f"Risk: {row[risk_col]:.2f}",
            ).add_to(m)

        # Title overlay
        title_html = (
            f'<h3 style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
            f'background:rgba(0,0,0,0.6);color:white;padding:8px 16px;'
            f'border-radius:6px;z-index:1000;">{title}</h3>'
        )
        m.get_root().html.add_child(folium.Element(title_html))

        out_path = self.output_dir / output_filename
        m.save(str(out_path))
        logger.info("Folium map saved to %s", out_path)
        return m

    # ------------------------------------------------------------------
    # Matplotlib — static research figures
    # ------------------------------------------------------------------

    def plot_risk_distribution(
        self,
        risk_df: pd.DataFrame,
        risk_col: str = "risk_score",
        output_filename: str = "risk_distribution.png",
    ):
        """Plot the distribution of risk scores across all grid cells.

        Parameters
        ----------
        risk_df : pd.DataFrame
        risk_col : str
        output_filename : str

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for static figures. "
                "Install it with: pip install matplotlib"
            ) from exc

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        scores = risk_df[risk_col].dropna()

        # Histogram
        axes[0].hist(scores, bins=30, color="#e74c3c", edgecolor="white", alpha=0.8)
        axes[0].set_xlabel("Risk Score")
        axes[0].set_ylabel("Number of Grid Cells")
        axes[0].set_title("Distribution of Risk Scores")
        axes[0].axvline(scores.median(), color="black", linestyle="--", label="Median")
        axes[0].legend()

        # Cumulative
        sorted_scores = np.sort(scores)
        cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[1].plot(sorted_scores, cdf, color="#e74c3c", linewidth=2)
        axes[1].set_xlabel("Risk Score")
        axes[1].set_ylabel("Cumulative Proportion")
        axes[1].set_title("Cumulative Risk Score Distribution")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("ABC-HP Risk Score Summary", fontsize=13, fontweight="bold")
        fig.tight_layout()

        out_path = self.output_dir / output_filename
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        logger.info("Risk distribution figure saved to %s", out_path)
        return fig

    def plot_feature_importance(
        self,
        importances: pd.Series,
        top_n: int = 15,
        output_filename: str = "feature_importance.png",
    ):
        """Horizontal bar chart of feature importances.

        Parameters
        ----------
        importances : pd.Series
            From :meth:`MLPredictionEngine.get_feature_importance`.
        top_n : int
            Number of top features to display.
        output_filename : str

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required. Install it with: pip install matplotlib"
            ) from exc

        top = importances.head(top_n).sort_values()
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
        colours = [
            "#e74c3c" if v >= top.quantile(0.75) else "#3498db" for v in top.values
        ]
        ax.barh(top.index, top.values, color=colours, edgecolor="white")
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        out_path = self.output_dir / output_filename
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        logger.info("Feature importance figure saved to %s", out_path)
        return fig

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------

    def export_risk_scores(
        self,
        risk_df: pd.DataFrame,
        fmt: str = "csv",
        output_filename: Optional[str] = None,
    ) -> Path:
        """Export the risk score DataFrame to CSV or GeoJSON.

        Parameters
        ----------
        risk_df : pd.DataFrame
        fmt : str
            ``"csv"`` or ``"geojson"``.
        output_filename : str, optional
            Defaults to ``risk_scores.csv`` / ``risk_scores.geojson``.

        Returns
        -------
        Path
            Path to the saved file.
        """
        if fmt not in ("csv", "geojson"):
            raise ValueError("fmt must be 'csv' or 'geojson'.")

        if output_filename is None:
            output_filename = f"risk_scores.{fmt}"

        out_path = self.output_dir / output_filename

        if fmt == "csv":
            risk_df.to_csv(str(out_path), index=True)
            logger.info("Risk scores exported to CSV: %s", out_path)
        else:
            try:
                import geopandas as gpd  # noqa: PLC0415
                from shapely.geometry import Point  # noqa: PLC0415

                gdf = gpd.GeoDataFrame(
                    risk_df,
                    geometry=[
                        Point(row["longitude"], row["latitude"])
                        for _, row in risk_df.iterrows()
                    ],
                    crs="EPSG:4326",
                )
                gdf.to_file(str(out_path), driver="GeoJSON")
                logger.info("Risk scores exported to GeoJSON: %s", out_path)
            except ImportError as exc:
                raise ImportError(
                    "geopandas and shapely are required for GeoJSON export."
                ) from exc

        return out_path

    def generate_summary_report(
        self,
        risk_df: pd.DataFrame,
        risk_col: str = "risk_score",
        output_filename: str = "summary_report.txt",
    ) -> str:
        """Write a plain-text summary of the hotspot prediction results.

        Parameters
        ----------
        risk_df : pd.DataFrame
        risk_col : str
        output_filename : str

        Returns
        -------
        str
            Report text.
        """
        scores = risk_df[risk_col].dropna()
        high_risk = (scores >= 0.75).sum()
        medium_risk = ((scores >= 0.50) & (scores < 0.75)).sum()
        low_risk = (scores < 0.50).sum()

        lines = [
            "=" * 60,
            "  ABC-HP: Adaptive Bias-Corrected Hotspot Prediction",
            "  Summary Report",
            "=" * 60,
            f"  Total grid cells analysed : {len(scores):,}",
            f"  High-risk cells (≥0.75)   : {high_risk:,}",
            f"  Medium-risk cells (0.50–0.75): {medium_risk:,}",
            f"  Low-risk cells (<0.50)    : {low_risk:,}",
            "",
            f"  Mean risk score  : {scores.mean():.4f}",
            f"  Median risk score: {scores.median():.4f}",
            f"  Max risk score   : {scores.max():.4f}",
            "=" * 60,
        ]
        report = "\n".join(lines)

        out_path = self.output_dir / output_filename
        out_path.write_text(report)
        logger.info("Summary report saved to %s", out_path)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_colour(score: float) -> str:
        """Map a 0–1 risk score to an HTML colour from the risk ramp."""
        idx = min(int(score * len(_RISK_COLOURS)), len(_RISK_COLOURS) - 1)
        return _RISK_COLOURS[idx]
