"""
M4 — Spatial Grid Engine
========================
Divides the study area into adaptive spatial prediction units.

Supported grid types
--------------------
* **H3 hexagonal grid** (recommended) — Uber H3 library
* **Square grid**                      — pure-Python fallback
* **Administrative zones**             — GeoPandas polygon overlay

The H3 hex grid is the primary implementation as it provides:
* Uniform area cells with equal-distance neighbours
* Hierarchical resolution (resolution 8 ≈ 0.74 km²)
* Efficient neighbour look-up for spatial-lag features
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class SpatialGridEngine:
    """Assign accident records to spatial grid cells.

    Parameters
    ----------
    resolution : int
        H3 hexagonal resolution (0 = coarsest, 15 = finest).
        Resolution 8 gives cells of ~0.74 km²; recommended for city-scale
        hotspot detection.  Default: 8.
    grid_type : str
        ``"h3"`` (default) or ``"square"``.
    cell_size_deg : float
        Side length of square cells in degrees.  Only used when
        ``grid_type="square"``.  Default: 0.01 (≈ 1.1 km at the equator).
    """

    def __init__(
        self,
        resolution: int = 8,
        grid_type: str = "h3",
        cell_size_deg: float = 0.01,
    ) -> None:
        if grid_type not in ("h3", "square"):
            raise ValueError("grid_type must be 'h3' or 'square'.")
        self.resolution = resolution
        self.grid_type = grid_type
        self.cell_size_deg = cell_size_deg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``grid_id`` column to *df* based on (latitude, longitude).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``latitude`` and ``longitude`` columns.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with a ``grid_id`` column.
        """
        df = df.copy()
        if self.grid_type == "h3":
            df["grid_id"] = self._assign_h3(df)
        else:
            df["grid_id"] = self._assign_square(df)
        logger.info(
            "Assigned %d records to %d unique %s grid cells",
            len(df),
            df["grid_id"].nunique(),
            self.grid_type,
        )
        return df

    def get_neighbours(self, cell_id: str, k: int = 1) -> List[str]:
        """Return the k-ring neighbours of *cell_id* (H3 only).

        Parameters
        ----------
        cell_id : str
            H3 cell index.
        k : int
            Ring distance.  k=1 returns the 6 immediate neighbours.

        Returns
        -------
        list of str
            Neighbour cell IDs (excludes the origin cell).
        """
        if self.grid_type != "h3":
            raise NotImplementedError("Neighbour look-up is only supported for H3 grids.")
        h3 = self._import_h3()
        ring = h3.grid_disk(cell_id, k)
        return [c for c in ring if c != cell_id]

    def cell_to_latlng(self, cell_id: str) -> Tuple[float, float]:
        """Return the (lat, lng) centroid of an H3 cell.

        Parameters
        ----------
        cell_id : str
            H3 cell index.

        Returns
        -------
        tuple of (float, float)
        """
        if self.grid_type != "h3":
            raise NotImplementedError("cell_to_latlng is only supported for H3 grids.")
        h3 = self._import_h3()
        lat, lng = h3.cell_to_latlng(cell_id)
        return lat, lng

    def build_grid_summary(
        self,
        df: pd.DataFrame,
        weight_col: Optional[str] = "bias_weight",
    ) -> pd.DataFrame:
        """Aggregate per-cell statistics from accident records.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``grid_id``.  Optionally ``bias_weight``.
        weight_col : str or None
            Column to sum as the corrected accident count.  When *None* or
            absent, the raw count is used.

        Returns
        -------
        pd.DataFrame
            One row per grid cell with columns: ``grid_id``,
            ``accident_count``, ``corrected_count`` (if *weight_col* present).
        """
        if "grid_id" not in df.columns:
            raise ValueError("Call assign_cells() before build_grid_summary().")

        summary = df.groupby("grid_id").size().rename("accident_count").reset_index()

        if weight_col and weight_col in df.columns:
            wsum = (
                df.groupby("grid_id")[weight_col]
                .sum()
                .rename("corrected_count")
                .reset_index()
            )
            summary = summary.merge(wsum, on="grid_id", how="left")

        logger.info("Grid summary: %d cells", len(summary))
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_h3(self, df: pd.DataFrame) -> pd.Series:
        h3 = self._import_h3()
        return df.apply(
            lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], self.resolution),
            axis=1,
        )

    def _assign_square(self, df: pd.DataFrame) -> pd.Series:
        """Assign square grid IDs encoded as ``"lat_bin:lon_bin"`` strings."""
        lat_bin = (df["latitude"] / self.cell_size_deg).apply(math.floor)
        lon_bin = (df["longitude"] / self.cell_size_deg).apply(math.floor)
        return lat_bin.astype(str) + ":" + lon_bin.astype(str)

    @staticmethod
    def _import_h3():
        try:
            import h3  # noqa: PLC0415

            return h3
        except ImportError as exc:
            raise ImportError(
                "The 'h3' package is required for H3 grid support. "
                "Install it with: pip install h3"
            ) from exc
