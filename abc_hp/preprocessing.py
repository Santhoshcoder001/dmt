"""
M2 — Preprocessing
==================
Cleans, aligns, and normalises all data sources into a unified
feature-ready geospatial dataset.

Steps
-----
1. Coordinate standardisation — reprojects to EPSG:4326
2. Temporal feature extraction — hour, weekday, month, weekend flag
3. Missing value handling — per-column strategies
4. Outlier detection — removes records >100 m from road network
5. Spatial data fusion — left-joins accident points to road attributes
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Public holidays placeholder — extend with ``holidays`` library if available
_DEFAULT_PUBLIC_HOLIDAYS: set = set()


class Preprocessor:
    """Transform raw accident records into a clean, feature-ready DataFrame.

    Parameters
    ----------
    max_road_distance_m : float
        Accidents recorded more than this distance from the road network are
        dropped as outliers.  Default: 100 m.
    public_holidays : set of str, optional
        ISO-formatted date strings (``"YYYY-MM-DD"``) considered public
        holidays.  Used to set the ``is_public_holiday`` flag.
    """

    def __init__(
        self,
        max_road_distance_m: float = 100.0,
        public_holidays: Optional[set] = None,
    ) -> None:
        self.max_road_distance_m = max_road_distance_m
        self.public_holidays: set = (
            public_holidays if public_holidays is not None else _DEFAULT_PUBLIC_HOLIDAYS
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw, normalised accident records from :class:`DataAcquisition`.

        Returns
        -------
        pd.DataFrame
            Cleaned, feature-enriched DataFrame.
        """
        df = df.copy()
        df = self._standardise_coordinates(df)
        df = self._extract_temporal_features(df)
        df = self._handle_missing_values(df)
        df = self._filter_road_outliers(df)
        logger.info("Preprocessing complete: %d records", len(df))
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 1: coordinate standardisation
    # ------------------------------------------------------------------

    @staticmethod
    def _standardise_coordinates(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows with valid WGS-84 coordinates."""
        before = len(df)
        df = df.dropna(subset=["latitude", "longitude"])
        df = df[
            df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
        ]
        logger.debug("Coordinate filter: kept %d / %d rows", len(df), before)
        return df

    # ------------------------------------------------------------------
    # Step 2: temporal feature extraction
    # ------------------------------------------------------------------

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive hour, day_of_week, month, is_weekend, is_public_holiday."""
        if "datetime" not in df.columns:
            logger.warning("'datetime' column missing — temporal features skipped")
            return df

        dt = pd.to_datetime(df["datetime"], errors="coerce")
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek  # 0=Monday … 6=Sunday
        df["month"] = dt.dt.month
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        df["is_public_holiday"] = (
            dt.dt.date.astype(str).isin(self.public_holidays)
        ).astype(int)
        return df

    # ------------------------------------------------------------------
    # Step 3: missing value handling
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Apply per-column imputation strategies from the specification."""
        # GPS — rows already dropped in step 1
        # Severity — mode imputation
        if "severity" in df.columns and df["severity"].isna().any():
            mode_val = df["severity"].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else 1
            df["severity"] = df["severity"].fillna(fill_val)

        # Weather — forward/backward fill as a simple time-based interpolation
        if "weather" in df.columns:
            df["weather"] = df["weather"].ffill().bfill()

        # Road type — fill with "unknown" until spatial join provides values
        if "road_type" in df.columns:
            df["road_type"] = df["road_type"].fillna("unknown")

        # Traffic speed/congestion — linear interpolation where numeric
        for col in ("speed", "congestion_index"):
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear", limit_direction="both")

        return df

    # ------------------------------------------------------------------
    # Step 4: outlier detection (distance to road network)
    # ------------------------------------------------------------------

    def _filter_road_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove accidents recorded far from the road network.

        When a ``road_distance_m`` column is present the threshold is applied
        directly.  Otherwise this step is a no-op (road distances must be
        computed externally via OSMnx / Shapely before calling this method).
        """
        if "road_distance_m" not in df.columns:
            logger.debug("'road_distance_m' not present — road-outlier filter skipped")
            return df

        before = len(df)
        df = df[df["road_distance_m"] <= self.max_road_distance_m]
        dropped = before - len(df)
        if dropped:
            logger.info(
                "Road-outlier filter: removed %d rows (threshold %.0f m)",
                dropped,
                self.max_road_distance_m,
            )
        return df

    # ------------------------------------------------------------------
    # Utility: spatial data fusion stub
    # ------------------------------------------------------------------

    @staticmethod
    def spatial_join(accidents_gdf, road_gdf, how: str = "left"):
        """Left-join accident points to road attributes via GeoPandas.

        This is a thin wrapper kept here for API completeness; callers must
        supply valid GeoDataFrames.

        Parameters
        ----------
        accidents_gdf : geopandas.GeoDataFrame
        road_gdf : geopandas.GeoDataFrame
        how : str
            Join type passed to ``gpd.sjoin``.

        Returns
        -------
        geopandas.GeoDataFrame
        """
        try:
            import geopandas as gpd  # noqa: PLC0415

            return gpd.sjoin(accidents_gdf, road_gdf, how=how, predicate="within")
        except ImportError as exc:
            raise ImportError(
                "geopandas is required for spatial_join. "
                "Install it with: pip install geopandas"
            ) from exc

    # ------------------------------------------------------------------
    # Utility: IDW weather interpolation helper
    # ------------------------------------------------------------------

    @staticmethod
    def idw_interpolate(
        values: np.ndarray,
        known_coords: np.ndarray,
        query_coords: np.ndarray,
        power: float = 2.0,
    ) -> np.ndarray:
        """Inverse Distance Weighting interpolation for weather data.

        Parameters
        ----------
        values : ndarray of shape (n,)
            Known weather values at observation points.
        known_coords : ndarray of shape (n, 2)
            (lat, lon) pairs for observation points.
        query_coords : ndarray of shape (m, 2)
            (lat, lon) pairs where values are to be estimated.
        power : float
            IDW power parameter.  Default: 2.

        Returns
        -------
        ndarray of shape (m,)
            Interpolated values at query locations.
        """
        interpolated = np.empty(len(query_coords))
        for i, qp in enumerate(query_coords):
            dists = np.sqrt(np.sum((known_coords - qp) ** 2, axis=1))
            if np.any(dists == 0):
                interpolated[i] = values[dists == 0][0]
            else:
                weights = 1.0 / dists**power
                interpolated[i] = np.dot(weights, values) / weights.sum()
        return interpolated
