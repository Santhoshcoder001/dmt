"""
M5 — Feature Engineering
=========================
Extracts spatio-temporal and contextual features for each H3 grid cell.

Each H3 cell becomes one row in the ML training dataset with 21+ features
covering:

| Category       | Features                                              |
|----------------|-------------------------------------------------------|
| Accident hist. | corrected_count, raw_count, accident_density          |
| Temporal       | peak_hour_ratio, weekend_ratio, night_ratio           |
| Weather        | rain_accident_ratio, fog_ratio, severe_weather_ratio  |
| Traffic        | congestion_index, mean_speed                          |
| Infrastructure | junction_count, road_length_km, road_type_primary     |
| Population     | pop_density                                           |
| POI proximity  | school_within_500m, hospital_within_500m              |
| Spatial lag    | neighbour_corrected_count (mean of k-ring neighbours) |
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Hour ranges used for temporal feature computation
_PEAK_HOURS = frozenset(range(7, 10)) | frozenset(range(16, 20))  # AM/PM peaks
_NIGHT_HOURS = frozenset(range(0, 6)) | frozenset({22, 23})


class FeatureEngineer:
    """Build the per-cell feature matrix from accident records and a grid summary.

    Parameters
    ----------
    grid_engine : SpatialGridEngine, optional
        Used to look up H3 neighbour cells for spatial-lag features.
        When *None*, spatial-lag features are skipped.
    """

    def __init__(self, grid_engine=None) -> None:
        self.grid_engine = grid_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        grid_summary: pd.DataFrame,
        contextual: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Construct the ML feature matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed, bias-weighted accident records.  Must contain
            ``grid_id``.
        grid_summary : pd.DataFrame
            Per-cell aggregate produced by
            :meth:`SpatialGridEngine.build_grid_summary`.
        contextual : pd.DataFrame, optional
            Per-cell contextual features (pop_density, junction_count, etc.).
            Merged on ``grid_id`` when provided.

        Returns
        -------
        pd.DataFrame
            Feature matrix indexed by ``grid_id`` with one row per cell.
        """
        features = grid_summary.copy()

        # Accident-history features
        features = self._add_accident_history(features, df)

        # Temporal features
        features = self._add_temporal_features(features, df)

        # Weather features
        features = self._add_weather_features(features, df)

        # Merge contextual (infrastructure, population, POI, traffic)
        if contextual is not None:
            features = features.merge(contextual, on="grid_id", how="left")

        # Spatial-lag feature
        if self.grid_engine is not None:
            features = self._add_spatial_lag(features)

        # Fill any remaining NaN with 0
        features = features.fillna(0)

        logger.info(
            "Feature matrix: %d cells × %d features",
            len(features),
            len(features.columns) - 1,  # exclude grid_id
        )
        return features.set_index("grid_id")

    # ------------------------------------------------------------------
    # Accident history features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_accident_history(
        features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        if "corrected_count" not in features.columns:
            features["corrected_count"] = (
                features["accident_count"] if "accident_count" in features.columns else 0
            )
        # Accident density proxy (normalised by max count)
        max_count = features["corrected_count"].max()
        features["accident_density"] = (
            features["corrected_count"] / max_count if max_count > 0 else 0.0
        )
        return features

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_temporal_features(
        features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        if "grid_id" not in df.columns:
            return features

        temporal_agg: dict = {}

        if "hour" in df.columns:
            df_tmp = df.copy()
            df_tmp["_is_peak"] = df_tmp["hour"].isin(_PEAK_HOURS).astype(int)
            df_tmp["_is_night"] = df_tmp["hour"].isin(_NIGHT_HOURS).astype(int)

            peak = df_tmp.groupby("grid_id")["_is_peak"].mean().rename("peak_hour_ratio")
            night = df_tmp.groupby("grid_id")["_is_night"].mean().rename("night_ratio")
            temporal_agg["peak_hour_ratio"] = peak
            temporal_agg["night_ratio"] = night

        if "is_weekend" in df.columns:
            weekend = df.groupby("grid_id")["is_weekend"].mean().rename("weekend_ratio")
            temporal_agg["weekend_ratio"] = weekend

        if temporal_agg:
            t_df = pd.concat(temporal_agg.values(), axis=1).reset_index()
            features = features.merge(t_df, on="grid_id", how="left")

        return features

    # ------------------------------------------------------------------
    # Weather features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_weather_features(
        features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        if "weather" not in df.columns or "grid_id" not in df.columns:
            return features

        weather_lower = df["weather"].fillna("").str.lower()
        df_tmp = df.copy()
        df_tmp["_is_rain"] = weather_lower.str.contains("rain|drizzle", regex=True).astype(int)
        df_tmp["_is_fog"] = weather_lower.str.contains("fog|mist", regex=True).astype(int)
        df_tmp["_is_severe"] = weather_lower.str.contains(
            "storm|snow|ice|hail|thunder", regex=True
        ).astype(int)

        weather_agg = (
            df_tmp.groupby("grid_id")[["_is_rain", "_is_fog", "_is_severe"]]
            .mean()
            .rename(
                columns={
                    "_is_rain": "rain_accident_ratio",
                    "_is_fog": "fog_ratio",
                    "_is_severe": "severe_weather_ratio",
                }
            )
            .reset_index()
        )
        features = features.merge(weather_agg, on="grid_id", how="left")
        return features

    # ------------------------------------------------------------------
    # Spatial lag feature
    # ------------------------------------------------------------------

    def _add_spatial_lag(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add mean corrected_count of H3 k=1 ring neighbours."""
        if "corrected_count" not in features.columns:
            return features

        count_map = features.set_index("grid_id")["corrected_count"].to_dict()
        lag_values: List[float] = []

        for cell_id in features["grid_id"]:
            try:
                neighbours = self.grid_engine.get_neighbours(cell_id, k=1)
                neighbour_counts = [count_map.get(n, 0.0) for n in neighbours]
                lag_values.append(float(np.mean(neighbour_counts)) if neighbour_counts else 0.0)
            except Exception:
                lag_values.append(0.0)

        features = features.copy()
        features["neighbour_corrected_count"] = lag_values
        return features

    # ------------------------------------------------------------------
    # Utility: build contextual feature template
    # ------------------------------------------------------------------

    @staticmethod
    def make_contextual_template(grid_ids: List[str]) -> pd.DataFrame:
        """Return an empty contextual-features DataFrame for the given cells.

        Users populate this template with real values from OSM, WorldPop, HERE
        Maps, etc.  Columns with NaN will be filled with 0 by
        :meth:`build_feature_matrix`.

        Parameters
        ----------
        grid_ids : list of str

        Returns
        -------
        pd.DataFrame
        """
        return pd.DataFrame(
            {
                "grid_id": grid_ids,
                "pop_density": np.nan,
                "junction_count": np.nan,
                "road_length_km": np.nan,
                "road_type_primary": np.nan,
                "school_within_500m": np.nan,
                "hospital_within_500m": np.nan,
                "congestion_index": np.nan,
                "mean_speed": np.nan,
            }
        )
