"""
M1 — Data Acquisition
=====================
Handles downloading and loading of accident records and contextual datasets
from public sources (Kaggle US Accidents, STATS19, FARS, OSM, weather APIs).

Only the *loading/schema-normalisation* layer is implemented here; heavy
network calls are guarded by thin adapter methods that can be replaced with
real API clients in production.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column aliases to normalise heterogeneous source schemas
# ---------------------------------------------------------------------------
_SCHEMA_MAP: Dict[str, Dict[str, str]] = {
    "kaggle_us": {
        "Start_Lat": "latitude",
        "Start_Lng": "longitude",
        "Severity": "severity",
        "Weather_Condition": "weather",
        "Start_Time": "datetime",
        "ID": "accident_id",
    },
    "stats19": {
        "latitude": "latitude",
        "longitude": "longitude",
        "accident_severity": "severity",
        "date": "datetime",
        "accident_index": "accident_id",
    },
    "fars": {
        "LATITUDE": "latitude",
        "LONGITUD": "longitude",
        "FATALS": "severity",
        "HOUR": "hour",
        "YEAR": "year",
        "ST_CASE": "accident_id",
    },
}

# Canonical columns required by downstream modules
REQUIRED_COLUMNS: List[str] = ["accident_id", "latitude", "longitude", "datetime", "severity"]


class DataAcquisition:
    """Load and normalise accident records from multiple sources.

    Parameters
    ----------
    data_dir : str or Path, optional
        Root directory containing raw data files.  Defaults to ``./data/raw``.
    """

    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path("data/raw")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_csv(self, filepath: str | Path, source: str = "kaggle_us") -> pd.DataFrame:
        """Load a CSV accident dataset and normalise its column names.

        Parameters
        ----------
        filepath : str or Path
            Path to the CSV file.
        source : str
            Schema identifier: ``"kaggle_us"``, ``"stats19"``, or ``"fars"``.
            Controls which column-alias map is applied.

        Returns
        -------
        pd.DataFrame
            Normalised DataFrame with :data:`REQUIRED_COLUMNS` where available.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Accident data file not found: {filepath}")

        logger.info("Loading %s dataset from %s", source, filepath)
        df = pd.read_csv(filepath, low_memory=False)
        df = self._rename_columns(df, source)
        df = self._coerce_types(df)
        logger.info("Loaded %d records from %s", len(df), filepath)
        return df

    def load_multiple(
        self, file_source_pairs: List[tuple[str | Path, str]]
    ) -> pd.DataFrame:
        """Load and concatenate accident records from multiple files.

        Parameters
        ----------
        file_source_pairs : list of (filepath, source) tuples

        Returns
        -------
        pd.DataFrame
            Combined, normalised DataFrame.
        """
        frames: List[pd.DataFrame] = []
        for filepath, source in file_source_pairs:
            try:
                frames.append(self.load_csv(filepath, source))
            except FileNotFoundError as exc:
                logger.warning("Skipping missing file: %s", exc)
        if not frames:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["accident_id"], keep="first")
        logger.info("Combined dataset: %d records total", len(combined))
        return combined

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows that are missing mandatory coordinate or identifier columns.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with invalid rows removed.
        """
        mandatory = ["latitude", "longitude", "accident_id"]
        before = len(df)
        df = df.dropna(subset=[c for c in mandatory if c in df.columns])
        dropped = before - len(df)
        if dropped:
            logger.info("Validation dropped %d rows missing mandatory fields", dropped)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rename_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
        mapping = _SCHEMA_MAP.get(source, {})
        df = df.rename(columns=mapping)
        return df

    @staticmethod
    def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for col in ("latitude", "longitude"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "severity" in df.columns:
            df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
        return df
