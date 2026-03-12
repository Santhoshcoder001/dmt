"""
Tests for M1 — DataAcquisition
"""

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from abc_hp.data_acquisition import DataAcquisition, REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KAGGLE_CSV = """\
ID,Start_Lat,Start_Lng,Severity,Weather_Condition,Start_Time
A-1,37.337,-122.032,2,Clear,2021-01-01 08:00:00
A-2,34.052,-118.244,3,Rain,2021-01-02 17:30:00
A-3,,,-1,,
"""

STATS19_CSV = """\
accident_index,latitude,longitude,accident_severity,date
S-1,51.5074,-0.1278,1,2021-03-10
S-2,53.4808,-2.2426,2,2021-03-11
"""


def _write_tmp_csv(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataAcquisitionLoadCsv:
    def test_load_kaggle_renames_columns(self):
        path = _write_tmp_csv(KAGGLE_CSV)
        da = DataAcquisition()
        df = da.load_csv(path, source="kaggle_us")
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        assert "severity" in df.columns
        assert "accident_id" in df.columns

    def test_load_stats19_renames_columns(self):
        path = _write_tmp_csv(STATS19_CSV)
        da = DataAcquisition()
        df = da.load_csv(path, source="stats19")
        assert "latitude" in df.columns
        assert "accident_id" in df.columns

    def test_datetime_coercion(self):
        path = _write_tmp_csv(KAGGLE_CSV)
        da = DataAcquisition()
        df = da.load_csv(path, source="kaggle_us")
        assert pd.api.types.is_datetime64_any_dtype(df["datetime"])

    def test_missing_file_raises(self):
        da = DataAcquisition()
        with pytest.raises(FileNotFoundError):
            da.load_csv("/nonexistent/path/file.csv")

    def test_unknown_source_returns_original_columns(self):
        path = _write_tmp_csv(KAGGLE_CSV)
        da = DataAcquisition()
        df = da.load_csv(path, source="unknown_source")
        # No renaming — original column names preserved
        assert "Start_Lat" in df.columns


class TestDataAcquisitionValidate:
    def test_drops_rows_missing_coords(self):
        da = DataAcquisition()
        path = _write_tmp_csv(KAGGLE_CSV)
        df = da.load_csv(path, source="kaggle_us")
        valid = da.validate(df)
        # Row with empty lat/lon should be gone
        assert valid["latitude"].notna().all()
        assert valid["longitude"].notna().all()

    def test_valid_rows_preserved(self):
        da = DataAcquisition()
        path = _write_tmp_csv(KAGGLE_CSV)
        df = da.load_csv(path, source="kaggle_us")
        valid = da.validate(df)
        assert len(valid) == 2


class TestDataAcquisitionLoadMultiple:
    def test_concatenates_sources(self):
        p1 = _write_tmp_csv(KAGGLE_CSV)
        p2 = _write_tmp_csv(STATS19_CSV)
        da = DataAcquisition()
        df = da.load_multiple([(p1, "kaggle_us"), (p2, "stats19")])
        # 2 valid kaggle + 2 stats19 rows (duplicate check on accident_id)
        assert len(df) >= 2

    def test_skips_missing_files_gracefully(self):
        da = DataAcquisition()
        df = da.load_multiple([("/nonexistent/file.csv", "kaggle_us")])
        assert len(df) == 0
        assert list(df.columns) == REQUIRED_COLUMNS
