"""
Tests for core database export utilities.
"""

import json
from datetime import datetime

import numpy as np
import pytest

from gently.core.database import (
    add_embryo_to_database,
    format_embryo_calibration_for_json,
    format_embryo_entry_for_json,
    format_timestamp,
    get_embryo_calibration,
    list_embryos,
    load_multi_embryo_database,
    numpy_to_python,
    save_multi_embryo_database,
)


class TestNumpyToPython:
    """numpy_to_python converts numpy types for JSON serialization."""

    def test_ndarray(self):
        assert numpy_to_python(np.array([1, 2, 3])) == [1, 2, 3]

    def test_integer(self):
        assert numpy_to_python(np.int64(42)) == 42
        assert isinstance(numpy_to_python(np.int64(42)), int)

    def test_float(self):
        assert numpy_to_python(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(numpy_to_python(np.float64(3.14)), float)

    def test_bool(self):
        assert numpy_to_python(np.bool_(True)) is True
        assert isinstance(numpy_to_python(np.bool_(True)), bool)

    def test_nested_dict(self):
        d = {"a": np.int64(1), "b": np.array([2, 3])}
        result = numpy_to_python(d)
        assert result == {"a": 1, "b": [2, 3]}

    def test_nested_list(self):
        result = numpy_to_python([np.float64(1.0), np.int64(2)])
        assert result == [1.0, 2]

    def test_passthrough_native(self):
        assert numpy_to_python("hello") == "hello"
        assert numpy_to_python(42) == 42

    def test_result_is_json_serializable(self):
        data = {
            "array": np.zeros(3),
            "int": np.int32(5),
            "float": np.float32(2.5),
            "nested": {"x": np.array([1, 2])},
        }
        result = numpy_to_python(data)
        # Must not raise
        json.dumps(result)


class TestFormatTimestamp:
    """format_timestamp returns ISO 8601 strings."""

    def test_default_now(self):
        ts = format_timestamp()
        # Should be parseable
        datetime.fromisoformat(ts)

    def test_specific_time(self):
        dt = datetime(2025, 6, 15, 12, 30, 45)
        ts = format_timestamp(dt)
        assert "2025-06-15" in ts


class TestFormatCalibration:
    """format_embryo_calibration_for_json fills missing fields."""

    def test_fills_missing_required_fields(self):
        result = format_embryo_calibration_for_json({})
        assert "slope_um_per_deg" in result
        assert "timestamp" in result
        assert result["sample_type"] == "embryo"

    def test_preserves_existing_data(self):
        data = {"slope_um_per_deg": 95.5, "offset_um": 3.0}
        result = format_embryo_calibration_for_json(data)
        assert result["slope_um_per_deg"] == 95.5

    def test_converts_numpy_values(self):
        data = {"slope_um_per_deg": np.float64(95.5)}
        result = format_embryo_calibration_for_json(data)
        assert isinstance(result["slope_um_per_deg"], float)


class TestFormatEmbryoEntry:
    """format_embryo_entry_for_json structures embryo data."""

    def test_basic_entry(self):
        data = {
            "embryo_number": 1,
            "pixel_x": 500.0,
            "pixel_y": 300.0,
            "initial_stage_x": 1000.0,
            "initial_stage_y": 200.0,
        }
        result = format_embryo_entry_for_json(data)
        assert result["embryo_number"] == 1
        assert result["bottom_camera_position_pixel"]["x"] == 500.0

    def test_with_calibration(self):
        data = {"embryo_number": 1, "calibration": {"slope_um_per_deg": 95.0}}
        result = format_embryo_entry_for_json(data)
        assert "calibration" in result


class TestDatabaseFileOps:
    """Load/save/modify multi-embryo database."""

    def test_load_missing_returns_empty(self, tmp_path):
        db = load_multi_embryo_database(tmp_path / "nonexistent.json")
        assert db["embryos"] == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "db.json"
        db = {"created": format_timestamp(), "embryos": {}, "last_updated": ""}
        add_embryo_to_database(db, "embryo_001", {"embryo_number": 1})
        save_multi_embryo_database(db, path)

        loaded = load_multi_embryo_database(path)
        assert "embryo_001" in loaded["embryos"]
        assert loaded["last_updated"]  # Should be updated

    def test_add_embryo(self):
        db = {"created": "", "embryos": {}, "last_updated": ""}
        add_embryo_to_database(db, "embryo_001", {"embryo_number": 1})
        assert "embryo_001" in db["embryos"]

    def test_get_calibration(self):
        db = {"embryos": {"embryo_001": {"calibration": {"slope_um_per_deg": 95.0}}}}
        cal = get_embryo_calibration(db, "embryo_001")
        assert cal["slope_um_per_deg"] == 95.0

    def test_get_calibration_missing(self):
        db = {"embryos": {}}
        assert get_embryo_calibration(db, "embryo_001") is None

    def test_list_embryos_sorted(self):
        db = {
            "embryos": {
                "embryo_003": {"embryo_number": 3},
                "embryo_001": {"embryo_number": 1},
                "embryo_002": {"embryo_number": 2},
            }
        }
        result = list_embryos(db)
        assert result == ["embryo_001", "embryo_002", "embryo_003"]
