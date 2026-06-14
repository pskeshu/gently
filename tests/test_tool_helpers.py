"""Tests for gently.harness.tools.helpers"""

import types
from unittest.mock import MagicMock

from gently.harness.tools.helpers import (
    build_snapshot_metadata,
    format_duration,
    get_embryo_or_error,
    get_timestamp_string,
    require_agent,
    require_databroker,
    require_developmental_tracker,
    require_interaction_logger,
    require_microscope,
    require_timelapse_orchestrator,
)

# ── Context extractors ──────────────────────────────────────────────


class TestRequireAgent:
    def test_returns_agent_when_present(self):
        agent = MagicMock()
        result, err = require_agent({"agent": agent})
        assert result is agent
        assert err is None

    def test_returns_error_when_missing(self):
        result, err = require_agent({})
        assert result is None
        assert "No agent" in err

    def test_returns_error_when_none(self):
        result, err = require_agent({"agent": None})
        assert result is None
        assert err is not None


class TestRequireMicroscope:
    def test_returns_client_when_present(self):
        client = MagicMock()
        result, err = require_microscope({"client": client})
        assert result is client
        assert err is None

    def test_returns_error_when_missing(self):
        result, err = require_microscope({})
        assert result is None
        assert "connect_microscope" in err


class TestRequireInteractionLogger:
    def test_returns_logger_when_present(self):
        agent = MagicMock()
        agent.interaction_logger = MagicMock()
        result, err = require_interaction_logger(agent)
        assert result is agent.interaction_logger
        assert err is None

    def test_returns_error_when_none(self):
        agent = MagicMock()
        agent.interaction_logger = None
        result, err = require_interaction_logger(agent)
        assert result is None
        assert err is not None

    def test_returns_error_when_attr_missing(self):
        agent = types.SimpleNamespace()
        result, err = require_interaction_logger(agent)
        assert result is None
        assert err is not None


class TestRequireDevelopmentalTracker:
    def test_returns_tracker_when_present(self):
        agent = MagicMock()
        agent.developmental_tracker = MagicMock()
        result, err = require_developmental_tracker(agent)
        assert result is agent.developmental_tracker
        assert err is None

    def test_returns_error_when_none(self):
        agent = MagicMock()
        agent.developmental_tracker = None
        _, err = require_developmental_tracker(agent)
        assert "classify_embryo_stage" in err


class TestRequireTimelapseOrchestrator:
    def test_returns_orchestrator_when_present(self):
        agent = MagicMock()
        agent.timelapse_orchestrator = MagicMock()
        result, err = require_timelapse_orchestrator(agent)
        assert result is agent.timelapse_orchestrator
        assert err is None

    def test_returns_error_when_none(self):
        agent = MagicMock()
        agent.timelapse_orchestrator = None
        _, err = require_timelapse_orchestrator(agent)
        assert err is not None


class TestRequireDatabroker:
    def test_returns_databroker_when_present(self):
        agent = MagicMock()
        agent.databroker = MagicMock()
        result, err = require_databroker(agent)
        assert result is agent.databroker
        assert err is None

    def test_returns_error_when_none(self):
        agent = MagicMock()
        agent.databroker = None
        _, err = require_databroker(agent)
        assert "databroker" in err.lower()


class TestGetEmbryoOrError:
    def test_returns_embryo_when_found(self):
        embryo = MagicMock()
        agent = MagicMock()
        agent.experiment.get_embryo_by_any_name.return_value = embryo
        result, err = get_embryo_or_error(agent, "E1")
        assert result is embryo
        assert err is None

    def test_returns_error_when_not_found(self):
        agent = MagicMock()
        agent.experiment.get_embryo_by_any_name.return_value = None
        result, err = get_embryo_or_error(agent, "missing")
        assert result is None
        assert "missing" in err


# ── Utility functions ────────────────────────────────────────────────


class TestGetTimestampString:
    def test_format(self):
        ts = get_timestamp_string()
        assert len(ts) == 15  # YYYYMMDD_HHMMSS
        assert ts[8] == "_"


class TestFormatDuration:
    def test_seconds(self):
        assert format_duration(30) == "30s"

    def test_minutes(self):
        assert format_duration(150) == "2.5m"

    def test_hours(self):
        assert format_duration(7200) == "2.0h"

    def test_boundary_60(self):
        assert format_duration(60) == "1.0m"

    def test_boundary_3600(self):
        assert format_duration(3600) == "1.0h"

    def test_zero(self):
        assert format_duration(0) == "0s"


# ── build_snapshot_metadata ──────────────────────────────────────────


class TestBuildSnapshotMetadata:
    def test_basic_fields(self):
        meta = build_snapshot_metadata(
            stage_position=(1500.0, 200.0),
            image_shape=(1024, 1280),
        )
        assert meta["stage_x"] == 1500.0
        assert meta["stage_y"] == 200.0
        assert meta["image_width"] == 1280
        assert meta["image_height"] == 1024

    def test_coordinate_transform(self):
        meta = build_snapshot_metadata(
            stage_position=(0, 0),
            image_shape=(1024, 1280),
            pixel_size_um=6.5,
            objective_mag=10.0,
        )
        ct = meta["coordinate_transform"]
        assert ct["pixel_size_um"] == 6.5
        assert ct["objective_mag"] == 10.0
        assert ct["um_per_pixel"] == 0.65
        assert ct["image_center_x"] == 640.0
        assert ct["image_center_y"] == 512.0

    def test_default_safety_perimeter(self):
        meta = build_snapshot_metadata(
            stage_position=(0, 0),
            image_shape=(100, 100),
        )
        sp = meta["safety_perimeter"]
        assert sp["x"] == (2000.0, 4000.0)
        assert sp["y"] == (-1000.0, 1000.0)

    def test_custom_safety_limits(self):
        limits = {"x": (0, 5000), "y": (-2000, 2000)}
        meta = build_snapshot_metadata(
            stage_position=(0, 0),
            image_shape=(100, 100),
            safety_limits=limits,
        )
        assert meta["safety_perimeter"] is limits

    def test_no_embryos_when_experiment_none(self):
        meta = build_snapshot_metadata(
            stage_position=(0, 0),
            image_shape=(100, 100),
            experiment=None,
        )
        assert "embryos" not in meta

    def test_no_embryos_when_experiment_empty(self):
        exp = MagicMock()
        exp.embryos = {}
        meta = build_snapshot_metadata(
            stage_position=(0, 0),
            image_shape=(100, 100),
            experiment=exp,
        )
        assert "embryos" not in meta

    def test_embryos_included(self):
        emb1 = MagicMock()
        emb1.stage_position = {"x": 1000.0, "y": 500.0}
        emb1.nickname = "lefty"

        emb2 = MagicMock()
        emb2.stage_position = {"x": 1200.0, "y": -100.0}
        emb2.nickname = None

        exp = MagicMock()
        exp.embryos = {"E1": emb1, "E2": emb2}

        meta = build_snapshot_metadata(
            stage_position=(1100.0, 200.0),
            image_shape=(1024, 1280),
            experiment=exp,
        )
        assert len(meta["embryos"]) == 2
        e1 = next(e for e in meta["embryos"] if e["embryo_id"] == "E1")
        assert e1["stage_x"] == 1000.0
        assert e1["stage_y"] == 500.0
        assert e1["nickname"] == "lefty"
        e2 = next(e for e in meta["embryos"] if e["embryo_id"] == "E2")
        assert e2["nickname"] is None

    def test_embryo_with_no_position(self):
        emb = MagicMock()
        emb.stage_position = None
        emb.nickname = None
        exp = MagicMock()
        exp.embryos = {"E1": emb}

        meta = build_snapshot_metadata(
            stage_position=(0, 0),
            image_shape=(100, 100),
            experiment=exp,
        )
        e = meta["embryos"][0]
        # Should not crash — returns None for x/y
        assert e["stage_x"] is None
        assert e["stage_y"] is None

    def test_three_channel_image(self):
        meta = build_snapshot_metadata(
            stage_position=(0, 0),
            image_shape=(1024, 1280, 3),
        )
        assert meta["image_height"] == 1024
        assert meta["image_width"] == 1280

    def test_custom_optics(self):
        meta = build_snapshot_metadata(
            stage_position=(0, 0),
            image_shape=(512, 512),
            pixel_size_um=3.45,
            objective_mag=20.0,
        )
        ct = meta["coordinate_transform"]
        assert ct["um_per_pixel"] == 3.45 / 20.0

    def test_metadata_is_json_serializable(self):
        """Metadata must round-trip through JSON for DB storage."""
        import json

        emb = MagicMock()
        emb.stage_position = {"x": 1000.0, "y": 500.0}
        emb.nickname = "test"
        exp = MagicMock()
        exp.embryos = {"E1": emb}

        meta = build_snapshot_metadata(
            stage_position=(1500.0, 200.0),
            image_shape=(1024, 1280),
            experiment=exp,
        )
        roundtripped = json.loads(json.dumps(meta))
        assert roundtripped["stage_x"] == 1500.0
        assert len(roundtripped["embryos"]) == 1
