"""Tests for Task 6 & Task 7: acquisition temperature stamp.

Concerns:
1. temperature_stamp(None) returns None (pure helper, no I/O).
2. Volume metadata round-trip: a stamp injected via put_volume(metadata={"temperature":
   stamp}) lands correctly in the sidecar YAML and is readable via get_volume_meta.
3. Burst stamp: _persist_burst_to_disk injects temperature into both burst.yaml and
   each frame's .meta.yaml when a temperature_provider is supplied.

Confirmed from file_store.py:
- create_session(session_id, name=None, ...) — session_id is the first positional arg.
- register_embryo(session_id, embryo_id, ...) — embryo_id is the key; returns None.
- put_volume(session_id, embryo_id, timepoint, volume, metadata=None) — nests `metadata`
  under the "metadata" key in the sidecar YAML.
- get_volume_meta(session_id, embryo_id, timepoint) — added by Task 6; reads sidecar YAML.

Note: tifffile is not installed in the test environment; it is mocked so put_volume /
_persist_burst_to_disk write only the YAML files (the contracts under test).
"""

import sys
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from gently.app.temperature_sampler import temperature_stamp


def test_stamp_none_when_no_reading():
    assert temperature_stamp(None) is None


def test_volume_metadata_carries_temperature(file_store):
    sid = file_store.create_session(str(uuid.uuid4()), name="s")
    emb = "embryo_1"
    file_store.register_embryo(sid, emb, position_x=0.0, position_y=0.0)
    stamp = temperature_stamp(
        {
            "t": "2026-06-27T10:00:00+00:00",
            "water_c": 28.4,
            "setpoint_c": 32.0,
            "state": "heating",
        }
    )
    vol = np.zeros((2, 4, 4), dtype="uint16")

    # tifffile is not installed in this environment; mock it so put_volume can
    # proceed to writing the sidecar YAML (the part under test).
    tifffile_mock = MagicMock()
    with patch.dict(sys.modules, {"tifffile": tifffile_mock}):
        with patch.object(file_store, "_generate_projection", return_value=None):
            file_store.put_volume(
                sid, emb, timepoint=0, volume=vol, metadata={"temperature": stamp}
            )

    meta = file_store.get_volume_meta(sid, emb, 0)
    assert meta["metadata"]["temperature"]["water_c"] == 28.4


def test_burst_stamp_writes_temperature(file_store):
    """_persist_burst_to_disk stamps temperature into burst.yaml and frame .meta.yaml."""
    from gently.app.orchestration.exclusive import _persist_burst_to_disk

    sid = file_store.create_session(str(uuid.uuid4()), name="burst-stamp-test")
    embryo_id = "embryo_burst"
    file_store.register_embryo(sid, embryo_id, position_x=0.0, position_y=0.0)

    # Minimal orchestrator stand-in: only _store and _session_id are needed.
    orch = MagicMock()
    orch._store = file_store
    orch._session_id = sid

    # Minimal embryo stand-in.
    embryo = MagicMock()
    embryo.stage_position = {"x": 0.0, "y": 0.0}
    embryo.num_slices = 2
    embryo.exposure_ms = 50.0

    frames_data = [{"volume": np.zeros((2, 4, 4), dtype="uint16"), "acquired_at_epoch": None}]

    tifffile_mock = MagicMock()
    with patch.dict(sys.modules, {"tifffile": tifffile_mock}):
        burst_dir = _persist_burst_to_disk(
            orchestrator=orch,
            embryo=embryo,
            embryo_id=embryo_id,
            request_id="req-burst-001",
            mode="1hz",
            frames_requested=1,
            frames_data=frames_data,
            loop_start=datetime(2026, 6, 27, 10, 0, 0, tzinfo=timezone.utc),
            duration_s=1.0,
            sustained_hz=1.0,
            galvo_amplitude=100.0,
            galvo_center=0.0,
            piezo_amplitude=50.0,
            piezo_center=0.0,
            laser_power_488_pct=10.0,
            temperature_provider=lambda: {
                "t": "2026-06-27T10:00:00+00:00",
                "water_c": 28.4,
                "setpoint_c": 32.0,
                "state": "heating",
            },
        )

    assert burst_dir is not None, "_persist_burst_to_disk returned None — check store/session setup"

    # burst.yaml must have a top-level temperature with water_c == 28.4.
    manifest = yaml.safe_load((burst_dir / "burst.yaml").read_text())
    assert manifest["temperature"]["water_c"] == 28.4

    # Frame meta.yaml must have metadata.temperature.water_c == 28.4.
    frame_meta = yaml.safe_load((burst_dir / "t0001.meta.yaml").read_text())
    assert frame_meta["metadata"]["temperature"]["water_c"] == 28.4
