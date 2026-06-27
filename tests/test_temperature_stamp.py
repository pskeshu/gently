"""Tests for Task 6: acquisition temperature stamp.

Two concerns:
1. temperature_stamp(None) returns None (pure helper, no I/O).
2. Volume metadata round-trip: a stamp injected via put_volume(metadata={"temperature":
   stamp}) lands correctly in the sidecar YAML and is readable via get_volume_meta.

Confirmed from file_store.py:
- create_session(session_id, name=None, ...) — session_id is the first positional arg.
- register_embryo(session_id, embryo_id, ...) — embryo_id is the key; returns None.
- put_volume(session_id, embryo_id, timepoint, volume, metadata=None) — nests `metadata`
  under the "metadata" key in the sidecar YAML.
- get_volume_meta(session_id, embryo_id, timepoint) — added by Task 6; reads sidecar YAML.

Note: tifffile is not installed in the test environment; it is mocked so put_volume
writes only the sidecar YAML (the contract under test) without the TIFF.
"""

import sys
import uuid
from unittest.mock import MagicMock, patch

import numpy as np

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
            file_store.put_volume(sid, emb, timepoint=0, volume=vol, metadata={"temperature": stamp})

    meta = file_store.get_volume_meta(sid, emb, 0)
    assert meta["metadata"]["temperature"]["water_c"] == 28.4
