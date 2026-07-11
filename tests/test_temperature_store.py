"""Tests for FileStore temperature log methods."""


def _new_session(file_store):
    """Create a test session and return its ID."""
    # Use UUID-based session ID for uniqueness
    import uuid

    session_id = str(uuid.uuid4())
    return file_store.create_session(session_id, name="temp-test")


def test_append_and_read_roundtrip(file_store):
    """Test appending and reading temperature samples."""
    sid = _new_session(file_store)
    file_store.append_temperature_sample(
        sid,
        {"t": "2026-06-27T10:00:00+00:00", "water_c": 28.0, "setpoint_c": 32.0, "state": "heating"},
    )
    file_store.append_temperature_sample(
        sid,
        {"t": "2026-06-27T10:00:01+00:00", "water_c": 28.3, "setpoint_c": 32.0, "state": "heating"},
    )
    rows = file_store.read_temperature_log(sid)
    assert [r["water_c"] for r in rows] == [28.0, 28.3]


def test_read_since_filters(file_store):
    """Test that read_temperature_log filters by since parameter."""
    sid = _new_session(file_store)
    for i, t in enumerate(
        ["2026-06-27T10:00:00+00:00", "2026-06-27T10:00:01+00:00", "2026-06-27T10:00:02+00:00"]
    ):
        file_store.append_temperature_sample(
            sid, {"t": t, "water_c": 28.0 + i, "setpoint_c": 32.0, "state": "heating"}
        )
    rows = file_store.read_temperature_log(sid, since="2026-06-27T10:00:01+00:00")
    assert [r["water_c"] for r in rows] == [29.0, 30.0]


def test_read_unknown_session_is_empty(file_store):
    """Test that reading from a non-existent session returns empty list."""
    assert file_store.read_temperature_log("does-not-exist") == []


def test_truncated_trailing_line_is_skipped(file_store, tmp_path):
    """A truncated trailing line (e.g. after a crash mid-append) is skipped gracefully."""

    sid = _new_session(file_store)

    # Append two valid samples via the normal API.
    file_store.append_temperature_sample(
        sid,
        {"t": "2026-06-27T10:00:01+00:00", "water_c": 28.1, "setpoint_c": 32.0, "state": "heating"},
    )
    file_store.append_temperature_sample(
        sid,
        {"t": "2026-06-27T10:00:02+00:00", "water_c": 28.2, "setpoint_c": 32.0, "state": "heating"},
    )

    # Append a raw truncated line directly to the JSONL file.
    # Locate the temperature.jsonl via the store's internal path.
    sd = file_store._session_dir(sid)
    log_path = sd / "temperature.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write('{"t": "2026-06-27T10:00:03+00:00", "water_c":\n')

    # read_temperature_log must return only the two valid rows without raising.
    rows = file_store.read_temperature_log(sid)
    assert len(rows) == 2
    assert [r["water_c"] for r in rows] == [28.1, 28.2]
