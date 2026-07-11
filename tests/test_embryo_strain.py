"""Tests for per-embryo strain field (D2 Task 1).

Covers:
- register_embryo with strain persists it through get_embryo
- update without strain coalesces (keeps existing value)
- absent strain on create → None
- /api/embryos/positions includes strain
"""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

# ===========================================================================
# FileStore tests
# ===========================================================================


def test_register_embryo_with_strain(file_store):
    """register_embryo with strain; get_embryo returns it."""
    file_store.create_session("s1", name="test")
    file_store.register_embryo(
        "s1",
        "embryo_1",
        position_coarse={"x": 10.0, "y": 20.0},
        strain="pan-nuclear GFP",
    )
    emb = file_store.get_embryo("s1", "embryo_1")
    assert emb is not None
    assert emb.get("strain") == "pan-nuclear GFP"


def test_register_embryo_strain_absent_is_none(file_store):
    """register_embryo without strain → strain is None in get_embryo."""
    file_store.create_session("s2", name="test")
    file_store.register_embryo("s2", "embryo_1", position_coarse={"x": 0.0, "y": 0.0})
    emb = file_store.get_embryo("s2", "embryo_1")
    assert emb is not None
    assert emb.get("strain") is None


def test_register_embryo_strain_coalesces_on_update(file_store):
    """Update without strain keeps the originally set value (coalesce)."""
    file_store.create_session("s3", name="test")
    # Create with strain
    file_store.register_embryo(
        "s3",
        "embryo_1",
        position_coarse={"x": 5.0, "y": 5.0},
        strain="H2B-mCherry",
    )
    # Update with a new position but no strain — existing strain must be kept
    file_store.register_embryo(
        "s3",
        "embryo_1",
        position_coarse={"x": 6.0, "y": 6.0},
    )
    emb = file_store.get_embryo("s3", "embryo_1")
    assert emb is not None
    assert emb.get("strain") == "H2B-mCherry"


def test_register_embryo_strain_can_be_overwritten(file_store):
    """Passing a new strain on update replaces the old value."""
    file_store.create_session("s4", name="test")
    file_store.register_embryo(
        "s4", "embryo_1", position_coarse={"x": 0.0, "y": 0.0}, strain="wild-type"
    )
    file_store.register_embryo(
        "s4", "embryo_1", position_coarse={"x": 0.0, "y": 0.0}, strain="pan-nuclear GFP"
    )
    emb = file_store.get_embryo("s4", "embryo_1")
    assert emb is not None
    assert emb.get("strain") == "pan-nuclear GFP"


# ===========================================================================
# /api/embryos/positions endpoint test
# ===========================================================================


def _build_client_with_tracker(embryos_dict):
    """Build a FastAPI TestClient whose server has a timelapse_tracker."""
    from gently.ui.web.routes.data import create_router

    app = FastAPI()
    server = MagicMock()

    tracker = MagicMock()
    tracker.embryos = embryos_dict
    server.timelapse_tracker = tracker

    # store.get_embryo_ids needed by /api/embryos (not under test here)
    server.store.get_embryo_ids.return_value = []

    app.include_router(create_router(server))
    return TestClient(app)


def test_positions_endpoint_includes_strain():
    """embryo_positions response includes the strain key from tracker."""
    client = _build_client_with_tracker(
        {
            "embryo_1": {
                "stage_x_um": 10.0,
                "stage_y_um": 20.0,
                "role": "test",
                "strain": "pan-nuclear GFP",
                "uid": "abc",
                "user_label": None,
                "confidence": 0.9,
                "cadence_phase": None,
                "is_complete": False,
            }
        }
    )
    resp = client.get("/api/embryos/positions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["embryos"]) == 1
    point = data["embryos"][0]
    assert "strain" in point
    assert point["strain"] == "pan-nuclear GFP"


def test_positions_endpoint_strain_none_when_absent():
    """embryo_positions returns strain=None when tracker dict has no strain key."""
    client = _build_client_with_tracker(
        {
            "embryo_1": {
                "stage_x_um": 1.0,
                "stage_y_um": 2.0,
                "role": "test",
                # no "strain" key
            }
        }
    )
    resp = client.get("/api/embryos/positions")
    assert resp.status_code == 200
    point = resp.json()["embryos"][0]
    assert "strain" in point
    assert point["strain"] is None


# ===========================================================================
# TimelapseStateTracker – Fix 1 (strain surfaces through tracker)
# ===========================================================================


def test_tracker_embryo_detected_sets_strain():
    """EMBRYO_DETECTED with strain populates the tracker embryo dict."""
    from gently.ui.web.timelapse_tracker import TimelapseStateTracker

    tracker = TimelapseStateTracker()
    tracker.handle_event(
        "EMBRYO_DETECTED",
        {
            "embryo_id": "e1",
            "x": 100.0,
            "y": 200.0,
            "role": "test",
            "strain": "pan-nuclear GFP",
            "uid": "abc",
        },
    )
    assert tracker.embryos["e1"]["strain"] == "pan-nuclear GFP"


def test_tracker_embryo_detected_strain_none_propagates():
    """EMBRYO_DETECTED with explicit strain=None still writes the key."""
    from gently.ui.web.timelapse_tracker import TimelapseStateTracker

    tracker = TimelapseStateTracker()
    tracker.handle_event(
        "EMBRYO_DETECTED",
        {
            "embryo_id": "e2",
            "x": 0.0,
            "y": 0.0,
            "role": "test",
            "strain": None,
        },
    )
    assert "strain" in tracker.embryos["e2"]
    assert tracker.embryos["e2"]["strain"] is None


def test_tracker_seed_from_experiment_includes_strain():
    """seed_from_experiment copies strain from EmbryoState into tracker."""
    from types import SimpleNamespace

    from gently.ui.web.timelapse_tracker import TimelapseStateTracker

    emb = SimpleNamespace(
        stage_position={"x": 50.0, "y": 60.0},
        uid="u1",
        role="test",
        strain="H2B-mCherry",
        user_label=None,
        detection_confidence=0.95,
    )
    experiment = SimpleNamespace(embryos={"e3": emb})

    tracker = TimelapseStateTracker()
    seeded = tracker.seed_from_experiment(experiment)

    assert seeded == 1
    assert tracker.embryos["e3"]["strain"] == "H2B-mCherry"


def test_tracker_seed_from_experiment_strain_absent_is_none():
    """seed_from_experiment with no strain attribute → tracker strain is None."""
    from types import SimpleNamespace

    from gently.ui.web.timelapse_tracker import TimelapseStateTracker

    emb = SimpleNamespace(
        stage_position={"x": 10.0, "y": 20.0},
        uid="u2",
        role="control",
        # no .strain attribute
        user_label=None,
        detection_confidence=None,
    )
    experiment = SimpleNamespace(embryos={"e4": emb})

    tracker = TimelapseStateTracker()
    tracker.seed_from_experiment(experiment)

    assert "strain" in tracker.embryos["e4"]
    assert tracker.embryos["e4"]["strain"] is None
