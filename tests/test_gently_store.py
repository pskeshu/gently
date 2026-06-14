"""
Tests for GentlyStore — unified data storage.

Tests cover:
- Session CRUD
- Embryo registration
- put_volume() round-trip (write TIFF + projection + DB rows)
- register_volume() zero-copy path (rename + projection + DB)
- Projection generation and retrieval
- Perception run lifecycle (create → store predictions → complete)
- Ground truth CRUD
- EmbryoDataset integration with GentlyStore schema
- Device-side serialize_value() file-ref protocol
- Client-side _resolve_file_refs()
"""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from gently.core.store import GentlyStore


@pytest.fixture
def store_dir(tmp_path):
    """Create a temporary directory for the store."""
    return tmp_path / "gently_test"


@pytest.fixture
def store(store_dir):
    """Create a fresh GentlyStore instance."""
    s = GentlyStore(store_dir)
    yield s
    s.close()


# =========================================================================
# Session tests
# =========================================================================


class TestSessions:
    def test_create_session(self, store):
        sid = store.create_session("sess1", name="Test Session")
        assert sid == "sess1"

    def test_get_session(self, store):
        store.create_session("sess1", name="Test", description="desc")
        s = store.get_session("sess1")
        assert s is not None
        assert s["session_id"] == "sess1"
        assert s["name"] == "Test"
        assert s["description"] == "desc"

    def test_get_nonexistent_session(self, store):
        assert store.get_session("nope") is None

    def test_list_sessions(self, store):
        store.create_session("a")
        store.create_session("b", name="Second")
        sessions = store.list_sessions()
        assert len(sessions) == 2
        ids = {s["session_id"] for s in sessions}
        assert ids == {"a", "b"}

    def test_touch_session(self, store):
        store.create_session("sess1")
        s1 = store.get_session("sess1")
        store.touch_session("sess1")
        s2 = store.get_session("sess1")
        assert s2["last_active"] >= s1["last_active"]

    def test_session_snapshot_roundtrip(self, store):
        store.create_session("sess1")
        snapshot = {"conversation": ["hello", "world"], "state": {"foo": 1}}
        store.save_session_snapshot("sess1", snapshot)
        loaded = store.load_session_snapshot("sess1")
        assert loaded == snapshot

    def test_load_missing_snapshot(self, store):
        assert store.load_session_snapshot("nope") is None

    def test_create_session_with_metadata(self, store):
        store.create_session("sess1", metadata={"key": "value"})
        s = store.get_session("sess1")
        assert s["metadata"] == {"key": "value"}

    def test_create_duplicate_session_is_noop(self, store):
        store.create_session("sess1", name="First")
        store.create_session("sess1", name="Second")  # INSERT OR IGNORE
        s = store.get_session("sess1")
        assert s["name"] == "First"  # not overwritten


# =========================================================================
# Embryo tests
# =========================================================================


class TestEmbryos:
    def test_register_embryo(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "embryo_1", position_x=100.0, position_y=200.0)
        e = store.get_embryo("s1", "embryo_1")
        assert e is not None
        assert e["embryo_id"] == "embryo_1"
        assert e["position_x"] == 100.0
        assert e["position_y"] == 200.0

    def test_register_embryo_update(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "embryo_1", position_x=100.0)
        store.register_embryo("s1", "embryo_1", position_y=200.0)
        e = store.get_embryo("s1", "embryo_1")
        # COALESCE keeps old non-null values, updates new
        assert e["position_x"] == 100.0
        assert e["position_y"] == 200.0

    def test_list_embryos(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "embryo_1")
        store.register_embryo("s1", "embryo_2")
        embryos = store.list_embryos("s1")
        assert len(embryos) == 2

    def test_get_nonexistent_embryo(self, store):
        store.create_session("s1")
        assert store.get_embryo("s1", "nope") is None

    def test_embryo_with_calibration(self, store):
        store.create_session("s1")
        cal = {"pixel_size_um": 0.325, "z_step_um": 1.0}
        store.register_embryo("s1", "e1", calibration=cal)
        e = store.get_embryo("s1", "e1")
        assert e["calibration"] == cal


# =========================================================================
# Volume tests
# =========================================================================


def _make_volume(shape=(2, 20, 64, 64), dtype=np.uint16):
    """Create a test volume with some structure."""
    rng = np.random.RandomState(42)
    return rng.randint(100, 5000, size=shape, dtype=dtype)


class TestVolumes:
    def test_put_volume_roundtrip(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")
        volume = _make_volume()

        path = store.put_volume("s1", "e1", 0, volume)
        assert path.exists()
        assert path.suffix == ".tif"
        assert "e1_t0000" in path.name

        # Read back
        loaded = store.get_volume("s1", "e1", 0)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, volume)

    def test_put_volume_creates_projection(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")
        volume = _make_volume()
        store.put_volume("s1", "e1", 0, volume)

        proj_path = store.get_projection_path("s1", "e1", 0)
        assert proj_path is not None
        assert proj_path.exists()
        assert proj_path.suffix == ".jpg"

    def test_put_volume_db_row(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")
        volume = _make_volume()
        store.put_volume("s1", "e1", 0, volume)

        vols = store.list_volumes("s1", "e1")
        assert len(vols) == 1
        v = vols[0]
        assert v["timepoint"] == 0
        assert v["embryo_id"] == "e1"
        assert v["shape"] == list(volume.shape)
        assert v["dtype"] == "uint16"

    def test_register_volume_zero_copy(self, store, store_dir):
        """register_volume renames an incoming TIFF to canonical location."""
        store.create_session("s1")
        store.register_embryo("s1", "e1")

        # Write a TIFF to the incoming directory
        volume = _make_volume()
        incoming = store.incoming_dir / "test_uuid.tif"
        tifffile.imwrite(str(incoming), volume)
        assert incoming.exists()

        # Register it
        canonical = store.register_volume("s1", "e1", 5, incoming)
        assert canonical.exists()
        assert "e1_t0005" in canonical.name
        assert not incoming.exists()  # moved, not copied

        # Verify DB row
        vol_path = store.get_volume_path("s1", "e1", 5)
        assert vol_path is not None
        assert vol_path.exists()

        # Verify projection was generated
        proj = store.get_projection_path("s1", "e1", 5)
        assert proj is not None
        assert proj.exists()

    def test_register_volume_overwrites(self, store):
        """Re-registering same (session, embryo, timepoint) overwrites."""
        store.create_session("s1")
        store.register_embryo("s1", "e1")

        v1 = _make_volume(shape=(2, 10, 32, 32))
        store.put_volume("s1", "e1", 0, v1)

        v2 = _make_volume(shape=(2, 20, 64, 64))
        incoming = store.incoming_dir / "v2.tif"
        tifffile.imwrite(str(incoming), v2)
        store.register_volume("s1", "e1", 0, incoming)

        loaded = store.get_volume("s1", "e1", 0)
        np.testing.assert_array_equal(loaded, v2)

    def test_get_volume_nonexistent(self, store):
        assert store.get_volume("nope", "nope", 0) is None

    def test_get_volume_path_nonexistent(self, store):
        assert store.get_volume_path("nope", "nope", 0) is None

    def test_list_volumes_filter_by_embryo(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")
        store.register_embryo("s1", "e2")

        store.put_volume("s1", "e1", 0, _make_volume())
        store.put_volume("s1", "e1", 1, _make_volume())
        store.put_volume("s1", "e2", 0, _make_volume())

        all_vols = store.list_volumes("s1")
        assert len(all_vols) == 3

        e1_vols = store.list_volumes("s1", "e1")
        assert len(e1_vols) == 2

    def test_multiple_timepoints(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")

        for tp in range(5):
            store.put_volume("s1", "e1", tp, _make_volume())

        vols = store.list_volumes("s1", "e1")
        assert len(vols) == 5
        timepoints = [v["timepoint"] for v in vols]
        assert timepoints == [0, 1, 2, 3, 4]


# =========================================================================
# Projection tests
# =========================================================================


class TestProjections:
    def test_projection_b64(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")
        store.put_volume("s1", "e1", 0, _make_volume())

        b64 = store.get_projection_b64("s1", "e1", 0)
        assert b64 is not None
        assert len(b64) > 100  # non-trivial base64 string

    def test_list_projections(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")
        for tp in range(3):
            store.put_volume("s1", "e1", tp, _make_volume())

        projs = store.list_projections("s1", "e1")
        assert len(projs) == 3

    def test_projection_nonexistent(self, store):
        assert store.get_projection_path("nope", "nope", 0) is None
        assert store.get_projection_b64("nope", "nope", 0) is None


# =========================================================================
# Perception run lifecycle tests
# =========================================================================


class TestPerception:
    def test_create_and_complete_run(self, store):
        store.create_session("s1")
        run_id = store.create_perception_run(
            "s1",
            "test_run",
            "vlm_stage_classification",
            model_name="claude-opus-4-5-20251101",
        )
        assert isinstance(run_id, int)
        assert run_id > 0

        store.complete_perception_run(run_id, status="completed")
        # No exception = success

    def test_store_prediction(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")
        store.put_volume("s1", "e1", 0, _make_volume())

        run_id = store.create_perception_run("s1", "run1", "vlm")
        pred_id = store.store_prediction(
            run_id=run_id,
            session_id="s1",
            embryo_id="e1",
            timepoint=0,
            predicted_stage="early",
            confidence=0.85,
            reasoning="Oval shape, smooth texture",
        )
        assert isinstance(pred_id, int)
        assert pred_id > 0

    def test_store_prediction_with_trace(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")
        store.put_volume("s1", "e1", 0, _make_volume())

        run_id = store.create_perception_run("s1", "run1", "vlm")
        trace_data = {
            "perception_result": {"stage": "early"},
            "steps": ["step1", "step2"],
        }
        store.store_prediction(
            run_id=run_id,
            session_id="s1",
            embryo_id="e1",
            timepoint=0,
            predicted_stage="early",
            confidence=0.9,
            trace_data=trace_data,
        )

        # Verify trace file was written
        trace_path = store.root / "traces" / "s1" / "e1_t0000.json"
        assert trace_path.exists()
        with open(trace_path) as f:
            loaded = json.load(f)
        assert loaded == trace_data

    def test_store_prediction_with_observed_features(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")
        store.put_volume("s1", "e1", 0, _make_volume())

        run_id = store.create_perception_run("s1", "run1", "vlm")
        features = {"shape": "oval", "curvature": "slight", "texture": "smooth"}
        store.store_prediction(
            run_id=run_id,
            session_id="s1",
            embryo_id="e1",
            timepoint=0,
            predicted_stage="early",
            observed_features=features,
        )

        preds = store.get_predictions("s1", "e1")
        assert len(preds) == 1
        assert preds[0]["observed_features"] == features

    def test_get_predictions_filter_by_run(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")
        store.put_volume("s1", "e1", 0, _make_volume())
        store.put_volume("s1", "e1", 1, _make_volume())

        run1 = store.create_perception_run("s1", "run1", "vlm")
        run2 = store.create_perception_run("s1", "run2", "vlm")

        store.store_prediction(run1, "s1", "e1", 0, "early", confidence=0.8)
        store.store_prediction(run2, "s1", "e1", 0, "bean", confidence=0.6)
        store.store_prediction(run2, "s1", "e1", 1, "comma", confidence=0.7)

        all_preds = store.get_predictions("s1", "e1")
        assert len(all_preds) == 3

        run2_preds = store.get_predictions("s1", "e1", run_id=run2)
        assert len(run2_preds) == 2

    def test_perception_run_with_config(self, store):
        store.create_session("s1")
        config = {"interval": 120, "stop_condition": "hatched"}
        run_id = store.create_perception_run(
            "s1",
            "run1",
            "vlm",
            config=config,
        )
        assert run_id > 0

    def test_complete_run_failed(self, store):
        store.create_session("s1")
        run_id = store.create_perception_run("s1", "run1", "vlm")
        store.complete_perception_run(run_id, status="failed", error_message="OOM")
        # No exception = success


# =========================================================================
# Ground truth tests
# =========================================================================


class TestGroundTruth:
    def test_set_and_get_ground_truth(self, store):
        store.create_session("s1")
        store.set_ground_truth("s1", "e1", "early", 0, end_timepoint=42)
        store.set_ground_truth("s1", "e1", "bean", 43, end_timepoint=55)

        gt = store.get_ground_truth("s1", "e1")
        assert len(gt) == 2
        assert gt[0]["stage"] == "early"
        assert gt[0]["start_timepoint"] == 0
        assert gt[0]["end_timepoint"] == 42
        assert gt[1]["stage"] == "bean"

    def test_ground_truth_upsert(self, store):
        store.create_session("s1")
        store.set_ground_truth("s1", "e1", "early", 0)
        store.set_ground_truth("s1", "e1", "early", 5)  # update start

        gt = store.get_ground_truth("s1", "e1")
        assert len(gt) == 1
        assert gt[0]["start_timepoint"] == 5


# =========================================================================
# Stats and utility tests
# =========================================================================


class TestUtility:
    def test_stats(self, store):
        store.create_session("s1")
        store.register_embryo("s1", "e1")
        store.put_volume("s1", "e1", 0, _make_volume())

        stats = store.stats()
        assert stats["sessions"] == 1
        assert stats["embryos"] == 1
        assert stats["volumes"] == 1
        assert stats["projections"] == 1
        assert stats["disk_usage_mb"] > 0

    def test_db_path_property(self, store, store_dir):
        assert store.db_path == store_dir / "gently.db"

    def test_incoming_dir_property(self, store, store_dir):
        assert store.incoming_dir == store_dir / "incoming"
        assert store.incoming_dir.exists()

    def test_directory_structure(self, store_dir, store):
        for subdir in ("incoming", "volumes", "projections", "traces", "sessions"):
            assert (store_dir / subdir).is_dir()
        assert (store_dir / "gently.db").is_file()

    def test_context_manager(self, store_dir):
        with GentlyStore(store_dir) as s:
            s.create_session("ctx_test")
        # After context exit, connection should be closed
        assert s._conn is None

    def test_repr(self, store, store_dir):
        assert str(store_dir) in repr(store)


# =========================================================================
# EmbryoDataset integration with GentlyStore
# =========================================================================


class TestEmbryoDatasetIntegration:
    def test_from_store(self, store):
        from gently.dataset.embryo_dataset import EmbryoDataset

        store.create_session("s1")
        store.register_embryo("s1", "e1")
        for tp in range(5):
            store.put_volume("s1", "e1", tp, _make_volume())

        dataset = EmbryoDataset.from_store(store)
        assert dataset.is_gently_schema is True

    def test_iter_embryos_gently_schema(self, store):
        from gently.dataset.embryo_dataset import EmbryoDataset

        store.create_session("s1")
        store.register_embryo("s1", "e1")
        store.register_embryo("s1", "e2")
        for tp in range(3):
            store.put_volume("s1", "e1", tp, _make_volume())
        for tp in range(2):
            store.put_volume("s1", "e2", tp, _make_volume())

        dataset = EmbryoDataset.from_store(store)
        embryos = list(dataset.iter_embryos(session_id="s1"))
        assert len(embryos) == 2

        # Check e1
        e1 = next(e for e in embryos if e.embryo_id == "e1")
        assert e1.num_volumes == 3
        assert e1.timepoint_range == (0, 2)

    def test_iter_images_gently_schema(self, store):
        from gently.dataset.embryo_dataset import EmbryoDataset

        store.create_session("s1")
        store.register_embryo("s1", "e1")
        for tp in range(3):
            store.put_volume("s1", "e1", tp, _make_volume())

        dataset = EmbryoDataset.from_store(store)
        images = list(dataset.iter_images("e1", session_id="s1"))
        assert len(images) == 3

        img0 = images[0]
        assert img0.embryo_id == "e1"
        assert img0.timepoint == 0
        assert img0.session_id == "s1"
        assert img0.volume_path is not None

    def test_get_image(self, store):
        from gently.dataset.embryo_dataset import EmbryoDataset

        store.create_session("s1")
        store.register_embryo("s1", "e1")
        store.put_volume("s1", "e1", 5, _make_volume())

        dataset = EmbryoDataset.from_store(store)
        img = dataset.get_image("e1", 5, session_id="s1")
        assert img is not None
        assert img.timepoint == 5

    def test_get_image_by_uid_returns_none_for_gently(self, store):
        from gently.dataset.embryo_dataset import EmbryoDataset

        dataset = EmbryoDataset.from_store(store)
        assert dataset.get_image_by_uid("some_uid") is None

    def test_store_prediction_gently_schema(self, store):
        from gently.dataset.embryo_dataset import EmbryoDataset

        store.create_session("s1")
        store.register_embryo("s1", "e1")
        store.put_volume("s1", "e1", 0, _make_volume())

        dataset = EmbryoDataset.from_store(store)
        run_id = dataset.create_perception_run(
            name="test",
            perception_method="vlm",
            session_id="s1",
        )
        pred_id = dataset.store_prediction(
            run_id=run_id,
            embryo_id="e1",
            timepoint=0,
            predicted_stage="early",
            confidence=0.9,
            session_id="s1",
        )
        assert pred_id > 0

    def test_complete_perception_run_gently_schema(self, store):
        from gently.dataset.embryo_dataset import EmbryoDataset

        store.create_session("s1")
        dataset = EmbryoDataset.from_store(store)
        run_id = dataset.create_perception_run(
            name="test",
            perception_method="vlm",
            session_id="s1",
        )
        dataset.complete_perception_run(run_id, status="completed")
        # No exception = success

    def test_compute_run_metrics_gently_schema(self, store):
        from gently.dataset.embryo_dataset import EmbryoDataset

        store.create_session("s1")
        store.register_embryo("s1", "e1")
        store.put_volume("s1", "e1", 0, _make_volume())
        store.put_volume("s1", "e1", 1, _make_volume())

        # Set ground truth
        store.set_ground_truth("s1", "e1", "early", 0, end_timepoint=2)  # exclusive

        dataset = EmbryoDataset.from_store(store)
        run_id = dataset.create_perception_run(
            name="test",
            perception_method="vlm",
            session_id="s1",
        )
        dataset.store_prediction(
            run_id=run_id,
            embryo_id="e1",
            timepoint=0,
            predicted_stage="early",
            confidence=0.9,
            session_id="s1",
        )
        dataset.store_prediction(
            run_id=run_id,
            embryo_id="e1",
            timepoint=1,
            predicted_stage="bean",
            confidence=0.6,
            session_id="s1",
        )

        metrics = dataset.compute_run_metrics(run_id)
        assert metrics["total"] == 2
        assert metrics["correct"] == 1  # "early" is correct, "bean" is wrong
        assert 0.49 < metrics["accuracy"] < 0.51


# =========================================================================
# Device-side serialize_value file-ref protocol tests
# =========================================================================


class TestSerializeValueFileRef:
    """Test the file-ref protocol in simple_server.serialize_value()."""

    def test_large_array_becomes_file_ref(self, tmp_path):
        """Arrays > 1MB should be written as TIFF and return a file ref dict."""
        # Import the function under test
        import sys

        backend_path = str(Path(__file__).parent.parent / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        try:
            from simple_server import serialize_value
        except ImportError:
            pytest.skip("backend/simple_server.py not importable")

        volume_dir = tmp_path / "incoming"
        volume_dir.mkdir()

        # Create a large array (> 1MB)
        arr = np.zeros((50, 256, 256), dtype=np.uint16)

        result = serialize_value(arr, volume_dir=str(volume_dir))

        assert isinstance(result, dict)
        assert result.get("__file_ref__") is True
        assert "path" in result
        assert Path(result["path"]).exists()
        assert result["shape"] == list(arr.shape)
        assert result["dtype"] == "uint16"

    def test_small_array_stays_inline(self, tmp_path):
        """Arrays < 1MB should be serialized normally (as list)."""
        import sys

        backend_path = str(Path(__file__).parent.parent / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        try:
            from simple_server import serialize_value
        except ImportError:
            pytest.skip("backend/simple_server.py not importable")

        volume_dir = tmp_path / "incoming"
        volume_dir.mkdir()

        # Small array (< 1MB)
        arr = np.array([1, 2, 3], dtype=np.int32)

        result = serialize_value(arr, volume_dir=str(volume_dir))
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_no_volume_dir_falls_back(self):
        """Without volume_dir, serialize_value behaves as before (list)."""
        import sys

        backend_path = str(Path(__file__).parent.parent / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        try:
            from simple_server import serialize_value
        except ImportError:
            pytest.skip("backend/simple_server.py not importable")

        arr = np.zeros((50, 256, 256), dtype=np.uint16)
        result = serialize_value(arr, volume_dir=None)
        assert isinstance(result, list)


# =========================================================================
# Client-side _resolve_file_refs tests
# =========================================================================


class TestResolveFileRefs:
    """Test _resolve_file_refs() in QueueServerClient."""

    def test_resolve_file_ref_dict(self, tmp_path):
        """A file ref dict should be resolved to numpy array."""
        try:
            from gently.app.queue_server_client import QueueServerClient
        except ImportError:
            pytest.skip("QueueServerClient not importable")

        # Write a test TIFF
        volume = _make_volume(shape=(2, 20, 64, 64))
        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(str(tiff_path), volume)

        file_ref = {
            "__file_ref__": True,
            "path": str(tiff_path),
            "shape": list(volume.shape),
            "dtype": "uint16",
        }

        client = QueueServerClient.__new__(QueueServerClient)
        resolved, path = client._resolve_file_ref(file_ref)

        np.testing.assert_array_equal(resolved, volume)
        assert Path(path) == tiff_path

    def test_resolve_file_refs_nested(self, tmp_path):
        """_resolve_file_refs should walk nested dicts and replace file refs."""
        try:
            from gently.app.queue_server_client import QueueServerClient
        except ImportError:
            pytest.skip("QueueServerClient not importable")

        volume = _make_volume(shape=(2, 10, 32, 32))
        tiff_path = tmp_path / "nested.tif"
        tifffile.imwrite(str(tiff_path), volume)

        data = {
            "status": "success",
            "result": {
                "data": {
                    "__file_ref__": True,
                    "path": str(tiff_path),
                    "shape": list(volume.shape),
                    "dtype": "uint16",
                },
                "other_key": "kept",
            },
        }

        client = QueueServerClient.__new__(QueueServerClient)
        resolved = client._resolve_file_refs(data)

        assert resolved["status"] == "success"
        assert resolved["result"]["other_key"] == "kept"
        np.testing.assert_array_equal(resolved["result"]["data"], volume)
        paths = resolved.get("__resolved_paths__", {})
        assert len(paths) == 1
        assert Path(list(paths.values())[0]) == tiff_path

    def test_is_file_ref(self):
        """Test file ref detection."""
        try:
            from gently.app.queue_server_client import QueueServerClient
        except ImportError:
            pytest.skip("QueueServerClient not importable")

        client = QueueServerClient.__new__(QueueServerClient)

        assert client._is_file_ref({"__file_ref__": True, "path": "/a.tif"})
        assert not client._is_file_ref({"key": "value"})
        assert not client._is_file_ref("string")
        assert not client._is_file_ref(42)
