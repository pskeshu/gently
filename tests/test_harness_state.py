"""
Tests for harness state management: FocusDataPoint, EmbryoState, ExperimentState.
"""

from datetime import datetime, timedelta

from gently.harness.state import EmbryoState, ExperimentState, FocusDataPoint

# ===========================================================================
# FocusDataPoint
# ===========================================================================


class TestFocusDataPoint:
    """FocusDataPoint serialization and backward compatibility."""

    def test_creation(self):
        fp = FocusDataPoint(
            z=50.0,
            secondary_axis=1.5,
            score=0.9,
            r_squared=0.95,
            timestamp=datetime.now(),
            method="calibration",
        )
        assert fp.z == 50.0
        assert fp.secondary_axis == 1.5

    def test_backward_compat_properties(self):
        fp = FocusDataPoint(
            z=50.0,
            secondary_axis=1.5,
            score=0.9,
            r_squared=0.95,
            timestamp=datetime.now(),
            method="fine_focus",
        )
        assert fp.piezo == 50.0
        assert fp.galvo == 1.5

    def test_to_dict_has_both_keys(self):
        fp = FocusDataPoint(
            z=50.0,
            secondary_axis=1.5,
            score=0.9,
            r_squared=0.95,
            timestamp=datetime.now(),
            method="manual",
        )
        d = fp.to_dict()
        # New keys
        assert d["z"] == 50.0
        assert d["secondary_axis"] == 1.5
        # Backward-compat keys
        assert d["piezo"] == 50.0
        assert d["galvo"] == 1.5

    def test_from_dict_new_keys(self):
        d = {
            "z": 60.0,
            "secondary_axis": 2.0,
            "score": 0.8,
            "r_squared": 0.9,
            "timestamp": datetime.now().isoformat(),
            "method": "calibration",
        }
        fp = FocusDataPoint.from_dict(d)
        assert fp.z == 60.0
        assert fp.secondary_axis == 2.0

    def test_from_dict_old_keys(self):
        """Deserializing data with old key names (piezo/galvo) must work."""
        d = {
            "piezo": 70.0,
            "galvo": 3.0,
            "score": 0.85,
            "r_squared": 0.92,
            "timestamp": datetime.now().isoformat(),
            "method": "fine_focus",
        }
        fp = FocusDataPoint.from_dict(d)
        assert fp.z == 70.0
        assert fp.secondary_axis == 3.0

    def test_roundtrip(self):
        original = FocusDataPoint(
            z=42.0,
            secondary_axis=1.1,
            score=0.7,
            r_squared=0.88,
            timestamp=datetime.now(),
            method="calibration",
            algorithm="gradient",
        )
        restored = FocusDataPoint.from_dict(original.to_dict())
        assert restored.z == original.z
        assert restored.secondary_axis == original.secondary_axis
        assert restored.algorithm == original.algorithm


# ===========================================================================
# EmbryoState — Focus Methods
# ===========================================================================


class TestEmbryoFocus:
    """EmbryoState focus tracking and analysis."""

    def _make_embryo_with_focus(self):
        e = EmbryoState(id="embryo_1")
        now = datetime.now()
        for i in range(5):
            e.focus_history.append(
                FocusDataPoint(
                    z=50.0 + i * 2,
                    secondary_axis=1.0 + i * 0.1,
                    score=0.9,
                    r_squared=0.95,
                    timestamp=now - timedelta(hours=4 - i),
                    method="calibration",
                )
            )
        return e

    def test_add_focus_datapoint_new_args(self):
        e = EmbryoState(id="e1")
        e.add_focus_datapoint(z=50.0, secondary_axis=1.0, score=0.9, r_squared=0.95)
        assert len(e.focus_history) == 1
        assert e.focus_history[0].z == 50.0

    def test_add_focus_datapoint_old_args(self):
        """Backward-compatible kwargs (piezo=, galvo=) must work."""
        e = EmbryoState(id="e1")
        e.add_focus_datapoint(piezo=50.0, galvo=1.0, score=0.9, r_squared=0.95)
        assert len(e.focus_history) == 1
        assert e.focus_history[0].z == 50.0
        assert e.focus_history[0].secondary_axis == 1.0

    def test_get_focus_at_secondary(self):
        e = self._make_embryo_with_focus()
        result = e.get_focus_at_secondary(1.0)
        assert result is not None
        assert isinstance(result, float)

    def test_get_focus_at_galvo_alias(self):
        e = self._make_embryo_with_focus()
        result = e.get_focus_at_galvo(1.0)
        assert result is not None

    def test_get_focus_empty_history(self):
        e = EmbryoState(id="e1")
        assert e.get_focus_at_secondary(1.0) is None

    def test_get_focus_min_r_squared_filter(self):
        e = EmbryoState(id="e1")
        e.focus_history.append(
            FocusDataPoint(
                z=50.0,
                secondary_axis=0.0,
                score=0.5,
                r_squared=0.3,
                timestamp=datetime.now(),
                method="manual",
            )
        )
        # With default min_r_squared=0.5, this low-quality point is filtered
        assert e.get_focus_at_secondary(0.0) is None

    def test_z_axis_fit(self):
        e = self._make_embryo_with_focus()
        fit = e.get_z_axis_fit()
        assert fit is not None
        slope, intercept = fit
        assert isinstance(slope, float)
        assert isinstance(intercept, float)

    def test_z_axis_fit_insufficient_data(self):
        e = EmbryoState(id="e1")
        assert e.get_z_axis_fit() is None
        e.add_focus_datapoint(z=50.0, score=0.9, r_squared=0.95)
        assert e.get_z_axis_fit() is None  # Need at least 2

    def test_piezo_galvo_fit_alias(self):
        e = self._make_embryo_with_focus()
        assert e.get_piezo_galvo_fit() == e.get_z_axis_fit()

    def test_needs_refocus_no_history(self):
        e = EmbryoState(id="e1")
        assert e.needs_refocus() is True

    def test_needs_refocus_recent(self):
        e = EmbryoState(id="e1")
        e.add_focus_datapoint(z=50.0, score=0.9, r_squared=0.95)
        assert e.needs_refocus(max_age_minutes=60) is False

    def test_needs_refocus_old_data(self):
        e = EmbryoState(id="e1")
        e.focus_history.append(
            FocusDataPoint(
                z=50.0,
                secondary_axis=0.0,
                score=0.9,
                r_squared=0.95,
                timestamp=datetime.now() - timedelta(hours=2),
                method="manual",
            )
        )
        assert e.needs_refocus(max_age_minutes=60) is True

    def test_needs_refocus_galvo_kwarg(self):
        e = EmbryoState(id="e1")
        e.add_focus_datapoint(z=50.0, score=0.9, r_squared=0.95)
        assert e.needs_refocus(galvo_position=0.0) is False


# ===========================================================================
# EmbryoState — Detection Methods
# ===========================================================================


class TestEmbryoDetection:
    """EmbryoState detection result tracking."""

    def test_add_detection_result(self):
        e = EmbryoState(id="e1")
        e.add_detection_result("comma_stage", {"timepoint": 50, "detected": True})
        assert len(e.detection_results["comma_stage"]) == 1

    def test_get_latest_detection(self):
        e = EmbryoState(id="e1")
        e.add_detection_result("comma_stage", {"timepoint": 50, "detected": False})
        e.add_detection_result("comma_stage", {"timepoint": 60, "detected": True})
        latest = e.get_latest_detection("comma_stage")
        assert latest["timepoint"] == 60

    def test_get_latest_detection_unknown(self):
        e = EmbryoState(id="e1")
        assert e.get_latest_detection("unknown") is None

    def test_was_detected(self):
        e = EmbryoState(id="e1")
        e.add_detection_result("comma_stage", {"timepoint": 50, "detected": False})
        assert e.was_detected("comma_stage") is False
        e.add_detection_result("comma_stage", {"timepoint": 60, "detected": True})
        assert e.was_detected("comma_stage") is True

    def test_was_detected_require_verified(self):
        e = EmbryoState(id="e1")
        e.add_detection_result("hatching", {"timepoint": 100, "detected": True})
        assert e.was_detected("hatching", require_verified=False) is True
        assert e.was_detected("hatching", require_verified=True) is False

    def test_mark_detection_verified(self):
        e = EmbryoState(id="e1")
        e.add_detection_result("hatching", {"timepoint": 100, "detected": True})
        assert e.mark_detection_verified("hatching") is True
        assert e.was_detected("hatching", require_verified=True) is True

    def test_mark_detection_verified_by_timepoint(self):
        e = EmbryoState(id="e1")
        e.add_detection_result("hatching", {"timepoint": 100, "detected": True})
        e.add_detection_result("hatching", {"timepoint": 110, "detected": True})
        e.mark_detection_verified("hatching", timepoint=100)
        results = e.detection_results["hatching"]
        assert results[0].get("verified") is True
        assert results[1].get("verified") is None


# ===========================================================================
# EmbryoState — CV Results
# ===========================================================================


class TestEmbryoCVResults:
    """EmbryoState CV analysis result tracking."""

    def test_add_cv_result_updates_quick_access(self):
        e = EmbryoState(id="e1")
        e.add_cv_result("nuclei_count", {"timepoint": 5, "num_nuclei": 66})
        assert e.latest_nuclei_count == 66
        e.add_cv_result("stage_classification", {"timepoint": 5, "stage": "comma"})
        assert e.latest_developmental_stage == "comma"

    def test_get_cv_result_latest(self):
        e = EmbryoState(id="e1")
        e.add_cv_result("nuclei_count", {"timepoint": 5, "num_nuclei": 50})
        e.add_cv_result("nuclei_count", {"timepoint": 10, "num_nuclei": 100})
        result = e.get_cv_result("nuclei_count")
        assert result["num_nuclei"] == 100

    def test_get_cv_result_by_timepoint(self):
        e = EmbryoState(id="e1")
        e.add_cv_result("nuclei_count", {"timepoint": 5, "num_nuclei": 50})
        e.add_cv_result("nuclei_count", {"timepoint": 10, "num_nuclei": 100})
        result = e.get_cv_result("nuclei_count", timepoint=5)
        assert result["num_nuclei"] == 50


# ===========================================================================
# EmbryoState — Exposure Tracking
# ===========================================================================


class TestEmbryoExposure:
    """EmbryoState light exposure tracking."""

    def test_record_exposure(self):
        e = EmbryoState(id="e1")
        e.record_exposure(exposure_ms=10.0, num_frames=50)
        assert e.exposure_count == 1
        assert e.total_exposure_ms == 500.0

    def test_cumulative_exposure(self):
        e = EmbryoState(id="e1")
        e.record_exposure(exposure_ms=10.0, num_frames=50)
        e.record_exposure(exposure_ms=10.0, num_frames=50)
        assert e.exposure_count == 2
        assert e.total_exposure_ms == 1000.0

    def test_exposure_summary(self):
        e = EmbryoState(id="e1")
        assert "No light exposure" in e.get_exposure_summary()
        e.record_exposure(exposure_ms=10.0, num_frames=50)
        summary = e.get_exposure_summary()
        assert "1 exposures" in summary


# ===========================================================================
# EmbryoState — Serialization
# ===========================================================================


class TestEmbryoSerialization:
    """EmbryoState to_dict serialization."""

    def test_to_dict_basic(self):
        e = EmbryoState(id="e1", nickname="the fast one")
        d = e.to_dict()
        assert d["id"] == "e1"
        assert d["nickname"] == "the fast one"
        assert d["timepoints_acquired"] == 0

    def test_to_dict_with_focus_history(self):
        e = EmbryoState(id="e1")
        e.add_focus_datapoint(z=50.0, score=0.9, r_squared=0.95)
        d = e.to_dict()
        assert len(d["focus_history"]) == 1
        assert d["focus_history"][0]["z"] == 50.0


# ===========================================================================
# ExperimentState
# ===========================================================================


class TestExperimentState:
    """ExperimentState management."""

    def test_add_embryo_starts_experiment(self):
        exp = ExperimentState()
        assert exp.start_time is None
        exp.add_embryo("e1", position={"x": 1000, "y": 500})
        assert exp.start_time is not None
        assert "e1" in exp.embryos

    def test_remove_embryo(self):
        exp = ExperimentState()
        exp.add_embryo("e1")
        assert exp.remove_embryo("e1") is True
        assert "e1" not in exp.embryos
        assert exp.remove_embryo("e1") is False

    def test_assign_nickname(self):
        exp = ExperimentState()
        exp.add_embryo("e1")
        exp.assign_nickname("e1", "the fast one")
        assert exp.embryos["e1"].nickname == "the fast one"

    def test_get_embryo_by_id(self):
        exp = ExperimentState()
        exp.add_embryo("embryo_1")
        assert exp.get_embryo_by_any_name("embryo_1") is not None

    def test_get_embryo_by_nickname(self):
        exp = ExperimentState()
        exp.add_embryo("embryo_1")
        exp.assign_nickname("embryo_1", "speedy")
        assert exp.get_embryo_by_any_name("speedy") is not None

    def test_get_embryo_by_number(self):
        exp = ExperimentState()
        exp.add_embryo("embryo_1")
        assert exp.get_embryo_by_any_name("embryo 1") is not None

    def test_get_embryo_by_padded_number(self):
        exp = ExperimentState()
        exp.add_embryo("embryo_003")
        assert exp.get_embryo_by_any_name("embryo 3") is not None

    def test_get_embryo_not_found(self):
        exp = ExperimentState()
        assert exp.get_embryo_by_any_name("nonexistent") is None

    def test_to_dict(self):
        exp = ExperimentState()
        exp.add_embryo("e1")
        d = exp.to_dict()
        assert d["embryo_count"] == 1
        assert "e1" in d["embryos"]
        assert "calibration_prior" in d

    def test_summary_no_experiment(self):
        exp = ExperimentState()
        assert "No active experiment" in exp.get_summary()
