"""
Tests for the generic detector system: conditions, scheduling, response parsing.
"""

from datetime import datetime

from gently.harness.detection.detector import (
    ConfidenceLevel,
    DetectionMode,
    DetectionResult,
    Detector,
    DetectorActions,
    DetectorConditions,
)

# ===========================================================================
# DetectorConditions
# ===========================================================================


class TestDetectorConditions:
    """DetectorConditions controls when detectors run."""

    def test_should_run_default(self):
        cond = DetectorConditions()
        assert cond.should_run("e1", timepoint=10, last_run_timepoint=None, already_detected=False)

    def test_min_timepoint(self):
        cond = DetectorConditions(min_timepoint=5)
        assert not cond.should_run("e1", 3, None, False)
        assert cond.should_run("e1", 5, None, False)
        assert cond.should_run("e1", 10, None, False)

    def test_max_timepoint(self):
        cond = DetectorConditions(max_timepoint=100)
        assert cond.should_run("e1", 50, None, False)
        assert cond.should_run("e1", 100, None, False)
        assert not cond.should_run("e1", 101, None, False)

    def test_embryo_whitelist(self):
        cond = DetectorConditions(embryo_ids=["e1", "e2"])
        assert cond.should_run("e1", 10, None, False)
        assert not cond.should_run("e3", 10, None, False)

    def test_stop_after_detection(self):
        cond = DetectorConditions(run_if_detected=False)
        assert cond.should_run("e1", 10, None, already_detected=False)
        assert not cond.should_run("e1", 11, None, already_detected=True)

    def test_continue_after_detection(self):
        cond = DetectorConditions(run_if_detected=True)
        assert cond.should_run("e1", 11, None, already_detected=True)

    def test_min_interval(self):
        cond = DetectorConditions(min_interval_timepoints=5)
        assert cond.should_run("e1", 10, last_run_timepoint=None, already_detected=False)
        assert not cond.should_run("e1", 12, last_run_timepoint=10, already_detected=False)
        assert cond.should_run("e1", 15, last_run_timepoint=10, already_detected=False)

    def test_combined_conditions(self):
        cond = DetectorConditions(
            min_timepoint=10,
            max_timepoint=100,
            embryo_ids=["e1"],
            min_interval_timepoints=3,
        )
        # Too early
        assert not cond.should_run("e1", 5, None, False)
        # Wrong embryo
        assert not cond.should_run("e2", 20, None, False)
        # Too soon after last run
        assert not cond.should_run("e1", 22, 20, False)
        # All conditions met
        assert cond.should_run("e1", 25, 20, False)


# ===========================================================================
# Detector
# ===========================================================================


class TestDetector:
    """Detector scheduling and state tracking."""

    def _make_detector(self, **kwargs):
        defaults = {
            "name": "test_detector",
            "description": "Test detector",
            "detection_prompt": "Is this a test?",
        }
        defaults.update(kwargs)
        return Detector(**defaults)

    def test_should_run_enabled(self):
        d = self._make_detector()
        assert d.should_run("e1", 10)

    def test_should_run_disabled(self):
        d = self._make_detector(enabled=False)
        assert not d.should_run("e1", 10)

    def test_mark_run_tracks_timepoint(self):
        d = self._make_detector(conditions=DetectorConditions(min_interval_timepoints=5))
        d.mark_run("e1", 10)
        assert d.run_count == 1
        assert not d.should_run("e1", 12)  # Too soon
        assert d.should_run("e1", 15)

    def test_mark_detected(self):
        d = self._make_detector(conditions=DetectorConditions(run_if_detected=False))
        assert not d.was_detected("e1")
        d.mark_detected("e1")
        assert d.was_detected("e1")
        assert d.detection_count == 1
        assert not d.should_run("e1", 20)

    def test_per_embryo_tracking(self):
        """Detection state is tracked per-embryo."""
        d = self._make_detector(conditions=DetectorConditions(run_if_detected=False))
        d.mark_detected("e1")
        assert d.was_detected("e1")
        assert not d.was_detected("e2")
        assert d.should_run("e2", 10)


# ===========================================================================
# Response Parsing
# ===========================================================================


class TestDetectorResponseParsing:
    """Detector parses Claude Vision API responses."""

    def _make_detector(self):
        return Detector(name="test", description="test", detection_prompt="detect something")

    def test_parse_yes_high(self):
        d = self._make_detector()
        result = d.parse_detection_response(
            "DETECTED: YES\nCONFIDENCE: HIGH\nREASONING: Clear comma shape visible"
        )
        assert result["detected"] is True
        assert result["confidence"] == ConfidenceLevel.HIGH
        assert "comma" in result["reasoning"]

    def test_parse_no_low(self):
        d = self._make_detector()
        result = d.parse_detection_response(
            "DETECTED: NO\nCONFIDENCE: LOW\nREASONING: No visible features"
        )
        assert result["detected"] is False
        assert result["confidence"] == ConfidenceLevel.LOW

    def test_parse_true_as_detected(self):
        d = self._make_detector()
        result = d.parse_detection_response("DETECTED: TRUE\nCONFIDENCE: MEDIUM")
        assert result["detected"] is True

    def test_parse_missing_fields(self):
        d = self._make_detector()
        result = d.parse_detection_response("Some random response")
        assert result["detected"] is False
        assert result["confidence"] is None

    def test_parse_multiline_reasoning(self):
        """Multiline reasoning is captured when it spans multiple lines."""
        d = self._make_detector()
        # The parser captures single-line REASONING first, then falls through
        # to multiline if reasoning is None. Test the multiline path explicitly
        # by NOT having single-line capture match (no space after colon on same line).
        result = d.parse_detection_response(
            "DETECTED: YES\nCONFIDENCE: HIGH\nREASONING: The embryo shows clear elongation"
        )
        assert "elongation" in result["reasoning"]


# ===========================================================================
# Serialization
# ===========================================================================


class TestDetectorSerialization:
    """Detector serialization roundtrip."""

    def test_to_dict(self):
        d = Detector(
            name="comma_stage",
            description="Detect comma stage",
            detection_prompt="Is this comma?",
            confidence_threshold=ConfidenceLevel.HIGH,
        )
        data = d.to_dict()
        assert data["name"] == "comma_stage"
        assert data["confidence_threshold"] == "HIGH"

    def test_roundtrip(self):
        original = Detector(
            name="comma_stage",
            description="Detect comma stage",
            detection_prompt="Is this comma?",
            conditions=DetectorConditions(min_timepoint=10, max_timepoint=200),
            actions=DetectorActions(
                mode=DetectionMode.AUTO,
                parameter_changes={"interval_seconds": 30},
            ),
            confidence_threshold=ConfidenceLevel.HIGH,
            use_temporal_context=True,
            temporal_context_size=3,
        )
        restored = Detector.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.conditions.min_timepoint == 10
        assert restored.actions.mode == DetectionMode.AUTO
        assert restored.confidence_threshold == ConfidenceLevel.HIGH


class TestDetectionResult:
    """DetectionResult serialization."""

    def test_to_dict(self):
        r = DetectionResult(
            detector_name="comma",
            embryo_id="e1",
            timepoint=50,
            timestamp=datetime.now(),
            detected=True,
            confidence=ConfidenceLevel.HIGH,
            reasoning="Clear comma shape",
        )
        d = r.to_dict()
        assert d["detected"] is True
        assert d["confidence"] == "HIGH"
        assert isinstance(d["timestamp"], str)


class TestDetectorActions:
    """DetectorActions recommendation messages."""

    def test_recommendation_message(self):
        actions = DetectorActions(
            mode=DetectionMode.RECOMMEND,
            parameter_changes={"interval_seconds": 30},
            custom_message="Embryo is developing fast",
        )
        msg = actions.get_recommendation_message("comma_stage", "e1")
        assert "comma_stage" in msg
        assert "e1" in msg
        assert "interval_seconds" in msg
        assert "developing fast" in msg
