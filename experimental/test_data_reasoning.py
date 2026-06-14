"""
Tests for data reasoning engine — coverage, assessment, gap planning.
"""

from unittest.mock import MagicMock

import pytest
from gently.data_reasoning.assessment import DataAssessmentEngine
from gently.data_reasoning.coverage import CoverageAnalyzer
from gently.data_reasoning.gap_planner import GapPlanner
from gently.data_reasoning.models import (
    CoverageReport,
)


def _make_mock_store(sessions=None, embryos_per_session=None, gt_per_embryo=None):
    """Build a mock GentlyStore with controlled data."""
    store = MagicMock()
    sessions = sessions or []
    embryos_per_session = embryos_per_session or {}
    gt_per_embryo = gt_per_embryo or {}

    mock_sessions = []
    for sid, name in sessions:
        s = MagicMock()
        s.session_id = sid
        s.name = name
        s.created_at = "2024-01-01"
        s.last_active = "2024-01-01"
        mock_sessions.append(s)
    store.list_sessions.return_value = mock_sessions

    def list_embryos(sid):
        emb_ids = embryos_per_session.get(sid, [])
        result = []
        for eid in emb_ids:
            e = MagicMock()
            e.embryo_id = eid
            e.nickname = eid
            result.append(e)
        return result

    store.list_embryos.side_effect = list_embryos

    def list_volumes(sid, eid):
        return [MagicMock(timepoint=i) for i in range(3)]

    store.list_volumes.side_effect = list_volumes

    def get_ground_truth(sid, eid):
        stages = gt_per_embryo.get((sid, eid), [])
        result = []
        for stage in stages:
            gt = MagicMock()
            gt.stage = stage
            gt.start_tp = 0
            gt.end_tp = 10
            result.append(gt)
        return result

    store.get_ground_truth.side_effect = get_ground_truth

    return store


class TestCoverageAnalyzer:
    def test_empty_store(self):
        store = _make_mock_store()
        analyzer = CoverageAnalyzer(gently_store=store)
        report = analyzer.analyze()
        assert report.total_embryos == 0
        assert report.coverage_pct == 0.0

    def test_full_coverage(self):
        store = _make_mock_store(
            sessions=[("s1", "Session 1")],
            embryos_per_session={"s1": ["e1", "e2"]},
            gt_per_embryo={("s1", "e1"): ["early"], ("s1", "e2"): ["comma"]},
        )
        analyzer = CoverageAnalyzer(gently_store=store)
        report = analyzer.analyze()
        assert report.total_embryos == 2
        assert report.annotated_embryos == 2
        assert report.coverage_pct == 100.0
        assert "early" in report.stage_counts
        assert "comma" in report.stage_counts

    def test_partial_coverage(self):
        store = _make_mock_store(
            sessions=[("s1", "Session 1")],
            embryos_per_session={"s1": ["e1", "e2", "e3", "e4"]},
            gt_per_embryo={("s1", "e1"): ["early"], ("s1", "e2"): ["comma"]},
        )
        analyzer = CoverageAnalyzer(gently_store=store)
        report = analyzer.analyze()
        assert report.coverage_pct == 50.0
        assert report.annotated_embryos == 2

    def test_imbalance_detection(self):
        store = _make_mock_store(
            sessions=[("s1", "Session 1")],
            embryos_per_session={"s1": ["e1", "e2", "e3"]},
            gt_per_embryo={
                ("s1", "e1"): ["early", "early", "early"],  # 3 early
                ("s1", "e2"): ["comma"],  # 1 comma
                ("s1", "e3"): ["early"],  # 1 more early
            },
        )
        analyzer = CoverageAnalyzer(gently_store=store)
        report = analyzer.analyze()
        assert report.imbalance_ratio > 1.0

    def test_gap_detection(self):
        store = _make_mock_store(
            sessions=[("s1", "Session 1")],
            embryos_per_session={"s1": ["e1"]},
            gt_per_embryo={("s1", "e1"): ["early"]},
        )
        analyzer = CoverageAnalyzer(gently_store=store)
        report = analyzer.analyze()
        # Should find gaps for missing known stages
        assert len(report.gaps) > 0

    def test_recommendations_generated(self):
        store = _make_mock_store(
            sessions=[("s1", "Session 1")],
            embryos_per_session={"s1": ["e1"]},
            gt_per_embryo={("s1", "e1"): ["early"]},
        )
        analyzer = CoverageAnalyzer(gently_store=store)
        report = analyzer.analyze()
        assert len(report.recommendations) > 0

    def test_no_store(self):
        analyzer = CoverageAnalyzer(gently_store=None)
        report = analyzer.analyze()
        assert report.total_embryos == 0


class TestDataAssessment:
    def test_local_inventory(self):
        store = _make_mock_store(
            sessions=[("s1", "Session 1"), ("s2", "Session 2")],
            embryos_per_session={"s1": ["e1", "e2"], "s2": ["e3"]},
            gt_per_embryo={("s1", "e1"): ["early"]},
        )
        engine = DataAssessmentEngine(gently_store=store)
        sessions = engine.inventory_local()
        assert len(sessions) == 2
        assert sessions[0].session_id == "s1"
        assert sessions[0].embryo_count == 2
        assert sessions[0].is_remote is False

    def test_no_store(self):
        engine = DataAssessmentEngine()
        sessions = engine.inventory_local()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_build_inventory(self):
        store = _make_mock_store(
            sessions=[("s1", "Session 1")],
            embryos_per_session={"s1": ["e1"]},
            gt_per_embryo={("s1", "e1"): ["early"]},
        )
        engine = DataAssessmentEngine(gently_store=store)
        inventory = await engine.build_inventory(include_remote=False)
        assert inventory.total_embryos == 1
        assert len(inventory.local_sessions) == 1
        assert len(inventory.remote_sessions) == 0


class TestGapPlanner:
    def test_no_context_store(self):
        planner = GapPlanner(context_store=None)
        report = CoverageReport(stage_counts={"early": 10})
        ids = planner.plan_annotation_campaign("c1", report)
        assert ids == []

    def test_creates_items_for_gaps(self, context_store):
        campaign_id = context_store.create_campaign(description="ML Training")
        report = CoverageReport(
            stage_counts={"early": 10, "comma": 5},
            gaps=["early underrepresented", "comma underrepresented"],
        )
        planner = GapPlanner(context_store=context_store)
        ids = planner.plan_annotation_campaign(
            campaign_id=campaign_id,
            coverage_report=report,
            target_per_stage=50,
        )
        # Should create items for stages below target + missing known stages
        assert len(ids) > 0
