"""
CoverageAnalyzer — Annotation coverage analysis and gap detection.
"""

import logging

from .models import CoverageReport

logger = logging.getLogger(__name__)

# Known C. elegans embryonic stages (common ordering)
KNOWN_STAGES = [
    "early",
    "2-cell",
    "4-cell",
    "8-cell",
    "16-cell",
    "32-cell",
    "64-cell",
    "gastrulation",
    "bean",
    "comma",
    "1.5-fold",
    "2-fold",
    "pretzel",
    "3-fold",
    "hatching",
]

# Minimum recommended samples per stage for training
MIN_SAMPLES_PER_STAGE = 30


class CoverageAnalyzer:
    """Analyzes annotation coverage and generates recommendations.

    Parameters
    ----------
    gently_store : optional
        Local GentlyStore for direct queries.
    """

    def __init__(self, gently_store=None):
        self._store = gently_store

    def analyze(self, session_ids: list[str] | None = None) -> CoverageReport:
        """Analyze annotation coverage across specified sessions (or all).

        Parameters
        ----------
        session_ids : list of str, optional
            Sessions to analyze. If None, analyzes all sessions.

        Returns
        -------
        CoverageReport
        """
        if self._store is None:
            return CoverageReport()

        total_embryos = 0
        annotated_embryos = 0
        stage_counts: dict[str, int] = {}

        try:
            sessions = self._store.list_sessions()
            for sess in sessions:
                sid = sess.session_id if hasattr(sess, "session_id") else sess.get("session_id", "")
                if session_ids and sid not in session_ids:
                    continue

                embryos = self._store.list_embryos(sid)
                total_embryos += len(embryos)

                for emb in embryos:
                    eid = emb.embryo_id if hasattr(emb, "embryo_id") else emb.get("embryo_id", "")
                    try:
                        gts = self._store.get_ground_truth(sid, eid)
                        if gts:
                            annotated_embryos += 1
                            for gt in gts:
                                stage = gt.stage if hasattr(gt, "stage") else gt.get("stage", "")
                                if stage:
                                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            return CoverageReport()

        coverage_pct = (annotated_embryos / total_embryos * 100) if total_embryos else 0.0

        # Compute imbalance ratio
        counts = list(stage_counts.values())
        if counts and min(counts) > 0:
            imbalance_ratio = max(counts) / min(counts)
        else:
            imbalance_ratio = 0.0

        # Find gaps and generate recommendations
        gaps = self._find_gaps(stage_counts)
        recommendations = self._generate_recommendations(
            total_embryos,
            annotated_embryos,
            coverage_pct,
            stage_counts,
            gaps,
        )

        return CoverageReport(
            total_embryos=total_embryos,
            annotated_embryos=annotated_embryos,
            coverage_pct=round(coverage_pct, 1),
            stage_counts=stage_counts,
            imbalance_ratio=round(imbalance_ratio, 2),
            gaps=gaps,
            recommendations=recommendations,
        )

    def analyze_from_inventory(self, inventory) -> CoverageReport:
        """Build a coverage report from a NetworkDataInventory."""
        stage_counts: dict[str, int] = {}
        total_embryos = inventory.total_embryos
        annotated_embryos = inventory.total_annotated

        for sess in inventory.all_sessions:
            for stage in sess.stages_covered:
                # Approximate: we know stages covered but not exact counts from remote
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

        coverage_pct = (annotated_embryos / total_embryos * 100) if total_embryos else 0.0
        counts = list(stage_counts.values())
        imbalance_ratio = (max(counts) / min(counts)) if counts and min(counts) > 0 else 0.0
        gaps = self._find_gaps(stage_counts)
        recommendations = self._generate_recommendations(
            total_embryos,
            annotated_embryos,
            coverage_pct,
            stage_counts,
            gaps,
        )

        return CoverageReport(
            total_embryos=total_embryos,
            annotated_embryos=annotated_embryos,
            coverage_pct=round(coverage_pct, 1),
            stage_counts=stage_counts,
            imbalance_ratio=round(imbalance_ratio, 2),
            gaps=gaps,
            recommendations=recommendations,
        )

    def _find_gaps(self, stage_counts: dict[str, int]) -> list[str]:
        """Identify underrepresented stages."""
        gaps = []
        if not stage_counts:
            return ["No annotations found"]

        avg = sum(stage_counts.values()) / len(stage_counts) if stage_counts else 0

        # Stages with too few samples
        for stage, count in stage_counts.items():
            if count < MIN_SAMPLES_PER_STAGE:
                gaps.append(
                    f"{stage} underrepresented ({count} samples, need {MIN_SAMPLES_PER_STAGE})"
                )
            elif count < avg * 0.5:
                gaps.append(f"{stage} below average ({count} vs avg {avg:.0f})")

        # Known stages completely missing
        present = set(stage_counts.keys())
        for stage in KNOWN_STAGES:
            if stage not in present:
                gaps.append(f"{stage} missing entirely")

        return gaps

    def _generate_recommendations(
        self,
        total_embryos: int,
        annotated: int,
        coverage_pct: float,
        stage_counts: dict[str, int],
        gaps: list[str],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs = []

        if total_embryos == 0:
            recs.append("No embryo data found. Acquire some imaging sessions first.")
            return recs

        if coverage_pct < 50:
            recs.append(
                f"Annotation coverage is {coverage_pct:.0f}%. "
                f"Annotate at least {total_embryos - annotated} more embryos to reach 50%."
            )

        if coverage_pct < 80:
            recs.append("Consider a focused annotation campaign to improve coverage.")

        for stage in KNOWN_STAGES:
            count = stage_counts.get(stage, 0)
            if count < MIN_SAMPLES_PER_STAGE:
                needed = MIN_SAMPLES_PER_STAGE - count
                recs.append(f"Need {needed} more {stage} annotations (have {count}).")

        if not gaps and coverage_pct >= 80:
            recs.append("Good coverage! Data is ready for training.")

        return recs
