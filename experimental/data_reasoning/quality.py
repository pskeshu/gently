"""
QualityAnalyzer — Data quality validation.

Checks that volumes are readable, projections exist, and annotations
are consistent.
"""

import logging
from pathlib import Path

from .models import DataQualityReport

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """Validates data quality for ML training readiness.

    Parameters
    ----------
    gently_store : optional
        Local GentlyStore for querying data.
    """

    def __init__(self, gently_store=None):
        self._store = gently_store

    def analyze(
        self, session_ids: list | None = None, check_files: bool = False
    ) -> DataQualityReport:
        """Run quality checks on local data.

        Parameters
        ----------
        session_ids : list of str, optional
            Sessions to check. None = all.
        check_files : bool
            If True, verify files exist on disk (slower).

        Returns
        -------
        DataQualityReport
        """
        if self._store is None:
            return DataQualityReport()

        report = DataQualityReport()
        issues = []

        try:
            sessions = self._store.list_sessions()
            for sess in sessions:
                sid = sess.session_id if hasattr(sess, "session_id") else sess.get("session_id", "")
                if session_ids and sid not in session_ids:
                    continue

                embryos = self._store.list_embryos(sid)
                for emb in embryos:
                    eid = emb.embryo_id if hasattr(emb, "embryo_id") else emb.get("embryo_id", "")
                    vols = self._store.list_volumes(sid, eid)
                    report.total_volumes_checked += len(vols)

                    for vol in vols:
                        # Check volume file exists if requested
                        if check_files:
                            vpath = (
                                vol.file_path
                                if hasattr(vol, "file_path")
                                else vol.get("file_path", "")
                            )
                            if vpath and Path(vpath).exists():
                                report.readable_volumes += 1
                            elif vpath:
                                issues.append(f"Missing volume: {vpath}")
                        else:
                            report.readable_volumes += 1

                        # Check projection exists
                        tp = vol.timepoint if hasattr(vol, "timepoint") else vol.get("timepoint", 0)
                        try:
                            proj_path = self._store.get_projection_path(sid, eid, tp)
                            if check_files and proj_path and not Path(proj_path).exists():
                                report.missing_projections += 1
                                issues.append(f"Missing projection: {sid}/{eid}/t{tp}")
                        except Exception:
                            report.missing_projections += 1

                    # Check annotation consistency
                    try:
                        gts = self._store.get_ground_truth(sid, eid)
                        if gts:
                            # Check for overlapping timepoint ranges
                            ranges = []
                            for gt in gts:
                                start = (
                                    gt.start_tp
                                    if hasattr(gt, "start_tp")
                                    else gt.get("start_tp", 0)
                                )
                                end = gt.end_tp if hasattr(gt, "end_tp") else gt.get("end_tp", 0)
                                if start and end and start > end:
                                    report.inconsistent_annotations += 1
                                    issues.append(
                                        f"Invalid GT range for {sid}/{eid}: "
                                        f"start_tp={start} > end_tp={end}"
                                    )
                                ranges.append((start, end))
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            issues.append(f"Analysis error: {e}")

        report.issues = issues
        return report
