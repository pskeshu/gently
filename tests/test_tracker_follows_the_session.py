"""The timelapse tracker is the live session's, and nothing else's.

Switching to a session with no embryos used to show the previous session's
embryos as ghost tiles, because rehydration never cleared the tracker.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gently.core.file_store import FileStore
from gently.ui.web.server import VisualizationServer
from gently.ui.web.timelapse_tracker import TimelapseStateTracker as TimelapseTracker

pytest.importorskip("tifffile")


def _session_with_volumes(store, sid, embryo_ids):
    store.create_session(sid)
    for i, eid in enumerate(embryo_ids):
        store.register_embryo(sid, eid, position_x=1.0, position_y=2.0, role="test")
        for tp in range(1, 3 + i):
            store.put_volume(sid, eid, tp, np.full((2, 4, 4), tp, dtype=np.uint16))
    return sid


class TestARestoredSessionReplacesTheTracker:
    def test_session_restored_clears_the_previous_run(self):
        t = TimelapseTracker()
        t.handle_event(
            "ACQUISITION_STARTED", {"embryo_ids": ["embryo_1", "embryo_2"], "interval_seconds": 60}
        )
        t.handle_event("VOLUME_ACQUIRED", {"embryo_id": "embryo_1", "timepoint": 4})
        assert t.status == "RUNNING" and set(t.embryos) == {"embryo_1", "embryo_2"}
        t.handle_event("SESSION_RESTORED", {"session_id": "old00001"})
        assert t.session_id == "old00001"
        assert t.status == "IDLE" and t.embryos == {} and t.total_timepoints == 0

    def test_rehydration_lists_the_sessions_own_embryos_and_no_others(self, tmp_path):
        store = FileStore(root=tmp_path)
        _session_with_volumes(store, "aaaa0001", ["embryo_1", "embryo_2", "embryo_3"])
        store.create_session("bbbb0002")  # brightfield-only: frames, no embryos
        tracker = TimelapseTracker()
        tracker.handle_event(
            "ACQUISITION_STARTED", {"embryo_ids": ["embryo_9"], "interval_seconds": 60}
        )
        fake = SimpleNamespace(gently_store=store, timelapse_tracker=tracker, store=None)

        VisualizationServer.rehydrate_session(fake, "aaaa0001")
        assert sorted(tracker.embryos) == ["embryo_1", "embryo_2", "embryo_3"]
        assert tracker.embryos["embryo_3"]["timepoints"] == 4  # how far its volumes got
        assert tracker.status == "IDLE" and tracker.session_id == "aaaa0001"

        VisualizationServer.rehydrate_session(fake, "bbbb0002")
        assert tracker.embryos == {} and tracker.session_id == "bbbb0002"
