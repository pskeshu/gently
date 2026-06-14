"""
Tests for TransferService + TransferTracker.
"""

import time

import pytest

from gently.mesh.transfer.models import TransferJob, TransferStatus, TransferType
from gently.mesh.transfer.tracker import TransferTracker


class TestTransferTracker:
    def test_add_and_get_job(self, config_dir):
        tracker = TransferTracker(config_dir)
        job = TransferJob(
            id="job-1",
            transfer_type=TransferType.DATASET.value,
            peer_instance_id="peer-a",
            peer_hostname="peer-b",
            status=TransferStatus.PENDING.value,
        )
        tracker.add_job(job)
        fetched = tracker.get_job("job-1")
        assert fetched is not None
        assert fetched.id == "job-1"
        assert fetched.transfer_type == TransferType.DATASET.value

    def test_update_job(self, config_dir):
        tracker = TransferTracker(config_dir)
        job = TransferJob(
            id="job-2",
            transfer_type=TransferType.MODEL_WEIGHTS.value,
            status=TransferStatus.PENDING.value,
        )
        tracker.add_job(job)
        tracker.update_job(
            "job-2", status=TransferStatus.TRANSFERRING.value, bytes_transferred=1024
        )
        updated = tracker.get_job("job-2")
        assert updated.status == TransferStatus.TRANSFERRING.value
        assert updated.bytes_transferred == 1024

    def test_list_jobs(self, config_dir):
        tracker = TransferTracker(config_dir)
        for i in range(3):
            tracker.add_job(
                TransferJob(
                    id=f"job-{i}",
                    status=TransferStatus.PENDING.value
                    if i < 2
                    else TransferStatus.COMPLETED.value,
                )
            )
        all_jobs = tracker.list_jobs()
        assert len(all_jobs) == 3
        pending = tracker.list_jobs(status=TransferStatus.PENDING.value)
        assert len(pending) == 2

    def test_persistence(self, config_dir):
        """Jobs persist across tracker instances."""
        tracker1 = TransferTracker(config_dir)
        tracker1.add_job(TransferJob(id="persist-1", status=TransferStatus.PENDING.value))

        tracker2 = TransferTracker(config_dir)
        assert tracker2.get_job("persist-1") is not None

    def test_get_resumable(self, config_dir):
        tracker = TransferTracker(config_dir)
        # Job in progress with partial transfer
        tracker.add_job(
            TransferJob(
                id="resume-1",
                status=TransferStatus.TRANSFERRING.value,
                bytes_transferred=5000,
                total_bytes=10000,
            )
        )
        # Completed job (not resumable)
        tracker.add_job(
            TransferJob(
                id="done-1",
                status=TransferStatus.COMPLETED.value,
                bytes_transferred=10000,
                total_bytes=10000,
            )
        )
        resumable = tracker.get_resumable()
        assert len(resumable) == 1
        assert resumable[0].id == "resume-1"

    def test_cleanup_completed(self, config_dir):
        tracker = TransferTracker(config_dir)
        tracker.add_job(
            TransferJob(
                id="old-1",
                status=TransferStatus.COMPLETED.value,
                completed_at=time.time() - 100000,  # very old
            )
        )
        tracker.add_job(
            TransferJob(
                id="recent-1",
                status=TransferStatus.COMPLETED.value,
                completed_at=time.time(),
            )
        )
        tracker.cleanup_completed(max_age_hours=1.0)
        assert tracker.get_job("old-1") is None
        assert tracker.get_job("recent-1") is not None

    def test_empty_tracker(self, config_dir):
        tracker = TransferTracker(config_dir)
        assert tracker.list_jobs() == []
        assert tracker.get_resumable() == []
        assert tracker.get_job("nonexistent") is None


class TestTransferServiceLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop(self, tmp_path):
        from gently.mesh.transfer.server import TransferService

        dest = tmp_path / "received"
        dest.mkdir()
        svc = TransferService(dest_dir=dest, port=0)
        # Just test that on_start/on_stop don't crash
        await svc.on_start()
        assert svc._server is not None
        await svc.on_stop()
        assert svc._server is None

    @pytest.mark.asyncio
    async def test_active_transfer_count(self, tmp_path):
        from gently.mesh.transfer.server import TransferService

        dest = tmp_path / "received"
        dest.mkdir()
        svc = TransferService(dest_dir=dest, port=0)
        assert svc.active_transfer_count == 0
