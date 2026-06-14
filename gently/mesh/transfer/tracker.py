"""
TransferTracker — Persists transfer state for resume on restart.
"""

import json
import logging
from pathlib import Path

from .models import TransferJob, TransferStatus

logger = logging.getLogger(__name__)


class TransferTracker:
    """Persists transfer state to a JSON file.

    Parameters
    ----------
    config_dir : Path
        Directory for the transfers state file.
    """

    def __init__(self, config_dir: Path):
        self._config_dir = config_dir
        self._state_file = config_dir / "mesh_transfers.json"
        self._jobs: dict[str, TransferJob] = {}
        self._load()

    def _load(self):
        """Load transfer state from disk."""
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text())
            for entry in data:
                job = TransferJob.from_dict(entry)
                if job.id:
                    self._jobs[job.id] = job
            logger.info(f"TransferTracker: loaded {len(self._jobs)} transfers")
        except Exception as e:
            logger.warning(f"TransferTracker: failed to load: {e}")

    def _save(self):
        """Persist transfer state to disk."""
        data = [job.to_dict() for job in self._jobs.values()]
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"TransferTracker: failed to save: {e}")

    def add_job(self, job: TransferJob):
        """Track a new transfer job."""
        self._jobs[job.id] = job
        self._save()

    def update_job(self, job_id: str, **kwargs):
        """Update a transfer job."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)
        self._save()

    def get_job(self, job_id: str) -> TransferJob | None:
        """Get a transfer job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self, status: str | None = None) -> list[TransferJob]:
        """List all jobs, optionally filtered by status."""
        if status:
            return [j for j in self._jobs.values() if j.status == status]
        return list(self._jobs.values())

    def get_resumable(self) -> list[TransferJob]:
        """Get transfers that were interrupted and can be resumed."""
        return [
            j
            for j in self._jobs.values()
            if j.status == TransferStatus.TRANSFERRING.value
            and j.bytes_transferred > 0
            and j.bytes_transferred < j.total_bytes
        ]

    def cleanup_completed(self, max_age_hours: float = 24.0):
        """Remove old completed/failed transfers."""
        import time

        cutoff = time.time() - (max_age_hours * 3600)
        to_remove = [
            jid
            for jid, j in self._jobs.items()
            if j.status in (TransferStatus.COMPLETED.value, TransferStatus.FAILED.value)
            and j.completed_at > 0
            and j.completed_at < cutoff
        ]
        for jid in to_remove:
            del self._jobs[jid]
        if to_remove:
            self._save()
