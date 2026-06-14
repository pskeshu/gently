"""
LocalTrainer — Single-GPU PyTorch training in a subprocess.

Training runs in a subprocess via asyncio.create_subprocess_exec.
Progress is reported via JSON lines to a progress file.
A monitoring coroutine tails the file and emits events.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from ..core.event_bus import EventType, get_event_bus
from .models import TrainingRun, TrainingStatus

logger = logging.getLogger(__name__)


class LocalTrainer:
    """Manages local training runs as subprocesses.

    Parameters
    ----------
    run_dir : Path
        Directory for training outputs (weights, metrics, progress).
    """

    def __init__(self, run_dir: Path):
        self._run_dir = run_dir
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._process: asyncio.subprocess.Process | None = None
        self._monitor_task: asyncio.Task | None = None

    @property
    def progress_file(self) -> Path:
        return self._run_dir / "progress.jsonl"

    @property
    def weights_dir(self) -> Path:
        d = self._run_dir / "weights"
        d.mkdir(exist_ok=True)
        return d

    @property
    def metrics_file(self) -> Path:
        return self._run_dir / "metrics.json"

    async def start_training(
        self,
        run: TrainingRun,
        data_root: Path,
        labels_file: Path,
    ) -> TrainingRun:
        """Launch training as a subprocess.

        Parameters
        ----------
        run : TrainingRun
            Training run configuration.
        data_root : Path
            Root directory containing projection images.
        labels_file : Path
            JSON file mapping image paths to stage labels.

        Returns
        -------
        TrainingRun
            Updated run with status=training.
        """
        run.status = TrainingStatus.TRAINING.value
        run.started_at = datetime.now().isoformat()
        run.total_epochs = run.training_config.epochs if run.training_config else 50

        # Build the training script arguments
        config = {
            "run_id": run.id,
            "model_config": run.model_config.to_dict() if run.model_config else {},
            "training_config": run.training_config.to_dict() if run.training_config else {},
            "data_root": str(data_root),
            "labels_file": str(labels_file),
            "output_dir": str(self._run_dir),
            "progress_file": str(self.progress_file),
            "weights_dir": str(self.weights_dir),
            "metrics_file": str(self.metrics_file),
        }

        config_file = self._run_dir / "train_config.json"
        config_file.write_text(json.dumps(config, indent=2))

        # Launch subprocess
        train_script = Path(__file__).parent / "_train_worker.py"
        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(train_script),
            str(config_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._run_dir),
        )

        # Start progress monitor
        self._monitor_task = asyncio.create_task(self._monitor_progress(run.id, run.pipeline_id))

        logger.info(f"Training started: run={run.id}, pid={self._process.pid}")
        return run

    async def _monitor_progress(self, run_id: str, pipeline_id: str):
        """Tail progress.jsonl and emit events."""
        bus = get_event_bus()
        last_pos = 0

        while self._process and self._process.returncode is None:
            await asyncio.sleep(2.0)
            try:
                if self.progress_file.exists():
                    with open(self.progress_file) as f:
                        f.seek(last_pos)
                        for line in f:
                            line = line.strip()
                            if line:
                                data = json.loads(line)
                                data["run_id"] = run_id
                                data["pipeline_id"] = pipeline_id
                                bus.publish(
                                    EventType.ML_TRAINING_PROGRESS,
                                    data,
                                    source="ml_trainer",
                                )
                        last_pos = f.tell()
            except Exception as e:
                logger.debug(f"Progress monitor error: {e}")

        # Process completed — read final metrics
        if self._process:
            returncode = self._process.returncode
            if returncode == 0:
                bus.publish(
                    EventType.ML_TRAINING_COMPLETED,
                    {"run_id": run_id, "pipeline_id": pipeline_id},
                    source="ml_trainer",
                )
            else:
                stderr = ""
                if self._process.stderr:
                    try:
                        stderr_bytes = await self._process.stderr.read()
                        stderr = stderr_bytes.decode(errors="replace")[-500:]
                    except Exception:
                        pass
                bus.publish(
                    EventType.ML_TRAINING_FAILED,
                    {"run_id": run_id, "pipeline_id": pipeline_id, "error": stderr},
                    source="ml_trainer",
                )

    async def wait_for_completion(self) -> int:
        """Wait for the training subprocess to finish."""
        if self._process:
            return await self._process.wait()
        return -1

    async def cancel(self):
        """Cancel the running training."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except asyncio.TimeoutError:
                self._process.kill()
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

    def get_latest_progress(self) -> dict | None:
        """Read the last line from progress.jsonl."""
        if not self.progress_file.exists():
            return None
        try:
            lines = self.progress_file.read_text().strip().split("\n")
            if lines:
                return json.loads(lines[-1])
        except Exception:
            pass
        return None
