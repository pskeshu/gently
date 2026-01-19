"""
TracePersister - Hybrid trace persistence for live experiments.

Stores reasoning traces to BOTH:
1. JSON files on disk (source of truth for full trace data)
2. SQLite database (indexed for querying)

Usage:
    persister = TracePersister(
        dataset=dataset,
        session_id="abc123",
        trace_type='perception',
    )

    # Store a trace (writes file + DB record)
    prediction_id = await persister.store_trace(
        embryo_id="embryo_1",
        timepoint=42,
        result=perception_result,
    )
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .embryo_dataset import EmbryoDataset

logger = logging.getLogger(__name__)

# Default base path for trace files
TRACE_BASE_PATH = Path("D:/Gently/traces")


class TracePersister:
    """
    Hybrid storage for reasoning traces during live experiments.

    Writes traces to both:
    - JSON files: `{base_path}/{session_id}/{embryo_id}_T{timepoint:04d}_{trace_type}_run{run_id:04d}.json`
    - Database: Indexed in perception_runs/predictions/reasoning_traces tables

    The file is the source of truth; the database can be rebuilt via aggregation.

    Parameters
    ----------
    dataset : EmbryoDataset
        Dataset instance for database operations
    session_id : str
        Session identifier (links traces to experiment)
    trace_type : str
        Type of traces ('perception', 'hatching_detector', etc.)
    source : str
        Origin of traces ('live', 'benchmark', 'replay')
    perception_method : str
        Method identifier for the run
    model_name : str, optional
        Model used for perception
    base_path : Path, optional
        Base directory for trace files (default: D:/Gently/traces)
    """

    def __init__(
        self,
        dataset: "EmbryoDataset",
        session_id: str,
        trace_type: str = 'perception',
        source: str = 'live',
        perception_method: str = 'vlm_v3',
        model_name: Optional[str] = None,
        base_path: Optional[Path] = None,
    ):
        self.dataset = dataset
        self.session_id = session_id
        self.trace_type = trace_type
        self.source = source
        self.perception_method = perception_method
        self.model_name = model_name

        # Lazy run initialization
        self._run_id: Optional[int] = None

        # Set up trace directory
        self._base_path = base_path or TRACE_BASE_PATH
        self._trace_dir = self._base_path / session_id
        self._trace_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"TracePersister initialized: session={session_id}, "
            f"type={trace_type}, dir={self._trace_dir}"
        )

    @property
    def run_id(self) -> int:
        """Get run ID, creating run on first access (lazy initialization)."""
        if self._run_id is None:
            self._run_id = self._create_run()
        return self._run_id

    def _create_run(self) -> int:
        """Create a perception run record for this session."""
        run_name = f"{self.source}_{self.session_id}_{self.trace_type}"
        run_id = self.dataset.create_perception_run(
            name=run_name,
            perception_method=self.perception_method,
            model_name=self.model_name,
            trace_type=self.trace_type,
            source=self.source,
            session_id=self.session_id,
            description=f"Live trace persistence for {self.session_id}",
        )
        logger.info(f"Created perception run {run_id}: {run_name}")
        return run_id

    async def store_trace(
        self,
        embryo_id: str,
        timepoint: int,
        result: Any,  # PerceptionResult
        image_uid: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
    ) -> int:
        """
        Store a trace to both file and database.

        Parameters
        ----------
        embryo_id : str
            Embryo identifier
        timepoint : int
            Timepoint number
        result : PerceptionResult
            Perception result containing stage, confidence, and traces
        image_uid : str, optional
            UID of the source image
        execution_time_ms : float, optional
            Time taken for perception

        Returns
        -------
        int
            Prediction ID in database
        """
        run_id = self.run_id  # Ensures run is created

        # Extract trace dict
        trace_dict = None
        if hasattr(result, 'reasoning_trace') and result.reasoning_trace:
            trace_dict = result.reasoning_trace.to_dict()
        elif hasattr(result, 'multi_phase_trace') and result.multi_phase_trace:
            trace_dict = result.multi_phase_trace.to_dict()

        # Build full trace data for file
        timestamp = datetime.now()
        trace_data = self._build_trace_data(
            embryo_id=embryo_id,
            timepoint=timepoint,
            result=result,
            trace_dict=trace_dict,
            timestamp=timestamp,
        )

        # 1. Write JSON file (source of truth)
        file_path = self._write_trace_file(
            embryo_id=embryo_id,
            timepoint=timepoint,
            run_id=run_id,
            trace_data=trace_data,
        )

        # 2. Store in database for querying
        observed_features_dict = None
        if hasattr(result, 'observed_features') and result.observed_features:
            observed_features_dict = {
                'shape': result.observed_features.shape,
                'curvature': result.observed_features.curvature,
                'shell_status': result.observed_features.shell_status,
                'body_segments': getattr(result.observed_features, 'body_segments_visible', None),
                'emergence': result.observed_features.emergence,
                'movement': getattr(result.observed_features, 'movement', None),
                'texture': getattr(result.observed_features, 'texture', None),
            }

        prediction_id = self.dataset.store_prediction(
            run_id=run_id,
            embryo_id=embryo_id,
            timepoint=timepoint,
            predicted_stage=result.stage,
            confidence=result.confidence,
            reasoning=result.reasoning,
            image_uid=image_uid,
            session_id=self.session_id,
            is_transitional=getattr(result, 'is_transitional', False),
            observed_features=observed_features_dict,
            reasoning_trace=trace_dict,
            execution_time_ms=execution_time_ms,
            trace_file_path=str(file_path),
        )

        logger.debug(
            f"Stored trace: {embryo_id} T{timepoint} -> prediction {prediction_id}, "
            f"file: {file_path.name}"
        )

        return prediction_id

    def _build_trace_data(
        self,
        embryo_id: str,
        timepoint: int,
        result: Any,
        trace_dict: Optional[Dict],
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """Build the full trace data dict for file storage."""
        data = {
            # Identifiers
            "session_id": self.session_id,
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "trace_type": self.trace_type,
            "run_id": self.run_id,

            # Results
            "predicted_stage": result.stage,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "is_hatching": getattr(result, 'is_hatching', False),
            "is_transitional": getattr(result, 'is_transitional', False),
            "transition_between": getattr(result, 'transition_between', None),

            # Metadata
            "timestamp": timestamp.isoformat(),
            "perception_method": self.perception_method,
            "model_name": self.model_name,

            # Full reasoning trace
            "reasoning_trace": trace_dict,
        }

        # Add observed features if available
        if hasattr(result, 'observed_features') and result.observed_features:
            data["observed_features"] = {
                'shape': result.observed_features.shape,
                'curvature': result.observed_features.curvature,
                'shell_status': result.observed_features.shell_status,
                'body_segments': getattr(result.observed_features, 'body_segments_visible', None),
                'emergence': result.observed_features.emergence,
            }

        # Add contrastive reasoning if available
        if hasattr(result, 'contrastive_reasoning') and result.contrastive_reasoning:
            data["contrastive_reasoning"] = {
                'why_not_previous': result.contrastive_reasoning.why_not_previous_stage,
                'why_not_next': result.contrastive_reasoning.why_not_next_stage,
            }

        # Add verification info if available
        if hasattr(result, 'verification_triggered') and result.verification_triggered:
            data["verification"] = {
                'triggered': True,
                'result': result.verification_result.to_dict() if result.verification_result else None,
            }

        # Add multi-phase trace if available (separate from single-phase)
        if hasattr(result, 'multi_phase_trace') and result.multi_phase_trace:
            data["multi_phase_trace"] = result.multi_phase_trace.to_dict()

        return data

    def _write_trace_file(
        self,
        embryo_id: str,
        timepoint: int,
        run_id: int,
        trace_data: Dict[str, Any],
    ) -> Path:
        """Write trace data to JSON file."""
        # Filename format: {embryo_id}_T{timepoint:04d}_{trace_type}_run{run_id:04d}.json
        filename = f"{embryo_id}_T{timepoint:04d}_{self.trace_type}_run{run_id:04d}.json"
        file_path = self._trace_dir / filename

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)

        return file_path

    def complete_run(self, status: str = "completed", error_message: Optional[str] = None):
        """Mark the perception run as completed."""
        if self._run_id is not None:
            self.dataset.complete_perception_run(
                run_id=self._run_id,
                status=status,
                error_message=error_message,
            )
            logger.info(f"Completed perception run {self._run_id}: {status}")

    @property
    def trace_dir(self) -> Path:
        """Get the trace directory path."""
        return self._trace_dir

    def get_trace_count(self) -> int:
        """Get the number of traces stored in this run."""
        if self._run_id is None:
            return 0
        return len(list(self._trace_dir.glob(f"*_{self.trace_type}_run{self._run_id:04d}.json")))
