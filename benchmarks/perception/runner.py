"""
Perception Benchmark Runner.

Runs perception engine against offline test data and collects results.
"""

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .ground_truth import GroundTruth
from .metrics import PerceptionMetrics, compute_metrics
from .testset import OfflineTestset

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    # Model settings
    model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.0
    max_tokens: int = 8000

    # Engine settings
    enable_tools: bool = True
    enable_view_embryo: bool = True
    enable_view_reference: bool = True
    enable_view_previous: bool = True
    enable_verification: bool = True  # Multi-phase verification with subagents

    # Test settings
    start_timepoint: int = 0
    max_timepoints_per_embryo: int | None = None
    embryo_ids: list[str] | None = None  # None = all

    # Ablation toggles
    include_temporal_context: bool = True
    include_previous_observations: bool = True

    # Custom system prompt override
    system_prompt_override: str | None = None

    # Metadata
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_tools": self.enable_tools,
            "enable_view_embryo": self.enable_view_embryo,
            "enable_view_reference": self.enable_view_reference,
            "enable_view_previous": self.enable_view_previous,
            "enable_verification": self.enable_verification,
            "include_temporal_context": self.include_temporal_context,
            "include_previous_observations": self.include_previous_observations,
            "start_timepoint": self.start_timepoint,
            "max_timepoints_per_embryo": self.max_timepoints_per_embryo,
            "embryo_ids": self.embryo_ids,
            "system_prompt_override": self.system_prompt_override,
            "description": self.description,
        }


@dataclass
class PredictionResult:
    """Result of a single perception prediction."""

    timepoint: int
    predicted_stage: str
    ground_truth_stage: str | None
    confidence: float
    is_transitional: bool
    transition_between: list[str] | None
    reasoning: str
    reasoning_trace: dict[str, Any] | None  # Serialized ReasoningTrace
    tool_calls: int
    tools_used: list[str]

    # Multi-phase verification fields
    verification_triggered: bool = False
    phase_count: int = 1
    verification_result: dict[str, Any] | None = None
    candidate_stages: list[dict[str, Any]] | None = None

    @property
    def is_correct(self) -> bool:
        """Check if prediction matches ground truth exactly."""
        return self.predicted_stage == self.ground_truth_stage

    @property
    def is_adjacent_correct(self) -> bool:
        """Check if prediction is within 1 stage of ground truth."""
        from gently.harness.perception.stages import DevelopmentalStage

        if self.ground_truth_stage is None:
            return False

        try:
            pred_order = DevelopmentalStage.get_order(self.predicted_stage)
            gt_order = DevelopmentalStage.get_order(self.ground_truth_stage)
            return abs(pred_order - gt_order) <= 1
        except ValueError:
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timepoint": self.timepoint,
            "predicted_stage": self.predicted_stage,
            "ground_truth_stage": self.ground_truth_stage,
            "confidence": self.confidence,
            "is_transitional": self.is_transitional,
            "transition_between": self.transition_between,
            "reasoning": self.reasoning,
            "reasoning_trace": self.reasoning_trace,
            "tool_calls": self.tool_calls,
            "tools_used": self.tools_used,
            "is_correct": self.is_correct,
            "is_adjacent_correct": self.is_adjacent_correct,
            "verification_triggered": self.verification_triggered,
            "phase_count": self.phase_count,
            "verification_result": self.verification_result,
            "candidate_stages": self.candidate_stages,
        }


@dataclass
class EmbryoResult:
    """Results for a single embryo run."""

    embryo_id: str
    predictions: list[PredictionResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None

    @property
    def accuracy(self) -> float:
        """Exact match accuracy."""
        if not self.predictions:
            return 0.0
        correct = sum(1 for p in self.predictions if p.is_correct)
        return correct / len(self.predictions)

    @property
    def adjacent_accuracy(self) -> float:
        """Within-1-stage accuracy."""
        if not self.predictions:
            return 0.0
        correct = sum(1 for p in self.predictions if p.is_adjacent_correct)
        return correct / len(self.predictions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "embryo_id": self.embryo_id,
            "predictions": [p.to_dict() for p in self.predictions],
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "accuracy": self.accuracy,
            "adjacent_accuracy": self.adjacent_accuracy,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    config: BenchmarkConfig
    embryo_results: list[EmbryoResult] = field(default_factory=list)
    metrics: PerceptionMetrics | None = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    session_id: str | None = None

    @property
    def total_predictions(self) -> int:
        return sum(len(r.predictions) for r in self.embryo_results)

    @property
    def overall_accuracy(self) -> float:
        all_preds = [p for r in self.embryo_results for p in r.predictions]
        if not all_preds:
            return 0.0
        return sum(1 for p in all_preds if p.is_correct) / len(all_preds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "embryo_results": [r.to_dict() for r in self.embryo_results],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "session_id": self.session_id,
            "total_predictions": self.total_predictions,
            "overall_accuracy": self.overall_accuracy,
        }

    def save_json(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class PerceptionBenchmark:
    """
    Benchmark runner for perception engine.

    Runs the engine against a testset and collects detailed results.
    """

    def __init__(
        self,
        testset: OfflineTestset,
        config: BenchmarkConfig,
        engine: Any | None = None,  # PerceptionEngine
    ):
        """
        Parameters
        ----------
        testset : OfflineTestset
            Test data with ground truth
        config : BenchmarkConfig
            Configuration for this benchmark run
        engine : PerceptionEngine, optional
            Pre-configured engine. If None, creates one based on config.
        """
        self.testset = testset
        self.config = config
        self._engine = engine

    async def _get_engine(self):
        """Get or create perception engine."""
        if self._engine is not None:
            return self._engine

        # Lazy import to avoid circular dependencies
        import anthropic
        from gently.harness.perception.engine import PerceptionEngine

        client = anthropic.Anthropic()

        # Find examples path (ExampleStore expects parent of 'stages/' folder)
        examples_path = Path("gently/examples")
        if not examples_path.exists():
            # Try alternate path
            examples_path = Path("gently/agent/perception/examples")
            if not examples_path.exists():
                examples_path = None

        engine = PerceptionEngine(
            claude_client=client,
            examples_path=examples_path,
            enable_verification=self.config.enable_verification,
            include_temporal_context=self.config.include_temporal_context,
            include_previous_observations=self.config.include_previous_observations,
        )

        self._engine = engine
        return engine

    async def run_embryo(self, embryo_id: str) -> EmbryoResult:
        """
        Run perception on a single embryo sequence.

        Parameters
        ----------
        embryo_id : str
            Embryo to run

        Returns
        -------
        EmbryoResult
            Results for this embryo
        """
        from gently.harness.perception.session import PerceptionSession

        start_time = datetime.now()
        result = EmbryoResult(embryo_id=embryo_id)

        try:
            engine = await self._get_engine()
            session = PerceptionSession(embryo_id)

            # Determine timepoint range
            end_tp = None
            if self.config.max_timepoints_per_embryo:
                end_tp = self.config.max_timepoints_per_embryo

            for test_case in self.testset.iter_embryo(
                embryo_id,
                start_timepoint=self.config.start_timepoint,
                end_timepoint=end_tp,
            ):
                logger.info(
                    f"[{embryo_id}] Processing T{test_case.timepoint} "
                    f"(GT: {test_case.ground_truth_stage})"
                )

                # Run perception
                perception_result = await engine.perceive(
                    image_b64=test_case.image_b64,
                    session=session,
                    timepoint=test_case.timepoint,
                    volume=test_case.volume,
                )

                # Record result
                trace_dict = None
                if perception_result.reasoning_trace:
                    trace_dict = perception_result.reasoning_trace.to_dict()

                # Serialize verification result if present
                verification_dict = None
                if perception_result.verification_result:
                    verification_dict = perception_result.verification_result.to_dict()

                # Serialize candidate stages if present
                candidates_list = None
                if perception_result.candidate_stages:
                    candidates_list = [c.to_dict() for c in perception_result.candidate_stages]

                pred = PredictionResult(
                    timepoint=test_case.timepoint,
                    predicted_stage=perception_result.stage,
                    ground_truth_stage=test_case.ground_truth_stage,
                    confidence=perception_result.confidence,
                    is_transitional=perception_result.is_transitional,
                    transition_between=perception_result.transition_between,
                    reasoning=perception_result.reasoning,
                    reasoning_trace=trace_dict,
                    tool_calls=trace_dict.get("total_tool_calls", 0) if trace_dict else 0,
                    tools_used=trace_dict.get("tools_used", []) if trace_dict else [],
                    verification_triggered=perception_result.verification_triggered,
                    phase_count=perception_result.phase_count,
                    verification_result=verification_dict,
                    candidate_stages=candidates_list,
                )
                result.predictions.append(pred)

                # Add observation to session for temporal context
                session.add_observation(
                    timepoint=test_case.timepoint,
                    stage=perception_result.stage,
                    is_hatching=perception_result.is_hatching,
                    confidence=perception_result.confidence,
                    reasoning=perception_result.reasoning,
                    is_transitional=perception_result.is_transitional,
                    transition_between=perception_result.transition_between,
                    timestamp=test_case.acquired_at,
                )

                logger.info(
                    f"[{embryo_id}] T{test_case.timepoint}: "
                    f"pred={perception_result.stage}, GT={test_case.ground_truth_stage}, "
                    f"{'CORRECT' if pred.is_correct else 'WRONG'}"
                )

        except Exception as e:
            logger.error(f"[{embryo_id}] Error: {e}")
            result.error = str(e)

        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        return result

    async def run_all(self) -> BenchmarkReport:
        """
        Run benchmark on all embryos in testset.

        Returns
        -------
        BenchmarkReport
            Complete benchmark report with metrics
        """
        report = BenchmarkReport(
            config=self.config,
            session_id=self.testset.ground_truth.session_id,
        )

        # Determine which embryos to run
        embryo_ids = self.config.embryo_ids
        if embryo_ids is None:
            embryo_ids = self.testset.embryo_ids

        logger.info(f"Running benchmark on {len(embryo_ids)} embryos")

        for embryo_id in embryo_ids:
            logger.info(f"Starting embryo: {embryo_id}")
            embryo_result = await self.run_embryo(embryo_id)
            report.embryo_results.append(embryo_result)
            logger.info(
                f"Completed {embryo_id}: accuracy={embryo_result.accuracy:.1%}, "
                f"adjacent={embryo_result.adjacent_accuracy:.1%}"
            )

        report.completed_at = datetime.now()

        # Compute metrics
        report.metrics = compute_metrics(report)

        logger.info(
            f"Benchmark complete: {report.total_predictions} predictions, "
            f"overall accuracy={report.overall_accuracy:.1%}"
        )

        return report


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run perception benchmark")
    parser.add_argument(
        "--session",
        required=True,
        help="Session ID or path to session directory",
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--output",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--embryo",
        action="append",
        help="Specific embryo(s) to run (can specify multiple)",
    )
    parser.add_argument(
        "--start-timepoint",
        type=int,
        default=0,
        help="First timepoint index to process (skip earlier frames)",
    )
    parser.add_argument(
        "--max-timepoints",
        type=int,
        help="End timepoint index (exclusive). With --start-timepoint, processes [start, max).",
    )
    parser.add_argument(
        "--no-temporal-context",
        action="store_true",
        help="Ablation: omit the TEMPORAL CONTEXT block from the prompt",
    )
    parser.add_argument(
        "--no-previous-observations",
        action="store_true",
        help="Ablation: omit the PREVIOUS OBSERVATIONS block from the prompt",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Description for this benchmark run",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # The perception engine reads stage definitions etc. from the active
    # organism module, which is normally loaded by launch_gently.py.
    from gently.organisms import load_organism

    load_organism("celegans")

    # Find session path
    session_path = Path(args.session)
    if not session_path.exists():
        # Try as session ID
        base_paths = [
            Path("Z:/embryo_data"),
            Path("D:/embryo_data"),
            Path.home() / "embryo_data",
        ]
        for base in base_paths:
            candidate = base / args.session
            if candidate.exists():
                session_path = candidate
                break

    if not session_path.exists():
        print(f"Session not found: {args.session}")
        sys.exit(1)

    # Load ground truth
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"Ground truth not found: {args.ground_truth}")
        sys.exit(1)

    ground_truth = GroundTruth.from_json(gt_path)
    logger.info(f"Loaded ground truth for {len(ground_truth.embryo_ids)} embryos")

    # Create testset
    testset = OfflineTestset(
        session_path=session_path,
        ground_truth=ground_truth,
        load_volumes=True,
    )
    logger.info(f"Testset has {len(testset.embryo_ids)} embryos with both data and ground truth")

    # Create config
    config = BenchmarkConfig(
        embryo_ids=args.embryo,
        start_timepoint=args.start_timepoint,
        max_timepoints_per_embryo=args.max_timepoints,
        include_temporal_context=not args.no_temporal_context,
        include_previous_observations=not args.no_previous_observations,
        description=args.description,
    )

    # Run benchmark
    benchmark = PerceptionBenchmark(testset=testset, config=config)
    report = await benchmark.run_all()

    # Save results
    if args.output:
        output_path = Path(args.output)
        report.save_json(output_path)
        logger.info(f"Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total predictions: {report.total_predictions}")
    print(f"Overall accuracy: {report.overall_accuracy:.1%}")
    if report.metrics:
        print(f"Adjacent accuracy: {report.metrics.adjacent_accuracy:.1%}")
        print(f"Mean confidence: {report.metrics.mean_confidence:.2f}")
    print()
    for er in report.embryo_results:
        print(f"  {er.embryo_id}: accuracy={er.accuracy:.1%}, adjacent={er.adjacent_accuracy:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
