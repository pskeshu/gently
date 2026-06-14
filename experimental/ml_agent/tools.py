"""
ML-specific tools for the ML subagent.

These are registered on the subagent's tool registry and also
available to the agent in plan mode.
"""

import json
import logging
from datetime import datetime

from gently.harness.tools.registry import ToolCategory, ToolParameter

logger = logging.getLogger(__name__)


def register_ml_tools(
    registry, context_store=None, gently_store=None, verse_map=None, peer_client=None
):
    """Register ML tools on a tool registry.

    Parameters
    ----------
    registry : ToolRegistry
        Registry to register tools on.
    context_store : ContextStore, optional
    gently_store : GentlyStore, optional
    verse_map : VerseMap, optional
    peer_client : PeerClient, optional
    """

    @registry.register(
        name="inventory_datasets",
        description=(
            "Survey all datasets across the mesh (local + remote peers). "
            "Returns a NetworkDataInventory with session summaries, "
            "embryo counts, volume counts, and annotation status."
        ),
        category=ToolCategory.ML,
        parameters=[
            ToolParameter(
                name="include_remote",
                type="boolean",
                description="Include remote peer datasets (default true)",
                required=False,
                default=True,
            ),
        ],
    )
    async def inventory_datasets(include_remote: bool = True, **kwargs) -> str:
        from gently.data_reasoning.assessment import DataAssessmentEngine

        engine = DataAssessmentEngine(
            gently_store=gently_store,
            peer_client=peer_client,
            verse_map=verse_map,
        )
        inventory = await engine.build_inventory(include_remote=include_remote)
        return json.dumps(inventory.to_dict(), indent=2)

    @registry.register(
        name="check_annotation_coverage",
        description=(
            "Analyze annotation coverage across datasets. Reports stage distribution, "
            "class imbalance, gaps, and recommendations for training readiness."
        ),
        category=ToolCategory.ML,
        parameters=[
            ToolParameter(
                name="session_ids",
                type="string",
                description="Comma-separated session IDs to check (empty = all)",
                required=False,
            ),
        ],
    )
    async def check_annotation_coverage(session_ids: str = "", **kwargs) -> str:
        from gently.data_reasoning.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(gently_store=gently_store)
        sid_list = [s.strip() for s in session_ids.split(",") if s.strip()] or None
        report = analyzer.analyze(session_ids=sid_list)
        return json.dumps(report.to_dict(), indent=2)

    @registry.register(
        name="select_architecture",
        description=(
            "Get suitable model architectures for the given dataset size and hardware. "
            "Returns a ranked list with reasoning hints."
        ),
        category=ToolCategory.ML,
        parameters=[
            ToolParameter(
                name="dataset_size",
                type="integer",
                description="Number of annotated samples",
                required=True,
            ),
            ToolParameter(
                name="vram_gb",
                type="number",
                description="Available GPU VRAM in GB",
                required=True,
            ),
        ],
    )
    async def select_architecture(dataset_size: int, vram_gb: float, **kwargs) -> str:
        from gently.ml.architectures import get_suitable_architectures

        results = get_suitable_architectures(dataset_size, vram_gb)
        return json.dumps(results, indent=2)

    @registry.register(
        name="configure_training",
        description=(
            "Create an ML pipeline with model config and training hyperparameters. "
            "Stores the configuration in the context store."
        ),
        category=ToolCategory.ML,
        parameters=[
            ToolParameter(
                name="campaign_id",
                type="string",
                description="Campaign this pipeline belongs to",
                required=True,
            ),
            ToolParameter(name="name", type="string", description="Pipeline name", required=True),
            ToolParameter(
                name="architecture",
                type="string",
                description="Model architecture ID",
                required=True,
            ),
            ToolParameter(
                name="num_classes",
                type="integer",
                description="Number of output classes",
                required=True,
            ),
            ToolParameter(
                name="batch_size",
                type="integer",
                description="Training batch size",
                required=False,
                default=32,
            ),
            ToolParameter(
                name="epochs",
                type="integer",
                description="Number of training epochs",
                required=False,
                default=50,
            ),
            ToolParameter(
                name="learning_rate",
                type="number",
                description="Learning rate",
                required=False,
                default=1e-4,
            ),
        ],
    )
    async def configure_training(
        campaign_id: str,
        name: str,
        architecture: str,
        num_classes: int,
        batch_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        **kwargs,
    ) -> str:
        if context_store is None:
            return "Error: No context store available"
        pipeline = context_store.create_ml_pipeline(
            campaign_id=campaign_id,
            name=name,
            model_config={
                "architecture": architecture,
                "num_classes": num_classes,
                "pretrained": True,
                "input_channels": 1,
            },
            training_config={
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "mixed_precision": True,
            },
        )
        return json.dumps(pipeline, indent=2)

    @registry.register(
        name="start_local_training",
        description=(
            "Start a local training run for a configured pipeline. "
            "Runs in a subprocess, reports progress via events."
        ),
        category=ToolCategory.ML,
        parameters=[
            ToolParameter(
                name="pipeline_id",
                type="string",
                description="Pipeline ID to train",
                required=True,
            ),
        ],
    )
    async def start_local_training(pipeline_id: str, **kwargs) -> str:
        if context_store is None:
            return "Error: No context store available"

        pipeline = context_store.get_ml_pipeline(pipeline_id)
        if pipeline is None:
            return f"Error: Pipeline {pipeline_id} not found"

        # Create a training run
        run_data = context_store.create_training_run(
            pipeline_id=pipeline_id,
            model_config=pipeline.get("model_config"),
            training_config=pipeline.get("training_config"),
            data_split=pipeline.get("data_split"),
        )

        # Update pipeline status
        context_store.update_ml_pipeline(pipeline_id, status="training")

        # Build labels from store
        if gently_store is None:
            return "Error: No data store available"

        from gently.ml.data_loader import build_labels_from_store

        labels = build_labels_from_store(gently_store)

        if not labels.get("samples"):
            return "Error: No labeled data found in store"

        # Write labels file
        from gently.settings import settings

        run_dir = settings.storage.base_path / "ml_runs" / run_data["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        labels_file = run_dir / "labels.json"
        labels_file.write_text(json.dumps(labels, indent=2))

        # Start trainer
        from gently.ml.models import ModelConfig, TrainingConfig, TrainingRun
        from gently.ml.trainer import LocalTrainer

        trainer = LocalTrainer(run_dir)
        run = TrainingRun(
            id=run_data["id"],
            pipeline_id=pipeline_id,
            model_config=ModelConfig.from_dict(run_data["model_config"] or {}),
            training_config=TrainingConfig.from_dict(run_data["training_config"] or {}),
        )
        await trainer.start_training(
            run,
            data_root=settings.storage.base_path,
            labels_file=labels_file,
        )

        context_store.update_training_run(
            run_data["id"],
            status="training",
            started_at=datetime.now().isoformat(),
        )

        return json.dumps(
            {
                "status": "training_started",
                "run_id": run_data["id"],
                "pipeline_id": pipeline_id,
            }
        )

    @registry.register(
        name="get_ml_status",
        description="Get the status of ML pipelines and training runs.",
        category=ToolCategory.ML,
        parameters=[
            ToolParameter(
                name="pipeline_id",
                type="string",
                description="Pipeline ID (empty = list all)",
                required=False,
            ),
        ],
    )
    async def get_ml_status(pipeline_id: str = "", **kwargs) -> str:
        if context_store is None:
            return "Error: No context store available"
        if pipeline_id:
            pipeline = context_store.get_ml_pipeline(pipeline_id)
            runs = context_store.list_training_runs(pipeline_id)
            return json.dumps({"pipeline": pipeline, "runs": runs}, indent=2)
        else:
            pipelines = context_store.list_ml_pipelines()
            return json.dumps({"pipelines": pipelines}, indent=2)

    @registry.register(
        name="plan_annotation_campaign",
        description=(
            "Create plan items to fill annotation gaps for ML training. "
            "Analyzes coverage and creates tasks for underrepresented stages."
        ),
        category=ToolCategory.ML,
        parameters=[
            ToolParameter(
                name="campaign_id",
                type="string",
                description="Campaign to add items to",
                required=True,
            ),
            ToolParameter(
                name="target_per_stage",
                type="integer",
                description="Target annotations per stage (default 50)",
                required=False,
                default=50,
            ),
        ],
    )
    async def plan_annotation_campaign(
        campaign_id: str,
        target_per_stage: int = 50,
        **kwargs,
    ) -> str:
        from gently.data_reasoning.coverage import CoverageAnalyzer
        from gently.data_reasoning.gap_planner import GapPlanner

        analyzer = CoverageAnalyzer(gently_store=gently_store)
        report = analyzer.analyze()

        planner = GapPlanner(context_store=context_store)
        created_ids = planner.plan_annotation_campaign(
            campaign_id=campaign_id,
            coverage_report=report,
            target_per_stage=target_per_stage,
        )

        return json.dumps(
            {
                "created_plan_items": len(created_ids),
                "item_ids": created_ids,
                "coverage_before": report.to_dict(),
            },
            indent=2,
        )
