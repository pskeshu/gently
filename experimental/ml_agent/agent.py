"""
MLSubagent — Autonomous background agent for ML training tasks.

Spawned by the agent via asyncio.create_task(). Runs its own tool
execution loop, communicating back via the event bus.
"""

import json
import logging

from gently.core.event_bus import EventType, get_event_bus
from gently.harness.tools.registry import ToolRegistry

from .tools import register_ml_tools

logger = logging.getLogger(__name__)


class MLSubagent:
    """Autonomous ML training agent.

    Parameters
    ----------
    context_store : ContextStore
    gently_store : GentlyStore, optional
    verse_map : VerseMap, optional
    peer_client : PeerClient, optional
    """

    def __init__(
        self,
        context_store,
        gently_store=None,
        verse_map=None,
        peer_client=None,
    ):
        self._context_store = context_store
        self._gently_store = gently_store
        self._verse_map = verse_map
        self._peer_client = peer_client
        self._registry = ToolRegistry()
        self._running = False
        self._task: str | None = None
        self._campaign_id: str | None = None

        # Register ML tools on our private registry
        register_ml_tools(
            self._registry,
            context_store=context_store,
            gently_store=gently_store,
            verse_map=verse_map,
            peer_client=peer_client,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    async def run(self, task: str, campaign_id: str = ""):
        """Execute an ML task autonomously.

        This is the main entry point, called via asyncio.create_task().

        Parameters
        ----------
        task : str
            The ML task description (e.g., "Train an embryo classifier").
        campaign_id : str
            Campaign to associate pipelines with.
        """
        self._running = True
        self._task = task
        self._campaign_id = campaign_id
        bus = get_event_bus()

        try:
            bus.publish(
                EventType.ML_SUBAGENT_STATUS,
                {"status": "started", "task": task},
                source="ml_subagent",
            )

            # Step 1: Inventory datasets
            bus.publish(
                EventType.ML_SUBAGENT_STATUS,
                {"status": "assessing_data", "detail": "Inventorying datasets..."},
                source="ml_subagent",
            )
            inventory_result = await self._registry.execute(
                "inventory_datasets", {"include_remote": True}
            )
            inventory = json.loads(inventory_result)

            # Step 2: Check coverage
            bus.publish(
                EventType.ML_SUBAGENT_STATUS,
                {
                    "status": "checking_coverage",
                    "detail": "Analyzing annotation coverage...",
                },
                source="ml_subagent",
            )
            coverage_result = await self._registry.execute("check_annotation_coverage", {})
            coverage = json.loads(coverage_result)

            inventory.get("total_annotated", 0)
            total_gt = inventory.get("total_ground_truth", 0)

            # Step 3: Check if we have enough data
            if total_gt < 50:
                bus.publish(
                    EventType.ML_SUBAGENT_STATUS,
                    {
                        "status": "insufficient_data",
                        "detail": (
                            f"Only {total_gt} ground truth annotations found. "
                            f"Need at least 50 for minimal training. "
                            f"Gaps: {coverage.get('gaps', [])}"
                        ),
                        "coverage": coverage,
                    },
                    source="ml_subagent",
                )
                self._running = False
                return

            # Step 4: Select architecture
            bus.publish(
                EventType.ML_SUBAGENT_STATUS,
                {"status": "selecting_architecture"},
                source="ml_subagent",
            )

            # Determine VRAM
            vram = 24.0  # default A5000
            inventory.get("local_sessions", [])

            arch_result = await self._registry.execute(
                "select_architecture",
                {"dataset_size": total_gt, "vram_gb": vram},
            )
            architectures = json.loads(arch_result)
            if not architectures:
                bus.publish(
                    EventType.ML_SUBAGENT_STATUS,
                    {"status": "error", "detail": "No suitable architectures found"},
                    source="ml_subagent",
                )
                self._running = False
                return

            best_arch = architectures[0]
            num_classes = len(coverage.get("stage_counts", {})) or 8

            # Step 5: Configure and start training
            bus.publish(
                EventType.ML_SUBAGENT_STATUS,
                {
                    "status": "configuring",
                    "architecture": best_arch["architecture_id"],
                    "reason": best_arch.get("reason", ""),
                },
                source="ml_subagent",
            )

            config_result = await self._registry.execute(
                "configure_training",
                {
                    "campaign_id": campaign_id,
                    "name": f"Embryo classifier ({best_arch['name']})",
                    "architecture": best_arch["architecture_id"],
                    "num_classes": num_classes,
                },
            )
            pipeline = json.loads(config_result)

            # Step 6: Start training
            bus.publish(
                EventType.ML_SUBAGENT_STATUS,
                {"status": "training", "pipeline_id": pipeline.get("id", "")},
                source="ml_subagent",
            )

            train_result = await self._registry.execute(
                "start_local_training",
                {"pipeline_id": pipeline.get("id", "")},
            )

            bus.publish(
                EventType.ML_SUBAGENT_STATUS,
                {"status": "training_launched", "detail": train_result},
                source="ml_subagent",
            )

        except Exception as e:
            logger.error(f"ML subagent error: {e}")
            bus.publish(
                EventType.ML_SUBAGENT_STATUS,
                {"status": "error", "detail": str(e)},
                source="ml_subagent",
            )
        finally:
            self._running = False

    async def stop(self):
        """Request the subagent to stop."""
        self._running = False
