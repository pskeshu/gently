"""
GapPlanner — Creates PlanItems in ContextStore to fill annotation gaps.
"""

import logging

from .models import CoverageReport

logger = logging.getLogger(__name__)


class GapPlanner:
    """Creates plan items to address annotation gaps.

    Parameters
    ----------
    context_store : optional
        ContextStore for creating PlanItems.
    """

    def __init__(self, context_store=None):
        self._context_store = context_store

    def plan_annotation_campaign(
        self,
        campaign_id: str,
        coverage_report: CoverageReport,
        target_per_stage: int = 50,
    ) -> list:
        """Create PlanItems based on coverage gaps.

        Parameters
        ----------
        campaign_id : str
            Campaign to add items to.
        coverage_report : CoverageReport
            Coverage analysis results.
        target_per_stage : int
            Target number of annotations per stage.

        Returns
        -------
        list
            Created PlanItem IDs.
        """
        if self._context_store is None:
            return []

        created_ids = []

        for stage, count in coverage_report.stage_counts.items():
            if count < target_per_stage:
                needed = target_per_stage - count
                try:
                    item = self._context_store.create_plan_item(
                        campaign_id=campaign_id,
                        type="analysis",
                        title=f"Annotate {needed} more {stage} embryos",
                        description=(
                            f"Current count: {count}/{target_per_stage}. "
                            f"Need {needed} more ground truth annotations for "
                            f"the {stage} stage to reach training threshold."
                        ),
                    )
                    item_id = item.id if hasattr(item, "id") else item
                    created_ids.append(item_id)
                except Exception as e:
                    logger.error(f"Failed to create plan item for {stage}: {e}")

        # Create items for completely missing stages
        from .coverage import KNOWN_STAGES

        present_stages = set(coverage_report.stage_counts.keys())
        for stage in KNOWN_STAGES:
            if stage not in present_stages:
                try:
                    item = self._context_store.create_plan_item(
                        campaign_id=campaign_id,
                        type="imaging",
                        title=f"Acquire and annotate {stage} embryos",
                        description=(
                            f"No {stage} stage data exists. "
                            f"Need at least {target_per_stage} annotated examples. "
                            f"Requires targeted imaging session."
                        ),
                    )
                    item_id = item.id if hasattr(item, "id") else item
                    created_ids.append(item_id)
                except Exception as e:
                    logger.error(f"Failed to create plan item for missing {stage}: {e}")

        logger.info(f"GapPlanner: created {len(created_ids)} plan items for campaign {campaign_id}")
        return created_ids
