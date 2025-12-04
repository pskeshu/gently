"""
FastAPI routes for CV Subagent service

Provides the intent-based API for computer vision analysis.
"""

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, HTTPException, Query

from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    TaskStatus,
    TaskStatusEnum,
    TaskListResponse,
    ServiceStatus,
    TaskQueueStats,
)

if TYPE_CHECKING:
    from ..service import CVSubagentService

logger = logging.getLogger(__name__)


def create_router(service: "CVSubagentService") -> APIRouter:
    """
    Create FastAPI router with service reference

    Parameters
    ----------
    service : CVSubagentService
        The CV subagent service instance

    Returns
    -------
    APIRouter
        Configured FastAPI router
    """
    router = APIRouter(tags=["CV Subagent"])

    # =========================================================================
    # Analysis Endpoints
    # =========================================================================

    @router.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
        """
        Submit a CV analysis request

        The CV subagent receives high-level intent and autonomously
        determines which tools to use for analysis.

        ## Example Intents:
        - "classify embryo stage"
        - "count cells and track divisions over last 5 timepoints"
        - "detect developmental anomalies"
        - "measure embryo morphology"

        ## How it Works:
        1. The CV agent (Claude) receives your intent
        2. It plans which tools to use (Cellpose, StarDist, Claude Vision, etc.)
        3. It executes the plan, enriching context at each step
        4. Results are returned with visual verification

        ## Response:
        Returns a task ID for tracking. Use GET /tasks/{task_id} to check status.
        """
        try:
            result = await service.analyze(
                intent=request.intent,
                embryo_id=request.embryo_id,
                timepoints=request.timepoints,
                volume_uids=request.volume_uids,
                context=request.context,
            )

            return AnalyzeResponse(
                task_id=result.get("task_id", ""),
                status=TaskStatusEnum(result.get("status", "queued")),
                plan=result.get("plan", []),
                estimated_time_seconds=result.get("estimated_time_seconds"),
            )

        except Exception as e:
            logger.error(f"Analysis request failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Task Management Endpoints
    # =========================================================================

    @router.get("/tasks/{task_id}", response_model=TaskStatus)
    async def get_task(task_id: str) -> TaskStatus:
        """
        Get status of a specific task

        Returns current status, progress, and result (if completed).
        """
        try:
            status = await service.get_task_status(task_id)

            if status is None:
                raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

            return TaskStatus(
                task_id=status["task_id"],
                status=TaskStatusEnum(status["status"]),
                created_at=status["created_at"],
                started_at=status.get("started_at"),
                completed_at=status.get("completed_at"),
                progress=status.get("progress"),
                current_step=status.get("current_step"),
                result=status.get("result"),
                error=status.get("error"),
                plan=status.get("plan", []),
                metadata=status.get("metadata", {}),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get task status: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/tasks/{task_id}")
    async def cancel_task(task_id: str) -> dict:
        """
        Cancel a queued or processing task

        Returns success status. Note: tasks that have already completed
        cannot be cancelled.
        """
        try:
            success = await service.cancel_task(task_id)

            if not success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not cancel task: {task_id} (may be completed or not found)",
                )

            return {"success": True, "task_id": task_id, "message": "Task cancelled"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/tasks", response_model=TaskListResponse)
    async def list_tasks(
        status: Optional[TaskStatusEnum] = Query(None, description="Filter by status"),
        limit: int = Query(20, ge=1, le=100, description="Maximum tasks to return"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
    ) -> TaskListResponse:
        """
        List tasks with optional filtering

        Supports filtering by status and pagination.
        """
        try:
            # Get tasks from queue
            tasks = await service.task_queue.list_tasks(
                status=status.value if status else None,
                limit=limit,
                offset=offset,
            )

            total = await service.task_queue.count_tasks(
                status=status.value if status else None,
            )

            return TaskListResponse(
                tasks=[
                    TaskStatus(
                        task_id=t["task_id"],
                        status=TaskStatusEnum(t["status"]),
                        created_at=t["created_at"],
                        started_at=t.get("started_at"),
                        completed_at=t.get("completed_at"),
                        progress=t.get("progress"),
                        current_step=t.get("current_step"),
                        result=t.get("result"),
                        error=t.get("error"),
                        plan=t.get("plan", []),
                        metadata=t.get("metadata", {}),
                    )
                    for t in tasks
                ],
                total=total,
                limit=limit,
                offset=offset,
            )

        except Exception as e:
            logger.error(f"Failed to list tasks: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Service Status Endpoints
    # =========================================================================

    @router.get("/status", response_model=ServiceStatus)
    async def get_status() -> ServiceStatus:
        """
        Get detailed service status

        Returns information about:
        - Service state and uptime
        - GPU availability and memory
        - Task queue stats
        - Loaded models
        - Available capabilities
        """
        try:
            status = service.get_service_status()

            return ServiceStatus(
                name=status["name"],
                version=status["version"],
                state=status["state"],
                host=status["host"],
                port=status["port"],
                gpu_available=status["gpu_available"],
                gpu_memory_used_mb=status.get("gpu_memory_used_mb"),
                gpu_memory_total_mb=status.get("gpu_memory_total_mb"),
                models_loaded=status.get("models_loaded", []),
                task_queue=TaskQueueStats(
                    active_tasks=status["task_queue"]["active_tasks"],
                    queued_tasks=status["task_queue"]["queued_tasks"],
                    completed_tasks=status["task_queue"].get("completed_tasks", 0),
                    failed_tasks=status["task_queue"].get("failed_tasks", 0),
                    max_concurrent=status["task_queue"]["max_concurrent"],
                ),
                capabilities=status.get("capabilities", []),
                uptime_seconds=status.get("uptime_seconds"),
            )

        except Exception as e:
            logger.error(f"Failed to get service status: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Auto-Analysis Control Endpoints
    # =========================================================================

    @router.get("/auto-analysis")
    async def get_auto_analysis_config() -> dict:
        """
        Get current auto-analysis configuration

        Auto-analysis automatically processes new volumes when they are acquired.
        """
        return service.get_auto_analysis_config()

    @router.post("/auto-analysis/enable")
    async def enable_auto_analysis() -> dict:
        """
        Enable auto-analysis

        When enabled, the CV subagent will automatically analyze
        new volumes as they are acquired from the microscope.
        """
        service.set_auto_analysis_enabled(True)
        return {
            "success": True,
            "message": "Auto-analysis enabled",
            "config": service.get_auto_analysis_config(),
        }

    @router.post("/auto-analysis/disable")
    async def disable_auto_analysis() -> dict:
        """
        Disable auto-analysis

        When disabled, volumes will only be analyzed when explicitly requested.
        """
        service.set_auto_analysis_enabled(False)
        return {
            "success": True,
            "message": "Auto-analysis disabled",
            "config": service.get_auto_analysis_config(),
        }

    return router
