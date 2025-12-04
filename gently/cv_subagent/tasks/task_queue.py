"""
Task Queue for CV Subagent

Provides async task management with priority-based scheduling
and GPU-aware resource allocation.
"""

import asyncio
import heapq
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Awaitable

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels (lower value = higher priority)"""
    URGENT = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=True)
class CVTask:
    """
    A task in the CV processing queue

    Tasks are ordered by (priority, created_at) for fair scheduling.
    """
    # For ordering
    priority_value: int = field(compare=True)
    created_at: datetime = field(compare=True)

    # Task details (not used for ordering)
    task_id: str = field(compare=False)
    task_type: str = field(compare=False)
    params: Dict[str, Any] = field(compare=False, default_factory=dict)
    priority: TaskPriority = field(compare=False, default=TaskPriority.NORMAL)
    status: TaskStatus = field(compare=False, default=TaskStatus.QUEUED)
    started_at: Optional[datetime] = field(compare=False, default=None)
    completed_at: Optional[datetime] = field(compare=False, default=None)
    result: Any = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)
    callback_event: Optional[str] = field(compare=False, default=None)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)
    plan: List[str] = field(compare=False, default_factory=list)
    progress: Optional[float] = field(compare=False, default=None)
    current_step: Optional[str] = field(compare=False, default=None)

    @classmethod
    def create(
        cls,
        task_type: str,
        params: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        callback_event: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CVTask":
        """Create a new task"""
        now = datetime.now()
        return cls(
            priority_value=priority.value,
            created_at=now,
            task_id=f"{task_type}_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            params=params,
            priority=priority,
            callback_event=callback_event,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "priority": self.priority.name.lower(),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "current_step": self.current_step,
            "result": self.result,
            "error": self.error,
            "plan": self.plan,
            "metadata": self.metadata,
        }


class TaskQueue:
    """
    Async task queue for CV operations

    Features:
    - Priority-based scheduling
    - Concurrent task limits
    - Progress tracking
    - Event bus integration for results
    - Graceful shutdown
    """

    def __init__(
        self,
        max_concurrent: int = 2,
        event_bus=None,
    ):
        """
        Initialize task queue

        Parameters
        ----------
        max_concurrent : int
            Maximum concurrent tasks
        event_bus : EventBus, optional
            Event bus for publishing results
        """
        self.max_concurrent = max_concurrent
        self.event_bus = event_bus

        # Task storage
        self._queue: List[CVTask] = []  # Priority heap
        self._tasks: Dict[str, CVTask] = {}  # All tasks by ID
        self._active: Dict[str, asyncio.Task] = {}  # Running asyncio tasks

        # Synchronization
        self._lock = asyncio.Lock()
        self._queue_event = asyncio.Event()

        # State
        self._running = False
        self._workers: List[asyncio.Task] = []

        # Processors (registered by agent)
        self._processors: Dict[str, Callable[..., Awaitable[Any]]] = {}

        # Stats
        self._completed_count = 0
        self._failed_count = 0

    async def start(self):
        """Start the task queue workers"""
        if self._running:
            return

        self._running = True

        # Start worker tasks
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        logger.info(f"Task queue started with {self.max_concurrent} workers")

    async def stop(self):
        """Stop the task queue gracefully"""
        if not self._running:
            return

        self._running = False

        # Signal workers to stop
        self._queue_event.set()

        # Cancel active tasks
        for task_id, task in self._active.items():
            task.cancel()

        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()

        logger.info("Task queue stopped")

    def register_processor(
        self,
        task_type: str,
        processor: Callable[..., Awaitable[Any]],
    ):
        """
        Register a processor for a task type

        Parameters
        ----------
        task_type : str
            Type of task to process
        processor : callable
            Async function to process the task
        """
        self._processors[task_type] = processor
        logger.debug(f"Registered processor for task type: {task_type}")

    async def submit(
        self,
        task_type: str,
        params: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        callback_event: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        plan: Optional[List[str]] = None,
    ) -> CVTask:
        """
        Submit a task to the queue

        Parameters
        ----------
        task_type : str
            Type of task
        params : dict
            Task parameters
        priority : TaskPriority
            Task priority
        callback_event : str, optional
            Event to publish on completion
        metadata : dict, optional
            Additional metadata
        plan : list, optional
            Execution plan steps

        Returns
        -------
        CVTask
            The created task
        """
        task = CVTask.create(
            task_type=task_type,
            params=params,
            priority=priority,
            callback_event=callback_event,
            metadata=metadata,
        )

        if plan:
            task.plan = plan

        async with self._lock:
            heapq.heappush(self._queue, task)
            self._tasks[task.task_id] = task

        # Signal workers
        self._queue_event.set()

        # Publish queued event
        if self.event_bus:
            try:
                from gently.core.event_bus import EventType
                self.event_bus.publish(
                    event_type=EventType.STATUS_CHANGED,
                    data={
                        'task_id': task.task_id,
                        'status': 'queued',
                        'task_type': task_type,
                    },
                    source='cv-subagent',
                )
            except Exception as e:
                logger.debug(f"Could not publish queue event: {e}")

        logger.info(f"Task queued: {task.task_id} (type={task_type}, priority={priority.name})")
        return task

    async def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status by ID"""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return task.to_dict()

    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a task

        Returns True if cancelled, False if not found or already completed.
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False

            # Cancel if active
            if task_id in self._active:
                self._active[task_id].cancel()

            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()

            # Remove from queue if still queued
            self._queue = [t for t in self._queue if t.task_id != task_id]
            heapq.heapify(self._queue)

        logger.info(f"Task cancelled: {task_id}")
        return True

    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List tasks with optional filtering"""
        tasks = list(self._tasks.values())

        # Filter by status
        if status:
            tasks = [t for t in tasks if t.status.value == status]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        # Paginate
        tasks = tasks[offset:offset + limit]

        return [t.to_dict() for t in tasks]

    async def count_tasks(self, status: Optional[str] = None) -> int:
        """Count tasks with optional filtering"""
        if status is None:
            return len(self._tasks)
        return sum(1 for t in self._tasks.values() if t.status.value == status)

    def update_progress(
        self,
        task_id: str,
        progress: float,
        current_step: Optional[str] = None,
    ):
        """Update task progress"""
        task = self._tasks.get(task_id)
        if task:
            task.progress = progress
            if current_step:
                task.current_step = current_step

    async def _worker(self, worker_id: int):
        """Worker coroutine that processes tasks"""
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            # Wait for tasks
            await self._queue_event.wait()

            # Get next task
            task = await self._get_next_task()
            if task is None:
                # No tasks, clear event and wait again
                self._queue_event.clear()
                continue

            # Process task
            try:
                await self._process_task(task)
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                self._failed_count += 1
                logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)

                # Publish failure event
                if self.event_bus:
                    try:
                        from gently.core.event_bus import EventType
                        self.event_bus.publish(
                            event_type=EventType.CV_TASK_FAILED,
                            data={
                                'task_id': task.task_id,
                                'task_type': task.task_type,
                                'error': str(e),
                                **task.metadata,
                            },
                            source='cv-subagent',
                        )
                    except Exception as pub_error:
                        logger.warning(f"Could not publish failure event: {pub_error}")
            finally:
                async with self._lock:
                    self._active.pop(task.task_id, None)

        logger.debug(f"Worker {worker_id} stopped")

    async def _get_next_task(self) -> Optional[CVTask]:
        """Get the next task to process"""
        async with self._lock:
            # Check if we can run more tasks
            if len(self._active) >= self.max_concurrent:
                return None

            # Get highest priority task
            while self._queue:
                task = heapq.heappop(self._queue)

                # Skip cancelled tasks
                if task.status == TaskStatus.CANCELLED:
                    continue

                return task

            return None

    async def _process_task(self, task: CVTask):
        """Process a single task"""
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now()

        logger.info(f"Processing task: {task.task_id}")

        # Get processor
        processor = self._processors.get(task.task_type)
        if processor is None:
            raise ValueError(f"No processor for task type: {task.task_type}")

        # Create asyncio task for tracking
        async with self._lock:
            process_task = asyncio.create_task(processor(task))
            self._active[task.task_id] = process_task

        # Wait for completion
        try:
            result = await process_task
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            self._completed_count += 1
            logger.info(f"Task completed: {task.task_id}")

            # Publish completion event
            if self.event_bus:
                try:
                    from gently.core.event_bus import EventType

                    # Publish task-specific callback event if specified
                    if task.callback_event:
                        self.event_bus.publish(
                            event_type=EventType[task.callback_event],
                            data={
                                'task_id': task.task_id,
                                'result': result,
                                **task.metadata,
                            },
                            source='cv-subagent',
                        )

                    # Always publish CV_TASK_COMPLETED
                    self.event_bus.publish(
                        event_type=EventType.CV_TASK_COMPLETED,
                        data={
                            'task_id': task.task_id,
                            'task_type': task.task_type,
                            'result': result,
                            **task.metadata,
                        },
                        source='cv-subagent',
                    )
                except Exception as e:
                    logger.warning(f"Could not publish completion event: {e}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Task processing error: {e}")
            raise
