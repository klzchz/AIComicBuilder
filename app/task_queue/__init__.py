"""DB-backed background task queue — Python port of src/lib/task-queue.

Public API (per the porting contract):
    - ``start_worker()``      start the background worker (idempotent)
    - ``register_handler()``  register an async handler for a task type
    - ``enqueue()``           insert a pending Task, return its id

Also re-exports the queue/worker primitives for parity with the TS index.ts
(enqueue_task, complete_task, fail_task, get_tasks_by_project, register_handlers,
stop_worker).

Import-safe: importing this package starts no worker and touches no DB. The
DB is only accessed when queue functions run; the worker only spins up in
``start_worker()``.
"""
from __future__ import annotations

from app.task_queue.queue import (
    complete_task,
    dequeue_task,
    enqueue_task,
    fail_task,
    get_tasks_by_project,
)
from app.task_queue.worker import (
    POLL_INTERVAL_MS,
    register_handler,
    register_handlers,
    start_worker,
    stop_worker,
)


def enqueue(
    task_type: str,
    payload: dict,
    *,
    project_id: str | None = None,
    episode_id: str | None = None,
    max_retries: int = 3,
    scheduled_at: int | None = None,
) -> str:
    """Insert a pending Task row and return its id.

    Thin, ergonomic wrapper over :func:`enqueue_task` matching the contract
    signature the pipeline calls.
    """
    return enqueue_task(
        type=task_type,
        payload=payload,
        project_id=project_id,
        episode_id=episode_id,
        max_retries=max_retries,
        scheduled_at=scheduled_at,
    )


__all__ = [
    "enqueue",
    "enqueue_task",
    "register_handler",
    "register_handlers",
    "start_worker",
    "stop_worker",
    "complete_task",
    "fail_task",
    "dequeue_task",
    "get_tasks_by_project",
    "POLL_INTERVAL_MS",
]
