"""Task queue DB operations — Python port of src/lib/task-queue/queue.ts.

All functions open a short-lived session via db_session(); nothing touches
the DB at import time.
"""
from __future__ import annotations

import json
import time
from typing import Any

from sqlalchemy import select, update

from app.core.ids import new_id
from app.db.models import Task
from app.db.session import db_session


def enqueue_task(
    *,
    type: str,
    payload: Any = None,
    project_id: str | None = None,
    max_retries: int = 3,
    scheduled_at: int | None = None,
    episode_id: str | None = None,
) -> str:
    """Insert a pending task and return its id. Port of enqueueTask.

    payload is stored as a JSON string in the TEXT column (Drizzle did the
    same JSON serialization under the hood). scheduled_at is unix seconds.
    """
    task_id = new_id()
    with db_session() as session:
        session.add(
            Task(
                id=task_id,
                type=type,
                project_id=project_id,
                payload=json.dumps(payload) if payload is not None else None,
                max_retries=max_retries,
                scheduled_at=scheduled_at,
                episode_id=episode_id,
            )
        )
    return task_id


def dequeue_task() -> Task | None:
    """Atomically claim the oldest due pending task, marking it running.

    Port of dequeueTask. The TS version used a single UPDATE with a
    correlated subquery; here the select + guarded UPDATE run inside one
    transaction, and the ``status == 'pending'`` guard on the UPDATE makes
    the claim safe against concurrent workers (rowcount 0 → lost the race).
    """
    now = int(time.time())
    with db_session() as session:
        candidate_id = session.execute(
            select(Task.id)
            .where(
                Task.status == "pending",
                (Task.scheduled_at.is_(None)) | (Task.scheduled_at <= now),
            )
            .order_by(Task.created_at.asc())
            .limit(1)
        ).scalar_one_or_none()

        if candidate_id is None:
            return None

        claimed = session.execute(
            update(Task)
            .where(Task.id == candidate_id, Task.status == "pending")
            .values(status="running")
        )
        if claimed.rowcount != 1:
            return None  # another worker claimed it first

        return session.get(Task, candidate_id)


def complete_task(id: str, result: Any) -> None:
    """Mark a task completed and store its result as JSON. Port of completeTask."""
    with db_session() as session:
        session.execute(
            update(Task)
            .where(Task.id == id)
            .values(status="completed", result=json.dumps(result) if result is not None else None)
        )


def fail_task(id: str, error: str) -> None:
    """Record a failure: retry (back to pending) until max_retries, then fail.

    Port of failTask — faithful to the source: retries is incremented and the
    task is re-queued as "pending" (picked up on a subsequent poll; the 2s
    poll interval is the effective backoff) while ``retries < max_retries``;
    otherwise it is marked "failed".
    """
    with db_session() as session:
        task = session.get(Task, id)
        if task is None:
            return

        new_retries = (task.retries or 0) + 1
        max_retries = task.max_retries if task.max_retries is not None else 3

        if new_retries < max_retries:
            task.status = "pending"
        else:
            task.status = "failed"
        task.retries = new_retries
        task.error = error


def get_tasks_by_project(project_id: str) -> list[Task]:
    """Return all tasks for a project, oldest first. Port of getTasksByProject."""
    with db_session() as session:
        return list(
            session.execute(
                select(Task).where(Task.project_id == project_id).order_by(Task.created_at.asc())
            ).scalars()
        )
