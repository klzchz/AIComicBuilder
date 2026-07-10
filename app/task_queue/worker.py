"""Background task worker — Python port of src/lib/task-queue/worker.ts.

The TS original ran a ``setTimeout`` poll loop on Node's single event loop. In
Python there is no ambient event loop for a library, so the worker runs its own
asyncio event loop inside a dedicated **daemon thread**: the loop polls the DB
every ``POLL_INTERVAL_MS`` for a due pending task, hands it to the registered
async handler, and records the result (or a failure/retry).

Import-safe: importing this module starts no thread and touches no DB. The
worker only spins up when :func:`start_worker` is called, and it is idempotent
(a double call is a no-op, guarded by a lock).

Handlers are ``async def handler(task: Task) -> Any`` — mirroring the TS
``TaskHandler`` signature ``(task) => Promise<unknown>``. Handlers run on the
worker's event loop, so they may freely ``await`` other coroutines.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Awaitable, Callable

from app.db.models import Task
from app.task_queue.queue import complete_task, dequeue_task, fail_task

# Poll cadence, faithful to the TS constant. Exposed in seconds for asyncio.sleep.
POLL_INTERVAL_MS = 2000
_POLL_INTERVAL_S = POLL_INTERVAL_MS / 1000.0

TaskHandler = Callable[[Task], Awaitable[Any]]

# ── Module state (all mutations guarded by _lock) ─────────────────────────
_handlers: dict[str, TaskHandler] = {}
_lock = threading.Lock()
_running = False
_thread: threading.Thread | None = None
_loop: asyncio.AbstractEventLoop | None = None


def register_handler(task_type: str, handler: TaskHandler) -> None:
    """Register (or replace) the async handler for a single task type."""
    with _lock:
        _handlers[task_type] = handler


def register_handlers(new_handlers: dict[str, TaskHandler]) -> None:
    """Merge a mapping of task_type -> handler. Port of registerHandlers (spread merge)."""
    with _lock:
        _handlers.update(new_handlers)


def _parse_payload(task: Task) -> Any:
    """Return the task payload as a parsed Python object (JSON), or None."""
    if not task.payload:
        return None
    try:
        return json.loads(task.payload)
    except (json.JSONDecodeError, TypeError):
        # Not valid JSON — hand the raw string to the handler as-is.
        return task.payload


async def _process_task(task: Task) -> None:
    """Run one task's handler and record the outcome. Port of processTask.

    The handler is called with the Task ORM object; ``task.parsed_payload`` is
    attached as a convenience so handlers get the decoded dict without touching
    the raw JSON column. On success the return value is stored as the result;
    any exception is routed to fail_task (which retries or marks failed).
    """
    handler = _handlers.get(task.type) if task.type else None
    if handler is None:
        fail_task(task.id, f"No handler registered for task type: {task.type}")
        return

    # Convenience: expose the decoded payload without mutating the DB column.
    try:
        task.parsed_payload = _parse_payload(task)  # type: ignore[attr-defined]
    except Exception:
        task.parsed_payload = None  # type: ignore[attr-defined]

    try:
        result = await handler(task)
        complete_task(task.id, result)
    except Exception as err:  # noqa: BLE001 — mirror TS catch-all
        message = str(err) if str(err) else err.__class__.__name__
        fail_task(task.id, message)


async def _poll_once() -> None:
    """Claim and process a single due task, if any. Port of one poll() iteration."""
    task = dequeue_task()
    if task is not None:
        await _process_task(task)


async def _run_loop() -> None:
    """Async poll loop — runs until stop_worker() clears the running flag."""
    print(f"[TaskWorker] Started polling every {POLL_INTERVAL_MS} ms")
    while _running:
        try:
            await _poll_once()
        except Exception as err:  # noqa: BLE001
            print(f"[TaskWorker] Poll error: {err}")
        # Sleep between polls even after work, matching the TS setTimeout cadence.
        await asyncio.sleep(_POLL_INTERVAL_S)
    print("[TaskWorker] Stopped")


def _thread_main() -> None:
    global _loop
    loop = asyncio.new_event_loop()
    _loop = loop
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_loop())
    finally:
        loop.close()
        _loop = None


def start_worker() -> None:
    """Start the background worker thread. Idempotent. Port of startWorker.

    Spins up a daemon thread hosting a private asyncio event loop that polls the
    task queue. A second call while already running is a no-op (guarded).
    """
    global _running, _thread
    with _lock:
        if _running:
            return
        _running = True
        _thread = threading.Thread(
            target=_thread_main, name="task-queue-worker", daemon=True
        )
        _thread.start()


def stop_worker() -> None:
    """Signal the worker loop to stop. Port of stopWorker.

    Clears the running flag; the loop exits after its current poll/sleep. Does
    not join the daemon thread (it winds down on its own and dies with the
    process). Safe to call when not running.
    """
    global _running
    with _lock:
        _running = False


def is_running() -> bool:
    """Return whether the worker loop is currently active."""
    return _running
