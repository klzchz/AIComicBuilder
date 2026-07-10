"""Generation pipeline — Python port of src/lib/pipeline/.

Public API:
    - ``register_pipeline_handlers()``  register the async task handlers with the
      task queue (mirrors src/lib/pipeline/index.ts registerPipelineHandlers).
    - each stage individually callable as ``run_*`` (the API layer calls these
      directly) and as ``handle_*`` (the worker calls these with a Task).

Task-type -> handler map (from index.ts):
    script_outline    -> handle_script_outline
    script_parse      -> handle_script_parse
    character_extract -> handle_character_extract
    character_image   -> handle_character_image
    shot_split        -> handle_shot_split
    frame_generate    -> handle_frame_generate
    video_generate    -> handle_video_generate
    video_assemble    -> handle_video_assemble

Import-safe: importing this package pulls in only the stage modules (which
lazy-import app.ai / app.ai.prompts / app.task_queue / app.video inside their
functions), so there is no import cycle and no side effect at import time.
"""
from __future__ import annotations

from app.pipeline.character_extract import handle_character_extract, run_character_extract
from app.pipeline.character_image import handle_character_image, run_character_image
from app.pipeline.continuity_check import check_continuity
from app.pipeline.frame_generate import handle_frame_generate, run_frame_generate
from app.pipeline.script_outline import handle_script_outline, run_script_outline
from app.pipeline.script_parse import handle_script_parse, run_script_parse
from app.pipeline.shot_split import handle_shot_split, run_shot_split
from app.pipeline.video_assemble import handle_video_assemble, run_video_assemble
from app.pipeline.video_generate import handle_video_generate, run_video_generate
from app.pipeline.video_quality_check import check_video_quality


def register_pipeline_handlers() -> None:
    """Register all pipeline handlers with the task queue.

    Port of registerPipelineHandlers. ``register_handler`` is imported lazily to
    avoid an import cycle (the queue package and the pipeline reference each
    other) and to keep this module import-safe.
    """
    from app.task_queue import register_handler  # lazy

    handlers = {
        "script_outline": handle_script_outline,
        "script_parse": handle_script_parse,
        "character_extract": handle_character_extract,
        "character_image": handle_character_image,
        "shot_split": handle_shot_split,
        "frame_generate": handle_frame_generate,
        "video_generate": handle_video_generate,
        "video_assemble": handle_video_assemble,
    }
    for task_type, handler in handlers.items():
        register_handler(task_type, handler)


__all__ = [
    "register_pipeline_handlers",
    # run_* stage functions (called directly by the API layer)
    "run_script_outline",
    "run_script_parse",
    "run_character_extract",
    "run_character_image",
    "run_shot_split",
    "run_frame_generate",
    "run_video_generate",
    "run_video_assemble",
    # handle_* task handlers (called by the worker)
    "handle_script_outline",
    "handle_script_parse",
    "handle_character_extract",
    "handle_character_image",
    "handle_shot_split",
    "handle_frame_generate",
    "handle_video_generate",
    "handle_video_assemble",
    # quality/continuity checks
    "check_continuity",
    "check_video_quality",
]
