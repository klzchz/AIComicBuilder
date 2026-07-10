"""SQLAlchemy models — faithful port of src/lib/db/schema.ts (Drizzle).

Conventions mirrored from the source:
- Text primary keys (nanoid, see app.core.ids.new_id).
- Timestamps stored as integer unix seconds (Drizzle `integer(mode:"timestamp")`).
- JSON columns stored as TEXT holding JSON (payload/result/metadata/slots).
- Column *names* kept snake_case identical to the SQLite table so an existing
  aicomic.db stays compatible.
"""
import time

from sqlalchemy import Integer, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base
from app.core.ids import new_id


def _now() -> int:
    return int(time.time())


def _pk() -> Mapped[str]:
    return mapped_column(Text, primary_key=True, default=new_id)


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    title: Mapped[str] = mapped_column(Text, nullable=False)
    idea: Mapped[str] = mapped_column(Text, default="")
    script: Mapped[str] = mapped_column(Text, default="")
    outline: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(Text, nullable=False, default="draft")  # draft|processing|completed
    final_video_url: Mapped[str | None] = mapped_column(Text)
    generation_mode: Mapped[str] = mapped_column(Text, nullable=False, default="keyframe")  # keyframe|reference
    use_project_prompts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    color_palette: Mapped[str] = mapped_column(Text, default="")
    world_setting: Mapped[str] = mapped_column(Text, default="")
    target_duration: Mapped[int] = mapped_column(Integer, default=0)
    bgm_url: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now, onupdate=_now)


class Episode(Base):
    __tablename__ = "episodes"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    idea: Mapped[str] = mapped_column(Text, default="")
    script: Mapped[str] = mapped_column(Text, default="")
    outline: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(Text, nullable=False, default="draft")
    generation_mode: Mapped[str] = mapped_column(Text, nullable=False, default="keyframe")
    description: Mapped[str] = mapped_column(Text, default="")
    keywords: Mapped[str] = mapped_column(Text, default="")
    script_hash: Mapped[str] = mapped_column(Text, default="")
    color_palette: Mapped[str] = mapped_column(Text, default="")
    target_duration: Mapped[int] = mapped_column(Integer, default=0)
    bgm_url: Mapped[str] = mapped_column(Text, default="")
    final_video_url: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now, onupdate=_now)


class Character(Base):
    __tablename__ = "characters"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    visual_hint: Mapped[str] = mapped_column(Text, default="")
    reference_image: Mapped[str | None] = mapped_column(Text)
    reference_image_history: Mapped[str] = mapped_column(Text, default="[]")
    scope: Mapped[str] = mapped_column(Text, nullable=False, default="main")  # main|guest
    performance_style: Mapped[str] = mapped_column(Text, default="")
    height_cm: Mapped[int] = mapped_column(Integer, default=0)
    body_type: Mapped[str] = mapped_column(Text, default="average")
    is_stale: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    episode_id: Mapped[str | None] = mapped_column(Text, ForeignKey("episodes.id", ondelete="CASCADE"))


class EpisodeCharacter(Base):
    __tablename__ = "episode_characters"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    episode_id: Mapped[str] = mapped_column(Text, ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False)
    character_id: Mapped[str] = mapped_column(Text, ForeignKey("characters.id", ondelete="CASCADE"), nullable=False)


class StoryboardVersion(Base):
    __tablename__ = "storyboard_versions"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    version_num: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
    episode_id: Mapped[str | None] = mapped_column(Text, ForeignKey("episodes.id", ondelete="CASCADE"))


class Scene(Base):
    __tablename__ = "scenes"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    episode_id: Mapped[str] = mapped_column(Text, ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False)
    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, default="")
    lighting: Mapped[str] = mapped_column(Text, default="")
    color_palette: Mapped[str] = mapped_column(Text, default="")
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class ShotAsset(Base):
    __tablename__ = "shot_assets"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    shot_id: Mapped[str] = mapped_column(Text, ForeignKey("shots.id", ondelete="CASCADE"), nullable=False)
    # first_frame|last_frame|reference|keyframe_video|reference_video
    type: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_in_type: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    asset_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    file_url: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")  # pending|generating|completed|failed
    characters: Mapped[str | None] = mapped_column(Text)  # JSON array
    model_provider: Mapped[str | None] = mapped_column(Text)
    model_id: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[str | None] = mapped_column(Text)  # JSON
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now, onupdate=_now)


class Shot(Base):
    __tablename__ = "shots"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, default="")
    motion_script: Mapped[str | None] = mapped_column(Text)
    camera_direction: Mapped[str] = mapped_column(Text, default="static")
    duration: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    video_script: Mapped[str | None] = mapped_column(Text)
    video_prompt: Mapped[str | None] = mapped_column(Text)
    transition_in: Mapped[str] = mapped_column(Text, default="cut")
    transition_out: Mapped[str] = mapped_column(Text, default="cut")
    episode_id: Mapped[str | None] = mapped_column(Text, ForeignKey("episodes.id", ondelete="CASCADE"))
    version_id: Mapped[str | None] = mapped_column(Text, ForeignKey("storyboard_versions.id", ondelete="CASCADE"))
    scene_id: Mapped[str | None] = mapped_column(Text)
    composition_guide: Mapped[str] = mapped_column(Text, default="")
    focal_point: Mapped[str] = mapped_column(Text, default="")
    depth_of_field: Mapped[str] = mapped_column(Text, default="medium")
    sound_design: Mapped[str] = mapped_column(Text, default="")
    music_cue: Mapped[str] = mapped_column(Text, default="")
    costume_overrides: Mapped[str] = mapped_column(Text, default="")
    is_stale: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")


class Dialogue(Base):
    __tablename__ = "dialogues"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    shot_id: Mapped[str] = mapped_column(Text, ForeignKey("shots.id", ondelete="CASCADE"), nullable=False)
    character_id: Mapped[str] = mapped_column(Text, ForeignKey("characters.id", ondelete="CASCADE"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    audio_url: Mapped[str | None] = mapped_column(Text)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    start_ratio: Mapped[str] = mapped_column(Text, default="0")
    end_ratio: Mapped[str] = mapped_column(Text, default="1")


class ImportLog(Base):
    __tablename__ = "import_logs"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")  # running|done|error
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    metadata_: Mapped[str | None] = mapped_column("metadata", Text)  # JSON
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_key: Mapped[str] = mapped_column(Text, nullable=False)
    slot_key: Mapped[str | None] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(Text, nullable=False, default="global")  # global|project
    project_id: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now, onupdate=_now)


class PromptVersion(Base):
    __tablename__ = "prompt_versions"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    template_id: Mapped[str] = mapped_column(Text, ForeignKey("prompt_templates.id", ondelete="CASCADE"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class PromptPreset(Base):
    __tablename__ = "prompt_presets"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(Text)
    prompt_key: Mapped[str] = mapped_column(Text, nullable=False)
    slots: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class CharacterRelation(Base):
    __tablename__ = "character_relations"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    character_a_id: Mapped[str] = mapped_column(Text, ForeignKey("characters.id", ondelete="CASCADE"), nullable=False)
    character_b_id: Mapped[str] = mapped_column(Text, ForeignKey("characters.id", ondelete="CASCADE"), nullable=False)
    relation_type: Mapped[str] = mapped_column(Text, nullable=False, default="neutral")
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class CharacterCostume(Base):
    __tablename__ = "character_costumes"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    character_id: Mapped[str] = mapped_column(Text, ForeignKey("characters.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False, default="default")
    description: Mapped[str] = mapped_column(Text, default="")
    reference_image: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class MoodBoardImage(Base):
    __tablename__ = "mood_board_images"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    annotation: Mapped[str] = mapped_column(Text, default="")
    extracted_style: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class ShotAction(Base):
    __tablename__ = "shot_actions"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    shot_id: Mapped[str] = mapped_column(Text, ForeignKey("shots.id", ondelete="CASCADE"), nullable=False)
    character_id: Mapped[str | None] = mapped_column(Text)
    body_part: Mapped[str] = mapped_column(Text, default="full_body")
    motion: Mapped[str] = mapped_column(Text, nullable=False, default="")
    start_time: Mapped[str] = mapped_column(Text, default="0")
    end_time: Mapped[str] = mapped_column(Text, default="0")
    intensity: Mapped[str] = mapped_column(Text, default="normal")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class PromptAbTest(Base):
    __tablename__ = "prompt_ab_tests"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    prompt_key: Mapped[str] = mapped_column(Text, nullable=False)
    variant_a: Mapped[str] = mapped_column(Text, nullable=False)
    variant_b: Mapped[str] = mapped_column(Text, nullable=False)
    shot_id: Mapped[str | None] = mapped_column(Text)
    result_a_url: Mapped[str | None] = mapped_column(Text)
    result_b_url: Mapped[str | None] = mapped_column(Text)
    preferred: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    project_id: Mapped[str | None] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"))
    # script_outline|script_parse|character_extract|character_image|shot_split|frame_generate|video_generate|video_assemble
    type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")  # pending|running|completed|failed
    payload: Mapped[str | None] = mapped_column(Text)  # JSON
    result: Mapped[str | None] = mapped_column(Text)  # JSON
    error: Mapped[str | None] = mapped_column(Text)
    retries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
    scheduled_at: Mapped[int | None] = mapped_column(Integer)
    episode_id: Mapped[str | None] = mapped_column(Text, ForeignKey("episodes.id", ondelete="CASCADE"))


class Agent(Base):
    __tablename__ = "agents"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # script_outline|script_generate|script_parse|character_extract|shot_split|keyframe_prompts|video_prompts|ref_image_prompts|ref_video_prompts
    category: Mapped[str] = mapped_column(Text, nullable=False)
    platform: Mapped[str] = mapped_column(Text, nullable=False, default="bailian")  # bailian|dify|coze
    app_id: Mapped[str] = mapped_column(Text, nullable=False)
    api_key: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now, onupdate=_now)


class AgentBinding(Base):
    __tablename__ = "agent_bindings"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    agent_id: Mapped[str | None] = mapped_column(Text, ForeignKey("agents.id", ondelete="SET NULL"))
