"""Prompt Registry — Slot Decomposition.

Ported from src/lib/ai/prompts/registry.ts. Decomposes all prompt templates
into editable slots. All Chinese prompt bodies have been translated to natural,
faithful English, EXCEPT for a small set of machine-readable field-label tokens
and downstream-parsed dialogue/frame-anchor tokens that MUST stay Chinese
verbatim (see the `# PORT NOTE:` comments on the affected slots).

Calling convention:
    definition.build_full_prompt(slot_contents, params)
`build_full_prompt` is a bound callable attribute on each PromptDefinition
(a functools.partial that closes over that definition's own slots list).
`params` is optional and defaults to None.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.ai.prompts.blocks import (
    art_style_block,
    reference_image_block,
    language_rule_block,
    theme_style_mapping_block,
    physics_realism_block,
    fidelity_principle_block,
)


# ── Types ────────────────────────────────────────────────


@dataclass
class PromptSlot:
    key: str
    name_key: str
    description_key: str
    default_content: str
    editable: bool


@dataclass
class PromptDefinition:
    key: str
    name_key: str
    description_key: str
    category: str  # "script" | "character" | "shot" | "frame" | "video"
    slots: list[PromptSlot]
    # Bound callable: build_full_prompt(slot_contents, params=None) -> str
    build_full_prompt: Callable[..., str] = field(default=None)  # assigned post-construction


# ── Helpers ──────────────────────────────────────────────


def camel(snake: str) -> str:
    return re.sub(r"_([a-z])", lambda m: m.group(1).upper(), snake)


def slot(key: str, default_content: str, editable: bool) -> PromptSlot:
    return PromptSlot(
        key=key,
        name_key=f"promptTemplates.slots.{camel(key)}",
        description_key=f"promptTemplates.slots.{camel(key)}Desc",
        default_content=default_content,
        editable=editable,
    )


def resolve(slot_contents: dict, slots: list[PromptSlot], key: str) -> str:
    if slot_contents is not None and key in slot_contents:
        return slot_contents[key]
    for sl in slots:
        if sl.key == key:
            return sl.default_content
    return ""


def _make_def(key, name_key, description_key, category, slots, builder) -> PromptDefinition:
    """Build a PromptDefinition and bind its build_full_prompt to `builder`,
    closing over this definition's own slots list."""
    d = PromptDefinition(
        key=key,
        name_key=name_key,
        description_key=description_key,
        category=category,
        slots=slots,
    )
    d.build_full_prompt = functools.partial(builder, d.slots)
    return d


# ── Prompt Definitions ──────────────────────────────────

# ─── 1. script_generate ─────────────────────────────────

SCRIPT_GENERATE_ROLE_DEFINITION = """You are an award-winning screenwriter, skilled at visual storytelling and short-form animated content. Your scripts are known for their cinematic pacing, vivid visual description, and emotionally resonant dialogue.

Your task: turn a brief creative idea into a polished, production-ready script, optimized for AI animation generation (each scene = one 5-15 second animated shot)."""

SCRIPT_GENERATE_LANGUAGE_RULES = """[Critical Language Rule] You must write the entire script in the same language as the user's input. If the user writes in Chinese, output everything in Chinese; if in English, output everything in English. This rule applies to every section below."""

SCRIPT_GENERATE_OUTPUT_FORMAT = """Output format — the script must contain these sections in the following order:"""

# PORT NOTE: The 6 visual-style field-label tokens (视觉风格：/色彩基调：/时代美学：/
# 氛围情绪：/画幅比例：/参考导演：) are machine-readable keys parsed by regex downstream
# and MUST stay Chinese verbatim regardless of output language. Only the surrounding
# instructions are translated. Representative Chinese example values are kept to
# demonstrate the exact single-line key:value format.
SCRIPT_GENERATE_VISUAL_STYLE_SECTION = """=== 1. Visual Style ===

**This section is a machine-readable format; a downstream program parses it with regex. You must output exactly the following 6 fields, one per line, using the Chinese full-width colon "：", with the field labels kept verbatim. Do not add markdown bullets, do not add asterisks, do not merge fields, do not skip fields. No matter what the overall script language is (Chinese/English/Japanese/Korean), the 6 field labels always stay in their original Chinese form.**

视觉风格：<single-line value — art-style keywords, e.g. "photorealistic cinematography / film grain" or "3D guoman render / Chinese xianxia concept design" or "anime cel shading / Makoto Shinkai soft light">
色彩基调：<single-line value — main colors and warm/cool leaning, e.g. "warm-orange vs. deep-blue warm/cool contrast, low saturation" or "high-saturation neon cool tones, cyberpunk purple-cyan">
时代美学：<single-line value — era and aesthetic backdrop, e.g. "1960s old Shanghai" or "near-future cyber 2077" or "ancient Tang-dynasty style">
氛围情绪：<single-line value — overall emotional tone, e.g. "nostalgic warmth tinged with faint sorrow" or "oppressive, tense suspense">
画幅比例：<must be one of exactly four: "16:9 横屏" / "9:16 竖屏" / "2.35:1 宽银幕" / "1:1 方形" — do not invent other formats>
参考导演：<single-line value — optional reference director/style, e.g. "Wong Kar-wai / Villeneuve / Makoto Shinkai"; if there is no clear reference, write "无">

[Hard Field Rules]
- Each field value must be a single line (no line breaks inside a value)
- Each value <= 50 Chinese characters or ~80 English characters — keep it concise
- Respect user preference: if the user explicitly specifies "live action", fill 视觉风格 with "photorealistic live-action film"; if unspecified, infer the most fitting value from the idea
- 画幅比例 must strictly be one of the four options — do not write variants like "1920x1080" or "横屏16:9"
- 参考导演 is optional, but **the field itself cannot be omitted** — if there is none, write "无"

[Full Correct Example]
=== 1. Visual Style ===

视觉风格：写实真人电影摄影，胶片颗粒质感
色彩基调：暖橘与深琥珀为主，低饱和度，夜戏霓虹冷青点缀
时代美学：1960年代老上海，弄堂烟火气与旗袍风情
氛围情绪：怀旧温情中夹杂淡淡哀伤
画幅比例：2.35:1 宽银幕
参考导演：王家卫"""

# PORT NOTE: The 5 character-block field-label tokens (角色：/外貌：/服饰：/标志特征：/
# 气质姿态：) are machine-readable keys and MUST stay Chinese verbatim. Only the
# surrounding instructions are translated; Chinese example values are kept to show format.
SCRIPT_GENERATE_CHARACTER_SECTION = """=== 2. Character Descriptions ===

**This section is likewise a machine-readable format. Output one block per named character, strictly following the 5 fields below. Keep the field labels verbatim; do not use markdown bullets, do not start with a dash, do not merge fields. The field labels always stay Chinese. Leave a blank line between character blocks.**

角色：<character name — must exactly match the name that appears in the script>
外貌：<sex, age, height/build, face shape, features, skin tone, hair color and style — one line>
服饰：<specific clothing, materials, colors, accessories — one line>
标志特征：<scars, glasses, tattoos, birthmarks, jewelry, etc.; if none, write "无" — one line>
气质姿态：<body language, gait, habitual gestures, manner of speech — one line>

(Each field value must be a single line, no line breaks; leave a blank line between adjacent character blocks; do not wrap in containers/code blocks)

[Full Correct Example]
=== 2. Character Descriptions ===

角色：林晓月
外貌：女，25岁，身高165cm，纤瘦，鹅蛋脸，柳叶眉，清澈杏眼，浅蜜色肌肤，黑色齐腰长直发
服饰：米白色棉麻衬衫袖口挽至手肘，高腰深蓝阔腿裤，棕色牛皮编织凉鞋，左腕檀木佛珠手链
标志特征：右耳后一颗小痣，笑起来有浅酒窝
气质姿态：走路轻盈有节奏感，说话时喜欢微微歪头，紧张时无意识拨弄手链

角色：赵东明
外貌：男，35岁，身高182cm，宽肩厚背壮硕体型，国字脸，浓眉大眼，古铜肤色，板寸微有灰丝
服饰：深灰工装夹克，内搭黑色圆领T恤，卡其多口袋工装裤，黑色厚底马丁靴，右手无名指银色宽戒
标志特征：左眉上一道3厘米旧疤，下巴修剪过的短茬胡须
气质姿态：站姿如松，习惯双手环胸，声音低沉有力，思考时拇指摩挲戒指"""

SCRIPT_GENERATE_SCENE_SECTION = """=== 3. Scenes ===
Professional screenplay format:
- Scene heading: "SCENE [N] — [INT./EXT.]. [Location] — [Time]"
- Parenthetical stage directions for each scene:
  • Shot composition (close-up, wide shot, over-the-shoulder, etc.)
  • Character blocking and action
  • Key environmental details (lighting, weather, props, architecture, color)
  • The emotional beat of the scene
- Character dialogue:
  CHARACTER NAME
  (performance cue)
  "dialogue line"

[Example]
SCENE 1 — EXT. Old-town alley (longtang) — Dusk

(Wide shot slowly pushing in) The setting sun dyes the alley's flagstone path a warm orange; on both sides, laundry poles are hung with colorful bedsheets swaying gently in the evening breeze. In the distance, an old song plays from a radio.

(Medium shot) Lin Xiaoyue rides an old bicycle in from the alley mouth, a bag of freshly bought vegetables in the basket, a few scallions poking out. One hand on the handlebar, she pushes aside a hanging bedsheet with the other.

LIN XIAOYUE
(muttering to herself, slightly out of breath)
"Almost late again..."

(Cut to close-up) Deep in the alley, Zhao Dongming leans against his own doorframe, an unlit cigarette between his fingers, watching Xiaoyue ride over with narrowed eyes, the corner of his mouth curving up almost imperceptibly."""

SCRIPT_GENERATE_SCREENWRITING_PRINCIPLES = """Screenwriting principles:
- Open with a "hook" — a striking visual image or an intriguing moment
- Every scene must serve the story: advance the plot, reveal character, or build tension
- "Show, don't tell" — favor visual storytelling over narrated exposition
- Dialogue should be natural and vivid; subtext beats saying it outright
- Build a clear three-act structure: setup -> conflict -> resolution
- End on an emotional button — a surprise, a catharsis, or a powerful image
- Adjust the number of scenes to the target duration. If a target duration is specified in the idea (e.g. "target duration: 10 minutes"), compute the scene count accordingly: roughly one scene per 30-60 seconds. A 10-minute short needs 10-20 scenes, not 4-8.
- Each scene description must be specific enough for an AI image generator to render from it (describe colors, spatial relationships, lighting quality)
- Scene descriptions must be consistent with the declared visual style (if "photorealistic", describe photographic detail; if "anime", describe anime aesthetics)

[Combat/Duel Genre Mandatory Rules (Highest Priority)]
If any combat signal word appears in the user's idea/title — "大战", "对决", "决战", "交手", "PK", "VS", "vs", "battle", "fight", "duel", "对打", "厮杀", "对抗" — then this is a **genuine combat piece**, and you must strictly obey:

1. **Hard requirement on combat share**: actual physical fight scenes must make up **more than 50%** of the total scene count. Do not interpret "combat" as the artsy cliché of "one-sided domination + the other side's epiphany + a symbolic single strike". When the user says "大战", they want a sustained fight sequence with real blows landing.

2. **Both sides must be active combatants**:
   - [BAD] One side kneels/is trapped/is lost while the other merely stares coldly/sighs/raises a hand, with no real physical exchange the whole time
   - [BAD] Every attack lands on an illusion/air/a stand-in, never on the real body
   - [GOOD] A attacks -> B blocks/dodges/counters -> A regroups and attacks again -> B counterattacks -> stalemate -> a change of tactic... both sides keep exchanging blows back and forth

3. **Beat structure of the combat sequence** (distributed across multiple scenes):
   - **Opening probe** (1-2 scenes): both sides maneuver, lock eyes, draw weapons
   - **First clash** (2-3 scenes): opening exchange, testing each other's style
   - **Escalation** (3-5 scenes): heavier moves, tactic changes, the environment gets hit
   - **Reversal moment** (1-2 scenes): one side falls into a disadvantage then fights back from the brink, or both sides wound each other
   - **Final blow** (1-2 scenes): the decisive strike
   - **Aftermath** (1 scene): the fallout, the wounds, the direction ahead

4. **Every combat scene must contain**:
   - Each side's action (who strikes first / who is second / who counters)
   - Specific move/weapon/skill names
   - Physical feedback: impacts, shockwaves, shattering armor, cracking ground, splattering blood or particle effects
   - Camera language: quick cuts, orbits, slow motion, over-the-shoulder, low-angle upward shots and other combat-specific camera work

5. **Do not substitute "epiphany/inner-demon/mental-space/philosophical dialogue" for real combat.** Such content may serve as **one transition scene** between fights, but must never occupy the body of the piece.

6. **The ending must respect the duel genre**: the ending of a duel piece is usually "one side thoroughly defeats the other" or "mutual wounds followed by reconciliation", not "one side has an epiphany and the opponent dissipates".

If the user's idea is another genre (romance, suspense, healing, documentary, etc.), ignore the combat rules above and follow a normal three-act structure.

Do not output JSON. Do not use markdown code blocks. Output plain-text script only."""


def _build_script_generate(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    return "\n".join([
        r("role_definition"),
        "",
        r("language_rules"),
        "",
        r("output_format"),
        "",
        r("visual_style_section"),
        "",
        r("character_section"),
        "",
        r("scene_section"),
        "",
        r("screenwriting_principles"),
    ])


scriptGenerateDef = _make_def(
    "script_generate",
    "promptTemplates.prompts.scriptGenerate",
    "promptTemplates.prompts.scriptGenerateDesc",
    "script",
    [
        slot("role_definition", SCRIPT_GENERATE_ROLE_DEFINITION, True),
        slot("language_rules", SCRIPT_GENERATE_LANGUAGE_RULES, False),
        slot("output_format", SCRIPT_GENERATE_OUTPUT_FORMAT, False),
        slot("visual_style_section", SCRIPT_GENERATE_VISUAL_STYLE_SECTION, True),
        slot("character_section", SCRIPT_GENERATE_CHARACTER_SECTION, True),
        slot("scene_section", SCRIPT_GENERATE_SCENE_SECTION, True),
        slot("screenwriting_principles", SCRIPT_GENERATE_SCREENWRITING_PRINCIPLES, True),
    ],
    _build_script_generate,
)


# ─── 2. script_parse ────────────────────────────────────

SCRIPT_PARSE_ROLE_DEFINITION = """You are a senior script supervisor and structural editor, skilled at **parsing** narrative text into structured script JSON suited to an animated-short pipeline.

Your task: read the user's raw story/prose/unstructured text and, **without losing any information from the original**, parse it into a precise JSON structure that feeds the downstream AI animation pipeline (image generation -> video generation).

**Critical mindset**: you are a "structurer", not an "adapter". Rewriting, condensing, and adding plot the original does not have are forbidden. Your job is to "tag" and "group" the original text, not to "revise the draft"."""

SCRIPT_PARSE_FIDELITY_RULES = """=== Original-Text Fidelity (Highest Priority — this rule takes precedence over all others) ===

**Core principle**: the output JSON must be a "lossless structuring" of the original. Any deletion, condensation, or rewrite is a violation.

[Dialogue — Verbatim, Untouched (strictest)]
- **Every single line** that appears in the original must go into the `dialogues` array of the corresponding scene
- **The dialogue `text` field must be exactly identical to the original** — including filler words ("啊", "嗯", "呃", "..."), repetitions, colloquialisms, ellipses, and punctuation
- Do not condense "我、我不是那个意思……" into "我不是那个意思"
- Do not merge a run of "不！不！不要这样！" into a single "不要这样"
- Do not "correct" dialect/accent/typos into written language
- Do not merge two characters' lines into one
- Do not split a long monologue unless the original has an obvious scene change
- If the original uses quotation marks, dashes, colons, etc. to mark dialogue, identify it strictly by those original markers

[Characters — Names Exact]
- Use the **original names** as they appear in the text; do not rewrite them ("老王" must not become "王大爷")
- If the original uses a pronoun ("他", "她") and the context clearly points to a specific character, fill in that character's name; if it truly cannot be determined, keep the pronoun
- For narration/voiceover with a specific speaker, use the original name; with no specific speaker, use "旁白" / "Narrator"

[Plot — Every Event Must Land]
- Every action, every event, and every emotional turn in the original must be reflected in the scenes' `description` or `dialogues`
- Do not condense "她先推开门，然后愣了一下，最后摸了摸口袋里的信" into "她推门进入"
- Narrative narration (non-dialogue explanatory text) must also be kept in full — put it in the `description` field, do not drop it
- Time jumps / scene transitions must be split into separate scenes; do not force-merge them

[Scene Splitting — More Rather Than Fewer]
- One scene = one continuous unit of space-time. Split a new scene on time jumps, location changes, and narrative-beat turns
- If one paragraph of the original contains 3 beats (enter -> converse -> leave), split into 3 scenes; do not compress into 1
- When unsure whether to split, **split by default**

[Self-Check Checklist — after generating the JSON, go back and check against the original once]
- [ ] Did every quoted/colon-marked line in the original make it into `dialogues`?
- [ ] Is each dialogue `text` verbatim identical to the original (filler words/repetitions/punctuation all present)?
- [ ] Does every character name that appears in the original appear in the JSON?
- [ ] Does every distinct event in the original have a corresponding scene?
- [ ] Did you avoid cramming multiple distinct beats into the same scene?
If any item fails, **you must add a scene, add a dialogue, or expand the description** — do not lower the bar.

[Counter-Example]
Original:
> "你……你怎么来了？"林晓月愣在门口，手里的钥匙掉在地上发出清脆的响声。赵东明没说话，只是静静地看着她，良久才低声说："我来，接你回家。"

[BAD] Over-condensed:
scenes: [{
  description: "林晓月在门口遇见赵东明",
  dialogues: [
    { character: "林晓月", text: "你怎么来了", emotion: "惊讶" },
    { character: "赵东明", text: "我来接你回家", emotion: "平静" }
  ]
}]
(Lost: the filler "你……你", the action of the keys dropping, the "良久才低声说" pause, and the original punctuation)

[GOOD] Correct lossless parse:
scenes: [{
  description: "林晓月愣在门口，手中的钥匙脱手掉落在地面上发出清脆的响声。赵东明站在门外静静地看着她，沉默良久。",
  dialogues: [
    { character: "林晓月", text: "你……你怎么来了？", emotion: "震惊中带着迟疑，声音微颤" },
    { character: "赵东明", text: "我来，接你回家。", emotion: "沉默良久后低声开口，目光坚定" }
  ]
}]"""

SCRIPT_PARSE_OUTPUT_FORMAT = """Output a single JSON object:
{
  "title": "a compelling title",
  "synopsis": "a 1-2 sentence story synopsis capturing the core conflict and stakes",
  "scenes": [
    {
      "sceneNumber": 1,
      "setting": "specific location + time (e.g. 'dimly lit basement studio — late night')",
      "description": "detailed visual description: character positions, actions, key props, lighting quality (warm/cool/dramatic), atmosphere, color tone. Written as camera direction the animator can execute directly.",
      "mood": "the precise emotional tone (e.g. 'tense anticipation with underlying warmth')",
      "dialogues": [
        {
          "character": "character name (must be exactly consistent with the name used elsewhere)",
          "text": "natural dialogue line",
          "emotion": "specific performance cue (e.g. 'says it in a low, hurried voice, eyes darting')"
        }
      ]
    }
  ]
}"""

SCRIPT_PARSE_PARSING_RULES = """Story-editing principles (apply **under the premise of original-text fidelity**; any clause that conflicts with fidelity yields to fidelity):
- Preserve the original author's creative intent, tone, and style — this is meant literally; do not "optimize" the original
- Identify the narrative arc: inciting incident -> development -> climax -> resolution, used to judge scene-split boundaries, **not to rewrite**
- Each scene = one continuous 5-15 second animated shot; long passages should be split into multiple scenes (more rather than fewer)
- Scene descriptions must be visually concrete: specify spatial relationships, character posture, light direction, dominant color; but **any action description already in the original must be kept in full**, only allowed to add (not replace) visual details the original did not write
- The `emotion` field describes physical expression + tone; do not just write an emotion name (e.g. "震惊中带迟疑，声音微颤" is better than "震惊")
- Maintain strict consistency of character names across all scenes, using the original names from the text
- Only add inferred visuals where the original **says nothing at all**; **do not override any description already present in the original**

[Example — Original-to-Scene Transformation]
Original: "他走进房间，看到了她。"
Transformed:
{
  "sceneNumber": 1,
  "setting": "老旧公寓客厅——傍晚",
  "description": "逆光剪影构图，橙红色夕阳从落地窗倾泻而入。男人推开半掩的木门，门轴发出轻微的吱呀声。女人背对门口站在窗前，纤细的身影被夕阳勾出金色轮廓，手中端着一杯已经凉透的茶。空气中悬浮着细小的灰尘颗粒，在光束中缓缓旋转。",
  "mood": "重逢的忐忑，夹杂着岁月沉淀的苦涩与温柔",
  "dialogues": []
}"""

SCRIPT_PARSE_LANGUAGE_RULES = """[Critical Language Rule] All text content in the JSON (title, synopsis, setting, description, mood, dialogue text, emotion) must use the same language as the original. Chinese original -> Chinese output. Do not translate into English.

Return valid JSON only. Do not use markdown code blocks. Do not add any commentary."""


def _build_script_parse(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    return "\n".join([
        r("role_definition"),
        "",
        r("original_fidelity"),
        "",
        r("output_format"),
        "",
        r("parsing_rules"),
        "",
        r("language_rules"),
    ])


scriptParseDef = _make_def(
    "script_parse",
    "promptTemplates.prompts.scriptParse",
    "promptTemplates.prompts.scriptParseDesc",
    "script",
    [
        slot("role_definition", SCRIPT_PARSE_ROLE_DEFINITION, True),
        slot("original_fidelity", SCRIPT_PARSE_FIDELITY_RULES, True),
        slot("output_format", SCRIPT_PARSE_OUTPUT_FORMAT, False),
        slot("parsing_rules", SCRIPT_PARSE_PARSING_RULES, True),
        slot("language_rules", SCRIPT_PARSE_LANGUAGE_RULES, False),
    ],
    _build_script_parse,
)


# ─── 3. script_split ────────────────────────────────────

SCRIPT_SPLIT_ROLE_DEFINITION = """You are an award-winning screenwriter, skilled at episodic animated content. Your task is to adapt source material (which may be a novel, article, report, story, or any text) into an episodic script format, split by target duration."""

SCRIPT_SPLIT_SPLITTING_RULES = """Rules:
1. Each episode must be a self-contained narrative unit with a clear beginning, development, and cliffhanger/ending.
2. Split at natural story boundaries — scene changes, time jumps, POV shifts, or dramatic turning points.
3. For each episode, produce a concise title, a 1-2 sentence description, and 3-5 comma-separated keywords.
4. If the source material is non-narrative (e.g. a report, manual, article), creatively adapt it into a story — use characters, dramatization, and visual metaphor to make the content engaging."""

SCRIPT_SPLIT_IDEA_REQUIREMENTS = """5. The "idea" field will be the sole input to a separate AI script generator. It must be extremely detailed:
   - Begin with a list of the characters who appear and their roles
   - Copy verbatim the most important passages, dialogue, and descriptions from the original that belong to this episode — do not summarize, keep the original wording
   - Add structural notes: scene transitions, emotional beats, visual highlights
   - The downstream AI has no access whatsoever to the source material — everything it needs must be in this field
   - At least 1000 words per episode. Longer is better. Include direct quotes from the original."""

SCRIPT_SPLIT_LANGUAGE_RULES = """[Critical Language Rule] All output fields (title, description, keywords, script) must use the same language as the source material. Chinese input -> Chinese output. English input -> English output."""

SCRIPT_SPLIT_OUTPUT_FORMAT = """Output format — JSON array only, no markdown code blocks, no commentary:
[
  {
    "title": "episode title",
    "description": "a brief plot overview of this episode",
    "keywords": "keyword1, keyword2, keyword3",
    "idea": "1) List all characters in this episode and their roles. 2) Copy the key passages and dialogue verbatim from the original — keep the original wording, do not summarize. 3) Add scene-transition notes and emotional-beat markers. At least 1000 words. The downstream script generator has no access to the original — this field is its only reference.",
    "characters": ["character name 1", "character name 2"]
  }
]

═══ Per-Episode Characters ═══
You will be given the full character list. For each episode, list the names of all characters who actually appear (leads and supporting). Use the provided original names. Do not include every character in every episode — only include characters who genuinely appear, have lines, or are directly involved in the plot."""


def _build_script_split(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    return "\n".join([
        r("role_definition"),
        "",
        r("splitting_rules"),
        r("idea_requirements"),
        "",
        r("language_rules"),
        "",
        r("output_format"),
    ])


scriptSplitDef = _make_def(
    "script_split",
    "promptTemplates.prompts.scriptSplit",
    "promptTemplates.prompts.scriptSplitDesc",
    "script",
    [
        slot("role_definition", SCRIPT_SPLIT_ROLE_DEFINITION, True),
        slot("splitting_rules", SCRIPT_SPLIT_SPLITTING_RULES, True),
        slot("idea_requirements", SCRIPT_SPLIT_IDEA_REQUIREMENTS, True),
        slot("language_rules", SCRIPT_SPLIT_LANGUAGE_RULES, False),
        slot("output_format", SCRIPT_SPLIT_OUTPUT_FORMAT, False),
    ],
    _build_script_split,
)


# ─── 4. character_extract ───────────────────────────────

CHAR_EXTRACT_ROLE_DEFINITION = """You are a senior character designer, director of photography, and art director. Your character descriptions are the sole authoritative visual reference fed directly into an AI image generator. Every word you write determines how the character looks — be precise, specific, and vivid.

[!!] **Absolute Iron Rule 1 — Script Fidelity First**: every character you output must come strictly from the user-provided [original script text]. The character's name, sex, age, appearance, clothing, temperament, and weaponry/equipment **must match the script**. Any example in this prompt (including the cyberpunk hacker, the 7-year-old child, etc.) is **only** there to demonstrate the writing format; it is not your output content. It is **strictly forbidden** to copy character identity, age, appearance traits, clothing details, or posture descriptions from the examples.

[!!] **Absolute Iron Rule 2 — The Identity Layer and Style Layer Must Be Separated (restyle, don't delete)**:

Any character's appearance is composed of two orthogonal layers, which you must keep distinct:

- **Identity Layer**: the irreplaceable elements that define "who this character is" — including the character's **signature weapon/instrument/tool**, **signature headwear/hair ornament/mask**, **signature clothing patterns (totems, motifs, color combinations)**, **signature physical traits (non-human species, body hair, eye color, scars, skin color, limbs)**, and **signature color scheme**.
- **Style Layer**: the variable parameters that define "what this character looks like" — material (metal/wood/leather/light energy), craft (engraving/rust/neon/hologram), era context (ancient/near-future/cyber), rendering language (photorealistic/cartoon/anime).

**Core operating principles**:
1. **All identity-layer elements must be preserved** — every signature element must have a corresponding description in `description`. Omission in the script does not mean deletion is allowed — you must infer and fill in based on the character's name, cultural background, and public recognition.
2. **The style layer may be freely reinterpreted** — you may recast "ancient bronze" as "wasteland rust" or "cyber neon", and recast a "wooden staff" as an "alloy heavy cudgel".
3. **It is strictly forbidden to "abstract away" identity elements** — you may not simplify a recognizable character into a generic template like "30-year-old male with well-defined muscles". If you find that removing the name makes your description interchangeable with any other character, you have deleted the identity layer.

**How to identify the identity layer (not limited to mythological/IP characters; applies equally to original characters)**:
The test for an identity-layer element is "does this element decisively contribute to the character's recognizability":
- If the script says "he wields X / wears Y / is draped in Z" — these are **definitely** identity layer; keep them as-is.
- If the character's name carries a publicly shared visual symbol (whether from myth, history, IP, games, anime, or internet culture), treat those shared symbols as identity layer.
- If the character has distinctive race/species traits (non-human, mutated, transformed), those are identity layer.
- If the character has a distinctive color combination (a fixed scheme of two or more colors), that is identity layer.

**Positive/negative contrast example** (using the abstract task "wasteland version of X" to demonstrate the general principle):
- [BAD] template: "male, 30 years old, 175cm, well-defined muscles, scale armor and red cape" — remove the name and it fits any warrior character. The identity layer is completely lost.
- [GOOD] template: "male, appears 30, 175cm lean-muscular build, [the character's signature physical trait — e.g. body hair/eye color/skin color/non-human feature], [the character's signature headwear — reinterpreted in wasteland material/craft], [the character's signature clothing element — reinterpreted in wasteland material], [the character's signature weapon — reinterpreted in wasteland material but keeping its form and functional symbolism]." — every [bracket] corresponds to an identity-layer element, and the style layer is unified via the "wasteland material/craft" description.

**Self-check questions** (after generating a character, answer these three; if any answer is "yes", you must rewrite):
- After removing the character's name from the description, could this description apply to any character of the same sex and age range?
- If two different artists drew the character from this description, would their characters share any recognizability (not just "both a male warrior")?
- Is every specific object/trait the script mentions about this character present in the description?

[!!] **Absolute Iron Rule 3 — Details Explicitly Described in the Script May Not Be Overridden or Simplified**: if the script already states the character's specific appearance/clothing/weapon, you must incorporate it into `description` **exactly as-is** — no "optimizing", "redesigning", or replacing it with a more generic phrasing.

Your task: extract every character who needs to appear on screen (whether or not explicitly named) and produce a professional-grade visual spec sheet at the level of a real film-production bible.

Important: not only extract named characters, but also the following types:
- Characters who appear by reference (e.g. "he", "that man", "the old man") — coin a short identifier for them (e.g. "the man in the memorial photo", "the mysterious elder")
- Characters who appear only via photos, memories, hallucinations, etc. but need visual representation
- Characters who have dialogue or plot impact but are not given a name
- Extras with a distinctive appearance description

When naming unnamed characters, use the most common way the script refers to them or their most salient feature as the identifier."""

CHAR_EXTRACT_STYLE_DETECTION = """═══ Step One — Identify the Visual Style ═══
Identify the style declared or implied in the script:
- "live action" / "photorealistic" / "shot on camera" / "photo-grade" -> describe as real photography or high-end CG cinema, never using any anime aesthetics.
- "anime" / "manga" / "anime" / "manga" -> describe with anime proportions, stylized features, vivid colors.
- "3D CG" / "Pixar" -> describe as a 3D render pipeline.
- "2D cartoon" -> describe as cartoon illustration.
This style must appear in every character's description. A live-action script must never produce anime-style descriptions."""

CHAR_EXTRACT_OUTPUT_FORMAT = """═══ Output Format ═══
JSON object only — no markdown code blocks, no commentary:
{
  "characters": [
    {
      "name": "character name, exactly as in the script",
      "scope": "main" or "guest",
      "description": "complete visual spec — a single paragraph, containing all requirements below",
      "visualHint": "a 2-4 character visual identifier for dialogue labels (e.g. silver-hair-gold-eyes, red-robe-long-hair). Must be recognizable at a glance — focus on the most salient appearance feature.",
      "personality": "2-3 core personality traits that shape posture, expression, and movement",
      "heightCm": "estimated height in cm, e.g. 175. Infer from clues in the script.",
      "bodyType": "slim | average | athletic | heavy | petite | tall",
      "performanceStyle": "performance-style description — movement amplitude (exaggerated/subtle), signature gestures, emotional-expression pattern"
    }
  ],
  "relationships": [
    {
      "characterA": "name of character A, exactly matching a name in `characters`",
      "characterB": "name of character B, exactly matching a name in `characters`",
      "relationType": "ally | enemy | lover | family | mentor | rival | stranger | neutral",
      "description": "a short description of the specific nature of the relationship, e.g. 'master and disciple, also close friends', 'secretly in love but never confessed'"
    }
  ]
}

═══ Relationship-Extraction Rules ═══
- Only extract character pairs with an explicit interaction or implied relationship in the script
- `relationType` must be the closest match from the given options
- Each pair only needs to appear once (A->B, no need to also write B->A)
- If there is no obvious relationship between characters, do not force one
- `description` describes the core of the relationship in a single concise sentence"""

CHAR_EXTRACT_SCOPE_RULES = """═══ Character Classification Rules ═══
- "main": core characters who drive the story, appear in multiple scenes, or are crucial to the plot — protagonists, important supporting roles, key antagonists, and pivotal figures who appear via photo/memory but need visual representation
- "guest": minor/auxiliary characters who appear briefly — passersby, one-off bit parts, unimportant background characters
When unsure, prefer "main". A character with substantial dialogue, plot impact, or a need for visual representation (even just a photo/memorial portrait) is "main".

═══ Full Character Coverage (Hard Constraint) ═══
- **Every named character in the script must appear in the `characters` array** — no omissions, no merging
- Including: named supporting characters who appear only once, characters who appear via memory/photo/memorial portrait, and named characters mentioned in voiceover/narration
- If the script already contains a fixed-format "=== 2. 角色描述 ===" block (the 角色/外貌/服饰/标志特征/气质姿态 five-field format produced by script_generate), you **must** extract every character as-is, without condensing, trimming, or rewriting the character names
- Self-check: after generating, scan the script line by line again to confirm that every character who is given lines via quotes or colons, and every person named in a scene description, is in `characters`"""

CHAR_EXTRACT_DESCRIPTION_REQUIREMENTS = """═══ Description Requirements ═══
Write a dense, precise paragraph covering all of the following. This description will be passed to the image generator exactly as-is — write it in the voice of a professional DP briefing a photographer:

0. Style tag: begin with the art style (e.g. "photorealistic live-action cinematic style, 85mm lens —" or "Japanese anime style —") to anchor the downstream renderer.

1. Physique and bearing: sex, apparent age, sense of height (tall/petite/medium), build (lean/slender/athletic/stocky), natural posture and demeanor.

2. Face — describe as a close-up:
   - Bone structure: face shape, cheekbones, jawline (sharp/soft/angular), brow ridge
   - Eyes: shape (almond/round/phoenix/monolid), size, iris color (be specific, e.g. "storm gray", "amber brown", "deep ink black"), lash density
   - Nose: bridge height, tip shape, nostril width
   - Lips: thickness, cupid's-bow curve, natural resting expression
   - Skin: describe tone with precise modifiers (e.g. "cool porcelain white", "warm honey gold", "deep sandalwood with blue undertone"), texture (translucent/matte/rough), spots/moles, etc.
   - Overall: state the attractiveness positioning directly — model-tier beauty, rugged handsome, or the approachable girl/boy-next-door?

3. Hair: exact color (hue + undertone, e.g. "blue-black with deep indigo sheen"), length relative to the body, texture (straight/big waves/tight curls), style (how it lifts, falls, moves), hair ornaments.

4. Clothing — main look (full outfit breakdown):
   - Top: style, cut, material (e.g. "slim-fit lime wool mandarin-collar coat"), color
   - Bottom: pants/skirt type, material, color
   - Footwear: style, material
   - Outerwear/armor: if any, describe layer by layer
   - Accessories: jewelry (metal, gems, style), belt, bag, gloves, hat — be specific

5. Weapons and equipment (if any):
   - Melee weapon: blade length, blade shape, guard style, grip-wrap material, surface finish (bluing/polish/engraving), how it's carried
   - Ranged weapon: bow/gun type, surface finish, modification details
   - Armor: material (plate/mail/leather), surface finish, insignia or engravings
   - Other equipment: describe function and appearance

6. Signature traits: scars (location, shape, old/new), tattoos (design, location), glasses (frame type, lens tint), cybernetic prosthetics, non-human features (ears, wings, horns, tail) — describe the precise visual look.

7. Character color palette: list 3-5 primary colors that define this character's visual identity (e.g. "deep red, worn gold, charcoal black").

[Example]
赛博朋克风格，35mm广角镜头低角度——男，约30岁，190cm精瘦高挑身形，站立姿态，双脚与肩同宽微微前后错开，重心偏右腿，脊背微弓前倾，左手插在夹克口袋，右手自然垂在身侧。棱角分明的长脸，颧骨高耸投下锐利阴影，下颌线锋利笔直，眉骨突出。狭长上挑的丹凤眼，左眼瞳色自然灰绿、右眼为机械义眼散发幽蓝冷光，睫毛稀疏。高挺鹰钩鼻，鼻尖略下弯，鼻翼窄。薄唇苍白，唇角自然下垂。肤色病态苍白偏冷青调，质感哑光粗粝，左颊从眼角到嘴角一道细长的银色机械缝合疤痕，沿疤痕嵌有微型蓝色LED指示灯。阴郁危险的暗夜猎手气质。头发铂银白色带荧光紫挑染，右侧剃至3mm露出头皮上的电路纹身，左侧长发遮住半边脸垂至下巴，发梢参差不齐。上身破旧的哑光黑色合成皮夹克，立领，左肩焊接一块钛合金护甲片，内搭深灰色高科技速干背心，胸口印有褪色的红色骷髅标志。下身黑色工装机能裤，膝盖处缝有凯夫拉补丁，裤腿束入小腿处。脚穿磨损严重的黑色高帮军靴，鞋底加厚，鞋舌外翻。左前臂从手肘到手腕整段替换为钛合金机械义肢，关节处露出液压管线和微型齿轮，指尖是碳纤维材质。右手无名指戴一枚氧化发黑的钨钢戒指。腰后别一把折叠式等离子短刀，刀柄缠绕磨旧的红色伞绳。角色色彩调色板：哑光黑、铂银白、荧光紫、幽蓝冷光、锈红。"""

CHAR_EXTRACT_WRITING_RULES = f"""═══ Writing Rules ═══
- A single continuous paragraph — no bullet points or line breaks inside the `description` field
- Specific enough that two different AI image generators would produce recognizably the same character
- Use precise color names: not "red" but "blood red" or "rose pink"
- Attractiveness matters — if the script implies the character is attractive, write genuinely stunning beauty. Use the professional vocabulary of high-end fashion photography and film casting.
- For non-human characters, describe their distinctive features with the same anatomical precision

═══ Layered Posture Writing (Key — Downstream Generates a Four-View Reference Sheet) ═══

**Top-level rule**: downstream uses the `description` field to generate a "four-view character reference sheet" (front / 3-4 side / side / back), so the posture in `description` **must be a neutral standing full-body pose**, not a specific in-story action moment.

[Posture in the `description` field — must strictly follow these standards]
- **Must be standing**: standing pose / natural full-body stand / standing facing the viewer — no "crouching", "sitting", "kneeling", "prone", "leaping" or other non-standing poses
- **Foot position**: standing naturally with feet shoulder-width apart / standing with feet together (only if the character is extremely prim)
- **Body orientation**: facing the viewer head-on (the default pose for the front view of a four-view sheet)
- **Arms and hands**: hanging naturally at the sides / one hand holding a weapon and the other hanging naturally — no dramatic actions like "hands clasped to the chest", "hugging knees", "hands braced on the ground"
- **Expression**: calm neutral or a micro-expression — no strong-emotion expressions like "looking up in terror", "laughing hard", "weeping"
- **No abstract temperament words**: do not just write "timid", "aloof", "elegant" — but, while keeping a neutral standing pose, convey temperament through posture detail (e.g. "shoulders slightly hunched forward, head slightly lowered" conveys timidity; "back straight, hands clasped behind" conveys haughtiness)

[Signature Poses/Actions — write to the `performanceStyle` field]
The character's signature in-story action (e.g. "crouching and gripping an iron hoop while looking up", "arms crossed, sneering", "drawing a sword") **must not go into `description`**; write it in the `performanceStyle` field, e.g.:
- performanceStyle: "a common action is crouching down and curling up, both hands tightly gripping the iron hoop carried on their person and holding it to the chest while looking up at the speaker; small movement amplitude, frequently lowering the head, speaking in a voice as thin as a mosquito's"

This way, when downstream generates the shot breakdown, the LLM can automatically apply these signature actions to each shot's motionScript, while the character reference sheet itself stays neutral standing, reusable and consistent.

[Layered-Posture Syntax Example — demonstrates structure only, do not copy as content; for the real character, rewrite strictly per the script]

[BAD] pattern (polluting `description` with a specific in-story action):
description: "…[crouch/kneel/leap/hugging knees/hands braced on ground and other dramatic actions]…"

[GOOD] pattern:
description: "…[neutral standing pose + foot position + body orientation + arm position + micro-expression]…"
performanceStyle: "signature action: [the character's common in-story pose/action/emotional-expression style]"

[Key Reminder — Prevent Example Contamination]
The above is only a **syntax-structure example**. You must rewrite `description` entirely based on the character identity, sex, age, appearance, and clothing in the [original script text], and absolutely never copy character setup (age/appearance/clothing/posture wording, etc.) from any example. Your output must correspond one-to-one with the actual characters in the script.

{physics_realism_block()}"""

CHAR_EXTRACT_LANGUAGE_RULES = """[Critical Language Rule] All fields must use the same language as the script. Chinese script -> Chinese output. English script -> English output. Character names must exactly match the script.

Return the JSON array only. No markdown. No commentary."""


def _build_character_extract(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    return "\n".join([
        r("role_definition"),
        "",
        r("style_detection"),
        "",
        r("output_format"),
        "",
        r("scope_rules"),
        "",
        r("description_requirements"),
        "",
        r("writing_rules"),
        "",
        r("language_rules"),
    ])


characterExtractDef = _make_def(
    "character_extract",
    "promptTemplates.prompts.characterExtract",
    "promptTemplates.prompts.characterExtractDesc",
    "character",
    [
        slot("role_definition", CHAR_EXTRACT_ROLE_DEFINITION, True),
        slot("style_detection", CHAR_EXTRACT_STYLE_DETECTION, True),
        slot("output_format", CHAR_EXTRACT_OUTPUT_FORMAT, False),
        slot("scope_rules", CHAR_EXTRACT_SCOPE_RULES, True),
        slot("description_requirements", CHAR_EXTRACT_DESCRIPTION_REQUIREMENTS, True),
        slot("writing_rules", CHAR_EXTRACT_WRITING_RULES, True),
        slot("language_rules", CHAR_EXTRACT_LANGUAGE_RULES, False),
    ],
    _build_character_extract,
)


# ─── 5. import_character_extract ────────────────────────

IMPORT_CHAR_ROLE_DEFINITION = """You are a senior character designer, director of photography, and art director. Your task is to extract all named characters from the given text, estimate their frequency, and produce a professional-grade visual spec sheet for each."""

IMPORT_CHAR_EXTRACTION_RULES = """Rules:
1. Extract every named character in the text
2. Count roughly how many times each character appears/is mentioned
3. Characters mentioned more than twice are likely main characters
4. Merge obvious aliases (e.g. "小明" and "明哥" refer to the same person)

═══ Step One — Identify the Visual Style ═══
Identify the style declared or implied in the text:
- "live action" / "photorealistic" / "shot on camera" / historical subject -> describe in a photorealistic cinematic style, using no anime aesthetics.
- "anime" / "manga" / "anime" / "manga" -> describe with anime proportions and stylized features.
- "3D CG" / "Pixar" -> describe as a 3D render.
- If no style is specified, infer from the content (historical text -> photorealistic historical-drama style).

═══ Description Requirements ═══
The "description" field must be a dense paragraph covering all of the following, written in the voice of a professional DP:

0. Style tag: begin with the art style (e.g. "cinematic photorealistic historical-drama style, no filter, 85mm close-up —")
1. [Physique]: sex, apparent age, height/build, posture, bearing
2. [Face]: face shape, jawline, brow ridge, eye shape/iris color, nose shape, lips, skin tone (described precisely), skin texture, attractiveness positioning
3. [Hair]: exact color, length, style, hair ornaments
4. [Clothing]: full outfit breakdown — top, bottom, footwear, outerwear, accessories, noting material and color
5. [Weapons/Equipment] (if any): detailed description of weapons, armor, equipment
6. [Color Palette]: 3-5 primary colors that define this character's visual identity

[Example]
电影级写实历史正剧风格，无滤镜，85mm镜头特写——男，约45岁，身高约178cm，体型魁梧厚实但不臃肿，站姿沉稳如山，双肩微微后展透出帝王威压。方正国字脸，颧骨高耸，下颌线刚硬如刀削，眉骨隆起投下深邃阴影。丹凤眼窄长上挑，瞳色极深近乎纯黑，目光阴鸷锐利如鹰隼。鼻梁高挺笔直，鼻尖略呈鹰钩，鼻翼不宽。薄唇紧抿，唇线下弯，自然流露出冷峻威严。肤色深麦色暖调，面部肌理粗粝，法令纹深刻，额角有隐约的岁月痕迹。属于令人畏惧的帝王级气场。花白短髯修剪齐整，头戴十二旒冕冠，黑色旒珠垂落遮挡部分面容。身穿明黄色龙袍，五爪金龙盘踞前胸，金线满绣云纹海水江崖纹，袖口镶赤金色回纹宽边。腰系白玉带钩嵌红宝石的御带。脚蹬黑色缎面朝靴。角色色彩调色板：明黄、赤金、纯黑、白玉色、深麦色。

═══ Visual Hint ═══
The "visualHint" field must be a 2-4 character appearance tag for instant visual recognition (e.g. "龙袍金冠阴沉脸", "大红直身佩刀"). It must describe appearance, not action.

[Critical Language Rule] All output fields must use the same language as the original text."""

IMPORT_CHAR_OUTPUT_FORMAT = """Output format — JSON object only, no markdown code blocks, no commentary:
{
  "characters": [
    {
      "name": "character name, matching the text",
      "frequency": 5,
      "description": "complete visual spec — a dense paragraph following all requirements above",
      "visualHint": "a 2-4 character appearance identifier"
    }
  ],
  "relationships": [
    {
      "characterA": "name of character A",
      "characterB": "name of character B",
      "relationType": "ally | enemy | lover | family | mentor | rival | stranger | neutral",
      "description": "a short relationship description"
    }
  ]
}

Return the JSON object only. No markdown. No commentary."""


def _build_import_character_extract(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    return "\n".join([
        r("role_definition"),
        "",
        r("extraction_rules"),
        "",
        r("output_format"),
    ])


importCharacterExtractDef = _make_def(
    "import_character_extract",
    "promptTemplates.prompts.importCharacterExtract",
    "promptTemplates.prompts.importCharacterExtractDesc",
    "character",
    [
        slot("role_definition", IMPORT_CHAR_ROLE_DEFINITION, True),
        slot("extraction_rules", IMPORT_CHAR_EXTRACTION_RULES, True),
        slot("output_format", IMPORT_CHAR_OUTPUT_FORMAT, False),
    ],
    _build_import_character_extract,
)


# ─── 6. character_image ─────────────────────────────────

CHAR_IMAGE_STYLE_MATCHING = f"""=== Critical: Art-Style Matching (Highest Priority) ===
Read the character description below carefully. It specifies or implies an art style (e.g. anime, manga, photorealistic, cartoon, watercolor, pixel art, oil painting, etc.).
You must match that art style precisely. Do not default to photorealistic. Do not override the style in the description.
- If the description mentions "anime"/"manga" -> produce an anime/manga-style illustration
- If the description mentions "photorealistic"/"live action"/"photorealistic" -> produce a photorealistic render
- If the description implies another style -> follow that style faithfully
- If no style is mentioned at all -> infer the most fitting style from the character's background and type

{theme_style_mapping_block()}

**Writing language**: describe each part in natural Chinese prose; do not use weight syntax "（xx：1.99）", do not use structured tags "Scene:" "Style:" — Seedance / Jimeng-family image models understand natural language best."""

CHAR_IMAGE_FACE_DETAIL = """=== Face — High Precision ===
Render the face at high precision appropriate to the chosen art style:
- Clear, consistent facial features: bone structure, eye shape, nose shape, mouth shape — all matching the appearance in the description
- Eyes: expressive, richly detailed, with catchlights and a sense of depth — adjust to the art style (anime-style eyes for anime, fine iris detail for photorealistic)
- Hair: clear volume, color, and sense of motion, rendered in the manner suited to the style (individual strands for photorealistic, large hair clumps with highlight bands for anime)
- Skin: rendered to fit the style — smooth cel shading for anime, pore-level detail for photorealistic
- Overall: the face should be recognizable and memorable, with strong visual features"""

CHAR_IMAGE_FOUR_VIEW_LAYOUT = """=== Four-View Layout (Must Be Strictly Followed — This Is the Core Output Form of a Character Design Sheet) ===
**Mandatory four-view output**: the final image must contain four separate viewpoints, arranged horizontally left to right on a pure white canvas. **Do not output a single-view portrait, do not draw only two or three views, do not place the character in a scene** — this is a professional character turnaround sheet (three-view / four-view).

Precise requirements for the four views (left to right):
1. **Front (Front / 0°)** — the character faces the viewer, shoulders parallel to the frame, arms relaxed and hanging naturally at the sides, feet shoulder-width apart standing naturally, showing the full front of the clothing, belt, weapon fittings, chest accessories. Calm neutral expression, to ease later derivation.
2. **Three-quarter view (3/4 View / ~45°)** — the character rotated about 45° to the right, showing the facial depth, cheekbone and nose-bridge contour, and the layering of the front-side clothing structure and cape/robe.
3. **Profile (Profile / 90°)** — a standard 90° orientation facing the right of the frame, clearly showing the nose-to-chin contour line, the side volume of the hair, the position of the weapon strap, the cape hem, and the side of the boots.
4. **Back (Back / 180°)** — fully facing away from the viewer, showing the back-of-head hairstyle and ornaments, the back pattern/embroidery of the clothing, the full cape/cloak, and back equipment (scabbard, quiver, backpack, etc.).

**Composition and layout requirements**:
- The image's aspect ratio should be 16:9 or wider, to give the four views ample display space
- The canvas background must be **pure white and textureless**, with appropriate spacing between the four views so they do not overlap
- The four views must be **aligned at the top of the head, at the waistline, and at the soles**, neat and uniform like a professional design sheet
- Uniform shot size — all in a standing full-body view (from top of head to soles, including shoes/boots), to fully show clothing and posture
- If the character holds a weapon, the front view clearly shows the grip, and the other views should show at least part of the weapon"""

CHAR_IMAGE_LIGHTING_RENDERING = """=== Lighting and Rendering ===
- Clean, professional three-point lighting: key light entering from the upper front at about 45°, fill light softening shadows from the opposite side, and a rim light behind that cleanly "cuts" the character out from the pure white background
- The quality of light matches the style — soft studio light for photorealistic, clear cel-shaded light/shadow boundaries for anime, subtle volumetric light to enhance atmosphere for xianxia
- Pure white background with no gradient, no texture, no ground shadow (or only a very faint contact shadow), ensuring the character is cleanly separated for later cutout reuse
- **The four views must keep exactly the same light direction and color temperature**, avoiding a "daylight front / dusk side" discontinuity
- Pursue the highest rendering quality within the chosen art style: material detail, fabric folds, metal reflections, and skin texture must all meet the technical standards of the style"""

CHAR_IMAGE_CONSISTENCY_RULES = """=== Four-View Consistency (The Lifeline of the Downstream Pipeline) ===
This reference sheet is reused as the authoritative reference for all subsequent shot generation — any inconsistency will be amplified into continuity errors in the finished film. Enforce strictly:
- **Identity consistency**: the four views must be the same person — same facial bone structure, same height proportion, same feature placement, same skin tone
- **Clothing consistency**: every garment, accessory, belt buckle, button, embroidery, and pocket position aligns one-to-one, with exactly the same color values (do not make the front deep blue and the back light blue)
- **Hair consistency**: hair color, volume, length, bang shape, ornament position — the four views may show different sides, but must be the same hairstyle from different angles
- **Weapon/equipment consistency**: the weapon's color, length, grip style, and mount position — if it hangs on the left of the waist in front, it must be on the left of the waist in back (which is the right side seen from behind)
- **Physique consistency**: shoulder width, waist size, and leg-length proportion align view by view — do not make the front slender and the back stocky
- **Expression and temperament consistency**: all four views keep the same neutral/micro-expression, conveying the same personality (cold / gentle / aloof), with no mix of smiling and angry faces"""

# The name_label slot is locked because it is dynamically generated from the character name
CHAR_IMAGE_NAME_LABEL = """=== Character Name Label ===
{{NAME_LABEL_PLACEHOLDER}}"""


def _build_character_image(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    params = params or {}
    character_name = params.get("characterName")
    description = params.get("description") or ""

    # Resolve name label dynamically
    if character_name:
        name_label_text = (
            f'=== Character Name Label ===\n'
            f'Display the character name "{character_name}" centered below the four-view layout. '
            f'Use a modern sans-serif font, dark text on a white background, center-aligned. '
            f'The name is clearly legible, in a professional design-sheet style.'
        )
    else:
        name_label_text = "=== Character Name Label ===\nNo character name label needed."

    return "\n".join([
        "Character four-view reference design sheet — a professional character design document.",
        "**The final output must be a single horizontal-layout design sheet containing the four views \"front / three-quarter / profile / back\"**, on a pure white background, with the four views aligned at the top of the head / waistline / soles. It is strictly forbidden to output a single-view portrait, a scene illustration, or an unfinished sheet with only two or three views.",
        "",
        r("style_matching"),
        "",
        "=== Character Description ===",
        f"{('Name: ' + character_name + chr(10)) if character_name else ''}{description}",
        "",
        r("face_detail"),
        "",
        "=== Weapons and Equipment (if any) ===",
        "- Render all weapons, armor, and equipment in the same art style as the character",
        "- Show material detail suited to the style: signs of wear for photorealistic, clean stylized lines for anime/cartoon",
        "- All equipment must be proportional to the character's body",
        "",
        r("four_view_layout"),
        "",
        r("lighting_rendering"),
        "",
        r("consistency_rules"),
        "",
        name_label_text,
        "",
        "=== Final Output Standard ===",
        "A professional character-design reference sheet. Achieve the highest quality within the chosen art style. Zero AI artifacts, zero inconsistency between views. This is the sole authoritative reference — every subsequently generated image must precisely reproduce this character in this style.",
    ])


characterImageDef = _make_def(
    "character_image",
    "promptTemplates.prompts.characterImage",
    "promptTemplates.prompts.characterImageDesc",
    "character",
    [
        slot("style_matching", CHAR_IMAGE_STYLE_MATCHING, True),
        slot("face_detail", CHAR_IMAGE_FACE_DETAIL, True),
        slot("four_view_layout", CHAR_IMAGE_FOUR_VIEW_LAYOUT, True),
        slot("lighting_rendering", CHAR_IMAGE_LIGHTING_RENDERING, True),
        slot("consistency_rules", CHAR_IMAGE_CONSISTENCY_RULES, True),
        slot("name_label", CHAR_IMAGE_NAME_LABEL, False),
    ],
    _build_character_image,
)


# ─── 7. shot_split ──────────────────────────────────────

SHOT_SPLIT_ROLE_DEFINITION = """You are an experienced storyboard director and director of photography, skilled at animated-short production. The shot lists you plan are visually dynamic, narratively efficient, and optimized for an AI video-generation pipeline (first frame -> last frame -> interpolated video).

Your task: break the script into a precise shot list, where each shot becomes a {{MIN_DURATION}}-{{MAX_DURATION}} second AI-generated video clip."""

SHOT_SPLIT_FIDELITY_RULES = """=== Script Fidelity (Highest Priority — this rule takes precedence over all others) ===

You are the director, not the editor. **No condensing, no compressing, no omitting** any narrative content in the script. Your duty is to fully "translate" the script into shot language, not to "condense" it into a summary.

[!!] **Hard minimum-word constraints on sceneDescription and motionScript**:
- **sceneDescription**: each scene's description must be **at least 150 Chinese characters**, and must include all the environment/prop/atmosphere detail of that scene from the script; do not write an empty shell like "凌霄宝殿废墟". If N specific environmental elements appear in the script (architecture, props, weather, sound, smell, light and shadow), sceneDescription must have N corresponding descriptions.
- **motionScript**: no single shot may exceed 15 seconds. Each shot's motionScript must narrate in time segments "0-3s / 4-6s / ...", each segment a dense 50-80 character description. If a passage of the script is rich in content, you must **split it into multiple shots** rather than compress it into one long shot.
- **Rejected answer**: if you write a sceneDescription shorter than 150 characters, or a motionScript that crams 3 distinct actions into the same timestamp, the entire output is judged non-compliant and must be rewritten.

[!!] **Hard constraint on shot count**:
- For every distinct visual beat in the script (action, transition, dialogue exchange, emotional change), there **must** be a corresponding separate shot.
- It is strictly forbidden to compress a 4-beat sequence like "character enters + walks to target + performs action + reacts" into a single 12-second shot. Such a sequence is at least 3-4 shots.
- **Mathematical lower bound**: if a passage of the script has K action verbs or K dialogue lines, the shot count must be >= K. Before generating, mentally list the action verbs in the script and confirm the shot count is not fewer than the verb count.
- If unsure how many shots to split into, **split more by default** — the finer the granularity, the more accurate the downstream image/video generation.
- One shot = one atomic beat. Multiple beats = must split.

[Content That Must Be 100% Covered]
Read the script line by line; each of the following must have an explicit visual landing point in the output shot list:

1. **Every event/action**: every specific action mentioned in the script ("she pushes the door open", "he lights a cigarette", "the teacup on the table suddenly tips over") must appear in some time segment of some shot's motionScript — not a "similar action", but the original action itself.
2. **Every dialogue line**: every line in the script must go into some shot's `dialogues` array; no omitting or rewriting. A too-long line may span shots, but must not be deleted.
3. **Every emotional beat**: emotional turns in the script (hesitation -> resolve, anger -> breakdown, calm -> surprise) must be embodied as a distinct beat in motionScript, corresponding to at least one time segment's micro-expression/body change.
4. **Every specific object/prop**: named props, clothing details, and environmental objects the script mentions ("that worn leather briefcase", "the yellowed family photo on the wall", "the half-cup of cold coffee") must appear in at least one of startFrame/endFrame/sceneDescription.
5. **Every specific scene/location**: whenever the script switches to a new scene, a new shot must begin; multiple narrative beats within the same scene must also be split into multiple shots.
6. **Space-time markers**: the time written in the script ("two a.m.", "a clear morning after rain"), weather, season, and specific landmarks — must go into sceneDescription.
7. **Subtext and atmosphere words**: atmosphere description in the script ("the air froze", "so oppressive it was hard to breathe", "the cicadas outside suddenly went silent") must be turned into concrete visual/auditory detail in motionScript or sceneDescription.

[Self-Check Checklist — after generating the shot list, check against the script once]
- [ ] Did every narrative passage of the script produce at least 1 shot?
- [ ] Did every dialogue line in the script go into some shot's `dialogues`?
- [ ] Does every named object the script mentions appear in some frame description?
- [ ] Can the script's emotional turns each be pointed out in a motionScript time segment?
- [ ] Did you avoid cramming multiple distinct events into the same shot?
If any item fails, **you must add shots or expand descriptions** — do not lower the bar.

[Counter-Example — Forbidden Condensation]
Original script:
> 林晓月推开吱呀作响的木门，门外的雨还在下。她愣了一下，抬手摸了摸口袋里那封没寄出的信，嘴角牵起一丝自嘲的笑。远处传来卖馄饨老人沙哑的吆喝声。

[BAD] Over-condensed: "林晓月推门出去，雨中露出苦笑。" (Lost: the creaking door sound, the letter-touching action, the self-mocking emotional turn, the distant vendor's call, and the letter itself as a symbolic object)
[GOOD] Correct expansion: split into 1-2 shots, with the motionScript explicitly stating "push open the creaking wooden door -> freeze in the curtain of rain -> right hand reaches into the trench-coat pocket and finds that unsent letter -> fingertip pauses for a moment -> the corner of the mouth curves into a self-mocking arc", the sceneDescription writing "late-night rainy alley, the hoarse call of an old wonton vendor drifting from afar", and the letter appearing as a key prop in the composition of the startFrame or endFrame.

[Dialogue-Coverage Principle — Every Shot Must Have Sound]
- **Every shot must have dialogues**. A video with no lines is like a silent play; the audience drifts off.
- If the script has explicit lines for that passage, use them directly.
- If the script has no explicit lines, you must **add reasonable lines** based on the plot and character personality, including but not limited to:
  * The character's improvised reaction ("什么？！", "不可能...", "终于来了")
  * Inner monologue (set offscreen: true, like narration)
  * Short dialogue between characters ("你看到了吗？", "小心！")
  * Environment-related muttering ("这里好冷", "有人来了")
- The only exception: a pure establishing shot with no characters, in which case add narration or a voiceover.
- Keep each line short and punchy, 1-2 sentences, no long speeches.

[Shot-Count Principle]
- More rather than fewer. If a passage is information-dense, 3-5 shots is normal.
- One shot carries one core beat. Multiple beats must be split.
- The only compression license: a pure scene transition / time jump ("three days later"), for which one brief transition shot suffices.

[Combat/Duel Scene Mandatory Rules]
If a combat/duel sequence appears in the script (identified by these signals: "大战/对决/交手/厮杀/VS" in the title or character relationships; weapon/move/attack verbs in the plot; two hostile parties present at once in the `characters` list) — you must split shots by these rules:

1. **Give both sides shots**: in a combat sequence, both hostile sides must have shots as **active attackers**; one side must not merely "dodge/block/sigh/raise a hand to suppress" the whole time. It is strictly forbidden to give one side 5 shots of attacking and the other only 1 shot of "raising a hand" — such a lopsided distribution.

2. **Attack-defense alternating beat template** (a combat passage must contain the following shot types):
   - **A charges up/strikes**: the moment the body exerts force and the weapon swings out
   - **B blocks/dodges**: the body's reaction, weapons clashing
   - **Impact**: a wide shot of weapons colliding, shockwaves, environmental destruction
   - **B counterattacks**: strikes back seizing the momentum
   - **A is hit/dodges**: knocked back / flesh torn / armor shattered
   - **Pull back to a wide shot**: showing the whole battlefield's state of destruction

3. **One exchange = multiple shots**: one complete attack-defense exchange (A attacks -> B defends -> impact -> B counters -> A defends) must be split into at least **4-6 shots**. Do not compress one exchange into one shot.

4. **Do not substitute "mental space/epiphany" for real combat**: if a "mental world/inner drama/epiphany" passage appears in the script, it may be kept but **must not exceed 30% of the total combat shot count**. The audience came for an action film, not to watch meditation.

5. **If the script itself is short on combat**: **add** specific combat-action detail in sceneDescription / motionScript — because the script author may have written "the two exchanged thirty rounds" very tersely, and as the storyboard director you have the duty to expand it into a 6-10 shot attack-defense sequence. This is not deviating from the script; it is the normal work of "translating narrative language into shot language"."""

SHOT_SPLIT_OUTPUT_FORMAT_TEMPLATE = """Output a JSON array (only the shared shot metadata; downstream will use the same metadata to separately generate the first/last frames and reference images):
[
  {
    "sequence": 1,
    "sceneDescription": "scene/environment description — must preserve all environmental elements of the scene from the script (set, architecture, props, weather, time, sound, smell, light and shadow, atmosphere), >= 150 characters",
    "motionScript": "time-segment narration, split as 0-3s / 4-6s / ..., each segment 50-80 characters, describing all actions and emotional beats in this shot",
    "videoScript": "30-60 character Seedance-style prose that drives the video-generation model",
    "duration": {{MIN_DURATION}}-{{MAX_DURATION}},
    "dialogues": [
      { "character": "exact character name", "text": "the original line (kept verbatim, including filler words and punctuation)" }
    ],
    "cameraDirection": "static / dolly in / pan left / push in / orbit left / ... English keyword",
    "characters": ["names of characters appearing in this shot (exactly matching the character list)"]
  }
]"""

SHOT_SPLIT_START_END_FRAME_RULES = """=== First-Frame and Last-Frame Requirements (Key — Directly Drives Image Generation) ===
Each frame must be a self-sufficient image-generation prompt, containing:
- Composition: frame layout — foreground/midground/background layers, character position (left/center/right, rule of thirds), depth of field
- Character: use the exact character name; describe the current pose, expression, action, clothing (matching the character design sheet)
- Camera: shot size (extreme close-up/close-up/medium/wide/extreme wide), angle (eye level/low angle/high angle/bird's-eye/Dutch angle)
- Light: direction, quality, color temperature — for this frame's specific moment
- Do not include dialogue text in the first or last frame

=== First-Frame-Specific Rules ===
- Show the initial state before the action begins
- The character is at the starting position, with an opening expression
- The camera is at the starting position/composition

=== Last-Frame-Specific Rules ===
- Show the ending state after the action completes
- The character has moved to a new position, with an expression reflecting the result of the action
- The camera is at the final position/composition (after the cameraDirection movement)
- Must be visually stable (not in the middle of motion) — this frame is reused as the opening reference for the next shot
- The composition must stand as an independent image

[Example]
startFrame: "全景，三分法构图。画面左侧三分之一处，林晓月（米白衬衫、黑色长直发）骑着旧自行车从巷口驶入，车篮里的葱叶在晚风中微微摆动。弄堂两侧晾衣竿上的花色被单在暖橘色夕阳中轻轻飘荡。青石板路面反射着金色余晖，远处弄堂尽头隐约可见几户人家的灯光。自然光线从画面右上方45度照入，色温偏暖。"
endFrame: "中景偏近，林晓月在画面中央偏右位置停下自行车，左脚点地，右手拨开眼前垂落的花被单，微微喘气的嘴角带着一丝无奈的笑意。背景中弄堂深处的赵东明（深灰工装夹克）的模糊身影倚在门框上，作为画面的视觉锚点。夕阳从背后打出暖色轮廓光。\""""

SHOT_SPLIT_MOTION_SCRIPT_RULES = """=== motionScript Requirements ===
- motionScript is the full expansion of the script's beats, not an action summary. Every action, every emotional change, and every mentioned object interaction in the script passage covered by this shot must explicitly appear in some time segment.
- Narrate in time segments: "0-2s: [action]. 2-4s: [action]. 4-6s: [action]. ..."
- Strict rule: each time segment is at most 3 seconds. A 10-second shot = at least 4 segments. Never write a segment longer than 3 seconds.
- Beat-mapping requirement: if the script passage has N narrative beats (action/emotional turn/object interaction), the number of motionScript time segments must be >= N. Do not cram multiple beats into one segment.
- Each segment is a dense long sentence (50-80 characters) that simultaneously weaves four layers:
  • Character: precise physical movement — knuckles whitening, tendons taut, pupils contracting, breath held, jaw clenched; specify speed and force
  • Environment: the world's reaction — cracks spidering across the ground, a lamppost bending, sparks pouring, black smoke rolling, debris trajectories
  • Camera: precise shot size + movement + speed — "camera plunges to a ground-level ultra-wide then rapidly rises" / "camera holds an extreme close-up then whips right"
  • Physics/atmosphere: material detail — the sound of shattering metal, ripples of shockwave in the air, heat distortion, color-temperature shift, particle behavior

[Example]
- Bad (too vague, too long a span): "0-6s: the iron beast swings its claw and destroys the street. Camera pushes in."
- Good (specific, at most 3s): "0-2s: the iron beast's right foreleg slams down with a bone-shaking thud, spiderweb cracks radiating six meters outward from the impact, three sets of mechanical claw-teeth rising simultaneously trailing hydraulic white mist, its sensor eye pulsing dark red; camera slowly tilts up in a low-angle wide. 2-4s: the foreclaw sweeps horizontally at subsonic speed, cutting a burst of blue-white sparks through the mid-section of a lamppost, the severed upper half spinning off at a 45° angle, asphalt chunks and metal shards scattering downward; camera holds a medium then snaps in. 4-6s: black smoke gushing from a ruptured pipe rolls and fills the frame over the heat shockwave, debris still falling, the iron beast's sensor eye locking onto the next target with a shrill hydraulic screech; camera slowly orbits right in a low angle, finally freezing on the iron beast's silhouette.\""""

SHOT_SPLIT_VIDEO_SCRIPT_RULES = """=== videoScript Requirements (Seedance 2.0 Style) ===
- Purpose: the primary input to the video-generation model — drives all motion; must be natural Seedance-style prose.
- Forbidden: structured tags like Scene:/Action:/Performance:/Detail:; weight syntax "（xx：1.5）"; dialogue text (goes in the `dialogues` array).
- Language: same as the script.

Format tiered by shot duration:

**4-8 second short shots**: 30-60 characters, a single fluent prose paragraph
  • Begin with "character name (brief visual identifier in parentheses)"
  • One core action + one camera movement + one atmosphere/emotion detail
  • The camera movement is embedded at the end of the sentence, using concrete words ("camera slowly pushes in" / "low-angle tilt up" / "locked-off shot" / "orbiting pan")

**9-12 second medium shots**: 60-120 characters, using 2-3 timestamp segments, e.g. "0-4s: ... 5-8s: ... 9-12s: ..."

**13-15 second long shots**: 120-200 characters, mandatorily using 3-4 timestamp segments "0-3s / 4-8s / 9-12s / 13-15s", each a dense long sentence weaving four layers:
  • Character: precise physical movement (gripping, turning, staggering, breath pausing), speed and force
  • Environment: the world's reaction (robe hems flying, light patches sweeping, leaves lifting, debris trajectories)
  • Camera: concrete shot size + movement + speed ("low-angle wide slowly tilting up" / "orbiting pan quick cut" / "freeze slow-motion")
  • Physics/atmosphere: material detail, light/shadow color temperature, sound cues

[Example — 8-second prose]
陆云舟（月白长袍，玉簪束发）从棋盘上缓缓抬眼，头微侧转向斜后方，嘴角牵出一抹含笑弧度，月白纱衣随晨风轻轻摆动，镜头从中景缓慢推近至近景特写。

[Example — 15-second timestamp segments]
15 秒仙侠高燃战斗镜头，金红暖色调。0-3秒：低角度特写陆云舟（月白长袍、玉簪束发）双手紧握雷纹巨剑，剑刃赤红电光持续爆闪，衣摆被热浪吹得猎猎翻飞，远处魔兵嘶吼冲锋，镜头低角度缓缓上摇。4-8秒：环绕摇镜快切，陆云舟旋身挥剑，剑刃撕裂空气迸射红色冲击波，前排魔兵被击飞碎裂成灰烬粒子四散，镜头从环绕切到猛推。9-12秒：仰拍拉远定格慢放，陆云舟跃起腾空，剑刃凝聚巨型雷光电弧劈向魔兵群，金红粒子向四周爆散。13-15秒：缓推特写陆云舟落地收剑姿态，衣摆余波微动，冷峻侧脸定格，背景火光渐弱。

[Example]
- Bad (has tags): "Scene: 湖畔垂柳。Action: 陆云舟落棋。Performance: 神情淡然。"
- Bad (separate camera line): "陆云舟落棋。Camera: dolly out。"
- Good (prose, ~45 chars):
  "陆云舟（月白长袍，玉簪束发）从棋盘上缓缓抬眼，头微侧转向斜后方，嘴角牵出一抹含笑弧度，月白纱衣随晨风轻轻摆动，镜头缓慢推近。"
- Good (English, ~45 words):
  "The Veteran (black helmet, calm eyes) leans forward over the steering wheel, one hand adjusting the visor with practiced ease, the rain-blurred dashboard lights casting green on his face as the camera slowly pushes in."

=== sceneDescription Requirements ===
- The environmental context shared by the two frames — includes environment detail **and** the narrative environmental elements from the script
- Must include: set, architecture, specific props (especially symbolic objects named in the script), weather, time (to the specific moment), season
- Must include: lighting scheme (key/fill/rim, direction, quality, color temperature), color tone
- Must include: the script's atmospheric mood and subtext turned into concrete environmental detail ("the air froze" -> "the cicadas outside abruptly stop, the ceiling fan droning"; "oppressive" -> "the curtains sealed against light, the desktop lit only by the yellow glow of a single lamp")
- Must include: off-screen environmental elements mentioned in the script (distant sounds, scent hints, off-frame movement), written in with phrasing like "from afar comes..." / "the air is filled with..."
- Do not include the character's specific actions or poses — those go in startFrame/endFrame/motionScript (but you may state the fact that a character is already present)

[Example]
sceneDescription: "老城区弄堂黄昏。窄长的青石板巷道两侧是斑驳的灰白色砖墙，二层木阳台上晾满花色被单。弄堂尽头可见一棵老梧桐树的枝叶剪影。自然光为落日暖橘色调，从巷口方向斜照入，在石板路面形成长长的影子。色彩基调：暖橘、灰白、深绿、旧木棕。氛围：烟火气十足的市井温情，带有时光流逝的怀旧感。\""""

SHOT_SPLIT_CAMERA_DIRECTIONS = """Camera-movement directives (for the cameraDirection field only):

**Important: the cameraDirection field is technical metadata; its value must be one of the English keywords in the list below** (the downstream video generator recognizes shot type by the English). Meanwhile, when describing the camera in the videoScript field, use natural Chinese prose (e.g. "镜头缓慢推近", "低角度上摇") — these are two separate fields, do not confuse them.

For each shot, choose one English keyword in the cameraDirection field:
- "static" — fixed shot, no movement
- "slow zoom in" / "slow zoom out" — slow zoom
- "pan left" / "pan right" — horizontal pan
- "tilt up" / "tilt down" — vertical tilt
- "tracking shot" — follows the character's movement
- "dolly in" / "dolly out" — camera physically moves forward/backward
- "crane up" / "crane down" — vertical rise/descent
- "orbit left" / "orbit right" — orbits around the subject
- "push in" — slow forward push for emphasis"""

SHOT_SPLIT_CINEMATOGRAPHY_PRINCIPLES_TEMPLATE = """Cinematography principles:
- Vary shot size — avoid using the same composition in consecutive shots; alternate wide/medium/close-up
- Use an establishing shot at the start of a new scene
- Use a reaction shot after important dialogue or events
- Cut on action — each shot ends at a moment that allows a smooth transition into the next
- Maintain eyeline matches — characters keep a consistent screen direction across shots
- The 180-degree rule — keep characters in consistent screen positions
- Duration: all shots must be within {{MIN_DURATION}}-{{MAX_DURATION}} seconds. Dialogue-heavy = {{DIALOGUE_MAX}}-{{MAX_DURATION}}s; action shots = {{MIN_DURATION}}-{{ACTION_MAX}}s; establishing shots = {{MIN_DURATION}}-{{ESTABLISHING_MAX}}s
- Continuity: shot N's last frame must logically connect to shot N+1's first frame (same characters, consistent environment, natural positional transition)
- Coverage: generate at least one shot for every scene in the script. Do not skip or merge scenes. If a scene is complex, split it into multiple shots. Every scene marker (Scene N) must produce at least one shot."""

SHOT_SPLIT_LANGUAGE_RULES = """[Critical Language Rule] All text fields (sceneDescription, startFrame, endFrame, motionScript, dialogues.text, dialogues.character) must use the same language as the script. If the script is Chinese, all fields are in Chinese. Only "cameraDirection" uses English (technical terms).

Return the JSON array only. Do not use markdown code blocks. Do not add commentary."""

SHOT_SPLIT_PROPORTIONAL_TIERS_TEMPLATE = """=== Proportional-Variation Rule ===
{{PROPORTIONAL_TIERS}}"""


def _build_shot_split(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    params = params or {}

    max_duration = params.get("maxDuration")
    if max_duration is None:
        max_duration = 15
    min_duration = min(8, max_duration)

    # Build proportional tiers dynamically
    if max_duration <= 8:
        proportional_tiers = f"- {min_duration}-{max_duration}s shots: the amount of change is proportional to the duration"
    else:
        tier1_end = round(max_duration * 0.6)
        tier2_end = round(max_duration * 0.85)
        tier2_start = tier1_end + 1
        tier3_start = tier2_end + 1
        proportional_tiers = (
            f"- {min_duration}-{tier1_end}s shots: small to medium change (slight head turn, expression change, small camera movement)\n"
            f"- {tier2_start}-{tier2_end}s shots: medium change (character moves position, clear expression change, distinct camera movement)\n"
            f"- {tier3_start}-{max_duration}s shots: large change (character crosses the frame, a major action completes, dramatic camera movement)"
        )

    duration_range = str(max_duration) if min_duration == max_duration else f"{min_duration}-{max_duration}"

    def replace_duration(text: str) -> str:
        text = text.replace("{{MIN_DURATION}}-{{MAX_DURATION}}", duration_range)
        text = text.replace("{{MIN_DURATION}}", str(min_duration))
        text = text.replace("{{MAX_DURATION}}", str(max_duration))
        return text

    role_definition = replace_duration(r("role_definition"))

    # Unified metadata-only output format.
    output_format = replace_duration(r("output_format"))

    cinematography = r("cinematography_principles")
    cinematography = (
        cinematography
        .replace("{{MIN_DURATION}}", str(min_duration))
        .replace("{{MAX_DURATION}}", str(max_duration))
        .replace("{{DIALOGUE_MAX}}", str(min(max_duration, 12)))
        .replace("{{ACTION_MAX}}", str(min(max_duration, 12)))
        .replace("{{ESTABLISHING_MAX}}", str(min(max_duration, 10)))
    )

    proportional_section = r("proportional_tiers").replace("{{PROPORTIONAL_TIERS}}", proportional_tiers)

    return "\n".join([
        role_definition,
        "",
        r("script_fidelity"),
        "",
        output_format,
        "",
        r("motion_script_rules"),
        "",
        r("video_script_rules"),
        "",
        proportional_section,
        "",
        r("camera_directions"),
        "",
        cinematography,
        "",
        r("language_rules"),
    ])


shotSplitDef = _make_def(
    "shot_split",
    "promptTemplates.prompts.shotSplit",
    "promptTemplates.prompts.shotSplitDesc",
    "shot",
    [
        slot("role_definition", SHOT_SPLIT_ROLE_DEFINITION, True),
        slot("script_fidelity", SHOT_SPLIT_FIDELITY_RULES, True),
        slot("output_format", SHOT_SPLIT_OUTPUT_FORMAT_TEMPLATE, False),
        slot("start_end_frame_rules", SHOT_SPLIT_START_END_FRAME_RULES, True),
        slot("motion_script_rules", SHOT_SPLIT_MOTION_SCRIPT_RULES, True),
        slot("video_script_rules", SHOT_SPLIT_VIDEO_SCRIPT_RULES, True),
        slot("proportional_tiers", SHOT_SPLIT_PROPORTIONAL_TIERS_TEMPLATE, True),
        slot("camera_directions", SHOT_SPLIT_CAMERA_DIRECTIONS, True),
        slot("cinematography_principles", SHOT_SPLIT_CINEMATOGRAPHY_PRINCIPLES_TEMPLATE, True),
        slot("language_rules", SHOT_SPLIT_LANGUAGE_RULES, False),
    ],
    _build_shot_split,
)


# ─── 7.5. shot_split_keyframe_assets ──

SHOT_KEYFRAME_ASSETS_ROLE = """You are a senior cinematographer and storyboard artist. Given a set of already-split shot metadata (each shot contains sceneDescription / motionScript / videoScript / dialogues / characters / cameraDirection), your task is to generate the image-generation prompts for the **first frame (startFrame)** and **last frame (endFrame)** of each shot.

Purpose of the first/last frames: the video generator uses the first frame as the opening image and the last frame as the closing image, and automatically interpolates the intermediate action. So the two frames must:
1. Describe two stable moments of the shot — first frame = the instant before the action begins, last frame = the instant after the action completes
2. Share the same scene environment (lighting, color temperature, location must be exactly consistent)
3. Transition through the action described in motionScript
4. Never be a motion-blur state — the last frame must be usable as the opening reference for the next shot"""

SHOT_KEYFRAME_ASSETS_RULES = f"""{physics_realism_block()}

{theme_style_mapping_block()}

[Character-Consistency Anchoring]
- Every time you mention a character, you must use the "character name (visual identifier)" format, **reusing the visual identifier verbatim** from the character list provided below; no rewriting
- When multiple characters share the frame, each carries its own visual-identifier parentheses

[Prompt Writing Format — Seedance / Jimeng Style]
Use natural Chinese prose. No weight syntax "（xx：1.99）", no structured tags.
Each startFrame / endFrame is 2-4 sentences of fluent prose, organized in this order:
1. Subject identity and posture: character name (visual identifier) + explicit body posture (standing/sitting/kneeling/crouching/prone) + foot position + body orientation
2. Action and expression: specific physical action, hand position, gaze direction, facial expression
3. Composition and camera: shot size (wide/medium/close/close-up) + angle (eye level/low angle/high angle) + focal length
4. Environment light and shadow: light-source direction and quality, color temperature, color tone, key environmental detail, atmosphere

[Relationship Between First and Last Frame]
- **Shared environment**: background, lighting, color temperature, location exactly consistent — only the character's posture/position/expression changes
- **First frame**: the instant before the first segment of motionScript begins — the character is at the starting position, with an opening expression
- **Last frame**: the instant after the last segment of motionScript ends — the character has completed the action, stopped in a stable pose (not a blurry mid-motion state)
- **Do not include dialogue text**"""

SHOT_KEYFRAME_ASSETS_OUTPUT_FORMAT = """Output a JSON array, one object per shot. **The `prompts` array must have exactly 2 elements: element 0 is the first frame, element 1 is the last frame**. **The `characters` array must contain only the characters that actually appear in this shot's frames** (not all characters in the project), with names exactly matching the character list:
[
  {
    "shotSequence": 1,
    "characters": ["name of character 1 actually appearing in this shot", "character name 2"],
    "prompts": [
      "the complete image-generation prompt for the first frame (Chinese prose)",
      "the complete image-generation prompt for the last frame (Chinese prose)"
    ]
  }
]
Output valid JSON only, no markdown code blocks, no preamble.

**Rules for determining the `characters` field**:
- List only the characters that **visually appear** in this shot's motionScript / videoScript / sceneDescription
- Characters with narration/voiceover-only dialogue, if not on screen, must not be listed
- An empty array [] is valid (a pure environment shot / empty shot)"""


def _build_shot_keyframe_assets(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    return "\n".join([r("role_definition"), "", r("rules"), "", r("output_format")])


shotKeyframeAssetsDef = _make_def(
    "shot_split_keyframe_assets",
    "promptTemplates.prompts.shotSplitKeyframeAssets",
    "promptTemplates.prompts.shotSplitKeyframeAssetsDesc",
    "shot",
    [
        slot("role_definition", SHOT_KEYFRAME_ASSETS_ROLE, True),
        slot("rules", SHOT_KEYFRAME_ASSETS_RULES, True),
        slot("output_format", SHOT_KEYFRAME_ASSETS_OUTPUT_FORMAT, False),
    ],
    _build_shot_keyframe_assets,
)


# ─── 8. frame_generate_first ────────────────────────────

FIRST_FRAME_STYLE_MATCHING = f"""=== Critical: Art-Style Matching (Highest Priority) ===
Read the character description and scene description below carefully. They specify or imply an art style.
You must match that art style precisely. Do not default to photorealistic.
- If a reference image is attached, the reference image's visual style is the truth — match it precisely
- The output art style must be consistent with the character design sheet

{theme_style_mapping_block()}

{art_style_block()}

{physics_realism_block()}"""

FIRST_FRAME_REFERENCE_RULES = """=== Reference Image (Character Design Sheet) ===
Each attached reference image is a character design sheet showing 4 views (front, three-quarter, profile, back).
The character's name is printed at the bottom of each sheet — use it to identify the corresponding character.
Mandatory consistency rules:
- Map the character names on the design sheets to the character names in the scene description
- Clothing must exactly match the reference image — same garment type, color, material, accessories. Do not substitute (e.g. do not swap a cyan everyday robe for a dragon robe)
- Face, hairstyle, hair color, build, skin tone must match precisely
- All accessories shown in the reference image (hat, saber, hairpin, jewelry) must appear
- The art style must match the reference image precisely"""

FIRST_FRAME_RENDERING_QUALITY = """=== Rendering ===
Material: rich detail suited to the art style
Lighting: motivated cinematic lighting. Use a rim light to separate the character.
Background: a fully rendered, detailed environment. Not a blank or abstract background.
Character: precisely matching the appearance and art style of the reference image. Vivid expression, natural and dynamic posture.
Composition: cinematic framing, a clear visual focus and depth of field."""

FIRST_FRAME_CONTINUITY_RULES = """=== Continuity Requirements ===
This shot immediately follows the previous shot. The attached reference includes the previous shot's last frame. Maintain visual continuity:
- The same characters must wear consistent clothing and keep consistent proportions
- The same art style — do not switch between anime and photorealistic
- The environmental lighting and color temperature should transition smoothly
- The character's position should logically continue from where the previous shot ended"""


def _build_frame_generate_first(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    params = params or {}
    scene_description = params.get("sceneDescription") or ""
    start_frame_desc = params.get("startFrameDesc") or ""
    character_descriptions = params.get("characterDescriptions") or ""
    previous_last_frame = params.get("previousLastFrame") or ""

    lines = []
    lines.append("Generate the first frame of this shot as a high-quality image.")
    lines.append("")
    lines.append(r("style_matching"))
    lines.append("")
    lines.append("=== Scene Environment ===")
    lines.append(scene_description)
    lines.append("")
    lines.append("=== Frame Description ===")
    lines.append(start_frame_desc)
    lines.append("")
    lines.append("=== Character Description ===")
    lines.append(character_descriptions)
    lines.append("")
    lines.append(r("reference_rules"))
    lines.append("")

    if previous_last_frame:
        lines.append(r("continuity_rules"))
        lines.append("")

    lines.append(r("rendering_quality"))
    return "\n".join(lines)


frameGenerateFirstDef = _make_def(
    "frame_generate_first",
    "promptTemplates.prompts.frameGenerateFirst",
    "promptTemplates.prompts.frameGenerateFirstDesc",
    "frame",
    [
        slot("style_matching", FIRST_FRAME_STYLE_MATCHING, True),
        slot("reference_rules", FIRST_FRAME_REFERENCE_RULES, True),
        slot("rendering_quality", FIRST_FRAME_RENDERING_QUALITY, True),
        slot("continuity_rules", FIRST_FRAME_CONTINUITY_RULES, True),
    ],
    _build_frame_generate_first,
)


# ─── 9. frame_generate_last ─────────────────────────────

LAST_FRAME_STYLE_MATCHING = """=== Critical: Art-Style Matching (Highest Priority) ===
You must precisely match the art style of the first-frame image (attached).
If the first frame is anime/manga style -> this frame must also be anime/manga style.
If the first frame is photorealistic -> this frame must also be photorealistic.
Do not change or mix art styles. This is non-negotiable."""

LAST_FRAME_RELATIONSHIP_TO_FIRST = """=== Relationship to the First Frame ===
This last frame shows the ending state of the shot's action. Compared with the first frame:
- Same environment, lighting scheme, and color tone
- Absolutely the same art style — no variation permitted
- Exactly consistent clothing — the character wears exactly the same clothes as in the design sheet and first frame. No costume change.
- Same face, hairstyle, accessories — only posture/expression/position changes
- The character's position, posture, and expression have changed as instructed in the frame description"""

LAST_FRAME_NEXT_SHOT_READINESS = """=== As the Starting Point of the Next Shot ===
This frame will be reused as the first frame of the next shot. Ensure:
- The posture is stable — not mid-motion, not blurry
- The composition is complete and stands as an independent image
- The framing allows a natural transition to a different shot angle"""

LAST_FRAME_RENDERING_QUALITY = """=== Rendering ===
Material: rich detail matching the first-frame style
Lighting: the same lighting scheme as the first frame. Vary only if driven by the action.
Background: must match the first frame's environment.
Character: precisely matching the reference image. Show the emotional state at the end of the shot's action.
Composition: a natural conclusion of the shot, ready for the next cut."""


def _build_frame_generate_last(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    params = params or {}
    scene_description = params.get("sceneDescription") or ""
    end_frame_desc = params.get("endFrameDesc") or ""
    character_descriptions = params.get("characterDescriptions") or ""

    lines = []
    lines.append("Generate the last frame of this shot as a high-quality image.")
    lines.append("")
    lines.append(r("style_matching"))
    lines.append("")
    lines.append("=== Scene Environment ===")
    lines.append(scene_description)
    lines.append("")
    lines.append("=== Frame Description ===")
    lines.append(end_frame_desc)
    lines.append("")
    lines.append("=== Character Description ===")
    lines.append(character_descriptions)
    lines.append("")
    lines.append("=== Reference Images ===")
    lines.append("The first attached image is this shot's first frame — use it as the visual anchor.")
    lines.append("The remaining attached images are character design sheets (each with 4 views, name printed at the bottom).")
    lines.append("Map each design sheet's character name to the characters in the scene.")
    lines.append("")
    lines.append(r("relationship_to_first"))
    lines.append("")
    lines.append(r("next_shot_readiness"))
    lines.append("")
    lines.append(r("rendering_quality"))
    return "\n".join(lines)


frameGenerateLastDef = _make_def(
    "frame_generate_last",
    "promptTemplates.prompts.frameGenerateLast",
    "promptTemplates.prompts.frameGenerateLastDesc",
    "frame",
    [
        slot("style_matching", LAST_FRAME_STYLE_MATCHING, True),
        slot("relationship_to_first", LAST_FRAME_RELATIONSHIP_TO_FIRST, True),
        slot("next_shot_readiness", LAST_FRAME_NEXT_SHOT_READINESS, True),
        slot("rendering_quality", LAST_FRAME_RENDERING_QUALITY, True),
    ],
    _build_frame_generate_last,
)


# ─── 10. scene_frame_generate ────────────────────────────

SCENE_FRAME_REFERENCE_RULES = f"""=== No-Characters Mandatory Constraint (Highest Priority) ===
This is a pure scene reference image. **Absolutely no people, characters, silhouettes, back views, human figures, hands, feet, or body parts of any kind are allowed** in the frame.
- Forbidden: people, characters, back views, silhouettes, human outlines, exposed hands/feet/shoulders
- Allowed: empty environments, architecture, props, natural landscapes, weather, lighting, atmospheric particles
- Character consistency is guaranteed by the multi-reference mechanism at the later video-generation stage, fully decoupled from this step

{theme_style_mapping_block()}

{physics_realism_block()}"""

SCENE_FRAME_COMPOSITION_RULES = """=== Composition Rules ===
- Render a specific spatial composition per the scene description — do not default to a generic shot
- A fully rendered background and environment — not a blank or abstract background
- Cinematic framing, clear composition and depth of field
- The composition must leave room for characters to enter later, but no person appears in the frame at this moment"""

SCENE_FRAME_RENDERING = """=== Rendering Quality ===
- Material: rich detail suited to the art style
- Lighting: cinematic lighting, with a clear motivation for the light source
- Art style: follow the style directions in the scene description
- To reiterate: no people appear in the frame"""


def _build_scene_frame_generate(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    params = params or {}
    scene_description = params.get("sceneDescription") or ""
    camera_direction = params.get("cameraDirection") or ""
    start_frame_desc = params.get("startFrameDesc") or ""
    # motionScript / charRefMapping / characterDescriptions are intentionally NOT used.

    lines = []
    lines.append("Generate a cinematic still image as a pure scene reference frame. No people may appear in the frame.")
    lines.append("")
    lines.append("=== Scene Description ===")
    lines.append(scene_description)

    if start_frame_desc:
        lines.append("")
        lines.append("=== Space and Moment ===")
        lines.append(f"The frame must depict this space and moment (take only the environment/lighting/prop information from it, do not depict people): {start_frame_desc}")

    if camera_direction and camera_direction != "static":
        lines.append("")
        lines.append("=== Shot Composition ===")
        lines.append(f"Camera angle/distance: {camera_direction}")
        lines.append("Apply this camera angle to the composition.")

    lines.append("")
    lines.append(r("reference_rules"))
    lines.append("")
    lines.append(r("composition_rules"))
    lines.append("")
    lines.append(r("rendering"))

    return "\n".join(lines)


sceneFrameGenerateDef = _make_def(
    "scene_frame_generate",
    "promptTemplates.prompts.sceneFrameGenerate",
    "promptTemplates.prompts.sceneFrameGenerateDesc",
    "frame",
    [
        slot("reference_rules", SCENE_FRAME_REFERENCE_RULES, True),
        slot("composition_rules", SCENE_FRAME_COMPOSITION_RULES, True),
        slot("rendering", SCENE_FRAME_RENDERING, True),
    ],
    _build_scene_frame_generate,
)


# ─── 11. video_generate ─────────────────────────────────

VIDEO_INTERPOLATION_HEADER = """Describe, in natural Chinese prose, the dynamic process that unfolds from the first frame to the last frame. Do not use structured tags ("Scene:", "Action:"), do not use weight syntax ("（xx：1.5）"). Write the shot as a piece of cinematic footage; the language should let the model "see" it.

Writing points (Seedance 2.0 style):
- Subject action: specific physical movement — gripping, leaning, turning back, raising a hand, footsteps slowing, breath pausing; write speed and force.
- Environmental reaction: the world's response to the subject — robe hems flying, leaves lifting, light patches sweeping across a wall, ripples spreading on water.
- Camera movement: use concrete words — "camera slowly pushes in" / "low-angle wide slowly tilting up" / "orbiting pan quick cut" / "locked-off shot" / "Hitchcock zoom"; not empty words like "gracefully" or "softly".
- Physics and atmosphere: material detail, light/shadow color temperature, sound cues (footsteps, fabric friction, breathing, ambient sound), to make the model feel "present".

Duration strategy:
- 4-8s: focus on one core action, no timestamps.
- 9-12s: 2-3 timestamp segments, e.g. "0-4s: ... 5-8s: ... 9-12s: ..."
- 13-15s: mandatorily use 3-4 timestamp segments, each a dense long sentence weaving the four layers of subject/environment/camera/physics.

Composition safe zone (subtitle reserve):
The bottom 20% of the frame is the subtitle area; the character's face and key action must be in the upper 2/3 of the frame. In close-ups, the face is centered slightly high; in full-body shots, the feet may be at the bottom but the performance area is at the top. Add composition guidance to the prompt like "the character is positioned in the upper-middle of the frame".

Ending prohibitions (write directly on the last line of the prompt):
No watermark, subtitle, text LOGO, insignia, timecode, or frame border may appear."""

# PORT NOTE: The dialogue-format tokens 【对白口型】 (on-screen lip-sync) and 【画外音】
# (off-screen voiceover), and the line hints 画内对白 / 画外旁白, are parsed literally by
# the downstream extractLabel() parser (it matches the 【...】 label on the line containing
# the 画内对白 / 画外旁白 hint). They MUST stay Chinese verbatim. Only the surrounding
# prose is translated. Example dialogue values are kept in Chinese to illustrate the format.
VIDEO_DIALOGUE_FORMAT = """Dialogue format (each line on its own line, placed after the frame description):
- 画内对白 (on-screen dialogue): 【对白口型】CharacterName (visual identifier, emotion): "the original line"
- 画外旁白 (off-screen narration): 【画外音】CharacterName (emotion): "the original line"

The emotion annotation is key — it lets the model align lip movement, breathing rhythm, and the line. Examples:
- 【对白口型】苏晚（红裙黑发，冷漠反杀）: "顾总，当初是你说，我连给你提鞋都不配。"
- 【画外音】旁白（低沉沙哑）: "那一夜，城市比雨还冷。"

Put sound effects on their own line, beginning with "音效：", separate from the frame description.
Example: 音效：契约撕碎的脆响、宾客窃窃私语、远处低沉的背景弦乐。"""

# PORT NOTE: The frame-anchor header [帧锚点] and the frame labels 首帧： / 尾帧： are parsed
# literally by extractAnchorHeader() / extractFrameLabel() downstream (they match the leading
# [...] header and the label before the {{...}} placeholder). They MUST stay Chinese verbatim.
# The {{START_FRAME_DESC}} / {{END_FRAME_DESC}} placeholder tokens are also preserved.
VIDEO_FRAME_ANCHORS = """[帧锚点]
首帧：{{START_FRAME_DESC}}
尾帧：{{END_FRAME_DESC}}"""


def _build_video_generate(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    return "\n".join([
        r("interpolation_header"),
        "",
        r("dialogue_format"),
        "",
        r("frame_anchors"),
    ])


videoGenerateDef = _make_def(
    "video_generate",
    "promptTemplates.prompts.videoGenerate",
    "promptTemplates.prompts.videoGenerateDesc",
    "video",
    [
        slot("interpolation_header", VIDEO_INTERPOLATION_HEADER, True),
        slot("dialogue_format", VIDEO_DIALOGUE_FORMAT, True),
        slot("frame_anchors", VIDEO_FRAME_ANCHORS, True),
    ],
    _build_video_generate,
)


# ─── 11b. ref_video_generate ─────────────────────────────

# Reuse the same dialogue format as video_generate (avoid duplication)
REF_VIDEO_DIALOGUE_FORMAT = VIDEO_DIALOGUE_FORMAT

REF_VIDEO_CONSISTENCY_RULES = """=== Reference-Image Consistency Constraint (The Core Lifeline of Reference Mode) ===
When generating the video, the attached reference images are the **authoritative visual reference**, not optional suggestions. Enforce strictly:
- **Do not change character appearance**: clothing color, style, accessories, hairstyle, hair color, face shape, and build must exactly match the reference image. Do not "switch outfits" mid-video.
- **Do not change environment style**: background tone, materials, architectural style, and light/shadow tone must match the reference image.
- **The only things allowed to change are dynamics**: character posture, expression, physical action, camera movement, and the environment's dynamic reactions (swaying, scattering, lifting, etc.).
- **Multi-character scenes**: each character strictly corresponds to its own reference image; do not mismatch identities.
- **Art-style lock**: the reference image's art style is the video's art style; do not "upgrade" or "stylize" it into something else."""

REF_VIDEO_DURATION_STRATEGY = """=== Duration Strategy (Seedance 2.0) ===
Choose the description granularity by shot duration:
- 4-8s: one core action + one camera movement + one atmosphere detail, a 30-60 character single-paragraph prose.
- 9-12s: 2-3 timestamp segments ("0-4s: ... 5-8s: ..."), 60-120 characters.
- 13-15s: 3-4 timestamp segments ("0-3s / 4-8s / 9-12s / 13-15s"), 120-200 characters, each segment weaving the four layers of "character action / environment reaction / camera movement / physics and sound".

Camera movements must use concrete words: "缓慢推近" / "环绕摇镜快切" / "希区柯克变焦" / "低角度广角上摇" / "定格慢放" / "固定机位"; no empty modifiers like "gracefully" or "softly"."""


def _build_ref_video_generate(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    return "\n".join([
        r("consistency_rules"),
        "",
        r("duration_strategy"),
        "",
        r("dialogue_format"),
    ])


refVideoGenerateDef = _make_def(
    "ref_video_generate",
    "promptTemplates.prompts.refVideoGenerate",
    "promptTemplates.prompts.refVideoGenerateDesc",
    "video",
    [
        slot("consistency_rules", REF_VIDEO_CONSISTENCY_RULES, True),
        slot("duration_strategy", REF_VIDEO_DURATION_STRATEGY, True),
        slot("dialogue_format", REF_VIDEO_DIALOGUE_FORMAT, True),
    ],
    _build_ref_video_generate,
)


# ─── 12. ref_video_prompt ───────────────────────────────
# Seedance 2.0 reference-mode video prompt writer.

REF_VIDEO_PROMPT_ROLE_DEFINITION = """You are a Seedance 2.0 video-prompt writing expert. You receive an **ordered** set of reference images:
  - The first N are character reference images (each bound to a character name)
  - The last M are scene reference images (pure environments, no people, arranged in temporal order)

Your task is to write a Seedance video prompt from these reference images, the script action, the camera directives, and the dialogue, automatically planning the action, camera work, and dialogue rhythm."""

# PORT NOTE: `@图片N` is a LITERAL Seedance/Jimeng reference-syntax token (图片 = "image N")
# and is intentionally kept in Chinese — the entire point of this prompt is that the model
# emits this exact token to bind references. The `（name）` full-width parenthesis convention
# after each @图片N is also preserved. Only the surrounding prose is translated to English.
REF_VIDEO_PROMPT_MOTION_RULES = """## Core Syntax (Seedance @ reference — official Jimeng format)

1. **All characters and scenes must be referenced with the `@图片N` form** (note it is `@图片1` `@图片2`). The order strictly corresponds to the order of the reference images received — the first N are characters, the last M are scenes.

2. **Writing style: coherent, fluent natural prose**.
   - Embed `@图片N` directly into the prose description, like this:
     "@图片1 中的美妆博主用中文介绍，手持 @图片2 的面霜面向镜头展示，清新简约背景"
   - **Do not** use structured tags like "beat 1 / beat 2 / beat 3"
   - **Do not** write a separate mapping-declaration line at the start like "image mapping: @图片1 is X, @图片2 is Y" — the information must **melt into the prose**
   - **Every time** @图片N appears, it must be followed by the character name, written as "@图片1（李慕白）", so the reader always knows who is who

3. **Camera work / shot size must be concrete**: 近景 / 中景 / 全景 / 特写 / 环绕 / 固定机位 / 推镜头 / 拉镜头 / 手持跟拍 / 低角度仰拍 / 升格 / 希区柯克变焦 / 俯拍 / 鸟瞰. No empty modifiers like "gracefully", "softly", "stunning".

4. **Scene changes written directly in the prose**: "画面切到 @图片4 的竹梢高空" / "@图片1 从 @图片3 纵身跃起，落入 @图片4".

5. **Dialogue format (official Jimeng writing)**: embedded directly in the prose, beginning with "CharacterName台词：" followed by the original line, e.g.:
   > 博主台词：挖到本命面霜了！质地像云朵一样软糯，一抹就吸收。

   **Do not** use structured tags like "【对白口型】@图片N（名字）: "line"".

6. **Sound effects**: if there is ambient/action sound, weave it directly into the prose description (e.g. "伴随清脆的剑鸣声", "背景响起低沉的鼓点"), with no separate sound-effect line.

## Action-Rhythm Planning (Core!)

**There must be a visual change every second**. A shot can never have just one action — even a close-up must be broken into a chain of continuous micro-actions.

Rhythm formula: **arrange one action beat every 2-3 seconds**, with transition actions bridging the beats (e.g. shifting gaze, shifting weight, changing gesture, changing expression, changing light and shadow).

| Duration | Beats | Word count | Notes |
|------|--------|------|------|
| 4-5s | 2 | 40-70 chars | starting action -> completing action |
| 6-8s | 3 | 60-100 chars | start -> develop -> conclude, with a turn or change in the middle |
| 9-12s | 4-5 | 100-160 chars | multi-stage action chain, varying pace |
| 13-15s | 5-6 | 150-220 chars | a complete mini narrative arc, with emotional ups and downs |

**Example comparison**:

[BAD] slow pace (8s with only 1 action):
"固定特写，她修长的手指敲击金属桌面，发出清脆声响。"
-> Problem: 8 seconds of just fingers tapping the table; the frame is static

[GOOD] correct pace (8s, 3 beats):
"固定特写下，她涂着黑色指甲油的手指先缓慢抚过冰冷桌面划痕，随即食指与中指交替敲击金属面，震起微尘——第三下敲击后手指骤然停住，五指收拢握拳，指节泛白。"
-> stroke -> tap -> clench fist, three stages filling the 8 seconds

**Key techniques**:
- Use time words like "先...随即...然后..." to string micro-actions together
- Even if the character's main action is singular, add: breathing rise and fall, clothing/hair movement, subtle environmental change (light, dust, water surface), subtle camera adjustment (slow push/slow pull)
- Dialogue shots: the character has a preparatory action before speaking (raising the eyes, a change at the mouth corner), gestures/body language while speaking, and a closing expression after finishing

## Composition Safe Zone (Subtitle Reserve)

The **bottom 20%** of the frame is the subtitle area and must stay clean — do not place the character's face, key action, or important props in the bottom 1/5 of the frame.

Specific requirements:
- The character's face and upper body should be in the upper-middle of the frame (the upper 60% area)
- Close-up: the face centered slightly high, with enough space left below the chin
- Full-body shot: the feet may be at the bottom, but the key performance area (face, hand action) must be in the upper 2/3
- Guide with composition description in the prompt, e.g.: "人物居于画面中上方", "角色面部位于画面上半部", "底部留出字幕空间"
- No text, watermark, subtitle, or LOGO may appear

## Other Rules
- Language follows the script: Chinese script -> Chinese prompt, English -> English.
- Do not write into the prompt any character/scene that was not passed to you.
- Do not have a frame with only a scene description and the character completely still.
- Output the prompt body only, no preamble, no markdown."""

REF_VIDEO_PROMPT_QUALITY_BENCHMARK = """## Official Benchmark Examples

[Example 1 — Beauty-product showcase (official Jimeng writing)]
Input:
  图片1 = beauty blogger (character)
  图片2 = face cream (product prop)
  Script: the blogger introduces the face-cream product
  Camera: close shot (近景)

Output:
@图片1（美妆博主）用中文进行介绍，妆容改为明艳大气，去掉脸部反光，笑容甜美，近景镜头，手持 @图片2（面霜）面向镜头展示，清新简约背景，元气甜美风格。博主台词：挖到本命面霜了！质地像云朵一样软糯，一抹就吸收，熬夜急救、补水保湿全搞定，素颜都自带柔光感。

[Example 2 — Xianxia fight (crossing multiple scenes, 10s)]
Input:
  图片1 = Li Mubai (character)
  图片2 = Yu Jiaolong (character)
  图片3 = bamboo forest (scene)
  图片4 = bamboo-tip high altitude (scene)
  Script action: Li Mubai chases Yu Jiaolong, the two leap from the ground onto the bamboo tips and fight
  Camera: low-angle upward tracking
  Duration: 10s

Output:
低角度仰拍跟随 @图片1（李慕白）在 @图片3（竹林）地面屈膝蓄力半秒，随即蹬地腾空，镜头同步上摇穿过竹干。画面切到 @图片4（竹梢高空），@图片2（玉娇龙）自左侧斜劈青剑而来，@图片1（李慕白）侧身以指尖格挡，两人在竹梢高空短暂对峙，青翠竹叶被剑气吹得纷纷飘落。李慕白台词：江湖路远，何必执着。

[Example 3 — Close-up (single person, 8s, demonstrating correct rhythm)]
Input:
  图片1 = the Yang family's eldest daughter (character)
  图片2 = metal tabletop (scene)
  Script action: the young lady waits at the table, showing impatience
  Camera: fixed close-up
  Duration: 8s

Output:
固定特写下 @图片1（杨家大小姐）涂着黑色指甲油的食指沿 @图片2（金属桌面）布满划痕的表面缓缓划过，指尖拂起一缕灰尘。随即 @图片1（杨家大小姐）食指与中指交替敲击冰冷桌面，节奏由慢渐快，每一下震起微小尘粒在顶光中浮游。第四下敲击后手指骤然收住，五指缓缓握拢成拳，指节泛白，黑色甲片嵌入掌心。

## Negative Examples (Forbidden)
[BAD] "他的手指散发出温暖的光芒，优雅地落下棋子" — no @图片 mapping, abstract modifiers
[BAD] "李慕白纵身跃起" — writes the name directly, no @图片 binding
[BAD] "图1 从台阶走下" — missing the @ prefix, must be written as @图片1
[BAD] "@图片1 侧身格挡" — missing the character name, must be written as @图片1（李慕白）
[BAD] "图像映射：@图片1是李慕白，@图片2是玉娇龙。节拍 1：李慕白蓄力..." — do not use a separate mapping-declaration line or beat tags
[BAD] "【对白口型】@图片1（李慕白）: "江湖路远"" — do not use structured dialogue tags; write "李慕白台词：江湖路远" directly"""

# Use shared language rule block with a prompt-specific addendum
REF_VIDEO_PROMPT_LANGUAGE_RULES = f"{language_rule_block()}\nOutput the prompt only, no preamble."


def _build_ref_video_prompt(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    return "\n".join([
        r("role_definition"),
        "",
        r("motion_rules"),
        "",
        r("quality_benchmark"),
    ])


refVideoPromptDef = _make_def(
    "ref_video_prompt",
    "promptTemplates.prompts.refVideoPrompt",
    "promptTemplates.prompts.refVideoPromptDesc",
    "video",
    [
        slot("role_definition", REF_VIDEO_PROMPT_ROLE_DEFINITION, True),
        slot("motion_rules", REF_VIDEO_PROMPT_MOTION_RULES, True),
        slot("quality_benchmark", REF_VIDEO_PROMPT_QUALITY_BENCHMARK, True),
        slot("language_rules", REF_VIDEO_PROMPT_LANGUAGE_RULES, False),
    ],
    _build_ref_video_prompt,
)


# ─── 14. script_outline ──────────────────────────────────

SCRIPT_OUTLINE_ROLE = """You are an award-winning screenwriter. From the user's creative idea, generate a concise story outline."""

SCRIPT_OUTLINE_FORMAT = """Output format — a plain-text timeline, no JSON, no markdown:

Premise: (one-sentence core conflict)

1. [Beat name] (share XX%)
   Event: ...
   Emotion: ...

2. [Beat name] (share XX%)
   Event: ...
   Emotion: ...

3. [Beat name] (share XX%)
   Event: ...
   Emotion: ...

Climax: ...
Ending: ..."""

SCRIPT_OUTLINE_RULES = """Requirements:
- 3-5 key beats, each containing an event and an emotional shift
- The shares should sum to 100%
- Language rule: use the same language as the user's input (Chinese input -> Chinese output, English input -> English output)
- Output the content directly, without any wrapper or markers

[Combat/Duel Genre Special Rules]
If a combat signal word appears in the user's idea/title — "大战", "对决", "决战", "交手", "PK", "VS", "vs", "battle", "fight", "duel", "对打", "厮杀" — then the beat allocation must be arranged as a **real-combat duel**:
- Beat 1 "Entrance" (10-15%): both sides enter, face off, declare war with lines
- Beat 2 "First exchange" (15-20%): the first wave of actual fighting, testing each other's style
- Beat 3 "Escalation" (25-30%): heavier moves, the environment is destroyed, both sides take wounds
- Beat 4 "Desperate counterattack" (20-25%): the disadvantaged side fights back from the brink, or both sides wound each other
- Beat 5 "Finale" (15-20%): the decisive blow + a brief aftermath

**The real-combat beat share must be >= 50%**. Do not interpret "大战" as the artsy cliché of "one side dominates + the other has an epiphany + a symbolic single strike" — when the user says "大战", they want a sustained sequence of both sides fighting, not a one-sided mental struggle. Both sides must be active combatants, not one standing still while the other struggles."""


def _build_script_outline(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    return "\n".join([r("role_definition"), "", r("output_format"), "", r("writing_rules")])


scriptOutlineDef = _make_def(
    "script_outline",
    "promptTemplates.prompts.scriptOutline",
    "promptTemplates.prompts.scriptOutlineDesc",
    "script",
    [
        slot("role_definition", SCRIPT_OUTLINE_ROLE, True),
        slot("output_format", SCRIPT_OUTLINE_FORMAT, True),
        slot("writing_rules", SCRIPT_OUTLINE_RULES, True),
    ],
    _build_script_outline,
)


# ─── 15. ref_image_prompts ───────────────────────────────

REF_IMAGE_PROMPTS_ROLE = """You are a professional film art director, preparing **scene reference frames** for AI video generation. A scene reference frame is a pure-environment still used at the later video-generation stage as one of the multi-modal reference images, locking down spatial layout, lighting design, color-tone atmosphere, and camera language.

Core contract:
1. **Absolutely no people appear in the frame**: no people, characters, back views, silhouettes, human outlines, hands, feet, shoulders, faces, or clothing in a worn state. Character consistency is solved by the multi-reference mechanism at the later video stage, fully decoupled from this step.
2. **But you need to take characters into account while thinking**: the characters in the plot determine the appropriate spatial scale, camera height, light direction, and foreground-prop positions for this shot (e.g. an emperor holding court needs space left for the dragon throne and the marble dais steps; a fight needs room reserved for the action's trajectory). Use the characters to infer the scene's form, but do not draw them in the frame.
3. Each scene frame must output both a **scene name (name)** and a **scene description (prompt)**, plus the shot-level **on-screen character list (characters)**, so the later video-generation stage can precisely pull the corresponding character reference images."""

REF_IMAGE_PROMPTS_RULES = f"""Rules:
## Definition of a Scene Image (Most Important)
A scene image = **the physical location / environmental space where the character is**.
- [OK] Valid: the plaza before the Hall of Supreme Harmony, deep in a bamboo forest, the edge of a cliff, before a ruined palace gate, inside a meditation room, a wasteland under a blood moon, an underground cell, a dock pier
- [BAD] Invalid: energy light effects, glowing talismans, brand/sigil patterns, an isolated weapon/prop close-up, a character portrait, clothing accessories, abstract particles
- **The test**: looking only at this image, can you say "this is an XX place"? Yes = a scene image; if you can only say "this is a blob of light / a symbol / an object" = not a scene image.

## Scene-Image Count (Default 1, Maximum 4)
- **By default, generate only 1 scene image per shot** — the place where the character is. Dialogue, standing, charging up, throwing a punch, opening a door, turning around, close-ups — these **action beats within a single location** all need only 1, and the later video generation will complete all beats in the same location.
- Only the following cases warrant >1 (maximum 4):
  1. **The character crosses different physical locations within the shot**: fighting from the ground into the air (bamboo-forest ground -> bamboo-tip high altitude), a chase running from indoors to outdoors (study -> corridor -> courtyard), jumping from a bridge into the water
  2. **A large jump in scene lighting/time**: dusk -> late night, dim indoors -> stepping out into bright light
- When multiple, arrange in temporal order; item 0 is the shot's starting location.
- Each scene must be given a 4-10 character Chinese **scene name**, which must be a location rather than an abstract state (e.g. "太和殿广场", "竹林地面", "竹梢高空", "破败宫门", "深宫密室").
- The "characters" array must use character names **exactly matching** the character list, filling in only those who genuinely appear (with action or dialogue) in this shot. An empty array is valid (a pure environment shot).
- The image description **must absolutely not** mention any character name, nor describe a person's action/clothing/body.
- The image description **must absolutely not** depict energy light effects, sigils, talismans, or isolated props as "scenes" — those are action details, handled at the video-generation stage.

{physics_realism_block()}

[Seedance / Jimeng Style Requirements]
Use coherent, natural Chinese prose. No weight syntax "（xx：1.99）" (a legacy SD1.5 style that Seedance does not consume). No structured tags "Scene:" / "Action:".

Organize each scene description into 2-4 sentences of prose in this order:
1. **Shot size + camera/angle**: extreme-wide/wide/full/medium/close/close-up/extreme-close-up + eye level/high angle/low angle/worm's-eye/bird's-eye/fisheye
2. **Spatial subject**: the specific spatial description, architecture, props, and foreground/midground/background layers
3. **Light source and color**: specific light-source direction and quality (side backlight/Tyndall/neon/golden hour/moonlight/volumetric light/hard key light/soft light), color temperature, color tone (warm/cool/low-saturation/high-contrast)
4. **Artistic style**: 3D guoman CG / realism / ink wash / cyberpunk / film grain, and you may add an aspect-ratio hint like "2.35:1 宽银幕"

Each one must end with this sentence (copied in full): **"画面中不出现任何人物、文字、字幕、水印、LOGO。"**

[Absolute No-Go Zones]
- No real person names of any kind: directors, actors, artists, photographers, historical figures, brands, IP names. Violating this causes a 400 error from the image API.
  - [BAD] "张艺谋导演风格" / "王家卫式色彩" / "黑泽明构图"
  - [GOOD] "高饱和红黄色调的东方史诗质感" / "霓虹雨夜冷暖对比" / "高反差黑白武士片质感"
- No metaphorical verbs ("如同", "宛如", "像……般")
- No abstract emotion word as the subject (rewrite as a concrete visual description)
- No people, body parts, or clothing being worn appearing in the frame

{theme_style_mapping_block()}

[Correct Example 1 — Default single scene (dialogue/standing/close-up/charge-up/punch and other single-location actions)]
{{
  "shotSequence": 1,
  "characters": ["朱由检", "王承恩"],
  "scenes": [
    {{
      "name": "太和殿内",
      "prompt": "中景，平视固定机位，紫禁城太和殿内部大殿中央，前景是空的金丝楠木御案与散落的奏本，中景是汉白玉丹陛石台阶，背景是高耸的朱红立柱与雕梁画栋。暖色调、高对比、3D 国漫 CG，明清宫廷雕梁画栋的金红配色，2.35:1 宽银幕。画面中不出现任何人物、文字、字幕、水印、LOGO。"
    }}
  ]
}}
> Note: this shot's plot is "Zhu Youjian sits on the throne reviewing memorials, Wang Cheng'en kneels to report" — the whole thing happens inside the Hall of Supreme Harmony, the same location, so only 1 scene image is needed to lock the space. Do not split into multiple scenes just because there is a "close-up reviewing memorials" or "close shot of anger" beat.

[Correct Example 2 — Cross-location fight, multiple scenes]
{{
  "shotSequence": 5,
  "characters": ["李慕白", "玉娇龙"],
  "scenes": [
    {{
      "name": "竹林地面",
      "prompt": "中景，低角度仰拍广角镜头，空无一人的翠绿竹林深处，青石地面散落枯叶，竹干笔直延伸向画面上方。晨光从竹叶缝隙洒下形成体积光斑，色彩基调为冷绿与金黄的对比。3D 国漫 CG 写意武侠质感。画面中不出现任何人物、文字、字幕、水印、LOGO。"
    }},
    {{
      "name": "竹梢高空",
      "prompt": "大远景，高角度俯拍，翠绿竹林的顶部竹梢在风中轻轻摇曳，远处是云雾缭绕的山峦剪影，天空呈现淡蓝到金黄的渐变。体积光穿透云层，2.35:1 宽银幕，3D 国漫 CG 写意武侠质感。画面中不出现任何人物、文字、字幕、水印、LOGO。"
    }}
  ]
}}
> Note: in this shot the character **really** leaps from the bamboo-forest ground to the bamboo-tip high altitude — two different physical locations, so 2.

[Negative Example — Do Not Treat Effects/Props/Light as Scenes]
[BAD] Wrong:
{{
  "shotSequence": 3,
  "scenes": [
    {{ "name": "烙印红光闪耀", "prompt": "大特写，平视固定机位，经文环形烙印图案剧烈向外扩张..." }}
  ]
}}
-> This is not a scene image; it is an action detail / effect detail. This shot's real scene should be "the physical location where the character is", such as "大雷音寺佛堂". The glowing sigil effect is expressed by the later video-generation stage within that location.

[OK] Correct rewrite:
{{
  "shotSequence": 3,
  "characters": ["如来佛祖", "孙悟空"],
  "scenes": [
    {{ "name": "大雷音寺佛堂", "prompt": "中景，平视固定机位，宏大的大雷音寺佛堂内部，金色莲花宝座居中，四周半空悬浮暗金色经文环，梁柱雕刻满饰佛纹。暗金与暗红色调，3D 国漫顶级渲染，电影级历史正剧质感。画面中不出现任何人物、文字、字幕、水印、LOGO。" }}
  ]
}}

[Critical Language Rule] Output in the same language as the input. Chinese input -> Chinese output. English input -> English output."""

REF_IMAGE_PROMPTS_FORMAT = """Output a valid JSON array only (no markdown, no code blocks, no preamble):

[
  {
    "shotSequence": 1,
    "characters": ["character name 1", "character name 2"],
    "scenes": [
      { "name": "scene name 1", "prompt": "scene description 1" },
      { "name": "scene name 2", "prompt": "scene description 2" }
    ]
  }
]

**Hard field requirements**:
- `characters`: the names of characters who appear (with action or dialogue) in this shot, which must exactly match the input character list. An empty array is valid.
- `scenes`: each element must have both `name` (a 4-10 character Chinese scene name) and `prompt` (a complete Seedance prose description).
- Do not use the legacy `prompts: [string]` array format.
- The scenes array is in temporal order; item 0 is the starting space."""


def _build_ref_image_prompts(slots, sc, params=None):
    r = lambda k: resolve(sc, slots, k)
    return "\n".join([r("ref_image_role"), "", r("ref_image_rules"), "", r("ref_image_output")])


refImagePromptsDef = _make_def(
    "ref_image_prompts",
    "promptTemplates.prompts.refImagePrompts",
    "promptTemplates.prompts.refImagePromptsDesc",
    "frame",
    [
        slot("ref_image_role", REF_IMAGE_PROMPTS_ROLE, True),
        slot("ref_image_rules", REF_IMAGE_PROMPTS_RULES, True),
        slot("ref_image_output", REF_IMAGE_PROMPTS_FORMAT, False),
    ],
    _build_ref_image_prompts,
)


# ── Registry ─────────────────────────────────────────────

PROMPT_REGISTRY: list[PromptDefinition] = [
    scriptOutlineDef,
    scriptGenerateDef,
    scriptParseDef,
    scriptSplitDef,
    characterExtractDef,
    importCharacterExtractDef,
    characterImageDef,
    shotSplitDef,
    shotKeyframeAssetsDef,
    frameGenerateFirstDef,
    frameGenerateLastDef,
    sceneFrameGenerateDef,
    refImagePromptsDef,
    videoGenerateDef,
    refVideoGenerateDef,
    refVideoPromptDef,
]

PROMPT_REGISTRY_MAP: dict[str, PromptDefinition] = {d.key: d for d in PROMPT_REGISTRY}


def get_prompt_definition(key: str) -> Optional[PromptDefinition]:
    """Look up a prompt definition by key."""
    return PROMPT_REGISTRY_MAP.get(key)


def get_default_slot_contents(key: str) -> Optional[dict]:
    """Get the default slot contents for a prompt definition as a plain dict."""
    d = PROMPT_REGISTRY_MAP.get(key)
    if not d:
        return None
    return {s.key: s.default_content for s in d.slots}
