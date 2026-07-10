"""Multi-platform agent caller — Python port of src/lib/ai/agent-caller.ts.

Supports three agent platforms: Bailian (DashScope), Dify, and Coze.
Chinese messages in the source have been translated to natural English.
Streaming is exposed as an async generator of decoded text chunks (the Python
equivalent of the TS ReadableStream<Uint8Array>).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)

# "bailian" | "dify" | "coze"
AgentPlatform = str


@dataclass
class AgentConfig:
    platform: AgentPlatform
    app_id: str
    api_key: str


# ── Unified streaming caller ────────────────────────────────────────
# Yields decoded text chunks. Raises on error.
async def call_agent_stream(config: AgentConfig, prompt: str) -> AsyncIterator[str]:
    if config.platform == "bailian":
        async for chunk in _call_bailian_agent_stream(config, prompt):
            yield chunk
    elif config.platform == "dify":
        async for chunk in _call_dify_agent_stream(config, prompt):
            yield chunk
    elif config.platform == "coze":
        # Coze workflow doesn't have native SSE for run_workflow — fall back to full text
        text = await _call_coze_agent(config, prompt)
        yield text
    else:
        raise ValueError(f"Unsupported agent platform: {config.platform}")


async def _call_bailian_agent_stream(config: AgentConfig, prompt: str) -> AsyncIterator[str]:
    url = f"https://dashscope.aliyuncs.com/api/v1/apps/{config.app_id}/completion"

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
                "X-DashScope-SSE": "enable",
            },
            json={"input": {"prompt": prompt}, "parameters": {"incremental_output": True}},
        ) as res:
            if res.status_code >= 400:
                err_text = (await res.aread()).decode("utf-8", "ignore")
                raise RuntimeError(
                    f"Bailian agent request failed: {res.status_code} {err_text[:300]}"
                )

            async for line in res.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue  # skip malformed line
                if obj.get("code"):
                    raise RuntimeError(
                        f"Bailian agent error [{obj['code']}]: {obj.get('message', 'unknown')}"
                    )
                chunk = (obj.get("output") or {}).get("text") or ""
                # Unwrap result wrapper
                try:
                    wrapper = json.loads(chunk)
                    if isinstance(wrapper, dict) and isinstance(wrapper.get("result"), str):
                        chunk = wrapper["result"]
                except (json.JSONDecodeError, TypeError):
                    pass  # not wrapped
                if chunk:
                    yield chunk


async def _call_dify_agent_stream(config: AgentConfig, prompt: str) -> AsyncIterator[str]:
    base_url = config.app_id.rstrip("/")
    url = f"{base_url}/v1/workflows/run"

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
            json={
                "inputs": {"query": prompt, "input": prompt},
                "response_mode": "streaming",
                "user": "aicomic-user",
            },
        ) as res:
            if res.status_code >= 400:
                err_text = (await res.aread()).decode("utf-8", "ignore")
                raise RuntimeError(
                    f"Dify workflow request failed: {res.status_code} {err_text[:300]}"
                )

            async for line in res.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                event = obj.get("event")
                data = obj.get("data") or {}
                if event == "text_chunk" and data.get("text"):
                    yield data["text"]
                elif event == "node_finished" and data.get("outputs"):
                    # For workflows that don't use text_chunk, emit final output once
                    out = data["outputs"]
                    txt = out.get("text") or out.get("result") or out.get("output")
                    if isinstance(txt, str):
                        yield txt


# ── Unified non-streaming caller ────────────────────────────────────


async def call_agent(config: AgentConfig, prompt: str) -> str:
    if config.platform == "bailian":
        return await call_bailian_agent(config, prompt)
    if config.platform == "dify":
        return await _call_dify_agent(config, prompt)
    if config.platform == "coze":
        return await _call_coze_agent(config, prompt)
    raise ValueError(f"Unsupported agent platform: {config.platform}")


# ── Bailian (DashScope) ─────────────────────────────────────────────


async def call_bailian_agent(config: AgentConfig, prompt: str) -> str:
    url = f"https://dashscope.aliyuncs.com/api/v1/apps/{config.app_id}/completion"

    async with httpx.AsyncClient(timeout=300.0) as client:
        res = await client.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
            json={"input": {"prompt": prompt}, "parameters": {}},
        )

    if res.status_code >= 400:
        err_text = res.text
        raise RuntimeError(
            f"Bailian agent request failed: {res.status_code} {err_text[:300]}"
        )

    obj = res.json()

    if obj.get("code"):
        raise RuntimeError(
            f"Bailian agent error [{obj['code']}]: {obj.get('message', 'unknown')}"
        )

    raw_text = (obj.get("output") or {}).get("text")
    if not raw_text:
        raise RuntimeError("Bailian agent returned empty output")
    text = raw_text

    # Bailian's workflow mode wraps the result in {"result": "..."} — unwrap it.
    try:
        wrapper = json.loads(text)
        if isinstance(wrapper, dict) and isinstance(wrapper.get("result"), str):
            text = wrapper["result"]
    except (json.JSONDecodeError, TypeError):
        pass  # text is not a JSON wrapper — use the raw value

    return text


# ── Dify ─────────────────────────────────────────────────────────────
# API: POST {appId}/v1/workflows/run  (appId is the Dify instance base URL)
#  or  POST https://api.dify.ai/v1/workflows/run
# Auth: Bearer {apiKey}
# Body: { inputs: { query: prompt }, response_mode: "blocking", user: "aicomic" }
# Response: { data: { outputs: { result: "..." } } }


async def _call_dify_agent(config: AgentConfig, prompt: str) -> str:
    # app_id is the Dify base URL (e.g. https://api.dify.ai or a self-hosted URL)
    base_url = config.app_id.rstrip("/")
    url = f"{base_url}/v1/workflows/run"

    async with httpx.AsyncClient(timeout=300.0) as client:
        res = await client.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
            json={
                "inputs": {"query": prompt},
                "response_mode": "blocking",
                "user": "aicomic-user",
            },
        )

    if res.status_code >= 400:
        raise RuntimeError(
            f"Dify workflow request failed: {res.status_code} {res.text[:300]}"
        )

    obj = res.json()

    if obj.get("code"):
        raise RuntimeError(f"Dify error [{obj['code']}]: {obj.get('message', 'unknown')}")

    data = obj.get("data") or {}
    if data.get("error"):
        raise RuntimeError(f"Dify workflow execution failed: {data['error']}")

    outputs = data.get("outputs")
    if not outputs:
        raise RuntimeError("Dify workflow returned empty output")

    text = (
        outputs.get("result")
        or outputs.get("text")
        or outputs.get("output")
        or (next(iter(outputs.values()), None))
    )
    if not text:
        raise RuntimeError(f"Dify workflow output is empty: {json.dumps(outputs)}")

    return text


# ── Coze ─────────────────────────────────────────────────────────────
# API: POST https://api.coze.cn/v1/workflow/run
# Auth: Bearer {apiKey} (Personal Access Token)
# Body: { workflow_id: appId, parameters: { input: prompt } }
# Response: { code: 0, data: "..." } or { code: 0, data: "{json}" }


async def _call_coze_agent(config: AgentConfig, prompt: str) -> str:
    url = "https://api.coze.cn/v1/workflow/run"

    async with httpx.AsyncClient(timeout=300.0) as client:
        res = await client.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
            json={"workflow_id": config.app_id, "parameters": {"input": prompt}},
        )

    if res.status_code >= 400:
        raise RuntimeError(
            f"Coze workflow request failed: {res.status_code} {res.text[:300]}"
        )

    obj = res.json()

    if obj.get("code") != 0:
        raise RuntimeError(f"Coze error [{obj.get('code')}]: {obj.get('msg', 'unknown')}")

    if not obj.get("data"):
        raise RuntimeError("Coze workflow returned empty output")

    # Coze workflow returns a JSON string like {"result":"..."} — extract the result.
    try:
        parsed = json.loads(obj["data"])
        if isinstance(parsed, dict) and parsed.get("result") is not None:
            return parsed["result"]
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("[Coze] failed to parse data field as JSON, returning raw value: %s", e)

    return obj["data"]


# ── JSON extraction ─────────────────────────────────────────────────

import re as _re

_CODE_BLOCK_RE = _re.compile(r"```(?:json)?\s*\n?([\s\S]*?)\n?```")
_JSON_RE = _re.compile(r"(\[[\s\S]*\]|\{[\s\S]*\})")


def _extract_json(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _JSON_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


# ── Schema validation ───────────────────────────────────────────────

# "script_outline" | "script_generate" | "script_parse" | "character_extract"
# | "shot_split" | "keyframe_prompts" | "video_prompts" | "ref_image_prompts"
# | "ref_video_prompts"
AgentCategory = str


def validate_agent_output(category: AgentCategory, raw_text: str) -> Any:
    json_str = _extract_json(raw_text)
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        raise RuntimeError(
            "The agent's output is not valid JSON. Please adjust the agent's "
            f"output format.\nRaw output: {raw_text[:500]}"
        )

    logger.info(
        "[AgentValidate] category=%s, parsed keys: %s",
        category,
        list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__,
    )
    logger.info("[AgentValidate] rawText (first 1000): %s", raw_text[:1000])

    if category in ("script_outline", "script_generate"):
        # Both return free-form text — wrap in {outline}/{script} loosely
        return _validate_script_outline(parsed)
    if category == "script_parse":
        return _validate_script_parse(parsed)
    if category == "character_extract":
        return _validate_character_extract(parsed)
    if category == "shot_split":
        return _validate_shot_split(parsed)
    if category in ("keyframe_prompts", "video_prompts", "ref_image_prompts", "ref_video_prompts"):
        return parsed
    return parsed


def _assert_field(obj: dict, field: str, type_: str, context: str) -> None:
    if field not in obj or obj[field] is None:
        raise RuntimeError(f'Agent output is missing required field "{field}" ({context})')
    val = obj[field]
    if type_ == "string" and not isinstance(val, str):
        raise RuntimeError(f'Agent output field "{field}" should be a string ({context})')
    if type_ == "number" and not isinstance(val, (int, float)):
        raise RuntimeError(f'Agent output field "{field}" should be a number ({context})')
    if type_ == "array" and not isinstance(val, list):
        raise RuntimeError(f'Agent output field "{field}" should be an array ({context})')


def _validate_script_outline(parsed: Any) -> dict:
    if isinstance(parsed, str):
        return {"outline": parsed}
    _assert_field(parsed, "outline", "string", "script_outline")
    return {"outline": parsed["outline"]}


def _validate_script_parse(parsed: Any) -> Any:
    _assert_field(parsed, "title", "string", "script_parse")
    _assert_field(parsed, "synopsis", "string", "script_parse")
    _assert_field(parsed, "scenes", "array", "script_parse")
    scenes = parsed["scenes"]
    for i, s in enumerate(scenes):
        _assert_field(s, "sceneNumber", "number", f"script_parse.scenes[{i}]")
        _assert_field(s, "setting", "string", f"script_parse.scenes[{i}]")
        _assert_field(s, "description", "string", f"script_parse.scenes[{i}]")
    return parsed


def _validate_character_extract(parsed: Any) -> Any:
    if isinstance(parsed, list):
        for i, c in enumerate(parsed):
            _assert_field(c, "name", "string", f"character[{i}]")
            _assert_field(c, "description", "string", f"character[{i}]")
        return {"characters": parsed}

    _assert_field(parsed, "characters", "array", "character_extract")
    chars = parsed["characters"]
    for i, c in enumerate(chars):
        _assert_field(c, "name", "string", f"characters[{i}]")
        _assert_field(c, "description", "string", f"characters[{i}]")
    return parsed


def _validate_shot_split(parsed: Any) -> Any:
    if not isinstance(parsed, list):
        raise RuntimeError("Agent output shot_split should be an array")
    if len(parsed) == 0:
        return parsed

    first = parsed[0]

    # Format A: grouped by scene [{ sceneTitle, shots: [...] }]
    if isinstance(first, dict) and "sceneTitle" in first and "shots" in first:
        for i, scene in enumerate(parsed):
            _assert_field(scene, "sceneTitle", "string", f"scene[{i}]")
            _assert_field(scene, "shots", "array", f"scene[{i}]")
            for j, shot in enumerate(scene["shots"]):
                _assert_field(shot, "sequence", "number", f"scene[{i}].shots[{j}]")
        return parsed

    # Format B: flat array [{ sequence, prompt/startFrame, ... }] — common agent output
    for i, shot in enumerate(parsed):
        _assert_field(shot, "sequence", "number", f"shot[{i}]")
    # Wrap into Format A so downstream can handle both uniformly
    return [
        {
            "sceneTitle": "Scene 1",
            "sceneDescription": "",
            "lighting": "",
            "colorPalette": "",
            "shots": parsed,
        }
    ]
