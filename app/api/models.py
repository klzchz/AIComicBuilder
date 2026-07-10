"""Models router — port of models/list/route.ts.

POST /models/list — probe a provider for its available models. Static lists
for kling / ucloud-seedance / wan / dashscope; live fetch for OpenAI-protocol
and Gemini endpoints. (Model display names translated from Chinese.)
"""
from __future__ import annotations

import urllib.parse

import httpx
from fastapi import APIRouter, Request

from app.api._common import json_error

router = APIRouter()

_STATIC_MODELS = {
    "kling": [
        {"id": "kling-v1", "name": "Kling v1"},
        {"id": "kling-v1-5", "name": "Kling v1.5"},
        {"id": "kling-v1-6", "name": "Kling v1.6"},
        {"id": "kling-v2", "name": "Kling v2"},
        {"id": "kling-v2-new", "name": "Kling v2 New"},
        {"id": "kling-v2-1", "name": "Kling v2.1"},
        {"id": "kling-v2-master", "name": "Kling v2 Master"},
        {"id": "kling-v2-1-master", "name": "Kling v2.1 Master"},
        {"id": "kling-v2-5-turbo", "name": "Kling v2.5 Turbo"},
    ],
    "ucloud-seedance": [
        {"id": "doubao-seedance-1-5-pro-251215", "name": "Seedance 1.5 Pro (UCloud)"},
        {"id": "doubao-seedance-2-0-260128", "name": "Seedance 2.0 (UCloud)"},
    ],
    "wan": [
        {"id": "wan2.7-t2v", "name": "Wan 2.7 Text-to-Video"},
        {"id": "wan2.7-r2v", "name": "Wan 2.7 Reference-to-Video"},
        {"id": "wan2.6-t2v", "name": "Wan 2.6 Text-to-Video"},
        {"id": "wan2.6-i2v-flash", "name": "Wan 2.6 Image-to-Video Flash"},
        {"id": "wan2.6-i2v", "name": "Wan 2.6 Image-to-Video"},
        {"id": "wan2.6-r2v", "name": "Wan 2.6 Reference-to-Video"},
        {"id": "wan2.6-r2v-flash", "name": "Wan 2.6 Reference-to-Video Flash"},
    ],
    "dashscope": [
        {"id": "wan2.7-image-pro", "name": "Wan 2.7 Image Pro (4K)"},
        {"id": "wan2.7-image", "name": "Wan 2.7 Image"},
        {"id": "qwen-image-2.0-pro", "name": "Qwen Image 2.0 Pro"},
        {"id": "qwen-image-2.0", "name": "Qwen Image 2.0"},
        {"id": "qwen-image-max", "name": "Qwen Image Max"},
        {"id": "qwen-image-plus", "name": "Qwen Image Plus"},
        {"id": "z-image-turbo", "name": "Z-Image Turbo"},
    ],
}


def _build_models_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    # If baseUrl already ends with /v1, don't duplicate
    if url.endswith("/v1"):
        return url + "/models"
    return url + "/v1/models"


async def _fetch_models(base_url: str, api_key: str) -> list[dict]:
    url = _build_models_url(base_url)
    print(f"[models/list] Fetching: {url}")
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    if res.status_code >= 400:
        raise RuntimeError(f"{res.status_code} {res.text[:200]}")
    data = res.json()
    if not isinstance(data.get("data"), list):
        raise RuntimeError("Unexpected response format: missing data array")
    return [{"id": m["id"], "name": m["id"]} for m in data["data"]]


async def _fetch_gemini_models(base_url: str, api_key: str) -> list[dict]:
    base = base_url.rstrip("/")
    url = f"{base}/v1beta/models?key={urllib.parse.quote(api_key)}"
    print(f"[models/list] Fetching Gemini: {url.replace(api_key, '***')}")
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.get(url)
    if res.status_code >= 400:
        raise RuntimeError(f"{res.status_code} {res.text[:200]}")
    data = res.json()
    if not isinstance(data.get("models"), list):
        raise RuntimeError("Unexpected Gemini response format: missing models array")
    out = []
    for m in data["models"]:
        model_id = (m.get("name") or "").removeprefix("models/")
        out.append({"id": model_id, "name": m.get("displayName") or model_id})
    return out


@router.post("/models/list")
async def list_models(request: Request):
    try:
        body = await request.json()
        protocol = body.get("protocol")

        if protocol in _STATIC_MODELS:
            return {"models": _STATIC_MODELS[protocol]}

        if not body.get("baseUrl"):
            return json_error(400, "Base URL is required")
        if not body.get("apiKey"):
            return json_error(400, "API Key is required")

        if protocol == "gemini":
            models = await _fetch_gemini_models(body["baseUrl"], body["apiKey"])
        else:
            models = await _fetch_models(body["baseUrl"], body["apiKey"])
        return {"models": models}
    except Exception as err:  # noqa: BLE001 — mirror TS catch-all
        message = str(err)
        print(f"[models/list] Error: {message}")
        return json_error(502, message)
