"""Parse the first JSON object from model output (handles ``` fences)."""

from __future__ import annotations

import json


def first_json_object(text: str) -> dict | None:
    raw = _strip_json_markdown_fence(text.strip())
    if not raw:
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _strip_json_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if len(lines) < 2:
        return text
    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()
