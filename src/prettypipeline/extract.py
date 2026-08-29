"""Cheap cloud structuring (GPT-5.4 nano) plus local review flags."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from prettypipeline.ocr import clean_ocr_output

DEFAULT_MODEL = os.environ.get("PRETTYPIPELINE_MODEL", "gpt-5.4-nano")
API_KEY_ENV = "OPENAI_API_KEY"

_GARBLED = re.compile(r"[\ufffd]|[^\w\s.,:$€£¥%/@+#()\\-]{4,}")

# Lower number = higher priority when deduplicating review reasons per field.
_REASON_PRIORITY = {
    "garbled_ocr": 0,
    "not_in_source": 1,
    "uncertain": 2,
    "null": 3,
}


def require_api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise SystemExit(
            f"{API_KEY_ENV} is not set. Export it, then re-run:\n"
            f"  export {API_KEY_ENV}=sk-...\n"
            f"  prettypipeline run <file.pdf> --schema <schema.json>"
        )
    return key


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _in_source(value: str, source_text: str) -> bool:
    if not value or not source_text:
        return False
    needle = _normalize(str(value))
    if len(needle) < 3:
        return True
    hay = _normalize(source_text)
    if needle in hay:
        return True
    if len(needle) >= 12:
        words = [w for w in needle.split() if len(w) >= 3]
        if words and sum(1 for w in words if w in hay) / len(words) >= 0.6:
            return True
    return False


def _make_strict(obj: dict[str, Any]) -> dict[str, Any]:
    """OpenAI strict mode requires every property key in required."""
    out = dict(obj)
    if out.get("type") == "object" and "properties" in out:
        props = out["properties"]
        out["additionalProperties"] = False
        out["required"] = list(props.keys())
        out["properties"] = {k: _make_strict(v) if isinstance(v, dict) else v for k, v in props.items()}
    if out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = _make_strict(out["items"])
    return out


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict = _make_strict(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": re.sub(r"[^A-Za-z0-9_]", "_", schema.get("title", "extract"))[:64] or "extract",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "data": strict,
                    "uncertain_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["data", "uncertain_fields"],
            },
        },
    }


def _openai_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    kwargs: dict[str, Any] = {"api_key": api_key or require_api_key()}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def structure(
    source_text: str,
    schema: dict[str, Any],
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    images: list[dict[str, Any]] | None = None,
    image_detail: str = "low",
) -> dict[str, Any]:
    client = _openai_client(api_key, base_url)
    model_name = model or DEFAULT_MODEL
    cleaned = clean_ocr_output(source_text) if "<|det|>" in source_text or "<PAGE>" in source_text else source_text

    prompt_text = (
        "Target schema for data:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Document text:\n"
        f"{cleaned or '(no embedded text — use attached figures if present)'}"
    )
    if images:
        prompt_text += (
            f"\n\n{len(images)} figure(s) attached below for visual context only. "
            "Do not transcribe figures as CSV or tables — use them to fill schema fields."
        )

    user_content: list[dict[str, Any]] | str
    if images:
        parts: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for img in images:
            parts.append({"type": "text", "text": img.get("label", "Figure")})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['mime']};base64,{img['b64']}",
                        "detail": img.get("detail", image_detail),
                    },
                }
            )
        user_content = parts
    else:
        user_content = prompt_text

    messages = [
        {
            "role": "system",
            "content": (
                "Extract fields from document text and any attached figures into JSON. "
                "Use null when a field is missing or unreadable. "
                "Do not invent values. Prefer null over a guess. "
                "List dotted field paths you are not confident about in uncertain_fields."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format=_strict_schema(schema),
    )
    raw = resp.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    data = parsed.get("data", parsed)
    uncertain = parsed.get("uncertain_fields") or []
    if not isinstance(uncertain, list):
        uncertain = []
    cleaned_uncertain = []
    for x in uncertain:
        path = str(x)
        if path.startswith("data."):
            path = path[5:]
        cleaned_uncertain.append(path)

    usage = resp.usage
    token_usage = None
    if usage is not None:
        token_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    return {
        "data": data,
        "uncertain_fields": cleaned_uncertain,
        "token_usage": token_usage,
        "vision_images": len(images or []),
    }


def _walk(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            out.extend(_walk(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk(v, f"{prefix}[{i}]"))
    else:
        out.append((prefix, obj))
    return out


def looks_garbled(value: str) -> bool:
    if not value or not str(value).strip():
        return False
    s = str(value)
    if s.count("\ufffd") >= 1:
        return True
    if _GARBLED.search(s):
        return True
    alnum = sum(c.isalnum() for c in s)
    return len(s) >= 8 and alnum / len(s) < 0.35


def ocr_near_field_garbled(ocr_text: str, value: str) -> bool:
    if not value or not ocr_text:
        return False
    needle = str(value).strip()
    if len(needle) < 3:
        return False
    idx = ocr_text.lower().find(needle.lower())
    if idx < 0:
        return False
    window = ocr_text[max(0, idx - 40) : idx + len(needle) + 40]
    return looks_garbled(window)


def needs_review(
    data: Any,
    source_text: str = "",
    uncertain: list[str] | None = None,
) -> list[dict[str, str]]:
    """One reason per field — highest-priority reason wins."""
    best: dict[str, str] = {}

    def consider(field: str, reason: str) -> None:
        if field not in best:
            best[field] = reason
            return
        if _REASON_PRIORITY[reason] < _REASON_PRIORITY.get(best[field], 99):
            best[field] = reason

    for path in uncertain or []:
        consider(path, "uncertain")
    for path, value in _walk(data):
        if value is None:
            consider(path, "null")
        elif isinstance(value, str):
            if looks_garbled(value) or ocr_near_field_garbled(source_text, value):
                consider(path, "garbled_ocr")
            elif not _in_source(value, source_text):
                consider(path, "not_in_source")
        elif isinstance(value, (int, float)) and not _in_source(str(value), source_text):
            consider(path, "not_in_source")

    return [{"field": field, "reason": reason} for field, reason in sorted(best.items())]
