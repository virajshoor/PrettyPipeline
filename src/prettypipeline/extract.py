"""Cheap cloud structuring (GPT-5.4 nano) plus local review flags."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

MODEL = "gpt-5.4-nano"
API_KEY_ENV = "OPENAI_API_KEY"

_GARBLED = re.compile(r"[\ufffd]|[^\w\s.,:$€£¥%/@+#()\\-]{4,}")


def require_api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise SystemExit(
            f"{API_KEY_ENV} is not set. Export it, then re-run:\n"
            f"  export {API_KEY_ENV}=sk-...\n"
            f"  prettypipeline run <file.pdf> --schema <schema.json>"
        )
    return key


def _clean_ocr(ocr_text: str) -> str:
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", " ", ocr_text, flags=re.DOTALL)
    return re.sub(r"<PAGE>", "\n", text)


def structure(ocr_text: str, schema: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
    client = OpenAI(api_key=api_key or require_api_key())
    messages = [
        {
            "role": "system",
            "content": (
                "Extract fields from OCR text into JSON. "
                "Return exactly: {\"data\": <object matching the schema>, "
                "\"uncertain_fields\": [<dotted paths you are not confident about>]}. "
                "Use null when a required or optional field is missing or unreadable. "
                "Do not invent values. Prefer null over a guess. "
                "Ignore repeated garbage tokens from OCR degeneration."
            ),
        },
        {
            "role": "user",
            "content": (
                "JSON schema (target shape for data):\n"
                f"{json.dumps(schema, indent=2)}\n\n"
                "OCR text:\n"
                f"{_clean_ocr(ocr_text)}"
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
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
    return {"data": data, "uncertain_fields": cleaned_uncertain}


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
    ocr_text: str = "",
    uncertain: list[str] | None = None,
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(field: str, reason: str) -> None:
        key = (field, reason)
        if key not in seen:
            seen.add(key)
            flags.append({"field": field, "reason": reason})

    for path in uncertain or []:
        add(path, "uncertain")
    for path, value in _walk(data):
        if value is None:
            add(path, "null")
        elif isinstance(value, str) and (looks_garbled(value) or ocr_near_field_garbled(ocr_text, value)):
            add(path, "garbled_ocr")
    return flags
