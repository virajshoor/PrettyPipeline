"""Local Ollama server backend (WIP).

Uses Ollama's native /api/chat endpoint:
- Images: base64 strings in message ``images`` (no data-URI prefix) — official REST format.
- Structured JSON: ``format`` JSON schema parameter.

Docs: https://docs.ollama.com/capabilities/vision
      https://docs.ollama.com/capabilities/structured-outputs
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from prettypipeline.extract import _make_strict, clean_ocr_output

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")


def _ollama_format_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict = _make_strict(schema)
    return {
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
    }


def _build_user_message(
    source_text: str,
    schema: dict[str, Any],
    images: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    cleaned = clean_ocr_output(source_text) if "<|det|>" in source_text or "<PAGE>" in source_text else source_text
    content = (
        "Target schema for data:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Document text:\n"
        f"{cleaned or '(no embedded text — use attached figures if present)'}"
    )
    if images:
        labels = "\n".join(f"- {img.get('label', 'Figure')}" for img in images)
        content += (
            f"\n\n{len(images)} figure(s) attached via Ollama images[] for visual context:\n{labels}\n"
            "Use figures to fill schema fields. Return JSON matching the format schema."
        )

    msg: dict[str, Any] = {"role": "user", "content": content}
    if images:
        # Official REST API: raw base64, no data: URI prefix (SDK accepts paths; REST expects base64).
        msg["images"] = [img["b64"] for img in images]
    return msg


def structure_ollama(
    source_text: str,
    schema: dict[str, Any],
    *,
    host: str | None = None,
    model: str | None = None,
    images: list[dict[str, Any]] | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Call Ollama /api/chat with structured output and optional vision images."""
    base = (host or DEFAULT_OLLAMA_HOST).rstrip("/")
    model_name = model or DEFAULT_OLLAMA_MODEL
    url = f"{base}/api/chat"

    payload = {
        "model": model_name,
        "stream": False,
        "format": _ollama_format_schema(schema),
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract fields from document text and any attached figures into JSON. "
                    "Use null when a field is missing or unreadable. "
                    "Do not invent values. Prefer null over a guess. "
                    "List dotted field paths you are not confident about in uncertain_fields."
                ),
            },
            _build_user_message(source_text, schema, images),
        ],
        "options": {"temperature": 0},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base}. Start the server: ollama serve\n  {exc.reason}"
        ) from exc

    raw = (body.get("message") or {}).get("content") or "{}"
    parsed = json.loads(raw)
    data_out = parsed.get("data", parsed)
    uncertain = parsed.get("uncertain_fields") or []
    if not isinstance(uncertain, list):
        uncertain = []
    cleaned_uncertain = []
    for x in uncertain:
        path = str(x)
        if path.startswith("data."):
            path = path[5:]
        cleaned_uncertain.append(path)

    usage = {
        "prompt_tokens": body.get("prompt_eval_count"),
        "completion_tokens": body.get("eval_count"),
        "total_tokens": (body.get("prompt_eval_count") or 0) + (body.get("eval_count") or 0) or None,
    }

    return {
        "data": data_out,
        "uncertain_fields": cleaned_uncertain,
        "token_usage": usage,
        "vision_images": len(images or []),
        "llm_provider": "ollama",
        "ollama_model": model_name,
    }


def warn_wip() -> None:
    print(
        "warning: Ollama backend is WIP — structured output and vision may be buggy; "
        "use a vision model (e.g. llama3.2-vision, llava, gemma3).",
        file=sys.stderr,
    )
