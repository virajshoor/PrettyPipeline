"""Export extraction results to CSV and LLM fine-tuning formats."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

EXPORT_FORMATS = ("csv", "alpaca", "sharegpt", "openai", "qa")

_SYSTEM = (
    "Extract structured fields from the document into JSON matching the user's schema. "
    "Use null for missing fields. Do not invent values."
)


def _instruction(schema: dict[str, Any]) -> str:
    title = schema.get("title") or "document"
    return (
        f"Extract {title} fields from the document into JSON matching this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def _flatten_rows(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            path = f"{prefix}.{k}" if prefix else k
            rows.extend(_flatten_rows(v, path))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            rows.extend(_flatten_rows(v, f"{prefix}[{i}]"))
    else:
        rows.append((prefix, data))
    return rows


def to_csv(data: Any) -> str:
    """Flatten extracted JSON to two-column CSV (field, value)."""
    rows = _flatten_rows(data)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["field", "value"])
    for field, value in rows:
        w.writerow([field, "" if value is None else value])
    return buf.getvalue()


def to_alpaca_record(
    schema: dict[str, Any],
    data: Any,
    *,
    source_text: str = "",
) -> dict[str, str]:
    return {
        "instruction": _instruction(schema),
        "input": source_text[:8000] if source_text else "Document provided (text and/or figures).",
        "output": json.dumps(data, ensure_ascii=False),
    }


def to_sharegpt_record(
    schema: dict[str, Any],
    data: Any,
    *,
    source_text: str = "",
) -> dict[str, Any]:
    user = (
        f"{_instruction(schema)}\n\nDocument text:\n{source_text[:6000]}"
        if source_text
        else _instruction(schema)
    )
    return {
        "conversations": [
            {"from": "system", "value": _SYSTEM},
            {"from": "human", "value": user},
            {"from": "gpt", "value": json.dumps(data, ensure_ascii=False)},
        ]
    }


def to_openai_record(
    schema: dict[str, Any],
    data: Any,
    *,
    source_text: str = "",
) -> dict[str, Any]:
    user = (
        f"{_instruction(schema)}\n\nDocument text:\n{source_text[:6000]}"
        if source_text
        else _instruction(schema)
    )
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
        ]
    }


def to_qa_record(
    schema: dict[str, Any],
    data: Any,
    *,
    source_text: str = "",
) -> dict[str, str]:
    title = schema.get("title") or "document"
    question = f"What are the {title} fields in this document?"
    if source_text:
        question += f"\n\nContext:\n{source_text[:4000]}"
    return {
        "question": question,
        "answer": json.dumps(data, ensure_ascii=False),
    }


def _record_for_format(
    fmt: str,
    schema: dict[str, Any],
    data: Any,
    *,
    source_text: str = "",
) -> Any:
    if fmt == "alpaca":
        return to_alpaca_record(schema, data, source_text=source_text)
    if fmt == "sharegpt":
        return to_sharegpt_record(schema, data, source_text=source_text)
    if fmt == "openai":
        return to_openai_record(schema, data, source_text=source_text)
    if fmt == "qa":
        return to_qa_record(schema, data, source_text=source_text)
    raise ValueError(f"unsupported format: {fmt}")


def write_export(
    fmt: str,
    path: str | Path,
    *,
    schema: dict[str, Any],
    data: Any,
    source_text: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        path.write_text(to_csv(data), encoding="utf-8")
        return
    record = _record_for_format(fmt, schema, data, source_text=source_text)
    if path.suffix == ".jsonl":
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_exports_from_result(
    formats: list[str],
    output_stem: Path,
    *,
    schema: dict[str, Any],
    result: dict[str, Any],
    source_text: str = "",
) -> list[Path]:
    """Write one or more export files next to a JSON result."""
    data = result.get("data")
    if data is None:
        return []
    written: list[Path] = []
    for fmt in formats:
        fmt = fmt.strip().lower()
        if fmt not in EXPORT_FORMATS:
            continue
        stem = output_stem.with_suffix("")
        out = stem.with_suffix(".csv") if fmt == "csv" else Path(f"{stem}.{fmt}.jsonl")
        write_export(fmt, out, schema=schema, data=data, source_text=source_text)
        written.append(out)
    return written
