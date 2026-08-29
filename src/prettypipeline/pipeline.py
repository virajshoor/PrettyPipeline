"""High-level API shared by CLI and library callers."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prettypipeline._version import __version__
from prettypipeline.export import write_exports_from_result
from prettypipeline.extract import needs_review, require_api_key, structure
from prettypipeline.ocr import count_pdf_pages, device_name, extract_text
from prettypipeline.segments import segment_pdf

DEFAULT_MODEL = os.environ.get("PRETTYPIPELINE_MODEL", "gpt-5.4-nano")
DEFAULT_BASE_URL = os.environ.get("PRETTYPIPELINE_BASE_URL", "").strip() or None


@dataclass
class RunResult:
    data: Any | None
    needs_review: list[dict[str, str]]
    source: str
    ocr_text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_ocr_text: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "data": self.data,
            "needs_review": self.needs_review,
            "source": self.source,
            "_meta": self.meta,
        }
        if include_ocr_text and self.ocr_text is not None:
            out["ocr_text"] = self.ocr_text
        return out


def run(
    pdf: str | Path,
    schema: dict[str, Any] | str | Path,
    *,
    ocr_only: bool = False,
    force_ocr: bool = False,
    embedded_only: bool = False,
    use_vision: bool = True,
    image_detail: str = "low",
    device: str = "",
    dpi: int = 300,
    max_length: int | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    include_ocr_text: bool = False,
    export_formats: list[str] | None = None,
    export_stem: Path | None = None,
) -> RunResult:
    """Extract structured JSON from a PDF (same behavior as the CLI)."""
    pdf_path = Path(pdf)
    if isinstance(schema, (str, Path)):
        schema_obj = json.loads(Path(schema).read_text())
    else:
        schema_obj = schema

    t0 = time.perf_counter()
    pages = count_pdf_pages(str(pdf_path))
    device_name_str = device_name(device)
    model_name = model or DEFAULT_MODEL

    segments = segment_pdf(str(pdf_path))
    text, source = extract_text(
        str(pdf_path),
        force_ocr=force_ocr,
        embedded_only=embedded_only,
        dpi=dpi,
        device=device,
        max_length=max_length,
    )

    vision_images = [img.to_api_dict(detail=image_detail) for img in segments.images] if use_vision else []

    meta: dict[str, Any] = {
        "version": __version__,
        "elapsed_ms": 0,
        "pages": pages,
        "device": device_name_str,
        "model": model_name if not ocr_only else None,
        "token_usage": None,
        "vision_images": len(vision_images),
        "segments": {"text_chars": len(text), "figures": len(vision_images), "text_source": source},
    }

    if ocr_only:
        meta["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        return RunResult(
            data=None,
            needs_review=[],
            source=source,
            ocr_text=text if include_ocr_text else None,
            meta=meta,
        )

    key = api_key or require_api_key()
    extracted = structure(
        text,
        schema_obj,
        api_key=key,
        model=model_name,
        base_url=base_url or DEFAULT_BASE_URL,
        images=vision_images or None,
        image_detail=image_detail,
    )
    flags = needs_review(extracted["data"], text, extracted["uncertain_fields"])
    meta["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    meta["token_usage"] = extracted.get("token_usage")
    meta["vision_images"] = extracted.get("vision_images", len(vision_images))

    result = RunResult(
        data=extracted["data"],
        needs_review=flags,
        source=source,
        ocr_text=text if include_ocr_text else None,
        meta=meta,
    )

    if export_formats and export_stem is not None and result.data is not None:
        write_exports_from_result(
            export_formats,
            export_stem,
            schema=schema_obj,
            result=result.to_dict(),
            source_text=text,
        )

    return result
