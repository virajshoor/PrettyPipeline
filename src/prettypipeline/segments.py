"""Segregate PDF text from embedded figures — send only figures to vision, not full pages."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import pymupdf as fitz

from prettypipeline.ocr import looks_like_digital_pdf

_MIME = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


@dataclass
class ImageSegment:
    page: int
    index: int
    label: str
    mime: str
    b64: str
    size_bytes: int
    bbox: tuple[float, float, float, float] | None = None

    def to_api_dict(self, *, detail: str = "low") -> dict[str, Any]:
        return {
            "label": self.label,
            "mime": self.mime,
            "b64": self.b64,
            "detail": detail,
            "page": self.page,
            "index": self.index,
        }


@dataclass
class DocumentSegments:
    text: str
    source: str
    images: list[ImageSegment] = field(default_factory=list)
    pages: int = 0

    @property
    def image_count(self) -> int:
        return len(self.images)


def _page_text(page: fitz.Page) -> str:
    return page.get_text().strip()


def _extract_embedded_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
    *,
    seen_xrefs: set[int],
    min_bytes: int,
    max_images: int,
    out: list[ImageSegment],
) -> None:
    if len(out) >= max_images:
        return
    for img_idx, img in enumerate(page.get_images(full=True)):
        if len(out) >= max_images:
            return
        xref = int(img[0])
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            info = doc.extract_image(xref)
        except Exception:
            continue
        raw = info.get("image") or b""
        if len(raw) < min_bytes:
            continue
        ext = (info.get("ext") or "png").lower()
        mime = _MIME.get(ext, f"image/{ext}")
        rects = page.get_image_rects(xref)
        bbox = None
        if rects:
            r = rects[0]
            bbox = (r.x0, r.y0, r.x1, r.y1)
        out.append(
            ImageSegment(
                page=page_num,
                index=img_idx + 1,
                label=f"Embedded figure on page {page_num} (#{img_idx + 1})",
                mime=mime,
                b64=base64.standard_b64encode(raw).decode("ascii"),
                size_bytes=len(raw),
                bbox=bbox,
            )
        )


def _scan_page_image(page: fitz.Page, page_num: int, *, dpi: int = 120) -> ImageSegment | None:
    """Low-res page render for scan-only pages — avoids full-DPI token waste."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    png = page.get_pixmap(matrix=mat).tobytes("png")
    if len(png) < 500:
        return None
    return ImageSegment(
        page=page_num,
        index=1,
        label=f"Scan page {page_num} (low-res context)",
        mime="image/png",
        b64=base64.standard_b64encode(png).decode("ascii"),
        size_bytes=len(png),
        bbox=None,
    )


def segment_pdf(
    pdf_path: str,
    *,
    min_image_bytes: int = 4_000,
    max_images: int = 16,
    scan_fallback_dpi: int = 120,
) -> DocumentSegments:
    """
    Split a PDF into text and figure segments.

    Digital pages: embedded text + embedded images only (never full-page raster).
    Scan pages with no text: one low-res page image for GPT vision context.
    """
    doc = fitz.open(pdf_path)
    text_parts: list[str] = []
    images: list[ImageSegment] = []
    seen_xrefs: set[int] = set()
    used_pdf_text = False
    used_scan_image = False
    multi_page = len(doc) > 1

    try:
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            page_text = _page_text(page)
            has_digital_text = looks_like_digital_pdf(page_text)

            if has_digital_text:
                if multi_page:
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}")
                else:
                    text_parts.append(page_text)
                used_pdf_text = True
                _extract_embedded_images(
                    doc,
                    page,
                    page_num,
                    seen_xrefs=seen_xrefs,
                    min_bytes=min_image_bytes,
                    max_images=max_images,
                    out=images,
                )
            else:
                before = len(images)
                _extract_embedded_images(
                    doc,
                    page,
                    page_num,
                    seen_xrefs=seen_xrefs,
                    min_bytes=min_image_bytes,
                    max_images=max_images,
                    out=images,
                )
                if len(images) == before and len(images) < max_images:
                    seg = _scan_page_image(page, page_num, dpi=scan_fallback_dpi)
                    if seg is not None:
                        images.append(seg)
                        used_scan_image = True

        text = "\n\n".join(text_parts)
        if used_pdf_text and (images and any("Embedded" in i.label for i in images)):
            source = "mixed"
        elif used_pdf_text and not images:
            source = "pdf_text"
        elif used_pdf_text and used_scan_image:
            source = "mixed"
        elif images and not used_pdf_text:
            source = "vision"
        elif used_pdf_text:
            source = "pdf_text"
        else:
            source = "ocr"

        return DocumentSegments(text=text, source=source, images=images, pages=len(doc))
    finally:
        doc.close()
