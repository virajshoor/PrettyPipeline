"""Local OCR via baidu/Unlimited-OCR (Transformers). CUDA, Apple Silicon MPS, or CPU."""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from functools import lru_cache

import pymupdf as fitz

MODEL_ID = "baidu/Unlimited-OCR"
MPS_DEFAULT_MAX_LENGTH = 8192


def _require_ocr_deps() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "OCR requires ML dependencies. Install with:\n"
            "  pip install prettypipeline-ocr[ocr]"
        ) from e


def device_name(explicit: str = "") -> str:
    """Human-readable device label; does not require OCR deps."""
    if explicit:
        return explicit
    try:
        return str(pick_device(None))
    except SystemExit:
        return "n/a"


def pick_device(explicit: str | None = None):
    _require_ocr_deps()
    import torch

    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_max_length(device: str = "") -> int:
    try:
        dev = pick_device(device or None)
        return MPS_DEFAULT_MAX_LENGTH if dev.type == "mps" else 32768
    except SystemExit:
        return 32768


def _patch_cuda_calls(device) -> None:
    """Unlimited-OCR hardcodes Tensor.cuda() and autocast('cuda')."""
    import torch

    if device.type == "cuda":
        return

    target = device

    def _tensor_cuda(self, *args, **kwargs):
        return self.to(target)

    def _module_cuda(self, *args, **kwargs):
        return self.to(target)

    _orig_autocast = torch.autocast

    def _autocast(device_type=None, dtype=None, *args, **kwargs):
        if device_type == "cuda":
            if device.type == "mps":
                kwargs.pop("device_type", None)
                try:
                    return _orig_autocast("mps", dtype=dtype, *args, **kwargs)
                except TypeError:
                    return _orig_autocast("cpu", dtype=dtype, *args, **kwargs)
            return _orig_autocast("cpu", dtype=dtype, *args, **kwargs)
        return _orig_autocast(device_type, dtype=dtype, *args, **kwargs)

    torch.Tensor.cuda = _tensor_cuda  # type: ignore[method-assign]
    torch.nn.Module.cuda = _module_cuda  # type: ignore[method-assign]
    torch.autocast = _autocast  # type: ignore[assignment]


def _dtype_for(device):
    import torch

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def count_pdf_pages(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    try:
        return "\n\n".join(page.get_text().strip() for page in doc if page.get_text().strip())
    finally:
        doc.close()


def looks_like_digital_pdf(text: str) -> bool:
    if len(text) < 40:
        return False
    alnum = sum(c.isalnum() for c in text)
    return alnum / max(len(text), 1) >= 0.35


def _normalize_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _looks_garbled_simple(text: str) -> bool:
    if not text or len(text.strip()) < 8:
        return False
    s = text.strip()
    alnum = sum(c.isalnum() for c in s)
    return alnum / len(s) < 0.25


def merge_texts(embedded: str, ocr: str) -> str:
    """
    Combine embedded PDF text with Baidu OCR output.

    Keeps embedded text as the base and appends OCR blocks that are not already
    present (e.g. text inside images, stamps, scan layers).
    """
    embedded = embedded.strip()
    ocr = clean_ocr_output(ocr).strip()

    if not embedded:
        return ocr
    if not ocr:
        return embedded

    hay = _normalize_compare(embedded)
    ocr_norm = _normalize_compare(ocr)
    if ocr_norm in hay or ocr_norm == hay:
        return embedded

    supplements: list[str] = []
    for block in re.split(r"\n{2,}", ocr):
        block = block.strip()
        if len(block) < 5 or _looks_garbled_simple(block):
            continue
        block_norm = _normalize_compare(block)
        if block_norm in hay:
            continue
        words = [w for w in block_norm.split() if len(w) >= 3]
        if words and sum(1 for w in words if w in hay) / len(words) > 0.85:
            continue
        supplements.append(block)

    if not supplements:
        return embedded

    header = "--- OCR supplement (text not in embedded layer) ---"
    return embedded + f"\n\n{header}\n\n" + "\n\n".join(supplements)


def cut_repetition(text: str) -> str:
    """Stop at the first long repeat — Unlimited-OCR can loop on MPS."""
    m = re.search(r"(.)\1{7,}", text)
    if m:
        text = text[: m.start()]
    words = text.split()
    if len(words) >= 8:
        for n in range(4, 1, -1):
            if len(words) >= n * 3 and words[-n:] == words[-2 * n : -n]:
                text = " ".join(words[: -n])
                break
    return text.strip()


def clean_ocr_output(text: str) -> str:
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<PAGE>", "\n", text)
    return cut_repetition(text)


def pdf_to_images(pdf_path: str, dpi: int = 300) -> tuple[list[str], str]:
    """Rasterize PDF pages the way Unlimited-OCR documents: PyMuPDF at `dpi`."""
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    try:
        for i, page in enumerate(doc):
            out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
            page.get_pixmap(matrix=mat).save(out)
            paths.append(out)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        doc.close()
    if not paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError(f"no pages in {pdf_path}")
    return paths, tmp_dir


def _rasterize_page(page, dpi: int, tmp_dir: str, index: int) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    out = os.path.join(tmp_dir, f"page_{index + 1:04d}.png")
    page.get_pixmap(matrix=mat).save(out)
    return out


@lru_cache(maxsize=1)
def load_model(device_str: str = "") -> tuple[object, object, object]:
    _require_ocr_deps()
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = pick_device(device_str or None)
    _patch_cuda_calls(device)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    dtype = _dtype_for(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    )
    model = model.eval().to(device)
    return model, tokenizer, device


def _run_ocr(model, tokenizer, paths: list[str], out: str, max_length: int) -> str:
    if len(paths) == 1:
        result = model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=paths[0],
            output_path=out,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=max_length,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=False,
        )
        if not result or result[0] is None:
            return ""
        text, _ = result
    else:
        result = model.infer_multi(
            tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=paths,
            output_path=out,
            image_size=1024,
            max_length=max_length,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=False,
        )
        if not result or result[0] is None:
            return ""
        text, _ = result
    return clean_ocr_output(text)


def ocr_pdf(
    pdf_path: str,
    dpi: int = 300,
    device: str = "",
    output_dir: str | None = None,
    max_length: int | None = None,
) -> str:
    model, tokenizer, _ = load_model(device)
    if max_length is None:
        max_length = default_max_length(device)
    paths, tmp_dir = pdf_to_images(pdf_path, dpi=dpi)
    out = output_dir or tempfile.mkdtemp(prefix="ocr_out_")
    own_out = output_dir is None
    try:
        return _run_ocr(model, tokenizer, paths, out, max_length)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if own_out:
            shutil.rmtree(out, ignore_errors=True)


def _ocr_page_image(
    model,
    tokenizer,
    image_path: str,
    max_length: int,
) -> str:
    out = tempfile.mkdtemp(prefix="ocr_page_")
    try:
        return _run_ocr(model, tokenizer, [image_path], out, max_length)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _warn_mps_ocr(device: str, *, context: str) -> None:
    try:
        dev = pick_device(device or None)
    except SystemExit:
        return
    if dev.type == "mps":
        print(
            f"warning: Baidu OCR on Apple Silicon (MPS) may be slow or garbled; "
            f"CUDA recommended ({context}).",
            file=sys.stderr,
        )


def fill_scan_text(
    pdf_path: str,
    segments,
    *,
    dpi: int = 300,
    device: str = "",
    max_length: int | None = None,
) -> int:
    """
    OCR only the scan pages of an already-segmented PDF, filling
    ``segments.page_texts`` in place. Returns the number of pages OCR'd.

    Digital pages are never touched — their embedded text is free and accurate.
    """
    scan = [i for i, kind in enumerate(segments.page_kinds) if kind == "scan"]
    if not scan:
        return 0

    try:
        _require_ocr_deps()
    except SystemExit:
        if segments.text.strip():
            print(
                f"note: install prettypipeline-ocr[ocr] to OCR {len(scan)} scan page(s); "
                "using embedded text only",
                file=sys.stderr,
            )
            return 0
        raise SystemExit(
            "No embedded text found and OCR dependencies missing.\n"
            "  pip install prettypipeline-ocr[ocr]"
        ) from None

    _warn_mps_ocr(device, context="scan-page OCR")
    if max_length is None:
        max_length = default_max_length(device)

    model = tokenizer = None
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_scan_")
    filled = 0
    try:
        multi_page = segments.pages > 1
        for i in scan:
            if model is None:
                model, tokenizer, _ = load_model(device)
            img_path = _rasterize_page(doc[i], dpi, tmp_dir, i)
            try:
                page_text = _ocr_page_image(model, tokenizer, img_path, max_length)
            except Exception as exc:
                print(f"warning: OCR failed on page {i + 1} ({exc})", file=sys.stderr)
                continue
            if page_text.strip():
                if multi_page:
                    page_text = f"--- Page {i + 1} ---\n{page_text}"
                segments.page_texts[i] = page_text
                filled += 1
    finally:
        doc.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return filled


def ocr_supplement_text(
    pdf_path: str,
    text: str,
    *,
    dpi: int = 300,
    device: str = "",
    max_length: int | None = None,
) -> tuple[str, bool]:
    """
    Merge full-document Baidu OCR into `text` (catches stamps, signatures,
    text inside figures). Returns (merged_text, added_anything).
    """
    try:
        _require_ocr_deps()
    except SystemExit:
        print(
            "note: install prettypipeline-ocr[ocr] for Baidu OCR text supplement",
            file=sys.stderr,
        )
        return text, False

    _warn_mps_ocr(device, context="OCR supplement")
    try:
        ocr_text = ocr_pdf(pdf_path, dpi=dpi, device=device, max_length=max_length)
    except Exception as exc:
        print(f"warning: Baidu OCR failed ({exc}); using embedded text only", file=sys.stderr)
        return text, False

    if not ocr_text.strip():
        return text, False
    if not text.strip():
        return ocr_text, True
    merged = merge_texts(text, ocr_text)
    if merged == text.strip():
        return text, False
    return merged, True


def extract_text(
    pdf_path: str,
    *,
    force_ocr: bool = False,
    embedded_only: bool = False,
    ocr_supplement: bool = False,
    dpi: int = 300,
    device: str = "",
    max_length: int | None = None,
) -> tuple[str, str]:
    """
    Return (text, source).

    Default: embedded text plus OCR for scan-only pages.
    force_ocr: OCR only, ignore embedded text.
    embedded_only: embedded text only — never loads the OCR model.
    ocr_supplement: also merge full-document OCR (stamps, image text).
    """
    if force_ocr:
        _warn_mps_ocr(device, context="--force-ocr")
        return ocr_pdf(pdf_path, dpi=dpi, device=device, max_length=max_length), "ocr"

    from prettypipeline.segments import segment_pdf

    segments = segment_pdf(pdf_path)
    if not embedded_only:
        fill_scan_text(pdf_path, segments, dpi=dpi, device=device, max_length=max_length)
    text, source = segments.text, segments.source
    if ocr_supplement and not embedded_only:
        text, added = ocr_supplement_text(
            pdf_path, text, dpi=dpi, device=device, max_length=max_length
        )
        if added:
            source = "pdf_text+ocr" if "digital" in segments.page_kinds else "ocr"
    return text, source
