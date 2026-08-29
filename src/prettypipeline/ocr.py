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
        text, _ = model.infer(
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
    else:
        text, _ = model.infer_multi(
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


def _warn_mps_force_ocr(device: str) -> None:
    dev = pick_device(device or None)
    if dev.type == "mps":
        print(
            "warning: --force-ocr on Apple Silicon (MPS) may produce garbled output; "
            "CUDA is recommended for OCR.",
            file=sys.stderr,
        )


def extract_text(
    pdf_path: str,
    *,
    force_ocr: bool = False,
    dpi: int = 300,
    device: str = "",
    max_length: int | None = None,
) -> tuple[str, str]:
    """Return (text, source) where source is 'pdf_text', 'ocr', or 'mixed'."""
    if force_ocr:
        _warn_mps_force_ocr(device)
        return ocr_pdf(pdf_path, dpi=dpi, device=device, max_length=max_length), "ocr"

    if max_length is None:
        max_length = default_max_length(device)

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_hybrid_")
    parts: list[str] = []
    used_pdf_text = False
    used_ocr = False
    model = tokenizer = None

    try:
        multi_page = len(doc) > 1
        for i, page in enumerate(doc):
            page_text = page.get_text().strip()
            if looks_like_digital_pdf(page_text):
                if multi_page:
                    parts.append(f"--- Page {i + 1} ---\n{page_text}")
                else:
                    parts.append(page_text)
                used_pdf_text = True
            else:
                if model is None:
                    model, tokenizer, _ = load_model(device)
                img_path = _rasterize_page(page, dpi, tmp_dir, i)
                ocr_text = _ocr_page_image(model, tokenizer, img_path, max_length)
                if multi_page:
                    parts.append(f"--- Page {i + 1} ---\n{ocr_text}")
                else:
                    parts.append(ocr_text)
                used_ocr = True
    finally:
        doc.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not parts:
        return ocr_pdf(pdf_path, dpi=dpi, device=device, max_length=max_length), "ocr"

    source = "mixed" if used_pdf_text and used_ocr else ("pdf_text" if used_pdf_text else "ocr")
    return "\n\n".join(parts), source
