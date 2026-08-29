"""Local OCR via baidu/Unlimited-OCR (Transformers). CUDA, Apple Silicon MPS, or CPU."""

from __future__ import annotations

import os
import shutil
import tempfile
from functools import lru_cache

import pymupdf as fitz
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "baidu/Unlimited-OCR"


def pick_device(explicit: str | None = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _patch_cuda_calls(device: torch.device) -> None:
    """Unlimited-OCR hardcodes Tensor.cuda() and autocast('cuda')."""
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


def _dtype_for(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


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


@lru_cache(maxsize=1)
def load_model(device_str: str = "") -> tuple[object, object, torch.device]:
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


def ocr_pdf(
    pdf_path: str,
    dpi: int = 300,
    device: str = "",
    output_dir: str | None = None,
    max_length: int = 32768,
) -> str:
    model, tokenizer, _ = load_model(device)
    paths, tmp_dir = pdf_to_images(pdf_path, dpi=dpi)
    out = output_dir or tempfile.mkdtemp(prefix="ocr_out_")
    own_out = output_dir is None
    try:
        text, _tokens = model.infer_multi(
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
        return text
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if own_out:
            shutil.rmtree(out, ignore_errors=True)
