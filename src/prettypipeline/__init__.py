"""PrettyPipeline — PDF to structured JSON via local OCR + cheap LLM."""

from prettypipeline._version import __version__
from prettypipeline.extract import needs_review, structure
from prettypipeline.ocr import extract_pdf_text, extract_text, looks_like_digital_pdf
from prettypipeline.pipeline import RunResult, run

__all__ = [
    "__version__",
    "RunResult",
    "extract_pdf_text",
    "extract_text",
    "looks_like_digital_pdf",
    "needs_review",
    "run",
    "structure",
]
