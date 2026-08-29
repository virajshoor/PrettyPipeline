"""PrettyPipeline — PDF to structured JSON via local OCR + cheap LLM."""

from prettypipeline._version import __version__
from prettypipeline.export import EXPORT_FORMATS, to_csv
from prettypipeline.extract import needs_review, structure
from prettypipeline.ocr import extract_pdf_text, extract_text, looks_like_digital_pdf, merge_texts
from prettypipeline.pipeline import RunResult, run
from prettypipeline.segments import segment_pdf

__all__ = [
    "__version__",
    "EXPORT_FORMATS",
    "RunResult",
    "extract_pdf_text",
    "extract_text",
    "looks_like_digital_pdf",
    "needs_review",
    "run",
    "segment_pdf",
    "merge_texts",
    "structure",
    "to_csv",
]
