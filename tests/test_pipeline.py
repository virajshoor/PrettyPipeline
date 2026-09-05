import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from prettypipeline import run


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "tests/fixtures/sample_invoice.pdf"
SCHEMA = ROOT / "examples/invoice.schema.json"

FAKE_EXTRACTED = {
    "data": {"invoice_number": "INV-3337", "total": 93.5},
    "uncertain_fields": [],
    "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    "vision_images": 0,
    "llm_provider": "openai",
}


def test_run_ocr_only_digital_pdf():
    result = run(SAMPLE, SCHEMA, ocr_only=True, embedded_only=True, include_ocr_text=True)
    assert result.source == "pdf_text"
    assert result.data is None
    assert "Sliced" in (result.ocr_text or "")


def test_default_digital_pdf_never_loads_ocr_model():
    with (
        mock.patch("prettypipeline.ocr.load_model") as load_model,
        mock.patch("prettypipeline.pipeline.structure", return_value=FAKE_EXTRACTED) as structure,
    ):
        result = run(SAMPLE, SCHEMA, api_key="sk-test")
    load_model.assert_not_called()
    structure.assert_called_once()
    assert result.source == "pdf_text"
    assert result.data["invoice_number"] == "INV-3337"
    assert result.meta["pages"] == 1
    assert result.meta["token_usage"]["total_tokens"] == 15
    assert result.needs_review == []


def test_embedded_only_never_loads_ocr_model():
    with (
        mock.patch("prettypipeline.ocr.load_model") as load_model,
        mock.patch("prettypipeline.pipeline.structure", return_value=FAKE_EXTRACTED),
    ):
        result = run(SAMPLE, SCHEMA, embedded_only=True, api_key="sk-test")
    load_model.assert_not_called()
    assert result.source == "pdf_text"


def test_ocr_supplement_merges_full_document_ocr():
    merged = "Invoice INV-3337 total 93.5\n\n--- OCR supplement (text not in embedded layer) ---\n\nPAID IN FULL"
    with (
        mock.patch("prettypipeline.pipeline.structure", return_value=FAKE_EXTRACTED) as structure,
        mock.patch(
            "prettypipeline.pipeline.ocr_supplement_text", return_value=(merged, True)
        ) as supplement,
    ):
        result = run(SAMPLE, SCHEMA, ocr_supplement=True, api_key="sk-test")
    supplement.assert_called_once()
    assert result.source == "pdf_text+ocr"
    assert "PAID IN FULL" in structure.call_args[0][0]


def test_ocr_supplement_skipped_when_embedded_only():
    with (
        mock.patch("prettypipeline.pipeline.structure", return_value=FAKE_EXTRACTED),
        mock.patch("prettypipeline.pipeline.ocr_supplement_text") as supplement,
    ):
        run(SAMPLE, SCHEMA, embedded_only=True, ocr_supplement=True, api_key="sk-test")
    supplement.assert_not_called()


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="live OpenAI integration")
def test_run_full_pipeline_live():
    result = run(SAMPLE, SCHEMA, embedded_only=True)
    assert result.source == "pdf_text"
    assert result.data["invoice_number"] == "INV-3337"
    assert result.data["total"] == 93.5
    assert result.meta["pages"] == 1
    assert result.meta["version"] == "0.5.1"
    assert result.meta["token_usage"] is not None


def test_run_to_dict_has_meta():
    result = run(SAMPLE, SCHEMA, ocr_only=True, embedded_only=True, include_ocr_text=True)
    d = result.to_dict(include_ocr_text=True)
    assert "_meta" in d
    assert d["ocr_text"]


def test_cli_run_subprocess():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "prettypipeline.cli",
            "run",
            str(SAMPLE),
            "--schema",
            str(SCHEMA),
            "--ocr-only",
            "--embedded-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["source"] == "pdf_text"
