import json
import subprocess
import sys
from pathlib import Path

from prettypipeline import run


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "tests/fixtures/sample_invoice.pdf"
SCHEMA = ROOT / "examples/invoice.schema.json"


def test_run_ocr_only_digital_pdf():
    result = run(SAMPLE, SCHEMA, ocr_only=True, include_ocr_text=True)
    assert result.source == "pdf_text"
    assert result.data is None
    assert "Sliced" in (result.ocr_text or "")


def test_run_full_pipeline():
    result = run(SAMPLE, SCHEMA)
    assert result.source == "pdf_text"
    assert result.data["invoice_number"] == "INV-3337"
    assert result.data["total"] == 93.5
    assert result.meta["pages"] == 1
    assert result.meta["version"] == "0.3.0"
    assert result.meta["token_usage"] is not None


def test_run_to_dict_has_meta():
    result = run(SAMPLE, SCHEMA, ocr_only=True, include_ocr_text=True)
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
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["source"] == "pdf_text"
