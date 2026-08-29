import json
from pathlib import Path

from prettypipeline.export import (
    to_alpaca_record,
    to_csv,
    to_openai_record,
    to_qa_record,
    to_sharegpt_record,
    write_export,
)
from prettypipeline.segments import segment_pdf

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "tests/fixtures/sample_invoice.pdf"
SCHEMA = {
    "title": "Invoice",
    "type": "object",
    "properties": {"total": {"type": "number"}, "vendor": {"type": "string"}},
}


def test_segment_digital_pdf():
    seg = segment_pdf(str(SAMPLE))
    assert seg.pages == 1
    assert seg.source in ("pdf_text", "mixed")
    assert "Sliced" in seg.text or "Invoice" in seg.text


def test_to_csv_flattens():
    data = {"vendor": "Acme", "line_items": [{"amount": 10}]}
    csv_text = to_csv(data)
    assert "vendor,Acme" in csv_text.replace(" ", "")
    assert "line_items[0].amount,10" in csv_text.replace(" ", "")


def test_training_record_shapes():
    data = {"total": 93.5, "vendor": "Demo"}
    alpaca = to_alpaca_record(SCHEMA, data, source_text="Invoice total 93.5")
    assert "instruction" in alpaca and "output" in alpaca
    share = to_sharegpt_record(SCHEMA, data)
    assert share["conversations"][0]["from"] == "system"
    openai = to_openai_record(SCHEMA, data)
    assert openai["messages"][-1]["role"] == "assistant"
    qa = to_qa_record(SCHEMA, data)
    assert "question" in qa and "answer" in qa


def test_write_export_alpaca(tmp_path):
    out = tmp_path / "train.jsonl"
    write_export("alpaca", out, schema=SCHEMA, data={"total": 1}, source_text="x")
    row = json.loads(out.read_text().strip())
    assert row["output"] == '{"total": 1}'
