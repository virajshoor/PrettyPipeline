from prettypipeline.extract import needs_review
from prettypipeline.ocr import (
    clean_ocr_output,
    count_pdf_pages,
    cut_repetition,
    extract_pdf_text,
    looks_like_digital_pdf,
)


def test_null_uncertain_garbled():
    data = {"date": None, "vendor": "Acme", "total": "@@@@####~~~~xx"}
    flags = needs_review(data, source_text="Acme invoice", uncertain=["vendor"])
    reasons = {(f["field"], f["reason"]) for f in flags}
    assert ("date", "null") in reasons
    assert ("vendor", "uncertain") in reasons
    assert ("total", "garbled_ocr") in reasons


def test_not_in_source():
    data = {"vendor": "Phantom Corp", "total": 99.0}
    flags = needs_review(data, source_text="DEMO - Sliced Invoices total $93.50")
    reasons = {(f["field"], f["reason"]) for f in flags}
    assert ("vendor", "not_in_source") in reasons


def test_review_dedup_one_reason_per_field():
    data = {"currency": None}
    flags = needs_review(data, uncertain=["currency"])
    by_field = {f["field"]: f["reason"] for f in flags}
    assert by_field["currency"] == "uncertain"
    assert len(flags) == 1


def test_cut_repetition():
    assert cut_repetition("hello zzzzzzzzzz world") == "hello"
    assert cut_repetition("a b c a b c a b c") == "a b c a b c"


def test_digital_pdf_detection():
    assert looks_like_digital_pdf("Invoice 12345 from Acme Corp dated Jan 1 2024 total $100")
    assert not looks_like_digital_pdf("!!! ???")
    assert not looks_like_digital_pdf("ab")


def test_sample_invoice_has_embedded_text():
    text = extract_pdf_text("tests/fixtures/sample_invoice.pdf")
    assert "Sliced Invoices" in text or "SlicedInvoices" in text
    assert looks_like_digital_pdf(text)


def test_count_pdf_pages():
    assert count_pdf_pages("tests/fixtures/sample_invoice.pdf") == 1


def test_clean_ocr_output_strips_det_tokens():
    raw = "<|det|>header [1,2]<|/det|>Hello world"
    assert clean_ocr_output(raw) == "Hello world"
