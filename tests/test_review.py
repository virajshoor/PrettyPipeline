from prettypipeline.extract import needs_review
from prettypipeline.ocr import (
    clean_ocr_output,
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


if __name__ == "__main__":
    test_null_uncertain_garbled()
    test_not_in_source()
    test_cut_repetition()
    test_digital_pdf_detection()
    test_sample_invoice_has_embedded_text()
    print("ok")
