from prettypipeline.extract import looks_garbled, needs_review


def test_null_uncertain_garbled():
    data = {"date": None, "vendor": "Acme", "total": "@@@@####~~~~xx"}
    flags = needs_review(data, ocr_text="Acme invoice", uncertain=["vendor"])
    reasons = {(f["field"], f["reason"]) for f in flags}
    assert ("date", "null") in reasons
    assert ("vendor", "uncertain") in reasons
    assert ("total", "garbled_ocr") in reasons


def test_clean_string_not_garbled():
    assert not looks_garbled("Acme Corp")
    assert looks_garbled("bad\ufffdtext")


if __name__ == "__main__":
    test_null_uncertain_garbled()
    test_clean_string_not_garbled()
    print("ok")
