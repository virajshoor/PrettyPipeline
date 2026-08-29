from prettypipeline.ollama import _build_user_message, _ollama_format_schema


def test_ollama_user_message_images_base64_only():
    msg = _build_user_message(
        "Invoice total 93.5",
        {"title": "Invoice", "type": "object", "properties": {"total": {"type": "number"}}},
        [{"label": "Figure on page 1", "b64": "abc123", "mime": "image/png"}],
    )
    assert msg["role"] == "user"
    assert msg["images"] == ["abc123"]
    assert "data:" not in str(msg["images"])
    assert "Figure on page 1" in msg["content"]


def test_ollama_format_schema_wraps_data():
    schema = {
        "type": "object",
        "properties": {"total": {"type": "number"}},
    }
    fmt = _ollama_format_schema(schema)
    assert fmt["required"] == ["data", "uncertain_fields"]
    assert "data" in fmt["properties"]
