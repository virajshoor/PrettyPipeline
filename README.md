# PrettyPipeline

[![PyPI](https://img.shields.io/pypi/v/prettypipeline-ocr.svg)](https://pypi.org/project/prettypipeline-ocr/)
[![Python](https://img.shields.io/pypi/pyversions/prettypipeline-ocr.svg)](https://pypi.org/project/prettypipeline-ocr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Turn PDFs into structured JSON. OCR runs on **your GPU** when needed. Field extraction uses a **cheap cloud LLM**. You define the schema.

```
PDF  →  embedded text (digital PDFs) or Unlimited-OCR (scans)
                              ↓
        GPT-5.4 nano (OpenAI)  →  JSON + needs_review + _meta
```

## Why this is cheap

Cloud OCR APIs charge per page. PrettyPipeline does vision work locally with [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) (MIT, 3B params, 32K context) when a page has no usable embedded text.

The only paid call is [GPT-5.4 nano](https://developers.openai.com/api/docs/models/gpt-5.4-nano) on document text (~$0.20 / 1M input tokens), prompted with **your** JSON schema.

**Digital PDFs** use embedded text automatically (fast, accurate). OCR runs per-page only for scans, or for everything with `--force-ocr`.

Nulls, uncertain fields, values not found in source text, and garbled OCR are flagged in `needs_review` (one reason per field).

## Install

Python **3.10–3.13** (3.12 recommended).

**Digital PDFs only** (no local OCR, smallest install):

```bash
pip install prettypipeline-ocr
```

**Full install** (local OCR for scans and `--force-ocr`):

```bash
pip install prettypipeline-ocr[ocr]
```

The CLI is `prettypipeline`. From this repo:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[ocr,dev]"
```

**NVIDIA Linux** — install a CUDA PyTorch wheel first if needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install prettypipeline-ocr[ocr]
```

The first OCR run downloads `baidu/Unlimited-OCR` from Hugging Face (~6GB).

**Docker (CUDA):**

```bash
docker build -f Dockerfile.cuda -t prettypipeline:cuda .
docker run --gpus all -e OPENAI_API_KEY=sk-... \
  -v "$PWD/data:/data" prettypipeline:cuda \
  run /data/invoice.pdf --schema /data/schema.json -o /data/out.json
```

## Quick start

```bash
export OPENAI_API_KEY=sk-...

prettypipeline run invoice.pdf --schema examples/invoice.schema.json
prettypipeline run invoice.pdf --schema examples/invoice.schema.json -o out.json
prettypipeline run invoice.pdf --schema examples/invoice.schema.json --ocr-only
```

Demo invoice in this repo:

```bash
prettypipeline run tests/fixtures/sample_invoice.pdf \
  --schema examples/invoice.schema.json -o out.json
```

**Batch:**

```bash
prettypipeline batch ./invoices/*.pdf \
  --schema examples/invoice.schema.json \
  --output-dir ./out/ \
  --summary ./out/summary.json
```

## Schema examples

| File | Use case |
|---|---|
| `examples/invoice.schema.json` | Invoices |
| `examples/receipt.schema.json` | Receipts |
| `examples/purchase_order.schema.json` | Purchase orders |

`--schema` is any JSON Schema — swap the file to extract a different document type.

## CLI

```
prettypipeline run FILE.pdf --schema SCHEMA.json
  -o, --output PATH           write JSON (also printed)
  --ocr-only                  skip OpenAI step
  --force-ocr                 always OCR (skip embedded text)
  --include-ocr-text          include raw text in output
  --device cuda|mps|cpu       override auto-detect
  --dpi N                     PDF raster DPI (default 300)
  --max-length N              OCR cap (8192 MPS, 32768 CUDA/CPU)
  --model NAME                LLM model (default gpt-5.4-nano)
  --base-url URL              OpenAI-compatible API endpoint

prettypipeline batch FILES... --schema SCHEMA.json --output-dir DIR
  --summary PATH              optional batch summary JSON
```

Environment variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required unless `--ocr-only` |
| `PRETTYPIPELINE_MODEL` | Default LLM model |
| `PRETTYPIPELINE_BASE_URL` | OpenAI-compatible base URL |

## Output

```json
{
  "data": {
    "date": "January 25, 2016",
    "vendor": "DEMO - Sliced Invoices",
    "invoice_number": "INV-3337",
    "total": 93.5
  },
  "needs_review": [],
  "source": "pdf_text",
  "_meta": {
    "version": "0.3.0",
    "elapsed_ms": 4200,
    "pages": 1,
    "device": "mps",
    "model": "gpt-5.4-nano",
    "token_usage": {
      "prompt_tokens": 1200,
      "completion_tokens": 180,
      "total_tokens": 1380
    }
  }
}
```

| `source` | meaning |
|---|---|
| `pdf_text` | all pages used embedded text |
| `ocr` | all pages OCR'd |
| `mixed` | some pages embedded, some OCR |

| `needs_review` reason | meaning |
|---|---|
| `null` | model returned null |
| `uncertain` | model flagged low confidence |
| `garbled_ocr` | value or nearby OCR looks corrupted |
| `not_in_source` | value not found in document text |

## Programmatic use

```python
from prettypipeline import run

result = run(
    "invoice.pdf",
    "examples/invoice.schema.json",
)
print(result.data)
print(result.needs_review)
print(result.meta)
```

Lower-level imports: `extract_text`, `structure`, `needs_review`.

## Hardware

| Device | Support |
|---|---|
| NVIDIA CUDA | Native. Recommended for OCR. |
| Apple Silicon (MPS) | Supported with CUDA-call patching. `--force-ocr` may produce garbled output; digital PDFs are unaffected. |
| CPU | Works, slow. |

On MPS, default `--max-length` is **8192** (input + generation). CUDA/CPU default is **32768**.

## Development

```bash
pip install -e ".[ocr,dev]"
pytest
python -m build && twine check dist/*
```

## License

MIT. [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) is also MIT.

Sample invoice: [Sliced Invoices](https://slicedinvoices.com/pdf/wordpress-pdf-invoice-plugin-sample.pdf).
