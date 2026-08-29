# PrettyPipeline

[![PyPI](https://img.shields.io/pypi/v/prettypipeline-ocr.svg)](https://pypi.org/project/prettypipeline-ocr/)
[![Python](https://img.shields.io/pypi/pyversions/prettypipeline-ocr.svg)](https://pypi.org/project/prettypipeline-ocr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Turn PDFs into structured JSON. OCR runs on **your GPU**. Field extraction uses a **cheap cloud LLM**. You define the schema.

```
PDF  →  PyMuPDF (pages → images)  →  Unlimited-OCR (local)  →  raw text
                                                              ↓
                                    GPT-5.4 nano (OpenAI)  →  JSON + needs_review
```

## Why this is cheap

Cloud OCR APIs charge per page. PrettyPipeline does the expensive vision work locally with [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) (MIT, 3B params, 32K context). A whole PDF is parsed in one `infer_multi()` pass — no page-chunking.

The only paid call is [GPT-5.4 nano](https://developers.openai.com/api/docs/models/gpt-5.4-nano) on that text (~$0.20 / 1M input tokens), prompted with **your** JSON schema. You pay for a small text completion, not for pixels.

Nulls, model-marked uncertain fields, values not found in the source text, and OCR that looks garbled are flagged in `needs_review` instead of being silently accepted.

**Digital PDFs** — embedded text is used when available (fast, accurate). OCR runs only for scans or with `--force-ocr`.

## Install

Python **3.10–3.13** (3.12 recommended). NVIDIA GPU (CUDA) or Apple Silicon (MPS). CPU works but is slow.

```bash
pip install prettypipeline-ocr
```

The CLI name is `prettypipeline`. From this repo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**NVIDIA Linux** — install a CUDA wheel of PyTorch first if `pip` gave you CPU-only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install prettypipeline-ocr
```

The first run downloads `baidu/Unlimited-OCR` from Hugging Face (~6GB).

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

Digital PDFs use embedded text automatically. Use `--force-ocr` to always run Unlimited-OCR.

## Schema

`--schema` is any JSON Schema. Nothing is hardcoded — swap the file to extract a different document type.

```json
{
  "type": "object",
  "properties": {
    "date": { "type": ["string", "null"] },
    "vendor": { "type": ["string", "null"] },
    "line_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": { "type": ["string", "null"] },
          "amount": { "type": ["number", "null"] }
        }
      }
    },
    "total": { "type": ["number", "null"] }
  }
}
```

See `examples/invoice.schema.json` for a fuller invoice example.

## CLI

```
prettypipeline run FILE.pdf --schema SCHEMA.json
  -o, --output PATH     write JSON to a file (also printed)
  --ocr-only            skip the OpenAI step
  --force-ocr           always OCR, skip embedded PDF text
  --device cuda|mps|cpu override auto-detect (cuda → mps → cpu)
  --dpi N               PDF raster DPI (default 300)
  --max-length N        OCR cap (default 8192 on MPS, 32768 on CUDA/CPU)
```

`OPENAI_API_KEY` is required unless you pass `--ocr-only`. It is never hardcoded.

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
  "source": "pdf_text"
}
```

| `reason` | meaning |
|---|---|
| `null` | model returned null |
| `uncertain` | model marked the field as a guess |
| `garbled_ocr` | extracted text (or nearby OCR) looks corrupted |
| `not_in_source` | extracted value not found in document text |

## Hardware

| Device | Support |
|---|---|
| NVIDIA CUDA (target: RTX 4060 Ti) | Native. Unlimited-OCR is written for CUDA. |
| Apple Silicon (MPS) | Supported. The model hardcodes `.cuda()`; PrettyPipeline remaps those calls to MPS. Short docs: `--max-length 2048` — generation can loop on MPS. |
| CPU | Works, slow. |

## License

MIT. [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) is also MIT.

Sample invoice fixture: [Sliced Invoices](https://slicedinvoices.com/pdf/wordpress-pdf-invoice-plugin-sample.pdf).
