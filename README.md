# PrettyPipeline

<!-- version: 0.5.1 -->

<p align="left">
  <img
    src="https://raw.githubusercontent.com/virajshoor/PrettyPipeline/main/docs/logo.jpg?v=0.5.1"
    alt="PrettyPipeline — PDF, JSON, and CSV through a pipeline into OCR"
    width="372"
  />
</p>

[![PyPI](https://img.shields.io/pypi/v/prettypipeline-ocr.svg)](https://pypi.org/project/prettypipeline-ocr/)
[![Python](https://img.shields.io/pypi/pyversions/prettypipeline-ocr.svg)](https://pypi.org/project/prettypipeline-ocr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Latest release: [v0.5.1](https://pypi.org/project/prettypipeline-ocr/0.5.1/)** · Requires `pip install prettypipeline-ocr` (add `[ocr]` for Baidu OCR)

Turn PDFs into structured JSON. Text comes from embedded PDF content or local OCR. **Figures are segregated and sent directly to GPT-5.4** — never the whole PDF as one image. Export to CSV and common LLM fine-tuning formats.

## Architecture

```
                         ┌─────────────────────────────────────┐
                         │              PDF input              │
                         └─────────────────┬───────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
              ▼                            ▼                            ▼
     ┌────────────────┐         ┌──────────────────┐         ┌──────────────────┐
     │ Embedded text  │         │ Embedded figures │         │ Scan-only pages  │
     │ (digital PDF)  │         │ (PyMuPDF crop)   │         │ (low-res fallback│
     └───────┬────────┘         └────────┬─────────┘         │  120 DPI max)    │
             │                           │                   └────────┬─────────┘
             │                           │                            │
             │              ┌────────────┴────────────┐               │
             │              │  NOT full-page images   │               │
             │              │  detail=low by default  │               │
             │              └────────────┬────────────┘               │
             │                           │                            │
             └───────────────────────────┼────────────────────────────┘
                                         │
                                         ▼
              ┌────────────────────────┐
              │   Embedded PDF text    │
              └───────────┬────────────┘
                          │ merge (dedupe)
              ┌───────────▼────────────┐
              │ Baidu Unlimited-OCR    │  ← default: always runs ([ocr] extra)
              │ (full document)        │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  Combined document text │
              │  + segregated figures   │──► GPT-5.4 nano
              └────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     ┌─────────┐   ┌─────────────┐  ┌──────────────┐
     │  JSON   │   │needs_review │  │Export formats│
     │ + _meta │   │   flags     │  │ csv alpaca…  │
     └─────────┘   └─────────────┘  └──────────────┘
```

Local OCR (Unlimited-OCR) runs on **every document by default** (requires `[ocr]` extra),
merged with embedded PDF text so nothing is missed — text in images, stamps, scan layers.
Use `--embedded-only` to skip OCR (faster). Use `--force-ocr` for OCR-only.

## Why this is cheap

- **Digital PDFs** — embedded text is free and accurate; no GPU needed.
- **Figures only** — segregated embedded images go to GPT vision at `detail=low` (not full-page raster).
- **Baidu OCR + merge** — full-document OCR merged with embedded text by default (`[ocr]` extra).
- **GPT-5.4 nano** — ~$0.20 / 1M input tokens for structuring ([pricing](https://developers.openai.com/api/docs/models/gpt-5.4-nano)).

## Install

Python **3.10–3.13** (3.12 recommended).

```bash
# Digital PDFs + GPT vision (smallest install)
pip install prettypipeline-ocr

# + local OCR for scans
pip install prettypipeline-ocr[ocr]
```

From this repo:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[ocr,dev]"
```

**NVIDIA Linux:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install prettypipeline-ocr[ocr]
```

**Docker (CUDA):** see `Dockerfile.cuda`.

## Quick start

```bash
export OPENAI_API_KEY=sk-...

# Extract JSON (text + segregated figures → GPT)
prettypipeline run invoice.pdf --schema examples/invoice.schema.json

# Also export fine-tuning formats
prettypipeline run invoice.pdf --schema examples/invoice.schema.json \
  -o out.json --export csv,alpaca,sharegpt,openai,qa

# Convert existing JSON to a training file
prettypipeline export out.json --schema examples/invoice.schema.json \
  --format sharegpt -o train.jsonl
```

Demo invoice:

```bash
prettypipeline run tests/fixtures/sample_invoice.pdf \
  --schema examples/invoice.schema.json -o out.json
```

## Figure segregation (vision)

PrettyPipeline **does not** rasterize entire PDF pages for GPT unless a page is scan-only with no embedded figures.

| Page type | Text source | Vision sent to GPT |
|---|---|---|
| Digital + figures | Embedded PDF text | Embedded figures only (`detail=low`) |
| Digital, no figures | Embedded PDF text | Nothing |
| Scan, no figures | Local OCR or low-res page | One low-res page image (120 DPI) |
| `--no-vision` | Same as above | Disabled |

Images are **visual context only** — no CSV or table dump is generated from figures.

## Local Ollama (WIP)

Run structuring against a local [Ollama](https://ollama.com) server — no OpenAI key required.

```bash
ollama serve
ollama pull llama3.2-vision   # or llava, gemma3, etc.

prettypipeline run invoice.pdf --schema examples/invoice.schema.json --ollama
prettypipeline run invoice.pdf --schema examples/invoice.schema.json \
  --ollama --ollama-host http://localhost:11434 --model llama3.2-vision
```

**How it works** (per [Ollama vision docs](https://docs.ollama.com/capabilities/vision)):
- Native `/api/chat` endpoint (not experimental OpenAI shim for images)
- Figures sent as **raw base64** in the message `images[]` array (official REST format)
- Structured JSON via Ollama `format` JSON schema ([structured outputs](https://docs.ollama.com/capabilities/structured-outputs))

This backend is **WIP** — schema enforcement and vision quality vary by model. Use a vision-capable model.

Environment: `OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `llama3.2-vision`).

You can also point `--base-url http://localhost:11434/v1` at Ollama's OpenAI-compatible API with `--model llama3.2`, but vision images must be base64 data URIs there; the native `--ollama` path is recommended.

## Export formats (LLM training)

| Format | File | Use case |
|---|---|---|
| `csv` | `.csv` | Spreadsheets, flat field/value |
| `alpaca` | `.alpaca.jsonl` | Instruction tuning (LLaMA-Factory, Axolotl) |
| `sharegpt` | `.sharegpt.jsonl` | Multi-turn chat fine-tuning |
| `openai` | `.openai.jsonl` | OpenAI fine-tuning `messages` format |
| `qa` | `.qa.jsonl` | Simple question/answer pairs |

ShareGPT and OpenAI JSONL are preferred for chat models in 2026; Alpaca for single-turn extraction tasks.

## Schema examples

| File | Use case |
|---|---|
| `examples/invoice.schema.json` | Invoices |
| `examples/receipt.schema.json` | Receipts |
| `examples/purchase_order.schema.json` | Purchase orders |

## CLI

```
prettypipeline run FILE.pdf --schema SCHEMA.json
  -o, --output PATH           JSON output (also printed)
  --export csv,alpaca,...     comma-separated training exports
  --no-vision                 skip figure images to GPT
  --image-detail low|auto|high  vision token budget (default: low)
  --embedded-only             skip Baidu OCR supplement (embedded text only)
  --force-ocr                 OCR only — ignore embedded text
  --model, --base-url         OpenAI LLM settings
  --ollama                    local Ollama /api/chat (WIP)
  --ollama-host URL           Ollama server (default localhost:11434)

prettypipeline batch FILES... --schema SCHEMA.json --output-dir DIR
prettypipeline export RESULT.json --schema SCHEMA.json --format FORMAT -o OUT
```

Environment: `OPENAI_API_KEY`, `PRETTYPIPELINE_MODEL`, `PRETTYPIPELINE_BASE_URL`, `OLLAMA_HOST`, `OLLAMA_MODEL`.

## Output

```json
{
  "data": { "invoice_number": "INV-3337", "total": 93.5 },
  "needs_review": [],
  "source": "pdf_text",
  "_meta": {
    "version": "0.5.1",
    "elapsed_ms": 3200,
    "pages": 1,
    "llm_provider": "openai",
    "vision_images": 1,
    "segments": { "text_chars": 842, "figures": 1 },
    "token_usage": { "prompt_tokens": 1100, "completion_tokens": 95, "total_tokens": 1195 }
  }
}
```

## Programmatic use

```python
from prettypipeline import run, segment_pdf, to_csv

result = run("invoice.pdf", "examples/invoice.schema.json", export_formats=["alpaca"], export_stem=Path("out.json"))
seg = segment_pdf("invoice.pdf")  # inspect text vs figures before calling GPT
```

## Hardware

| Device | Role |
|---|---|
| NVIDIA CUDA | Recommended for local OCR (`[ocr]` extra) |
| Apple Silicon MPS | OCR supported; `--force-ocr` may degrade |
| CPU | OCR works, slow |
| None | Digital PDFs + GPT vision work without GPU |

## Development

```bash
pip install -e ".[ocr,dev]"
pytest
python -m build && twine check dist/*
```

## License

MIT. [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) is also MIT.

Fixtures: [Sliced Invoices sample](https://slicedinvoices.com/pdf/wordpress-pdf-invoice-plugin-sample.pdf), [IRS Form W-9](https://www.irs.gov/pub/irs-pdf/fw9.pdf).
