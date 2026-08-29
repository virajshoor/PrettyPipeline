"""CLI: prettypipeline run <file.pdf> --schema <schema.json>"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prettypipeline.extract import needs_review, require_api_key, structure


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="prettypipeline",
        description="PDF → local OCR → cheap LLM JSON extraction.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="OCR a PDF and extract JSON using a schema")
    run.add_argument("pdf", type=Path)
    run.add_argument("--schema", type=Path, required=True)
    run.add_argument("-o", "--output", type=Path, help="Write JSON here (also printed)")
    run.add_argument("--ocr-only", action="store_true", help="Skip the cloud structuring step")
    run.add_argument("--device", choices=("cuda", "mps", "cpu"), default="", help="Override auto device")
    run.add_argument("--dpi", type=int, default=300)
    run.add_argument("--max-length", type=int, default=32768, help="OCR generation cap (model supports 32768)")
    args = p.parse_args(argv)

    if args.cmd != "run":
        p.print_help()
        return 2
    if not args.pdf.is_file():
        print(f"not a file: {args.pdf}", file=sys.stderr)
        return 2
    schema = json.loads(args.schema.read_text())
    from prettypipeline.ocr import ocr_pdf, pick_device

    device = args.device or str(pick_device())
    print(f"OCR device: {device}", file=sys.stderr)
    text = ocr_pdf(str(args.pdf), dpi=args.dpi, device=args.device, max_length=args.max_length)
    if args.ocr_only:
        result = {"ocr_text": text, "data": None, "needs_review": []}
        _emit(result, args.output)
        return 0
    require_api_key()
    extracted = structure(text, schema)
    result = {
        "data": extracted["data"],
        "needs_review": needs_review(extracted["data"], text, extracted["uncertain_fields"]),
        "ocr_text": text,
    }
    _emit(result, args.output)
    return 0


def _emit(result: dict, output: Path | None) -> None:
    blob = json.dumps(result, indent=2, ensure_ascii=False)
    print(blob)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(blob + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
