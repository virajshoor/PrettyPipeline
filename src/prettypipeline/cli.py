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
    run.add_argument("--force-ocr", action="store_true", help="Always OCR, skip embedded PDF text")
    run.add_argument("--device", choices=("cuda", "mps", "cpu"), default="", help="Override auto device")
    run.add_argument("--dpi", type=int, default=300)
    run.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="OCR generation cap (default: 8192 on MPS, 32768 on CUDA/CPU)",
    )
    args = p.parse_args(argv)

    if args.cmd != "run":
        p.print_help()
        return 2
    if not args.pdf.is_file():
        print(f"not a file: {args.pdf}", file=sys.stderr)
        return 2
    schema = json.loads(args.schema.read_text())
    from prettypipeline.ocr import default_max_length, extract_text, pick_device

    device = args.device or str(pick_device())
    max_length = args.max_length if args.max_length is not None else default_max_length(args.device)
    print(f"device: {device}", file=sys.stderr)
    text, source = extract_text(
        str(args.pdf),
        force_ocr=args.force_ocr,
        dpi=args.dpi,
        device=args.device,
        max_length=max_length,
    )
    print(f"text source: {source}", file=sys.stderr)
    if args.ocr_only:
        result = {"data": None, "needs_review": [], "source": source, "ocr_text": text}
        _emit(result, args.output, include_text=True)
        return 0
    require_api_key()
    extracted = structure(text, schema)
    result = {
        "data": extracted["data"],
        "needs_review": needs_review(extracted["data"], text, extracted["uncertain_fields"]),
        "source": source,
    }
    _emit(result, args.output, include_text=False)
    return 0


def _emit(result: dict, output: Path | None, *, include_text: bool) -> None:
    out = dict(result)
    if not include_text:
        out.pop("ocr_text", None)
    blob = json.dumps(out, indent=2, ensure_ascii=False)
    print(blob)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(blob + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
