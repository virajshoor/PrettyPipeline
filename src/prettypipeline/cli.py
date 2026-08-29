"""CLI: prettypipeline run | batch"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prettypipeline.pipeline import run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="prettypipeline",
        description="PDF → local OCR → cheap LLM JSON extraction.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="OCR a PDF and extract JSON using a schema")
    _add_run_args(run_p)
    run_p.add_argument("pdf", type=Path)

    batch_p = sub.add_parser("batch", help="Process multiple PDFs with one schema")
    _add_run_args(batch_p)
    batch_p.add_argument("pdfs", nargs="+", type=Path, help="PDF files or directories")
    batch_p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Write one JSON file per PDF here",
    )
    batch_p.add_argument(
        "--summary",
        type=Path,
        help="Optional path for batch summary JSON",
    )

    args = p.parse_args(argv)

    if args.cmd == "run":
        return _run_one(args)
    if args.cmd == "batch":
        return _run_batch(args)
    p.print_help()
    return 2


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--schema", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, help="Write JSON here (also printed)")
    p.add_argument("--ocr-only", action="store_true", help="Skip the cloud structuring step")
    p.add_argument("--force-ocr", action="store_true", help="Always OCR, skip embedded PDF text")
    p.add_argument("--include-ocr-text", action="store_true", help="Include raw OCR/text in output")
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default="", help="Override auto device")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="OCR generation cap (default: 8192 on MPS, 32768 on CUDA/CPU)",
    )
    p.add_argument("--model", default="", help="LLM model (default: gpt-5.4-nano or PRETTYPIPELINE_MODEL)")
    p.add_argument("--base-url", default="", help="OpenAI-compatible API base URL")


def _run_kwargs(args) -> dict:
    from prettypipeline.ocr import default_max_length, device_name

    device = device_name(args.device)
    max_length = args.max_length if args.max_length is not None else default_max_length(args.device)
    print(f"device: {device}", file=sys.stderr)

    kw = {
        "force_ocr": args.force_ocr,
        "dpi": args.dpi,
        "device": args.device,
        "max_length": max_length,
        "ocr_only": args.ocr_only,
        "include_ocr_text": args.include_ocr_text,
    }
    if args.model:
        kw["model"] = args.model
    if args.base_url:
        kw["base_url"] = args.base_url
    return kw


def _run_one(args) -> int:
    if not args.pdf.is_file():
        print(f"not a file: {args.pdf}", file=sys.stderr)
        return 2
    schema = json.loads(args.schema.read_text())
    kw = _run_kwargs(args)
    result = run(args.pdf, schema, **kw)
    print(f"text source: {result.source}", file=sys.stderr)
    _emit(result.to_dict(include_ocr_text=args.include_ocr_text), args.output)
    return 0


def _collect_pdfs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".pdf":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.glob("*.pdf")))
    return out


def _run_batch(args) -> int:
    pdfs = _collect_pdfs(args.pdfs)
    if not pdfs:
        print("no PDF files found", file=sys.stderr)
        return 2
    schema = json.loads(args.schema.read_text())
    kw = _run_kwargs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"ok": [], "failed": []}
    for pdf in pdfs:
        out_path = args.output_dir / f"{pdf.stem}.json"
        try:
            result = run(pdf, schema, **kw)
            blob = json.dumps(result.to_dict(include_ocr_text=args.include_ocr_text), indent=2, ensure_ascii=False)
            out_path.write_text(blob + "\n")
            summary["ok"].append({"file": str(pdf), "output": str(out_path), "source": result.source})
            print(f"ok {pdf.name} → {out_path}", file=sys.stderr)
        except SystemExit as e:
            summary["failed"].append({"file": str(pdf), "error": str(e)})
            print(f"fail {pdf.name}: {e}", file=sys.stderr)
        except Exception as e:
            summary["failed"].append({"file": str(pdf), "error": str(e)})
            print(f"fail {pdf.name}: {e}", file=sys.stderr)

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if not summary["failed"] else 1


def _emit(result: dict, output: Path | None) -> None:
    blob = json.dumps(result, indent=2, ensure_ascii=False)
    print(blob)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(blob + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
