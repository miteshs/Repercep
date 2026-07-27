#!/usr/bin/env python3
"""Render a Markdown file to an on-brand, printable PDF.

For docs meant to be read on paper or a tablet rather than in a terminal — the
speaker script and investor Q&A, mainly. Uses the same type and colour tokens as
the deck and the benchmark report so a printed page looks like it came from the
same company.

Dependencies are ephemeral by design: run it through uv so nothing is added to
the project venv or the lockfile.

    uv run --with markdown --no-project python scripts/md_to_pdf.py decks/VC_QA.md

Emits <input>.pdf next to the input unless --out is given. Requires Chrome for
the print step (same headless recipe as reports/ and decks/).
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

import markdown

_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

_CSS = """
@page { size: letter; margin: 16mm 15mm 14mm; }
:root{
  --ink:#171B23; --ink-2:#5A6274; --ink-3:#858C9C;
  --line:#DFE2E9; --line-strong:#C6CAD4;
  --teal:#0F6C5C; --teal-bg:#E7F3EF;
  --amber-bg:#FBF1DF; --amber:#A96A10;
  --red:#B03030; --red-bg:#FBECEC;
  --sans:'Archivo',-apple-system,'Helvetica Neue',sans-serif;
  --mono:'SF Mono',Menlo,monospace;
  --serif:Georgia,'Iowan Old Style',serif;
}
*{box-sizing:border-box}
body{font-family:var(--serif);color:var(--ink);font-size:10.4pt;line-height:1.55;margin:0;
  -webkit-font-smoothing:antialiased}
h1{font-family:var(--sans);font-size:22pt;font-weight:800;letter-spacing:-.01em;
  line-height:1.15;margin:0 0 6pt;border-bottom:2px solid var(--ink);padding-bottom:8pt}
h2{font-family:var(--sans);font-size:14pt;font-weight:800;margin:20pt 0 7pt;
  padding-top:9pt;border-top:1px solid var(--line-strong);break-after:avoid}
h3{font-family:var(--sans);font-size:11.4pt;font-weight:700;margin:14pt 0 5pt;
  color:var(--teal);break-after:avoid}
h4{font-family:var(--sans);font-size:10.4pt;font-weight:700;margin:11pt 0 4pt;break-after:avoid}
p{margin:0 0 8pt}
strong{font-family:var(--sans);font-weight:700}
em{font-style:italic}
a{color:var(--teal);text-decoration:none}
ul,ol{margin:0 0 9pt;padding-left:17pt}
li{margin-bottom:4pt}
hr{border:none;border-top:1px solid var(--line-strong);margin:16pt 0}
code{font-family:var(--mono);font-size:.86em;background:#F1F2F5;padding:1px 4px;border-radius:2px}
pre{background:#F7F8FA;border:1px solid var(--line);border-radius:3px;padding:9pt 11pt;
  overflow-x:auto;break-inside:avoid}
pre code{background:none;padding:0;font-size:8.6pt;line-height:1.5}
blockquote{margin:10pt 0;padding:9pt 13pt;background:var(--teal-bg);
  border-left:3px solid var(--teal);break-inside:avoid}
blockquote p{margin:0 0 5pt}
blockquote p:last-child{margin:0}
table{width:100%;border-collapse:collapse;margin:10pt 0 13pt;font-size:9pt;break-inside:avoid}
th{font-family:var(--mono);font-size:7.4pt;letter-spacing:.05em;text-transform:uppercase;
  color:var(--ink-2);text-align:left;padding:0 8pt 6pt 0;border-bottom:1.5px solid var(--ink);
  font-weight:500;vertical-align:bottom}
td{padding:6pt 8pt 6pt 0;border-bottom:1px solid var(--line);vertical-align:top;line-height:1.4}
tr:last-child td{border-bottom:1.5px solid var(--line-strong)}
h1+p em, h1+p strong{color:var(--ink-2)}
.warn{background:var(--amber-bg)}
"""


def _emoji_to_class(body: str) -> str:
    """Tint callout paragraphs that lead with a warning glyph."""
    return re.sub(r"<p>(⚠️|❗)", r'<p class="warn">\1', body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.src.is_file():
        print(f"no such file: {args.src}", file=sys.stderr)
        return 2

    body = markdown.markdown(
        args.src.read_text(),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    body = _emoji_to_class(body)
    title = html.escape(args.src.stem.replace("_", " ").title())
    page = f"<!doctype html><meta charset='utf-8'><title>{title}</title><style>{_CSS}</style>{body}"

    out = args.out or args.src.with_suffix(".pdf")
    tmp = args.src.with_suffix(".render.html")
    tmp.write_text(page)
    try:
        subprocess.run(
            [
                _CHROME,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={out}",
                "--virtual-time-budget=5000",
                f"file://{tmp.resolve()}",
            ],
            check=True,
            capture_output=True,
        )
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{args.src} -> {out}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
