#!/usr/bin/env python3
"""Build the anonymous TACL revision response letter from its Markdown source."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "revision" / "RESPONSE_LETTER_DRAFT.md"
HEADER = ROOT / "revision" / "response_letter_header.tex"
TEX = ROOT / "revision" / "RESPONSE_LETTER.tex"
PDF = ROOT / "revision" / "RESPONSE_LETTER.pdf"


def _tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise SystemExit(f"required executable not found: {name}")
    return resolved


def _body() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise SystemExit(f"expected an H1 title at the start of {SOURCE}")
    body = "\n".join(lines[1:]).lstrip() + "\n"
    forbidden = ("AUTHOR_INPUT_NEEDED", "TODO", "TBD", "PLACEHOLDER")
    hits = [token for token in forbidden if token in body]
    if hits:
        raise SystemExit(f"unresolved response-letter markers: {hits}")
    return body


def build(*, compile_pdf: bool = True) -> Path:
    pandoc = _tool("pandoc")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", encoding="utf-8", delete=False
    ) as handle:
        handle.write(_body())
        temporary_markdown = Path(handle.name)
    try:
        subprocess.run(
            [
                pandoc,
                str(temporary_markdown),
                "--standalone",
                "--from=markdown",
                "--to=latex",
                "--output",
                str(TEX),
                "--variable",
                "documentclass=article",
                "--variable",
                "classoption=10pt",
                "--variable",
                "classoption=a4paper",
                "--variable",
                "geometry:margin=0.72in",
                "--metadata",
                "title=Response to the Action Editor and Reviewers",
                "--metadata",
                "subtitle=TACL Submission 11241 — Major Revision",
                "--metadata",
                "author=Anonymous authors",
                "--metadata",
                "date=",
                "--include-in-header",
                str(HEADER),
            ],
            cwd=ROOT,
            check=True,
        )
    finally:
        temporary_markdown.unlink(missing_ok=True)

    # Pandoc loads the bookmark package by default. Under the current
    # Tectonic/xdvipdfmx toolchain, those generated outline destinations all
    # resolve to the first page, which can make Preview appear to jump from
    # page 1 to the final section. The response letter does not need a PDF
    # outline, so disable bookmarks at package load time while retaining normal
    # hyperlink support.
    tex = TEX.read_text(encoding="utf-8")
    bookmark_loader = (
        r"\IfFileExists{bookmark.sty}{\usepackage{bookmark}}{\usepackage{hyperref}}"
    )
    if bookmark_loader not in tex:
        raise SystemExit("expected pandoc bookmark loader was not found")
    TEX.write_text(
        tex.replace(bookmark_loader, r"\usepackage[bookmarks=false]{hyperref}", 1),
        encoding="utf-8",
    )

    if compile_pdf:
        tectonic = _tool("tectonic")
        subprocess.run(
            [tectonic, TEX.name, "--keep-logs"],
            cwd=TEX.parent,
            check=True,
        )
        if not PDF.is_file():
            raise SystemExit(f"expected PDF was not produced: {PDF}")
        return PDF
    return TEX


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tex-only", action="store_true", help="generate LaTeX without compiling PDF"
    )
    args = parser.parse_args()
    output = build(compile_pdf=not args.tex_only)
    print(output)


if __name__ == "__main__":
    main()
