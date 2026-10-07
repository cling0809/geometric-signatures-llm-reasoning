#!/usr/bin/env python3
"""Build the anonymized original TACL decision letter and reviews."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "revision" / "ORIGINAL_DECISION_AND_REVIEWS_ANONYMIZED.md"
HEADER = ROOT / "revision" / "decision_letter_header.tex"
TEX = ROOT / "revision" / "ORIGINAL_DECISION_AND_REVIEWS_ANONYMIZED.tex"
PDF = ROOT / "revision" / "ORIGINAL_DECISION_AND_REVIEWS_ANONYMIZED.pdf"


def _tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise SystemExit(f"required executable not found: {name}")
    return resolved


def build() -> Path:
    text = SOURCE.read_text(encoding="utf-8")
    # Author-identity tokens live in a private, gitignored file that is never
    # part of the anonymous release; the archived copy of this script therefore
    # carries no author names.  Missing file -> identity check is skipped.
    identity_file = ROOT / "revision" / ".private" / "author_identity_tokens.txt"
    identity_tokens = tuple(
        line.strip()
        for line in identity_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ) if identity_file.is_file() else ()
    forbidden = identity_tokens + (
        "AUTHOR_INPUT_NEEDED",
        "TODO",
        "TBD",
        "PLACEHOLDER",
    )
    hits = [token for token in forbidden if token in text]
    if hits:
        raise SystemExit(f"decision/reviews file is not anonymized: {hits}")

    subprocess.run(
        [
            _tool("pandoc"),
            str(SOURCE),
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
            "geometry:margin=0.78in",
            "--metadata",
            "title=Original Decision Letter and Reviews (Anonymized)",
            "--metadata",
            "author=",
            "--metadata",
            "date=",
            "--include-in-header",
            str(HEADER),
        ],
        cwd=ROOT,
        check=True,
    )

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

    subprocess.run(
        [_tool("tectonic"), TEX.name, "--keep-logs"], cwd=TEX.parent, check=True
    )
    if not PDF.is_file():
        raise SystemExit(f"expected PDF was not produced: {PDF}")
    return PDF


if __name__ == "__main__":
    print(build())
