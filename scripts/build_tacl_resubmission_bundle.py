#!/usr/bin/env python3
"""Build the official single-PDF TACL B-decision resubmission bundle."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from build_anonymized_decision import build as build_decision
from build_revision_response_letter import build as build_response

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "revision"
MANUSCRIPT = ROOT / "paper" / "main.pdf"
TEX = REVISION / "RESUBMISSION_BUNDLE.tex"
PDF = REVISION / "RESUBMISSION_BUNDLE.pdf"


def _tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise SystemExit(f"required executable not found: {name}")
    return resolved


def build() -> Path:
    build_response()
    build_decision()
    if not MANUSCRIPT.is_file():
        raise SystemExit(f"revised manuscript is missing: {MANUSCRIPT}")

    tex = r'''%% Based on the official TACL resubmission-pkg.tex v1.1 skeleton.
\documentclass[11pt,a4paper]{article}
\usepackage{times}
\usepackage{pdfpages}
\usepackage[bookmarks=false,hidelinks]{hyperref}
\pagestyle{plain}
\setlength{\footskip}{55pt}
\title{Resubmission of TACL \#11241\\[0.4em]
\large Geometric Signatures of Post-Training in LLM Reasoning Trajectories:\\
Separating Observability from Transferability}
\author{Anonymous TACL resubmission}
\date{}
\begin{document}
\maketitle
\tableofcontents
\section{Author cover letter responding to the original decision and reviews}
Starts on the next page.
\includepdf[pages=-,fitpaper=true]{RESPONSE_LETTER.pdf}
\section{Revised submission}
Starts on the next page.
\includepdf[pages=-,fitpaper=true]{../paper/main.pdf}
\section{Original decision letter and reviews}
Starts on the next page.
\includepdf[pages=-,fitpaper=true]{ORIGINAL_DECISION_AND_REVIEWS_ANONYMIZED.pdf}
\end{document}
'''
    TEX.write_text(tex, encoding="utf-8")
    subprocess.run(
        [_tool("tectonic"), TEX.name, "--keep-logs"], cwd=REVISION, check=True
    )
    if not PDF.is_file():
        raise SystemExit(f"expected resubmission bundle was not produced: {PDF}")

    # Fail closed if author identities escaped into any of the three anonymized parts.
    pdftotext = _tool("pdftotext")
    extracted = subprocess.check_output([pdftotext, str(PDF), "-"], text=True)
    identity_file = REVISION / ".private" / "author_identity_tokens.txt"
    identity_tokens = tuple(
        line.strip()
        for line in identity_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ) if identity_file.is_file() else ()
    identity_hits = [
        token for token in identity_tokens if token in extracted
    ]
    if identity_hits:
        raise SystemExit(f"resubmission bundle contains author identities: {identity_hits}")
    required = (
        "Original submission",
        "Hai Zhao",
        "justify scale-dependent signature stability",
        "Reviewer A",
        "Reviewer B",
        "Reviewer C",
        "Revised submission",
    )
    missing = [token for token in required if token not in extracted]
    if missing:
        raise SystemExit(f"resubmission bundle is incomplete: {missing}")
    return PDF


if __name__ == "__main__":
    print(build())
