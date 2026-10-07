"""MATH-500 loader + answer parsing.

MATH-500 (https://huggingface.co/datasets/HuggingFaceH4/MATH-500): 500 hand-
picked problems from the MATH dataset, used by the o1/R1 papers as their
standard "harder than GSM8K" math benchmark.

Schema per row:
  problem: str
  solution: str
  answer: str       (the numeric/closed-form answer — may be a fraction,
                     algebraic expression, or decimal)
  subject: str
  level: int
  unique_id: str

The original lightweight parser in this module is retained only for legacy
artifact inspection.  Formal revision OOD evaluation must use
``is_correct_math500_symbolic``: a pinned ``math-verify`` symbolic grader over
the complete generated text.  This distinction prevents a last-number fallback
from silently becoming the formal MATH-500 evaluator.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
# Match optional decimal only when digits follow the dot (so "42." -> "42").
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class MATH500Sample:
    id: int
    question: str
    gold_answer: str  # raw answer string (could be number, fraction, expr)
    subject: str
    level: int


def _normalize(s: str) -> str:
    """Light normalization for fuzzy answer match."""
    return s.strip().replace(" ", "").replace(",", "").replace("$", "").lower()


def _try_float(s: str) -> float | None:
    try:
        v = float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None
    import math

    return v if math.isfinite(v) else None


def _strip_outer_boxed(s: str) -> str:
    """If s is '\\boxed{...}', return the inner."""
    m = _BOXED_RE.search(s)
    return m.group(1) if m else s


def parse_predicted_math500(generated_text: str) -> str | None:
    """Best-effort answer extraction from a model's free-text MATH-500 output."""
    # Prefer the LAST \boxed{...} the model wrote (final answer)
    matches = list(_BOXED_RE.finditer(generated_text))
    if matches:
        return matches[-1].group(1).strip()
    # Fall back to the last number-like token (for cases where model forgot box)
    nums = _NUMBER_RE.findall(generated_text)
    if nums:
        return nums[-1].strip()
    return None


def is_correct_math500(pred: str | None, gold: str) -> bool:
    """Legacy lightweight checker for historical artifacts; not formal OOD scoring."""
    if pred is None:
        return False
    p_inner = _strip_outer_boxed(pred)
    g_inner = _strip_outer_boxed(gold)
    pf, gf = _try_float(p_inner), _try_float(g_inner)
    if pf is not None and gf is not None:
        return abs(pf - gf) < 1e-4
    return _normalize(p_inner) == _normalize(g_inner)


def is_correct_math500_symbolic(generated_text: str, gold: str) -> bool:
    """Grade a complete MATH-500 completion with pinned symbolic equivalence.

    ``math-verify==0.9.0`` is the required revision extra.  Gold answers in
    MATH-500 are often bare LaTex expressions, so we explicitly box the gold
    answer before extraction.  Predictions are parsed from the *full* decoded
    completion rather than a hand-selected last number; ``math-verify`` gives
    boxed answers priority while still accepting valid plain expressions.

    A missing/ill-formed expression is an incorrect answer.  An unavailable
    dependency is an explicit configuration error rather than a silent fallback
    to the legacy string/float checker.
    """
    if not isinstance(generated_text, str) or not generated_text.strip():
        return False
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    except ImportError as error:  # pragma: no cover - exercised by deployment configuration
        raise RuntimeError(
            "formal MATH-500 scoring requires math-verify==0.9.0; install geoprobe[revision]"
        ) from error

    configs = [LatexExtractionConfig(), ExprExtractionConfig()]
    normalized_gold = _strip_outer_boxed(gold)
    gold_candidates = parse(r"\\boxed{" + normalized_gold + "}", extraction_config=configs)
    prediction_candidates = parse(generated_text, extraction_config=configs)
    if not gold_candidates or not prediction_candidates:
        return False
    return bool(verify(gold_candidates, prediction_candidates, strict=True))


_MODELSCOPE_REPO = "AI-ModelScope/MATH-500"


def _ensure_jsonl() -> Path:
    """Get MATH-500 test.jsonl via modelscope snapshot_download.

    hf-mirror.com redirects (308) back to huggingface.co which is unreachable
    in some networks; modelscope hosts the file directly on Aliyun OSS.
    """
    cached = (
        Path.home()
        / ".cache"
        / "modelscope"
        / "hub"
        / "datasets"
        / "AI-ModelScope"
        / "MATH-500"
        / "test.jsonl"
    )
    if cached.exists():
        return cached

    from modelscope.hub.snapshot_download import snapshot_download

    repo_path = Path(snapshot_download(_MODELSCOPE_REPO, repo_type="dataset"))
    test = repo_path / "test.jsonl"
    if not test.exists():
        raise FileNotFoundError(f"MATH-500 test.jsonl missing in {repo_path}")
    return test


def load_math500(
    split: Literal["test"] = "test",
    n: int | None = None,
    source: str = "hf-mirror",  # forward-compat kwarg; only hf-mirror implemented
) -> list[MATH500Sample]:
    """Load MATH-500 test split. `source` kwarg ignored — we use direct HF-mirror download."""
    del source  # signature-compat for orchestrator
    if split != "test":
        raise ValueError(f"MATH-500 has only a test split; got {split!r}")
    path = _ensure_jsonl()
    out: list[MATH500Sample] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            row = json.loads(line)
            out.append(
                MATH500Sample(
                    id=i,
                    question=row["problem"],
                    gold_answer=row["answer"],
                    subject=row.get("subject", ""),
                    level=int(row.get("level", 0)),
                )
            )
    return out
