"""GSM8K loader + answer parsing.

GSM8K format (one JSON object per line):
    {"question": "...", "answer": "step-by-step reasoning ... #### <number>"}

We treat the integer/float after `####` as the gold label. For predictions, we
look for (in order): `\\boxed{...}`, the last number in the text, or None.

Source strategy: bypass modelscope's MsDataset (fragile against datasets-lib
version skew, see lessons_learned/2026-05-17_modelscope-msdataset-version-skew.md)
and pull the upstream JSONL directly from the public Aliyun OSS bucket that
ModelScope itself uses. Cached under $GEOPROBE_CACHE/datasets/gsm8k/.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from geoprobe.utils.paths import cache_dir

_GOLD_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
_BOXED_RE = re.compile(r"\\boxed\{\s*([-+]?\d[\d,]*\.?\d*)\s*\}")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_NEXT_PROBLEM_MARKERS = ("\n\nProblem:", "\nProblem:")

# Same files MsDataset would have fetched, just direct.
_OSS_URL = "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/gsm8k/{split}.jsonl"


@dataclass(frozen=True)
class GSM8KSample:
    id: int
    question: str
    gold_answer: float
    gold_text: str


def parse_gold(answer_text: str) -> float:
    m = _GOLD_RE.search(answer_text)
    if not m:
        raise ValueError(f"no #### answer found in: {answer_text[-80:]!r}")
    return float(m.group(1).replace(",", ""))


def _safe_float(s: str) -> float | None:
    """float() that returns None for non-finite (inf/nan) — e.g. '1e500' -> inf -> None."""
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_predicted(generated_text: str) -> float | None:
    # Base LMs sometimes answer the current problem, then continue by inventing
    # a fresh "Problem:" block. GSM8K evaluation should only parse the first
    # completion, not numbers from synthetic follow-up examples.
    text = generated_text
    for marker in _NEXT_PROBLEM_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
            break

    m = _BOXED_RE.search(text)
    if m:
        v = _safe_float(m.group(1))
        if v is not None:
            return v
    nums = _NUMBER_RE.findall(text)
    if not nums:
        return None
    return _safe_float(nums[-1])


def is_correct(pred: float | None, gold: float, atol: float = 1e-4) -> bool:
    if pred is None:
        return False
    return abs(pred - gold) <= atol


def _ensure_jsonl(split: str) -> Path:
    """Download GSM8K <split>.jsonl into local cache if not already present."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    target_dir = cache_dir() / "datasets" / "gsm8k"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{split}.jsonl"
    if not target.exists():
        url = _OSS_URL.format(split=split)
        tmp = target.with_suffix(".jsonl.tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(target)
    return target


def load_gsm8k(
    split: Literal["train", "test"] = "test",
    n: int | None = None,
    source: Literal["oss", "modelscope", "huggingface"] = "oss",
) -> list[GSM8KSample]:
    """Load GSM8K. `source` kept for forward compatibility; only 'oss' implemented."""
    if source != "oss":
        # We accept the kwarg from configs so existing yaml stays valid, but
        # only the direct-OSS path is implemented after the MsDataset version skew.
        # See lessons_learned/2026-05-17_modelscope-msdataset-version-skew.md
        pass

    path = _ensure_jsonl(split)
    out: list[GSM8KSample] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            row = json.loads(line)
            gold = parse_gold(row["answer"])
            out.append(
                GSM8KSample(
                    id=i,
                    question=row["question"],
                    gold_answer=gold,
                    gold_text=row["answer"],
                )
            )
    return out
