"""Pinned SVAMP test-set loader and numeric evaluation utilities.

SVAMP is a 1,000-item arithmetic word-problem benchmark.  The revision uses the
entire official test JSON as a preregistered second OOD task after GSM8K
selection has frozen.  The file is fetched from the official repository only
when absent and is checked against a recorded SHA-256 before parsing.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from geoprobe.datasets.gsm8k import is_correct, parse_predicted
from geoprobe.utils.paths import cache_dir

SVAMP_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"
SVAMP_SHA256 = "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"
SVAMP_EXPECTED_COUNT = 1000


@dataclass(frozen=True)
class SVAMPSample:
    id: int
    question: str
    gold_answer: float
    gold_text: str
    source_id: str
    problem_type: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_svamp() -> Path:
    target_dir = cache_dir() / "datasets" / "svamp"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "SVAMP.json"
    if not target.exists():
        temporary = target.with_suffix(".json.tmp")
        urllib.request.urlretrieve(SVAMP_URL, temporary)
        temporary.replace(target)
    actual = _sha256(target)
    if actual != SVAMP_SHA256:
        raise ValueError(
            "SVAMP file hash differs from the frozen revision dataset: "
            f"expected {SVAMP_SHA256}, found {actual}"
        )
    return target


def _parse_answer(value: object) -> float:
    try:
        answer = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"SVAMP answer is not numeric: {value!r}") from error
    if not math.isfinite(answer):
        raise ValueError(f"SVAMP answer is non-finite: {value!r}")
    return answer


def load_svamp(n: int | None = None, *, path: str | Path | None = None) -> list[SVAMPSample]:
    """Load the complete frozen SVAMP test JSON or a deterministic prefix."""
    dataset_path = Path(path) if path is not None else _ensure_svamp()
    if _sha256(dataset_path) != SVAMP_SHA256:
        raise ValueError("SVAMP loader refuses an unpinned dataset file")
    payload = json.loads(dataset_path.read_text())
    if not isinstance(payload, list) or len(payload) != SVAMP_EXPECTED_COUNT:
        raise ValueError(f"SVAMP must contain exactly {SVAMP_EXPECTED_COUNT} rows")
    if n is not None and (n < 0 or n > len(payload)):
        raise ValueError("n must be between zero and the SVAMP test size")
    selected = payload if n is None else payload[:n]
    samples: list[SVAMPSample] = []
    for index, row in enumerate(selected):
        if not isinstance(row, dict):
            raise ValueError(f"SVAMP row {index} is not an object")
        required = {"ID", "Body", "Question", "Answer", "Type"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"SVAMP row {index} is missing fields: {sorted(missing)}")
        body = str(row["Body"]).strip()
        question = str(row["Question"]).strip()
        if not body or not question:
            raise ValueError(f"SVAMP row {index} has an empty body/question")
        samples.append(
            SVAMPSample(
                id=index,
                question=f"{body}\n{question}",
                gold_answer=_parse_answer(row["Answer"]),
                gold_text=str(row["Answer"]),
                source_id=str(row["ID"]),
                problem_type=str(row["Type"]),
            )
        )
    return samples


__all__ = [
    "SVAMP_EXPECTED_COUNT",
    "SVAMP_SHA256",
    "SVAMP_URL",
    "SVAMPSample",
    "is_correct",
    "load_svamp",
    "parse_predicted",
]
