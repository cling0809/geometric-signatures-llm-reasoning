"""One-shot locked-test launch specification for frozen steering selections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LockedRunSpec:
    """The exact selected intervention permitted on the GSM8K locked set."""

    method: str
    vector_path: str
    source_ids: tuple[int, ...]
    layer: int
    alpha: float
    validation_accuracy: float
    validation_delta: float
    validation_run_dir: str
    locked_eval_start: int = 200
    locked_eval_count: int = 100
    schedule: str = "constant"
    position_mode: str = "decode_last"
    injection_mode: str = "absolute"
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    base_seed: int = 11241

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-canonical representation for launch-manifest equality.

        Preserve compatibility with existing greedy-512 launch artifacts while
        making every non-default decoder field explicit for official-context
        sampled runs.
        """
        payload = asdict(self)
        payload["source_ids"] = list(self.source_ids)
        if (
            self.max_new_tokens == 512
            and not self.do_sample
            and self.temperature == 1.0
            and self.top_p == 1.0
            and self.base_seed == 11241
        ):
            for key in ("do_sample", "temperature", "top_p", "base_seed"):
                payload.pop(key)
        return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _generation_from_validation_manifest(manifest: dict[str, object]) -> dict[str, object]:
    """Recover the frozen decoder from a validation manifest.

    Old greedy Qwen artifacts did not record this nested object.  They are
    canonically equivalent to the declared greedy defaults, while a sampled
    official-context run must persist and reuse its exact decoder and seed rule.
    """
    raw = manifest.get("generation", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("validation run has malformed generation metadata")
    do_sample = bool(raw.get("do_sample", False))
    if do_sample:
        required = {"temperature", "top_p", "base_seed", "seed_rule"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(
                "sampled validation run lacks frozen generation metadata: "
                f"{missing}"
            )
    temperature = float(raw.get("temperature", 1.0))
    top_p = float(raw.get("top_p", 1.0))
    base_seed = int(raw.get("base_seed", 11241))
    if not 0.0 < top_p <= 1.0:
        raise ValueError("validation run has invalid top_p")
    if do_sample and temperature <= 0.0:
        raise ValueError("sampled validation run has non-positive temperature")
    return {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "base_seed": base_seed,
    }


def frozen_generation_cli_args(spec: LockedRunSpec) -> list[str]:
    """Return CLI arguments that replay a sampled validation decoder exactly.

    Greedy runs intentionally add no decoder flags because the grid runner's
    declared greedy defaults are part of the frozen Qwen protocol.  Sampled
    official-context runs must carry every non-default decoding field through
    locked, OOD, long-context, and label-efficiency launchers.
    """
    if not spec.do_sample:
        return []
    return [
        "--do-sample",
        "--temperature",
        str(spec.temperature),
        "--top-p",
        str(spec.top_p),
        "--seed",
        str(spec.base_seed),
    ]


def _selection_method(payload: dict[str, object], method: str) -> dict[str, object]:
    if payload.get("protocol") != "tacl-11241-frozen-steering-grid-v1":
        raise ValueError("selection has unexpected protocol")
    if payload.get("selection_status") != "locked_before_locked_test_generation":
        raise ValueError("selection was not frozen before locked-test generation")
    entries = payload.get("methods")
    if not isinstance(entries, list):
        raise ValueError("selection has no method decisions")
    found = [item for item in entries if isinstance(item, dict) and item.get("method") == method]
    if len(found) != 1:
        raise ValueError(f"selection must contain exactly one decision for {method!r}")
    return found[0]


def load_locked_run_spec(selection_path: str | Path, method: str) -> LockedRunSpec:
    """Validate a frozen validation decision and recover its exact vector path."""
    selection_file = Path(selection_path)
    payload = json.loads(selection_file.read_text())
    entry = _selection_method(payload, method)
    decision = entry.get("decision")
    if not isinstance(decision, dict):
        raise ValueError(f"selection for {method!r} has no decision payload")
    if not bool(decision.get("eligible")):
        raise ValueError(f"{method!r} is ineligible: {decision.get('reason')}")
    if decision.get("reason") != "selected_by_preregistered_validation_rule":
        raise ValueError(f"{method!r} was not selected by the preregistered rule")
    layer = decision.get("layer")
    alpha = decision.get("alpha")
    if not isinstance(layer, int) or not isinstance(alpha, (int, float)):
        raise ValueError(f"{method!r} selection lacks a concrete layer/alpha")

    run_dir = Path(str(entry["run_dir"]))
    manifest_path = run_dir / "resolved_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != "tacl-11241-frozen-steering-grid-v1":
        raise ValueError(f"validation run for {method!r} has unexpected protocol")
    if manifest.get("position_mode") != "decode_last":
        raise ValueError(f"validation run for {method!r} was not post-prompt steering")
    if manifest.get("injection_mode", "absolute") != "absolute":
        raise ValueError("locked run only accepts the frozen absolute intervention")
    if manifest.get("schedule") != "constant":
        raise ValueError(f"validation run for {method!r} does not use the frozen constant schedule")
    max_new_tokens = int(manifest.get("max_new_tokens", -1))
    if max_new_tokens <= 0:
        raise ValueError(f"validation run for {method!r} has invalid max_new_tokens")
    generation = _generation_from_validation_manifest(manifest)
    vectors = manifest.get("vectors", {})
    if set(vectors) != {method}:
        raise ValueError(f"validation run has unexpected vector set {sorted(vectors)}")
    vector_path = Path(str(vectors[method]))
    if not vector_path.exists():
        raise FileNotFoundError(f"selected vector no longer exists: {vector_path}")
    source_ids = tuple(int(value) for value in manifest.get("source_ids", []))
    if not source_ids:
        raise ValueError("validation run does not declare source IDs")
    forbidden = set(range(200, 300)) & set(source_ids)
    if forbidden:
        raise ValueError(f"selected direction leaks locked IDs: {sorted(forbidden)}")
    validation_ids = {int(value) for value in entry.get("evaluation_ids", [])}
    if validation_ids != set(range(100, 200)):
        raise ValueError("selection did not come from the entire frozen validation ID range 100--199")

    return LockedRunSpec(
        method=method,
        vector_path=str(vector_path.resolve()),
        source_ids=source_ids,
        layer=layer,
        alpha=float(alpha),
        validation_accuracy=float(decision["validation_accuracy"]),
        validation_delta=float(decision["validation_delta"]),
        validation_run_dir=str(run_dir.resolve()),
        max_new_tokens=max_new_tokens,
        do_sample=bool(generation["do_sample"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        base_seed=int(generation["base_seed"]),
    )
