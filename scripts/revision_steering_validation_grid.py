#!/usr/bin/env python3
"""Frozen-grid steering evaluation for the TACL-11241 major revision.

This runner evaluates only predeclared method vectors/layers/alphas on a
problem-paired in-domain or frozen-OOD set, normalizes every vector to the same
RMS perturbation budget, uses post-prompt ``decode_last`` injection, and writes
complete per-problem artifacts.  For a declared sampling decoder, baseline
and every intervention cell reuse the same deterministic problem-indexed seed.
In-domain GSM8K evaluation rejects overlap with the direction's source-training
IDs.  Cross-dataset evaluation requires an explicit flag and records its
evaluator/version in the immutable manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from geoprobe.datasets import (
    is_correct,
    is_correct_math500_symbolic,
    load_gsm8k,
    load_math500,
    load_svamp,
    parse_predicted,
    parse_predicted_math500,
)
from geoprobe.datasets.svamp import SVAMP_SHA256
from geoprobe.models import load_model_and_tokenizer
from geoprobe.revision.behavior import summarize_text
from geoprobe.revision.generation import summarize_generated_tokens
from geoprobe.revision.pooled_trajectory import _generate_with_local_seed
from geoprobe.revision.stats import paired_binary_summary
from geoprobe.steering import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    LinearDecaySchedule,
    PrefixSchedule,
    SteeringHook,
    normalize_direction_rms,
)

_PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)


def _parse_csv_floats(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("layer list contains duplicates")
    return values


def _parse_vector(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--vector must have NAME=/path/to/vector.pt")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--vector must have a non-empty name and path")
    return name, Path(path)


def _make_schedule(name: str, parameter: float):
    if name == "constant":
        return ConstantSchedule()
    if name == "prefix":
        return PrefixSchedule(int(parameter))
    if name == "linear":
        return LinearDecaySchedule(int(parameter))
    if name == "exponential":
        return ExponentialDecaySchedule(float(parameter))
    raise ValueError(f"unknown schedule: {name}")


def _registry_source_ids(vector_paths: list[Path]) -> set[int]:
    """Read and require a consistent registry manifest next to each vector."""
    source_ids: set[int] | None = None
    for path in vector_paths:
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{path} lacks sibling manifest.json; formal evaluation refuses untracked vectors"
            )
        manifest = json.loads(manifest_path.read_text())
        ids = {int(value) for value in manifest.get("source_ids", [])}
        if not ids:
            raise ValueError(f"{manifest_path} does not record non-empty source_ids")
        if source_ids is None:
            source_ids = ids
        elif source_ids != ids:
            raise ValueError("all compared vectors must have the same declared source-training IDs")
    return source_ids or set()


def _validate_evaluation_scope(
    dataset: str,
    *,
    source_ids: set[int],
    eval_start: int,
    eval_count: int,
    allow_cross_dataset_eval: bool,
) -> None:
    """Keep GSM8K split isolation while making OOD opt-in and explicit."""
    if dataset == "gsm8k":
        overlap = source_ids & set(range(eval_start, eval_start + eval_count))
        if overlap:
            raise ValueError(
                f"evaluation overlaps vector source-training IDs: {sorted(overlap)[:10]}"
            )
        return
    if dataset in {"math500", "svamp"} and allow_cross_dataset_eval:
        return
    raise ValueError(
        "cross-dataset evaluation requires --allow-cross-dataset-eval; "
        "it may only be launched after an in-domain selection is frozen"
    )


def _dataset_evaluator_metadata(dataset: str) -> dict[str, str]:
    if dataset == "gsm8k":
        return {"name": "gsm8k_numeric_boxed_last_number", "version": "revision-v1"}
    if dataset == "math500":
        return {"name": "math_verify_symbolic_full_completion", "version": "math-verify==0.9.0"}
    if dataset == "svamp":
        return {"name": "svamp_numeric_boxed_last_number", "version": f"sha256:{SVAMP_SHA256}"}
    raise ValueError(f"unknown dataset: {dataset}")


def _load_samples(dataset: str, *, start: int, count: int):
    if dataset == "gsm8k":
        samples = load_gsm8k(split="test", n=start + count, source="oss")[start:]
    elif dataset == "math500":
        samples = load_math500(n=start + count)[start:]
    elif dataset == "svamp":
        samples = load_svamp(n=start + count)[start:]
    else:  # pragma: no cover - argparse validates choices
        raise ValueError(f"unknown dataset: {dataset}")
    if len(samples) != count:
        raise ValueError(f"requested evaluation range exceeds {dataset} data")
    return samples


def _score_completion(
    dataset: str, generated_text: str, gold: object
) -> tuple[object, object, bool]:
    """Return serializable gold/display prediction/correctness without silent grader fallback."""
    if dataset == "gsm8k":
        prediction = parse_predicted(generated_text)
        return (
            float(gold),
            None if prediction is None else float(prediction),
            bool(is_correct(prediction, gold)),
        )
    if dataset == "math500":
        if not isinstance(gold, str):
            raise TypeError("MATH-500 gold answer must be text")
        display_prediction = parse_predicted_math500(generated_text)
        return gold, display_prediction, bool(is_correct_math500_symbolic(generated_text, gold))
    if dataset == "svamp":
        prediction = parse_predicted(generated_text)
        return (
            float(gold),
            None if prediction is None else float(prediction),
            bool(is_correct(prediction, gold)),
        )
    raise ValueError(f"unknown dataset: {dataset}")


def _run_signature(args: argparse.Namespace, vectors: dict[str, Path]) -> dict[str, object]:
    return {
        "protocol": "tacl-11241-frozen-steering-grid-v1",
        "dataset": getattr(args, "dataset", "gsm8k"),
        "cross_dataset_eval": bool(getattr(args, "allow_cross_dataset_eval", False)),
        "evaluator": _dataset_evaluator_metadata(getattr(args, "dataset", "gsm8k")),
        "vectors": {name: str(path.resolve()) for name, path in vectors.items()},
        "target_model": args.target_model,
        "target_source": args.target_source,
        "target_device": args.target_device,
        "eval_start": args.eval_start,
        "eval_count": args.eval_count,
        "layers": args.layers,
        "alphas": args.alphas,
        "schedule": args.schedule,
        "schedule_parameter": args.schedule_parameter,
        "position_mode": "decode_last",
        "injection_mode": args.injection_mode,
        "normalization": "direction RMS=1 before alpha",
        "max_new_tokens": args.max_new_tokens,
        "generation": {
            "do_sample": bool(getattr(args, "do_sample", False)),
            "temperature": float(getattr(args, "temperature", 1.0)),
            "top_p": float(getattr(args, "top_p", 1.0)),
            "base_seed": int(getattr(args, "seed", 11241)),
            "seed_rule": (
                "seed * 100003 + sample_id * 1009"
                if bool(getattr(args, "do_sample", False))
                else None
            ),
            "num_beams": 1,
            "use_cache": True,
            "eos_token_id": "model.generation_config.eos_token_id",
        },
        "prompt_template": _PROMPT_TEMPLATE,
    }


def _problem_seed(base_seed: int, sample_id: int) -> int:
    """Frozen sampling seed shared by baseline and every intervention cell."""
    return base_seed * 100003 + sample_id * 1009


@torch.inference_mode()
def _generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    hook: SteeringHook | None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generation_seed: int | None = None,
) -> tuple[str, dict[str, object]]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Use the model's full EOS set, not just tokenizer.eos_token_id.  Qwen2.5-
    # Instruct's generation config declares both <|im_end|> (151645) and
    # <|endoftext|> (151643); bare-greedy generations terminate on the latter.
    # Passing only tokenizer.eos_token_id drops 151643 and forces every
    # completion to max_new_tokens (100% truncation).
    eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": eos_ids,
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_p": top_p})
    generation_kwargs = {**inputs, **kwargs}
    if hook is None:
        output = _generate_with_local_seed(
            model,
            generation_kwargs,
            generation_seed=generation_seed if do_sample else None,
            device=device,
        )
    else:
        with hook:
            output = _generate_with_local_seed(
                model,
                generation_kwargs,
                generation_seed=generation_seed if do_sample else None,
                device=device,
            )
    prompt_len = int(inputs["input_ids"].shape[1])
    telemetry = summarize_generated_tokens(
        output[0, prompt_len:], eos_token_id=eos_ids, max_new_tokens=max_new_tokens
    )
    text = tokenizer.decode(telemetry.generated_token_ids, skip_special_tokens=True)
    return text, {
        "n_generated_tokens": telemetry.n_generated_tokens,
        "n_content_tokens": telemetry.n_content_tokens,
        "stop_reason": telemetry.stop_reason,
        "truncated": telemetry.truncated,
    }


def _load_completed(path: Path) -> tuple[set[tuple[str, int, float, int]], list[dict[str, object]]]:
    done: set[tuple[str, int, float, int]] = set()
    rows: list[dict[str, object]] = []
    if not path.exists():
        return done, rows
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            done.add(
                (str(row["method"]), int(row["layer"]), float(row["alpha"]), int(row["sample_id"]))
            )
    return done, rows


def _append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = rows[rows["method"] == "baseline"].set_index("sample_id").sort_index()
    if baseline.empty:
        raise ValueError("no baseline rows available")
    baseline_correct = baseline["correct"].astype(int).to_numpy()
    summaries: list[dict[str, object]] = []
    for (method, layer, alpha), cell in rows[rows["method"] != "baseline"].groupby(
        ["method", "layer", "alpha"], sort=True
    ):
        cell = cell.set_index("sample_id").sort_index()
        if not cell.index.equals(baseline.index):
            raise ValueError(
                f"{method}/L{layer}/alpha={alpha} does not cover the same problems as baseline"
            )
        paired = paired_binary_summary(baseline_correct, cell["correct"].astype(int).to_numpy())
        summaries.append(
            {
                "method": method,
                "layer": int(layer),
                "alpha": float(alpha),
                "mean_text_tokens": float(cell["text_tokens"].mean()),
                "mean_generated_tokens": float(
                    cell.get("n_generated_tokens", cell["text_tokens"]).mean()
                ),
                "truncation_rate": float(
                    cell.get("truncated", pd.Series(False, index=cell.index)).astype(float).mean()
                ),
                "mean_repeated_4gram_fraction": float(cell["repeated_4gram_fraction"].mean()),
                **paired.__dict__,
            }
        )
    return pd.DataFrame(summaries).sort_values(["method", "layer", "alpha"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", action="append", type=_parse_vector, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument(
        "--target-source", choices=["modelscope", "huggingface"], default="modelscope"
    )
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "svamp"], default="gsm8k")
    parser.add_argument("--allow-cross-dataset-eval", action="store_true")
    parser.add_argument("--eval-start", type=int, required=True)
    parser.add_argument("--eval-count", type=int, required=True)
    parser.add_argument("--layers", type=_parse_csv_ints, required=True)
    parser.add_argument("--alphas", type=_parse_csv_floats, required=True)
    parser.add_argument(
        "--schedule", choices=["constant", "prefix", "linear", "exponential"], default="constant"
    )
    parser.add_argument("--schedule-parameter", type=float, default=64)
    parser.add_argument(
        "--injection-mode",
        choices=["absolute", "relative_hidden_rms"],
        default="absolute",
        help="absolute is required for validation grid V1; relative_hidden_rms is long-context only",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11241)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.eval_start < 0 or args.eval_count <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("evaluation offset/count and max-new-tokens must be positive")
    if not 0.0 < args.top_p <= 1.0:
        raise SystemExit("top-p must be in (0, 1]")
    if args.do_sample and args.temperature <= 0.0:
        raise SystemExit("sampling temperature must be positive")
    if len(args.vector) != len({name for name, _ in args.vector}):
        raise SystemExit("vector method names must be unique")
    vectors = dict(args.vector)
    source_ids = _registry_source_ids(list(vectors.values()))
    try:
        _validate_evaluation_scope(
            args.dataset,
            source_ids=source_ids,
            eval_start=args.eval_start,
            eval_count=args.eval_count,
            allow_cross_dataset_eval=args.allow_cross_dataset_eval,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    signature = _run_signature(args, vectors)
    signature["source_ids"] = sorted(source_ids)
    signature["signature_sha256"] = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "resolved_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != signature:
        raise SystemExit("output directory has a different resolved manifest; refuse mixed runs")
    manifest_path.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")

    loaded_vectors = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in vectors.items()
    }
    if any(vector.ndim != 2 for vector in loaded_vectors.values()):
        raise SystemExit("all vectors must have shape [layers, hidden]")
    shapes = {tuple(vector.shape) for vector in loaded_vectors.values()}
    if len(shapes) != 1:
        raise SystemExit(f"vector shapes differ: {sorted(shapes)}")
    n_layers, _ = next(iter(shapes))
    if any(layer < 1 or layer >= n_layers for layer in args.layers):
        raise SystemExit(
            f"layers must be in [1, {n_layers - 1}] for vector shape {next(iter(shapes))}"
        )
    schedule = _make_schedule(args.schedule, args.schedule_parameter)
    incremental = args.out / "per_sample.jsonl"
    done, existing_rows = _load_completed(incremental)
    samples = _load_samples(args.dataset, start=args.eval_start, count=args.eval_count)
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.target_model,
        source=args.target_source,
        dtype="bfloat16",
        device=args.target_device,
    )
    rows = list(existing_rows)
    for sample in samples:
        prompt = _PROMPT_TEMPLATE.format(question=sample.question)
        generation_seed = _problem_seed(args.seed, int(sample.id)) if args.do_sample else None
        base_key = ("baseline", 0, 0.0, int(sample.id))
        if base_key not in done:
            text, telemetry = _generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                hook=None,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                generation_seed=generation_seed,
            )
            gold, prediction, correct = _score_completion(args.dataset, text, sample.gold_answer)
            row: dict[str, object] = {
                "method": "baseline",
                "layer": 0,
                "alpha": 0.0,
                "sample_id": int(sample.id),
                "gold": gold,
                "pred": prediction,
                "correct": correct,
                "generated_text": text,
                "text_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                "generation_seed": generation_seed,
                **telemetry,
                **summarize_text(text),
            }
            _append_row(incremental, row)
            rows.append(row)
            done.add(base_key)
        for method, matrix in loaded_vectors.items():
            for layer in args.layers:
                vector = normalize_direction_rms(matrix[layer], rms=1.0)
                for alpha in args.alphas:
                    key = (method, layer, float(alpha), int(sample.id))
                    if key in done:
                        continue
                    hook = SteeringHook(
                        model,
                        layer_idx=layer,
                        vector=vector,
                        alpha=alpha,
                        schedule=schedule,
                        position_mode="decode_last",
                        injection_mode=args.injection_mode,
                    )
                    text, telemetry = _generate(
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens=args.max_new_tokens,
                        hook=hook,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        generation_seed=generation_seed,
                    )
                    gold, prediction, correct = _score_completion(
                        args.dataset, text, sample.gold_answer
                    )
                    row = {
                        "method": method,
                        "layer": int(layer),
                        "alpha": float(alpha),
                        "sample_id": int(sample.id),
                        "gold": gold,
                        "pred": prediction,
                        "correct": correct,
                        "generated_text": text,
                        "text_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                        "generation_seed": generation_seed,
                        **telemetry,
                        **summarize_text(text),
                    }
                    _append_row(incremental, row)
                    rows.append(row)
                    done.add(key)
    frame = pd.DataFrame(rows)
    frame.to_parquet(args.out / "per_sample.parquet", index=False)
    summary = _summarize(frame)
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "DONE").write_text("complete\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
