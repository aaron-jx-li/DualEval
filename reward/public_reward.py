#!/usr/bin/env python3
"""Score arena response files with the public Skywork reward model.

For every ``results/arena_eval/<domain>/responses.jsonl`` file produced by
``eval/eval_arena_*.py``, this script writes a matching
``results/public_reward/<domain>/responses.jsonl`` with a populated ``reward``
field from the Skywork-Reward-V2-Qwen3-8B model.

Example:
  python reward/public_reward.py \
    --arena-dir results/arena_eval \
    --output-dir results/public_reward
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARENA_DIR = ROOT / "results" / "arena_eval"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "public_reward"
DEFAULT_MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score results/arena_eval responses with Skywork public RM.",
    )
    parser.add_argument(
        "--arena-dir",
        default=str(DEFAULT_ARENA_DIR),
        help="Directory containing arena_eval domain subdirs with responses.jsonl.",
    )
    parser.add_argument(
        "--arena-jsonl",
        nargs="*",
        default=None,
        help="Optional explicit arena responses.jsonl files to score.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for public reward responses.jsonl files.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face reward model id.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of conversations to score per forward pass/request.",
    )
    parser.add_argument(
        "--backend",
        default="transformers",
        choices=("transformers", "vllm"),
        help="Scoring backend. vLLM uses pooling/classify and can be faster if installed.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=16384,
        help="Tokenizer max length. Skywork recommends inference within 16,384 tokens.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for tokenized tensors, e.g. cuda:0 or cpu. Defaults to cuda:0 when available.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional transformers device_map. Defaults to the resolved device for CUDA, else none.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Torch dtype for model loading. Defaults to bfloat16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional transformers attention implementation, e.g. flash_attention_2 or sdpa.",
    )
    parser.add_argument(
        "--vllm-dtype",
        default="bfloat16",
        help="vLLM dtype, e.g. auto, bfloat16, float16, float32.",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing output responses.jsonl files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap per input JSONL for smoke tests.",
    )
    parser.add_argument(
        "--skip-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep processing after a batch scoring error by writing reward_error rows.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_label",
        "num_items",
        "num_scored",
        "avg_reward",
        "max_reward",
        "min_reward",
        "generation_errors",
        "reward_errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def arena_paths(args: argparse.Namespace) -> list[Path]:
    if args.arena_jsonl:
        paths = [Path(p) for p in args.arena_jsonl]
    else:
        paths = sorted(Path(args.arena_dir).glob("*/responses.jsonl"))
    if not paths:
        raise SystemExit("No arena responses.jsonl files found.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing arena JSONL file(s): {', '.join(missing)}")
    return paths


def output_domain_dir(input_path: Path, arena_dir: Path, output_dir: Path) -> Path:
    try:
        rel = input_path.parent.relative_to(arena_dir)
        return output_dir / rel
    except ValueError:
        return output_dir / input_path.parent.name


def row_key(row: dict[str, Any]) -> str:
    return "::".join(
        [
            str(row.get("item_id") or row.get("example_id") or row.get("question_id") or ""),
            str(row.get("model_label") or ""),
            str(row.get("row_index") if row.get("row_index") is not None else ""),
        ],
    )


def load_completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(path):
        completed.add(row_key(row))
    return completed


def limited_rows(path: Path, max_rows: int | None) -> Iterable[dict[str, Any]]:
    for i, row in enumerate(read_jsonl(path)):
        if max_rows is not None and i >= max_rows:
            break
        yield row


def count_limited_rows(path: Path, max_rows: int | None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for count, _ in enumerate(f, start=1):
            if max_rows is not None and count >= max_rows:
                return count
    return count


def build_conversation(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = str(row.get("question") or row.get("prompt") or "")
    response = str(row.get("response_text") or "")
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def prepare_text(tokenizer: Any, row: dict[str, Any]) -> str:
    text = tokenizer.apply_chat_template(build_conversation(row), tokenize=False)
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token is not None and text.startswith(bos_token):
        text = text[len(bos_token) :]
    return text


def resolve_torch_dtype(torch: Any, dtype_name: str | None, device: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def load_transformers_scorer(args: argparse.Namespace) -> Callable[[list[dict[str, Any]]], list[float]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Scoring requires torch and transformers. Install them before running public_reward.py.",
        ) from exc

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_torch_dtype(torch, args.torch_dtype, device)
    device_map = args.device_map
    if device_map is None and device.startswith("cuda"):
        device_map = device

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "num_labels": 1,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)
    model.eval()
    if device_map is None:
        model.to(device)

    def scorer(rows: list[dict[str, Any]]) -> list[float]:
        return score_batch_transformers(rows, tokenizer, model, torch, device, args.max_length)

    return scorer


def score_batch_transformers(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
    max_length: int,
) -> list[float]:
    texts = [prepare_text(tokenizer, row) for row in rows]
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    if hasattr(tokenized, "to"):
        tokenized = tokenized.to(device)
    else:
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

    with torch.no_grad():
        logits = model(**tokenized).logits
    return [float(value) for value in logits.reshape(-1).detach().cpu().tolist()]


def extract_vllm_score(output: Any) -> float:
    data = getattr(getattr(output, "outputs", None), "data", None)
    if data is None:
        data = getattr(getattr(output, "outputs", None), "probs", None)
    if data is None:
        raise ValueError(f"Unexpected vLLM output shape: {output!r}")
    if hasattr(data, "tolist"):
        data = data.tolist()
    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError(f"Empty vLLM output data: {output!r}")
        return float(data[0])
    return float(data)


def load_vllm_scorer(args: argparse.Namespace) -> Callable[[list[dict[str, Any]]], list[float]]:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM
    except ImportError as exc:
        raise SystemExit("vLLM backend requires both vllm and transformers to be installed.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(
        model=args.model_name,
        task="classify",
        dtype=args.vllm_dtype,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.max_length,
    )

    def scorer(rows: list[dict[str, Any]]) -> list[float]:
        texts = [prepare_text(tokenizer, row) for row in rows]
        outputs = llm.classify(texts)
        return [extract_vllm_score(output) for output in outputs]

    return scorer


def load_scorer(args: argparse.Namespace) -> Callable[[list[dict[str, Any]]], list[float]]:
    if args.backend == "vllm":
        return load_vllm_scorer(args)
    return load_transformers_scorer(args)


def scored_row(
    row: dict[str, Any],
    reward: float | None,
    error: str | None,
    latency_s: float | None,
    model_name: str,
) -> dict[str, Any]:
    out = dict(row)
    out["source_reward"] = row.get("reward")
    out["reward"] = reward
    out["reward_model"] = model_name
    out["reward_model_type"] = "public"
    out["reward_latency_s"] = latency_s
    out["processed_at"] = datetime.now().isoformat()
    if reward is None:
        out["status"] = "generation_error" if row.get("status") == "generation_error" else "reward_error"
        out["error"] = error
    else:
        out["status"] = row.get("status") if row.get("status") not in (None, "reward_error") else "ok"
        out["error"] = row.get("error") if out["status"] != "ok" else None
    return out


def summarize_results(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("model_label") or "")].append(row)

    summary: list[dict[str, Any]] = []
    for model_label, group in sorted(grouped.items()):
        ok_rows = [row for row in group if row.get("status") == "ok"]
        rewards = [
            float(row["reward"])
            for row in ok_rows
            if isinstance(row.get("reward"), (int, float)) and math.isfinite(float(row["reward"]))
        ]
        summary.append(
            {
                "model_label": model_label,
                "num_items": len(group),
                "num_scored": len(rewards),
                "avg_reward": mean(rewards) if rewards else None,
                "max_reward": max(rewards) if rewards else None,
                "min_reward": min(rewards) if rewards else None,
                "generation_errors": sum(1 for row in group if row.get("status") == "generation_error"),
                "reward_errors": sum(1 for row in group if row.get("status") == "reward_error"),
            },
        )
    return summary


def write_summary_from_output(output_jsonl: Path) -> None:
    rows = list(read_jsonl(output_jsonl)) if output_jsonl.exists() else []
    write_summary_csv(output_jsonl.parent / "summary.csv", summarize_results(rows))


def flush_batch(
    batch: list[dict[str, Any]],
    output_jsonl: Path,
    scorer: Callable[[list[dict[str, Any]]], list[float]],
    args: argparse.Namespace,
) -> int:
    if not batch:
        return 0

    started = time.time()
    try:
        scores = scorer(batch)
        latency = round(time.time() - started, 2)
        rows = [
            scored_row(row, score, None, round(latency / max(len(batch), 1), 4), args.model_name)
            for row, score in zip(batch, scores)
        ]
    except Exception as exc:
        if not args.skip_errors:
            raise
        latency = round(time.time() - started, 2)
        rows = [
            scored_row(row, None, f"{type(exc).__name__}: {exc}", round(latency / max(len(batch), 1), 4), args.model_name)
            for row in batch
        ]

    append_jsonl(output_jsonl, rows)
    return len(rows)


def score_file(
    input_jsonl: Path,
    output_jsonl: Path,
    scorer: Callable[[list[dict[str, Any]]], list[float]],
    args: argparse.Namespace,
) -> int:
    completed = load_completed_keys(output_jsonl) if args.resume else set()
    if output_jsonl.exists() and not args.resume:
        output_jsonl.unlink()

    written = 0
    skipped = 0
    batch: list[dict[str, Any]] = []
    progress = tqdm(
        limited_rows(input_jsonl, args.max_rows),
        total=count_limited_rows(input_jsonl, args.max_rows),
        desc=input_jsonl.parent.name,
        unit="row",
    )
    for row in progress:
        if row_key(row) in completed:
            skipped += 1
            progress.set_postfix(written=written, skipped=skipped, pending=len(batch), refresh=False)
            continue
        if row.get("status") == "generation_error" or not row.get("response_text"):
            append_jsonl(
                output_jsonl,
                [
                    scored_row(
                        row,
                        None,
                        "No response_text to score" if not row.get("response_text") else row.get("error"),
                        None,
                        args.model_name,
                    ),
                ],
            )
            written += 1
            progress.set_postfix(written=written, skipped=skipped, pending=len(batch), refresh=False)
            continue

        batch.append(row)
        if len(batch) >= args.batch_size:
            written += flush_batch(batch, output_jsonl, scorer, args)
            batch.clear()
            progress.set_postfix(written=written, skipped=skipped, pending=len(batch), refresh=False)

    if batch:
        written += flush_batch(batch, output_jsonl, scorer, args)
        progress.set_postfix(written=written, skipped=skipped, pending=0, refresh=True)
    write_summary_from_output(output_jsonl)
    return written


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.max_length < 1:
        raise SystemExit("--max-length must be at least 1.")

    input_paths = arena_paths(args)
    output_dir = Path(args.output_dir)
    arena_dir = Path(args.arena_dir)

    scorer = load_scorer(args)

    for input_jsonl in input_paths:
        domain_dir = output_domain_dir(input_jsonl, arena_dir, output_dir)
        output_jsonl = domain_dir / "responses.jsonl"
        print(f"Scoring {input_jsonl} -> {output_jsonl}", flush=True)
        written = score_file(input_jsonl, output_jsonl, scorer, args)
        print(f"Wrote {written} new rows to {output_jsonl}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(1)
