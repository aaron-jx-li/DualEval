#!/usr/bin/env python3
"""Validate public reward-model pair preferences against Arena human feedback.

Examples:
  python reward/rm_validate.py \
    --public-rm-dir results/public_RM \
    --output results/rm_validation

  python reward/rm_validate.py \
    --public-rm-jsonl results/public_RM/v1_arena_coding.jsonl \
    --output /tmp/rm_validation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PUBLIC_RM_DIR = Path("results") / "public_RM"
DEFAULT_PUBLIC_VALIDATE_DIR = Path("results") / "public_validate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare public reward-model pair preferences with "
            "Arena human_winner labels."
        ),
    )
    parser.add_argument(
        "--public-rm-dir",
        default=str(DEFAULT_PUBLIC_RM_DIR),
        help="Directory containing public RM CSVs. Ignored when --public-rm-csv/--public-rm-jsonl is set.",
    )
    parser.add_argument(
        "--public-rm-csv",
        nargs="*",
        default=None,
        help="One or more public RM CSV files with human_winner, score_a, and score_b columns.",
    )
    parser.add_argument(
        "--public-rm-jsonl",
        nargs="*",
        default=None,
        help="One or more public RM JSONL files with human_winner, score_a, and score_b fields.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. Writes responses.jsonl and summary.csv inside it.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap per public RM input file, useful for smoke tests.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reward_model",
        "domain",
        "num_rows",
        "num_human_decisive",
        "num_scored_decisive",
        "num_correct_decisive",
        "num_pred_tie_decisive",
        "accuracy_decisive",
        "pearson_corr_decisive",
        "spearman_corr_decisive",
        "source_paths",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def public_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.public_rm_csv or args.public_rm_jsonl:
        paths = [Path(p) for p in args.public_rm_csv or []]
        paths.extend(Path(p) for p in args.public_rm_jsonl or [])
    else:
        public_dir = Path(args.public_rm_dir)
        if args.public_rm_dir == str(DEFAULT_PUBLIC_RM_DIR) and not public_dir.exists():
            public_dir = DEFAULT_PUBLIC_VALIDATE_DIR
        paths = sorted(public_dir.glob("*.csv"))
    if not paths:
        raise SystemExit("No public RM input files found.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing public RM input file(s): {', '.join(missing)}")
    return paths


def infer_domain(input_path: Path, first_row_domain: str | None) -> str:
    if first_row_domain:
        return first_row_domain
    stem = input_path.stem
    for prefix in ("v1_arena_", "arena_", "v1_"):
        if stem.startswith(prefix):
            return stem.removeprefix(prefix)
    return stem


def is_jsonl_path(path: Path) -> bool:
    return path.suffix.lower() in {".jsonl", ".json"}


def iter_input_rows(path: Path, required: set[str]) -> Iterable[dict[str, Any]]:
    if is_jsonl_path(path):
        for line_no, row in enumerate(read_jsonl(path), start=1):
            missing = required.difference(row)
            if missing:
                raise SystemExit(
                    f"{path}:{line_no} is missing required field(s): {', '.join(sorted(missing))}",
                )
            yield row
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing required column(s): {', '.join(sorted(missing))}")
        yield from reader


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def normalize_reward(raw_reward: Any) -> float | None:
    if isinstance(raw_reward, Exception):
        return None
    return parse_float(raw_reward)


def human_sign(value: Any) -> int | None:
    label = str(value or "").strip().lower()
    if label in {"model_a", "a", "response_a", "assistant_a", "winner_a"}:
        return 1
    if label in {"model_b", "b", "response_b", "assistant_b", "winner_b"}:
        return -1
    return None


def prediction_sign(score_diff_a_minus_b: float) -> int:
    if score_diff_a_minus_b > 0:
        return 1
    if score_diff_a_minus_b < 0:
        return -1
    return 0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    den = math.sqrt(den_x * den_y)
    if den == 0:
        return None
    return num / den


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        # 1-indexed average rank for ties.
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            out[order[k]] = avg_rank
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(ranks(xs), ranks(ys))


class MetricAccumulator:
    def __init__(self) -> None:
        self.human: list[float] = []
        self.score_diff: list[float] = []
        self.num_rows = 0
        self.num_decisive = 0
        self.num_scored = 0
        self.num_correct = 0
        self.num_pred_tie = 0

    def add(self, human: int | None, score_diff: float | None) -> None:
        self.num_rows += 1
        if human is None:
            return
        self.num_decisive += 1
        if score_diff is None:
            return
        self.num_scored += 1
        pred = prediction_sign(score_diff)
        if pred == 0:
            self.num_pred_tie += 1
        if pred == human:
            self.num_correct += 1
        self.human.append(float(human))
        self.score_diff.append(score_diff)

    def merge(self, other: "MetricAccumulator") -> None:
        self.human.extend(other.human)
        self.score_diff.extend(other.score_diff)
        self.num_rows += other.num_rows
        self.num_decisive += other.num_decisive
        self.num_scored += other.num_scored
        self.num_correct += other.num_correct
        self.num_pred_tie += other.num_pred_tie

    def metrics(self) -> dict[str, Any]:
        return {
            "num_rows": self.num_rows,
            "num_human_decisive": self.num_decisive,
            "num_scored_decisive": self.num_scored,
            "num_correct_decisive": self.num_correct,
            "num_pred_tie_decisive": self.num_pred_tie,
            "accuracy_decisive": self.num_correct / self.num_scored if self.num_scored else None,
            "pearson_corr_decisive": pearson(self.score_diff, self.human),
            "spearman_corr_decisive": spearman(self.score_diff, self.human),
        }


def metric_rows(
    rm_name: str,
    by_domain: dict[str, MetricAccumulator],
    source_paths: list[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overall = MetricAccumulator()
    for domain in sorted(by_domain):
        acc = by_domain[domain]
        overall.merge(acc)
        rows.append(
            {
                "reward_model": rm_name,
                "domain": domain,
                "source_paths": ";".join(str(path) for path in source_paths),
                **acc.metrics(),
            },
        )
    rows.append(
        {
            "reward_model": rm_name,
            "domain": "overall",
            "source_paths": ";".join(str(path) for path in source_paths),
            **overall.metrics(),
        },
    )
    return rows


def public_response_rows(
    input_path: Path,
    by_domain: dict[str, MetricAccumulator],
    max_rows: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first_domain: str | None = None
    required = {"human_winner", "model_a", "model_b", "score_a", "score_b"}
    rows_seen = 0
    for row_index, row in enumerate(iter_input_rows(input_path, required)):
        if first_domain is None:
            first_domain = row.get("domain")
        domain = infer_domain(input_path, row.get("domain") or first_domain)
        item_id = str(row.get("example_id") or row.get("item_id") or row_index)
        question = str(row.get("question") or row.get("prompt") or "")
        prompt = str(row.get("prompt") or question)
        human = human_sign(row.get("human_winner"))
        score_a = parse_float(row.get("score_a"))
        score_b = parse_float(row.get("score_b"))
        score_diff = score_a - score_b if score_a is not None and score_b is not None else None
        by_domain[domain].add(human, score_diff)

        for side, model_key, response_key, score in (
            ("a", "model_a", "response_a", score_a),
            ("b", "model_b", "response_b", score_b),
        ):
            rows.append(
                {
                    "item_id": item_id,
                    "row_index": row.get("row_index", row_index),
                    "model_label": row.get(model_key),
                    "model_provider": None,
                    "model_id": row.get(model_key),
                    "question": question,
                    "prompt": prompt,
                    "response_text": row.get(response_key),
                    "reward": score,
                    "status": "ok" if score is not None else "reward_error",
                    "error": None if score is not None else "Missing public RM score",
                    "generation_latency_s": None,
                    "reward_latency_s": None,
                    "total_latency_s": None,
                    "processed_at": None,
                    "reward_model": "public_rm",
                    "reward_model_source": str(input_path),
                    "pair_side": side,
                    "human_winner": row.get("human_winner"),
                    "domain": domain,
                },
            )

        rows_seen += 1
        if max_rows is not None and rows_seen >= max_rows:
            break
    return rows


def add_public_rm_metrics(
    input_path: Path,
    by_domain: dict[str, MetricAccumulator],
    max_rows: int | None,
) -> str:
    first_domain: str | None = None
    required = {"human_winner", "score_a", "score_b"}
    rows_seen = 0
    for row in iter_input_rows(input_path, required):
        if first_domain is None:
            first_domain = row.get("domain")
        domain = infer_domain(input_path, row.get("domain") or first_domain)
        human = human_sign(row.get("human_winner"))
        score_a = parse_float(row.get("score_a"))
        score_b = parse_float(row.get("score_b"))
        score_diff = score_a - score_b if score_a is not None and score_b is not None else None
        by_domain[domain].add(human, score_diff)
        rows_seen += 1
        if max_rows is not None and rows_seen >= max_rows:
            break
    return infer_domain(input_path, first_domain)


def main() -> None:
    args = parse_args()
    public_paths = public_input_paths(args)

    output_response_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    public_by_domain: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    for path in public_paths:
        output_response_rows.extend(public_response_rows(path, public_by_domain, args.max_rows))
    summary_rows.extend(metric_rows("public_rm", public_by_domain, public_paths))

    output_dir = Path(args.output)
    output_path = output_dir / "responses.jsonl"
    summary_path = output_dir / "summary.csv"
    write_jsonl(output_path, output_response_rows)
    write_summary_csv(summary_path, summary_rows)
    print(f"Wrote {len(output_response_rows)} response rows to {output_path}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            import os
            os._exit(1)
