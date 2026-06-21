#!/usr/bin/env python3
"""
Unified static question sampler for math and misc domains.

Usage examples:
    python demo/sample_static.py --domain math --config demo/config_static.yaml
    python demo/sample_static.py --domain misc --config demo/config_static.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm
import yaml

# Resolve eval/ from the repo root so eval_static can be imported from demo/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from eval_static import (
    _sanitize_for_json,
    DATASET_LOOKUP,
    DATASET_SPECS,
    build_eval_prompt,
    build_gold_answer,
    build_raw_question,
    build_sample_plan,
    get_item_metadata,
    load_env_file,
    load_jsonl,
    load_raw_rows,
    resolve_env_path,
    sample_rows,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Inlined misc helpers (from eval_static_misc, not available in public repo)
# ---------------------------------------------------------------------------

ANSWER_INSTRUCTION_MISC = (
    "Answer the question concisely and correctly. "
    "End your response with a final line of the form 'Final answer: <answer>'."
)

HLE_HF_PATH = "cais/hle"
HLE_SPLIT = "test"
HLE_DATASET_LABELS: dict[str, str] = {
    "humanities_social_science": "hle-humanities-social-science",
    "other": "hle-other",
    "biology_medicine": "hle-biology-medicine",
    "engineering": "hle-engineering",
}


def is_hle_text_only(row: dict[str, Any]) -> bool:
    return not (row.get("image") and str(row["image"]).strip())


def is_non_numeric_gold(answer: str) -> bool:
    """Heuristic: exclude pure numbers / simple fractions (for Humanities stratum)."""
    s = str(answer).strip()
    if not s:
        return False
    s_plain = re.sub(r"[\$\\]", "", s)
    compact = s_plain.replace(",", "").replace(" ", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", compact):
        return False
    if re.fullmatch(r"-?\d+/\d+", compact):
        return False
    return True


def filter_hle_humanities_misc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("category") != "Humanities/Social Science":
            continue
        if row.get("answer_type") != "exactMatch":
            continue
        if not is_hle_text_only(row):
            continue
        if not is_non_numeric_gold(str(row.get("answer", ""))):
            continue
        out.append(row)
    return out


def filter_hle_category_text_only(rows: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("category") == category and is_hle_text_only(r)]


def sample_deterministic(rows: list[dict[str, Any]], n: int, seed: int, salt: str) -> list[dict[str, Any]]:
    rng = random.Random(seed + sum(ord(c) for c in salt))
    pool = list(rows)
    rng.shuffle(pool)
    if n > len(pool):
        raise ValueError(f"Need {n} items for {salt!r} but only {len(pool)} eligible rows.")
    return pool[:n]


def build_hle_item(dataset_key: str, row: dict[str, Any], sample_index: int) -> dict[str, Any]:
    dataset = HLE_DATASET_LABELS[dataset_key]
    raw = _sanitize_for_json(dict(row))
    question = str(row["question"])
    return {
        "dataset": dataset,
        "dataset_kind": "hle",
        "hle_category": row.get("category"),
        "sample_index": sample_index,
        "question": question,
        "prompt": f"{question}\n\n{ANSWER_INSTRUCTION_MISC}",
        "gold_answer": str(row.get("answer", "")),
        "level": "Expert",
        "subject": str(row.get("category", "")),
        "hle_answer_type": row.get("answer_type"),
        "raw_item": raw,
    }


def build_simpleqa_item(row: dict[str, Any], meta: dict[str, Any], sample_index: int) -> dict[str, Any]:
    topic = str(meta.get("topic", "unknown"))
    q = str(row["problem"])
    return {
        "dataset": "simpleqa",
        "dataset_kind": "simpleqa",
        "sample_index": sample_index,
        "question": q,
        "prompt": f"{q}\n\n{ANSWER_INSTRUCTION_MISC}",
        "gold_answer": str(row.get("answer", "")),
        "level": None,
        "subject": topic,
        "topic": topic,
        "raw_item": {"problem": q, "answer": row.get("answer"), "metadata": meta},
    }


def build_misc_sampled_items(sampling_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    seed = int(sampling_cfg.get("seed", 0))
    hle_cfg = sampling_cfg.get("hle", {})
    simpleqa_cfg = sampling_cfg.get("simpleqa", {})

    ds_hle = load_dataset(HLE_HF_PATH, split=HLE_SPLIT)
    hle_rows = [dict(r) for r in ds_hle]

    items: list[dict[str, Any]] = []
    stratum_counts: dict[str, int] = {}

    # HLE strata (fixed order)
    hs_n = int(hle_cfg.get("humanities_social_science", 90))
    pool_hs = filter_hle_humanities_misc(hle_rows)
    sampled_hs = sample_deterministic(pool_hs, hs_n, seed, "hle_humanities")
    for row in sampled_hs:
        items.append(build_hle_item("humanities_social_science", row, len(items)))
    stratum_counts["hle_humanities_social_science"] = len(sampled_hs)

    o_n = int(hle_cfg.get("other", 83))
    pool_o = filter_hle_category_text_only(hle_rows, "Other")
    sampled_o = sample_deterministic(pool_o, o_n, seed, "hle_other")
    for row in sampled_o:
        items.append(build_hle_item("other", row, len(items)))
    stratum_counts["hle_other"] = len(sampled_o)

    b_n = int(hle_cfg.get("biology_medicine", 52))
    pool_b = filter_hle_category_text_only(hle_rows, "Biology/Medicine")
    sampled_b = sample_deterministic(pool_b, b_n, seed, "hle_biology")
    for row in sampled_b:
        items.append(build_hle_item("biology_medicine", row, len(items)))
    stratum_counts["hle_biology_medicine"] = len(sampled_b)

    e_n = int(hle_cfg.get("engineering", 25))
    pool_e = filter_hle_category_text_only(hle_rows, "Engineering")
    sampled_e = sample_deterministic(pool_e, e_n, seed, "hle_engineering")
    for row in sampled_e:
        items.append(build_hle_item("engineering", row, len(items)))
    stratum_counts["hle_engineering"] = len(sampled_e)

    # SimpleQA stratified
    sq_hf = str(simpleqa_cfg.get("hf_path", "basicv8vc/SimpleQA"))
    sq_split = str(simpleqa_cfg.get("split", "test"))
    topic_counts: dict[str, int] = {
        str(k): int(v) for k, v in (simpleqa_cfg.get("topics") or {}).items()
    }
    if not topic_counts:
        raise ValueError("simpleqa.topics is empty: add topic counts under sampling.simpleqa in the YAML config.")
    ds_sq = load_dataset(sq_hf, split=sq_split)

    # Index rows by topic
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for row in ds_sq:
        r = dict(row)
        meta = ast.literal_eval(r["metadata"])
        topic = str(meta["topic"])
        by_topic.setdefault(topic, []).append((r, meta))

    simpleqa_sampled: dict[str, int] = {}
    for topic, need in topic_counts.items():
        pool = by_topic.get(topic, [])
        if need > len(pool):
            raise ValueError(f"SimpleQA topic {topic!r}: need {need}, only {len(pool)} available.")
        # Deterministic per-topic sample
        rng = random.Random(seed + sum(ord(c) for c in f"simpleqa::{topic}"))
        rng.shuffle(pool)
        chosen = pool[:need]
        for r, meta in chosen:
            items.append(build_simpleqa_item(r, meta, len(items)))
        simpleqa_sampled[topic] = len(chosen)

    meta_out = {
        "seed": seed,
        "hle_stratum_counts": stratum_counts,
        "simpleqa_topic_counts": simpleqa_sampled,
        "total_items": len(items),
    }
    return items, meta_out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _expand_env(raw)


def build_output_dir(user_output_dir: str | None, domain: str) -> Path:
    if user_output_dir:
        return Path(user_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "static_samples" / f"{domain}_{timestamp}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and save a fixed static item set (math or misc).",
    )
    parser.add_argument(
        "--domain",
        choices=["math", "misc"],
        required=True,
        help="Which domain to sample: 'math' (competition datasets) or 'misc' (HLE strata + SimpleQA).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config file. Sampling settings are read from config[domain]['sampling'].",
    )
    # Math-only overrides
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=[spec.name for spec in DATASET_SPECS],
        help="(math only) Datasets to sample from. Defaults to all supported datasets.",
    )
    parser.add_argument(
        "--profile",
        choices=["pilot", "paper"],
        default=None,
        help="(math only) Sampling profile.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=None,
        help="(math only) Override and sample the same count from every selected dataset.",
    )
    # Shared overrides
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save sampled_items.jsonl and sampling_config.json.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Math sampling
# ---------------------------------------------------------------------------

def apply_math_defaults(args: argparse.Namespace, section: dict[str, Any]) -> argparse.Namespace:
    if args.datasets is None:
        args.datasets = section.get("datasets", [spec.name for spec in DATASET_SPECS])
    if args.profile is None:
        args.profile = section.get("profile", "pilot")
    if args.samples_per_dataset is None and "samples_per_dataset" in section:
        args.samples_per_dataset = int(section["samples_per_dataset"])
    if args.seed is None:
        args.seed = int(section.get("seed", 0))
    if args.output_dir is None:
        args.output_dir = section.get("output_dir")

    copy_from = section.get("copy_from", {})
    args.copy_from_file = copy_from.get("file")
    args.copy_from_datasets = list(copy_from.get("datasets", []))
    args.copy_from_responses_file = copy_from.get("responses_file")

    return args


def run_math(args: argparse.Namespace, section: dict[str, Any]) -> None:
    args = apply_math_defaults(args, section)

    output_dir = build_output_dir(args.output_dir, "math")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Copy items from an existing sampled_items.jsonl (e.g. v0) ---
    copied_items: list[dict[str, Any]] = []
    if args.copy_from_file and args.copy_from_datasets:
        copy_path = Path(args.copy_from_file)
        if not copy_path.exists():
            raise FileNotFoundError(f"copy_from file not found: {copy_path}")
        copy_ds = set(args.copy_from_datasets)
        all_v0 = load_jsonl(copy_path)
        copied_items = [item for item in all_v0 if item.get("dataset") in copy_ds]
        found_ds = {item["dataset"] for item in copied_items}
        missing = copy_ds - found_ds
        if missing:
            raise ValueError(f"copy_from datasets not found in {copy_path}: {missing}")
        print(
            f"Copied {len(copied_items)} items from {copy_path} "
            f"({', '.join(sorted(found_ds))})"
        )

    # --- Sample fresh datasets ---
    sample_plan = build_sample_plan(args.datasets, args.profile, args.samples_per_dataset)

    fresh_items: list[dict[str, Any]] = []
    dataset_progress = tqdm(args.datasets, desc="Sampling datasets", unit="dataset")
    for dataset_name in dataset_progress:
        spec = DATASET_LOOKUP[dataset_name]
        dataset_progress.set_postfix_str(
            f"{dataset_name} -> {sample_plan[dataset_name]} items", refresh=False
        )
        raw_rows = load_raw_rows(spec)
        if spec.kind == "math":
            stratify_field = "level"
        elif spec.kind == "olympiad":
            stratify_field = "subfield"
        else:
            stratify_field = None

        sampled_rows = sample_rows(
            raw_rows,
            n=sample_plan[dataset_name],
            seed=args.seed + sum(ord(ch) for ch in dataset_name),
            stratify_field=stratify_field,
        )

        for idx, item in enumerate(sampled_rows):
            metadata = get_item_metadata(spec, item)
            fresh_items.append(
                {
                    "dataset": dataset_name,
                    "dataset_kind": spec.kind,
                    "sample_index": idx,
                    "question": build_raw_question(spec, item),
                    "prompt": build_eval_prompt(spec, item),
                    "gold_answer": build_gold_answer(spec, item),
                    **metadata,
                    "raw_item": item,
                }
            )
    dataset_progress.close()

    # Copied items come first so aime/olympiad appear before hle-math
    all_items = copied_items + fresh_items
    write_jsonl(output_dir / "sampled_items.jsonl", all_items)

    # --- Copy responses for --resume support ---
    if args.copy_from_responses_file and args.copy_from_datasets:
        resp_path = Path(args.copy_from_responses_file)
        if not resp_path.exists():
            print(f"Warning: copy_from responses_file not found, skipping: {resp_path}")
        else:
            copy_ds = set(args.copy_from_datasets)
            all_responses = load_jsonl(resp_path)
            relevant = [r for r in all_responses if r.get("dataset") in copy_ds]
            write_jsonl(output_dir / "responses.jsonl", relevant)
            print(
                f"Copied {len(relevant)} response records to "
                f"{output_dir / 'responses.jsonl'} (for --resume)"
            )

    sampling_config = {
        "domain": "math",
        "datasets_sampled": args.datasets,
        "datasets_copied": args.copy_from_datasets,
        "copy_from_file": str(args.copy_from_file) if args.copy_from_file else None,
        "profile": args.profile,
        "sample_plan": sample_plan,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "num_items": len(all_items),
        "num_copied": len(copied_items),
        "num_fresh": len(fresh_items),
    }
    (output_dir / "sampling_config.json").write_text(
        json.dumps(sampling_config, indent=2), encoding="utf-8"
    )

    print(
        f"Saved {len(all_items)} items to {output_dir / 'sampled_items.jsonl'} "
        f"({len(copied_items)} copied, {len(fresh_items)} freshly sampled)"
    )


# ---------------------------------------------------------------------------
# Misc sampling
# ---------------------------------------------------------------------------

def apply_misc_defaults(args: argparse.Namespace, section: dict[str, Any]) -> argparse.Namespace:
    if args.seed is None:
        args.seed = int(section.get("seed", 0))
    if args.output_dir is None:
        args.output_dir = section.get("output_dir")
    return args


def run_misc(args: argparse.Namespace, section: dict[str, Any]) -> None:
    args = apply_misc_defaults(args, section)

    # build_misc_sampled_items reads seed and stratum counts from the section dict
    effective_section = dict(section)
    effective_section["seed"] = args.seed

    output_dir = build_output_dir(args.output_dir, "misc")
    output_dir.mkdir(parents=True, exist_ok=True)

    items, meta = build_misc_sampled_items(effective_section)
    write_jsonl(output_dir / "sampled_items.jsonl", items)

    sampling_config = {
        "domain": "misc",
        "seed": args.seed,
        "output_dir": str(output_dir),
        "num_items": len(items),
        **meta,
    }
    (output_dir / "sampling_config.json").write_text(
        json.dumps(sampling_config, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(items)} items to {output_dir / 'sampled_items.jsonl'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())

    config = load_yaml_config(args.config)

    # Config is read as config[domain]["sampling"]
    domain_cfg = config.get(args.domain, {})
    if not domain_cfg and args.config:
        raise SystemExit(
            f"Error: config file has no top-level '{args.domain}' key. "
            f"Expected config['{args.domain}']['sampling'] to exist."
        )
    section = domain_cfg.get("sampling", {})

    if args.domain == "math":
        run_math(args, section)
    else:
        run_misc(args, section)


if __name__ == "__main__":
    main()
