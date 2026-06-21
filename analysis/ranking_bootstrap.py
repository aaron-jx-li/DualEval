#!/usr/bin/env python3
"""Question-level cluster bootstrap for DualEval ranking stability (§4.1 Table 4).

For each (domain, method) pair:
  - Resample questions with replacement, stratified by source.
  - Each draw gets a unique question_id suffix so duplicates are treated as
    independent items by the IRT fit — the standard cluster-bootstrap behavior.
  - Reward z-scores are renormalized per replicate (matches the canonical
    arena-load step).
  - Refit the chosen method on the resample:
      dualeval  joint static+arena IRT (mode=both for static-anchored domains;
                mode=arena for generic).
      static    static-only 2PL IRT (lambda_arena=0, lambda_bb=0). Skipped for
                arena-only domains.
      bt        naive Bradley-Terry over arena pairs (fit_bt). No item params.
  - Record theta per model per replicate.

Output: results/ranking_bootstrap/{domain}_{method}_theta_replicates.csv, long
format with columns (model_name, theta, domain, seed, method).

Usage:
  python analysis/ranking_bootstrap.py -B 100 --workers 16
  python analysis/ranking_bootstrap.py -B 100 --workers 16 --methods dualeval static bt
  python analysis/ranking_bootstrap.py -B 4 --workers 4 --domains coding --methods bt
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DUALEVAL_PATH = REPO_ROOT / "ranking" / "dualeval.py"


_DUALEVAL = None


def _get_dualeval():
    global _DUALEVAL
    if _DUALEVAL is None:
        spec = importlib.util.spec_from_file_location("dualeval_module", DUALEVAL_PATH)
        mod = importlib.util.module_from_spec(spec)
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        spec.loader.exec_module(mod)
        _DUALEVAL = mod
    return _DUALEVAL


DOMAINS = {
    "coding": {
        "static": "results/static_eval/coding/responses.jsonl",
        "arena": "results/arena_eval/coding/responses.jsonl",
    },
    "math": {
        "static": "results/static_eval/math/responses.jsonl",
        "arena": "results/arena_eval/math/responses.jsonl",
    },
}

METHODS = ("dualeval", "static", "bt")


def is_valid(domain: str, method: str) -> bool:
    """Static needs static data; bt and dualeval need arena data."""
    info = DOMAINS[domain]
    if method == "static":
        return info["static"] is not None
    if method == "bt":
        return info["arena"] is not None
    if method == "dualeval":
        return info["static"] is not None or info["arena"] is not None
    raise ValueError(f"unknown method {method}")


DEFAULT_HYPERS = dict(
    num_epochs=2000,
    lr=0.02,
    lambda_static=1.0,
    lambda_arena=1.0,
    lambda_bb=0.2,
    reg_lambda=0.01,
    bb_ratio=0.15,
    tie_ratio=0.15,
)


def resample_per_source(
    static_df: pd.DataFrame | None,
    reward_df: pd.DataFrame | None,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Stratified question-level resample, per source.

    Each source's question pool is sampled with replacement to its original
    size. Drawn copies receive unique question_id suffixes so the IRT fit
    treats them as independent items (proper cluster bootstrap). Static and
    arena are sampled independently because their question_id namespaces are
    disjoint by source tag.
    """
    def _resample(df: pd.DataFrame) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        for _, group in df.groupby("source", sort=False):
            qids = group["question_id"].unique()
            n = len(qids)
            if n == 0:
                continue
            draws = rng.choice(qids, size=n, replace=True)
            qid_to_rows = {q: g for q, g in group.groupby("question_id", sort=False)}
            for k, qid in enumerate(draws):
                sub = qid_to_rows[qid].copy()
                sub["question_id"] = f"{qid}__r{k}"
                chunks.append(sub)
        if not chunks:
            return df.iloc[0:0].copy()
        return pd.concat(chunks, ignore_index=True)

    new_static = (
        _resample(static_df) if static_df is not None and not static_df.empty else None
    )
    new_reward = (
        _resample(reward_df) if reward_df is not None and not reward_df.empty else None
    )

    if new_reward is not None and not new_reward.empty:
        mu = float(new_reward["reward_raw"].mean())
        sigma = float(new_reward["reward_raw"].std(ddof=0))
        if not math.isfinite(sigma) or sigma < 1e-8:
            sigma = 1.0
        new_reward["reward_z"] = (new_reward["reward_raw"] - mu) / sigma

    return new_static, new_reward


@dataclass
class WorkerArgs:
    domain: str
    method: str
    static_jsonl: str | None
    arena_jsonl: str | None
    seed: int
    hypers: dict


def _fit_replicate(wa: WorkerArgs) -> pd.DataFrame:
    # Force CPU and single-thread torch BEFORE importing it. Otherwise N
    # spawned workers all try to allocate on the same GPU and OOM, and they
    # each launch OMP/MKL thread pools that oversubscribe the box.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)

    dualeval = _get_dualeval()
    rng = np.random.default_rng(wa.seed)

    need_static = wa.method in {"dualeval", "static"} and wa.static_jsonl is not None
    need_arena = wa.method in {"dualeval", "bt"} and wa.arena_jsonl is not None

    static_df = dualeval.load_static_jsonl([wa.static_jsonl]) if need_static else None
    reward_df = dualeval.load_arena_reward_jsonl([wa.arena_jsonl]) if need_arena else None
    new_static, new_reward = resample_per_source(static_df, reward_df, rng)

    if new_reward is not None and not new_reward.empty:
        bb_th, tie_d = dualeval.resolve_pairwise_thresholds(
            new_reward,
            bb_ratio=wa.hypers["bb_ratio"],
            tie_ratio=wa.hypers["tie_ratio"],
        )
        pairwise = dualeval.build_soft_pairwise_targets(
            new_reward, both_bad_threshold=bb_th, tie_delta=tie_d,
        )
    else:
        pairwise = None

    static_for_fit = new_static if (new_static is not None and not new_static.empty) else None

    if wa.method == "dualeval":
        model_params, _, _ = dualeval.fit_irt(
            static_for_fit,
            pairwise,
            num_epochs=wa.hypers["num_epochs"],
            lr=wa.hypers["lr"],
            lambda_static=wa.hypers["lambda_static"],
            lambda_arena=wa.hypers["lambda_arena"],
            lambda_bb=wa.hypers["lambda_bb"],
            reg_lambda=wa.hypers["reg_lambda"],
            verbose=False,
        )
    elif wa.method == "static":
        # Static-only 2PL: same fit_irt path with arena/bb losses disabled.
        model_params, _, _ = dualeval.fit_irt(
            static_for_fit,
            None,
            num_epochs=wa.hypers["num_epochs"],
            lr=wa.hypers["lr"],
            lambda_static=wa.hypers["lambda_static"],
            lambda_arena=0.0,
            lambda_bb=0.0,
            reg_lambda=wa.hypers["reg_lambda"],
            verbose=False,
        )
    elif wa.method == "bt":
        # Naive Bradley-Terry on arena pairs; no item parameters.
        model_params, _, _ = dualeval.fit_bt(
            pairwise,
            num_epochs=wa.hypers["num_epochs"],
            lr=wa.hypers["lr"],
            lambda_arena=wa.hypers["lambda_arena"],
            reg_lambda=wa.hypers["reg_lambda"],
            verbose=False,
        )
    else:
        raise ValueError(f"unknown method {wa.method}")

    out = model_params[["model_name", "theta"]].copy()
    out["domain"] = wa.domain
    out["seed"] = wa.seed
    out["method"] = wa.method
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS.keys()),
                    help="Subset of domains. Default: all four.")
    ap.add_argument("--methods", nargs="+", default=["dualeval"],
                    choices=list(METHODS),
                    help="Methods to bootstrap. Skips invalid combos "
                         "(static needs static data, bt needs arena data).")
    ap.add_argument("-B", type=int, default=100, help="Replicates per (domain, method).")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--output-dir", default="results/bootstrap_smoke")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[WorkerArgs] = []
    skipped: list[tuple[str, str]] = []
    for d in args.domains:
        if d not in DOMAINS:
            raise SystemExit(f"Unknown domain {d}. Choices: {list(DOMAINS)}")
        info = DOMAINS[d]
        for m in args.methods:
            if not is_valid(d, m):
                skipped.append((d, m))
                continue
            for b in range(args.B):
                jobs.append(WorkerArgs(
                    domain=d,
                    method=m,
                    static_jsonl=str(REPO_ROOT / info["static"]) if info["static"] else None,
                    arena_jsonl=str(REPO_ROOT / info["arena"]) if info["arena"] else None,
                    # Hash domain+method into seed so different (d, m) pairs
                    # draw uncorrelated replicates; reproducible within a (d, m).
                    seed=args.seed_offset + b + 10_000 * (hash((d, m)) % 1000),
                    hypers=DEFAULT_HYPERS,
                ))

    if skipped:
        print(f"Skipped invalid combos: {skipped}")
    print(f"Bootstrap smoke: {len(jobs)} fits across "
          f"{len(args.domains)} domains × {len(args.methods)} methods, "
          f"{args.workers} workers, B={args.B}")
    t0 = time.time()
    if args.workers == 1:
        results = [_fit_replicate(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_fit_replicate, jobs, chunksize=1))
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s total ({elapsed/len(jobs):.1f}s/fit)")

    by_key: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for r in results:
        key = (r["domain"].iloc[0], r["method"].iloc[0])
        by_key.setdefault(key, []).append(r)
    for (d, m), parts in by_key.items():
        df = pd.concat(parts, ignore_index=True)
        path = out_dir / f"{d}_{m}_theta_replicates.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {path}  ({df['seed'].nunique()} replicates × "
              f"{df['model_name'].nunique()} models = {len(df)} rows)")


if __name__ == "__main__":
    main()
