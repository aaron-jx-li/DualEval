#!/usr/bin/env python3
"""
IRT ranking for static benchmarks and arena reward signals.

Supported modes:
1. ``static`` — 2PL-IRT on binary correctness labels only
       P(y_{i,q}=1) = sigmoid(a_q * (theta_i - b_q))
2. ``arena``  — reward-distilled soft-pairwise IRT
       p*_{ij,q} = sigmoid(z_{i,q} - z_{j,q})
       P_hat(i>j|q) = sigmoid(gamma * (p_{i,q} - p_{j,q}))
3. ``both``   — joint static + arena on shared (theta, b, a) parameters
4. ``BT``     — naive Bradley-Terry baseline (arena data only)
       P(i > j | q) = sigmoid(s_i - s_j)
       No question-specific parameters; both-bad pairs included in loss.

Design notes:
  - Both-bad pairs are excluded from the arena BCE loss and only enter the
    bb anchoring term, so the two terms operate on disjoint pair sets. This
    avoids conflicting gradients on the same pairs (the arena loss pushing
    theta_i relative to theta_j while the bb loss pushes both toward failure).
  - bb_ratio and tie_ratio are the sole threshold interface.  Percentile-
    based thresholds are derived from the empirical pairwise z-score
    distribution, so the actual flagged fractions match the requested ratios
    regardless of domain-level reward biases.  Absolute z-score fallbacks
    are not supported.

Example:
    python dualeval/dualeval.py --config dualeval/config_dualeval.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODES = {"static", "arena", "both", "BT"}


def mode_uses_static(mode: str) -> bool:
    return mode in {"static", "both"}


def mode_uses_arena(mode: str) -> bool:
    return mode in {"arena", "both", "BT"}


# ---------------------------------------------------------------------------
# CLI / config
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit IRT rankings from static and/or arena reward JSONL data.",
    )
    p.add_argument("--config", default=None, help="YAML config file.")
    p.add_argument("--static-jsonl", nargs="*", default=None,
                   help="Static eval responses.jsonl files.")
    p.add_argument("--arena-reward-jsonl", nargs="*", default=None,
                   help="Arena eval responses.jsonl files with reward scores.")
    p.add_argument("--mode", default=None, choices=sorted(MODES),
                   help="Training mode: static | arena | both.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--save-plot", default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lambda-static", type=float, default=None,
                   help="Weight for static 2PL loss.")
    p.add_argument("--lambda-arena", type=float, default=None,
                   help="Weight for arena soft-pairwise loss.")
    p.add_argument("--lambda-bb", type=float, default=None,
                   help="Weight for both-bad anchoring loss.")
    p.add_argument("--reg-lambda", type=float, default=None,
                   help="L2 regularization coefficient.")
    p.add_argument(
        "--bb-ratio", type=float, default=None,
        help=(
            "Target fraction of pairwise comparisons to treat as both-bad "
            "(e.g. 0.15).  Threshold is the bb_ratio-th percentile of "
            "max(z_i, z_j) over all pairs.  Set to 0.0 to disable."
        ),
    )
    p.add_argument(
        "--tie-ratio", type=float, default=None,
        help=(
            "Target fraction of pairwise comparisons to treat as ties "
            "(e.g. 0.10).  Threshold is the tie_ratio-th percentile of "
            "|z_i - z_j| over all pairs.  Set to 0.0 to disable."
        ),
    )
    p.add_argument("--save-pairwise-targets", action="store_true",
                   help="Write distilled arena pairwise targets to CSV.")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


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
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _expand_env(raw)


def _ensure_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def _resolve_paths(paths: list[str] | None) -> list[str] | None:
    return None if paths is None else [str(_resolve_path(p)) for p in paths]


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_yaml_config(args.config)
    inp = cfg.get("input", {})
    tr = cfg.get("training", {})
    out = cfg.get("output", {})

    if args.static_jsonl is None:
        args.static_jsonl = _ensure_list(inp.get("static_jsonl"))
    if args.arena_reward_jsonl is None:
        args.arena_reward_jsonl = _ensure_list(inp.get("arena_reward_jsonl"))
    if args.mode is None:
        args.mode = str(tr.get("mode", "both")).strip()
    if args.mode not in MODES:
        raise SystemExit(f"Invalid mode '{args.mode}'. Choose from: {sorted(MODES)}")
    if args.output_dir is None:
        args.output_dir = str(_resolve_path(out.get("output_dir", "results/dualeval/default")))
    if args.save_plot is None and out.get("save_plot") is not None:
        args.save_plot = str(_resolve_path(str(out["save_plot"])))
    if args.num_epochs is None:
        args.num_epochs = int(tr.get("num_epochs", 2000))
    if args.lr is None:
        args.lr = float(tr.get("lr", 0.02))
    if args.lambda_static is None:
        args.lambda_static = float(tr.get("lambda_static", 1.0))
    if args.lambda_arena is None:
        args.lambda_arena = float(tr.get("lambda_arena", 1.0))
    if args.lambda_bb is None:
        args.lambda_bb = float(tr.get("lambda_bb", 0.2))
    if args.reg_lambda is None:
        args.reg_lambda = float(tr.get("reg_lambda", 1e-3))
    if args.bb_ratio is None and tr.get("bb_ratio") is not None:
        args.bb_ratio = float(tr["bb_ratio"])
    if args.tie_ratio is None and tr.get("tie_ratio") is not None:
        args.tie_ratio = float(tr["tie_ratio"])
    if not args.save_pairwise_targets:
        args.save_pairwise_targets = bool(out.get("save_pairwise_targets", False))
    if not args.no_plot:
        args.no_plot = bool(out.get("no_plot", False))
    if not args.quiet:
        args.quiet = bool(out.get("quiet", False))

    args.static_jsonl = _resolve_paths(args.static_jsonl)
    args.arena_reward_jsonl = _resolve_paths(args.arena_reward_jsonl)
    return args


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def source_tag_for_path(path: Path) -> str:
    parent_name = path.parent.name.strip()
    return parent_name if parent_name else path.stem


def load_static_jsonl(jsonl_paths: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for jsonl_path in jsonl_paths:
        path = Path(jsonl_path)
        tag = source_tag_for_path(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("model_label") is None or d.get("correct") is None:
                    continue
                dataset = str(d.get("dataset", "unknown"))
                sample_index = d.get("sample_index")
                rows.append({
                    "source": tag,
                    "benchmark": dataset,
                    "model_name": str(d["model_label"]),
                    "question_id": f"{tag}::{dataset}_{sample_index}",
                    "judge_result": int(bool(d["correct"])),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["model_name", "question_id"], keep="last").reset_index(drop=True)


def load_arena_reward_jsonl(jsonl_paths: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for jsonl_path in jsonl_paths:
        path = Path(jsonl_path)
        tag = source_tag_for_path(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("reward") is None:
                    continue
                rows.append({
                    "source": tag,
                    "benchmark": "Arena",
                    "model_name": str(d["model_label"]),
                    "question_id": f"{tag}::{d['item_id']}",
                    "reward_raw": float(d["reward"]),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["model_name", "question_id"], keep="last").reset_index(drop=True)
    mu = float(df["reward_raw"].mean())
    sigma = float(df["reward_raw"].std(ddof=0))
    if not math.isfinite(sigma) or sigma < 1e-8:
        sigma = 1.0
    df["reward_z"] = (df["reward_raw"] - mu) / sigma
    return df


def resolve_pairwise_thresholds(
    reward_df: pd.DataFrame,
    *,
    bb_ratio: float,
    tie_ratio: float,
) -> tuple[float, float]:
    """Derive both_bad_threshold and tie_delta from target pair-fraction ratios.

    Both thresholds come from the empirical pairwise z-score distribution so
    the actual flagged fractions match the requested ratios, regardless of
    domain-level reward biases in the global z-scoring.

    Args:
        bb_ratio:  target fraction of pairs to flag as both-bad.
                   A pair is both-bad when max(z_i, z_j) < threshold, i.e.
                   both models score below the threshold on that question.
                   Threshold = bb_ratio-th percentile of max(z_i, z_j) so
                   the flagged fraction matches bb_ratio exactly.
                   0.0 sets threshold to -inf, disabling both-bad flagging.
        tie_ratio: target fraction of pairs to flag as ties.
                   Threshold = tie_ratio-th percentile of |z_i - z_j|.
                   0.0 disables tie flagging.

    Returns:
        (both_bad_threshold, tie_delta)
    """
    max_scores: list[float] = []
    abs_diffs: list[float] = []
    for _, group in reward_df.groupby("question_id", sort=False):
        if len(group) < 2:
            continue
        zs = group["reward_z"].to_numpy()
        for i in range(len(zs)):
            for j in range(i + 1, len(zs)):
                max_scores.append(float(max(zs[i], zs[j])))
                abs_diffs.append(float(abs(zs[i] - zs[j])))

    if not max_scores:
        return -math.inf, 0.0

    max_arr = np.array(max_scores)
    diff_arr = np.array(abs_diffs)
    both_bad_threshold = float(np.percentile(max_arr, bb_ratio * 100))
    tie_delta = float(np.percentile(diff_arr, tie_ratio * 100)) if tie_ratio > 0.0 else 0.0
    return both_bad_threshold, tie_delta


def build_soft_pairwise_targets(
    reward_df: pd.DataFrame,
    *,
    both_bad_threshold: float,
    tie_delta: float,
) -> pd.DataFrame:
    """Convert per-response reward z-scores into soft pairwise comparison rows.

    For each question every C(n,2) model pair receives:
      target_prob  = sigmoid(z_i - z_j)           soft Bradley-Terry preference
      both_bad     = max(z_i, z_j) < threshold     both models clearly fail
      tie          = |z_i - z_j| < tie_delta        scores too close to signal

    The two flags are independent; a pair can be neither, either, or both.
    Both-bad and tie pairs are excluded from the arena BCE loss at fit time;
    the bb loss only sees both-bad non-tie pairs.
    """
    rows: list[dict[str, Any]] = []
    for question_id, group in reward_df.groupby("question_id", sort=False):
        if len(group) < 2:
            continue
        group = group.sort_values("model_name").reset_index(drop=True)
        for idx1, idx2 in combinations(range(len(group)), 2):
            r1, r2 = group.iloc[idx1], group.iloc[idx2]
            z1, z2 = float(r1["reward_z"]), float(r2["reward_z"])
            rows.append({
                "source": r1["source"],
                "benchmark": "Arena",
                "question_id": question_id,
                "model_1": r1["model_name"],
                "model_2": r2["model_name"],
                "reward_raw_1": float(r1["reward_raw"]),
                "reward_raw_2": float(r2["reward_raw"]),
                "reward_z_1": z1,
                "reward_z_2": z2,
                "target_prob": 1.0 / (1.0 + math.exp(-(z1 - z2))),
                "both_bad": bool(max(z1, z2) < both_bad_threshold),
                "tie": bool(abs(z1 - z2) < tie_delta),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    # This IRT fit is tiny (a few hundred questions / models), so it runs fast
    # on CPU and has no business claiming a shared GPU. Default to CPU; opt into
    # CUDA explicitly with DUALEVAL_USE_CUDA=1.
    if os.environ.get("DUALEVAL_USE_CUDA") == "1" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_irt(
    static_df: pd.DataFrame | None,
    pairwise_df: pd.DataFrame | None,
    *,
    num_epochs: int,
    lr: float,
    lambda_static: float,
    lambda_arena: float,
    lambda_bb: float,
    reg_lambda: float,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit joint IRT model from static benchmark and/or soft-pairwise arena data.

    IRT success probability (2PL):
        p_{i,q} = sigmoid(a_q * (theta_i - b_q)),  a_q = exp(k_q) > 0

    Arena pairwise preference (nested sigmoid):
        P_hat(i > j | q) = sigmoid(gamma * (p_{i,q} - p_{j,q}))

    Three loss terms, operating on disjoint pair subsets:
        loss_static  BCE(a_q*(theta_i - b_q), y_{i,q})   [static labels]
        loss_arena   soft BCE(gamma*(p_i - p_j), p*)      [hard pairs only]
        loss_bb      -E[log(1-p_i) + log(1-p_j)]         [both-bad pairs only]

    Hard pairs: non-tie AND non-both-bad.
    This partition ensures the arena BCE loss only signals relative preference
    between models of comparable quality.  The bb term independently anchors
    question difficulty without interfering with the arena gradient.

    Identifiability: after each optimiser step theta is zero-mean centred and
    the same shift is absorbed into b, preserving theta_i - b_q exactly.

    gamma prior: log(gamma) is L2-regularised toward log(4).
    Rationale: at perfect fit gamma*(p_i-p_j) = z_i-z_j.  By the MVT,
    p_i-p_j = sigma'(c)*(z_i-z_j), so gamma = 1/sigma'(c).  sigma'(0)=1/4
    is the global maximum, giving gamma_0 = 4 as the equilibrium lower bound.
    """
    static = static_df.copy() if static_df is not None else pd.DataFrame()
    pairwise = pairwise_df.copy() if pairwise_df is not None else pd.DataFrame()
    has_static = not static.empty
    has_arena = not pairwise.empty
    if not has_static and not has_arena:
        raise SystemExit("Need at least one non-empty data source.")

    # Build unified model/question vocabularies across both data sources
    model_series: list[pd.Series] = []
    question_series: list[pd.Series] = []
    if has_static:
        model_series.append(static["model_name"])
        question_series.append(static["question_id"])
    if has_arena:
        model_series.extend([pairwise["model_1"], pairwise["model_2"]])
        question_series.append(pairwise["question_id"])

    all_models = pd.Index(pd.unique(pd.concat(model_series, ignore_index=True)), name="model_name")
    all_questions = pd.Index(pd.unique(pd.concat(question_series, ignore_index=True)), name="question_id")
    model_to_idx = {m: i for i, m in enumerate(all_models)}
    q_to_idx = {q: i for i, q in enumerate(all_questions)}

    question_meta: dict[str, dict[str, str]] = {}
    if has_static:
        for _, row in static[["question_id", "source", "benchmark"]].drop_duplicates().iterrows():
            question_meta[str(row["question_id"])] = {
                "source": str(row["source"]), "benchmark": str(row["benchmark"]),
            }
        static["m_idx"] = static["model_name"].map(model_to_idx)
        static["q_idx"] = static["question_id"].map(q_to_idx)
    if has_arena:
        for _, row in pairwise[["question_id", "source", "benchmark"]].drop_duplicates().iterrows():
            question_meta[str(row["question_id"])] = {
                "source": str(row["source"]), "benchmark": str(row["benchmark"]),
            }
        pairwise["m1_idx"] = pairwise["model_1"].map(model_to_idx)
        pairwise["m2_idx"] = pairwise["model_2"].map(model_to_idx)
        pairwise["q_idx"] = pairwise["question_id"].map(q_to_idx)

    device = _get_device()

    # Static tensors
    m_s = q_s = y_s = None
    if has_static:
        m_s = torch.tensor(static["m_idx"].values, dtype=torch.long, device=device)
        q_s = torch.tensor(static["q_idx"].values, dtype=torch.long, device=device)
        y_s = torch.tensor(static["judge_result"].values, dtype=torch.float32, device=device)

    # Arena tensors — two disjoint subsets of non-tie pairs:
    #   hard  (non-tie, non-both-bad) → arena BCE loss
    #   bb    (both-bad, non-tie)     → bb anchoring loss only
    m1_t = m2_t = q_t = soft_t = None
    m1_bb = m2_bb = q_bb = None
    if has_arena:
        hard = pairwise[~pairwise["tie"] & ~pairwise["both_bad"]].copy()
        bb = pairwise[pairwise["both_bad"] & ~pairwise["tie"]].copy()
        if not hard.empty:
            m1_t = torch.tensor(hard["m1_idx"].values, dtype=torch.long, device=device)
            m2_t = torch.tensor(hard["m2_idx"].values, dtype=torch.long, device=device)
            q_t = torch.tensor(hard["q_idx"].values, dtype=torch.long, device=device)
            soft_t = torch.tensor(hard["target_prob"].values, dtype=torch.float32, device=device)
        if not bb.empty:
            m1_bb = torch.tensor(bb["m1_idx"].values, dtype=torch.long, device=device)
            m2_bb = torch.tensor(bb["m2_idx"].values, dtype=torch.long, device=device)
            q_bb = torch.tensor(bb["q_idx"].values, dtype=torch.long, device=device)

    n_models = len(all_models)
    n_questions = len(all_questions)

    theta = nn.Embedding(n_models, 1, device=device)
    b = nn.Embedding(n_questions, 1, device=device)
    k = nn.Embedding(n_questions, 1, device=device)
    nn.init.zeros_(theta.weight)
    nn.init.zeros_(b.weight)
    nn.init.zeros_(k.weight)

    _LOG_GAMMA_0 = math.log(1.0 / (0.5 * 0.5))  # log(4): equilibrium lower bound
    log_gamma = nn.Parameter(torch.tensor(_LOG_GAMMA_0, device=device))

    opt_params = [*theta.parameters(), *b.parameters(), *k.parameters()]
    if has_arena:
        opt_params.append(log_gamma)

    optimizer = optim.Adam(opt_params, lr=lr)
    bce_logits = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        loss_static = torch.tensor(0.0, device=device)
        if has_static and m_s is not None:
            logits_s = torch.exp(k(q_s).squeeze(-1)) * (theta(m_s).squeeze(-1) - b(q_s).squeeze(-1))
            loss_static = bce_logits(logits_s, y_s)

        loss_arena = torch.tensor(0.0, device=device)
        if has_arena and m1_t is not None:
            a_q = torch.exp(k(q_t).squeeze(-1))
            b_q = b(q_t).squeeze(-1)
            gamma = torch.exp(log_gamma)
            p1 = torch.sigmoid(a_q * (theta(m1_t).squeeze(-1) - b_q))
            p2 = torch.sigmoid(a_q * (theta(m2_t).squeeze(-1) - b_q))
            loss_arena = bce_logits(gamma * (p1 - p2), soft_t)

        loss_bb = torch.tensor(0.0, device=device)
        if has_arena and m1_bb is not None:
            a_q_bb = torch.exp(k(q_bb).squeeze(-1))
            b_q_bb = b(q_bb).squeeze(-1)
            p1_bb = torch.sigmoid(a_q_bb * (theta(m1_bb).squeeze(-1) - b_q_bb))
            p2_bb = torch.sigmoid(a_q_bb * (theta(m2_bb).squeeze(-1) - b_q_bb))
            loss_bb = -(
                torch.log(1.0 - p1_bb + 1e-6).mean()
                + torch.log(1.0 - p2_bb + 1e-6).mean()
            )

        reg = reg_lambda * (
            theta.weight.pow(2).mean()
            + b.weight.pow(2).mean()
            + k.weight.pow(2).mean()
        )
        if has_arena:
            reg = reg + reg_lambda * (log_gamma - _LOG_GAMMA_0).pow(2)

        total = lambda_static * loss_static + lambda_arena * loss_arena + lambda_bb * loss_bb + reg
        total.backward()
        optimizer.step()

        # Identifiability anchor: zero-mean theta; absorb same shift into b
        # so that theta_i - b_q is preserved exactly.
        with torch.no_grad():
            shift = theta.weight.mean()
            theta.weight.sub_(shift)
            b.weight.sub_(shift)

        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            gamma_val = torch.exp(log_gamma).item() if has_arena else float("nan")
            print(
                f"  Epoch {epoch:5d} | "
                f"static={loss_static.item():.4f}  "
                f"arena={loss_arena.item():.4f}  "
                f"bb={loss_bb.item():.4f}  "
                f"gamma={gamma_val:.4f}  "
                f"total={total.item():.4f}"
            )

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    b_np = b.weight.detach().cpu().numpy().squeeze(-1)
    k_np = k.weight.detach().cpu().numpy().squeeze(-1)
    gamma_learned = float(torch.exp(log_gamma).detach().cpu().item()) if has_arena else None

    model_params = (
        pd.DataFrame({"model_name": all_models, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )

    question_rows: list[dict[str, Any]] = []
    for i, qid in enumerate(all_questions):
        meta = question_meta.get(str(qid), {})
        question_rows.append({
            "question_id": qid,
            "source": meta.get("source", "unknown"),
            "benchmark": meta.get("benchmark", "unknown"),
            "difficulty_b": float(b_np[i]),
            "k_raw": float(k_np[i]),
            "discrimination_exp_k": math.exp(float(k_np[i])),
        })
    question_params = (
        pd.DataFrame(question_rows)
        .sort_values("difficulty_b", ascending=False)
        .reset_index(drop=True)
    )

    metadata: dict[str, Any] = {
        "learned_gamma": gamma_learned,
        "n_models": int(n_models),
        "n_questions": int(n_questions),
        "has_static": bool(has_static),
        "has_arena": bool(has_arena),
    }
    return model_params, question_params, metadata


# ---------------------------------------------------------------------------
# Bradley-Terry baseline
# ---------------------------------------------------------------------------

def fit_bt(
    pairwise_df: pd.DataFrame,
    *,
    num_epochs: int,
    lr: float,
    lambda_arena: float,
    reg_lambda: float,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit naive Bradley-Terry model from soft pairwise arena data.

    P(i > j | q) = sigmoid(s_i - s_j)

    No question-specific parameters (no difficulty, no discrimination, no gamma).
    Both-bad pairs are included in the loss — BT has no bb anchoring term and
    no structural reason to exclude them.  Tie pairs are excluded.

    Serves as a baseline: if IRT-structured modes do not clearly outperform
    BT on arena_hard_pair_accuracy, the question/discrimination structure is
    not contributing signal beyond raw model strength.
    """
    pairwise = pairwise_df.copy()

    all_models = pd.Index(
        pd.unique(pd.concat([pairwise["model_1"], pairwise["model_2"]], ignore_index=True)),
        name="model_name",
    )
    model_to_idx = {m: i for i, m in enumerate(all_models)}
    pairwise["m1_idx"] = pairwise["model_1"].map(model_to_idx)
    pairwise["m2_idx"] = pairwise["model_2"].map(model_to_idx)

    device = _get_device()

    non_tie = pairwise[~pairwise["tie"]].copy()
    if non_tie.empty:
        raise SystemExit("No non-tie pairwise data available for BT fitting.")

    m1_t = torch.tensor(non_tie["m1_idx"].values, dtype=torch.long, device=device)
    m2_t = torch.tensor(non_tie["m2_idx"].values, dtype=torch.long, device=device)
    soft_t = torch.tensor(non_tie["target_prob"].values, dtype=torch.float32, device=device)

    theta = nn.Embedding(len(all_models), 1, device=device)
    nn.init.zeros_(theta.weight)

    optimizer = optim.Adam(theta.parameters(), lr=lr)
    bce_logits = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logit = theta(m1_t).squeeze(-1) - theta(m2_t).squeeze(-1)
        loss = lambda_arena * bce_logits(logit, soft_t) + reg_lambda * theta.weight.pow(2).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            theta.weight.sub_(theta.weight.mean())

        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            print(f"  Epoch {epoch:5d} | bt_loss={loss.item():.4f}")

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    model_params = (
        pd.DataFrame({"model_name": all_models, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )
    metadata: dict[str, Any] = {
        "learned_gamma": None,
        "n_models": int(len(all_models)),
        "n_questions": 0,
        "has_static": False,
        "has_arena": True,
    }
    return model_params, pd.DataFrame(), metadata


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    static_df: pd.DataFrame | None,
    pairwise_df: pd.DataFrame | None,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
    *,
    learned_gamma: float | None,
) -> dict[str, Any]:
    theta_map = model_params.set_index("model_name")["theta"]
    q_map = (
        question_params.set_index("question_id")[["difficulty_b", "discrimination_exp_k"]]
        if not question_params.empty
        else pd.DataFrame(columns=["difficulty_b", "discrimination_exp_k"])
    )
    metrics: dict[str, Any] = {}

    if static_df is not None and not static_df.empty:
        st = static_df.copy()
        st["theta"] = st["model_name"].map(theta_map)
        st["difficulty_b"] = st["question_id"].map(q_map["difficulty_b"])
        st["a_q"] = st["question_id"].map(q_map["discrimination_exp_k"])
        st = st.dropna(subset=["theta", "difficulty_b", "a_q"])
        logits = st["a_q"] * (st["theta"] - st["difficulty_b"])
        pred = (logits >= 0).astype(int)
        metrics["static_accuracy"] = float(pred.eq(st["judge_result"].astype(int)).mean())

    if pairwise_df is not None and not pairwise_df.empty:
        ar = pairwise_df.copy()
        ar["theta_1"] = ar["model_1"].map(theta_map)
        ar["theta_2"] = ar["model_2"].map(theta_map)
        ar = ar.dropna(subset=["theta_1", "theta_2"])
        if not question_params.empty:
            # IRT modes: preference mediated through IRT success probabilities
            ar["difficulty_b"] = ar["question_id"].map(q_map["difficulty_b"])
            ar["a_q"] = ar["question_id"].map(q_map["discrimination_exp_k"])
            ar = ar.dropna(subset=["difficulty_b", "a_q"])
            a_q = ar["a_q"].to_numpy()
            b_q = ar["difficulty_b"].to_numpy()
            p1 = 1.0 / (1.0 + np.exp(-a_q * (ar["theta_1"].to_numpy() - b_q)))
            p2 = 1.0 / (1.0 + np.exp(-a_q * (ar["theta_2"].to_numpy() - b_q)))
            gamma = learned_gamma if learned_gamma is not None else 1.0
            pred_prob = 1.0 / (1.0 + np.exp(-gamma * (p1 - p2)))
        else:
            # BT mode: preference is direct strength difference
            logit = ar["theta_1"].to_numpy() - ar["theta_2"].to_numpy()
            pred_prob = 1.0 / (1.0 + np.exp(-logit))
        target = ar["target_prob"].to_numpy()
        eps = 1e-6
        metrics["arena_soft_logloss"] = float(
            -np.mean(target * np.log(pred_prob + eps) + (1 - target) * np.log(1 - pred_prob + eps))
        )
        both_bad_mask = ar["both_bad"].to_numpy(dtype=bool)
        tie_mask = ar["tie"].to_numpy(dtype=bool)
        hard_mask = ~both_bad_mask & ~tie_mask
        metrics["arena_both_bad_pairs"] = int(both_bad_mask.sum())
        metrics["arena_tie_pairs"] = int(tie_mask.sum())
        metrics["arena_hard_eval_pairs"] = int(hard_mask.sum())
        if hard_mask.any():
            metrics["arena_hard_pair_accuracy"] = float(
                ((pred_prob[hard_mask] >= 0.5) == (target[hard_mask] >= 0.5)).mean()
            )
        else:
            metrics["arena_hard_pair_accuracy"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_difficulty_and_ability(
    model_ranking: pd.DataFrame,
    question_ranking: pd.DataFrame,
    *,
    save_path: str | None,
    title: str,
) -> None:
    if question_ranking.empty or model_ranking.empty or not save_path:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    qr = question_ranking.copy()
    benchmarks = sorted(qr["benchmark"].fillna("unknown").unique())
    colors = plt.cm.tab10.colors[: max(1, len(benchmarks))]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bins = np.linspace(qr["difficulty_b"].min(), qr["difficulty_b"].max(), 41)
    centers = (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins)
    bottoms = np.zeros(len(centers))

    for idx, benchmark in enumerate(benchmarks):
        counts, _ = np.histogram(qr.loc[qr["benchmark"] == benchmark, "difficulty_b"], bins=bins)
        ax.bar(
            centers, counts, width=widths, bottom=bottoms,
            color=colors[idx % len(colors)], alpha=0.85,
            align="center", edgecolor="white", linewidth=0.3,
            label=benchmark,
        )
        bottoms += counts

    y_top = max(1.0, bottoms.max() * 1.5)
    ax.set_ylim(0, y_top)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.35)

    model_colors = plt.cm.Dark2.colors
    benchmark_handles = [
        Line2D([0], [0], color=colors[idx % len(colors)], lw=6, label=bm)
        for idx, bm in enumerate(benchmarks)
    ]
    model_handles: list[Line2D] = []
    for idx, (_, row) in enumerate(model_ranking.iterrows()):
        color = model_colors[idx % len(model_colors)]
        ax.axvline(row["theta"], 0, 0.85, color=color, linestyle="--", linewidth=1.3, alpha=0.9)
        ax.text(
            row["theta"], y_top * (0.88 + 0.03 * (idx % 2)),
            f"{idx + 1}", ha="center", va="bottom", fontsize=8, color=color,
        )
        model_handles.append(Line2D(
            [0], [0], color=color, linestyle="--", marker="o", markersize=5,
            label=f"{idx + 1}. {row['model_name']}",
        ))

    ax.set_xlabel(r"Latent scale (ability $\theta$ / difficulty $b$)")
    ax.set_ylabel("Number of questions")
    ax.set_title(title)

    n_model_cols = 2 if len(model_handles) > 8 else 1
    benchmark_legend = ax.legend(
        handles=benchmark_handles, title="Question Sources",
        frameon=True, framealpha=0.9, loc="upper left",
        bbox_to_anchor=(0.0, -0.14), bbox_transform=ax.transAxes,
        borderaxespad=0.0, fontsize=8, title_fontsize=9,
    )
    ax.add_artist(benchmark_legend)
    ax.legend(
        handles=model_handles, title="Model Rankings",
        frameon=True, framealpha=0.9, loc="upper left",
        bbox_to_anchor=(0.42, -0.14), bbox_transform=ax.transAxes,
        borderaxespad=0.0, ncol=n_model_cols, fontsize=8, title_fontsize=9,
    )

    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=500, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args = apply_config_defaults(args)
    verbose = not args.quiet

    if args.num_epochs < 1:
        raise SystemExit("--num-epochs must be at least 1.")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive.")
    if args.reg_lambda < 0:
        raise SystemExit("--reg-lambda must be non-negative.")

    use_static = mode_uses_static(args.mode)
    use_arena = mode_uses_arena(args.mode)

    if use_arena:
        if args.mode == "BT":
            # BT has no bb anchoring term; bb_ratio is optional (used only for
            # pair statistics reporting).  Default to 0.0 if not provided.
            if args.bb_ratio is None:
                args.bb_ratio = 0.0
        else:
            if args.bb_ratio is None:
                raise SystemExit(
                    "bb_ratio must be set for arena/both modes.  "
                    "Specify it in the config file or via --bb-ratio "
                    "(use 0.0 to disable both-bad anchoring)."
                )
        if not 0.0 <= args.bb_ratio <= 1.0:
            raise SystemExit(f"bb_ratio must be in [0, 1], got {args.bb_ratio}.")
        tie_ratio = args.tie_ratio if args.tie_ratio is not None else 0.0
        if not 0.0 <= tie_ratio <= 1.0:
            raise SystemExit(f"tie_ratio must be in [0, 1], got {tie_ratio}.")
    else:
        tie_ratio = 0.0

    # Load static data
    static_df = pd.DataFrame()
    if use_static:
        if not args.static_jsonl:
            raise SystemExit(f"Mode '{args.mode}' requires --static-jsonl.")
        print(f"Loading static JSONL: {len(args.static_jsonl)} file(s)")
        static_df = load_static_jsonl(args.static_jsonl)
        print(
            f"  {len(static_df)} rows, "
            f"{static_df['model_name'].nunique() if not static_df.empty else 0} models, "
            f"{static_df['question_id'].nunique() if not static_df.empty else 0} questions"
        )
    elif args.static_jsonl:
        print(f"Note: ignoring static_jsonl (mode='{args.mode}' does not use static data).")

    # Load arena data and build pairwise targets
    pairwise_df = pd.DataFrame()
    if use_arena:
        if not args.arena_reward_jsonl:
            raise SystemExit(f"Mode '{args.mode}' requires --arena-reward-jsonl.")
        print(f"Loading arena reward JSONL: {len(args.arena_reward_jsonl)} file(s)")
        reward_df = load_arena_reward_jsonl(args.arena_reward_jsonl)
        if reward_df.empty:
            print("  0 usable reward rows found.")
        else:
            print(
                f"  {len(reward_df)} rows, "
                f"{reward_df['model_name'].nunique()} models, "
                f"{reward_df['question_id'].nunique()} questions, "
                f"reward mean={reward_df['reward_raw'].mean():.3f}, "
                f"reward std={reward_df['reward_raw'].std(ddof=0):.3f}"
            )
            both_bad_threshold, tie_delta = resolve_pairwise_thresholds(
                reward_df,
                bb_ratio=args.bb_ratio,
                tie_ratio=tie_ratio,
            )
            print(
                f"  bb_ratio={args.bb_ratio} → both_bad_threshold={both_bad_threshold:.4f}  "
                f"tie_ratio={tie_ratio} → tie_delta={tie_delta:.4f}"
            )
            pairwise_df = build_soft_pairwise_targets(
                reward_df,
                both_bad_threshold=both_bad_threshold,
                tie_delta=tie_delta,
            )
            if not pairwise_df.empty:
                n_bb = int(pairwise_df["both_bad"].sum())
                n_tie = int(pairwise_df["tie"].sum())
                n_hard = int((~pairwise_df["both_bad"] & ~pairwise_df["tie"]).sum())
                if args.mode == "BT":
                    print(
                        f"  {len(pairwise_df)} total pairs: "
                        f"{n_hard + n_bb} in BT loss ({n_bb} both-bad included)  "
                        f"{n_tie} ties (excluded)"
                    )
                else:
                    print(
                        f"  {len(pairwise_df)} total pairs: "
                        f"{n_hard} hard (arena loss)  "
                        f"{n_bb} both-bad (bb loss only)  "
                        f"{n_tie} ties (excluded)"
                    )
    elif args.arena_reward_jsonl:
        print(f"Note: ignoring arena_reward_jsonl (mode='{args.mode}' does not use arena data).")

    if static_df.empty and pairwise_df.empty:
        raise SystemExit("No usable static or arena-reward data loaded.")

    print(f"\nFitting model (mode={args.mode}) ...")
    if args.mode == "BT":
        model_params, question_params, fit_meta = fit_bt(
            pairwise_df,
            num_epochs=args.num_epochs,
            lr=args.lr,
            lambda_arena=args.lambda_arena,
            reg_lambda=args.reg_lambda,
            verbose=verbose,
        )
    else:
        model_params, question_params, fit_meta = fit_irt(
            static_df if not static_df.empty else None,
            pairwise_df if not pairwise_df.empty else None,
            num_epochs=args.num_epochs,
            lr=args.lr,
            lambda_static=args.lambda_static,
            lambda_arena=args.lambda_arena,
            lambda_bb=args.lambda_bb,
            reg_lambda=args.reg_lambda,
            verbose=verbose,
        )

    metrics = compute_metrics(
        static_df if not static_df.empty else None,
        pairwise_df if not pairwise_df.empty else None,
        model_params,
        question_params,
        learned_gamma=fit_meta["learned_gamma"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_params.to_csv(output_dir / "model_ranking.csv", index=False)
    question_params.to_csv(output_dir / "question_ranking.csv", index=False)
    if args.save_pairwise_targets and not pairwise_df.empty:
        pairwise_df.to_csv(output_dir / "arena_pairwise_targets.csv", index=False)

    run_summary: dict[str, Any] = {
        "config": args.config,
        "mode": args.mode,
        "static_jsonl": args.static_jsonl,
        "arena_reward_jsonl": args.arena_reward_jsonl,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "lambda_static": args.lambda_static,
        "lambda_arena": args.lambda_arena,
        "lambda_bb": args.lambda_bb,
        "reg_lambda": args.reg_lambda,
        "bb_ratio": args.bb_ratio,
        "tie_ratio": args.tie_ratio,
        "save_plot": args.save_plot,
        "metrics": metrics,
        **fit_meta,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("\nModel ranking (highest ability first):")
    print(model_params.to_string(index=False))
    if not question_params.empty:
        print(f"\nQuestion difficulty (top 20 hardest of {len(question_params)}):")
        print(question_params.head(20).to_string(index=False))
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    if fit_meta["learned_gamma"] is not None:
        print(f"  learned_gamma: {fit_meta['learned_gamma']:.4f}")

    if not args.no_plot and args.save_plot:
        title_map = {
            "static": "Static IRT Ranking",
            "arena": "Arena IRT Ranking",
            "both": "Joint IRT Ranking",
            "BT": "Bradley-Terry Ranking",
        }
        plot_difficulty_and_ability(
            model_params, question_params,
            save_path=args.save_plot,
            title=title_map.get(args.mode, "IRT Ranking"),
        )
        print(f"Saved plot to {args.save_plot}")

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
