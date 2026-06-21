#!/usr/bin/env python3
"""
DualEval vs. static/arena baselines — §4.1 Static–Arena Integration results.

By default the script uses question-level holdout.  That is the right split for
arena hard-pair accuracy and rank-transfer metrics.  Static binary accuracy is
only defined when the fitted model has item parameters for the eval questions;
with true question-level static holdout it will therefore be reported as NaN
rather than estimated with leaked eval labels.  Use --split-level row for the
transductive static-binary diagnostic.

The script reports two direct application metrics:
  1. static_binary_accuracy: correctness prediction for held-out (model, question)
     static rows, using methods that have static item parameters.
  2. arena_hard_pair_accuracy: winner prediction for held-out non-tie,
     non-both-bad arena pairs.

It also reports static rank-transfer correlations for arena-only scorers.  That
is the fairer symmetric counterpart to "static-only predicts arena": arena-only
BT/average-reward scorers have no static item difficulty/discrimination, so they
should not be credited or penalized for per-question static binary prediction.

Example:
  python analysis/baseline_comparison.py \
      --config dualeval/config_dualeval.yaml \
      --output-dir results/baseline_comparison/coding_v1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DUALEVAL_PATH = REPO_ROOT / "ranking" / "dualeval.py"
dualeval_spec = importlib.util.spec_from_file_location("dualeval_module", DUALEVAL_PATH)
if dualeval_spec is None or dualeval_spec.loader is None:
    raise ImportError(f"Could not load DualEval module from {DUALEVAL_PATH}")
dualeval = importlib.util.module_from_spec(dualeval_spec)
dualeval_spec.loader.exec_module(dualeval)


DEFAULT_TEST_FRACTION = 0.2


def ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_paths(paths: list[str]) -> list[str]:
    return [str(resolve_path(path)) for path in paths]


def load_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cfg = dualeval.load_yaml_config(args.config)
    input_cfg = cfg.get("input", {})
    training_cfg = cfg.get("training", {})

    if args.static_jsonl is None:
        args.static_jsonl = ensure_list(input_cfg.get("static_jsonl"))
    if args.arena_reward_jsonl is None:
        args.arena_reward_jsonl = ensure_list(input_cfg.get("arena_reward_jsonl"))
    if args.num_epochs is None:
        args.num_epochs = int(training_cfg.get("num_epochs", 2000))
    if args.lr is None:
        args.lr = float(training_cfg.get("lr", 0.02))
    if args.lambda_static is None:
        args.lambda_static = float(training_cfg.get("lambda_static", 1.0))
    if args.lambda_arena is None:
        args.lambda_arena = float(training_cfg.get("lambda_arena", 1.0))
    if args.lambda_bb is None:
        args.lambda_bb = float(training_cfg.get("lambda_bb", 0.2))
    if args.reg_lambda is None:
        args.reg_lambda = float(training_cfg.get("reg_lambda", 1e-3))
    if args.bb_ratio is None:
        args.bb_ratio = float(training_cfg.get("bb_ratio", 0.15))
    if args.tie_ratio is None:
        args.tie_ratio = float(training_cfg.get("tie_ratio", 0.10))

    args.static_jsonl = resolve_paths(args.static_jsonl)
    args.arena_reward_jsonl = resolve_paths(args.arena_reward_jsonl)
    args.output_dir = resolve_path(args.output_dir)
    return args


def split_rows(df: pd.DataFrame, *, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or test_fraction <= 0.0:
        return df.copy(), df.copy()
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"--test-fraction must be in [0, 1), got {test_fraction}")

    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_fraction
    if not mask.any() and len(df) > 1:
        mask[rng.integers(0, len(df))] = True
    if mask.all() and len(df) > 1:
        mask[rng.integers(0, len(df))] = False

    return df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)


def split_questions(
    df: pd.DataFrame,
    *,
    question_col: str,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    if df.empty or test_fraction <= 0.0:
        return df.copy(), df.copy(), set()
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"--test-fraction must be in [0, 1), got {test_fraction}")

    questions = np.array(sorted(df[question_col].astype(str).unique()))
    if len(questions) < 2:
        return df.copy(), df.copy(), set(questions.tolist())

    rng = np.random.default_rng(seed)
    n_eval = max(1, min(len(questions) - 1, int(round(test_fraction * len(questions)))))
    eval_questions = set(rng.choice(questions, size=n_eval, replace=False).tolist())
    eval_mask = df[question_col].astype(str).isin(eval_questions)
    return (
        df.loc[~eval_mask].reset_index(drop=True),
        df.loc[eval_mask].reset_index(drop=True),
        eval_questions,
    )


def subsample_training_data(
    df: pd.DataFrame,
    *,
    fraction: float,
    seed: int,
    split_level: str,
    question_col: str,
) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df.copy(), 0
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Training fractions must be in (0, 1], got {fraction}")
    if fraction >= 1.0:
        n_questions = int(df[question_col].astype(str).nunique()) if question_col in df.columns else 0
        return df.copy().reset_index(drop=True), n_questions

    rng = np.random.default_rng(seed)
    if split_level == "question":
        questions = np.array(sorted(df[question_col].astype(str).unique()))
        n_select = max(1, min(len(questions), int(math.ceil(fraction * len(questions)))))
        selected = set(rng.choice(questions, size=n_select, replace=False).tolist())
        mask = df[question_col].astype(str).isin(selected)
        return df.loc[mask].reset_index(drop=True), int(n_select)

    n_select = max(1, min(len(df), int(math.ceil(fraction * len(df)))))
    selected_idx = rng.choice(df.index.to_numpy(), size=n_select, replace=False)
    subset = df.loc[selected_idx].reset_index(drop=True)
    n_questions = int(subset[question_col].astype(str).nunique()) if question_col in subset.columns else 0
    return subset, n_questions


def static_binary_accuracy(
    static_df: pd.DataFrame,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
) -> tuple[float, int]:
    if static_df.empty or question_params.empty:
        return float("nan"), 0

    theta = model_params.set_index("model_name")["theta"]
    q_params = question_params.set_index("question_id")[["difficulty_b", "discrimination_exp_k"]]
    eval_df = static_df.copy()
    eval_df["theta"] = eval_df["model_name"].map(theta)
    eval_df["difficulty_b"] = eval_df["question_id"].map(q_params["difficulty_b"])
    eval_df["a_q"] = eval_df["question_id"].map(q_params["discrimination_exp_k"])
    eval_df = eval_df.dropna(subset=["theta", "difficulty_b", "a_q", "judge_result"])
    if eval_df.empty:
        return float("nan"), 0

    logits = eval_df["a_q"] * (eval_df["theta"] - eval_df["difficulty_b"])
    pred = (logits >= 0.0).astype(int)
    return float(pred.eq(eval_df["judge_result"].astype(int)).mean()), int(len(eval_df))


def arena_hard_pair_accuracy(
    pairwise_df: pd.DataFrame,
    model_params: pd.DataFrame,
) -> tuple[float, int]:
    if pairwise_df.empty:
        return float("nan"), 0

    theta = model_params.set_index("model_name")["theta"]
    eval_df = pairwise_df[~pairwise_df["tie"].astype(bool) & ~pairwise_df["both_bad"].astype(bool)].copy()
    eval_df["theta_1"] = eval_df["model_1"].map(theta)
    eval_df["theta_2"] = eval_df["model_2"].map(theta)
    eval_df = eval_df.dropna(subset=["theta_1", "theta_2", "target_prob"])
    if eval_df.empty:
        return float("nan"), 0

    pred = eval_df["theta_1"] >= eval_df["theta_2"]
    target = eval_df["target_prob"] >= 0.5
    return float(pred.eq(target).mean()), int(len(eval_df))


def static_rank_transfer(
    static_df: pd.DataFrame,
    model_params: pd.DataFrame,
) -> tuple[float, float, int]:
    """Compare a method's model scores with held-out mean static correctness."""
    if static_df.empty:
        return float("nan"), float("nan"), 0

    ref = static_df.groupby("model_name", as_index=False)["judge_result"].mean()
    scores = model_params[["model_name", "theta"]].merge(ref, on="model_name", how="inner")
    if len(scores) < 2:
        return float("nan"), float("nan"), int(len(scores))

    spearman, _ = spearmanr(scores["theta"], scores["judge_result"])
    kendall, _ = kendalltau(scores["theta"], scores["judge_result"])
    return float(spearman), float(kendall), int(len(scores))


def arena_rank_transfer(
    pairwise_df: pd.DataFrame,
    model_params: pd.DataFrame,
) -> tuple[float, float, int]:
    """Compare a method's model scores with held-out arena soft-win rates."""
    if pairwise_df.empty:
        return float("nan"), float("nan"), 0

    pairs = pairwise_df[~pairwise_df["tie"].astype(bool) & ~pairwise_df["both_bad"].astype(bool)].copy()
    if pairs.empty:
        return float("nan"), float("nan"), 0

    ref = pd.concat(
        [
            pairs[["model_1", "target_prob"]].rename(
                columns={"model_1": "model_name", "target_prob": "arena_soft_win_rate"}
            ),
            pairs[["model_2", "target_prob"]]
            .assign(target_prob=lambda df: 1.0 - df["target_prob"])
            .rename(columns={"model_2": "model_name", "target_prob": "arena_soft_win_rate"}),
        ],
        ignore_index=True,
    )
    ref = ref.groupby("model_name", as_index=False)["arena_soft_win_rate"].mean()
    scores = model_params[["model_name", "theta"]].merge(ref, on="model_name", how="inner")
    if len(scores) < 2:
        return float("nan"), float("nan"), int(len(scores))

    spearman, _ = spearmanr(scores["theta"], scores["arena_soft_win_rate"])
    kendall, _ = kendalltau(scores["theta"], scores["arena_soft_win_rate"])
    return float(spearman), float(kendall), int(len(scores))


def fit_dualeval_both(
    static_train: pd.DataFrame,
    pairwise_train: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return dualeval.fit_irt(
        static_train,
        pairwise_train,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_static=args.lambda_static,
        lambda_arena=args.lambda_arena,
        lambda_bb=args.lambda_bb,
        reg_lambda=args.reg_lambda,
        verbose=not args.quiet,
    )


def fit_static_2pl(
    static_train: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return dualeval.fit_irt(
        static_train,
        None,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_static=args.lambda_static,
        lambda_arena=0.0,
        lambda_bb=0.0,
        reg_lambda=args.reg_lambda,
        verbose=not args.quiet,
    )


def fit_shared_static_2pl_bt(
    static_train: pd.DataFrame,
    pairwise_train: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit shared-theta baseline: static 2PL + direct BT arena loss.

    Static side:
        P(y_iq = 1) = sigmoid(a_q * (theta_i - b_q))

    Arena side:
        P(i > j) = sigmoid(theta_i - theta_j)

    This baseline shares model ability across static and arena data, but unlike
    DualEval it does not route arena comparisons through item response
    probabilities and has no both-bad anchoring term.
    """
    static = static_train.copy()
    pairwise = pairwise_train.copy()
    non_tie = pairwise[~pairwise["tie"].astype(bool)].copy()
    if static.empty:
        raise SystemExit("Shared 2PL+BT baseline needs non-empty static training data.")
    if non_tie.empty:
        raise SystemExit("Shared 2PL+BT baseline needs non-tie arena training pairs.")

    all_models = pd.Index(
        pd.unique(pd.concat([static["model_name"], non_tie["model_1"], non_tie["model_2"]], ignore_index=True)),
        name="model_name",
    )
    all_questions = pd.Index(pd.unique(static["question_id"]), name="question_id")
    model_to_idx = {model: idx for idx, model in enumerate(all_models)}
    q_to_idx = {question: idx for idx, question in enumerate(all_questions)}

    static["m_idx"] = static["model_name"].map(model_to_idx)
    static["q_idx"] = static["question_id"].map(q_to_idx)
    non_tie["m1_idx"] = non_tie["model_1"].map(model_to_idx)
    non_tie["m2_idx"] = non_tie["model_2"].map(model_to_idx)

    device = dualeval._get_device()
    m_s = torch.tensor(static["m_idx"].values, dtype=torch.long, device=device)
    q_s = torch.tensor(static["q_idx"].values, dtype=torch.long, device=device)
    y_s = torch.tensor(static["judge_result"].values, dtype=torch.float32, device=device)
    m1_t = torch.tensor(non_tie["m1_idx"].values, dtype=torch.long, device=device)
    m2_t = torch.tensor(non_tie["m2_idx"].values, dtype=torch.long, device=device)
    soft_t = torch.tensor(non_tie["target_prob"].values, dtype=torch.float32, device=device)

    theta = nn.Embedding(len(all_models), 1, device=device)
    b = nn.Embedding(len(all_questions), 1, device=device)
    k = nn.Embedding(len(all_questions), 1, device=device)
    nn.init.zeros_(theta.weight)
    nn.init.zeros_(b.weight)
    nn.init.zeros_(k.weight)

    optimizer = optim.Adam([*theta.parameters(), *b.parameters(), *k.parameters()], lr=args.lr)
    bce_logits = nn.BCEWithLogitsLoss()

    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        logits_static = torch.exp(k(q_s).squeeze(-1)) * (theta(m_s).squeeze(-1) - b(q_s).squeeze(-1))
        loss_static = bce_logits(logits_static, y_s)

        logits_bt = theta(m1_t).squeeze(-1) - theta(m2_t).squeeze(-1)
        loss_arena = bce_logits(logits_bt, soft_t)

        reg = args.reg_lambda * (theta.weight.pow(2).mean() + b.weight.pow(2).mean() + k.weight.pow(2).mean())
        loss = args.lambda_static * loss_static + args.lambda_arena * loss_arena + reg
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            shift = theta.weight.mean()
            theta.weight.sub_(shift)
            b.weight.sub_(shift)

        if not args.quiet and (epoch % 500 == 0 or epoch == args.num_epochs - 1):
            print(
                f"  Epoch {epoch:5d} | "
                f"static={loss_static.item():.4f}  "
                f"bt={loss_arena.item():.4f}  "
                f"total={loss.item():.4f}"
            )

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    b_np = b.weight.detach().cpu().numpy().squeeze(-1)
    k_np = k.weight.detach().cpu().numpy().squeeze(-1)

    model_params = (
        pd.DataFrame({"model_name": all_models, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )

    question_meta = (
        static[["question_id", "source", "benchmark"]]
        .drop_duplicates("question_id")
        .set_index("question_id")
        .to_dict(orient="index")
    )
    question_rows: list[dict[str, Any]] = []
    for idx, question_id in enumerate(all_questions):
        meta = question_meta.get(str(question_id), {})
        question_rows.append(
            {
                "question_id": question_id,
                "source": meta.get("source", "unknown"),
                "benchmark": meta.get("benchmark", "unknown"),
                "difficulty_b": float(b_np[idx]),
                "k_raw": float(k_np[idx]),
                "discrimination_exp_k": math.exp(float(k_np[idx])),
            }
        )
    question_params = (
        pd.DataFrame(question_rows)
        .sort_values("difficulty_b", ascending=False)
        .reset_index(drop=True)
    )
    metadata = {
        "learned_gamma": None,
        "n_models": int(len(all_models)),
        "n_questions": int(len(all_questions)),
        "has_static": True,
        "has_arena": True,
    }
    return model_params, question_params, metadata


def fit_arena_bt(
    pairwise_train: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return dualeval.fit_bt(
        pairwise_train,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_arena=args.lambda_arena,
        reg_lambda=args.reg_lambda,
        verbose=not args.quiet,
    )


def fit_arena_avg_reward(pairwise_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if pairwise_train.empty:
        raise SystemExit("No arena pairwise training rows available for average-reward baseline.")

    long_rewards = pd.concat(
        [
            pairwise_train[["model_1", "reward_z_1"]].rename(
                columns={"model_1": "model_name", "reward_z_1": "reward_z"}
            ),
            pairwise_train[["model_2", "reward_z_2"]].rename(
                columns={"model_2": "model_name", "reward_z_2": "reward_z"}
            ),
        ],
        ignore_index=True,
    )
    model_params = (
        long_rewards.groupby("model_name", as_index=False)["reward_z"]
        .mean()
        .rename(columns={"reward_z": "theta"})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )
    metadata = {
        "learned_gamma": None,
        "n_models": int(len(model_params)),
        "n_questions": 0,
        "has_static": False,
        "has_arena": True,
    }
    return model_params, pd.DataFrame(), metadata


def ref_rank_agreement(
    method_model: pd.DataFrame | None,
    ref_model: pd.DataFrame | None,
) -> tuple[float, float, int]:
    """Spearman/Kendall between a method's theta and a reference theta on shared models."""
    if method_model is None or ref_model is None or method_model.empty or ref_model.empty:
        return float("nan"), float("nan"), 0
    merged = method_model[["model_name", "theta"]].merge(
        ref_model[["model_name", "theta"]].rename(columns={"theta": "ref_theta"}),
        on="model_name",
        how="inner",
    )
    if len(merged) < 2:
        return float("nan"), float("nan"), int(len(merged))
    spearman, _ = spearmanr(merged["theta"], merged["ref_theta"])
    kendall, _ = kendalltau(merged["theta"], merged["ref_theta"])
    return float(spearman), float(kendall), int(len(merged))


def metrics_row(
    *,
    method: str,
    static_eval: pd.DataFrame,
    pairwise_eval: pd.DataFrame,
    static_model: pd.DataFrame | None,
    static_questions: pd.DataFrame | None,
    arena_model: pd.DataFrame | None,
    full_data_model: pd.DataFrame | None = None,
    static_ref_model: pd.DataFrame | None = None,
    arena_ref_model: pd.DataFrame | None = None,
    notes: str,
) -> dict[str, Any]:
    if static_model is not None and static_questions is not None:
        static_acc, n_static = static_binary_accuracy(static_eval, static_model, static_questions)
    else:
        static_acc, n_static = float("nan"), 0

    if arena_model is not None:
        arena_acc, n_arena = arena_hard_pair_accuracy(pairwise_eval, arena_model)
        transfer_spearman, transfer_kendall, n_transfer = static_rank_transfer(static_eval, arena_model)
        arena_spearman, arena_kendall, n_arena_rank = arena_rank_transfer(pairwise_eval, arena_model)
    else:
        arena_acc, n_arena = float("nan"), 0
        transfer_spearman, transfer_kendall, n_transfer = float("nan"), float("nan"), 0
        arena_spearman, arena_kendall, n_arena_rank = float("nan"), float("nan"), 0

    # Ref ρ uses the full-data fit of this method (Option C): cleaner semantics — no
    # split-induced noise. Falls back to the split fit if no full-data fit was supplied.
    if full_data_model is not None:
        ref_method_model = full_data_model
    else:
        ref_method_model = arena_model if arena_model is not None else static_model
    static_ref_spearman, static_ref_kendall, n_static_ref = ref_rank_agreement(ref_method_model, static_ref_model)
    arena_ref_spearman, arena_ref_kendall, n_arena_ref = ref_rank_agreement(ref_method_model, arena_ref_model)

    return {
        "method": method,
        "static_binary_accuracy": static_acc,
        "n_static_eval_rows": n_static,
        "arena_hard_pair_accuracy": arena_acc,
        "n_arena_hard_eval_pairs": n_arena,
        "arena_to_static_rank_spearman": transfer_spearman,
        "arena_to_static_rank_kendall": transfer_kendall,
        "n_static_rank_models": n_transfer,
        "static_to_arena_rank_spearman": arena_spearman,
        "static_to_arena_rank_kendall": arena_kendall,
        "n_arena_rank_models": n_arena_rank,
        "static_ref_spearman": static_ref_spearman,
        "static_ref_kendall": static_ref_kendall,
        "n_static_ref_models": n_static_ref,
        "arena_ref_spearman": arena_ref_spearman,
        "arena_ref_kendall": arena_ref_kendall,
        "n_arena_ref_models": n_arena_ref,
        "notes": notes,
    }


def run_baseline_comparison(
    *,
    static_train: pd.DataFrame,
    static_eval: pd.DataFrame,
    pairwise_train: pd.DataFrame,
    pairwise_eval: pd.DataFrame,
    args: argparse.Namespace,
    seed: int,
    train_fraction: float,
    n_static_train_questions: int,
    n_arena_train_questions: int,
    static_ref_model: pd.DataFrame | None = None,
    arena_ref_model: pd.DataFrame | None = None,
    full_data_models: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    both_models, both_questions, both_meta = fit_dualeval_both(static_train, pairwise_train, args)
    static_models, static_questions, static_meta = fit_static_2pl(static_train, args)
    bt_models, _bt_questions, bt_meta = fit_arena_bt(pairwise_train, args)
    avg_reward_models, _avg_reward_questions, avg_reward_meta = fit_arena_avg_reward(pairwise_train)

    full_data_models = full_data_models or {}
    rows = [
        metrics_row(
            method="dualeval_both",
            static_eval=static_eval,
            pairwise_eval=pairwise_eval,
            static_model=both_models,
            static_questions=both_questions,
            arena_model=both_models,
            full_data_model=full_data_models.get("dualeval_both"),
            static_ref_model=static_ref_model,
            arena_ref_model=arena_ref_model,
            notes="Shared 2PL latent ability trained on static labels and arena pairwise targets.",
        ),
        metrics_row(
            method="static_2pl_only",
            static_eval=static_eval,
            pairwise_eval=pairwise_eval,
            static_model=static_models,
            static_questions=static_questions,
            arena_model=static_models,
            full_data_model=full_data_models.get("static_2pl_only"),
            static_ref_model=static_ref_model,
            arena_ref_model=arena_ref_model,
            notes="Static 2PL baseline; arena winners are predicted by static theta ordering.",
        ),
        metrics_row(
            method="arena_bt_only",
            static_eval=static_eval,
            pairwise_eval=pairwise_eval,
            static_model=None,
            static_questions=None,
            arena_model=bt_models,
            full_data_model=full_data_models.get("arena_bt_only"),
            static_ref_model=static_ref_model,
            arena_ref_model=arena_ref_model,
            notes="Arena-only BT; static binary accuracy is undefined, rank transfer is reported instead.",
        ),
        metrics_row(
            method="arena_avg_reward_only",
            static_eval=static_eval,
            pairwise_eval=pairwise_eval,
            static_model=None,
            static_questions=None,
            arena_model=avg_reward_models,
            full_data_model=full_data_models.get("arena_avg_reward_only"),
            static_ref_model=static_ref_model,
            arena_ref_model=arena_ref_model,
            notes="Arena-only mean reward-z ranking; static binary accuracy is undefined, rank transfer is reported instead.",
        ),
    ]
    metrics = pd.DataFrame(rows)
    metrics.insert(0, "seed", int(seed))
    metrics.insert(1, "train_fraction", float(train_fraction))
    metrics["n_static_train_rows"] = int(len(static_train))
    metrics["n_arena_train_pairs"] = int(len(pairwise_train))
    metrics["n_static_train_questions"] = int(n_static_train_questions)
    metrics["n_arena_train_questions"] = int(n_arena_train_questions)

    rankings = {
        "dualeval_both": both_models,
        "static_2pl": static_models,
        "arena_bt": bt_models,
        "arena_avg_reward": avg_reward_models,
    }
    metadata = {
        "dualeval_both": both_meta,
        "static_2pl_only": static_meta,
        "arena_bt": bt_meta,
        "arena_avg_reward": avg_reward_meta,
    }
    return metrics, rankings, metadata


def summarise_sample_efficiency(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "static_binary_accuracy",
        "arena_hard_pair_accuracy",
        "arena_to_static_rank_spearman",
        "arena_to_static_rank_kendall",
        "static_to_arena_rank_spearman",
        "static_to_arena_rank_kendall",
        "static_ref_spearman",
        "static_ref_kendall",
        "arena_ref_spearman",
        "arena_ref_kendall",
    ]
    summary = (
        metrics.groupby(["method", "train_fraction"], as_index=False)
        .agg(
            **{
                f"{col}_mean": (col, "mean")
                for col in metric_cols
            },
            **{
                f"{col}_std": (col, "std")
                for col in metric_cols
            },
            n_trials=("seed", "nunique"),
            n_static_train_rows_mean=("n_static_train_rows", "mean"),
            n_arena_train_pairs_mean=("n_arena_train_pairs", "mean"),
            n_static_train_questions_mean=("n_static_train_questions", "mean"),
            n_arena_train_questions_mean=("n_arena_train_questions", "mean"),
        )
        .sort_values(["method", "train_fraction"])
        .reset_index(drop=True)
    )
    return summary


def compute_auc(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "arena_hard_pair_accuracy_mean",
        "arena_to_static_rank_spearman_mean",
        "arena_to_static_rank_kendall_mean",
        "static_to_arena_rank_spearman_mean",
        "static_to_arena_rank_kendall_mean",
    ]
    for method, part in summary.groupby("method", sort=False):
        part = part.sort_values("train_fraction")
        x = part["train_fraction"].to_numpy(dtype=float)
        row: dict[str, Any] = {"method": method}
        for col in metric_cols:
            valid = part[col].notna().to_numpy()
            row[f"{col.removesuffix('_mean')}_auc"] = (
                float(np.trapz(part.loc[valid, col].to_numpy(dtype=float), x[valid]))
                if valid.sum() >= 2
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sample_efficiency(summary: pd.DataFrame, save_path: Path) -> None:
    metrics = [
        ("arena_hard_pair_accuracy_mean", "arena_hard_pair_accuracy_std", "Arena hard-pair accuracy"),
        (
            "arena_to_static_rank_spearman_mean",
            "arena_to_static_rank_spearman_std",
            "Arena-to-static Spearman",
        ),
        (
            "arena_to_static_rank_kendall_mean",
            "arena_to_static_rank_kendall_std",
            "Arena-to-static Kendall",
        ),
        (
            "static_to_arena_rank_spearman_mean",
            "static_to_arena_rank_spearman_std",
            "Static-to-arena Spearman",
        ),
        (
            "static_to_arena_rank_kendall_mean",
            "static_to_arena_rank_kendall_std",
            "Static-to-arena Kendall",
        ),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(24, 4.2), sharex=True)
    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        for method, part in summary.groupby("method", sort=False):
            part = part.sort_values("train_fraction")
            x = part["train_fraction"].to_numpy(dtype=float)
            y = part[mean_col].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.7, label=method)
            if std_col in part.columns and part[std_col].notna().any():
                std = part[std_col].fillna(0.0).to_numpy(dtype=float)
                ax.fill_between(x, y - std, y + std, alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel("Fraction of training questions used")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, frameon=True)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ranking/config_dualeval.yaml")
    parser.add_argument("--static-jsonl", nargs="*", default=None)
    parser.add_argument("--arena-reward-jsonl", nargs="*", default=None)
    parser.add_argument("--output-dir", default="results/dualeval_baseline_comparison")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-static", type=float, default=None)
    parser.add_argument("--lambda-arena", type=float, default=None)
    parser.add_argument("--lambda-bb", type=float, default=None)
    parser.add_argument("--reg-lambda", type=float, default=None)
    parser.add_argument("--bb-ratio", type=float, default=None)
    parser.add_argument("--tie-ratio", type=float, default=None)
    parser.add_argument(
        "--split-level",
        choices=["question", "row"],
        default="question",
        help="Hold out whole question IDs by default; row split keeps static binary accuracy transductive.",
    )
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-fractions",
        nargs="+",
        type=float,
        default=None,
        help="Run sample-efficiency mode with these fractions of the training pool.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds for repeated sample-efficiency runs. Defaults to --seed.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = load_config_defaults(parse_args())
    if not args.static_jsonl:
        raise SystemExit("Need static input. Provide --static-jsonl or config input.static_jsonl.")
    if not args.arena_reward_jsonl:
        raise SystemExit("Need arena input. Provide --arena-reward-jsonl or config input.arena_reward_jsonl.")

    static_df = dualeval.load_static_jsonl(args.static_jsonl)
    reward_df = dualeval.load_arena_reward_jsonl(args.arena_reward_jsonl)
    if static_df.empty:
        raise SystemExit("No usable static rows loaded.")
    if reward_df.empty:
        raise SystemExit("No usable arena reward rows loaded.")

    bb_threshold, tie_delta = dualeval.resolve_pairwise_thresholds(
        reward_df,
        bb_ratio=args.bb_ratio,
        tie_ratio=args.tie_ratio,
    )
    pairwise_df = dualeval.build_soft_pairwise_targets(
        reward_df,
        both_bad_threshold=bb_threshold,
        tie_delta=tie_delta,
    )
    if pairwise_df.empty:
        raise SystemExit("No usable arena pairs built.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_fractions = args.train_fractions if args.train_fractions is not None else [1.0]
    train_fractions = sorted({float(fraction) for fraction in train_fractions})
    seeds = args.seeds if args.seeds is not None else [args.seed]

    if not args.quiet:
        print("\nFitting full-data static-only IRT reference (lambda_arena=0)...")
    static_ref_model, _static_ref_questions, _static_ref_meta = dualeval.fit_irt(
        static_df,
        None,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_static=args.lambda_static,
        lambda_arena=0.0,
        lambda_bb=0.0,
        reg_lambda=args.reg_lambda,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print("Fitting full-data arena BT reference...")
    arena_ref_model, _arena_ref_questions, _arena_ref_meta = dualeval.fit_bt(
        pairwise_df,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_arena=args.lambda_arena,
        reg_lambda=args.reg_lambda,
        verbose=not args.quiet,
    )
    static_ref_model.to_csv(args.output_dir / "static_ref_full_data_ranking.csv", index=False)
    arena_ref_model.to_csv(args.output_dir / "arena_ref_full_data_ranking.csv", index=False)

    # Full-data fits per method (Option C): used for Ref ρ. References double as the
    # full-data fits for static_2pl_only and arena_bt_only.
    if not args.quiet:
        print("Fitting full-data DualEval (both)...")
    dualeval_full_models, _dualeval_full_q, _dualeval_full_meta = fit_dualeval_both(static_df, pairwise_df, args)
    if not args.quiet:
        print("Fitting full-data arena average-reward baseline...")
    avg_reward_full_models, _avg_reward_full_q, _avg_reward_full_meta = fit_arena_avg_reward(pairwise_df)

    full_data_models = {
        "dualeval_both": dualeval_full_models,
        "static_2pl_only": static_ref_model,
        "arena_bt_only": arena_ref_model,
        "arena_avg_reward_only": avg_reward_full_models,
    }
    dualeval_full_models.to_csv(args.output_dir / "dualeval_both_full_data_ranking.csv", index=False)
    avg_reward_full_models.to_csv(args.output_dir / "arena_avg_reward_full_data_ranking.csv", index=False)

    all_metrics: list[pd.DataFrame] = []
    split_summaries: list[dict[str, Any]] = []
    last_rankings: dict[str, pd.DataFrame] = {}
    last_fit_metadata: dict[str, Any] = {}

    if not args.quiet:
        print(
            f"Pair thresholds: bb_ratio={args.bb_ratio} -> {bb_threshold:.4f}, "
            f"tie_ratio={args.tie_ratio} -> {tie_delta:.4f}"
        )
        if args.split_level == "question":
            print(
                "Note: static_binary_accuracy is undefined for true static question holdout "
                "because held-out questions have no fitted IRT item parameters."
            )

    for seed in seeds:
        if args.split_level == "question":
            static_train_pool, static_eval, static_eval_questions = split_questions(
                static_df,
                question_col="question_id",
                test_fraction=args.test_fraction,
                seed=seed,
            )
            pairwise_train_pool, pairwise_eval, arena_eval_questions = split_questions(
                pairwise_df,
                question_col="question_id",
                test_fraction=args.test_fraction,
                seed=seed + 1,
            )
        else:
            static_train_pool, static_eval = split_rows(static_df, test_fraction=args.test_fraction, seed=seed)
            pairwise_train_pool, pairwise_eval = split_rows(
                pairwise_df,
                test_fraction=args.test_fraction,
                seed=seed + 1,
            )
            static_eval_questions = set(static_eval["question_id"].astype(str).unique())
            arena_eval_questions = set(pairwise_eval["question_id"].astype(str).unique())

        split_summaries.append(
            {
                "seed": int(seed),
                "n_static_train_pool": int(len(static_train_pool)),
                "n_static_eval": int(len(static_eval)),
                "n_static_eval_questions": int(len(static_eval_questions)),
                "n_pairwise_train_pool": int(len(pairwise_train_pool)),
                "n_pairwise_eval": int(len(pairwise_eval)),
                "n_arena_eval_questions": int(len(arena_eval_questions)),
            }
        )
        if not args.quiet:
            print(
                f"\nSeed {seed}: static train-pool={len(static_train_pool)} eval={len(static_eval)} | "
                f"arena train-pool={len(pairwise_train_pool)} eval={len(pairwise_eval)} | "
                f"split={args.split_level}"
            )

        for fraction in train_fractions:
            static_train, n_static_train_questions = subsample_training_data(
                static_train_pool,
                fraction=fraction,
                seed=seed * 1009 + int(round(fraction * 10000)) + 17,
                split_level=args.split_level,
                question_col="question_id",
            )
            pairwise_train, n_arena_train_questions = subsample_training_data(
                pairwise_train_pool,
                fraction=fraction,
                seed=seed * 1009 + int(round(fraction * 10000)) + 29,
                split_level=args.split_level,
                question_col="question_id",
            )
            if not args.quiet:
                print(
                    f"  Train fraction {fraction:.3f}: "
                    f"static rows={len(static_train)} q={n_static_train_questions}; "
                    f"arena pairs={len(pairwise_train)} q={n_arena_train_questions}"
                )

            metrics_part, rankings, fit_metadata = run_baseline_comparison(
                static_train=static_train,
                static_eval=static_eval,
                pairwise_train=pairwise_train,
                pairwise_eval=pairwise_eval,
                args=args,
                seed=seed,
                train_fraction=fraction,
                n_static_train_questions=n_static_train_questions,
                n_arena_train_questions=n_arena_train_questions,
                static_ref_model=static_ref_model,
                arena_ref_model=arena_ref_model,
                full_data_models=full_data_models,
            )
            all_metrics.append(metrics_part)
            if seed == seeds[-1] and fraction == max(train_fractions):
                last_rankings = rankings
                last_fit_metadata = fit_metadata

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics_path = args.output_dir / "baseline_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    if last_rankings:
        last_rankings["dualeval_both"].to_csv(args.output_dir / "dualeval_both_model_ranking.csv", index=False)
        last_rankings["static_2pl"].to_csv(args.output_dir / "static_2pl_model_ranking.csv", index=False)
        last_rankings["arena_bt"].to_csv(args.output_dir / "arena_bt_model_ranking.csv", index=False)
        last_rankings["arena_avg_reward"].to_csv(args.output_dir / "arena_avg_reward_model_ranking.csv", index=False)

    if len(train_fractions) > 1 or len(seeds) > 1:
        efficiency_summary = summarise_sample_efficiency(metrics)
        efficiency_summary.to_csv(args.output_dir / "sample_efficiency_summary.csv", index=False)
        compute_auc(efficiency_summary).to_csv(args.output_dir / "sample_efficiency_auc.csv", index=False)
        plot_sample_efficiency(efficiency_summary, args.output_dir / "sample_efficiency.pdf")

    summary = {
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
        "both_bad_threshold": bb_threshold,
        "tie_delta": tie_delta,
        "split_level": args.split_level,
        "test_fraction": args.test_fraction,
        "seeds": seeds,
        "train_fractions": train_fractions,
        "split_summaries": split_summaries,
        "fit_metadata_last_max_fraction": last_fit_metadata,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False, float_format=lambda x: "nan" if math.isnan(x) else f"{x:.4f}"))
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
