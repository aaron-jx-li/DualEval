#!/usr/bin/env python3
"""
Item informativeness: learned discrimination enables efficient ranking recovery (§4.2).

For each saved dualeval run, this script:
  1. Plots item difficulty vs. learned sharpness and annotates correlations.
  2. Selects the top x% most discriminative questions using a configurable
     item-utility criterion, re-estimates model abilities from only those
     questions while keeping full-run item parameters fixed, and compares the
     recovered ranking to the full-question ranking.
  3. Adds random-question baselines using both the fixed-item DualEval recovery
     and a direct arena-only BT fit on the same random x% question subsets.

Usage examples:
  python analysis/item_informativeness.py \
      --run-dirs results/dualeval/coding_v1/arena \
      --benchmarks Arena \
      --fractions 0.05 0.10 0.20 \
      --selection-criterion effective \
      --output-dir results/dualeval_baselines/coding_v1_arena_efficiency

  python analysis/item_informativeness.py \
      --runs-root results/dualeval \
      --output-dir results/dualeval_baselines
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau, pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRACTIONS = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00]
DEFAULT_RANDOM_SEEDS = [0, 1, 2]
SELECTION_CRITERIA = {"raw", "effective", "fisher", "mean"}
FISHER_MODES = {"objective", "simple"}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def find_run_dirs(runs_root: Path) -> list[Path]:
    runs_root = resolve_path(runs_root)
    if not runs_root.exists():
        raise SystemExit(f"Runs root not found: {runs_root}")
    return sorted(
        path
        for path in runs_root.rglob("*")
        if path.is_dir()
        and (path / "model_ranking.csv").exists()
        and (path / "question_ranking.csv").exists()
        and (path / "run_summary.json").exists()
    )


def source_tag_for_path(path: Path) -> str:
    parent_name = path.parent.name.strip()
    return parent_name if parent_name else path.stem


def load_static_jsonl(jsonl_paths: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for jsonl_path in jsonl_paths:
        path = resolve_path(jsonl_path)
        source_tag = source_tag_for_path(path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("model_label") is None:
                    continue
                status = str(record.get("status", "")).strip().lower()
                is_errored = status != "ok"
                if not is_errored and record.get("correct") is None:
                    continue
                dataset = str(record.get("dataset", "unknown"))
                sample_index = record.get("sample_index")
                question_id = f"{source_tag}::{dataset}_{sample_index}"
                rows.append(
                    {
                        "source": source_tag,
                        "benchmark": dataset,
                        "model_name": str(record["model_label"]),
                        "question_id": question_id,
                        "judge_result": 0 if is_errored else int(bool(record["correct"])),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["source", "benchmark", "model_name", "question_id", "judge_result"])
    return pd.DataFrame(rows).drop_duplicates(["model_name", "question_id"], keep="last").reset_index(drop=True)


def load_arena_reward_jsonl(jsonl_paths: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for jsonl_path in jsonl_paths:
        path = resolve_path(jsonl_path)
        source_tag = source_tag_for_path(path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") != "ok" or record.get("reward") is None:
                    continue
                rows.append(
                    {
                        "source": source_tag,
                        "benchmark": "Arena",
                        "model_name": str(record["model_label"]),
                        "question_id": f"{source_tag}::{record['item_id']}",
                        "reward_raw": float(record["reward"]),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["source", "benchmark", "model_name", "question_id", "reward_raw", "reward_z"])
    df = pd.DataFrame(rows).drop_duplicates(["model_name", "question_id"], keep="last").reset_index(drop=True)
    reward_std = float(df["reward_raw"].std(ddof=0))
    if not math.isfinite(reward_std) or reward_std < 1e-8:
        reward_std = 1.0
    df["reward_z"] = (df["reward_raw"] - float(df["reward_raw"].mean())) / reward_std
    return df


def resolve_pairwise_thresholds(reward_df: pd.DataFrame, *, bb_ratio: float, tie_ratio: float) -> tuple[float, float]:
    max_scores: list[float] = []
    abs_diffs: list[float] = []
    for _, group in reward_df.groupby("question_id", sort=False):
        if len(group) < 2:
            continue
        zs = group["reward_z"].to_numpy()
        for idx1 in range(len(zs)):
            for idx2 in range(idx1 + 1, len(zs)):
                max_scores.append(float(max(zs[idx1], zs[idx2])))
                abs_diffs.append(float(abs(zs[idx1] - zs[idx2])))

    if not max_scores:
        return -math.inf, 0.0
    both_bad_threshold = float(np.percentile(np.array(max_scores), bb_ratio * 100))
    tie_delta = float(np.percentile(np.array(abs_diffs), tie_ratio * 100)) if tie_ratio > 0.0 else 0.0
    return both_bad_threshold, tie_delta


def build_soft_pairwise_targets(
    reward_df: pd.DataFrame,
    *,
    both_bad_threshold: float,
    tie_delta: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for question_id, group in reward_df.groupby("question_id", sort=False):
        if len(group) < 2:
            continue
        group = group.sort_values("model_name").reset_index(drop=True)
        for idx1, idx2 in combinations(range(len(group)), 2):
            r1, r2 = group.iloc[idx1], group.iloc[idx2]
            z1, z2 = float(r1["reward_z"]), float(r2["reward_z"])
            rows.append(
                {
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
                }
            )
    return pd.DataFrame(rows)


def rank_series(model_params: pd.DataFrame) -> pd.Series:
    ordered = model_params.sort_values("theta", ascending=False)["model_name"].tolist()
    return pd.Series({model: rank for rank, model in enumerate(ordered, start=1)}, name="rank")


def ranking_metrics(ref_params: pd.DataFrame, sample_params: pd.DataFrame, *, top_k: int) -> dict[str, float]:
    common = sorted(set(ref_params["model_name"]) & set(sample_params["model_name"]))
    if len(common) < 2:
        return {
            "spearman": float("nan"),
            "kendall": float("nan"),
            "exact_matches": float("nan"),
            "top_k_retention": float("nan"),
            "n_models": float(len(common)),
        }

    ref_rank = rank_series(ref_params[ref_params["model_name"].isin(common)])
    sample_rank = rank_series(sample_params[sample_params["model_name"].isin(common)])
    ref_values = [ref_rank[model] for model in common]
    sample_values = [sample_rank[model] for model in common]
    rho, _ = spearmanr(ref_values, sample_values)
    tau, _ = kendalltau(ref_values, sample_values)
    ref_top = {model for model, rank in ref_rank.items() if rank <= top_k}
    sample_top = {model for model, rank in sample_rank.items() if rank <= top_k}
    return {
        "spearman": float(rho),
        "kendall": float(tau),
        "exact_matches": float(np.mean([a == b for a, b in zip(ref_values, sample_values)])),
        "top_k_retention": float(len(ref_top & sample_top) / len(ref_top)) if ref_top else float("nan"),
        "n_models": float(len(common)),
    }


_CONTRAST_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#A65628", "#F781BF", "#666666",
    "#66C2A5", "#FC8D62",
]


_CRITERION_YLABEL: dict[str, str] = {
    "raw":       r"Sharpness $a_q$",
    "effective": r"Eff. disc. $a_q \cdot p_q(1-p_q)$",
    "fisher":    r"Fisher info about model ability",
    "mean":      r"Mean Fisher $\mathbb{E}_m[a_q p_{mq}(1\!-\!p_{mq})]$",
}


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -709.0, 709.0)))


def _simple_selection_scores(
    qp: pd.DataFrame,
    criterion: str,
    model_params: pd.DataFrame | None,
) -> np.ndarray:
    a = qp["sharpness_exp_k"].astype(float).to_numpy()
    b = qp["difficulty_b"].astype(float).to_numpy()
    if criterion == "raw" or model_params is None or model_params.empty:
        return a
    if criterion in {"effective", "fisher"}:
        theta_ref = float(model_params["theta"].median())
        p = _sigmoid_np(a * (theta_ref - b))
        return (a ** 2) * p * (1.0 - p) if criterion == "fisher" else a * p * (1.0 - p)
    # mean
    theta_vals = model_params["theta"].astype(float).to_numpy()
    logits = np.outer(theta_vals, a) - (a * b)[None, :]
    p = _sigmoid_np(logits)
    return (a[None, :] * p * (1.0 - p)).mean(axis=0)


def _hard_pairwise_for_fisher(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    pairwise = pairwise_df.copy()
    if {"tie", "both_bad"}.issubset(pairwise.columns):
        pairwise = pairwise[
            ~pairwise["tie"].astype(bool)
            & ~pairwise["both_bad"].astype(bool)
        ].copy()
    elif "tie" in pairwise.columns:
        pairwise = pairwise[~pairwise["tie"].astype(bool)].copy()
    return pairwise


def _objective_fisher_scores(
    qp: pd.DataFrame,
    model_params: pd.DataFrame,
    pairwise_df: pd.DataFrame | None,
    run_summary: dict[str, Any] | None,
) -> np.ndarray:
    a = qp["sharpness_exp_k"].astype(float).to_numpy()
    b = qp["difficulty_b"].astype(float).to_numpy()
    theta_vals = model_params["theta"].astype(float).to_numpy()
    lambda_static = float((run_summary or {}).get("lambda_static", 1.0))
    lambda_arena = float((run_summary or {}).get("lambda_arena", 1.0))
    gamma = float((run_summary or {}).get("learned_gamma") or 1.0)

    logits = np.outer(theta_vals, a) - (a * b)[None, :]
    p = _sigmoid_np(logits)
    scores = lambda_static * ((a[None, :] ** 2) * p * (1.0 - p)).mean(axis=0)

    if pairwise_df is None or pairwise_df.empty:
        return scores

    model_theta = dict(zip(model_params["model_name"].astype(str), theta_vals, strict=False))
    q_pos = {str(qid): idx for idx, qid in enumerate(qp["question_id"].astype(str))}
    arena_pairwise = _hard_pairwise_for_fisher(pairwise_df)
    if arena_pairwise.empty:
        return scores

    arena_sums: dict[str, float] = {}
    arena_counts: dict[str, int] = {}
    for row in arena_pairwise.itertuples(index=False):
        qid = str(getattr(row, "question_id"))
        pos = q_pos.get(qid)
        if pos is None:
            continue
        theta_1 = model_theta.get(str(getattr(row, "model_1")))
        theta_2 = model_theta.get(str(getattr(row, "model_2")))
        if theta_1 is None or theta_2 is None:
            continue

        aq = float(a[pos])
        bq = float(b[pos])
        p1 = float(_sigmoid_np(np.array([aq * (theta_1 - bq)]))[0])
        p2 = float(_sigmoid_np(np.array([aq * (theta_2 - bq)]))[0])
        mu = float(_sigmoid_np(np.array([gamma * (p1 - p2)]))[0])
        grad_sq_trace = (aq * p1 * (1.0 - p1)) ** 2 + (aq * p2 * (1.0 - p2)) ** 2
        contribution = mu * (1.0 - mu) * (gamma ** 2) * grad_sq_trace
        arena_sums[qid] = arena_sums.get(qid, 0.0) + contribution
        arena_counts[qid] = arena_counts.get(qid, 0) + 1

    for qid, total in arena_sums.items():
        scores[q_pos[qid]] = lambda_arena * total / arena_counts[qid]
    return scores


def _compute_selection_scores(
    qp: pd.DataFrame,
    criterion: str,
    model_params: pd.DataFrame | None,
    pairwise_df: pd.DataFrame | None = None,
    run_summary: dict[str, Any] | None = None,
    fisher_mode: str = "objective",
) -> np.ndarray:
    if criterion == "fisher" and fisher_mode == "objective" and model_params is not None and not model_params.empty:
        return _objective_fisher_scores(qp, model_params, pairwise_df, run_summary)
    return _simple_selection_scores(qp, criterion, model_params)


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#f8f8f8")
    ax.grid(color="white", linewidth=1.0, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")


def _scatter_by_benchmark(ax: plt.Axes, qp: pd.DataFrame, y_col: str) -> list[str]:
    benchmarks = sorted(qp["benchmark"].fillna("unknown").unique())
    for idx, benchmark in enumerate(benchmarks):
        color = _CONTRAST_COLORS[idx % len(_CONTRAST_COLORS)]
        part = qp[qp["benchmark"].fillna("unknown") == benchmark]
        ax.scatter(
            part["difficulty_b"],
            part[y_col],
            s=28, alpha=0.75, color=color,
            label=str(benchmark),
            edgecolors="white", linewidths=0.4, zorder=3,
        )
    return benchmarks


def _annotate(ax: plt.Axes, x: np.ndarray, y: np.ndarray, n: int) -> None:
    if len(x) > 1:
        pearson, _ = pearsonr(x, y)
        spearman, _ = spearmanr(x, y)
        ax.text(
            0.03, 0.97,
            f"Pearson $r$={pearson:.3f}\nSpearman $\\rho$={spearman:.3f}\n$n$={n}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                  "alpha": 0.9, "linewidth": 0.6, "edgecolor": "#cccccc"},
        )


def plot_difficulty_sharpness(
    question_params: pd.DataFrame,
    *,
    save_path: Path,
    title: str,
    model_params: pd.DataFrame | None = None,
    criterion: str = "effective",
) -> dict[str, float]:
    qp = question_params.dropna(subset=["difficulty_b", "sharpness_exp_k"]).copy()
    if qp.empty:
        return {"pearson": float("nan"), "spearman": float("nan"), "n_questions": 0.0}

    qp["log_sharpness"] = np.log(qp["sharpness_exp_k"].clip(lower=1e-12))
    pearson, _ = pearsonr(qp["difficulty_b"], qp["log_sharpness"]) if len(qp) > 1 else (float("nan"), None)
    spearman, _ = spearmanr(qp["difficulty_b"], qp["log_sharpness"]) if len(qp) > 1 else (float("nan"), None)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.2))

    _style_ax(ax)
    benchmarks = _scatter_by_benchmark(ax, qp, "log_sharpness")
    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=0.9, zorder=2)
    ax.axvline(0.0, color="#999999", linestyle="--", linewidth=0.9, zorder=2)
    ax.set_xlabel(r"Item difficulty $b_q$", fontsize=11)
    ax.set_ylabel(r"Log sharpness $\log a_q$", fontsize=11)
    ax.set_title("Difficulty vs Sharpness", fontsize=11, pad=8)
    _annotate(ax, qp["difficulty_b"].to_numpy(), qp["log_sharpness"].to_numpy(), len(qp))
    if len(benchmarks) > 1:
        ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="#cccccc", loc="best")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return {"pearson": float(pearson), "spearman": float(spearman), "n_questions": float(len(qp))}


def select_top_sharpness(
    question_params: pd.DataFrame,
    fraction: float,
    model_params: pd.DataFrame | None = None,
    criterion: str = "effective",
    pairwise_df: pd.DataFrame | None = None,
    run_summary: dict[str, Any] | None = None,
    fisher_mode: str = "objective",
) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError(f"Fractions must be in (0, 1], got {fraction}")
    if criterion not in SELECTION_CRITERIA:
        raise ValueError(f"Unknown selection criterion '{criterion}'. Choose from: {sorted(SELECTION_CRITERIA)}")
    if fisher_mode not in FISHER_MODES:
        raise ValueError(f"Unknown Fisher mode '{fisher_mode}'. Choose from: {sorted(FISHER_MODES)}")
    n_total = len(question_params)
    n_select = max(1, min(n_total, int(math.ceil(fraction * n_total))))
    qp = question_params.copy()
    if criterion == "raw" or model_params is None or model_params.empty or "theta" not in model_params.columns:
        sort_col = "sharpness_exp_k"
    else:
        qp["_selection_score"] = _compute_selection_scores(
            qp,
            criterion,
            model_params,
            pairwise_df=pairwise_df,
            run_summary=run_summary,
            fisher_mode=fisher_mode,
        )
        sort_col = "_selection_score"
    return (
        qp.sort_values(sort_col, ascending=False)
        .head(n_select)
        .drop(columns=["_selection_score"], errors="ignore")
        .copy()
        .reset_index(drop=True)
    )


def fit_abilities_with_fixed_items(
    *,
    full_model_params: pd.DataFrame,
    question_params: pd.DataFrame,
    static_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    selected_question_ids: set[str],
    run_summary: dict[str, Any],
    epochs: int,
    lr: float,
    reg_lambda: float,
) -> pd.DataFrame:
    model_names = full_model_params["model_name"].astype(str).tolist()
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}
    selected_qp = question_params[question_params["question_id"].astype(str).isin(selected_question_ids)].copy()
    if selected_qp.empty:
        return pd.DataFrame(columns=["model_name", "theta"])

    q_to_idx = {str(qid): idx for idx, qid in enumerate(selected_qp["question_id"].astype(str))}
    b_values = selected_qp["difficulty_b"].astype(float).to_numpy()
    a_values = selected_qp["sharpness_exp_k"].astype(float).to_numpy()

    static = static_df[
        static_df["question_id"].astype(str).isin(selected_question_ids)
        & static_df["model_name"].astype(str).isin(model_to_idx)
    ].copy()
    pairwise = pairwise_df[
        pairwise_df["question_id"].astype(str).isin(selected_question_ids)
        & pairwise_df["model_1"].astype(str).isin(model_to_idx)
        & pairwise_df["model_2"].astype(str).isin(model_to_idx)
    ].copy()
    reward = reward_df[
        reward_df["question_id"].astype(str).isin(selected_question_ids)
        & reward_df["model_name"].astype(str).isin(model_to_idx)
    ].copy()

    if static.empty and pairwise.empty and reward.empty:
        return pd.DataFrame(columns=["model_name", "theta"])

    # This fit has very few trainable parameters; CPU avoids GPU launch overhead.
    device = torch.device("cpu")
    theta = nn.Embedding(len(model_names), 1, device=device)
    nn.init.zeros_(theta.weight)
    optimizer = optim.Adam(theta.parameters(), lr=lr)
    b_fixed = torch.tensor(b_values, dtype=torch.float32, device=device)
    a_fixed = torch.tensor(a_values, dtype=torch.float32, device=device)
    bce_logits = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    tensors: dict[str, torch.Tensor] = {}
    if not static.empty:
        tensors["static_m"] = torch.tensor(static["model_name"].map(model_to_idx).to_numpy(), dtype=torch.long, device=device)
        tensors["static_q"] = torch.tensor(static["question_id"].astype(str).map(q_to_idx).to_numpy(), dtype=torch.long, device=device)
        tensors["static_y"] = torch.tensor(static["judge_result"].astype(float).to_numpy(), dtype=torch.float32, device=device)

    if not pairwise.empty:
        if {"tie", "both_bad"}.issubset(pairwise.columns):
            hard_pairwise = pairwise[
                ~pairwise["tie"].astype(bool)
                & ~pairwise["both_bad"].astype(bool)
            ].copy()
        elif "tie" in pairwise.columns:
            hard_pairwise = pairwise[~pairwise["tie"].astype(bool)].copy()
        else:
            hard_pairwise = pairwise.copy()
        if not hard_pairwise.empty:
            tensors["pair_m1"] = torch.tensor(hard_pairwise["model_1"].map(model_to_idx).to_numpy(), dtype=torch.long, device=device)
            tensors["pair_m2"] = torch.tensor(hard_pairwise["model_2"].map(model_to_idx).to_numpy(), dtype=torch.long, device=device)
            tensors["pair_q"] = torch.tensor(hard_pairwise["question_id"].astype(str).map(q_to_idx).to_numpy(), dtype=torch.long, device=device)
            tensors["pair_y"] = torch.tensor(hard_pairwise["target_prob"].astype(float).to_numpy(), dtype=torch.float32, device=device)
        if "both_bad" in pairwise.columns and "tie" in pairwise.columns:
            both_bad = pairwise[pairwise["both_bad"].astype(bool) & ~pairwise["tie"].astype(bool)].copy()
            if not both_bad.empty:
                tensors["bb_m1"] = torch.tensor(both_bad["model_1"].map(model_to_idx).to_numpy(), dtype=torch.long, device=device)
                tensors["bb_m2"] = torch.tensor(both_bad["model_2"].map(model_to_idx).to_numpy(), dtype=torch.long, device=device)
                tensors["bb_q"] = torch.tensor(both_bad["question_id"].astype(str).map(q_to_idx).to_numpy(), dtype=torch.long, device=device)

    if not reward.empty:
        tensors["reward_m"] = torch.tensor(reward["model_name"].map(model_to_idx).to_numpy(), dtype=torch.long, device=device)
        tensors["reward_q"] = torch.tensor(reward["question_id"].astype(str).map(q_to_idx).to_numpy(), dtype=torch.long, device=device)
        tensors["reward_y"] = torch.tensor(reward["reward_z"].astype(float).to_numpy(), dtype=torch.float32, device=device)

    lambda_static = float(run_summary.get("lambda_static", 1.0))
    lambda_arena = float(run_summary.get("lambda_arena", 1.0))
    lambda_bb = float(run_summary.get("lambda_bb", 0.0))
    gamma = float(run_summary.get("learned_gamma") or 1.0)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss_static = torch.tensor(0.0, device=device)
        if "static_m" in tensors:
            q_idx = tensors["static_q"]
            logits = a_fixed[q_idx] * (theta(tensors["static_m"]).squeeze(-1) - b_fixed[q_idx])
            loss_static = bce_logits(logits, tensors["static_y"])

        loss_arena = torch.tensor(0.0, device=device)
        if "pair_m1" in tensors:
            q_idx = tensors["pair_q"]
            p1 = torch.sigmoid(a_fixed[q_idx] * (theta(tensors["pair_m1"]).squeeze(-1) - b_fixed[q_idx]))
            p2 = torch.sigmoid(a_fixed[q_idx] * (theta(tensors["pair_m2"]).squeeze(-1) - b_fixed[q_idx]))
            loss_arena = bce_logits(gamma * (p1 - p2), tensors["pair_y"])
        elif "reward_m" in tensors:
            q_idx = tensors["reward_q"]
            pred = a_fixed[q_idx] * (theta(tensors["reward_m"]).squeeze(-1) - b_fixed[q_idx])
            loss_arena = mse(pred, tensors["reward_y"])

        loss_bb = torch.tensor(0.0, device=device)
        if "bb_m1" in tensors:
            q_idx = tensors["bb_q"]
            p1 = torch.sigmoid(a_fixed[q_idx] * (theta(tensors["bb_m1"]).squeeze(-1) - b_fixed[q_idx]))
            p2 = torch.sigmoid(a_fixed[q_idx] * (theta(tensors["bb_m2"]).squeeze(-1) - b_fixed[q_idx]))
            loss_bb = -(torch.log(1.0 - p1 + 1e-6).mean() + torch.log(1.0 - p2 + 1e-6).mean())

        loss = lambda_static * loss_static + lambda_arena * loss_arena + lambda_bb * loss_bb
        loss = loss + reg_lambda * theta.weight.pow(2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            theta.weight.sub_(theta.weight.mean())

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    return (
        pd.DataFrame({"model_name": model_names, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )


def fit_bt_with_selected_arena_questions(
    *,
    full_model_params: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    selected_question_ids: set[str],
    epochs: int,
    lr: float,
    reg_lambda: float,
) -> pd.DataFrame:
    model_names = full_model_params["model_name"].astype(str).tolist()
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}
    pairwise = pairwise_df[
        pairwise_df["question_id"].astype(str).isin(selected_question_ids)
        & pairwise_df["model_1"].astype(str).isin(model_to_idx)
        & pairwise_df["model_2"].astype(str).isin(model_to_idx)
    ].copy()
    if pairwise.empty:
        return pd.DataFrame(columns=["model_name", "theta"])

    if "tie" in pairwise.columns:
        pairwise = pairwise[~pairwise["tie"].astype(bool)].copy()
    if pairwise.empty:
        return pd.DataFrame(columns=["model_name", "theta"])

    # Direct BT is the arena-only baseline: no item parameters used.
    device = torch.device("cpu")
    theta = nn.Embedding(len(model_names), 1, device=device)
    nn.init.zeros_(theta.weight)
    optimizer = optim.Adam(theta.parameters(), lr=lr)
    bce_logits = nn.BCEWithLogitsLoss()

    m1 = torch.tensor(pairwise["model_1"].map(model_to_idx).to_numpy(), dtype=torch.long, device=device)
    m2 = torch.tensor(pairwise["model_2"].map(model_to_idx).to_numpy(), dtype=torch.long, device=device)
    target = torch.tensor(pairwise["target_prob"].astype(float).to_numpy(), dtype=torch.float32, device=device)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = theta(m1).squeeze(-1) - theta(m2).squeeze(-1)
        loss = bce_logits(logits, target) + reg_lambda * theta.weight.pow(2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            theta.weight.sub_(theta.weight.mean())

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    return (
        pd.DataFrame({"model_name": model_names, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )


def fit_2pl_bt_from_scratch(
    *,
    full_model_params: pd.DataFrame,
    static_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    selected_question_ids: set[str],
    epochs: int,
    lr: float,
    reg_lambda: float,
) -> pd.DataFrame:
    """2PL IRT on static questions + BT on arena questions, all learned from scratch."""
    model_names = full_model_params["model_name"].astype(str).tolist()
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    static = static_df[
        static_df["question_id"].astype(str).isin(selected_question_ids)
        & static_df["model_name"].astype(str).isin(model_to_idx)
    ].copy()
    pairwise = pairwise_df[
        pairwise_df["question_id"].astype(str).isin(selected_question_ids)
        & pairwise_df["model_1"].astype(str).isin(model_to_idx)
        & pairwise_df["model_2"].astype(str).isin(model_to_idx)
    ].copy()
    if not pairwise.empty and "tie" in pairwise.columns:
        pairwise = pairwise[~pairwise["tie"].astype(bool)].copy()

    if static.empty and pairwise.empty:
        return pd.DataFrame(columns=["model_name", "theta"])

    device = torch.device("cpu")
    theta = nn.Embedding(len(model_names), 1, device=device)
    nn.init.zeros_(theta.weight)
    trainable = list(theta.parameters())

    tensors: dict[str, torch.Tensor] = {}

    if not static.empty:
        static_qids = static["question_id"].astype(str).unique().tolist()
        static_q_to_idx = {qid: i for i, qid in enumerate(static_qids)}
        b_static = nn.Embedding(len(static_qids), 1, device=device)
        log_a_static = nn.Embedding(len(static_qids), 1, device=device)
        nn.init.zeros_(b_static.weight)
        nn.init.zeros_(log_a_static.weight)  # initialises a_q = exp(0) = 1
        trainable += list(b_static.parameters()) + list(log_a_static.parameters())
        tensors["static_m"] = torch.tensor(
            static["model_name"].map(model_to_idx).to_numpy(), dtype=torch.long, device=device
        )
        tensors["static_q"] = torch.tensor(
            static["question_id"].astype(str).map(static_q_to_idx).to_numpy(), dtype=torch.long, device=device
        )
        tensors["static_y"] = torch.tensor(
            static["judge_result"].astype(float).to_numpy(), dtype=torch.float32, device=device
        )

    if not pairwise.empty:
        tensors["pair_m1"] = torch.tensor(
            pairwise["model_1"].map(model_to_idx).to_numpy(), dtype=torch.long, device=device
        )
        tensors["pair_m2"] = torch.tensor(
            pairwise["model_2"].map(model_to_idx).to_numpy(), dtype=torch.long, device=device
        )
        tensors["pair_y"] = torch.tensor(
            pairwise["target_prob"].astype(float).to_numpy(), dtype=torch.float32, device=device
        )

    optimizer = optim.Adam(trainable, lr=lr)
    bce_logits = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        if "static_m" in tensors:
            q_idx = tensors["static_q"]
            a = torch.exp(log_a_static(q_idx)).squeeze(-1)
            b = b_static(q_idx).squeeze(-1)
            logits = a * (theta(tensors["static_m"]).squeeze(-1) - b)
            loss = loss + bce_logits(logits, tensors["static_y"])
            loss = loss + reg_lambda * (b_static.weight.pow(2).mean() + log_a_static.weight.pow(2).mean())
        if "pair_m1" in tensors:
            logits_bt = theta(tensors["pair_m1"]).squeeze(-1) - theta(tensors["pair_m2"]).squeeze(-1)
            loss = loss + bce_logits(logits_bt, tensors["pair_y"])
        loss = loss + reg_lambda * theta.weight.pow(2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            theta.weight.sub_(theta.weight.mean())

    theta_np = theta.weight.detach().cpu().numpy().squeeze(-1)
    return (
        pd.DataFrame({"model_name": model_names, "theta": theta_np})
        .sort_values("theta", ascending=False)
        .reset_index(drop=True)
    )


def subset_pairwise_stats(pairwise_df: pd.DataFrame, selected_question_ids: set[str]) -> dict[str, float]:
    if pairwise_df.empty:
        return {
            "n_pairwise_rows": 0.0,
            "hard_pair_fraction": float("nan"),
            "both_bad_fraction": float("nan"),
            "tie_fraction": float("nan"),
        }
    subset = pairwise_df[pairwise_df["question_id"].astype(str).isin(selected_question_ids)].copy()
    if subset.empty:
        return {
            "n_pairwise_rows": 0.0,
            "hard_pair_fraction": float("nan"),
            "both_bad_fraction": float("nan"),
            "tie_fraction": float("nan"),
        }
    tie = subset["tie"].astype(bool) if "tie" in subset.columns else pd.Series(False, index=subset.index)
    both_bad = (
        subset["both_bad"].astype(bool)
        if "both_bad" in subset.columns
        else pd.Series(False, index=subset.index)
    )
    hard = ~tie & ~both_bad
    return {
        "n_pairwise_rows": float(len(subset)),
        "hard_pair_fraction": float(hard.mean()),
        "both_bad_fraction": float(both_bad.mean()),
        "tie_fraction": float(tie.mean()),
    }


def subset_item_stats(question_params: pd.DataFrame, selected_question_ids: set[str]) -> dict[str, float]:
    subset = question_params[question_params["question_id"].astype(str).isin(selected_question_ids)].copy()
    if subset.empty:
        return {
            "mean_difficulty_b": float("nan"),
            "mean_sharpness_exp_k": float("nan"),
            "median_difficulty_b": float("nan"),
            "median_sharpness_exp_k": float("nan"),
        }
    return {
        "mean_difficulty_b": float(subset["difficulty_b"].mean()),
        "mean_sharpness_exp_k": float(subset["sharpness_exp_k"].mean()),
        "median_difficulty_b": float(subset["difficulty_b"].median()),
        "median_sharpness_exp_k": float(subset["sharpness_exp_k"].median()),
    }


def summarise_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby(
            ["selection_criterion", "selection_strategy", "target_fraction", "n_questions", "n_total_questions"],
            as_index=False,
        )
        .agg(
            actual_fraction=("actual_fraction", "mean"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            kendall_mean=("kendall", "mean"),
            kendall_std=("kendall", "std"),
            exact_matches_mean=("exact_matches", "mean"),
            top_k_retention_mean=("top_k_retention", "mean"),
            n_trials=("seed", "count"),
        )
        .sort_values(["selection_strategy", "n_questions"])
        .reset_index(drop=True)
    )


STRATEGY_LABELS = {
    "sharpness_top":       "Top Fisher info (DualEval)",
    "sharpness_only":      r"Top sharpness $a_q$ (DualEval)",
    "difficulty_matched":  "Difficulty-matched (DualEval)",
    "top_static_only":     "Top Fisher, static-only (DualEval)",
    "top_arena_only":      "Top Fisher, arena-only (DualEval)",
    "random":              "Random (DualEval)",
    "random_bt":           "Random (BT)",
    "random_2pl_bt":       "Random (2PL+BT joint)",
}


# Colors ordered best-to-worst recovery: green, blue, purple, orange, red.
STRATEGY_COLORS = {
    "sharpness_top":       "tab:green",
    "sharpness_only":      "tab:blue",
    "difficulty_matched":  "tab:purple",
    "random":              "tab:orange",
    "random_2pl_bt":       "tab:red",
    "random_bt":           "tab:red",
    "top_static_only":     "tab:brown",
    "top_arena_only":      "tab:gray",
}


def _select_top_n(
    qp: pd.DataFrame,
    n: int,
    *,
    model_params: pd.DataFrame | None,
    criterion: str,
    pairwise_df: pd.DataFrame | None,
    run_summary: dict[str, Any] | None,
    fisher_mode: str,
) -> pd.DataFrame:
    """Pick top-n items from ``qp`` by the given selection criterion.

    Unlike ``select_top_sharpness``, the count ``n`` is absolute, not a fraction,
    so the caller controls how the requested fraction maps onto a restricted pool.
    """
    if n <= 0 or qp.empty:
        return qp.head(0).copy()
    n = min(n, len(qp))
    if criterion == "raw" or model_params is None or model_params.empty or "theta" not in model_params.columns:
        sort_col = "sharpness_exp_k"
        return qp.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)
    qp = qp.copy()
    qp["_selection_score"] = _compute_selection_scores(
        qp, criterion, model_params,
        pairwise_df=pairwise_df, run_summary=run_summary, fisher_mode=fisher_mode,
    )
    return (
        qp.sort_values("_selection_score", ascending=False)
        .head(n)
        .drop(columns=["_selection_score"], errors="ignore")
        .reset_index(drop=True)
    )


def _select_difficulty_matched(
    qp: pd.DataFrame,
    n: int,
    model_params: pd.DataFrame | None,
) -> pd.DataFrame:
    """Pick items whose difficulty is closest to the median model ability."""
    if n <= 0 or qp.empty:
        return qp.head(0).copy()
    n = min(n, len(qp))
    if model_params is None or model_params.empty or "theta" not in model_params.columns:
        theta_ref = 0.0
    else:
        theta_ref = float(model_params["theta"].median())
    qp = qp.copy()
    qp["_distance_to_theta"] = (qp["difficulty_b"].astype(float) - theta_ref).abs()
    return (
        qp.sort_values("_distance_to_theta", ascending=True)
        .head(n)
        .drop(columns=["_distance_to_theta"], errors="ignore")
        .reset_index(drop=True)
    )


def plot_recovery(
    summary_df: pd.DataFrame,
    *,
    save_path: Path,
    title: str,
    y_min: float = 0.6,
    y_max: float = 1.02,
) -> None:
    """Two-panel recovery plot: left = Spearman, right = Kendall."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True, sharey=True)
    panels = [
        (summary_df, "spearman_mean", "spearman_std", "Spearman $\\rho$", ""),
        (summary_df, "kendall_mean",  "kendall_std",  "Kendall $\\tau$", ""),
    ]
    for ax, (df, mean_col, std_col, ylabel, panel_title) in zip(axes, panels):
        for strategy, part in df.groupby("selection_strategy", sort=False):
            part = part.sort_values("actual_fraction")
            label = STRATEGY_LABELS.get(str(strategy), str(strategy))
            color = STRATEGY_COLORS.get(str(strategy))
            ax.plot(part["actual_fraction"], part[mean_col], marker="o", linewidth=1.8, label=label, color=color)
            if std_col and std_col in part.columns and part[std_col].notna().any():
                lower = part[mean_col] - part[std_col].fillna(0.0)
                upper = part[mean_col] + part[std_col].fillna(0.0)
                ax.fill_between(part["actual_fraction"], lower, upper, alpha=0.15, color=color)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Fraction of questions used")
        ax.set_ylabel(ylabel)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.25)
        if panel_title:
            ax.set_title(panel_title)
    axes[0].legend(fontsize=8, frameon=True)
    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_recovery_multi(
    summaries: list[tuple[str, pd.DataFrame]],
    *,
    save_path: Path,
    title: str = "",
    metric: str = "spearman",
    y_min: float = 0.6,
    y_max: float = 1.02,
) -> None:
    """N-panel recovery plot. ``summaries`` is a list of (panel_title, summary_df)."""
    n = len(summaries)
    if n == 0:
        raise ValueError("plot_recovery_multi requires at least one summary.")
    fig_w = max(4.5 * n, 6.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.2), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    ylabel = "Spearman $\\rho$" if metric == "spearman" else metric

    # Render strategies in a deterministic order so legend order matches
    # the STRATEGY_COLORS / STRATEGY_LABELS ordering (best-to-worst).
    strategy_order = [name for name in STRATEGY_LABELS if any(
        name in set(df["selection_strategy"].astype(str)) for _, df in summaries
    )]

    legend_handles = []
    legend_labels = []
    for idx, (ax, (panel_title, df)) in enumerate(zip(axes, summaries)):
        present = set(df["selection_strategy"].astype(str))
        for strategy in strategy_order:
            if strategy not in present:
                continue
            part = df[df["selection_strategy"].astype(str) == strategy].sort_values("actual_fraction")
            label = STRATEGY_LABELS.get(str(strategy), str(strategy))
            color = STRATEGY_COLORS.get(str(strategy))
            line, = ax.plot(
                part["actual_fraction"], part[mean_col],
                marker="o", linewidth=1.8, label=label, color=color,
            )
            if std_col in part.columns and part[std_col].notna().any():
                lower = part[mean_col] - part[std_col].fillna(0.0)
                upper = part[mean_col] + part[std_col].fillna(0.0)
                ax.fill_between(part["actual_fraction"], lower, upper, alpha=0.15, color=color)
            if idx == 0 and label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Fraction of questions used")
        if idx == 0:
            ax.set_ylabel(ylabel)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.25)
        ax.set_title(panel_title)

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=len(legend_labels),
        frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=9,
    )
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def load_run_inputs(run_dir: Path, run_summary: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    static_paths = [str(path) for path in run_summary.get("static_jsonl", []) if path]
    reward_paths = [str(path) for path in run_summary.get("arena_reward_jsonl", []) if path]
    static_df = load_static_jsonl(static_paths) if static_paths else pd.DataFrame()
    reward_df = load_arena_reward_jsonl(reward_paths) if reward_paths else pd.DataFrame()
    pairwise_candidates = [
        run_dir / "arena_pairwise_targets.csv",
        run_dir / "arena_soft_pairwise_targets.csv",
    ]
    pairwise_df = pd.DataFrame()
    for pairwise_path in pairwise_candidates:
        if pairwise_path.exists():
            pairwise_df = pd.read_csv(pairwise_path)
            break
    if pairwise_df.empty and not reward_df.empty:
        both_bad_threshold, tie_delta = resolve_pairwise_thresholds(
            reward_df,
            bb_ratio=float(run_summary.get("bb_ratio", 0.0) or 0.0),
            tie_ratio=float(run_summary.get("tie_ratio", 0.0) or 0.0),
        )
        pairwise_df = build_soft_pairwise_targets(
            reward_df,
            both_bad_threshold=both_bad_threshold,
            tie_delta=tie_delta,
        )
    return static_df, pairwise_df, reward_df


def run_single_experiment(
    *,
    run_dir: Path,
    output_dir: Path,
    fractions: list[float],
    random_seeds: list[int],
    benchmarks: set[str] | None,
    selection_criterion: str,
    top_k: int,
    epochs: int,
    lr: float,
    reg_lambda: float,
    fisher_mode: str,
    y_min: float = 0.6,
    y_max: float = 1.02,
) -> pd.DataFrame:
    rel_name = run_dir.relative_to(REPO_ROOT / "results" / "ranking_rm") if run_dir.is_relative_to(REPO_ROOT / "results" / "ranking_rm") else run_dir.name
    filter_label = "_".join(sorted(benchmarks)) if benchmarks else "both"
    selection_label = f"{selection_criterion}_{fisher_mode}" if selection_criterion == "fisher" else selection_criterion
    run_label = f"{filter_label.lower()}_{selection_label}"
    run_output = output_dir / run_label
    run_output.mkdir(parents=True, exist_ok=True)

    model_params = pd.read_csv(run_dir / "model_ranking.csv")
    question_params = pd.read_csv(run_dir / "question_ranking.csv").rename(
        columns={"discrimination_exp_k": "sharpness_exp_k"}
    )
    if benchmarks:
        question_params = question_params[
            question_params["benchmark"].astype(str).isin(benchmarks)
        ].copy()
        if question_params.empty:
            raise SystemExit(f"No questions in {run_dir} matched benchmarks: {sorted(benchmarks)}")
    with (run_dir / "run_summary.json").open("r", encoding="utf-8") as fh:
        run_summary = json.load(fh)

    corr = plot_difficulty_sharpness(
        question_params,
        save_path=run_output / f"difficulty_vs_sharpness_{filter_label}.pdf",
        title=f"Difficulty vs. Sharpness — {rel_name} ({filter_label}, {selection_label})",
        model_params=model_params,
        criterion=selection_criterion,
    )
    pd.DataFrame([{**corr, "run_dir": str(run_dir), "question_filter": filter_label}]).to_csv(
        run_output / f"difficulty_sharpness_correlation_{filter_label}.csv",
        index=False,
    )

    static_df, pairwise_df, reward_df = load_run_inputs(run_dir, run_summary)
    if static_df.empty and pairwise_df.empty and reward_df.empty:
        print(f"Skipping subset recovery for {run_dir}: no saved/input response data found.", flush=True)
        return pd.DataFrame()

    has_both = not static_df.empty and not pairwise_df.empty
    all_qids = question_params["question_id"].astype(str).tolist()

    # Pre-compute full-data reference ranking for the alt baseline so each
    # method is evaluated against its own full-data fit, not DualEval's.
    print(f"  Fitting full-data alt reference ({('2PL+BT' if has_both else 'BT')})...", flush=True)
    if has_both:
        alt_ref_params = fit_2pl_bt_from_scratch(
            full_model_params=model_params,
            static_df=static_df,
            pairwise_df=pairwise_df,
            selected_question_ids=set(all_qids),
            epochs=epochs,
            lr=lr,
            reg_lambda=reg_lambda,
        )
    else:
        alt_ref_params = fit_bt_with_selected_arena_questions(
            full_model_params=model_params,
            pairwise_df=pairwise_df,
            selected_question_ids=set(all_qids),
            epochs=epochs,
            lr=lr,
            reg_lambda=reg_lambda,
        )

    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        selected = select_top_sharpness(
            question_params,
            fraction,
            model_params=model_params,
            criterion=selection_criterion,
            pairwise_df=pairwise_df,
            run_summary=run_summary,
            fisher_mode=fisher_mode,
        )
        selected_ids = set(selected["question_id"].astype(str))
        selected_stats = {
            **subset_item_stats(question_params, selected_ids),
            **subset_pairwise_stats(pairwise_df, selected_ids),
        }
        recovered = fit_abilities_with_fixed_items(
            full_model_params=model_params,
            question_params=question_params,
            static_df=static_df,
            pairwise_df=pairwise_df,
            reward_df=reward_df,
            selected_question_ids=selected_ids,
            run_summary=run_summary,
            epochs=epochs,
            lr=lr,
            reg_lambda=reg_lambda,
        )
        metrics = ranking_metrics(model_params, recovered, top_k=top_k)
        rows.append(
            {
                "run_dir": str(run_dir),
                "selection_criterion": selection_label,
                "selection_strategy": "sharpness_top",
                "target_fraction": fraction,
                "actual_fraction": len(selected_ids) / len(all_qids),
                "n_questions": len(selected_ids),
                "n_total_questions": len(all_qids),
                "seed": -1,
                **selected_stats,
                **metrics,
            }
        )
        selected.to_csv(run_output / f"top_sharpness_questions_{fraction:.3f}.csv", index=False)

        n_target = len(selected_ids)

        ablation_specs: list[tuple[str, pd.DataFrame]] = [
            (
                "sharpness_only",
                _select_top_n(
                    question_params, n_target,
                    model_params=model_params, criterion="raw",
                    pairwise_df=pairwise_df, run_summary=run_summary, fisher_mode=fisher_mode,
                ),
            ),
            (
                "difficulty_matched",
                _select_difficulty_matched(question_params, n_target, model_params),
            ),
        ]

        for strategy_name, selected_ablation in ablation_specs:
            if selected_ablation.empty:
                continue
            ablation_ids = set(selected_ablation["question_id"].astype(str))
            ablation_stats = {
                **subset_item_stats(question_params, ablation_ids),
                **subset_pairwise_stats(pairwise_df, ablation_ids),
            }
            recovered_ablation = fit_abilities_with_fixed_items(
                full_model_params=model_params,
                question_params=question_params,
                static_df=static_df,
                pairwise_df=pairwise_df,
                reward_df=reward_df,
                selected_question_ids=ablation_ids,
                run_summary=run_summary,
                epochs=epochs,
                lr=lr,
                reg_lambda=reg_lambda,
            )
            ablation_metrics = ranking_metrics(model_params, recovered_ablation, top_k=top_k)
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "selection_criterion": selection_label,
                    "selection_strategy": strategy_name,
                    "target_fraction": fraction,
                    "actual_fraction": len(ablation_ids) / len(all_qids),
                    "n_questions": len(ablation_ids),
                    "n_total_questions": len(all_qids),
                    "seed": -1,
                    **ablation_stats,
                    **ablation_metrics,
                }
            )

        for seed in random_seeds:
            rng = np.random.default_rng(seed)
            random_ids = set(rng.choice(all_qids, size=len(selected_ids), replace=False).tolist())
            random_stats = {
                **subset_item_stats(question_params, random_ids),
                **subset_pairwise_stats(pairwise_df, random_ids),
            }
            recovered_random = fit_abilities_with_fixed_items(
                full_model_params=model_params,
                question_params=question_params,
                static_df=static_df,
                pairwise_df=pairwise_df,
                reward_df=reward_df,
                selected_question_ids=random_ids,
                run_summary=run_summary,
                epochs=epochs,
                lr=lr,
                reg_lambda=reg_lambda,
            )
            random_metrics = ranking_metrics(model_params, recovered_random, top_k=top_k)
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "selection_criterion": selection_label,
                    "selection_strategy": "random",
                    "target_fraction": fraction,
                    "actual_fraction": len(random_ids) / len(all_qids),
                    "n_questions": len(random_ids),
                    "n_total_questions": len(all_qids),
                    "seed": seed,
                    **random_stats,
                    **random_metrics,
                }
            )
            if has_both:
                recovered_alt = fit_2pl_bt_from_scratch(
                    full_model_params=model_params,
                    static_df=static_df,
                    pairwise_df=pairwise_df,
                    selected_question_ids=random_ids,
                    epochs=epochs,
                    lr=lr,
                    reg_lambda=reg_lambda,
                )
                alt_strategy = "random_2pl_bt"
            else:
                recovered_alt = fit_bt_with_selected_arena_questions(
                    full_model_params=model_params,
                    pairwise_df=pairwise_df,
                    selected_question_ids=random_ids,
                    epochs=epochs,
                    lr=lr,
                    reg_lambda=reg_lambda,
                )
                alt_strategy = "random_bt"
            alt_metrics = ranking_metrics(alt_ref_params, recovered_alt, top_k=top_k)
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "selection_criterion": selection_label,
                    "selection_strategy": alt_strategy,
                    "target_fraction": fraction,
                    "actual_fraction": len(random_ids) / len(all_qids),
                    "n_questions": len(random_ids),
                    "n_total_questions": len(all_qids),
                    "seed": seed,
                    **random_stats,
                    **alt_metrics,
                }
            )
        print(
            f"{rel_name}: top {len(selected_ids)}/{len(all_qids)} questions "
            f"({len(selected_ids) / len(all_qids):.1%}, criterion={selection_label}) -> "
            f"rho={metrics['spearman']:.3f}, tau={metrics['kendall']:.3f}",
            flush=True,
        )

    metrics_df = pd.DataFrame(rows)
    summary_df = summarise_metrics(metrics_df)
    metrics_df.to_csv(run_output / f"subset_recovery_{filter_label}_raw.csv", index=False)
    summary_df.to_csv(run_output / f"subset_recovery_{filter_label}_summary.csv", index=False)
    plot_recovery(
        summary_df,
        save_path=run_output / f"subset_recovery_{filter_label}.pdf",
        title=f"Ranking Recovery from Discriminative Questions: ({selection_label})",
        y_min=y_min,
        y_max=y_max,
    )
    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "results" / "ranking_rm",
        help="Root directory to scan for ranking_rm runs.",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Specific run directories. Overrides --runs-root scanning.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "sharpness_value",
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Restrict question selection to benchmarks such as Arena, hle-math, olympiad-math.",
    )
    parser.add_argument(
        "--selection-criterion",
        choices=sorted(SELECTION_CRITERIA),
        default="effective",
        help=(
            "Question selection score: raw=a_q (raw sharpness); "
            "effective=a*p*(1-p) at median theta; "
            "fisher=Fisher information about model abilities; "
            "mean=mean over all model thetas of a*p*(1-p)."
        ),
    )
    parser.add_argument(
        "--fisher-mode",
        choices=sorted(FISHER_MODES),
        default="objective",
        help=(
            "Only used with --selection-criterion fisher. "
            "objective matches the fitted static/arena objectives and averages over models/pairs; "
            "simple keeps the old median-theta score a^2*p*(1-p)."
        ),
    )
    parser.add_argument(
        "--random-seeds",
        nargs="*",
        type=int,
        default=DEFAULT_RANDOM_SEEDS,
        help="Seeds for random-selection baseline. Pass no values to disable it.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=700, help="Ability-only fitting epochs per subset.")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=1e-4)
    parser.add_argument(
        "--multi-run-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Run dirs to plot as a single N-panel figure (one panel per dir).",
    )
    parser.add_argument(
        "--multi-run-labels",
        nargs="+",
        type=str,
        default=None,
        help="Panel titles for --multi-run-dirs (must match length). Defaults to run-dir name.",
    )
    parser.add_argument(
        "--multi-figure-out",
        type=Path,
        default=None,
        help="Path for the N-panel PDF (used with --multi-run-dirs).",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=0.6,
        help="Lower y-axis bound for recovery plots (default 0.6 zooms into the relevant region).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=1.02,
        help="Upper y-axis bound for recovery plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = {str(benchmark) for benchmark in args.benchmarks} if args.benchmarks else None

    # Multi-run mode: N panels, one per run dir. Renders a single figure with shared y-axis.
    if args.multi_run_dirs:
        run_paths = [resolve_path(p) for p in args.multi_run_dirs]
        labels_in = args.multi_run_labels or [p.parent.name for p in run_paths]
        if len(labels_in) != len(run_paths):
            raise SystemExit("--multi-run-labels must match --multi-run-dirs length.")
        common_kwargs = dict(
            output_dir=output_dir,
            fractions=args.fractions,
            random_seeds=args.random_seeds,
            benchmarks=benchmarks,
            selection_criterion=args.selection_criterion,
            top_k=args.top_k,
            epochs=args.epochs,
            lr=args.lr,
            reg_lambda=args.reg_lambda,
            fisher_mode=args.fisher_mode,
            y_min=args.y_min,
            y_max=args.y_max,
        )
        summaries: list[tuple[str, pd.DataFrame]] = []
        for run_path, label in zip(run_paths, labels_in):
            print(f"\n=== {label}: {run_path} ===", flush=True)
            metrics_df = run_single_experiment(run_dir=run_path, **common_kwargs)
            if metrics_df.empty:
                raise SystemExit(f"Empty metrics for {run_path}.")
            summaries.append((label, summarise_metrics(metrics_df)))
        fig_path = (
            resolve_path(args.multi_figure_out)
            if args.multi_figure_out is not None
            else output_dir / "subset_recovery_multi.pdf"
        )
        plot_recovery_multi(
            summaries,
            save_path=fig_path,
            title="",
            y_min=args.y_min,
            y_max=args.y_max,
        )
        print(f"\nSaved multi-panel PDF to {fig_path}", flush=True)
        return

    # Single-RM mode.
    run_dirs = [resolve_path(path) for path in args.run_dirs] if args.run_dirs else find_run_dirs(args.runs_root)
    if not run_dirs:
        raise SystemExit("No ranking_rm run directories found.")

    all_metrics: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        print(f"\n=== {run_dir} ===", flush=True)
        metrics_df = run_single_experiment(
            run_dir=run_dir,
            output_dir=output_dir,
            fractions=args.fractions,
            random_seeds=args.random_seeds,
            benchmarks=benchmarks,
            selection_criterion=args.selection_criterion,
            top_k=args.top_k,
            epochs=args.epochs,
            lr=args.lr,
            reg_lambda=args.reg_lambda,
            fisher_mode=args.fisher_mode,
            y_min=args.y_min,
            y_max=args.y_max,
        )
        if not metrics_df.empty:
            all_metrics.append(metrics_df)

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined.to_csv(output_dir / "subset_recovery_all_runs_raw.csv", index=False)
        summarise_metrics(combined).to_csv(output_dir / "subset_recovery_all_runs_summary.csv", index=False)
    print(f"\nOutputs saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
