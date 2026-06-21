#!/usr/bin/env python3
"""
Residual-based anomaly detection for DualEval (§4.3).

This experiment injects artificial contamination into either static benchmark
labels or arena reward scores, then refits DualEval on the contaminated data as
it would be used in practice. It asks whether residual diagnostics from that
observed-data fit recover the injected model-item cells better than simple
baselines. It is fully offline: it reuses existing correctness labels and reward
scores, and does not call model or reward APIs.

Example:
  python analysis/anomaly_detection.py \
      --config analysis/config_synthetic_recovery.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DUALEVAL_PATH = REPO_ROOT / "ranking" / "dualeval.py"
dualeval_spec = importlib.util.spec_from_file_location("dualeval_module", DUALEVAL_PATH)
if dualeval_spec is None or dualeval_spec.loader is None:
    raise ImportError(f"Could not load DualEval module from {DUALEVAL_PATH}")
dualeval = importlib.util.module_from_spec(dualeval_spec)
dualeval_spec.loader.exec_module(dualeval)


DEFAULT_INJECTION_RATES = [0.005, 0.01, 0.02, 0.05]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


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
    experiment_cfg = cfg.get("experiment", {})
    output_cfg = cfg.get("output", {})

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
        args.tie_ratio = float(training_cfg.get("tie_ratio", 0.15))
    if args.output_dir is None:
        args.output_dir = str(output_cfg.get("output_dir", "results/synthetic_recovery/default"))
    if args.injection_rates is None:
        args.injection_rates = [float(x) for x in experiment_cfg.get("injection_rates", DEFAULT_INJECTION_RATES)]
    if args.seeds is None:
        args.seeds = [int(x) for x in experiment_cfg.get("seeds", DEFAULT_SEEDS)]
    if args.rate_denominator is None:
        args.rate_denominator = str(experiment_cfg.get("rate_denominator", "all"))
    if args.injection_target is None:
        args.injection_target = str(experiment_cfg.get("injection_target", "static"))
    if args.candidate_pool_fraction is None:
        args.candidate_pool_fraction = float(experiment_cfg.get("candidate_pool_fraction", 0.30))
    if args.arena_reward_boost is None:
        args.arena_reward_boost = float(experiment_cfg.get("arena_reward_boost", 2.0))
    if args.top_cells_to_save is None:
        args.top_cells_to_save = int(output_cfg.get("top_cells_to_save", 500))
    if not args.quiet:
        args.quiet = bool(output_cfg.get("quiet", False))

    args.static_jsonl = resolve_paths(args.static_jsonl)
    args.arena_reward_jsonl = resolve_paths(args.arena_reward_jsonl)
    args.output_dir = resolve_path(args.output_dir)
    if args.rate_denominator not in {"all", "candidates"}:
        raise SystemExit("--rate-denominator must be either 'all' or 'candidates'.")
    if args.injection_target not in {"static", "arena"}:
        raise SystemExit("--injection-target must be either 'static' or 'arena'.")
    return args


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -709.0, 709.0)))


def add_irt_predictions(
    static_df: pd.DataFrame,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    theta = model_params.set_index("model_name")["theta"]
    q_params = question_params.set_index("question_id")[["difficulty_b", "discrimination_exp_k"]]

    scored = static_df.copy()
    scored[f"{prefix}_theta"] = scored["model_name"].map(theta)
    scored[f"{prefix}_difficulty_b"] = scored["question_id"].map(q_params["difficulty_b"])
    scored[f"{prefix}_a_q"] = scored["question_id"].map(q_params["discrimination_exp_k"])
    scored = scored.dropna(subset=[f"{prefix}_theta", f"{prefix}_difficulty_b", f"{prefix}_a_q"]).reset_index(drop=True)
    logits = scored[f"{prefix}_a_q"].to_numpy(dtype=float) * (
        scored[f"{prefix}_theta"].to_numpy(dtype=float) - scored[f"{prefix}_difficulty_b"].to_numpy(dtype=float)
    )
    scored[f"{prefix}_p"] = sigmoid_np(logits)
    return scored


def select_injection_candidates(
    static_df: pd.DataFrame,
    *,
    candidate_pool_fraction: float,
) -> pd.DataFrame:
    """Pick static contamination candidates using empirical statistics only.

    A cell is eligible if (a) judge_result == 0 and (b) the product of its
    per-model and per-item raw accuracy is among the lowest
    candidate_pool_fraction of the dataset. No IRT signal is used.
    """
    if static_df.empty:
        return static_df.head(0).copy()
    df = static_df.copy()
    model_acc = df.groupby("model_name")["judge_result"].mean()
    item_acc = df.groupby("question_id")["judge_result"].mean()
    df["empirical_model_acc"] = df["model_name"].map(model_acc)
    df["empirical_item_acc"] = df["question_id"].map(item_acc)
    df["empirical_expected_p"] = df["empirical_model_acc"] * df["empirical_item_acc"]

    failures = df[df["judge_result"].astype(int) == 0]
    if failures.empty:
        return failures.copy()
    target = max(1, int(round(candidate_pool_fraction * len(df))))
    target = min(target, len(failures))
    return failures.sort_values("empirical_expected_p").head(target).reset_index(drop=True)


def select_arena_injection_candidates(
    reward_df: pd.DataFrame,
    *,
    candidate_pool_fraction: float,
) -> pd.DataFrame:
    """Pick arena contamination candidates: rows in the bottom
    candidate_pool_fraction of raw reward_z. No IRT signal is used."""
    if reward_df.empty:
        return reward_df.head(0).copy()
    cutoff = float(reward_df["reward_z"].quantile(candidate_pool_fraction))
    return reward_df[reward_df["reward_z"].astype(float) <= cutoff].reset_index(drop=True)


def inject_static_contamination(
    static_df: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    injection_rate: float,
    seed: int,
    denominator: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if denominator == "all":
        target_n = int(math.ceil(injection_rate * len(static_df)))
    elif denominator == "candidates":
        target_n = int(math.ceil(injection_rate * len(candidates)))
    else:
        raise ValueError(f"Unknown denominator: {denominator}")

    target_n = max(1, target_n) if injection_rate > 0.0 else 0
    target_n = min(target_n, len(candidates))
    contaminated = static_df.copy()
    contaminated["synthetic_contaminated"] = False
    if target_n == 0:
        return contaminated, candidates.head(0).copy()

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(candidates.index.to_numpy(), size=target_n, replace=False)
    injected = candidates.loc[chosen_idx, ["model_name", "question_id"]].copy()
    injected["synthetic_contaminated"] = True

    key_to_row = pd.MultiIndex.from_frame(injected[["model_name", "question_id"]])
    static_keys = pd.MultiIndex.from_frame(contaminated[["model_name", "question_id"]])
    mask = static_keys.isin(key_to_row)
    contaminated.loc[mask, "judge_result"] = 1
    contaminated.loc[mask, "synthetic_contaminated"] = True
    return contaminated, injected.reset_index(drop=True)


def inject_arena_contamination(
    reward_df: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    injection_rate: float,
    seed: int,
    denominator: str,
    reward_boost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if denominator == "all":
        target_n = int(math.ceil(injection_rate * len(reward_df)))
    elif denominator == "candidates":
        target_n = int(math.ceil(injection_rate * len(candidates)))
    else:
        raise ValueError(f"Unknown denominator: {denominator}")

    target_n = max(1, target_n) if injection_rate > 0.0 else 0
    target_n = min(target_n, len(candidates))
    contaminated = reward_df.copy()
    contaminated["synthetic_contaminated"] = False
    if target_n == 0:
        return contaminated, candidates.head(0).copy()

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(candidates.index.to_numpy(), size=target_n, replace=False)
    injected = candidates.loc[chosen_idx, ["model_name", "question_id"]].copy()
    injected["synthetic_contaminated"] = True

    key_to_row = pd.MultiIndex.from_frame(injected[["model_name", "question_id"]])
    reward_keys = pd.MultiIndex.from_frame(contaminated[["model_name", "question_id"]])
    mask = reward_keys.isin(key_to_row)
    contaminated.loc[mask, "reward_z"] = contaminated.loc[mask, "reward_z"].astype(float) + reward_boost
    contaminated.loc[mask, "synthetic_contaminated"] = True
    return contaminated, injected.reset_index(drop=True)


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def roc_auc_score_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank_average(scores.astype(float))
    rank_sum_pos = float(ranks[y].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_score_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores.astype(float), kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1, dtype=float)
    precision = tp / ranks
    return float(precision[y_sorted].sum() / n_pos)


def topk_metrics(y_true: np.ndarray, scores: np.ndarray, *, k: int) -> tuple[float, float]:
    y = y_true.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0 or k <= 0:
        return float("nan"), float("nan")
    k = min(k, len(y))
    top = np.argsort(-scores.astype(float), kind="mergesort")[:k]
    hits = int(y[top].sum())
    precision = hits / k
    recall = hits / n_pos
    return float(precision), float(recall)


def evaluate_scores(scored: pd.DataFrame, *, method: str, score_col: str) -> dict[str, Any]:
    y_true = scored["synthetic_contaminated"].to_numpy(dtype=bool)
    scores = scored[score_col].to_numpy(dtype=float)
    n_pos = int(y_true.sum())
    precision_at_k, recall_at_k = topk_metrics(y_true, scores, k=n_pos)
    return {
        "method": method,
        "cell_auroc": roc_auc_score_binary(y_true, scores),
        "cell_auprc": average_precision_score_binary(y_true, scores),
        "cell_precision_at_k": precision_at_k,
        "cell_recall_at_k": recall_at_k,
    }


def evaluate_item_scores(scored: pd.DataFrame, *, method: str, score_col: str) -> dict[str, Any]:
    item = (
        scored.groupby("question_id", as_index=False)
        .agg(
            item_positive=("synthetic_contaminated", "max"),
            item_score=(score_col, "max"),
        )
    )
    y_true = item["item_positive"].to_numpy(dtype=bool)
    scores = item["item_score"].to_numpy(dtype=float)
    n_pos = int(y_true.sum())
    precision_at_k, recall_at_k = topk_metrics(y_true, scores, k=n_pos)
    return {
        "item_auroc": roc_auc_score_binary(y_true, scores),
        "item_auprc": average_precision_score_binary(y_true, scores),
        "item_precision_at_k": precision_at_k,
        "item_recall_at_k": recall_at_k,
    }


def score_contaminated_static(
    injected: pd.DataFrame,
    contam_dualeval_scored: pd.DataFrame,
    contam_static2pl_scored: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = [
        "source",
        "benchmark",
        "model_name",
        "question_id",
        "judge_result",
        "dualeval_p",
        "dualeval_a_q",
        "dualeval_difficulty_b",
    ]
    scored = contam_dualeval_scored[base_cols].rename(
        columns={"judge_result": "contaminated_judge_result"}
    ).copy()
    static_p = contam_static2pl_scored[[
        "model_name", "question_id",
        "static2pl_p", "static2pl_a_q", "static2pl_difficulty_b",
    ]]
    scored = scored.merge(static_p, on=["model_name", "question_id"], how="left")

    eps = 1e-6
    y = scored["contaminated_judge_result"].to_numpy(dtype=float)
    p_dual = np.clip(scored["dualeval_p"].to_numpy(dtype=float), eps, 1.0 - eps)
    p_static = np.clip(scored["static2pl_p"].to_numpy(dtype=float), eps, 1.0 - eps)
    scored["score_dualeval_residual"] = (y - p_dual) / np.sqrt(p_dual * (1.0 - p_dual) + eps)
    scored["score_static2pl_residual"] = (y - p_static) / np.sqrt(p_static * (1.0 - p_static) + eps)

    injected_keys = pd.MultiIndex.from_frame(injected[["model_name", "question_id"]]) if not injected.empty else pd.MultiIndex.from_arrays([[], []])
    scored_keys = pd.MultiIndex.from_frame(scored[["model_name", "question_id"]])
    scored["synthetic_contaminated"] = scored_keys.isin(injected_keys)
    scored["target"] = "static"
    scored["score_primary"] = scored["score_dualeval_residual"]
    return scored


def score_contaminated_arena(
    contaminated_reward: pd.DataFrame,
    injected: pd.DataFrame,
    contam_dualeval_arena_scored: pd.DataFrame,
    contam_arena2pl_scored: pd.DataFrame,
    contam_bt_models: pd.DataFrame,
    *,
    learned_gamma_dual: float | None,
    learned_gamma_arena2pl: float | None,
) -> pd.DataFrame:
    base_cols = [
        "source",
        "benchmark",
        "model_name",
        "question_id",
        "reward_raw",
        "reward_z",
        "dualeval_p",
        "dualeval_a_q",
        "dualeval_difficulty_b",
    ]
    scored = contam_dualeval_arena_scored[base_cols].rename(
        columns={"reward_z": "contaminated_reward_z"}
    ).copy()
    arena2pl_cols = contam_arena2pl_scored[[
        "model_name", "question_id",
        "arena2pl_p", "arena2pl_a_q", "arena2pl_difficulty_b",
    ]]
    scored = scored.merge(arena2pl_cols, on=["model_name", "question_id"], how="left")

    bt_theta = (
        contam_bt_models.set_index("model_name")["theta"]
        if not contam_bt_models.empty
        else pd.Series(dtype=float)
    )
    scored["bt_theta"] = scored["model_name"].map(bt_theta).fillna(0.0)

    n = len(scored)
    dual_sums = np.zeros(n, dtype=float)
    arena2pl_sums = np.zeros(n, dtype=float)
    bt_sums = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    gamma_dual = float(learned_gamma_dual) if learned_gamma_dual is not None else 4.0
    gamma_arena2pl = (
        float(learned_gamma_arena2pl) if learned_gamma_arena2pl is not None else 4.0
    )

    for _, group in scored.groupby("question_id", sort=False):
        m = len(group)
        if m < 2:
            continue
        idx = group.index.to_numpy()
        z = group["contaminated_reward_z"].to_numpy(dtype=float)
        p_dual = group["dualeval_p"].to_numpy(dtype=float)
        p_arena2pl = group["arena2pl_p"].to_numpy(dtype=float)
        bt = group["bt_theta"].to_numpy(dtype=float)

        # Antisymmetric M x M residual matrices; diagonals are 0 because
        # sigmoid(0) - sigmoid(0) = 0. Row sums give per-model residual sums.
        T = sigmoid_np(z[:, None] - z[None, :])
        P_dual = sigmoid_np(gamma_dual * (p_dual[:, None] - p_dual[None, :]))
        P_arena2pl = sigmoid_np(gamma_arena2pl * (p_arena2pl[:, None] - p_arena2pl[None, :]))
        P_bt = sigmoid_np(bt[:, None] - bt[None, :])
        dual_sums[idx] += (T - P_dual).sum(axis=1)
        arena2pl_sums[idx] += (T - P_arena2pl).sum(axis=1)
        bt_sums[idx] += (T - P_bt).sum(axis=1)
        counts[idx] += float(m - 1)

    counts = np.maximum(counts, 1.0)
    scored["score_dualeval_arena_residual"] = dual_sums / counts
    scored["score_arena2pl_arena_residual"] = arena2pl_sums / counts
    scored["score_bt_arena_residual"] = bt_sums / counts

    injected_keys = (
        pd.MultiIndex.from_frame(injected[["model_name", "question_id"]])
        if not injected.empty
        else pd.MultiIndex.from_arrays([[], []])
    )
    scored_keys = pd.MultiIndex.from_frame(scored[["model_name", "question_id"]])
    scored["synthetic_contaminated"] = scored_keys.isin(injected_keys)
    scored["target"] = "arena"
    scored["score_primary"] = scored["score_dualeval_arena_residual"]
    return scored


def fit_dualeval_both(
    static_df: pd.DataFrame,
    pairwise_df: pd.DataFrame | None,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return dualeval.fit_irt(
        static_df,
        pairwise_df if pairwise_df is not None and not pairwise_df.empty else None,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_static=args.lambda_static,
        lambda_arena=args.lambda_arena,
        lambda_bb=args.lambda_bb,
        reg_lambda=args.reg_lambda,
        verbose=not args.quiet,
    )


def fit_static_2pl(
    static_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return dualeval.fit_irt(
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


def fit_arena_bt(
    pairwise_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if pairwise_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"learned_gamma": None, "n_models": 0, "n_questions": 0}
    return dualeval.fit_bt(
        pairwise_df,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_arena=args.lambda_arena,
        reg_lambda=args.reg_lambda,
        verbose=not args.quiet,
    )


def fit_arena_only_2pl(
    pairwise_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit a 2PL IRT model using only pairwise arena data (no static labels).

    Single-modality counterpart to the joint DualEval fit. Item parameters
    (b_q, k_q) are identified by the both-bad anchoring term; gamma is learned
    from the arena BCE loss as in fit_irt.
    """
    if pairwise_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"learned_gamma": None, "n_models": 0, "n_questions": 0}
    return dualeval.fit_irt(
        None,
        pairwise_df,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_static=0.0,
        lambda_arena=args.lambda_arena,
        lambda_bb=args.lambda_bb,
        reg_lambda=args.reg_lambda,
        verbose=not args.quiet,
    )


def ranking_shift(
    clean_models: pd.DataFrame,
    contaminated_models: pd.DataFrame,
) -> pd.DataFrame:
    clean = clean_models.sort_values("theta", ascending=False).reset_index(drop=True).copy()
    contam = contaminated_models.sort_values("theta", ascending=False).reset_index(drop=True).copy()
    clean["clean_rank"] = np.arange(1, len(clean) + 1)
    contam["contaminated_rank"] = np.arange(1, len(contam) + 1)
    merged = clean[["model_name", "theta", "clean_rank"]].merge(
        contam[["model_name", "theta", "contaminated_rank"]],
        on="model_name",
        suffixes=("_clean", "_contaminated"),
    )
    merged["theta_shift"] = merged["theta_contaminated"] - merged["theta_clean"]
    merged["rank_shift"] = merged["clean_rank"] - merged["contaminated_rank"]
    return merged.sort_values("theta_shift", ascending=False).reset_index(drop=True)


def run_one_trial(
    *,
    static_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    clean_dualeval_models: pd.DataFrame,
    static_candidates: pd.DataFrame,
    arena_candidates: pd.DataFrame,
    injection_rate: float,
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored_frames: list[pd.DataFrame] = []
    contaminated_static = static_df.copy()
    contaminated_reward = reward_df.copy()
    injected_static = static_candidates.head(0).copy()
    injected_arena = arena_candidates.head(0).copy()

    if args.injection_target == "static":
        contaminated_static, injected_static = inject_static_contamination(
            static_df,
            static_candidates,
            injection_rate=injection_rate,
            seed=seed,
            denominator=args.rate_denominator,
        )

    contaminated_pairwise = pairwise_df
    if args.injection_target == "arena":
        contaminated_reward, injected_arena = inject_arena_contamination(
            reward_df,
            arena_candidates,
            injection_rate=injection_rate,
            seed=seed + 104729,
            denominator=args.rate_denominator,
            reward_boost=args.arena_reward_boost,
        )
        bb_threshold, tie_delta = dualeval.resolve_pairwise_thresholds(
            contaminated_reward,
            bb_ratio=args.bb_ratio,
            tie_ratio=args.tie_ratio,
        )
        contaminated_pairwise = dualeval.build_soft_pairwise_targets(
            contaminated_reward,
            both_bad_threshold=bb_threshold,
            tie_delta=tie_delta,
        )

    contam_static_clean = contaminated_static.drop(columns=["synthetic_contaminated"], errors="ignore")
    contaminated_models, contaminated_questions, contaminated_meta = fit_dualeval_both(
        contam_static_clean,
        contaminated_pairwise,
        args,
    )
    shifts = ranking_shift(clean_dualeval_models, contaminated_models)
    shifts.insert(0, "seed", int(seed))
    shifts.insert(1, "injection_rate", float(injection_rate))

    rows: list[dict[str, Any]] = []

    if args.injection_target == "static":
        contam_static_models, contam_static_questions, _ = fit_static_2pl(
            contam_static_clean,
            args,
        )
        contam_dualeval_scored_static = add_irt_predictions(
            contam_static_clean,
            contaminated_models,
            contaminated_questions,
            prefix="dualeval",
        )
        contam_static2pl_scored = add_irt_predictions(
            contam_static_clean,
            contam_static_models,
            contam_static_questions,
            prefix="static2pl",
        )
        static_scored = score_contaminated_static(
            injected_static,
            contam_dualeval_scored_static,
            contam_static2pl_scored,
        )
        static_scored.insert(0, "seed", int(seed))
        static_scored.insert(1, "injection_rate", float(injection_rate))
        scored_frames.append(static_scored)

        method_to_col = {
            "DualEval residual": "score_dualeval_residual",
            "Static 2PL residual": "score_static2pl_residual",
        }
        for method, score_col in method_to_col.items():
            row = evaluate_scores(static_scored, method=method, score_col=score_col)
            row.update(evaluate_item_scores(static_scored, method=method, score_col=score_col))
            row.update(
                {
                    "target": "static",
                    "seed": int(seed),
                    "injection_rate": float(injection_rate),
                    "n_rows": int(len(static_df)),
                    "n_candidates": int(len(static_candidates)),
                    "n_injected_cells": int(len(injected_static)),
                    "n_injected_items": int(injected_static["question_id"].nunique()) if not injected_static.empty else 0,
                    "contaminated_learned_gamma": contaminated_meta.get("learned_gamma"),
                }
            )
            rows.append(row)

    if args.injection_target == "arena":
        contam_bt_models, _, _contam_bt_meta = fit_arena_bt(contaminated_pairwise, args)
        contam_arena2pl_models, contam_arena2pl_questions, contam_arena2pl_meta = fit_arena_only_2pl(
            contaminated_pairwise, args
        )
        contam_dualeval_arena_scored = add_irt_predictions(
            contaminated_reward,
            contaminated_models,
            contaminated_questions,
            prefix="dualeval",
        )
        contam_arena2pl_scored = add_irt_predictions(
            contaminated_reward,
            contam_arena2pl_models,
            contam_arena2pl_questions,
            prefix="arena2pl",
        )
        arena_scored = score_contaminated_arena(
            contaminated_reward,
            injected_arena,
            contam_dualeval_arena_scored,
            contam_arena2pl_scored,
            contam_bt_models,
            learned_gamma_dual=contaminated_meta.get("learned_gamma"),
            learned_gamma_arena2pl=contam_arena2pl_meta.get("learned_gamma"),
        )
        arena_scored.insert(0, "seed", int(seed))
        arena_scored.insert(1, "injection_rate", float(injection_rate))
        scored_frames.append(arena_scored)

        method_to_col = {
            "DualEval arena residual": "score_dualeval_arena_residual",
            "Arena-only 2PL residual": "score_arena2pl_arena_residual",
            "BT arena residual": "score_bt_arena_residual",
        }
        for method, score_col in method_to_col.items():
            row = evaluate_scores(arena_scored, method=method, score_col=score_col)
            row.update(evaluate_item_scores(arena_scored, method=method, score_col=score_col))
            row.update(
                {
                    "target": "arena",
                    "seed": int(seed),
                    "injection_rate": float(injection_rate),
                    "n_rows": int(len(reward_df)),
                    "n_candidates": int(len(arena_candidates)),
                    "n_injected_cells": int(len(injected_arena)),
                    "n_injected_items": int(injected_arena["question_id"].nunique()) if not injected_arena.empty else 0,
                    "contaminated_learned_gamma": contaminated_meta.get("learned_gamma"),
                }
            )
            rows.append(row)

    metrics = pd.DataFrame(rows)
    scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    return metrics, scored, shifts


def summarise_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "cell_auroc",
        "cell_auprc",
        "cell_precision_at_k",
        "cell_recall_at_k",
        "item_auroc",
        "item_auprc",
        "item_precision_at_k",
        "item_recall_at_k",
    ]
    agg: dict[str, tuple[str, str]] = {}
    for col in metric_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")
    agg["n_trials"] = ("seed", "nunique")
    agg["n_injected_cells_mean"] = ("n_injected_cells", "mean")
    agg["n_injected_items_mean"] = ("n_injected_items", "mean")
    return (
        metrics.groupby(["target", "injection_rate", "method"], as_index=False)
        .agg(**agg)
        .sort_values(["target", "injection_rate", "method"])
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="analysis/config_synthetic_recovery.yaml")
    parser.add_argument("--static-jsonl", nargs="*", default=None)
    parser.add_argument("--arena-reward-jsonl", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--injection-rates", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--rate-denominator", choices=["all", "candidates"], default=None)
    parser.add_argument("--injection-target", choices=["static", "arena"], default=None)
    parser.add_argument("--candidate-pool-fraction", type=float, default=None)
    parser.add_argument("--arena-reward-boost", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-static", type=float, default=None)
    parser.add_argument("--lambda-arena", type=float, default=None)
    parser.add_argument("--lambda-bb", type=float, default=None)
    parser.add_argument("--reg-lambda", type=float, default=None)
    parser.add_argument("--bb-ratio", type=float, default=None)
    parser.add_argument("--tie-ratio", type=float, default=None)
    parser.add_argument("--top-cells-to-save", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return load_config_defaults(parser.parse_args())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.static_jsonl:
        raise SystemExit("Need static input. Provide --static-jsonl or config input.static_jsonl.")
    static_df = dualeval.load_static_jsonl(args.static_jsonl)
    if static_df.empty:
        raise SystemExit("No usable static rows loaded.")

    reward_df = (
        dualeval.load_arena_reward_jsonl(args.arena_reward_jsonl)
        if args.arena_reward_jsonl
        else pd.DataFrame()
    )
    pairwise_df = pd.DataFrame()
    if not reward_df.empty:
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

    if not args.quiet:
        print(f"Loaded static rows: {len(static_df)} from {len(args.static_jsonl)} file(s)")
        print(f"Loaded arena reward rows: {len(reward_df)}; pairwise rows: {len(pairwise_df)}")
        print("\nFitting clean DualEval reference...")
    clean_dualeval_models, clean_dualeval_questions, clean_dualeval_meta = fit_dualeval_both(static_df, pairwise_df, args)

    if not args.quiet:
        print("\nFitting clean static-only 2PL reference...")
    clean_static_models, clean_static_questions, clean_static_meta = fit_static_2pl(static_df, args)

    clean_bt_models = pd.DataFrame()
    clean_bt_meta: dict[str, Any] = {"learned_gamma": None, "n_models": 0, "n_questions": 0}
    if args.injection_target == "arena":
        if reward_df.empty or pairwise_df.empty:
            raise SystemExit("Arena injection requires arena_reward_jsonl data.")
        if not args.quiet:
            print("\nFitting clean arena BT reference...")
        clean_bt_models, _clean_bt_questions, clean_bt_meta = fit_arena_bt(pairwise_df, args)

    static_candidates = select_injection_candidates(
        static_df,
        candidate_pool_fraction=args.candidate_pool_fraction,
    )
    arena_candidates = (
        select_arena_injection_candidates(
            reward_df,
            candidate_pool_fraction=args.candidate_pool_fraction,
        )
        if args.injection_target == "arena" and not reward_df.empty
        else pd.DataFrame()
    )

    if args.injection_target == "static" and static_candidates.empty:
        raise SystemExit(
            "No static injection candidates found. Try raising --candidate-pool-fraction."
        )
    if args.injection_target == "arena" and arena_candidates.empty:
        raise SystemExit(
            "No arena injection candidates found. Try raising --candidate-pool-fraction."
        )

    clean_dualeval_models.to_csv(args.output_dir / "clean_dualeval_model_ranking.csv", index=False)
    clean_dualeval_questions.to_csv(args.output_dir / "clean_dualeval_question_ranking.csv", index=False)
    clean_static_models.to_csv(args.output_dir / "clean_static2pl_model_ranking.csv", index=False)
    clean_static_questions.to_csv(args.output_dir / "clean_static2pl_question_ranking.csv", index=False)
    static_candidates.to_csv(args.output_dir / "static_injection_candidates.csv", index=False)
    if not arena_candidates.empty:
        arena_candidates.to_csv(args.output_dir / "arena_injection_candidates.csv", index=False)
    if not clean_bt_models.empty:
        clean_bt_models.to_csv(args.output_dir / "clean_arena_bt_model_ranking.csv", index=False)

    if not args.quiet:
        if args.injection_target == "static":
            print(
                f"\nStatic injection candidates: {len(static_candidates)} "
                f"({len(static_candidates) / len(static_df):.2%} of static rows)"
            )
        if args.injection_target == "arena":
            print(
                f"Arena injection candidates: {len(arena_candidates)} "
                f"({len(arena_candidates) / len(reward_df):.2%} of arena reward rows)"
            )

    all_metrics: list[pd.DataFrame] = []
    top_cell_frames: dict[str, list[pd.DataFrame]] = {}
    all_shift_frames: list[pd.DataFrame] = []

    for injection_rate in args.injection_rates:
        for seed in args.seeds:
            if not args.quiet:
                print(f"\nTrial injection_rate={injection_rate:.4f}, seed={seed}")
            metrics, scored, shifts = run_one_trial(
                static_df=static_df,
                reward_df=reward_df,
                pairwise_df=pairwise_df,
                clean_dualeval_models=clean_dualeval_models,
                static_candidates=static_candidates,
                arena_candidates=arena_candidates,
                injection_rate=injection_rate,
                seed=seed,
                args=args,
            )
            all_metrics.append(metrics)
            all_shift_frames.append(shifts)

            if args.top_cells_to_save > 0 and not scored.empty:
                # score_primary lives in different scales across targets
                # (static cell-level residual vs. arena pairwise residual),
                # so save the top-K per target rather than concatenated.
                for target_name, target_df in scored.groupby("target", sort=False):
                    top = target_df.sort_values("score_primary", ascending=False).head(args.top_cells_to_save)
                    top_cell_frames.setdefault(str(target_name), []).append(top)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = summarise_metrics(metrics_df)
    shifts_df = pd.concat(all_shift_frames, ignore_index=True)

    metrics_df.to_csv(args.output_dir / "synthetic_recovery_metrics_raw.csv", index=False)
    summary_df.to_csv(args.output_dir / "synthetic_recovery_metrics_summary.csv", index=False)
    shifts_df.to_csv(args.output_dir / "contaminated_fit_model_shifts.csv", index=False)
    for target_name, frames in top_cell_frames.items():
        pd.concat(frames, ignore_index=True).to_csv(
            args.output_dir / f"top_dualeval_residual_cells_{target_name}.csv",
            index=False,
        )

    summary = {
        "config": args.config,
        "static_jsonl": args.static_jsonl,
        "arena_reward_jsonl": args.arena_reward_jsonl,
        "output_dir": str(args.output_dir),
        "injection_rates": args.injection_rates,
        "seeds": args.seeds,
        "rate_denominator": args.rate_denominator,
        "injection_target": args.injection_target,
        "candidate_pool_fraction": args.candidate_pool_fraction,
        "arena_reward_boost": args.arena_reward_boost,
        "n_static_rows": int(len(static_df)),
        "n_static_questions": int(static_df["question_id"].nunique()),
        "n_models": int(static_df["model_name"].nunique()),
        "n_arena_reward_rows": int(len(reward_df)),
        "n_pairwise_rows": int(len(pairwise_df)),
        "n_static_injection_candidates": int(len(static_candidates)),
        "n_arena_injection_candidates": int(len(arena_candidates)),
        "clean_dualeval_meta": clean_dualeval_meta,
        "clean_static2pl_meta": clean_static_meta,
        "clean_arena_bt_meta": clean_bt_meta,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSynthetic recovery summary:")
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: "nan" if math.isnan(x) else f"{x:.4f}",
        )
    )
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
