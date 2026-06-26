#!/usr/bin/env python3
"""Held-out calibration of DualEval predictions (Appendix).

We use a (model, item)-CELL-level 80/20 random holdout rather than the
question-level split used in baseline_comparison. This is required because
calibration measures p_{i,q} per cell, which depends on item-specific
parameters a_q, b_q. A question-level split would remove those entirely from
the fit, leaving no way to predict held-out cells. Cell-level holdout keeps
each item in training (typically ~14 of its 18 cells) while still evaluating
predictions on responses the fit never saw -- the standard IRT cross-validation
setup for out-of-sample calibration.

For each (domain, seed) pair:
  - Random 80/20 cell-level split of static rows and pairwise rows (same RNG seed
    used per split type).
  - Fit DualEval (joint static + arena) on the 80% train split.
  - On the held-out 20%:
      * Static: predict p_{i,q} = sigmoid(a_q(theta_i - b_q)),
        compare against observed binary judge_result.
      * Arena: predict mu_{ijq} = sigmoid(gamma (p_{i,q} - p_{j,q})) on
        hard pairs (non-tie, non-both-bad), compare against the binary
        target_prob >= 0.5 outcome.
  - Build 10-bin reliability diagrams and compute Expected Calibration Error.

Outputs CSVs in results/calibration/:
  - calibration_ece_summary.csv         : per (domain, seed) ECE for static/arena
  - calibration_static_bins.csv         : per-bin pred / observed / count for static
  - calibration_arena_bins.csv          : per-bin pred / observed / count for arena
  - calibration_static_held_out.csv     : per-(model, item) predictions & labels
  - calibration_arena_held_out.csv      : per-(model_1, model_2, item) predictions & targets

Usage:
  python analysis/calibration.py
  THREADS=16 python analysis/calibration.py
  python analysis/calibration.py --domains coding --seeds 0
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DUALEVAL_PATH = REPO_ROOT / "ranking" / "dualeval.py"
_spec = importlib.util.spec_from_file_location("dualeval_module", DUALEVAL_PATH)
dualeval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dualeval)


DOMAIN_INPUTS = {
    "coding": (
        "results/static_eval/coding/responses.jsonl",
        "results/arena_eval/coding/responses.jsonl",
    ),
    "math": (
        "results/static_eval/math/responses.jsonl",
        "results/arena_eval/math/responses.jsonl",
    ),
}

DEFAULT_SEEDS = [0, 1, 2]      # matches the three random splits in Table 2
DEFAULT_TEST_FRACTION = 0.2
N_BINS = 10                    # reliability diagram bins (deciles)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -709.0, 709.0)))


def split_cells(
    df: pd.DataFrame,
    *,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cell-level (row-level) random 80/20 split.

    Each item still appears in training -- only specific (model, item) responses
    (or specific arena pair rows) are held out. This preserves item parameters
    in the IRT fit so held-out cells can be predicted.
    """
    if df.empty or test_fraction <= 0.0:
        return df.copy(), df.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_fraction
    if not mask.any() and len(df) > 1:
        mask[rng.integers(0, len(df))] = True
    if mask.all() and len(df) > 1:
        mask[rng.integers(0, len(df))] = False
    return (
        df.loc[~mask].reset_index(drop=True),
        df.loc[mask].reset_index(drop=True),
    )


def predict_static_p(
    eval_df: pd.DataFrame,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
) -> pd.DataFrame:
    theta = model_params.set_index("model_name")["theta"]
    qp = question_params.set_index("question_id")[["difficulty_b", "discrimination_exp_k"]]
    out = eval_df[["model_name", "question_id", "judge_result"]].copy()
    out["theta"] = out["model_name"].map(theta)
    out["b"] = out["question_id"].map(qp["difficulty_b"])
    out["a"] = out["question_id"].map(qp["discrimination_exp_k"])
    out = out.dropna(subset=["theta", "b", "a", "judge_result"]).reset_index(drop=True)
    logits = out["a"].to_numpy(dtype=float) * (
        out["theta"].to_numpy(dtype=float) - out["b"].to_numpy(dtype=float)
    )
    out["p_pred"] = sigmoid_np(logits)
    out["y_obs"] = out["judge_result"].astype(float)
    return out[["model_name", "question_id", "p_pred", "y_obs"]]


def predict_arena_mu(
    eval_pairwise: pd.DataFrame,
    model_params: pd.DataFrame,
    question_params: pd.DataFrame,
    gamma: float,
) -> pd.DataFrame:
    theta = model_params.set_index("model_name")["theta"]
    qp = question_params.set_index("question_id")[["difficulty_b", "discrimination_exp_k"]]
    out = eval_pairwise[
        ["model_1", "model_2", "question_id", "target_prob", "tie", "both_bad"]
    ].copy()
    out["theta_1"] = out["model_1"].map(theta)
    out["theta_2"] = out["model_2"].map(theta)
    out["b"] = out["question_id"].map(qp["difficulty_b"])
    out["a"] = out["question_id"].map(qp["discrimination_exp_k"])
    out = out.dropna(subset=["theta_1", "theta_2", "b", "a", "target_prob"]).reset_index(drop=True)
    a = out["a"].to_numpy(dtype=float)
    b = out["b"].to_numpy(dtype=float)
    t1 = out["theta_1"].to_numpy(dtype=float)
    t2 = out["theta_2"].to_numpy(dtype=float)
    p1 = sigmoid_np(a * (t1 - b))
    p2 = sigmoid_np(a * (t2 - b))
    out["mu_pred"] = sigmoid_np(gamma * (p1 - p2))
    # Primary outcome: continuous soft preference target sigma(z_i - z_j).
    # This is the ground-truth quantity the joint fit predicts; comparing
    # against the binarised 1[target_prob >= 0.5] outcome introduces a
    # binarisation artifact (systematic under-confidence at high mu_pred
    # because moderately-confident predictions collapse to 1 in the binary
    # outcome distribution but stay soft in mu_pred).
    out["y_obs"] = out["target_prob"].to_numpy(dtype=float)
    # Secondary binarised outcome retained for completeness in the summary.
    out["y_obs_binary"] = (out["target_prob"].to_numpy(dtype=float) >= 0.5).astype(float)
    return out[[
        "model_1", "model_2", "question_id", "mu_pred", "y_obs", "y_obs_binary",
        "target_prob", "tie", "both_bad",
    ]]


def reliability_bins(
    predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = N_BINS,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(predictions, edges) - 1, 0, n_bins - 1)
    df = pd.DataFrame({"pred": predictions, "obs": outcomes, "bin": bin_idx})
    g = (
        df.groupby("bin", as_index=False)
        .agg(mean_pred=("pred", "mean"), mean_obs=("obs", "mean"), n=("obs", "size"))
    )
    # Ensure every bin index appears (NaN for empty bins)
    full = pd.DataFrame({"bin": np.arange(n_bins)})
    return full.merge(g, on="bin", how="left").fillna({"n": 0})


def ece(predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = N_BINS) -> float:
    bins = reliability_bins(predictions, outcomes, n_bins)
    bins = bins.dropna(subset=["mean_pred", "mean_obs"])
    if bins.empty:
        return float("nan")
    total = float(bins["n"].sum())
    if total == 0:
        return float("nan")
    return float(
        (bins["n"].to_numpy(dtype=float) / total
         * np.abs(bins["mean_pred"].to_numpy(dtype=float)
                  - bins["mean_obs"].to_numpy(dtype=float))).sum()
    )


def fit_joint(
    static_train: pd.DataFrame,
    pairwise_train: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return dualeval.fit_irt(
        static_train,
        pairwise_train if pairwise_train is not None and not pairwise_train.empty else None,
        num_epochs=args.num_epochs, lr=args.lr,
        lambda_static=args.lambda_static, lambda_arena=args.lambda_arena,
        lambda_bb=args.lambda_bb, reg_lambda=args.reg_lambda,
        verbose=False,
    )


def run_domain(domain: str, args: argparse.Namespace, seeds: list[int]):
    static_path, arena_path = DOMAIN_INPUTS[domain]
    static_df = dualeval.load_static_jsonl([str(REPO_ROOT / static_path)])
    reward_df = dualeval.load_arena_reward_jsonl([str(REPO_ROOT / arena_path)])
    bb_threshold, tie_delta = dualeval.resolve_pairwise_thresholds(
        reward_df, bb_ratio=args.bb_ratio, tie_ratio=args.tie_ratio,
    )
    pairwise_df = dualeval.build_soft_pairwise_targets(
        reward_df, both_bad_threshold=bb_threshold, tie_delta=tie_delta,
    )

    static_preds_all: list[pd.DataFrame] = []
    arena_preds_all: list[pd.DataFrame] = []
    static_bins_all: list[pd.DataFrame] = []
    arena_bins_all: list[pd.DataFrame] = []
    ece_rows: list[dict] = []

    for seed in seeds:
        t0 = time.time()
        static_train, static_test = split_cells(
            static_df, test_fraction=args.test_fraction, seed=seed
        )
        pairwise_train, pairwise_test = split_cells(
            pairwise_df, test_fraction=args.test_fraction, seed=seed
        )

        model_params, question_params, meta = fit_joint(
            static_train, pairwise_train, args
        )
        gamma = float(meta.get("learned_gamma") or 4.0)

        # Static held-out
        static_preds = predict_static_p(static_test, model_params, question_params)
        static_preds["seed"] = seed
        static_preds["domain"] = domain
        static_preds_all.append(static_preds)
        sbins = reliability_bins(
            static_preds["p_pred"].to_numpy(),
            static_preds["y_obs"].to_numpy(),
        )
        sbins["seed"] = seed; sbins["domain"] = domain
        static_bins_all.append(sbins)
        static_ece_v = ece(static_preds["p_pred"].to_numpy(), static_preds["y_obs"].to_numpy())

        # Arena held-out (hard pairs only — match Table 2 Pair Acc definition)
        pairwise_test_hard = pairwise_test[
            ~pairwise_test["tie"].astype(bool)
            & ~pairwise_test["both_bad"].astype(bool)
        ]
        arena_preds = predict_arena_mu(
            pairwise_test_hard, model_params, question_params, gamma=gamma,
        )
        arena_preds["seed"] = seed
        arena_preds["domain"] = domain
        arena_preds_all.append(arena_preds)
        abins = reliability_bins(
            arena_preds["mu_pred"].to_numpy(),
            arena_preds["y_obs"].to_numpy(),
        )
        abins["seed"] = seed; abins["domain"] = domain
        arena_bins_all.append(abins)
        # Primary arena ECE: predicted mu_{ijq} vs soft target sigma(z_i - z_j)
        arena_ece_v = ece(
            arena_preds["mu_pred"].to_numpy(), arena_preds["y_obs"].to_numpy()
        )
        # Secondary: ECE against binarised target, for transparency
        arena_ece_binary_v = ece(
            arena_preds["mu_pred"].to_numpy(), arena_preds["y_obs_binary"].to_numpy()
        )

        ece_rows.append({
            "domain": domain,
            "seed": int(seed),
            "static_ece": static_ece_v,
            "arena_ece": arena_ece_v,
            "arena_ece_binary": arena_ece_binary_v,
            "n_static_holdout": int(len(static_preds)),
            "n_arena_holdout": int(len(arena_preds)),
            "learned_gamma": gamma,
            "fit_seconds": time.time() - t0,
        })
        print(
            f"  {domain} seed={seed}: static ECE={static_ece_v:.4f}, "
            f"arena ECE={arena_ece_v:.4f} (soft) / {arena_ece_binary_v:.4f} (binary), "
            f"fit {time.time() - t0:.1f}s",
            flush=True,
        )

    return (
        pd.concat(static_preds_all, ignore_index=True),
        pd.concat(arena_preds_all, ignore_index=True),
        pd.concat(static_bins_all, ignore_index=True),
        pd.concat(arena_bins_all, ignore_index=True),
        pd.DataFrame(ece_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results" / "calibration")
    parser.add_argument("--domains", nargs="*", default=list(DOMAIN_INPUTS.keys()))
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lambda-static", type=float, default=1.0)
    parser.add_argument("--lambda-arena", type=float, default=1.0)
    parser.add_argument("--lambda-bb", type=float, default=0.2)
    parser.add_argument("--reg-lambda", type=float, default=0.01)
    parser.add_argument("--bb-ratio", type=float, default=0.15)
    parser.add_argument("--tie-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()
    static_preds_all: list[pd.DataFrame] = []
    arena_preds_all: list[pd.DataFrame] = []
    static_bins_all: list[pd.DataFrame] = []
    arena_bins_all: list[pd.DataFrame] = []
    ece_all: list[pd.DataFrame] = []

    for domain in args.domains:
        print(f"\n=== {domain} ===", flush=True)
        sp, ap, sb, ab, ec = run_domain(domain, args, args.seeds)
        static_preds_all.append(sp); arena_preds_all.append(ap)
        static_bins_all.append(sb); arena_bins_all.append(ab)
        ece_all.append(ec)
        print(
            f"  {domain} summary: static_ece={ec['static_ece'].mean():.4f}"
            f"±{ec['static_ece'].std():.4f}, arena_ece={ec['arena_ece'].mean():.4f}"
            f"±{ec['arena_ece'].std():.4f} (soft), "
            f"arena_ece_binary={ec['arena_ece_binary'].mean():.4f}"
            f"±{ec['arena_ece_binary'].std():.4f}",
            flush=True,
        )

    pd.concat(static_preds_all, ignore_index=True).to_csv(
        args.output_dir / "calibration_static_held_out.csv", index=False)
    pd.concat(arena_preds_all, ignore_index=True).to_csv(
        args.output_dir / "calibration_arena_held_out.csv", index=False)
    pd.concat(static_bins_all, ignore_index=True).to_csv(
        args.output_dir / "calibration_static_bins.csv", index=False)
    pd.concat(arena_bins_all, ignore_index=True).to_csv(
        args.output_dir / "calibration_arena_bins.csv", index=False)
    pd.concat(ece_all, ignore_index=True).to_csv(
        args.output_dir / "calibration_ece_summary.csv", index=False)

    print(
        f"\nDone in {time.time() - overall_start:.1f}s. Results in {args.output_dir}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
