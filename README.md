# DualEval

DualEval is an open-source Python package for ranking LLMs using **2-parameter Item Response Theory (IRT)**. Given binary correctness labels from static benchmarks and/or continuous reward scores from arena-style evaluations, DualEval jointly infers per-model ability, per-question difficulty, and per-question discrimination — producing a richer ranking than simple accuracy averaging.

If you already have evaluation results for your models, you can rank them in minutes. The full paper pipeline that produced those results is also reproducible end-to-end; see [Reproducing the Paper](#reproducing-the-paper).

> This is the official codebase for our paper: *DualEval: Jointly Ranking LLMs with Static Benchmarks and Arena Reward Signals via Item Response Theory*.

---

## Install

```bash
git clone https://github.com/aaron-jx-li/DualEval.git
cd DualEval
pip install -r requirements.txt
```

---

## Ranking Your Models

### Input format

**Static JSONL** — one JSON object per line, one row per `(model, question)` pair:

```json
{"model_label": "gpt-4o",          "dataset": "gsm8k", "sample_index": 0, "correct": 1}
{"model_label": "gpt-4o",          "dataset": "gsm8k", "sample_index": 1, "correct": 0}
{"model_label": "claude-3-5-sonnet","dataset": "gsm8k", "sample_index": 0, "correct": 1}
```

| Field | Type | Description |
|---|---|---|
| `model_label` | string | Unique model identifier |
| `dataset` | string | Benchmark name (e.g. `"gsm8k"`, `"humaneval"`) |
| `sample_index` | int | Question index within the dataset |
| `correct` | 0 or 1 | 1 if the model answered correctly |

Rows missing `model_label` or `correct` are skipped.

**Arena reward JSONL** — one JSON object per line, one row per `(model, prompt)` pair:

```json
{"model_label": "gpt-4o",          "item_id": "arena_math_0042", "reward": 3.71}
{"model_label": "gpt-4o",          "item_id": "arena_math_0043", "reward": 1.22}
{"model_label": "claude-3-5-sonnet","item_id": "arena_math_0042", "reward": 4.05}
```

| Field | Type | Description |
|---|---|---|
| `model_label` | string | Unique model identifier |
| `item_id` | string | Unique prompt identifier |
| `reward` | float | Reward model score for this `(model, prompt)` pair |

Rows missing `reward` are skipped. Rewards are z-score normalised across all rows before training. Multiple files can be concatenated — each file gets a source tag derived from its path, so `question_id` values remain unique across files.

### Config-driven ranking

Copy and edit `dualeval/config_dualeval.yaml` to point to your data files and set your preferred lambda hyperparameters:

```yaml
input:
  static_jsonl:
    - my_results/static/math.jsonl
    - my_results/static/coding.jsonl
  arena_reward_jsonl:
    - my_results/arena_rewards/math.jsonl
    - my_results/arena_rewards/coding.jsonl

training:
  mode: both          # static | arena | both | BT
  lambda_static: 1.0  # weight on binary correctness loss
  lambda_arena: 1.0   # weight on soft-pairwise reward loss
  lambda_bb: 0.2      # both-bad anchoring term
  reg_lambda: 0.001   # L2 regularisation

output:
  output_dir: my_results/dualeval_run
  no_plot: true
```

See `dualeval/config_dualeval.yaml` for a complete reference with all fields and inline documentation.

Then run:

```bash
python dualeval/dualeval.py --config my_config.yaml
```

Outputs `model_ranking.csv` (sorted θ), `question_ranking.csv` (per-question difficulty and discrimination), and `metrics.json`.

### Programmatic API

```python
from dualeval import fit_irt, load_static_jsonl, load_arena_reward_jsonl

static_df = load_static_jsonl(["my_results/static/math.jsonl"])
arena_df  = load_arena_reward_jsonl(["my_results/arena_rewards/math.jsonl"])

results = fit_irt(static_df=static_df, arena_df=arena_df, mode="both",
                  lambda_static=1.0, lambda_arena=1.0, lambda_bb=0.2)
print(results["model_ranking"])
```

### Fitting modes

| Mode | Signal | Description |
|------|--------|-------------|
| `static` | Binary correctness | 2PL-IRT on static benchmark labels only |
| `arena` | Continuous reward | Soft-pairwise distillation IRT on arena rewards |
| `both` | Both | Joint fitting on shared (θ, b, a) parameters |
| `BT` | Continuous reward | Bradley-Terry baseline (no question parameters) |

---

## Reproducing the Paper

The `demo/` directory contains the sampling scripts, evaluation configs, and a shell script that reproduces the paper's full experimental pipeline using the public Skywork reward model.

### Setup

Copy `.env.example` to `.env` and fill in your API keys:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...       # for Gemini models
OPENROUTER_API_KEY=...   # optional, for OpenRouter routing
```

### Running the pipeline

```bash
bash demo/run_pipeline.sh
```

This runs six sequential stages:

| Stage | Script | Output |
|-------|--------|--------|
| 1. Sample static questions | `demo/sample_static.py` | `results/static_eval/<domain>/sampled_items.jsonl` |
| 2. Evaluate static benchmarks | `eval/eval_static.py` | `results/static_eval/<domain>/responses.jsonl` |
| 3. Sample arena prompts | `demo/sample_arena.py` | `data/arena_*.jsonl` |
| 4. Generate arena responses | `eval/eval_arena.py` | `results/arena_eval/<domain>/responses.jsonl` |
| 5. Score with public Skywork RM | `reward/public_reward.py` | `results/public_reward/<domain>/responses.jsonl` |
| 6. Fit IRT model | `dualeval/dualeval.py` | `results/dualeval/my_run/` |

To evaluate a subset of models, edit the `models` lists in `demo/config_arena.yaml` and `demo/config_static.yaml`.

**Note on our reward model:** The paper uses a proprietary internal reward model. We are not releasing this reward model or its outputs. The public Skywork-Reward-V2-Qwen3-8B model is provided as a drop-in alternative that follows the same interface. A CUDA-capable GPU is required; see `reward/public_reward.py --help` for device and batch-size options.

### Individual steps

You can also run each stage manually:

```bash
# Static benchmarks
python demo/sample_static.py --domain math --config demo/config_static.yaml
python eval/eval_static.py   --domain math --config demo/config_static.yaml

python demo/sample_static.py --domain misc --config demo/config_static.yaml
python eval/eval_static.py   --domain misc --config demo/config_static.yaml

python eval/eval_static.py --domain coding --config demo/config_static_coding.yaml

# Arena
python demo/sample_arena.py --domain math    --config demo/config_arena.yaml
python demo/sample_arena.py --domain coding  --config demo/config_arena.yaml
python demo/sample_arena.py --domain generic --config demo/config_arena.yaml

python eval/eval_arena.py --domain math    --config demo/config_arena.yaml
python eval/eval_arena.py --domain coding  --config demo/config_arena.yaml
python eval/eval_arena.py --domain misc    --config demo/config_arena.yaml
python eval/eval_arena.py --domain generic --config demo/config_arena.yaml

# Reward scoring
python reward/public_reward.py \
    --arena-dir results/arena_eval \
    --output-dir results/public_reward

# IRT ranking
python dualeval/dualeval.py --config dualeval/config_dualeval.yaml
```

### Analysis and ablations

```bash
# §4.1 — DualEval vs. Bradley-Terry and static-only baselines
python analysis/baseline_comparison.py --config dualeval/config_dualeval.yaml \
    --output-dir results/baseline_comparison

# §4.1 — Bootstrap ranking stability (Table 4)
python analysis/ranking_bootstrap.py -B 100 --workers 16

# §4.2 — Item informativeness: discrimination-weighted question selection
python analysis/item_informativeness.py --runs-root results/dualeval \
    --output-dir results/item_informativeness

# §4.3 — Residual-based anomaly detection (synthetic contamination)
python analysis/anomaly_detection.py --config analysis/config_synthetic_recovery.yaml

# Appendix — Held-out calibration and ECE
python analysis/calibration.py
```

---

## Repository Layout

```
dualeval/                   Core IRT ranking package
  dualeval.py               fit_irt(), fit_bt(), data loaders — CLI or importable
  config_dualeval.yaml      Fully annotated config template (copy and edit)
  __init__.py               Package exports

eval/                       Evaluation scripts (generate responses, grade correctness)
  eval_static.py            Static eval for all domains (--domain math|misc|coding)
  eval_arena.py             Arena response generation for all domains (--domain)

reward/                     Reward model scoring
  public_reward.py          Score responses with Skywork-Reward-V2-Qwen3-8B
  rm_validate.py            Validate RM pair preferences against human labels

demo/                       End-to-end paper reproduction pipeline
  run_pipeline.sh           One-shot script: datasets → responses → rewards → rankings
  sample_static.py          Sample static benchmark questions (--domain math|misc)
  sample_arena.py           Sample arena prompts from HF datasets (--domain math|coding|generic)
  config_static.yaml        Config for static eval (math and misc domains)
  config_static_coding.yaml Config for coding static eval
  config_arena.yaml         Config for arena eval (all four domains)

analysis/                   Ablations and experiments (aligned with paper sections)
  baseline_comparison.py          §4.1 DualEval vs. Bradley-Terry / static-only baselines
  ranking_bootstrap.py            §4.1 Bootstrap ranking stability (Table 4)
  item_informativeness.py         §4.2 Discrimination-weighted question selection
  anomaly_detection.py            §4.3 Residual-based anomaly detection
  calibration.py                  Appendix held-out calibration and ECE
```

---

## Method

See the paper for the full technical formulation: the 2PL-IRT model for static data, the soft-pairwise distillation objective for arena rewards, and the joint training procedure.

---

## Citation

```bibtex
@article{dualeval2026,
  title   = {DualEval: Jointly Ranking LLMs with Static Benchmarks and Arena Reward Signals via Item Response Theory},
  journal = {arXiv preprint},
  year    = {2026},
}
```
