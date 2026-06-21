#!/usr/bin/env bash
# run_pipeline.sh — end-to-end paper reproduction using the public Skywork reward model
#
# Runs the full DualEval pipeline from raw HF datasets to IRT rankings:
#   1. Sample static benchmark questions (math, misc)
#   2. Evaluate static benchmarks (math, misc, coding)
#   3. Sample arena prompts from Chatbot Arena datasets (math, coding, generic)
#   4. Generate arena responses for all four domains (math, coding, misc, generic)
#   5. Score responses with the public Skywork reward model
#   6. Fit the DualEval IRT model and output ranked ability estimates
#
# Prerequisites:
#   pip install -r requirements.txt
#   Copy .env.example to .env and fill in your API keys.
#
# Run from the repo root:
#   bash demo/run_pipeline.sh
#
# To evaluate only a subset of domains or models, edit the `models` lists in
# demo/config_arena.yaml and demo/config_static.yaml before running.
#
# Expected wall time: several hours, depending on model API latency and
# the number of models in your config.

set -euo pipefail

STATIC_CFG="demo/config_static.yaml"
CODING_CFG="demo/config_static_coding.yaml"
ARENA_CFG="demo/config_arena.yaml"
RANKING_CFG="dualeval/config_dualeval.yaml"

echo "=== Step 1: Sample static benchmark questions ==="

echo "--- Math ---"
python demo/sample_static.py --domain math --config "$STATIC_CFG"

echo "--- Misc (HLE strata + SimpleQA) ---"
python demo/sample_static.py --domain misc --config "$STATIC_CFG"

echo ""
echo "=== Step 2: Evaluate static benchmarks ==="

echo "--- Math (LLM judge) ---"
python eval/eval_static.py --domain math --config "$STATIC_CFG"

echo "--- Misc (LLM judge) ---"
python eval/eval_static.py --domain misc --config "$STATIC_CFG"

echo "--- Coding (execution harness) ---"
python eval/eval_static.py --domain coding --config "$CODING_CFG"

echo ""
echo "=== Step 3: Sample arena prompts ==="

echo "--- Math arena ---"
python demo/sample_arena.py --domain math --config "$ARENA_CFG"

echo "--- Coding arena ---"
python demo/sample_arena.py --domain coding --config "$ARENA_CFG"

echo "--- Generic arena ---"
python demo/sample_arena.py --domain generic --config "$ARENA_CFG"

# misc uses a pre-existing sampled file; no sampling step needed.

echo ""
echo "=== Step 4: Generate arena responses ==="

for DOMAIN in math coding misc generic; do
    echo "--- Arena: $DOMAIN ---"
    python eval/eval_arena.py --domain "$DOMAIN" --config "$ARENA_CFG"
done

echo ""
echo "=== Step 5: Score responses with the public Skywork reward model ==="
echo "(Requires a CUDA-capable GPU; see reward/public_reward.py --help for options)"

python reward/public_reward.py \
    --arena-dir results/arena_eval \
    --output-dir results/public_reward

echo ""
echo "=== Step 6: Fit DualEval IRT model ==="

python dualeval/dualeval.py --config "$RANKING_CFG"

echo ""
echo "=== Done ==="
echo "Rankings written to: $(python -c "import yaml; c=yaml.safe_load(open('$RANKING_CFG')); print(c['output']['output_dir'])")"
