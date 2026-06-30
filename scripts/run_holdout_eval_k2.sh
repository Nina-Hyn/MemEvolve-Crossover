#!/usr/bin/env bash
# Hold-out GAIA evaluation on tasks 1-20 — evolved systems only.
# Baselines (agent_kb, voyager) were already run on 1-20; reuse those results.
set -euo pipefail

cd "$(dirname "$0")/../"

export http_proxy="${http_proxy:-http://127.0.0.1:7897}"
export https_proxy="${https_proxy:-http://127.0.0.1:7897}"

set -a
source .env
set +a

JUDGE="${DEFAULT_JUDGE_MODEL:-deepseek-v4-flash}"
MAX_STEPS="${MAX_STEPS:-40}"
OUT_ROOT="./gaia_output/holdout_k2_evolved_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"

PROVIDERS=(
  pathfinder
  adaptive_trajectory_knowledge
  echo_base
)

echo "=== Hold-out GAIA eval — K2 evolved systems (tasks 1-20) ==="
echo "Output: $OUT_ROOT"
echo "Judge: $JUDGE"
echo "Max steps: $MAX_STEPS"
echo "Providers: ${PROVIDERS[*]}"
echo ""
echo "Existing baselines (for comparison, do not re-run):"
echo "  agent_kb:  gaia_output/agent_kb_results_runs/agent_kb_20260627_195855/"
echo "  voyager:   gaia_output/voyager_results_runs/voyager_20260627_214726/"

for p in "${PROVIDERS[@]}"; do
  echo ""
  echo ">>> Running $p ..."
  mkdir -p "${OUT_ROOT}/${p}"
  python run_flash_searcher_mm_gaia.py \
    --infile ./data/gaia/validation/metadata.jsonl \
    --outfile "${OUT_ROOT}/${p}/results.jsonl" \
    --task_indices 1-20 \
    --memory_provider "$p" \
    --max_steps "$MAX_STEPS" \
    --judge_model "$JUDGE" \
    --concurrency 1 \
    --direct_output_dir "${OUT_ROOT}/${p}" \
    2>&1 | tee "${OUT_ROOT}/${p}_run.log"
  echo ">>> $p report:"
  cat "${OUT_ROOT}/${p}/report.txt" 2>/dev/null || true
done

echo ""
echo "=== Evolved systems summary ==="
printf "%-32s %10s %12s\n" "Provider" "Accuracy" "TotalTokens"
for p in "${PROVIDERS[@]}"; do
  report="${OUT_ROOT}/${p}/report.txt"
  if [ -f "$report" ]; then
    acc=$(grep -m1 "^Accuracy:" "$report" | awk '{print $2}')
    tok=$(grep -m1 "^Total Tokens:" "$report" | awk '{print $3}')
    printf "%-32s %10s %12s\n" "$p" "${acc:-N/A}" "${tok:-N/A}"
  fi
done

echo ""
echo "=== Done. Evolved results: $OUT_ROOT ==="
