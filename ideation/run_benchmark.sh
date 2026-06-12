#!/usr/bin/env bash
# End-to-end headline benchmark: does the graph reasoning make the SAME small model's answers better?
#
# For each task in $TASKS it generates two answers with synthesize.py -- both Llama-3.2-3B, same task:
#   SYSTEM   = with mined graph insights      (uses the reasoning graph from an ideate.py run)
#   BASELINE = --no-insights, single-shot      (identical model+task, no graph)
# then compare.py blind-judges every pair on multiple dimensions and plots system vs baseline.
#
# Configure via env vars, then run:  bash run_benchmark.sh
set -euo pipefail

RUN="${RUN:-runs/exp}"                                  # ideate.py run dir that holds the graph
TASKS="${TASKS:-benchmark_tasks.txt}"                   # one task per line
OUTDIR="${OUTDIR:-runs/exp/benchmark}"                  # where answers + figures land
MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"      # the small generator (same for both arms)
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"        # OpenAI-compatible endpoint serving $MODEL
MAX_ITER="${MAX_ITER:-}"                                # optional fair cross-run graph cutoff
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.5}"                   # judge (evaluation only; needs $OPENAI_API_KEY)
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"                    # optional OpenAI-compatible judge endpoint

mkdir -p "$OUTDIR/sys" "$OUTDIR/base" "$OUTDIR/figures"
mi_args=(); [ -n "$MAX_ITER" ] && mi_args=(--max-iter "$MAX_ITER")
jbu_args=(); [ -n "$JUDGE_BASE_URL" ] && jbu_args=(--judge-base-url "$JUDGE_BASE_URL")

i=0
while IFS= read -r task; do
  [ -z "$task" ] && continue
  i=$((i+1)); n=$(printf '%02d' "$i")
  echo "=== task $n: $task"
  python synthesize.py --run "$RUN" --task "$task" "${mi_args[@]}" \
      --backend openai --model "$MODEL" --base-url "$BASE_URL" \
      --out "$OUTDIR/sys/$n.md"
  python synthesize.py --run "$RUN" --task "$task" --no-insights \
      --backend openai --model "$MODEL" --base-url "$BASE_URL" \
      --out "$OUTDIR/base/$n.md"
done < "$TASKS"

echo "=== judging $i pairs with $JUDGE_MODEL"
python compare.py --tasks "$TASKS" --system "$OUTDIR/sys" --baseline "$OUTDIR/base" \
    --judge-model "$JUDGE_MODEL" "${jbu_args[@]}" --out "$OUTDIR/figures/benchmark"

echo "=== done -> $OUTDIR/figures/benchmark.{png,svg,pdf,json,md}"
