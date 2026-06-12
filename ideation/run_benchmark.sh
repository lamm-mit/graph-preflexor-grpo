#!/usr/bin/env bash
# Headline benchmark: validated idea-space COVERAGE.
#
# Does the graph reasoning let the SAME small model reach validated, non-obvious hypotheses that
# single-shot resampling — given MORE samples — cannot? For each task, compare.py has the small model
# turn each top graph lead into a hypothesis (graph arm, K) and independently sample the bare question
# many times (baseline arm, M>=2K), gates every hypothesis for plausibility+testability with a blind
# judge, then measures distinct validated ideas, coverage saturation, and the fraction of the graph's
# validated ideas the baseline never reaches. The generator is the SAME small model for both arms.
#
# Configure via env vars, then run:  bash run_benchmark.sh
set -euo pipefail

RUN="${RUN:-runs/exp}"                                  # ideate.py run dir (its insights.json = the leads)
TASKS="${TASKS:-benchmark_tasks.txt}"                   # one task per line
OUT="${OUT:-runs/exp/benchmark/coverage}"              # output basename
MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"      # the small generator (same for both arms)
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"        # OpenAI-compatible endpoint serving $MODEL
LEADS="${LEADS:-8}"                                     # K: graph leads -> graph hypotheses
BASELINE_SAMPLES="${BASELINE_SAMPLES:-}"               # M: baseline samples (default 2*K inside compare.py)
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.5}"                   # gate judge (needs $OPENAI_API_KEY)
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"                    # optional OpenAI-compatible judge endpoint
MAX_ITER="${MAX_ITER:-}"                                # optional matched-compute graph cutoff

mi=();  [ -n "$MAX_ITER" ] && mi=(--max-iter "$MAX_ITER")
jbu=(); [ -n "$JUDGE_BASE_URL" ] && jbu=(--judge-base-url "$JUDGE_BASE_URL")
ms=();  [ -n "$BASELINE_SAMPLES" ] && ms=(--baseline-samples "$BASELINE_SAMPLES")

# fresh insights → actionability ranking + full (untruncated) labels for the leads
if [ ! -f "$RUN/insights.json" ] || [ -n "$MAX_ITER" ]; then
  python insights.py --run "$RUN" --top 12 "${mi[@]}"
fi

python compare.py --run "$RUN" --tasks "$TASKS" --leads "$LEADS" "${ms[@]}" "${mi[@]}" \
    --model "$MODEL" --base-url "$BASE_URL" \
    --judge-model "$JUDGE_MODEL" "${jbu[@]}" --out "$OUT"

echo "=== done -> $OUT.{png,svg,pdf,json,md}"
