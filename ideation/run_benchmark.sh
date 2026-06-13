#!/usr/bin/env bash
# Headline benchmark: do DISTAL graph concepts make a small model more CREATIVE?
#
# Same small model, same "brainstorm N ideas" task, four conditions:
#   closed-book           the question only (the model's own priors — the floor)
#   near-RAG              + concepts most SIMILAR to the question (obvious — controls for "any retrieval?")
#   graph-RAG (optional)  + central anchors + UNUSUAL graph-connected angles + the reasoning chains
#                          linking them, offered as optional inspiration
#   graph-RAG (forced)    same context, but each idea MUST build on an unusual angle
# A blind judge scores each idea-set 1-5 (novelty/surprise/breadth/plausibility); a MANIPULATION CHECK
# measures whether the ideas actually used the injected concepts (incorporation). Read incorporation +
# novelty together: optional-ignored→forced-used tells you whether forcing helps or just parrots.
#
# Configure via env vars, then run:  bash run_benchmark.sh
set -euo pipefail

RUN="${RUN:-runs/exp}"                                  # ideate.py run dir (its graph.graphml is the KB)
TASKS="${TASKS:-benchmark_tasks.txt}"                   # one domain question per line
OUT="${OUT:-$RUN/benchmark/graphrag}"                 # output basename (tracks $RUN)
MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"      # the small generator (all three arms)
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"        # OpenAI-compatible endpoint serving $MODEL
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.5}"                   # blind absolute scorer (needs $OPENAI_API_KEY)
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"                    # optional OpenAI-compatible judge endpoint
MAX_ITER="${MAX_ITER:-}"                                # optional graph-compute cutoff (for the scaling story)

mi=();  [ -n "$MAX_ITER" ] && mi=(--max-iter "$MAX_ITER")
jbu=(); [ -n "$JUDGE_BASE_URL" ] && jbu=(--judge-base-url "$JUDGE_BASE_URL")

python compare.py --run "$RUN" --tasks "$TASKS" "${mi[@]}" \
    --model "$MODEL" --base-url "$BASE_URL" \
    --judge-model "$JUDGE_MODEL" "${jbu[@]}" --out "$OUT"

echo "=== done -> $OUT.{png,svg,pdf,json,md}"
echo "    (scaling story: re-run with MAX_ITER=250 / 750 / 1500 and compare — bigger graph, better RAG answers)"
