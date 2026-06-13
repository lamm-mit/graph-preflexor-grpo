#!/usr/bin/env bash
# Headline benchmark: the accumulated graph as a Graph-RAG knowledge base.
#
# Test-time compute spent by the graph-native reasoner is amortized into a reusable knowledge
# STRUCTURE (concepts + relationships). Does retrieving from it let the SAME small model answer
# domain questions better than closed-book? compare.py answers each task three ways with one model:
#   closed-book   question only (parametric knowledge — the floor)
#   flat-RAG      question + top retrieved concepts (bag of nodes, no edges)
#   graph-RAG     question + a retrieved subgraph (concepts AND their relationships, as triples)
# A blind judge scores each answer 1-5 (specificity/mechanism/relational/correctness); embeddings
# add objective grounding+coverage. graph-RAG >> closed-book = the graph helps; graph-RAG > flat-RAG
# = the STRUCTURE (not just the concepts) is what helps.
#
# Configure via env vars, then run:  bash run_benchmark.sh
set -euo pipefail

RUN="${RUN:-runs/exp}"                                  # ideate.py run dir (its graph.graphml is the KB)
TASKS="${TASKS:-benchmark_tasks.txt}"                   # one domain question per line
OUT="${OUT:-runs/exp/benchmark/graphrag}"             # output basename
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
