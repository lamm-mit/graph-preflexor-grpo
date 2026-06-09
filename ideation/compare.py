"""Head-to-head: run the SAME loop with the generator vs a baseline model, then
compare ideation metrics (+ optional LLM-judge). Stub — fill in for the paper.

Plan:
  1. For each seed topic, run ideate twice (generator, baseline) with identical
     strategy/context/budget/seed; collect summary.json from each.
  2. Tabulate metrics side by side (nodes, ideas/call, diversity, flexibility...).
  3. Optional: blind LLM-judge pairwise on extracted hypotheses (novelty/insight/
     testability) -> win-rate.
  4. Emit report.md + comparison figures (ideas-vs-compute, diversity-vs-compute).

For a fair graph comparison, give the baseline the SAME sentinel template so it
emits <graph_json>, and parse both with parse.py.
"""
TODO = True
