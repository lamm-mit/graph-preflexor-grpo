# Ideation loop

Turn Graph-PRefLexOR into a self-expanding **knowledge-graph ideation engine**: seed a
topic, generate a structured graph-native answer, accumulate its `<graph_json>` into a
growing NetworkX graph (with embedding de-duplication), and expand via follow-up questions
until a compute budget is spent — then score the result for ideation/creativity.

## 1. Serve both models (mistral.rs, one endpoint)

```bash
mistralrs from-config -f models.toml          # generator + questioner on :1234
curl -s http://localhost:1234/v1/models        # verify both loaded
python /path/to/mistral.rs/examples/server/responses.py   # verify /v1/responses works
```

## 2. Install + configure

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml             # then edit if needed
```

## 3. Run

```bash
python ideate.py --topic "self-healing biopolymer composites" \
    --strategy frontier --context-mode fresh --budget-calls 40 --out runs/exp1
```

Outputs in `runs/exp1/`: `graph.graphml` (open in Gephi/Cytoscape), `transcript.jsonl`,
`growth.csv` (ideas vs compute), `summary.json` (metrics).

## Dials

| Flag | Meaning |
|------|---------|
| `--strategy` | `frontier` (graph-analytic, default) · `node` (breadth) · `answer` (depth, LLM follow-ups) · `edge` (densify/missing links) · `novelty` · `mixed` |
| `--context-mode` | `fresh` (independent single turns — default, matches single-turn training) · `chained` / `branched` (multi-turn, **experimental**) |
| `--budget-calls / --budget-tokens / --max-iters` | compute budget (first to hit wins; novelty-stop also applies) |
| `--fanout` | questions spawned per step |
| `--dedup-threshold` | node-merge cosine (higher = stricter) |

## How it works

### The loop (`loop.py`)
Each step: **pop** the highest-priority question → **generate** a graph-native answer
(Graph-PRefLexOR) → **parse** its `<graph_json>` → **merge** it into one growing NetworkX
graph (with embedding de-dup) → **expand** by asking a strategy for follow-up questions →
**stop** when the budget or novelty-stop triggers.

### The frontier = a best-first priority queue
Questions live in a `heapq` used as a max-priority queue: items are pushed as
`(-priority, counter, candidate)`, where a candidate is `{"q": question, "anchor": node}`.
`-priority` turns the min-heap into max-first; `counter` is a FIFO tiebreaker that also keeps
Python from comparing the dicts. Priority (`_score`) is `1 / (1 + degree(anchor))`, so
**low-degree / peripheral nodes are expanded first** — the search fans out across the graph
frontier rather than drilling one path (DFS) or sweeping uniformly (BFS). A `seen_q` set
prevents re-asking a question, so the loop can't cycle. The seed topic enters with priority 1.0.

### Strategies and templates (`strategies.py`)
The **generator** (Graph-PRefLexOR) is called every step. The **questioner** LLM is called
*only* by the `answer` strategy — the rest derive the next question from the graph's
**structure** (NetworkX) and fill a fixed string template, so they need **no second LLM**.
Every templated question is wrapped by `_q(text, topic)` to stay anchored to the topic.

| Strategy | Target chosen by | Template | 2nd LLM? |
|---|---|---|---|
| `node` | each new node `n` | `By what mechanism does '{n}' operate, and what does it depend on?` | no |
| `frontier` | low-degree leaves + a high-betweenness hub `t` | `What are the key unresolved questions and underlying mechanisms concerning '{t}'?` | no |
| `edge` | embedding-close **unconnected** pair `(a,b)` | `How are '{a}' and '{b}' related, and what connects them?` | no |
| `novelty` | node `t` farthest from the embedding centroid | `What is an unconventional or overlooked aspect of '{t}', and why might it matter?` | no |
| `mixed` | rotates frontier→node→edge→novelty | (those) | no |
| `answer` | — | sends the prose answer to the questioner asking for `fanout` follow-ups | **yes** |

Cost per step: heuristic strategies = **1 LLM call** (generator); `answer` = **2** (generator +
questioner).

Templates are deliberately **standalone, self-contained questions** (matching the single-turn
training data) — the model is never told what is "already known" or "unexplored", because in
`fresh` mode it has no such context. The loop's memory lives entirely in *which* node is
selected from the accumulated graph (the strategy + frontier priority), **not** in the prompt
wording.

### Context modes (`previous_response_id`)
> **Graph-PRefLexOR is single-turn trained** (one standalone question → one trace+answer), so
> **`fresh` is the in-distribution default and recommended mode.** The loop does not need the
> model's conversation memory — accumulation happens in our client-side `GraphStore`, and the
> strategies ask self-contained questions that match the training distribution. `chained` and
> `branched` feed the model a multi-turn conversation it never saw in training; they rely on
> the *base* model's residual multi-turn ability and are **experimental** (may hurt format /
> `<graph_json>` adherence).

We **never inject or pre-write** the `<think>` block. Each generation returns a `response.id`;
the server keeps that turn server-side. We pass an id back as `previous_response_id` and the
server reconstructs the conversation thread. The modes differ only in **which id** we pass:
- `fresh` *(default)* — pass `None`: every question is an independent single turn (matches training).
- `chained` *(experimental)* — pass the **previous** response: one linear, multi-turn conversation.
- `branched` *(experimental)* — pass the response that **first introduced the anchor node**
  (`node_origin[node]`): a *tree* of conversations. (No anchor → falls back to the last response.)

## Metrics (`summary.json`)

- **Graph dynamics:** nodes, edges, density, components, largest-component fraction, clustering, longest path.
- **Semantic diversity:** mean pairwise embedding distance, embedding spread.
- **Creativity proxies:** fluency (ideas), ideas-per-call (efficiency), elaboration (edges/idea), flexibility (clusters).

## Plots (`plot_ideation.py`)

Each run logs a per-step `growth.csv` (nodes, edges, cumulative tokens, diversity). Render
journal-quality figures — pass several run dirs to overlay them:

```bash
python plot_ideation.py --runs runs/exp1 --labels "Graph-PRefLexOR-3B" --out figures/ideation
```
```bash
python plot_ideation.py --runs runs/exp1 runs/gpt4o \
    --labels "Graph-PRefLexOR-3B" "GPT-4o" --out figures/ideation
```


Produces (PNG + SVG + PDF each, shared styling):
- **`*_curves`** — 2×2: **(a) ideas vs compute**, **(b) semantic diversity vs compute**,
  **(c) elaboration** (edges/idea), **(d) connectivity** — overlaid across models.
- **`*_bars`** — final-metric comparison (fluency, ideas/call, diversity, flexibility, …).
- **`*_graph_<label>`** — spring-layout snapshot of the final accumulated idea graph.
- **`*_analytics_<label>`** — rich 2×3 graph-property panel: degree distribution (+ log-log),
  centrality distributions (betweenness/closeness/PageRank), top **hub ideas** by PageRank,
  **relation-type** frequency, **community** sizes (modularity), and a global-metrics card
  (**small-worldness** σ/ω, clustering, transitivity, avg path length, diameter, modularity Q,
  assortativity, reciprocity, density). Scalars also dumped to `*_analysis_<label>.json`.
- **`*_growth_<label>`** — montage of the graph **over iterations** (reconstructed from each
  node/edge's `iter` provenance, fixed layout so nodes hold position). Control with
  `--growth-frames N` (default 6; `0` to skip).
- **`*_movie_<label>.gif`** — animated growth (one frame per iteration). Add `--movie`
  (and `--movie-fps N`); needs `pillow`.

`transcript.jsonl` and `growth.csv` are written **incrementally** during a run (flushed each
step), so you can `tail -f` them live; `graph.graphml` and `summary.json` are written at the end.

The "ideas vs compute" and "diversity vs compute" curves are the headline result: more,
more-diverse ideas per generator call.

## Examples (end-to-end)

All of these use **a single LLM** (only Graph-PRefLexOR) and **`fresh` context mode** — i.e.
every generation is an independent single turn, matching how the model was trained. The
graph still grows because accumulation is client-side; only the *strategy* (how the next
question is chosen from the graph) varies.

```bash
# A. Frontier (graph-analytic, the recommended default)
python ideate.py --topic "self-healing biopolymer composites" \
    --strategy frontier --context-mode fresh --budget-calls 40 --out runs/heal_frontier
python plot_ideation.py --runs runs/heal_frontier --labels "frontier" --out figures/heal

# B. Node (breadth-first: one question per new concept)
python ideate.py --topic "mechanical metamaterials for energy absorption" \
    --strategy node --context-mode fresh --budget-calls 30 --fanout 4 --out runs/meta_node
python plot_ideation.py --runs runs/meta_node --labels "node" --out figures/meta

# C. Edge (densify: probe likely-but-missing links)
python ideate.py --topic "spider silk structure-property relationships" \
    --strategy edge --context-mode fresh --budget-calls 35 --out runs/silk_edge
python plot_ideation.py --runs runs/silk_edge --labels "edge" --out figures/silk

# D. Novelty (steer to under-explored regions of the idea space)
python ideate.py --topic "bioinspired underwater adhesion" \
    --strategy novelty --context-mode fresh --budget-calls 30 --out runs/adh_novelty
python plot_ideation.py --runs runs/adh_novelty --labels "novelty" --out figures/adh

# E. Same topic, two strategies, OVERLAID on one figure
python ideate.py --topic "collagen toughening mechanisms" \
    --strategy frontier --budget-calls 40 --out runs/coll_frontier
python ideate.py --topic "collagen toughening mechanisms" \
    --strategy node --budget-calls 40 --out runs/coll_node
python plot_ideation.py --runs runs/coll_frontier runs/coll_node \
    --labels "frontier" "node" --out figures/coll_compare
```

Notes:
- The `answer` strategy is the only one that also calls the questioner LLM (open-ended,
  prose-driven follow-ups instead of structure-driven ones).
- `--context-mode chained|branched` are experimental for this single-turn-trained model; leave
  it `fresh` unless you're deliberately testing multi-turn behavior.

## Evaluation vs other models

The experiment **fixes the loop and swaps the generator**:

1. Run the loop with Graph-PRefLexOR and with a **baseline** (frontier API or another local
   model via the `baseline:` endpoint) under **identical** topic / strategy / context-mode /
   budget / seed.
2. For fairness, give the baseline the **same sentinel template** so it emits `<graph_json>`,
   and parse both with `parse.py`.
3. Overlay with `plot_ideation.py`; optionally run a **blind LLM-judge** pairwise on extracted
   hypotheses (novelty / insight / testability) → win-rate.
4. Repeat over several seed topics; report mean ± std + significance.

`compare.py` (orchestrates the two runs + judge + report) is a **stub** — the loop, metrics,
and plots it needs are ready.

## Files

`ideate.py` (CLI) · `loop.py` (budget + context modes) · `strategies.py` (expansion policies) ·
`graphstore.py` (accumulate + embed dedup) · `parse.py` (`<graph_json>` extractor) ·
`clients.py` (Responses API) · `metrics.py` · `plot_ideation.py` (figures) · `compare.py` (baseline, TODO).
