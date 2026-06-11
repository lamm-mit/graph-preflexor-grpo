# Ideation loop

Turn Graph-PRefLexOR into a self-expanding **knowledge-graph ideation engine**: seed a
topic, generate a structured graph-native answer, accumulate its `<graph_json>` into a
growing NetworkX graph (with embedding de-duplication), and expand via follow-up questions
until a compute budget is spent — then score the result for ideation/creativity.


## Specific examples - Quick Start

### 1 Short example

```bash
# 1. Run the ideation loop  → writes runs/exp1/ (incl. per-iteration graphml/ snapshots)
python ideate.py --topic "self-healing biopolymer composites" \
    --strategy frontier --budget-calls 100 --out runs/exp1

# 2. Generate figures  → defaults into runs/exp1/figures/
python plot_ideation.py --runs runs/exp1 --labels "Graph-PRefLexOR-3B"
```
### 2 Long run, frontier strategy

```bash
# 1. Run the ideation loop  → writes runs/exp2/ (incl. per-iteration graphml/ snapshots)
python ideate.py --topic "self-healing biopolymer composites" --strategy frontier \
    --budget-calls 100000000 --budget-tokens 100000000000 --max-iter 100000000 \
    --out runs/exp2

# 2. Generate figures  → defaults into runs/exp2/figures/
python plot_ideation.py --runs runs/exp2 --labels "Graph-PRefLexOR-3B" --movie
```

### 3 Long run, novelty strategy

```bash
python ideate.py --topic "self-healing biopolymer composites" --strategy novelty \
    --budget-calls 100000000 --budget-tokens 100000000000 --max-iter 100000000 \
    --out runs/exp_novelty_2

python plot_ideation.py --runs runs/exp_novelty_2 --labels "Graph-PRefLexOR-3B" --movie
```
Then compare:

```bash
python plot_ideation.py --runs runs/exp2 runs/exp_novelty \
    --labels "frontier" "novelty" --out figures/algo_compare

```
Novelty:
```
python novelty.py --run runs/exp_novelty --out runs/exp_novelty/figures/novelty
python novelty.py --run runs/exp2 --out runs/exp2/figures/novelty
python novelty.py --runs runs/exp2 runs/exp_novelty runs/exp_leap \
    --labels frontier novelty leap --out figures/novelty_compare   # great for the paper
python novelty.py --run runs/exp2 --n-null 500                     # tighter p-values
```

### 4 Leap method (divergent)

```
python ideate.py --topic "self-healing biopolymer composites" --strategy leap \
    --budget-calls 100000000 --budget-tokens 100000000000 --max-iter 100000000 \
    --out runs/exp_leap
```

```bash
python plot_ideation.py --runs runs/exp_leap \
    --labels leap --out runs/exp_leap/figures/leap
```
Develop insights:
```bash
python insights.py --run runs/exp_leap --top 12
```
Multiple comparisons:

```bash
python plot_ideation.py --runs runs/exp2 runs/exp_novelty runs/exp_leap \
    --labels frontier novelty leap --out figures/strategy_compare
```

### Collecting runs from several machines (archive → HF dataset → analyze locally)

When runs live on different machines and you want to analyze them in one place, archive **only the
`ideate.py` outputs** (so you regenerate all figures/insights fresh locally), stage them on a
**private Hugging Face dataset**, then pull everything down.

**1. Archive each run** (one command per machine, from the `ideation/` dir). The explicit file list
captures exactly `ideate.py`'s outputs — `graph.graphml`, the `graphml/` snapshots, `transcript.jsonl`,
`growth.csv`, `summary.json` — and **nothing else** (no `figures/`, no `insights.*`). `--ignore-failed-read`
tolerates a missing `summary.json` on an **unfinished** run (it's only written when the loop stops):

```bash
tar czf exp2_ideate.tar.gz --ignore-failed-read \
  runs/exp2/graph.graphml runs/exp2/graphml \
  runs/exp2/transcript.jsonl runs/exp2/growth.csv runs/exp2/summary.json
```
(drop `runs/exp2/graphml` to skip the big per-iteration snapshots; repeat per run/machine.)

**2. Push to a private HF dataset** (`huggingface_hub` handles large files via LFS/Xet). One-time:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli repo create graph-preflexor-runs --type dataset --private   # → lamm-mit/graph-preflexor-runs
```
Then on each machine (headless-friendly — token via env, not stored):
```bash
export HF_TOKEN=hf_xxxxxxxx     # WRITE token: https://huggingface.co/settings/tokens
huggingface-cli upload lamm-mit/graph-preflexor-runs exp2_ideate.tar.gz \
    exp2_ideate.tar.gz --repo-type dataset
# syntax: upload <repo_id> <local_path> <path_in_repo> --repo-type dataset
```

**3. Pull everything down locally and extract** into `ideation/runs/`. The archives store relative
`runs/<run>/…` paths, so `tar xzf` recreates them under whatever your **current dir** is — extract while
your cwd is the `ideation/` dir so the analysis commands (`--runs runs/exp2 …`) find them:

```bash
cd ideation/        # the dir you run the analysis from
huggingface-cli download lamm-mit/graph-preflexor-runs --repo-type dataset --local-dir ./_hf_dl
for f in ./_hf_dl/*_ideate.tar.gz; do tar xzf "$f"; done   # → ideation/runs/exp2, runs/exp_leap, …
rm -rf ./_hf_dl                                             # optional: drop the downloaded archives
ls runs/                                                    # verify: exp2  exp_leap  exp_novelty_2
```

Analysis works on **unfinished** runs: `summary.json` is optional everywhere, and the tools fall back to
the newest stable `graphml/iter_*.graphml` snapshot if `graph.graphml` was caught mid-write. Without
`summary.json` the embed model isn't recorded, so pass `--embed-model` consistently (or accept the
embeddinggemma default), and use `--max-iter N` to compare runs of different (unfinished) lengths fairly.

Quick whole-dir variants (includes figures/insights if present — use when a run is finished and you want
*everything*): `tar czf exp2.tar.gz runs/exp2`. Non-HF alternatives: `rsync -avP exp2_ideate.tar.gz
user@server:/path/`, `scp`, or `rclone copy` to S3/GDrive.

### Full analysis pipeline (per-run + three-way comparison)

Complete recipe to analyze three runs (`exp2`/frontier, `exp_leap`/leap, `exp_novelty_2`/novelty) and
overlay them. Run from the `ideation/` dir. **Order matters:** run `insights.py` before `novelty.py` so
Panel (E) uses the canonical mined conceptual bridges (otherwise it falls back to an approximation).

For a fair journal comparison of runs at different (or unfinished) lengths, add **`--max-iter 1500`** to
*every* command below — it truncates all runs to `iter <= 1500` consistently. Drop it to use each run in full.

```bash
# 1) Mine insights for each run  → runs/<run>/insights.{json,md} + insights_map.*
python insights.py --run runs/exp2          --top 12
python insights.py --run runs/exp_leap      --top 12
python insights.py --run runs/exp_novelty_2 --top 12

# 2) Per-run growth / graph-analytics figures  → runs/<run>/figures/
python plot_ideation.py --runs runs/exp2          --labels frontier
python plot_ideation.py --runs runs/exp_leap      --labels leap
python plot_ideation.py --runs runs/exp_novelty_2 --labels novelty

# 3) Per-run novelty figures  → <out>_novelty_map.*  +  _novelty_stats.*  +  _novelty.json
python novelty.py --run runs/exp2          --out runs/exp2/figures/novelty
python novelty.py --run runs/exp_leap      --out runs/exp_leap/figures/novelty
python novelty.py --run runs/exp_novelty_2 --out runs/exp_novelty_2/figures/novelty

# 4) Three-way comparisons (overlaid)  → figures/strategy_compare*
#    (first run = primary for the map/stats panels; panel C overlays all three trajectories)
python plot_ideation.py --runs runs/exp2 runs/exp_novelty_2 runs/exp_leap \
    --labels frontier novelty leap --out figures/strategy_compare
python novelty.py --runs runs/exp2 runs/exp_novelty_2 runs/exp_leap \
    --labels frontier novelty leap --out figures/strategy_compare

# 5) Synthesize a final insight-enriched answer per run (local Llama-3.2-3B-Instruct)
#    → runs/<run>/answer.md   (gated model: huggingface-cli login once)
python synthesize.py --run runs/exp2          --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/exp2/answer.md
python synthesize.py --run runs/exp_leap      --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/exp_leap/answer.md
python synthesize.py --run runs/exp_novelty_2 --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/exp_novelty_2/answer.md
```

Tips: add `--n-null 500` to the `novelty.py` lines for tighter p-values in final figures; add
`--graph-snapshot` / `--growth-frames 6` to `plot_ideation.py` if you want the (slower) spring-layout
node-link figures; pass `--embed-model all-MiniLM-L6-v2` everywhere for a lighter/faster embedder.


## 1. Serve both models (mistral.rs, vLLM, etc.)

mistral.rs:
```bash
mistralrs from-config -f models.toml          # generator + questioner on :1234
curl -s http://localhost:1234/v1/models        # verify both loaded
python /path/to/mistral.rs/examples/server/responses.py   # verify /v1/responses works
```

vLLM:
```bash
HF_TOKEN="your_hf_token" vllm serve lamm-mit/Graph-Preflexor-3b_08012026 --port 1234 --gpu-memory-utilization 0.6
```

## 2. Install + configure

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml             # then edit if needed
```

**Embedding model.** Node dedup, semantic diversity, and every semantic analysis use a
sentence-embedding model, set by `embed_model` in `config.yaml`. The default is
**`google/embeddinggemma-300m`** (newer/stronger). It's a **gated** HF model, so once:

```bash
huggingface-cli login            # and accept the license at huggingface.co/google/embeddinggemma-300m
# needs sentence-transformers >= 5
```

It uses EmbeddingGemma's symmetric **`STS`** prompt automatically (right for concept-vs-concept
similarity). To avoid the gating/size, set `embed_model: all-MiniLM-L6-v2` (lighter, ungated).
Each run **records** its `embed_model` in `summary.json`, and the offline tools
(`plot_ideation.py`, `insights.py`, `synthesize.py`) re-embed with that same model by default —
override per-invocation with `--embed-model <id>`.

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
| `--strategy` | `frontier` (graph-analytic, default) · `node` (breadth) · `answer` (depth, LLM follow-ups) · `edge` (densify/missing links) · `novelty` · `leap` (aggressive exploration) · `mixed` |
| `--context-mode` | `fresh` (independent single turns — default, matches single-turn training) · `chained` / `branched` (multi-turn, **experimental**) |
| `--budget-calls / --budget-tokens / --max-iters` | compute budget (first to hit wins; novelty-stop also applies) |
| `--fanout` | questions spawned per step |
| `--dedup-threshold` | node-merge cosine (higher = stricter) |

**Truncating runs for fair figures.** All analysis tools accept **`--max-iter N`** — they keep
only nodes/edges/rows with `iter <= N` (provenance is on every node/edge), so you can cut every
run to a common length (e.g. 1500 iterations) for apples-to-apples journal plots:
`plot_ideation.py --max-iter 1500`, `insights.py --max-iter 1500`, `novelty.py --max-iter 1500`,
`synthesize.py --max-iter 1500`. When a cap is active, `plot_ideation` derives final metrics from
the capped data (not the full-run `summary.json`), and `insights`/`novelty` re-mine the capped
graph instead of reading the (uncapped) `insights.json`.

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
| `leap` | peripheral node `a` + its most embedding-**dissimilar** partner `b`; and peripheral nodes for cross-domain transfer | `What radically new approach … by combining '{a}' and '{b}' …?` and `What principle from a completely different field could transform '{t}' …?` | no |
| `mixed` | rotates frontier→node→edge→novelty→leap | (those) | no |
| `answer` | — | sends the prose answer to the questioner asking for `fanout` follow-ups | **yes** |

Cost per step: heuristic strategies = **1 LLM call** (generator); `answer` = **2** (generator +
questioner).

**`novelty` vs `leap` (exploration).** `novelty` *drifts* to the edge of what's known — it
re-examines the single most-peripheral concept ("an overlooked aspect of `t`"). `leap` *jumps
outside* it: it **recombines the most embedding-dissimilar concept pair** into a forced
mechanism and **imports principles from unrelated fields**, injecting ideas the graph doesn't
yet contain. Use `leap` when you want the search to fan out aggressively into new territory
rather than consolidate around the seed.

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
- **`*_curves`** — 2×2: **(a) ideas**, **(b) semantic diversity**, **(c) elaboration**
  (edges/idea), **(d) connectivity** — overlaid across models, x-axis = **reasoning depth**
  (hops of follow-up questions from the seed; aggregated from each node/edge's `depth`
  provenance).
- **`*_curves_index`** — the same four panels vs **reasoning index** (the per-step iteration
  counter). Both variants are written every run.
- **`*_bars`** — final-metric comparison (fluency, ideas/call, diversity, flexibility, …).
- **`*_graph_<label>`** — spring-layout node-link snapshot of the final idea graph. **OFF by
  default** (slow on big graphs); enable with `--graph-snapshot`. For interactive exploration,
  open the `graph.graphml` / `graphml/iter_*.graphml` files (written by `ideate.py`) in Gephi.
- **`*_analytics_<label>`** — rich 2×3 graph-property panel: degree distribution (+ log-log),
  centrality distributions (betweenness/closeness/PageRank), top **hub ideas** by PageRank,
  **relation-type** frequency, **community** sizes (modularity), and a global-metrics card
  (**small-worldness** σ/ω, clustering, transitivity, avg path length, diameter, modularity Q,
  assortativity, reciprocity, density). Scalars also dumped to `*_analysis_<label>.json`.
- **`*_structure_<label>`** — second 2×3 panel on the **shape of the reasoning** (skip with
  `--no-structure`):
  **(a) k-core decomposition** (the dense conceptual nucleus vs. the speculative periphery);
  **(b) broker scatter** (degree vs. betweenness, top brokers labelled — ideas that bridge
  separate clusters); **(c) critical-connector ideas** (articulation points ranked by how badly
  removing each fragments the graph); **(d) reasoning-depth profile** (new ideas + cumulative
  semantic diversity per hop from the seed); **(e) semantic map** (2D PCA of node embeddings,
  colored by community, sized by PageRank — the "idea landscape"); **(f) link homophily**
  (cosine similarity of *linked* vs. *random* idea pairs; Δ>0 = the model links semantically
  similar ideas, Δ≈0 = it makes creative leaps). Embeddings are re-derived offline from node
  labels; if `sentence-transformers` is missing the semantic sub-panels (e, f, and d's diversity
  line) are skipped.
- **`*_growth_<label>`** — spring-layout montage of the graph **over iterations** (fixed layout
  so nodes hold position). **OFF by default** (slow); enable with `--growth-frames N` (e.g. 6).
- **`*_movie_<label>.gif`** — animated growth (one frame per iteration). Add `--movie`
  (and `--movie-fps N`); needs `pillow`.

`transcript.jsonl` and `growth.csv` are written **incrementally** during a run (flushed each
step), so you can `tail -f` them live; `graph.graphml` and `summary.json` are written at the end.

The ideas- and diversity-vs-**reasoning-depth** curves are the headline result: how the idea
space broadens as the model reasons further from the seed.

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

## Insight mining (`insights.py`)

The accumulated graph is more than the sum of the answers: its **structure** encodes
hypotheses the model never stated in any single turn. `insights.py` mines a finished run's
`graph.graphml` for those, with **seven miners**, each emitting ranked, human-readable
candidates (a `score` plus a one-line `detail`):

| Miner | `kind` | What it surfaces | Needs embeddings? |
|---|---|---|---|
| **Conceptual bridges** | `conceptual_bridge` | shortest reasoning chains between *semantically distant* connected concepts — multi-step arguments the model implies but never wrote out. Score = embedding-distance × hops. | yes |
| **Latent links** | `latent_link` | link prediction over non-edges: structural **Adamic-Adar** × semantic cosine → relationships the graph "wants" but lacks. | optional (semantic term) |
| **Open triads** | `open_triad` | directed transitivity gaps `A→B→C` with no `A→C` → inferable relations, ranked by endpoint similarity. | optional |
| **Relational analogies** | `relational_analogy` | recurring relation-typed motifs (`A —r1→ B —r2→ C`) shared by **node-disjoint** instances → "A is to B as C is to D". Scored by embedding **parallelism** of the two steps × concept novelty. | optional (falls back to motif frequency) |
| **Feedback loops** | `feedback_loop` | directed simple cycles → candidate **self-reinforcing mechanisms** (apt for self-healing / homeostatic systems). Prefers short, coherent loops. | optional (coherence term) |
| **Semantic dissonance** | `semantic_dissonance` | pairs that are embedding-**similar** but graph-**distant** (≥3 hops or different components) — related in meaning, never linked in reasoning. | yes |
| **Broker ideas** | `broker_idea` | high-**betweenness**, multi-**community**, low-**Burt-constraint** nodes — interdisciplinary connectors where recombination/novelty concentrates. | no |

Embeddings are re-derived offline from node labels (they aren't stored in `graphml`), mirroring
the plotter. **Structural miners run without `sentence-transformers`**; the semantic ones (and
the semantic terms of the hybrid miners) light up when it's installed.

```bash
# Mine runs/exp2 → insights.json (structured) + insights.md (ranked report) + insights_map.* (figure)
python insights.py --run runs/exp2 --top 12

# Also expand the top leads into concrete, testable hypotheses via the generator (reuses config.yaml)
python insights.py --run runs/exp2 --llm
```

Outputs in the run dir (or `--out <base>`):
- **`insights.json`** — all candidates per miner (keyed by `kind`), plus any `--llm` expansions.
  This is the machine-readable artifact `synthesize.py` consumes.
- **`insights.md`** — a ranked, sectioned report you can read directly.
- **`insights_map.png/svg/pdf`** — the semantic PCA landscape with the **top conceptual bridges**
  drawn over it (skip with `--no-fig`).

`graph.graphml` is only written when a run **finishes**, so point `insights.py` at a completed
run — or copy a mid-run snapshot `runs/<exp>/graphml/iter_NNNN.graphml` to `graph.graphml` in a
scratch dir and mine that.

## Answer synthesis (`synthesize.py`)

Closes the loop: take the **original query** + the **mined insights** and have a language model
write one **complete, insight-enriched answer** — the reasoning the loop spread across many turns,
distilled back into prose and steered by the non-obvious connections the graph exposed. It's a thin
layer **on top of** `insights.py` (loads `insights.json`, or mines fresh with `--mine`).

**Backends** (`--backend`):

| `--backend` | Uses | Covers | Key flags |
|---|---|---|---|
| `openai` *(default)* | OpenAI Python SDK (chat-completions) | the **real OpenAI API** *and* **any OpenAI-compatible server** (vLLM, mistral.rs, llama.cpp, TGI, Together, Groq, …) | `--model`, `--base-url` (point at the server; omit for real OpenAI), `--api-key` (or `$OPENAI_API_KEY`) |
| `hf` | local **Hugging Face** `transformers` | any causal-LM repo id, run on this machine (uses the tokenizer's chat template when present) | `--model` (repo id), `--device`, `--dtype` |

**Prompting is deliberately flexible** — nothing is hard-coded to a domain:
- `--style {report,hypotheses,proposal,review,brief}` — built-in presets that swap the task wording.
- `--task "<free text>"` — override the task entirely (e.g. *"Write a 1-page Nature-style abstract
  proposing the single most novel mechanism."*).
- `--system "<free text>"` — replace the system prompt.
- `--max-per-kind N` — how many leads per miner to feed in (default 6).
- `--show-prompt` — print the assembled system+user prompt and exit (no model needed) to inspect/tune it.

```bash
# Real OpenAI
python synthesize.py --run runs/exp2 --backend openai --model gpt-4o \
    --api-key "$OPENAI_API_KEY" --out runs/exp2/answer.md

# Any OpenAI-compatible server (e.g. local vLLM on :8000)
python synthesize.py --run runs/exp2 --backend openai \
    --base-url http://localhost:8000/v1 --model meta-llama/Llama-3.1-70B-Instruct

# Local Hugging Face model, mining insights fresh, written as a research proposal
python synthesize.py --run runs/exp2 --backend hf \
    --model mistralai/Mistral-7B-Instruct-v0.3 --mine --style proposal

# Custom instruction; preview the prompt first
python synthesize.py --run runs/exp2 --show-prompt \
    --task "Rank the 3 most testable hypotheses and give a falsifying experiment for each."
```

**Per-run, with a local Llama-3.2-3B-Instruct** (HF backend; gated model — `huggingface-cli login`
+ accept its license once). Run `insights.py` on each run first (or add `--mine`):

```bash
python synthesize.py --run runs/exp2          --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/exp2/answer.md
python synthesize.py --run runs/exp_leap      --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/exp_leap/answer.md
python synthesize.py --run runs/exp_novelty_2 --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/exp_novelty_2/answer.md
```

Each writes `runs/<run>/answer.md`. Add `--mine` to skip the separate `insights.py` step (mines on
the fly), `--style proposal|hypotheses|review|brief` to change the format, and `--topic "…"` if the run
has no `summary.json` yet (unfinished runs). For the exact 1500-iter cutoff, add `--max-iter 1500`.

Writes **`answer.md`** (a provenance header — question, backend, model, style, #leads — followed by
the generated answer) and echoes it to stdout. `synthesize.py` needs `insights.json` (run
`insights.py` first, or pass `--mine` to compute it on the fly), and a `summary.json` in the run dir
for the topic (or supply `--topic`).

**Pipeline at a glance:**

```
ideate.py  →  graph.graphml  →  insights.py  →  insights.json  →  synthesize.py  →  answer.md
 (grow)        (accumulate)      (mine structure)  (ranked leads)   (LLM synthesis)   (final answer)
```

## Worked examples — idea → insight → answer (50 iterations each)

Short, self-contained runs (`--budget-calls 50 --max-iters 50`) that take a seed topic all the way
through the full pipeline and end with a **new design principle** written by a local
Llama-3.2-3B-Instruct, steered by the mined connections. Each block is independent — pick an area and
run it. (Swap `--strategy` to taste; `leap` favors cross-domain leaps, `frontier` balanced coverage.)

**Beginning → end.** First, one-time setup (serve the generator + install + authenticate the gated
embedding/synthesis models), then run any example block below:

```bash
# 0. one-time setup (see §1–§2 for details)
cd ideation/
pip install -r requirements.txt
cp config.example.yaml config.yaml                       # endpoints / model ids
huggingface-cli login                                    # for embeddinggemma-300m + Llama-3.2-3B (gated)
HF_TOKEN=... vllm serve lamm-mit/Graph-Preflexor-3b_08012026 --port 1234 &   # the generator (or mistral.rs)
curl -s http://localhost:1234/v1/models                  # verify it's up before running ideate.py
```

```bash
# A. Materials science — tough AND recyclable structural polymers
python ideate.py --topic "covalent adaptable networks for recyclable thermoset composites" \
    --strategy leap --budget-calls 50 --max-iters 50 --out runs/ex_recyclable
python insights.py     --run runs/ex_recyclable --top 12
python plot_ideation.py --runs runs/ex_recyclable --labels recyclable
python novelty.py      --run runs/ex_recyclable --out runs/ex_recyclable/figures/novelty
python synthesize.py   --run runs/ex_recyclable --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_recyclable/answer.md \
    --task "Propose ONE new design principle for a structural polymer that is both fracture-tough and \
fully recyclable. Ground it in the mined connections; give the mechanism and a falsifiable prediction."

# B. Bioinspired — passive radiative cooling surfaces (no energy input)
python ideate.py --topic "bioinspired passive radiative cooling surfaces" \
    --strategy frontier --budget-calls 50 --max-iters 50 --out runs/ex_cooling
python insights.py     --run runs/ex_cooling --top 12
python plot_ideation.py --runs runs/ex_cooling --labels cooling
python novelty.py      --run runs/ex_cooling --out runs/ex_cooling/figures/novelty
python synthesize.py   --run runs/ex_cooling --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_cooling/answer.md \
    --task "Explain a new design principle for a surface that cools itself below ambient with no power, \
inspired by a biological structure from the mined leads. State the mechanism and how to test it."

# C. Bioinspired mechanics — impact-resistant, lightweight architected materials
python ideate.py --topic "bioinspired impact-resistant architected metamaterials" \
    --strategy leap --budget-calls 50 --max-iters 50 --out runs/ex_impact
python insights.py     --run runs/ex_impact --top 12
python plot_ideation.py --runs runs/ex_impact --labels impact
python novelty.py      --run runs/ex_impact --out runs/ex_impact/figures/novelty
python synthesize.py   --run runs/ex_impact --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_impact/answer.md \
    --task "Lay out a design principle for an ultralight architected material that survives repeated \
impact, combining two unrelated mechanisms surfaced by the mined conceptual bridges."

# D. Cross-disciplinary — multifunctional structural energy storage
python ideate.py --topic "structural batteries: load-bearing energy storage composites" \
    --strategy mixed --budget-calls 50 --max-iters 50 --out runs/ex_structbattery
python insights.py     --run runs/ex_structbattery --top 12
python plot_ideation.py --runs runs/ex_structbattery --labels structbattery
python novelty.py      --run runs/ex_structbattery --out runs/ex_structbattery/figures/novelty
python synthesize.py   --run runs/ex_structbattery --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_structbattery/answer.md \
    --task "Articulate a design principle that lets one material carry mechanical load AND store energy \
without compromising either, using the analogies and feedback loops in the mined insights."

# E. Bioinspired — atmospheric water harvesting (no power)
python ideate.py --topic "bioinspired atmospheric water harvesting surfaces" \
    --strategy leap --budget-calls 50 --max-iters 50 --out runs/ex_water
python insights.py     --run runs/ex_water --top 12
python plot_ideation.py --runs runs/ex_water --labels water
python novelty.py      --run runs/ex_water --out runs/ex_water/figures/novelty
python synthesize.py   --run runs/ex_water --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_water/answer.md \
    --task "Propose a new design principle for a surface that pulls drinkable water from dry air with no \
power input, drawing on the cross-domain leads in the mined insights. Give the mechanism and a test."

# F. Energy materials — safe, fast solid-state electrolytes
python ideate.py --topic "solid-state fast-ion conductors for dendrite-free batteries" \
    --strategy frontier --budget-calls 50 --max-iters 50 --out runs/ex_electrolyte
python insights.py     --run runs/ex_electrolyte --top 12
python plot_ideation.py --runs runs/ex_electrolyte --labels electrolyte
python novelty.py      --run runs/ex_electrolyte --out runs/ex_electrolyte/figures/novelty
python synthesize.py   --run runs/ex_electrolyte --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_electrolyte/answer.md \
    --task "Explain a design principle for a solid electrolyte that is simultaneously highly ion-conductive \
and dendrite-proof, grounded in the mined connections. State the mechanism and a falsifiable prediction."

# G. Engineered living materials — grow and self-repair
python ideate.py --topic "engineered living materials that grow and self-repair" \
    --strategy leap --budget-calls 50 --max-iters 50 --out runs/ex_living
python insights.py     --run runs/ex_living --top 12
python plot_ideation.py --runs runs/ex_living --labels living
python novelty.py      --run runs/ex_living --out runs/ex_living/figures/novelty
python synthesize.py   --run runs/ex_living --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_living/answer.md \
    --task "Lay out a design principle for a material that biologically regrows after damage, combining two \
unrelated mechanisms surfaced by the mined conceptual bridges. Give the mechanism and how to test it."

# H. Green catalysis — enzyme-like selectivity at scale
python ideate.py --topic "bioinspired single-atom catalysts and enzyme mimics" \
    --strategy mixed --budget-calls 50 --max-iters 50 --out runs/ex_catalyst
python insights.py     --run runs/ex_catalyst --top 12
python plot_ideation.py --runs runs/ex_catalyst --labels catalyst
python novelty.py      --run runs/ex_catalyst --out runs/ex_catalyst/figures/novelty
python synthesize.py   --run runs/ex_catalyst --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_catalyst/answer.md \
    --task "Articulate a design principle for a heterogeneous catalyst that matches enzyme selectivity at \
industrial scale, using the analogies and feedback loops in the mined insights."

# --- Protein science ---

# I. De novo protein design — high-affinity, specific binders
python ideate.py --topic "de novo protein design for high-affinity target binding" \
    --strategy leap --budget-calls 50 --max-iters 50 --out runs/ex_binder
python insights.py     --run runs/ex_binder --top 12
python plot_ideation.py --runs runs/ex_binder --labels binder
python novelty.py      --run runs/ex_binder --out runs/ex_binder/figures/novelty
python synthesize.py   --run runs/ex_binder --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_binder/answer.md \
    --task "Propose a new design principle for de novo proteins that bind an arbitrary target with both high \
affinity and specificity, grounded in the mined bridges and analogies. Give the mechanism and a test."

# J. Protein folding — suppress pathological aggregation
python ideate.py --topic "protein misfolding and amyloid aggregation mechanisms" \
    --strategy frontier --budget-calls 50 --max-iters 50 --out runs/ex_amyloid
python insights.py     --run runs/ex_amyloid --top 12
python plot_ideation.py --runs runs/ex_amyloid --labels amyloid
python novelty.py      --run runs/ex_amyloid --out runs/ex_amyloid/figures/novelty
python synthesize.py   --run runs/ex_amyloid --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_amyloid/answer.md \
    --task "Explain a design principle to suppress pathological protein aggregation while preserving native \
function, drawn from the mined connections. State the mechanism and a falsifiable prediction."

# K. Enzyme engineering — active AND thermostable
python ideate.py --topic "enzyme active-site engineering and biocatalysis" \
    --strategy mixed --budget-calls 50 --max-iters 50 --out runs/ex_enzyme
python insights.py     --run runs/ex_enzyme --top 12
python plot_ideation.py --runs runs/ex_enzyme --labels enzyme
python novelty.py      --run runs/ex_enzyme --out runs/ex_enzyme/figures/novelty
python synthesize.py   --run runs/ex_enzyme --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_enzyme/answer.md \
    --task "Lay out a design principle for an engineered enzyme that is simultaneously highly active and \
thermostable, combining mechanisms surfaced by the mined conceptual bridges."

# L. Protein-based materials — beyond spider silk
python ideate.py --topic "structural proteins: silk, collagen, and elastin mechanics" \
    --strategy leap --budget-calls 50 --max-iters 50 --out runs/ex_silk
python insights.py     --run runs/ex_silk --top 12
python plot_ideation.py --runs runs/ex_silk --labels silk
python novelty.py      --run runs/ex_silk --out runs/ex_silk/figures/novelty
python synthesize.py   --run runs/ex_silk --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct --out runs/ex_silk/answer.md \
    --task "Articulate a design principle for a protein-based fiber that exceeds spider silk in toughness, \
using the cross-domain analogies in the mined insights. Give the mechanism and how to test it."
```

Each run lands in `runs/ex_<name>/` with the graph, figures, `insights.{json,md}`, novelty figures,
and a final `answer.md` (the design principle). To overlay several on one comparison figure, pass them
together: `python novelty.py --runs runs/ex_recyclable runs/ex_cooling runs/ex_impact --labels recyclable
cooling impact --out figures/examples_compare`. Llama-3.2-3B is gated (`huggingface-cli login` once); or
drop the `--backend`/`--model` flags and use `--show-prompt` to inspect the assembled prompt without a model.

## Novelty quantification (`novelty.py`)

Answers the reviewer's question — *how novel are these concepts and insights?* — with
**publication-grade figures** where every panel is a citeable method, plus a
**`<out>_novelty.json`** of the numbers so they can be quoted in the text. Two images are written
so each stands alone: **`<out>_novelty_map.*`** (the wide concept-space map, panel A) and
**`<out>_novelty_stats.*`** (panels B/C/D stacked, full-width). Needs embeddings.

Map image (**`_novelty_map`**): panels A, B.  Stats image (**`_novelty_stats`**): panels C, D, E.

| Panel | Method | The claim it supports |
|---|---|---|
| **(A) Concept space** | UMAP (or PCA) of every concept; **seed** marked; "established region" shaded by a KDE of the earliest-introduced concepts; points colored by **novelty-when-introduced** = `1 − cosine` to the nearest concept that already existed when it appeared. Kept label-free so the cloud stays readable. | A clean map of *all* ideas — new ones land **outside** the known region. |
| **(B) Ideation dynamics** | horizontal bar of the most novel concepts — **names on the y-axis** (always legible), bar length = novelty-when-introduced, colored by the **iteration** each first appeared. | *which* ideas were the novel ones and *when* they emerged (the names you can't read in the dense map). |
| **(C) Novelty vs reasoning** | mean nearest-prior-neighbour novelty per iteration ± bootstrap CI (open-ended **novelty-search**, Lehman & Stanley 2011); overlays runs if several are passed. | the engine keeps pushing into new territory; lets you **compare strategies** (frontier vs novelty vs leap). |
| **(D) Motif significance** | z-scores of relation-typed 2-step motifs vs a **relation-label-shuffled null** (network-motif significance, Milo et al. 2002); community **modularity** z (degree-preserving rewiring) + edge **heterophily** z (label permutation) annotated. | over-represented relational motifs (the **analogy** basis) and community structure are **beyond chance**. |
| **(E) Novel combinations** | combination **typicality** z of linked concept pairs vs the global pairwise-similarity distribution (adapting Uzzi et al. 2013); compares random pairs, existing edges, and **conceptual bridges**, with a Mann–Whitney p. | edges are locally homophilic, but the mined **bridges sit in the atypical tail** — the engine connects concepts across the full semantic diameter. |

```bash
# Single run → figure + json in runs/exp2/figures/
python novelty.py --run runs/exp2 --out runs/exp2/figures/novelty

# Compare strategies (first run drives A/C/D; all overlaid in B)
python novelty.py --runs runs/exp2 runs/exp_novelty runs/exp_leap \
    --labels frontier novelty leap --out figures/novelty_compare

# Tighter null distributions (more resamples; slower)
python novelty.py --run runs/exp2 --n-null 500
```

Reuses the run's recorded `embed_model` (override with `--embed-model`); pulls the conceptual
bridges from `insights.json` when present, else computes a fast approximation (**run `insights.py`
first** so Panel D uses the canonical mined bridges). `--n-null` trades runtime for tighter
p-values; `umap-learn` is used for panel (A) if installed, otherwise PCA. Embeddings are **batched**
and every heavy loop shows a **tqdm progress bar**; the per-concept-novelty, pairwise-stats, and
null-model computations are chunked/streamed so they stay memory-safe on large graphs (the exact
statistics are preserved — only the null **resample count** auto-scales down on very large graphs).

## Files

`ideate.py` (CLI) · `loop.py` (budget + context modes) · `strategies.py` (expansion policies) ·
`graphstore.py` (accumulate + embed dedup) · `parse.py` (`<graph_json>` extractor) ·
`clients.py` (Responses API) · `metrics.py` · `plot_ideation.py` (figures) ·
`insights.py` (mine the graph for novel leads) · `novelty.py` (novelty stats + paper figure) ·
`synthesize.py` (LLM answer from query + insights) · `compare.py` (baseline, TODO).
