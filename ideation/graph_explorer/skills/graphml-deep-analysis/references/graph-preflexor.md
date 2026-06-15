# Graph-PRefLexOR GraphML Reference

Use this reference when interpreting Graph Explorer or Graph-PRefLexOR graph snapshots.

## Run Layout

A typical run directory contains:

- `graph.graphml`: latest graph snapshot, rewritten during a run.
- `graphml/iter_####.graphml`: per-iteration snapshots, better for reproducible analysis.
- `transcript.jsonl`: per-iteration generation records.
- `growth.csv`: iteration-level node and edge counts.
- `summary.json`: topic, strategy, context mode, embedding model, and aggregate metrics.

In a hosted skill container, the app may upload only the selected `.graphml` file. If sidecar files are not attached, do not assume they exist.

## Graph Conventions

Graph-PRefLexOR builds an accumulating directed concept graph:

- Nodes are canonicalized concept labels after embedding-based de-duplication.
- Edges are generated relations between concepts.
- The graph may be read as a directed graph, but most community and bridge analysis should also use an undirected simple projection.
- Multi-edge GraphML from other tools should be folded into a simple graph before most algorithms.

Common node attributes:

- `label`: display label for the concept. If absent, use the node id.
- `iter`: reasoning iteration when the concept first appeared.
- `depth`: expansion depth or local reasoning depth.
- `question`: prompt/question that led to the concept.
- `response_id`: source generation id.

Common edge attributes:

- `relation`: generated relation label between concepts.
- `iter`: reasoning iteration when the edge first appeared.
- `depth`: expansion depth or local reasoning depth.
- `question`: prompt/question that led to the relation.
- `response_id`: source generation id.

All attributes are optional in arbitrary GraphML. Interpret only attributes that are present.

## Analysis Priorities

Graph Explorer is used for scientific ideation. Prefer evidence that helps a researcher decide what to inspect next:

- bridge concepts between modules
- relation chains that suggest mechanisms
- high-leverage modules with many external links
- late-arriving concepts that connect distant older modules
- isolates or small components that may indicate unexpanded ideas
- repeated questions that generated dense local clusters
- weakly justified edges or generic relation labels that need validation

## Snapshot Choice

Use `graphml/iter_####.graphml` when the question is about a specific step or when reproducibility matters. Use `graph.graphml` when the question asks about the latest graph or only the final result is attached.

If comparing snapshots, load each snapshot separately and compare:

- node and edge growth
- new modules or module merges
- new bridge nodes
- changes in top PageRank and betweenness nodes
- newly introduced cross-module paths

## Interpretation Rules

- A community is a candidate theme, not a ground-truth field.
- A high-betweenness node is a structural broker, not necessarily the best scientific idea.
- A relation label is generated text. Use it as a lead and identify what external evidence would validate it.
- Sparse or fragmented graphs need careful caveats; a missing edge may mean "not generated yet", not "no relationship exists".
- If a node's label is too generic, prefer paths or neighboring labels to infer meaning.
