---
name: graphml-deep-analysis
description: Analyze Graph-PRefLexOR and Graph Explorer GraphML snapshots for deep scientific ideation insight. Use when a user selects or uploads a .graphml file and asks about graph structure, communities or modules, bridge concepts, critical connectors, relation chains, novelty or opportunity gaps, run provenance, or next research questions.
metadata:
  version: "0.1.0"
---

# GraphML Deep Analysis

## Overview

Use this skill to answer research questions against one explicit GraphML snapshot from Graph Explorer or a Graph-PRefLexOR ideation run. The goal is not only to compute generic graph metrics; it is to turn graph evidence into scientific ideation insight: modules, bridges, latent relations, weak spots, promising next questions, and concrete node or path evidence.

Always ground claims in the selected GraphML file. Treat node labels, relation labels, questions, and provenance attributes as generated evidence that needs interpretation, not as verified scientific fact.

## Source File Contract

This skill analyzes a user-selected data file. The GraphML file is not bundled into the skill.

Preferred binding:

- The user or host app attaches the selected graph as an input file and names it in the prompt, for example `iter_0042.graphml`.
- If multiple GraphML files are present, use the filename named by the prompt.
- If no filename is named and exactly one `.graphml` file is visible in the working directory or `/mnt/data`, use it.
- If multiple plausible files are visible and the prompt does not disambiguate, ask which snapshot to use.

For Graph-PRefLexOR runs:

- Prefer immutable snapshots such as `graphml/iter_0042.graphml` for reproducible answers.
- Use `graph.graphml` only when the user asks for the latest state or when that is the only file attached.
- If a run is still active, read the selected file once and treat those bytes as the analysis snapshot.

Read `references/graph-preflexor.md` when you need details about Graph Explorer run layout, GraphML conventions, or how to map provenance attributes to analysis.

## Workflow

1. Identify the GraphML file.
   - Prefer the exact filename from the user request.
   - Otherwise inspect the current directory and `/mnt/data` for `.graphml` or `.xml` files.
   - Do not scan unrelated directories unless the user gives a path.

2. Run the bundled analyzer before synthesizing:

   ```bash
   python scripts/analyze_graphml.py iter_0042.graphml --question "Which nodes bridge the major communities?" --out graphml_analysis
   ```

   If the current working directory is not the skill directory, locate the bundled script:

   ```bash
   find . /mnt/data -path '*graphml-deep-analysis*' -name analyze_graphml.py 2>/dev/null
   ```

3. Read `graphml_analysis/analysis.md` first, then `graphml_analysis/analysis.json` if more detail is needed.

4. Answer the user directly, using graph evidence. Include the relevant node labels, module ids, relation chains, and metric names. Keep caveats precise: centrality and community structure indicate graph position, not external truth.

5. If the user asks for next-step ideation, translate graph findings into concrete candidate prompts or experiments. Tie each recommendation to a bridge, path, module gap, or provenance signal.

## Analysis Lenses

Use the user's question to choose the most relevant lenses. Do not dump every metric.

### Global Structure

Report node and edge counts, directedness, density, component count, isolate count, modularity, and major relation types when they clarify the answer. Use this to identify whether the graph is a coherent idea space, a fragmented frontier, or a hub-heavy map.

### Modules and Themes

Communities are candidate themes or subfields. For each important module, inspect:

- top terms from node labels
- top nodes by PageRank, degree, and betweenness
- internal vs external edges
- iteration and depth ranges
- source questions when present

Do not name a module from one node alone. Name it from recurring labels, relation types, and top nodes.

### Bridge Concepts

Bridge concepts are usually more important than raw hubs for ideation. Prioritize nodes that combine:

- high betweenness or PageRank
- neighbors in multiple modules
- many cross-module incident edges
- articulation-point or bridge-edge evidence
- useful provenance, such as appearing after a leap or novelty-seeking question

When answering "what bridges X and Y", prefer specific paths and relation labels over abstract centrality rankings.

### Relation Chains

A useful relation chain is a short path where intermediate concepts make the connection scientifically interpretable. For each chain, state:

- endpoint concepts
- intermediate concepts
- relation labels along the path
- why the path matters for the user's question
- what would need validation outside the graph

Avoid inventing missing mechanism. If a relation label is vague, say so and frame it as a lead.

### Novelty, Gaps, and Next Questions

For ideation, look for:

- under-connected modules with high-quality anchors
- short cross-module paths that suggest mechanism but have weak relation evidence
- boundary nodes with diverse neighbor modules
- modules that appear late in the run and connect to old anchors
- repeated source questions that yielded many nodes but few bridges
- isolates or small components that may be abandoned but promising

Turn these into actionable next questions, for example: "Explore whether A can mediate B through C because the graph has path A -r1-> C -r2-> B but no direct A-B edge."

### Provenance

Graph-PRefLexOR nodes and edges often carry `iter`, `depth`, `question`, and `response_id`. Use those attributes to explain when and why a concept entered the graph. If provenance is absent, avoid claims about run dynamics.

## Answer Style

For most questions, use this structure:

1. Direct answer in 2-4 sentences.
2. Evidence table or bullets with node labels, module ids, paths, and metric values.
3. Interpretation: what the evidence implies for the scientific question.
4. Next actions, if useful.

When the user asks for a ranked list, include enough evidence to make the ranking auditable. When the user asks for a deep dive, include module-level and path-level evidence, not just top-node lists.

## Scripts

The bundled `scripts/analyze_graphml.py` script is the default evidence-gathering tool. It reads a GraphML file with NetworkX and writes:

- `analysis.md`: human-readable report
- `analysis.json`: structured metrics and evidence

It is intentionally self-contained and does not require this repository's `profile_graph.py`. If the full Graph Explorer repo is available and the user asks for the richer PDF/profile workflow, `ideation/profile_graph.py` may be a better tool, but for a mounted hosted skill use the bundled script first.

Useful options:

```bash
python scripts/analyze_graphml.py <graph.graphml> \
  --question "<user question>" \
  --out graphml_analysis \
  --top 25 \
  --max-paths 30
```

## Guardrails

- Do not treat graph centrality as scientific validity.
- Do not claim a direct edge exists unless the GraphML contains it.
- Do not assume a file persists across containers or requests unless the host explicitly reuses the same container and the file is still visible.
- Do not analyze a stale `graph.graphml` when the user selected a specific `iter_####.graphml`.
- Do not expose raw `response_id` values unless they are needed for debugging or provenance.
