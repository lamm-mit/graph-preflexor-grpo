#!/usr/bin/env python
"""Per-task Graph-PRefLexOR benchmark.

This benchmark tests the expensive test-time-compute claim directly:

  A. Llama single-shot: answer each benchmark task directly, with no graph.
  C. Graph-PRefLexOR short run: for each task, build a small ideation graph with
     ideate.py, mine insights, synthesize an answer from a curated graph context
     packet, then judge graph answer vs baseline answer pairwise.

The graph arm feeds mined concepts by default and uses a two-call
candidate-generation/refinement step. Cleaner graph-relation modes can also feed
quality-filtered leads, paths, and edges. Both arms use the same strict final
answer format so the judge compares substance rather than prose shape.

The runner is resumable. Existing task graphs / insights / answers are reused
unless --force is passed.
"""
import argparse
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import networkx as nx

import compare
import insights as I
import synthesize as S


DEFAULT_ANSWER_TASK = (
    "Follow the benchmark task exactly. If it asks for multiple hypotheses or designs, provide "
    "the requested number; otherwise provide the single strongest idea. For each proposed idea, "
    "use this compact structure: Hypothesis/design claim; Mechanism; Non-obvious connection; "
    "Falsifying experiment; Expected discriminating result; Key controls or failure modes. "
    "Be concrete, mechanistic, and testable. Do not mention whether background notes were provided."
)

ANSWER_SYSTEM = (
    "You are a rigorous, inventive research scientist. Produce high-density hypothesis-generation "
    "answers: specific mechanisms, concrete experiments, falsifiable predictions, and clear controls. "
    "Do not reward yourself for broad surveys or generic literature-review prose."
)

GRAPH_CONTEXT_NOTE = (
    "The following are exploratory background notes mined from a graph built by extended reasoning "
    "about this exact task. They are noisy leads, not verified facts. Use them to spark concrete "
    "mechanisms, but do not copy an edge or path literally if it is vague, circular, generic, or "
    "chemically implausible. Reject placeholder labels and generic hub words. Do not cite the notes, "
    "do not mention a graph, and do not add provenance language to the final answer."
)

GENERIC_LABELS = {
    "a", "b", "c", "d", "e", "f", "g", "h",
    "healing", "repair", "damage", "autonomy", "biopolymer", "composite",
    "biopolymer composite", "biopolymercomposite", "mechanism", "mechanistic pathway",
    "self healing", "selfhealing", "autonomous healing", "underwater environment",
    "underwaterenvironment", "water", "trigger", "release", "regeneration",
    "microcapsules", "hydrogels", "enzymes", "mechanical stress", "mechanicalstress",
    "polymer matrix", "polymermatrix", "polymer chains", "polymerchains",
    "mechanical properties", "mechanicalproperties", "cross linking", "cross_linking",
    "stability", "biocompatibility", "no toxicity", "integration with biological systems",
    "continuous healing", "initiators", "repair pathways", "domain size", "domainsize",
    "tissue repair", "tissuerepair", "cell mediated healing", "cell-mediated healing",
}

GENERIC_RELATIONS = {
    "related_to", "provides", "enables", "causes", "contains", "influences",
    "mediated_by", "facilitates", "controls", "drives", "requires",
}


def _task_validity_constraints(task):
    """Small domain guardrails derived from the benchmark task.

    These are applied to BOTH arms' final answer prompts, and also to graph candidate
    generation/selection. They do not add graph information; they keep the small model
    from satisfying the format with mechanisms that do not answer the task.
    """
    t = task.lower()
    rules = []
    if "healing" in t or "self-healing" in t:
        rules += [
            "A self-healing answer must restore post-damage mechanical continuity, not merely increase toughness, crack resistance, or initial strength.",
            "Name the actual repair mechanism: reversible/dynamic bonds, healing-agent transport/reaction, interfacial rebonding, phase transition, capsule release, supramolecular reassociation, or another concrete pathway.",
            "Permanent crosslinking, static templating, reinforcement, or crack deflection is insufficient unless paired with a dynamic post-damage repair mechanism.",
        ]
    if "rate-limiting" in t:
        rules += [
            "Name one physical or chemical rate-limiting step and propose an intervention that accelerates that step; do not inhibit the stated bottleneck.",
            "Keep the setting as autonomic composite repair unless the task explicitly asks for biological wound healing.",
        ]
    if "underwater" in t:
        rules += [
            "For underwater healing, specify how the mechanism tolerates water: wet adhesion, ionic screening, swelling, diffusion, hydrolysis, oxygen/redox limits, or water-compatible bond exchange.",
        ]
    if "trade-off" in t:
        rules += [
            "For a trade-off task, explain how the design decouples variables rather than merely compromising between them.",
        ]
    if "distinguish intrinsic" in t or "extrinsic" in t:
        rules += [
            "For intrinsic-vs-extrinsic healing, the experiment must discriminate reversible bond reformation from released healing-agent/capsule chemistry.",
        ]
    if "self-reporting" in t or "signaling" in t or "damage signaling" in t:
        rules += [
            "For self-reporting, the same chemistry must plausibly produce both repair and a measurable signal.",
        ]
    if not rules:
        return ""
    return "\n".join(f"- {r}" for r in rules)


def _read_tasks(path, limit=None):
    tasks = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
    return tasks[:limit] if limit else tasks


def _slug(i, task):
    words = []
    for w in task.lower().replace("/", " ").replace("-", " ").split():
        w = "".join(ch for ch in w if ch.isalnum())
        if w:
            words.append(w)
        if len(words) >= 6:
            break
    return f"{i:03d}_" + ("_".join(words) or "task")


def _cmd_str(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _run(cmd, *, cwd, dry_run=False):
    print("+ " + _cmd_str(cmd), flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def _maybe_exists(path):
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def _write_task_file(path, tasks):
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t + "\n")


def _label(G, n):
    return str(G.nodes[n].get("label", n))


def _words(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    text = text.replace("_", " ").replace("-", " ")
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", text)]


def _label_key(label):
    return " ".join(_words(label))


def _is_placeholder_label(label):
    s = str(label).strip()
    if re.fullmatch(r"[A-Z]", s):
        return True
    if re.fullmatch(r"[A-Z][0-9]+", s):
        return True
    if re.fullmatch(r"(node|concept|idea|thing|entity)[ _-]?\d*", s, re.I):
        return True
    return False


def _specificity_score(label):
    if _is_placeholder_label(label):
        return -3.0
    key = _label_key(label)
    if key in GENERIC_LABELS:
        return -1.5
    ws = _words(label)
    if not ws:
        return -2.0
    score = 0.35 * len(ws)
    score += sum(0.25 for w in ws if len(w) >= 7)
    score += sum(0.35 for w in ws if w in {
        "catechol", "alginate", "chitosan", "collagen", "cellulose", "gelatin",
        "hydrophobic", "hydrophilic", "ionic", "redox", "boronate", "disulfide",
        "supramolecular", "sacrificial", "interfacial", "nanofibril", "silica",
        "polydopamine", "calcium", "zinc", "iron", "enzyme", "photothermal",
        "mechanophore", "hydrolysis", "diffusion", "crosslink", "micelle",
    })
    score -= sum(0.25 for w in ws if w in {
        "healing", "repair", "damage", "composite", "material", "mechanism",
        "autonomy", "properties", "system", "response", "process",
    })
    return score


def _is_specific_label(label):
    return _specificity_score(label) >= 0.55


def _relation(G, u, v):
    data = G.get_edge_data(u, v) or G.get_edge_data(v, u) or {}
    if isinstance(data, dict) and "relation" in data:
        return str(data.get("relation") or "related_to")
    if isinstance(data, dict) and data:
        first = next(iter(data.values()))
        if isinstance(first, dict):
            return str(first.get("relation") or "related_to")
    return "related_to"


def _relation_quality(rel):
    key = str(rel or "").strip().lower().replace("_", " ")
    if not key:
        return 0.0
    return 0.25 if key in GENERIC_RELATIONS else 0.75


def _chain_text(G, path):
    bits = []
    for i, n in enumerate(path):
        bits.append(_label(G, n))
        if i < len(path) - 1:
            bits.append(f"--{_relation(G, path[i], path[i + 1])}-->")
    return " ".join(bits)


def _flatten_insights(insights_json):
    if not Path(insights_json).exists():
        return []
    data = json.load(open(insights_json, encoding="utf-8"))
    flat = []
    for kind, rows in (data.get("miners") or {}).items():
        for rank, row in enumerate(rows or []):
            x = dict(row)
            x.setdefault("kind", kind)
            x["_rank"] = rank
            flat.append(x)
    flat.sort(key=lambda x: (float(x.get("actionability", 0.0)), -x.get("_rank", 0)), reverse=True)
    return flat


def _insight_nodes(ins):
    nodes = []
    for key in ("path", "chain", "cycle", "pair", "endpoints"):
        value = ins.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, list):
                    nodes.extend(item)
                else:
                    nodes.append(item)
    if ins.get("node") is not None:
        nodes.append(ins["node"])
    for key in ("instances", "routes"):
        for route in ins.get(key) or []:
            if isinstance(route, list):
                nodes.extend(route)
    return [n for n in nodes if n is not None]


def _insight_quality(G, ins):
    nodes = [n for n in _insight_nodes(ins) if n in G]
    labels = [_label(G, n) for n in nodes]
    if any(_is_placeholder_label(x) for x in labels):
        return -10.0
    if labels and not any(_is_specific_label(x) for x in labels):
        return -2.0
    spec = sum(_specificity_score(x) for x in labels) / max(1, len(labels))
    path = ins.get("path") or ins.get("chain") or ins.get("cycle")
    path_bonus = 0.0
    if isinstance(path, list) and len(path) >= 2 and all(n in G for n in path):
        if not _clean_path(G, path):
            return -2.0
        rels = [_relation(G, u, v) for u, v in zip(path, path[1:])]
        path_bonus += 0.25 * min(4, len(path) - 1)
        path_bonus += sum(_relation_quality(r) for r in rels) / max(1, len(rels))
    kind_bonus = {
        "conceptual_bridge": 0.5,
        "open_triad": 0.35,
        "latent_link": 0.25,
        "relational_analogy": 0.35,
        "feedback_loop": -0.2,
        "broker_idea": -0.4,
    }.get(ins.get("kind"), 0.0)
    return float(ins.get("actionability", 0.0)) + spec + path_bonus + kind_bonus


def _clean_path(G, path):
    labels = [_label(G, n) for n in path if n in G]
    if len(labels) < 2:
        return False
    keys = [_label_key(x) for x in labels]
    if any(_is_placeholder_label(x) for x in labels):
        return False
    generic = sum(1 for k in keys if k in GENERIC_LABELS)
    specific = sum(1 for x in labels if _is_specific_label(x))
    return specific >= 2 and generic <= 1


def _select_insights(G, insights, max_leads):
    scored = []
    seen = set()
    for ins in insights:
        path = ins.get("path") or ins.get("chain") or ins.get("cycle")
        if isinstance(path, list):
            key = tuple(str(x) for x in path)
        else:
            key = re.sub(r"\s+", " ", S._humanize(ins.get("title", "")).lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        q = _insight_quality(G, ins)
        if q > -1.0:
            scored.append((q, ins))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [ins for _, ins in scored[:max_leads]]
    if picked:
        return picked
    return insights[:max_leads]


def _format_insight(G, ins, i):
    lines = [f"{i}. [{ins.get('kind', 'insight')}] {S._humanize(ins.get('title', ''))}"]
    detail = " ".join(str(ins.get("detail", "")).split())
    if detail:
        lines.append(f"   detail: {detail[:900]}")
    for key in ("path", "chain", "cycle"):
        path = ins.get(key)
        if isinstance(path, list) and len(path) >= 2 and all(n in G for n in path):
            lines.append(f"   relation path: {_chain_text(G, path)}")
            break
    pair = ins.get("pair") or ins.get("endpoints")
    if isinstance(pair, list) and len(pair) >= 2 and all(n in G for n in pair[:2]):
        lines.append(f"   endpoints: {_label(G, pair[0])} <-> {_label(G, pair[1])}")
    return "\n".join(lines)


def _top_hubs_block(G, max_nodes, max_edges_per_node):
    U = G.to_undirected()
    hubs = sorted(G.nodes, key=lambda n: U.degree(n), reverse=True)[:max_nodes]
    rows = []
    for n in hubs:
        rows.append(f"- {_label(G, n)} (degree {U.degree(n)})")
        neigh = sorted(U.neighbors(n), key=lambda m: U.degree(m), reverse=True)[:max_edges_per_node]
        for m in neigh:
            rel = _relation(G, n, m)
            rows.append(f"  - {rel}: {_label(G, m)}")
    return "\n".join(rows)


def _graph_snapshot_block(G, max_nodes, max_edges, full_graph_nodes):
    U = G.to_undirected()
    nodes = list(G.nodes)
    if len(nodes) <= full_graph_nodes:
        chosen = nodes
        mode = "complete node list"
    else:
        chosen = sorted(nodes, key=lambda n: U.degree(n), reverse=True)[:max_nodes]
        mode = f"top {len(chosen)} nodes by degree"
    chosen_set = set(chosen)
    node_rows = [f"- {_label(G, n)} | degree={U.degree(n)} | iter={G.nodes[n].get('iter', '')}"
                 for n in chosen]
    edge_rows = []
    for u, v, d in G.edges(data=True):
        if u in chosen_set and v in chosen_set:
            edge_rows.append((U.degree(u) + U.degree(v), u, v, str(d.get("relation", "related_to"))))
    edge_rows.sort(reverse=True)
    edge_rows = edge_rows[:max_edges]
    edge_text = [f"- {_label(G, u)} --{rel}--> {_label(G, v)}" for _, u, v, rel in edge_rows]
    return (
        f"Snapshot mode: {mode}\n"
        f"Nodes:\n" + "\n".join(node_rows) + "\n\n"
        f"Edges among listed nodes:\n" + ("\n".join(edge_text) if edge_text else "(none)")
    )


def _paths_block(G, insights, max_paths):
    rows = []
    used = set()
    for ins in insights:
        path = ins.get("path") or ins.get("chain") or ins.get("cycle")
        if not isinstance(path, list) or len(path) < 2 or not all(n in G for n in path):
            continue
        if not _clean_path(G, path):
            continue
        key = tuple(path)
        if key in used:
            continue
        used.add(key)
        rows.append(f"- {_chain_text(G, path)}")
        if len(rows) >= max_paths:
            break
    return "\n".join(rows)


def _specific_concepts_block(G, max_nodes):
    U = G.to_undirected()
    candidates = [n for n in G.nodes if _is_specific_label(_label(G, n))]
    candidates.sort(key=lambda n: (_specificity_score(_label(G, n)), U.degree(n)), reverse=True)
    rows = []
    for n in candidates[:max_nodes]:
        rows.append(f"- {_label(G, n)} | degree={U.degree(n)} | iter={G.nodes[n].get('iter', '')}")
    return "\n".join(rows)


def _specific_edges_block(G, max_edges):
    U = G.to_undirected()
    rows = []
    for u, v, d in G.edges(data=True):
        lu, lv = _label(G, u), _label(G, v)
        if not (_is_specific_label(lu) or _is_specific_label(lv)):
            continue
        rel = str(d.get("relation", "related_to"))
        score = (_specificity_score(lu) + _specificity_score(lv) +
                 _relation_quality(rel) + 0.04 * (U.degree(u) + U.degree(v)))
        rows.append((score, lu, rel, lv))
    rows.sort(reverse=True)
    return "\n".join(f"- {u} --{rel}--> {v}" for _, u, rel, v in rows[:max_edges])


def _build_graph_context(task_run, insights_json, args):
    G = nx.read_graphml(task_run / "graph.graphml")
    raw_insights = _flatten_insights(insights_json)
    insights = _select_insights(G, raw_insights, args.max_leads)
    parts = [
        f"Graph summary: {G.number_of_nodes()} concepts, {G.number_of_edges()} relations.",
    ]

    if args.graph_context_mode in ("curated", "rich", "insights", "paths"):
        lead_rows = [_format_insight(G, x, i + 1) for i, x in enumerate(insights[:args.max_leads])]
        if lead_rows:
            parts.append("Curated structural leads:\n" + "\n\n".join(lead_rows))

    if args.graph_context_mode in ("curated", "rich", "paths"):
        paths = _paths_block(G, insights, args.path_leads)
        if paths:
            parts.append("Relation paths to consider:\n" + paths)

    if args.graph_context_mode in ("curated", "concepts"):
        concepts = _specific_concepts_block(G, args.max_context_nodes)
        if concepts:
            parts.append("Specific graph concepts worth considering:\n" + concepts)
    if args.graph_context_mode == "curated":
        edges = _specific_edges_block(G, args.max_context_edges)
        if edges:
            parts.append("Specific relation edges worth considering:\n" + edges)

    if args.graph_context_mode == "rich":
        hubs = _top_hubs_block(G, args.hub_nodes, args.neighbor_edges)
        if hubs:
            parts.append("High-connectivity neighborhoods:\n" + hubs)
        parts.append("Compact graph snapshot:\n" +
                     _graph_snapshot_block(G, args.max_context_nodes, args.max_context_edges,
                                           args.full_graph_nodes))

    if args.graph_context_mode == "full":
        parts.append("Graph node/edge table:\n" +
                     _graph_snapshot_block(G, args.max_context_nodes, args.max_context_edges,
                                           args.full_graph_nodes))

    text = "\n\n".join(parts)
    if len(text) > args.graph_context_chars:
        text = text[:args.graph_context_chars].rsplit("\n", 1)[0] + "\n...[context truncated]"
    return text, {
        "mode": args.graph_context_mode,
        "chars": len(text),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "insights_used": min(len(insights), args.max_leads),
        "raw_insights": len(raw_insights),
    }


def _build_answer_prompt(task, args, graph_context=None):
    constraints = _task_validity_constraints(task)
    if graph_context:
        background = (
            f"\n# Background notes\n{GRAPH_CONTEXT_NOTE}\n\n{graph_context}\n\n"
            "# How to use the background\n"
            "- Do not summarize the notes.\n"
            "- Do not output generic graph words such as healing, autonomy, or composite as mechanisms.\n"
            "- Convert one or more leads into a concrete material chemistry, interface design, or experiment.\n"
            "- Prefer a chemically plausible answer over literal obedience to a noisy relation path.\n"
        )
    else:
        background = ""
    constraint_block = f"\n# Validity constraints\n{constraints}\n" if constraints else ""
    return (
        f"# Benchmark task\n{task}\n"
        f"{background}\n"
        f"{constraint_block}\n"
        f"# Required answer behavior\n{args.answer_task}\n\n"
        "Write only the final answer. Do not include preamble, caveats about being an AI model, "
        "or meta-commentary about the benchmark."
    )


def _call_answer_model(args, prompt):
    if args.backend == "openai":
        return S.answer_openai(ANSWER_SYSTEM, prompt, model=args.model, base_url=args.base_url,
                               api_key=args.api_key, temperature=args.temperature,
                               max_tokens=args.max_tokens)
    return S.answer_hf(ANSWER_SYSTEM, prompt, model=args.model, temperature=args.temperature,
                       max_tokens=args.max_tokens, device=args.device, dtype=args.dtype or "auto")


def _call_answer_model_temp(args, prompt, temperature):
    old = args.temperature
    args.temperature = temperature
    try:
        return _call_answer_model(args, prompt)
    finally:
        args.temperature = old


def _write_answer(args, *, task, out, prompt_path, graph_context=None, dry_run=False):
    prompt = _build_answer_prompt(task, args, graph_context=graph_context)
    Path(prompt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(prompt_path).write_text(ANSWER_SYSTEM + "\n\n" + prompt, encoding="utf-8")
    if dry_run:
        print(f"+ generate answer -> {out}", flush=True)
        return
    answer = _call_answer_model(args, prompt).strip()
    header = f"*Topic:* {task}\n\n*Task:* {args.answer_task}\n\n---\n\n"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(header + answer + "\n", encoding="utf-8")
    print(f"[task_graph_benchmark] wrote {out}", flush=True)


def _candidate_prompt(task, args, graph_context):
    constraints = _task_validity_constraints(task)
    constraint_block = f"\n# Validity constraints\n{constraints}\n" if constraints else ""
    return (
        f"# Benchmark task\n{task}\n\n"
        f"# Curated graph context\n{GRAPH_CONTEXT_NOTE}\n\n{graph_context}\n\n"
        f"{constraint_block}\n"
        "# Candidate generation task\n"
        f"Generate {args.graph_candidates} distinct candidate answers inspired by different parts of "
        "the context. Each candidate must be concrete enough to test. For each candidate include:\n"
        "ID; hypothesis/design claim; exact material chemistry or interface/process; mechanism; "
        "why it is non-obvious; falsifying experiment; expected discriminating result; main risk.\n\n"
        "Rules: reject noisy literal paths, placeholders, and generic hub words. Do not use a candidate "
        "unless you can name a plausible chemistry, phase transition, transport process, interfacial "
        "mechanism, or measurement. If a candidate improves only toughness or crack resistance without "
        "post-damage repair, reject it. Be bold but scientifically coherent."
    )


def _refine_prompt(task, args, graph_context, candidates):
    constraints = _task_validity_constraints(task)
    constraint_block = f"\n# Validity constraints\n{constraints}\n" if constraints else ""
    return (
        f"# Benchmark task\n{task}\n\n"
        "# Curated graph context summary\n"
        f"{graph_context[:5000]}\n\n"
        "# Candidate hypotheses generated from graph context\n"
        f"{candidates}\n\n"
        f"{constraint_block}\n"
        "# Selection and final-answer task\n"
        "Choose the single best candidate, or recombine candidates if that yields a stronger answer. "
        "Optimize for novelty, mechanism, specificity, testability, and plausibility. Discard candidates "
        "that are generic, chemically incoherent, just paraphrase the context, or fail the validity "
        "constraints above.\n\n"
        f"# Required answer behavior\n{args.answer_task}\n\n"
        "Write only the final answer in the required structure. Do not mention candidates, background "
        "notes, graph context, or this selection process."
    )


def _write_graph_answer(args, *, task, out, prompt_path, candidate_path, graph_context, dry_run=False):
    Path(prompt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(candidate_path).parent.mkdir(parents=True, exist_ok=True)
    if args.graph_synthesis_mode == "direct":
        return _write_answer(args, task=task, out=out, prompt_path=prompt_path,
                             graph_context=graph_context, dry_run=dry_run)

    cand_prompt = _candidate_prompt(task, args, graph_context)
    Path(str(prompt_path).replace(".txt", ".candidates_prompt.txt")).write_text(
        ANSWER_SYSTEM + "\n\n" + cand_prompt, encoding="utf-8")
    if dry_run:
        print(f"+ generate graph candidates -> {candidate_path}", flush=True)
        print(f"+ refine graph answer -> {out}", flush=True)
        return
    candidates = _call_answer_model_temp(args, cand_prompt, args.candidate_temperature).strip()
    Path(candidate_path).write_text(candidates + "\n", encoding="utf-8")
    final_prompt = _refine_prompt(task, args, graph_context, candidates)
    Path(prompt_path).write_text(ANSWER_SYSTEM + "\n\n" + final_prompt, encoding="utf-8")
    answer = _call_answer_model(args, final_prompt).strip()
    header = f"*Topic:* {task}\n\n*Task:* {args.answer_task}\n\n---\n\n"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(header + answer + "\n", encoding="utf-8")
    print(f"[task_graph_benchmark] wrote {out}", flush=True)


def run(args):
    here = Path(__file__).resolve().parent
    out = Path(args.out).resolve()
    runs_dir = out / "task_runs"
    baseline_dir = out / "answers" / "baseline"
    graph_dir = out / "answers" / "graph"
    prompt_base_dir = out / "prompts" / "baseline"
    prompt_graph_dir = out / "prompts" / "graph"
    candidate_graph_dir = out / "candidates" / "graph"
    bench_dir = out / "benchmark"
    for d in (runs_dir, baseline_dir, graph_dir, prompt_base_dir, prompt_graph_dir,
              candidate_graph_dir, bench_dir):
        d.mkdir(parents=True, exist_ok=True)

    tasks = _read_tasks(args.tasks, args.limit)
    tasks_file = out / "tasks.txt"
    _write_task_file(tasks_file, tasks)

    manifest = {
        "mode": "task_graph_benchmark",
        "tasks_file": str(Path(args.tasks).resolve()),
        "out": str(out),
        "n_tasks": len(tasks),
        "ideate": {
            "strategy": args.strategy,
            "context_mode": args.context_mode,
            "budget_calls": args.budget_calls,
            "budget_tokens": args.budget_tokens,
            "max_iters": args.max_iters,
            "fanout": args.fanout,
            "dedup_threshold": args.dedup_threshold,
            "config": args.config,
        },
        "synthesize": {
            "backend": args.backend,
            "model": args.model,
            "base_url": args.base_url,
            "answer_task": args.answer_task,
            "max_leads": args.max_leads,
            "graph_context_mode": args.graph_context_mode,
            "graph_context_chars": args.graph_context_chars,
            "max_context_nodes": args.max_context_nodes,
            "max_context_edges": args.max_context_edges,
            "path_leads": args.path_leads,
            "hub_nodes": args.hub_nodes,
            "neighbor_edges": args.neighbor_edges,
            "graph_synthesis_mode": args.graph_synthesis_mode,
            "graph_candidates": args.graph_candidates,
        },
        "tasks": [],
    }

    for i, task in enumerate(tasks):
        name = _slug(i, task)
        task_run = runs_dir / name
        base_answer = baseline_dir / f"{i:03d}.md"
        graph_answer = graph_dir / f"{i:03d}.md"
        base_prompt = prompt_base_dir / f"{i:03d}.txt"
        graph_prompt = prompt_graph_dir / f"{i:03d}.txt"
        graph_candidates = candidate_graph_dir / f"{i:03d}.txt"
        print(f"\n[task_graph_benchmark] task {i + 1}/{len(tasks)}: {task}", flush=True)

        ideate_cmd = [
            "python", "ideate.py",
            "--topic", task,
            "--strategy", args.strategy,
            "--budget-calls", str(args.budget_calls),
            "--max-iters", str(args.max_iters),
            "--out", str(task_run),
        ]
        if args.config:
            ideate_cmd += ["--config", args.config]
        if args.context_mode:
            ideate_cmd += ["--context-mode", args.context_mode]
        if args.budget_tokens:
            ideate_cmd += ["--budget-tokens", str(args.budget_tokens)]
        if args.fanout is not None:
            ideate_cmd += ["--fanout", str(args.fanout)]
        if args.dedup_threshold is not None:
            ideate_cmd += ["--dedup-threshold", str(args.dedup_threshold)]

        graphml = task_run / "graph.graphml"
        if args.force or not _maybe_exists(graphml):
            _run(ideate_cmd, cwd=here, dry_run=args.dry_run)
        else:
            print(f"[task_graph_benchmark] reusing {graphml}", flush=True)

        insights_json = task_run / "insights.json"
        insights_cmd = ["python", "insights.py", "--run", str(task_run), "--top", str(args.insights_top)]
        if args.force or not _maybe_exists(insights_json):
            _run(insights_cmd, cwd=here, dry_run=args.dry_run)
        else:
            print(f"[task_graph_benchmark] reusing {insights_json}", flush=True)

        if args.make_run_figures:
            fig_dir = task_run / "figures"
            _run(["python", "plot_ideation.py", "--runs", str(task_run), "--labels", name,
                  "--out", str(fig_dir / "graph")], cwd=here, dry_run=args.dry_run)
            _run(["python", "novelty.py", "--run", str(task_run), "--out", str(fig_dir / "novelty")],
                 cwd=here, dry_run=args.dry_run)

        if args.force or not _maybe_exists(base_answer):
            _write_answer(args, task=task, out=str(base_answer), prompt_path=str(base_prompt),
                          graph_context=None, dry_run=args.dry_run)
        else:
            print(f"[task_graph_benchmark] reusing {base_answer}", flush=True)

        context_meta = None
        if args.force or not _maybe_exists(graph_answer):
            if args.dry_run:
                graph_context = "(dry run: graph context would be built from graph.graphml + insights.json)"
                context_meta = {"mode": args.graph_context_mode, "dry_run": True}
            else:
                graph_context, context_meta = _build_graph_context(task_run, insights_json, args)
            _write_graph_answer(args, task=task, out=str(graph_answer), prompt_path=str(graph_prompt),
                                candidate_path=str(graph_candidates), graph_context=graph_context,
                                dry_run=args.dry_run)
        else:
            print(f"[task_graph_benchmark] reusing {graph_answer}", flush=True)
            if _maybe_exists(graph_prompt):
                context_meta = {"mode": args.graph_context_mode, "reused": True}

        manifest["tasks"].append({
            "index": i,
            "task": task,
            "run": str(task_run),
            "baseline_answer": str(base_answer),
            "graph_answer": str(graph_answer),
            "baseline_prompt": str(base_prompt),
            "graph_prompt": str(graph_prompt),
            "graph_candidates": str(graph_candidates),
            "graph_context": context_meta,
            "commands": {
                "ideate": _cmd_str(ideate_cmd),
                "insights": _cmd_str(insights_cmd),
            },
        })
        with open(out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if args.no_judge or args.dry_run:
        print(f"\n[task_graph_benchmark] wrote answers under {out}", flush=True)
        print("[task_graph_benchmark] skipping pairwise judge", flush=True)
        return

    pairwise_out = str(bench_dir / "pairwise")
    print("\n[task_graph_benchmark] pairwise judging graph answers vs baseline...", flush=True)
    compare.run_pairwise(SimpleNamespace(
        tasks=str(tasks_file),
        system=str(graph_dir),
        baseline=str(baseline_dir),
        out=pairwise_out,
        seed=args.seed,
        jm=args.judge_model,
        jbu=args.judge_base_url,
        jak=args.judge_api_key,
        judge_effort=args.judge_effort,
    ))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", required=True, help="benchmark task file, one task per line")
    p.add_argument("--out", default="runs/task_graph_benchmark", help="benchmark output directory")
    p.add_argument("--limit", type=int, default=None, help="optional first-N task cap for smoke tests")
    p.add_argument("--force", action="store_true", help="rebuild existing task runs and answers")
    p.add_argument("--dry-run", action="store_true", help="print commands without executing them")

    # Ideation run per task.
    p.add_argument("--config", default=None, help="config.yaml passed to ideate.py")
    p.add_argument("--strategy", default="frontier",
                   help="ideate.py strategy: frontier|node|answer|edge|novelty|leap|converse|mixed")
    p.add_argument("--context-mode", dest="context_mode", default=None,
                   help="optional ideate.py context mode: fresh|chained|branched")
    p.add_argument("--budget-calls", type=int, default=50)
    p.add_argument("--budget-tokens", type=int, default=None)
    p.add_argument("--max-iters", dest="max_iters", type=int, default=50)
    p.add_argument("--fanout", type=int, default=None)
    p.add_argument("--dedup-threshold", dest="dedup_threshold", type=float, default=None)
    p.add_argument("--insights-top", dest="insights_top", type=int, default=12)
    p.add_argument("--make-run-figures", action="store_true",
                   help="also run plot_ideation.py and novelty.py for every task graph")

    # Synthesis model for both arms.
    p.add_argument("--backend", choices=["openai", "hf"], default="openai")
    p.add_argument("--model", required=True, help="synthesis model id")
    p.add_argument("--base-url", dest="base_url", default=None,
                   help="[openai] OpenAI-compatible generator endpoint")
    p.add_argument("--api-key", dest="api_key", default=None)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=1200)
    p.add_argument("--device", default=None, help="[hf] device_map")
    p.add_argument("--dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"],
                   help="[hf] torch dtype")
    p.add_argument("--max-leads", dest="max_leads", type=int, default=10,
                   help="max mined graph leads passed to the graph answer prompt")
    p.add_argument("--graph-context-mode", choices=["concepts", "curated", "rich", "insights", "paths", "full"],
                   default="concepts",
                   help="how to feed the per-task graph to the synthesis model. concepts is default: "
                        "specific mined graph concepts only, avoiding noisy edge semantics; curated adds "
                        "quality-filtered leads/paths/edges; rich keeps the older high-degree hub packet")
    p.add_argument("--graph-context-chars", dest="graph_context_chars", type=int, default=14000,
                   help="hard character cap for the graph context packet")
    p.add_argument("--max-context-nodes", dest="max_context_nodes", type=int, default=80,
                   help="max nodes in compact graph tables when the graph is larger than --full-graph-nodes")
    p.add_argument("--max-context-edges", dest="max_context_edges", type=int, default=180,
                   help="max edges in compact graph tables")
    p.add_argument("--full-graph-nodes", dest="full_graph_nodes", type=int, default=120,
                   help="if a per-task graph has at most this many nodes, include the complete node list")
    p.add_argument("--path-leads", dest="path_leads", type=int, default=8,
                   help="max relation paths/chains included from mined structural insights")
    p.add_argument("--hub-nodes", dest="hub_nodes", type=int, default=10,
                   help="max high-degree nodes included with local neighborhoods")
    p.add_argument("--neighbor-edges", dest="neighbor_edges", type=int, default=6,
                   help="max neighbor edges shown per hub node")
    p.add_argument("--graph-synthesis-mode", choices=["candidates", "direct"], default="candidates",
                   help="candidates uses two graph-arm Llama calls: generate multiple graph-seeded "
                        "hypotheses, then select/refine the best final answer. direct uses one call")
    p.add_argument("--graph-candidates", dest="graph_candidates", type=int, default=6,
                   help="number of graph-seeded candidate hypotheses generated before refinement")
    p.add_argument("--candidate-temperature", dest="candidate_temperature", type=float, default=0.85,
                   help="temperature for the graph candidate-generation call")
    p.add_argument("--answer-task", dest="answer_task", default=DEFAULT_ANSWER_TASK,
                   help="shared synthesis instruction appended to each benchmark task")

    # Judge.
    p.add_argument("--no-judge", action="store_true", help="generate answers but skip pairwise scoring")
    p.add_argument("--judge-model", dest="judge_model", default="gpt-5.5")
    p.add_argument("--judge-base-url", dest="judge_base_url", default=None)
    p.add_argument("--judge-api-key", dest="judge_api_key", default=None)
    p.add_argument("--judge-effort", dest="judge_effort", default="high",
                   choices=["minimal", "low", "medium", "high"])
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
