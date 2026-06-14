#!/usr/bin/env python
"""Concept-pair bridge benchmark over one mined Graph-PRefLexOR graph.

This benchmark asks a clean question:

  Can the same small model generate better testable hypotheses when it sees the
  true graph path/neighborhood connecting two concrete concepts?

For each sampled pair:

  A. baseline: concept A + concept B only
  B. graph:    concept A + concept B + true graph path/neighborhood

The script samples concrete endpoint pairs from an existing run's graph.graphml,
generates both answer sets with the same model, then reuses compare.py's blind
pairwise judge and plots.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

import compare
import insights as I
import synthesize as S
import task_graph_benchmark as T


SYSTEM = (
    "You are a rigorous, inventive research scientist. Produce one concrete, "
    "mechanistic, testable hypothesis. Do not write a broad review or a list."
)

ANSWER_INSTRUCTION = (
    "Propose ONE testable hypothesis that connects Concept A and Concept B in the "
    "given topic. The answer must use BOTH concepts non-trivially. Use this compact "
    "structure: Hypothesis/design claim; Mechanism; Non-obvious connection; "
    "Falsifying experiment; Expected discriminating result; Key controls or failure modes. "
    "Be specific, mechanistic, plausible, and experimentally decisive."
)

GRAPH_NOTE = (
    "The following path/neighborhood was mined from the Graph-PRefLexOR reasoning graph. "
    "It is a structural lead, not a verified fact. Use the concrete bridge concepts as "
    "mechanistic cues, but do not copy incoherent edge verbs literally."
)

DOMAIN_ANCHORS = {
    "adhesion", "adhesive", "alginate", "biomineralization", "biopolymer", "boronate",
    "capsule", "capillary", "catechol", "cellulose", "chitosan", "collagen", "composite",
    "crack", "crosslink", "diffusion", "disulfide", "dynamic", "enzyme", "enzymatic",
    "fiber", "fibril", "gelatin", "healing", "hydrogel", "hydrolysis", "ionic",
    "lignin", "matrix", "mechanophore", "membrane", "microcapsule", "microdomain",
    "mineralization", "nanocapsule", "nanofibril", "nanovesicle", "peptide", "polymer",
    "polymerization", "polysaccharide", "protein", "release", "repair", "reversible",
    "silk", "supramolecular", "swelling", "template", "templating", "vesicle",
}

META_PHRASES = {
    "novel idea", "untested idea", "benchmark", "mechanical testing", "testing protocol",
    "screening protocol", "candidate idea", "hypothesis generation", "idea generation",
}

SPECULATIVE_PHRASES = {
    "black hole", "cosmic", "entanglement", "gravitational", "quantum fluctuation",
    "quantum fluctuations", "quantum", "temporal entanglement", "time crystal",
    "topological insulator", "wormhole",
}

NOISY_PHRASES = {
    "catastrophic failure induced phase reconfigurable",
    "increased material topology",
    "nonequilibrium microdomains interact with degradation",
    "resource intensity increases environmental footprint",
}

LOCAL_GENERIC_LABELS = set(T.GENERIC_LABELS) | {
    "adaptive probes",
    "collective optimization",
    "healing agent",
    "healing reactions",
    "integrity restoration",
    "material properties",
    "mechanisms",
    "mechanistic understanding",
    "polymer matrix",
    "principles",
    "processing conditions",
    "repair mechanisms",
    "scalability",
    "self healing mechanisms",
    "self healing biopolymer composites",
    "simulations",
    "trade off",
    "trade offs",
    "interplay mechanisms",
}


def _display_label(label):
    s = str(label)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _display_chain(G, path):
    parts = [_display_label(T._label(G, path[0]))]
    for u, v in zip(path, path[1:]):
        parts.append(f"--{T._relation(G, u, v)}-->")
        parts.append(_display_label(T._label(G, v)))
    return " ".join(parts)


def _bridge_concepts(G, path):
    labels = []
    seen = set()
    for n in path[1:-1]:
        label = T._label(G, n)
        key = T._label_key(label)
        if key in seen or key in LOCAL_GENERIC_LABELS:
            continue
        seen.add(key)
        labels.append(_display_label(label))
    return labels


def _safe_name(i, a, b):
    txt = f"{a}_to_{b}".lower()
    txt = re.sub(r"[^a-z0-9]+", "_", txt).strip("_")[:80]
    return f"{i:03d}_{txt or 'pair'}"


def _read_graph(run):
    gp = Path(run) / "graph.graphml"
    if not gp.exists():
        raise SystemExit(f"missing graph: {gp}")
    G = nx.read_graphml(gp)
    if G.number_of_nodes() < 5:
        raise SystemExit(f"graph too small: {G.number_of_nodes()} nodes")
    return G


def _topic(run, G, override=None):
    return override or I.read_topic(str(run), G) or Path(run).name


def _has_phrase(key, phrases):
    return any(p in key for p in phrases)


def _domain_anchor_count(label):
    ws = set(T._words(label))
    return sum(1 for w in ws if w in DOMAIN_ANCHORS)


def _is_topic_hub(label, topic):
    key = T._label_key(label)
    topic_key = T._label_key(topic)
    if not key or not topic_key:
        return False
    if key == topic_key:
        return True
    topic_words = set(T._words(topic))
    label_words = set(T._words(label))
    return len(topic_words) >= 3 and topic_words.issubset(label_words)


def _label_rejection(label, topic, *, endpoint, quality_mode, allow_meta,
                     allow_speculative, allow_topic_hubs):
    key = T._label_key(label)
    if T._is_placeholder_label(label):
        return "placeholder"
    if _has_phrase(key, NOISY_PHRASES):
        return "known_noisy_phrase"
    if not allow_meta and _has_phrase(key, META_PHRASES):
        return "meta_or_method_label"
    if not allow_speculative and _has_phrase(key, SPECULATIVE_PHRASES):
        return "speculative_physics_label"
    if not allow_topic_hubs and _is_topic_hub(label, topic):
        return "topic_hub"
    if endpoint:
        if key in LOCAL_GENERIC_LABELS:
            return "generic_endpoint"
    return None


def _path_rejection(G, path, topic, args):
    labels = [T._label(G, n) for n in path]
    if not T._clean_path(G, path):
        return "unclean_path"
    for i, label in enumerate(labels):
        reason = _label_rejection(
            label,
            topic,
            endpoint=i in (0, len(labels) - 1),
            quality_mode=args.quality_mode,
            allow_meta=args.allow_meta,
            allow_speculative=args.allow_speculative,
            allow_topic_hubs=args.allow_topic_hubs,
        )
        if reason:
            return reason

    interior = labels[1:-1]
    if args.quality_mode == "strict":
        bridge_anchors = sum(_domain_anchor_count(x) for x in interior)
        if bridge_anchors < args.min_bridge_anchors:
            return "bridge_lacks_domain_anchors"
        generic_interior = sum(1 for x in interior if T._label_key(x) in LOCAL_GENERIC_LABELS)
        if generic_interior > args.max_generic_bridge_nodes:
            return "too_many_generic_bridge_nodes"
    return None


def _token_jaccard_distance(a, b):
    A, B = set(T._words(a)), set(T._words(b))
    if not A or not B:
        return 1.0
    return 1.0 - len(A & B) / max(1, len(A | B))


def _path_relation_quality(G, path):
    rels = [T._relation(G, u, v) for u, v in zip(path, path[1:])]
    return sum(T._relation_quality(r) for r in rels) / max(1, len(rels))


def _candidate_nodes(G, limit, topic, args):
    U = G.to_undirected()
    nodes = []
    for n in G.nodes:
        label = T._label(G, n)
        if not T._is_specific_label(label):
            continue
        if T._label_key(label) in LOCAL_GENERIC_LABELS:
            continue
        if _label_rejection(
            label,
            topic,
            endpoint=True,
            quality_mode=args.quality_mode,
            allow_meta=args.allow_meta,
            allow_speculative=args.allow_speculative,
            allow_topic_hubs=args.allow_topic_hubs,
        ):
            continue
        nodes.append(n)
    nodes.sort(key=lambda n: (_domain_anchor_count(T._label(G, n)),
                              T._specificity_score(T._label(G, n)),
                              -math.log1p(U.degree(n))),
               reverse=True)
    return nodes[:limit]


def _path_score(G, path):
    a, b = path[0], path[-1]
    la, lb = T._label(G, a), T._label(G, b)
    labels = [T._label(G, n) for n in path]
    interior = labels[1:-1]
    domain = sum(_domain_anchor_count(x) for x in labels)
    interior_domain = sum(_domain_anchor_count(x) for x in interior)
    generic_interior = sum(1 for x in interior if T._label_key(x) in LOCAL_GENERIC_LABELS)
    U = G.to_undirected()
    hub_penalty = sum(math.log1p(U.degree(n)) for n in path[1:-1]) / max(1, len(path) - 2)
    return (
        1.2 * domain
        + 0.8 * interior_domain
        + 0.8 * _path_relation_quality(G, path)
        + 0.6 * _token_jaccard_distance(la, lb)
        + 0.2 * (len(path) - 1)
        + 0.25 * (T._specificity_score(la) + T._specificity_score(lb))
        - 1.4 * generic_interior
        - 0.15 * hub_penalty
    )


def sample_pairs(G, *, n, seed, candidate_pool, candidate_pairs, min_hops, max_hops,
                 max_node_use, topic, args):
    U = G.to_undirected()
    nodes = _candidate_nodes(G, candidate_pool, topic, args)
    if len(nodes) < 2:
        raise SystemExit("not enough specific graph concepts to sample pairs")

    rng = random.Random(seed)
    pairs = list(itertools.combinations(nodes, 2))
    if candidate_pairs and candidate_pairs > 0 and candidate_pairs < len(pairs):
        rng.shuffle(pairs)
        pairs = pairs[:candidate_pairs]

    scored = []
    seen_paths = set()
    for a, b in pairs:
        try:
            path = nx.shortest_path(U, a, b)
        except nx.NetworkXNoPath:
            continue
        hops = len(path) - 1
        if hops < min_hops or hops > max_hops:
            continue
        reason = _path_rejection(G, path, topic, args)
        if reason:
            continue
        key = tuple(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        scored.append((_path_score(G, path), path))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen, use = [], {}
    label_pairs = set()
    for score, path in scored:
        a, b = path[0], path[-1]
        la, lb = T._label(G, a), T._label(G, b)
        pair_key = tuple(sorted((T._label_key(la), T._label_key(lb))))
        if pair_key in label_pairs:
            continue
        if use.get(a, 0) >= max_node_use or use.get(b, 0) >= max_node_use:
            continue
        label_pairs.add(pair_key)
        use[a] = use.get(a, 0) + 1
        use[b] = use.get(b, 0) + 1
        chosen.append({
            "concept_a": _display_label(la),
            "concept_b": _display_label(lb),
            "raw_concept_a": la,
            "raw_concept_b": lb,
            "node_a": str(a),
            "node_b": str(b),
            "path": [str(x) for x in path],
            "path_labels": [_display_label(T._label(G, x)) for x in path],
            "raw_path_labels": [T._label(G, x) for x in path],
            "chain": _display_chain(G, path),
            "raw_chain": T._chain_text(G, path),
            "bridge_concepts": _bridge_concepts(G, path),
            "hops": len(path) - 1,
            "score": float(score),
            "domain_anchors": int(sum(_domain_anchor_count(T._label(G, x)) for x in path)),
        })
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        raise SystemExit(f"only found {len(chosen)} usable concept pairs; lower --min-hops or --n")
    return chosen


def _neighborhood_block(G, path, neighbors_per_node, topic, args):
    U = G.to_undirected()
    rows = []
    path_set = set(path)
    for n in path:
        label = _display_label(T._label(G, n))
        rows.append(f"- {label}")
        neigh = []
        for m in U.neighbors(n):
            if m in path_set:
                continue
            ml = T._label(G, m)
            if T._is_placeholder_label(ml):
                continue
            if T._label_key(ml) in LOCAL_GENERIC_LABELS:
                continue
            if _label_rejection(
                ml,
                topic,
                endpoint=False,
                quality_mode=args.quality_mode,
                allow_meta=args.allow_meta,
                allow_speculative=args.allow_speculative,
                allow_topic_hubs=args.allow_topic_hubs,
            ):
                continue
            neigh.append((T._specificity_score(ml), U.degree(m), m))
        neigh.sort(reverse=True)
        for _, _, m in neigh[:neighbors_per_node]:
            rows.append(f"  - {T._relation(G, n, m)}: {_display_label(T._label(G, m))}")
    return "\n".join(rows)


def _task_text(topic, pair):
    return (
        f"Topic: {topic}\n"
        f"Concept A: {pair['concept_a']}\n"
        f"Concept B: {pair['concept_b']}\n"
        f"Task: {ANSWER_INSTRUCTION}"
    )


def _baseline_prompt(topic, pair):
    return (
        f"# Topic\n{topic}\n\n"
        f"# Concept A\n{pair['concept_a']}\n\n"
        f"# Concept B\n{pair['concept_b']}\n\n"
        f"# Task\n{ANSWER_INSTRUCTION}\n\n"
        "Write only the final answer."
    )


def _graph_prompt(topic, pair, neighborhood):
    return (
        f"# Topic\n{topic}\n\n"
        f"# Concept A\n{pair['concept_a']}\n\n"
        f"# Concept B\n{pair['concept_b']}\n\n"
        f"# Graph-derived bridge\n{GRAPH_NOTE}\n\n"
        f"Path: {pair['chain']}\n\n"
        f"Mechanistically useful bridge concepts: {', '.join(pair.get('bridge_concepts') or pair['path_labels'][1:-1])}\n\n"
        f"Local neighborhood around the path:\n{neighborhood}\n\n"
        f"# Task\n{ANSWER_INSTRUCTION}\n\n"
        "Rules: use both endpoint concepts; preferentially use the useful bridge concepts; "
        "convert the graph bridge into a scientifically plausible mechanism; ignore any graph "
        "relation that is too vague or implausible. "
        "Write only the final answer."
    )


def _answer(args, prompt):
    if args.backend == "openai":
        return S.answer_openai(SYSTEM, prompt, model=args.model, base_url=args.base_url,
                               api_key=args.api_key, temperature=args.temperature,
                               max_tokens=args.max_tokens)
    return S.answer_hf(SYSTEM, prompt, model=args.model, temperature=args.temperature,
                       max_tokens=args.max_tokens, device=args.device, dtype=args.dtype or "auto")


def _write_answer(path, topic, pair, answer):
    header = (
        f"*Topic:* {topic}\n\n"
        f"*Concept A:* {pair['concept_a']}\n\n"
        f"*Concept B:* {pair['concept_b']}\n\n---\n\n"
    )
    Path(path).write_text(header + answer.strip() + "\n", encoding="utf-8")


def _clear_generated_files(*dirs):
    for d in dirs:
        for p in Path(d).glob("*"):
            if p.is_file() and p.suffix in {".md", ".txt"}:
                p.unlink()


def run(args):
    run_dir = Path(args.run).resolve()
    out = Path(args.out).resolve()
    answer_base = out / "answers" / "baseline"
    answer_graph = out / "answers" / "graph"
    prompt_base = out / "prompts" / "baseline"
    prompt_graph = out / "prompts" / "graph"
    bench_dir = out / "benchmark"
    for d in (answer_base, answer_graph, prompt_base, prompt_graph, bench_dir):
        d.mkdir(parents=True, exist_ok=True)

    G = _read_graph(run_dir)
    topic = _topic(run_dir, G, args.topic)
    pairs_path = out / "pairs.json"
    resampled_pairs = False
    if pairs_path.exists() and not args.force_pairs:
        pairs = json.load(open(pairs_path, encoding="utf-8"))["pairs"][:args.n]
    else:
        resampled_pairs = True
        pairs = sample_pairs(
            G,
            n=args.n,
            seed=args.seed,
            candidate_pool=args.candidate_pool,
            candidate_pairs=args.candidate_pairs,
            min_hops=args.min_hops,
            max_hops=args.max_hops,
            max_node_use=args.max_node_use,
            topic=topic,
            args=args,
        )
        json.dump({"run": str(run_dir), "topic": topic, "pairs": pairs}, open(pairs_path, "w"), indent=2)
    if resampled_pairs:
        _clear_generated_files(answer_base, answer_graph, prompt_base, prompt_graph)

    tasks = [_task_text(topic, p) for p in pairs]
    tasks_file = out / "tasks.txt"
    tasks_file.write_text("\n".join(t.replace("\n", " ") for t in tasks) + "\n", encoding="utf-8")

    for i, pair in enumerate(pairs):
        name = _safe_name(i, pair["concept_a"], pair["concept_b"])
        base_md = answer_base / f"{name}.md"
        graph_md = answer_graph / f"{name}.md"
        base_prompt = _baseline_prompt(topic, pair)
        graph_prompt = _graph_prompt(topic, pair, _neighborhood_block(G, pair["path"], args.neighbors, topic, args))
        (prompt_base / f"{name}.txt").write_text(SYSTEM + "\n\n" + base_prompt, encoding="utf-8")
        (prompt_graph / f"{name}.txt").write_text(SYSTEM + "\n\n" + graph_prompt, encoding="utf-8")
        print(f"[path_pair_benchmark] {i+1}/{len(pairs)} {pair['concept_a']} <-> {pair['concept_b']}", flush=True)
        if args.force or not base_md.exists():
            if args.dry_run:
                print(f"+ generate baseline -> {base_md}")
            else:
                _write_answer(base_md, topic, pair, _answer(args, base_prompt))
        else:
            print(f"[path_pair_benchmark] reusing {base_md}", flush=True)
        if args.force or not graph_md.exists():
            if args.dry_run:
                print(f"+ generate graph -> {graph_md}")
            else:
                _write_answer(graph_md, topic, pair, _answer(args, graph_prompt))
        else:
            print(f"[path_pair_benchmark] reusing {graph_md}", flush=True)

    manifest = {
        "mode": "path_pair_benchmark",
        "run": str(run_dir),
        "topic": topic,
        "n": len(pairs),
        "sampling": {
            "seed": args.seed,
            "candidate_pool": args.candidate_pool,
            "candidate_pairs": args.candidate_pairs,
            "min_hops": args.min_hops,
            "max_hops": args.max_hops,
            "neighbors": args.neighbors,
            "quality_mode": args.quality_mode,
            "allow_meta": args.allow_meta,
            "allow_speculative": args.allow_speculative,
            "allow_topic_hubs": args.allow_topic_hubs,
            "min_bridge_anchors": args.min_bridge_anchors,
            "max_generic_bridge_nodes": args.max_generic_bridge_nodes,
        },
        "model": {"backend": args.backend, "model": args.model, "base_url": args.base_url},
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)

    if args.no_judge or args.dry_run:
        print(f"[path_pair_benchmark] wrote {out}; skipping judge", flush=True)
        return

    compare.run_pairwise(SimpleNamespace(
        tasks=str(tasks_file),
        system=str(answer_graph),
        baseline=str(answer_base),
        out=str(bench_dir / "pairwise"),
        seed=args.seed,
        jm=args.judge_model,
        jbu=args.judge_base_url,
        jak=args.judge_api_key,
        judge_effort=args.judge_effort,
    ))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="ideate.py run directory containing graph.graphml")
    p.add_argument("--out", default="runs/path_pair_benchmark")
    p.add_argument("--topic", help="override topic recovered from run metadata")
    p.add_argument("--n", type=int, default=10, help="number of concept-pair questions")
    p.add_argument("--force", action="store_true", help="regenerate answers")
    p.add_argument("--force-pairs", action="store_true", help="resample concept pairs")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--candidate-pool", type=int, default=260,
                   help="top specific graph concepts considered as endpoints")
    p.add_argument("--candidate-pairs", type=int, default=0,
                   help="sampled endpoint pairs scored before taking top N; 0 scores all pairs")
    p.add_argument("--min-hops", type=int, default=2)
    p.add_argument("--max-hops", type=int, default=5)
    p.add_argument("--max-node-use", type=int, default=2)
    p.add_argument("--neighbors", type=int, default=4,
                   help="specific off-path neighbors shown per path node in graph arm")
    p.add_argument("--quality-mode", choices=["strict", "permissive"], default="strict",
                   help="strict rejects meta/speculative/topic-hub paths; permissive only applies basic cleanup")
    p.add_argument("--allow-meta", action="store_true",
                   help="allow meta/method labels such as NovelIdea, UntestedIdea, or testing protocols")
    p.add_argument("--allow-speculative", action="store_true",
                   help="allow speculative physics labels such as quantum/topological/entanglement")
    p.add_argument("--allow-topic-hubs", action="store_true",
                   help="allow paths through a node whose label is the run topic")
    p.add_argument("--min-bridge-anchors", type=int, default=2,
                   help="minimum domain-anchor words required in interior path nodes in strict mode")
    p.add_argument("--max-generic-bridge-nodes", type=int, default=1,
                   help="maximum generic interior bridge nodes allowed in strict mode")

    p.add_argument("--backend", choices=["openai", "hf"], default="openai")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", dest="base_url", default=None)
    p.add_argument("--api-key", dest="api_key", default=None)
    p.add_argument("--temperature", type=float, default=0.65)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=1200)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"])

    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--judge-model", dest="judge_model", default="gpt-5.5")
    p.add_argument("--judge-base-url", dest="judge_base_url", default=None)
    p.add_argument("--judge-api-key", dest="judge_api_key", default=None)
    p.add_argument("--judge-effort", dest="judge_effort", default="high",
                   choices=["minimal", "low", "medium", "high"])
    run(p.parse_args())


if __name__ == "__main__":
    main()
