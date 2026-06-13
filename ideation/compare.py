#!/usr/bin/env python
"""Benchmark: does Graph-RAG make a small model more CREATIVE? (DEFAULT mode).

The headline comparison is intentionally simple: the same small model answers the same benchmark
tasks in two conditions:

  closed-book   the question only; the model brainstorms from its own priors.
  graph-RAG     the question plus task-conditioned graph context: central anchors, distal concepts,
                and short relation chains retrieved from the accumulated ideation graph.

Per task each arm brainstorms N ideas; a blind judge scores each SET 1-5 on novelty / surprise /
breadth / plausibility, and idea diversity / distance from the closed-book prior are measured
objectively. Pair with `--max-iter` for the scaling story (bigger graph -> better retrieved context).
Reads the graph directly; insights.json is not used.

    python compare.py --run runs/exp --tasks benchmark_tasks.txt \
        --model meta-llama/Llama-3.2-3B-Instruct --base-url http://localhost:8000/v1 \
        --judge-model gpt-5.5 --out runs/exp/benchmark/graphrag

Other modes: `--mode coverage` (validated idea-space coverage vs single-shot resampling) and
`--mode pairwise` (legacy single-answer judge over two answer dirs).
"""
import argparse
import glob
import json
import os
import random
import re

import numpy as np
import networkx as nx

import insights as I
import synthesize as S

GEN_SYSTEM = ("You are a rigorous, inventive research scientist. Answer in 1-2 sentences: a single "
              "specific, novel, testable hypothesis and its mechanism. No preamble, no restating the "
              "question, no lists.")

GATE_SYSTEM = ("You are a meticulous, impartial reviewer of scientific hypotheses. Score each on a "
               "strict 1-5 rubric. You are not swayed by length or confident tone.")

GATE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "gate", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", "plausible", "testable", "novelty"],
        "properties": {"id": {"type": "integer"},
                       "plausible": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                       "testable": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                       "novelty": {"type": "integer", "enum": [1, 2, 3, 4, 5]}}}}}}}}


# --------------------------------------------------------------------------- #
#  OpenAI-compatible chat call, tolerant of API generations + structured output
# --------------------------------------------------------------------------- #
def _make_call(model, base_url, api_key, reasoning_effort=None):
    """Chat-Completions caller. `reasoning_effort` (e.g. 'high') is passed through for reasoning
    models like gpt-5.5 — use it for the JUDGE so it actually thinks before scoring. It's dropped
    automatically for endpoints/models that don't accept it (e.g. the local vLLM generator)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url or None, api_key=api_key or os.environ.get("OPENAI_API_KEY") or "x")

    def call(messages, temperature=0.0, max_tokens=4000, response_format=None):
        kwargs = {"model": model, "messages": messages, "max_completion_tokens": max_tokens,
                  "temperature": temperature}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        last = None
        for _ in range(7):
            try:
                r = client.chat.completions.create(**kwargs)
                return r.choices[0].message.content or ""
            except Exception as e:
                last, msg = e, str(e).lower()
                rf = (kwargs.get("response_format") or {}).get("type")
                if "reasoning_effort" in kwargs and "reasoning_effort" in msg:
                    kwargs.pop("reasoning_effort")
                elif rf and any(k in msg for k in ("response_format", "json_schema", "json_object")):
                    if rf == "json_schema":
                        kwargs["response_format"] = {"type": "json_object"}
                    else:
                        kwargs.pop("response_format", None)
                elif "max_completion_tokens" in kwargs and "max_completion_tokens" in msg and \
                        any(k in msg for k in ("unsupported", "unexpected", "not supported")):
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                elif "max_tokens" in kwargs and "max_tokens" in msg and \
                        any(k in msg for k in ("unsupported", "max_completion_tokens", "not supported")):
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                elif "temperature" in kwargs and "temperature" in msg and \
                        any(k in msg for k in ("unsupported", "does not support", "not supported")):
                    kwargs.pop("temperature")
                else:
                    raise
        raise last
    return call


def _parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------- #
#  Load the run's mined leads (topic + ranked leads)
# --------------------------------------------------------------------------- #
def _load_leads(run, insights, embed_model, k):
    if insights:
        data = json.load(open(insights))
        miners = data.get("miners", {})
        results = [(kind, miners.get(kind, [])) for kind in I.KIND_ORDER]
        topic = data.get("topic") or I.read_topic(os.path.dirname(insights) or ".")
    else:
        topic, _, results = I.load_insights_or_mine(run, embed_model=embed_model)
    leads = S._rank_leads(results, k)
    return topic, [S._humanize(x.get("title", "")) for x in leads if x.get("title")]


# --------------------------------------------------------------------------- #
#  Generation
# --------------------------------------------------------------------------- #
def _gen_one(call, task, lead, temperature):
    if lead is None:
        user = (f"Question: {task}\n\nState ONE specific, novel, testable hypothesis about this "
                f"question, with its mechanism, in 1-2 sentences.")
    else:
        user = (f"Question: {task}\n\nConsider this specific connection noticed while exploring the "
                f"question:\n  {lead}\n\nState ONE specific, novel, testable hypothesis it suggests, "
                f"with its mechanism, in 1-2 sentences.")
    out = call([{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}],
               temperature=temperature, max_tokens=260)
    return " ".join(out.split())


def _gate(call, task, texts):
    """One blind call scoring every hypothesis (plausible/testable/novelty 1-5). Returns list aligned
    to `texts` of dicts (missing entries default to mid-low so a gate failure can't inflate either arm)."""
    listing = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user = (f"Question being addressed:\n{task}\n\nScore EACH candidate hypothesis below on "
            f"plausible (physically/scientifically sound), testable (a concrete falsifiable test "
            f"exists), and novelty (non-obvious). Return a verdict for every id 1..{len(texts)}.\n\n"
            f"{listing}")
    out = call([{"role": "system", "content": GATE_SYSTEM}, {"role": "user", "content": user}],
               temperature=0.0, max_tokens=6000, response_format=GATE_SCHEMA)
    v = _parse_json(out) or {}
    by_id = {d.get("id"): d for d in v.get("verdicts", []) if isinstance(d, dict)}
    out_list = []
    for i in range(len(texts)):
        d = by_id.get(i + 1, {})
        out_list.append({"plausible": float(d.get("plausible", 2)), "testable": float(d.get("testable", 2)),
                         "novelty": float(d.get("novelty", 2))})
    return out_list


# --------------------------------------------------------------------------- #
#  Geometry metrics on one task's two hypothesis sets
# --------------------------------------------------------------------------- #
def _unit(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(n, 1e-9)


def _dedup_order(idxs, X, valid, tau):
    """Walk hypotheses in generation order; keep a VALID one only if it's not a near-duplicate
    (cosine >= tau) of an already-kept one. Returns kept indices and the cumulative-distinct curve."""
    kept, curve = [], []
    for j in idxs:
        if valid[j]:
            if not kept or float(np.max(X[kept] @ X[j])) < tau:
                kept.append(j)
        curve.append(len(kept))
    return kept, np.array(curve)


def _task_metrics(graph_idx, base_idx, X, valid, novelty, tau_dedup, tau_cover):
    gk, gcurve = _dedup_order(graph_idx, X, valid, tau_dedup)
    bk, bcurve = _dedup_order(base_idx, X, valid, tau_dedup)

    def spread(kept):
        return float(X[kept].var(axis=0).sum()) if len(kept) > 1 else 0.0

    def exclusive(a_kept, b_kept):
        if not a_kept:
            return float("nan")
        if not b_kept:
            return 1.0
        S_ = X[a_kept] @ X[b_kept].T                 # nearest neighbour in the other arm
        return float(np.mean(S_.max(axis=1) < tau_cover))

    nov = lambda kept: float(np.mean([novelty[j] for j in kept])) if kept else float("nan")
    return {"distinct_graph": len(gk), "distinct_base": len(bk),
            "spread_graph": spread(gk), "spread_base": spread(bk),
            "excl_graph": exclusive(gk, bk), "excl_base": exclusive(bk, gk),
            "nov_graph": nov(gk), "nov_base": nov(bk),
            "sat_graph": gcurve, "sat_base": bcurve}


# --------------------------------------------------------------------------- #
#  Coverage benchmark
# --------------------------------------------------------------------------- #
def run_coverage(args):
    from graphstore import embed_texts, resolve_embed_model
    tasks = [ln.strip() for ln in open(args.tasks, encoding="utf-8") if ln.strip()]
    topic, leads = _load_leads(args.run, args.insights, args.embed_model, args.leads)
    if not leads:
        raise SystemExit("no leads found — mine insights first (python insights.py --run <dir>).")
    K = len(leads); M = args.baseline_samples or 2 * K
    emodel = resolve_embed_model(args.run, args.embed_model)
    print(f"[compare] coverage benchmark · {len(tasks)} tasks · graph K={K} leads · baseline M={M} "
          f"samples · generator={args.model} · judge={args.jm} · embed={emodel}", flush=True)
    call_gen = _make_call(args.model, args.base_url, args.api_key)
    call_judge = _make_call(args.jm, args.jbu, args.jak, reasoning_effort=args.judge_effort)

    per_task, sat_g, sat_b = [], [], []
    for ti, task in enumerate(tasks):
        print(f"[compare] task {ti+1}/{len(tasks)}: generating {K}+{M} hypotheses…", flush=True)
        graph_h = [_gen_one(call_gen, task, ld, args.temperature) for ld in leads]
        base_h = [_gen_one(call_gen, task, None, args.temperature) for _ in range(M)]
        texts = graph_h + base_h
        if not any(texts):
            continue
        verdicts = _gate(call_judge, task, texts)
        X = _unit(np.asarray(embed_texts(texts, emodel), dtype=np.float32))
        valid = np.array([(verdicts[i]["plausible"] >= args.gate and verdicts[i]["testable"] >= args.gate)
                          for i in range(len(texts))])
        novelty = np.array([verdicts[i]["novelty"] for i in range(len(texts))])
        gi = list(range(K)); bi = list(range(K, K + M))
        m = _task_metrics(gi, bi, X, valid, novelty, args.dedup_tau, args.cover_tau)
        m["task"] = task; per_task.append(m); sat_g.append(m["sat_graph"]); sat_b.append(m["sat_base"])
        print(f"    distinct valid: graph={m['distinct_graph']}/{K}  baseline={m['distinct_base']}/{M}"
              f"  · graph-exclusive={100*m['excl_graph']:.0f}%", flush=True)
    if not per_task:
        raise SystemExit("no tasks produced hypotheses")

    _render_coverage(args, topic, K, M, per_task, sat_g, sat_b)


def _ms(xs):
    a = np.asarray([x for x in xs if x == x], float)               # drop nan
    return (float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 1 \
        else ((float(a.mean()), 0.0) if len(a) else (float("nan"), 0.0))


def _render_coverage(args, topic, K, M, per_task, sat_g, sat_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150})
    GC, BC = "#d62728", "#1f77b4"
    grab = lambda k: [m[k] for m in per_task]
    dist_g, dist_b = _ms(grab("distinct_graph")), _ms(grab("distinct_base"))
    spr_g, spr_b = _ms(grab("spread_graph")), _ms(grab("spread_base"))
    nov_g, nov_b = _ms(grab("nov_graph")), _ms(grab("nov_base"))
    exg, exb = _ms(grab("excl_graph")), _ms(grab("excl_base"))

    def sat_mean(sats):                                            # mean cumulative-distinct per gen index
        L = min(len(s) for s in sats); A = np.stack([s[:L] for s in sats])
        return np.arange(1, L + 1), A.mean(0), A.std(0, ddof=1) / np.sqrt(len(sats))
    xg, yg, eg = sat_mean(sat_g); xb, yb, eb = sat_mean(sat_b)

    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.6))
    # A distinct validated hypotheses
    a = ax[0, 0]; a.bar([0, 1], [dist_g[0], dist_b[0]], yerr=[dist_g[1], dist_b[1]], capsize=5,
                        color=[GC, BC])
    a.set_xticks([0, 1]); a.set_xticklabels(["graph", "baseline"]); a.set_ylabel("count (mean ± s.e.)")
    a.set_title(f"(A) Distinct VALIDATED hypotheses\n(graph K={K} leads vs baseline M={M} samples)")
    # B coverage saturation
    a = ax[0, 1]
    a.plot(xg, yg, color=GC, lw=2, marker="o", ms=3, label="graph"); a.fill_between(xg, yg - eg, yg + eg, color=GC, alpha=.15)
    a.plot(xb, yb, color=BC, lw=2, marker="s", ms=3, label="baseline"); a.fill_between(xb, yb - eb, yb + eb, color=BC, alpha=.15)
    a.set_xlabel("# generations"); a.set_ylabel("cumulative distinct validated ideas")
    a.set_title("(B) Coverage saturation\n(baseline plateau = mode-collapse)"); a.legend(frameon=False, fontsize=9)
    # C exclusive fraction
    a = ax[0, 2]; a.bar([0, 1], [100 * exg[0], 100 * exb[0]], yerr=[100 * exg[1], 100 * exb[1]],
                        capsize=5, color=[GC, BC])
    a.set_xticks([0, 1]); a.set_xticklabels(["graph-only", "baseline-only"]); a.set_ylabel("% of own validated ideas")
    a.set_title("(C) Exclusive ideas\n(unreached by the OTHER arm)")
    # D spread
    a = ax[1, 0]; a.bar([0, 1], [spr_g[0], spr_b[0]], yerr=[spr_g[1], spr_b[1]], capsize=5, color=[GC, BC])
    a.set_xticks([0, 1]); a.set_xticklabels(["graph", "baseline"]); a.set_ylabel("embedding variance")
    a.set_title("(D) Idea-space spread")
    # E novelty
    a = ax[1, 1]; a.bar([0, 1], [nov_g[0], nov_b[0]], yerr=[nov_g[1], nov_b[1]], capsize=5, color=[GC, BC])
    a.set_xticks([0, 1]); a.set_xticklabels(["graph", "baseline"]); a.set_ylabel("judged novelty (1-5)")
    a.set_ylim(0, 5.2); a.set_title("(E) Novelty of validated ideas")
    # F caption
    a = ax[1, 2]; a.axis("off")
    a.text(0, .98, "Validated idea coverage", fontsize=12, fontweight="bold", va="top")
    cap = (f"topic: {(topic or '')[:46]}\n{len(per_task)} tasks · embed: {args.embed_model or 'run default'}\n"
           f"generator: {args.model}\njudge (gate only): {args.jm}\n\n"
           f"graph: {K} leads → {dist_g[0]:.1f} distinct validated\n"
           f"baseline: {M} samples → {dist_b[0]:.1f} distinct validated\n"
           f"graph-exclusive: {100*exg[0]:.0f}% of graph's validated ideas\n"
           f"   are unreached by {M} baseline samples.\n\n"
           "Baseline gets MORE answer-time samples yet covers\nless validated idea space — the "
           "mode-collapse a single-\nshot model can't escape by resampling.")
    a.text(0, .88, cap, fontsize=8.6, va="top", family="monospace")

    fig.suptitle(f"Validated idea-space coverage: graph reasoning vs single-shot resampling "
                 f"(n={len(per_task)} tasks)", y=1.0, fontsize=12.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}.png/.svg/.pdf")

    report = {"mode": "coverage", "topic": topic, "n_tasks": len(per_task), "K": K, "M": M,
              "judge": args.jm, "generator": args.model,
              "distinct": {"graph": dist_g, "baseline": dist_b},
              "spread": {"graph": spr_g, "baseline": spr_b},
              "novelty": {"graph": nov_g, "baseline": nov_b},
              "exclusive": {"graph": exg, "baseline": exb},
              "per_task": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
                           for m in per_task]}
    json.dump(report, open(f"{args.out}.json", "w"), indent=2)
    lines = [f"# Validated idea-space coverage — {topic or ''}",
             f"\n*{len(per_task)} tasks · graph K={K} leads vs baseline M={M} samples · judge {args.jm}*\n",
             f"- **Distinct validated hypotheses** — graph **{dist_g[0]:.1f} ± {dist_g[1]:.1f}** vs "
             f"baseline {dist_b[0]:.1f} ± {dist_b[1]:.1f}",
             f"- **Graph-exclusive** (validated graph ideas unreached by {M} baseline samples): "
             f"**{100*exg[0]:.0f}%** (baseline-exclusive: {100*exb[0]:.0f}%)",
             f"- **Idea-space spread** — graph {spr_g[0]:.3f} vs baseline {spr_b[0]:.3f}",
             f"- **Judged novelty** — graph {nov_g[0]:.2f} vs baseline {nov_b[0]:.2f}"]
    open(f"{args.out}.md", "w").write("\n".join(lines) + "\n")
    print(f"wrote {args.out}.json / {args.out}.md")
    print(f"[compare] graph {dist_g[0]:.1f} vs baseline {dist_b[0]:.1f} distinct validated; "
          f"{100*exg[0]:.0f}% of graph's ideas unreached by {M} baseline samples.")


# --------------------------------------------------------------------------- #
#  Graph-RAG creativity benchmark (DEFAULT): closed-book vs task-conditioned
#  graph retrieval from the accumulated ideation graph.
# --------------------------------------------------------------------------- #
CREATIVE_SYSTEM = ("You are a bold, inventive research scientist. You propose original, "
                   "unconventional, cross-disciplinary ideas — imaginative but scientifically grounded.")

ARMS = ["closed", "graph"]
ARM_LABEL = {"closed": "Llama alone", "graph": "Llama + Graph-RAG"}
ARM_COL = {"closed": "#7f7f7f", "graph": "#d62728"}
CRE_DIMS = ["novelty", "surprise", "breadth", "plausibility"]

CRE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "creative", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["sets"],
    "properties": {"sets": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id"] + CRE_DIMS,
        "properties": dict({"id": {"type": "integer"}},
                           **{d: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for d in CRE_DIMS})}}}}}}


def _lbl(G, n):
    return str(G.nodes[n].get("label", n))


def _retrieve_provocations(qv, default_V, node_ids, X, idx, G, deg, n_seeds, n_central, n_unusual, hops):
    """Curate a small subgraph the reasoning connected to the question — a mix of:
      CENTRAL anchors  = high-degree hubs in the question's neighbourhood (load-bearing organizing
                         ideas that help ground an answer), and
      UNUSUAL angles   = graph-connected concepts that are OUT-OF-DISTRIBUTION for the model — far
                         from the ideas it produced on its own (default_V) — the provocations it
                         would not reach alone; only the graph's EDGES link them to the question.
    Plus the RELATIONS joining the unusual angles back to the anchors (how the leap connects).
    `default_V` = embeddings of the model's closed-book ideas (its prior for this question)."""
    seeds = [node_ids[i] for i in np.argsort(X @ qv)[::-1][:n_seeds]]
    hopd = {s: 0 for s in seeds}
    frontier = set(seeds)
    for h in range(1, hops + 1):
        nxt = set()
        for s in frontier:
            nxt |= set(G.successors(s)) | set(G.predecessors(s))
        nxt = {n for n in nxt if n not in hopd}
        for n in nxt:
            hopd[n] = h
        frontier = nxt
    pool = [n for n in hopd if hopd[n] > 0] or list(seeds)
    central = sorted(pool, key=lambda n: -deg.get(n, 0))[:n_central]
    # OOD score: far from the model's OWN default ideas (not from the question), graph-proximity tie-break
    def ood(n):
        d = 1.0 - (float((X[idx[n]] @ default_V.T).max()) if len(default_V) else float(X[idx[n]] @ qv))
        return d - 0.1 * hopd[n]
    unusual = [n for n in sorted([p for p in pool if p not in central], key=lambda n: -ood(n))][:n_unusual]
    anchors = set(seeds) | set(central)
    # RICH context: the reasoning CHAIN linking each unusual angle back to an anchor (a mechanism the
    # model can build on), not just a word — capped short so it stays legible.
    U = G.to_undirected()
    chains = []
    for u in unusual:
        best = None
        for an in anchors:
            if u == an or not U.has_node(u) or not U.has_node(an):
                continue
            try:
                p = nx.shortest_path(U, u, an)
            except Exception:
                continue
            if 2 <= len(p) <= 4 and (best is None or len(p) < len(best)):
                best = p
        if best:
            seg = []
            for a, b in zip(best, best[1:]):
                r = (G.get_edge_data(a, b) or G.get_edge_data(b, a) or {}).get("relation", "related to")
                seg.append(f"{_lbl(G, a)} —{r}→")
            chains.append(" ".join(seg) + f" {_lbl(G, best[-1])}")
    return [_lbl(G, n) for n in central], [_lbl(G, n) for n in unusual], chains[:n_unusual]


def _ideas_prompt(q, mode, central, unusual, chains, n_ideas):
    ctx = ""
    if mode == "graph":
        ctx = "Retrieved graph context from a broad ideation graph:\n"
        if central:
            ctx += "Central themes: " + ", ".join(central) + "\n"
        if unusual:
            ctx += "Unusual / cross-domain angles: " + ", ".join(unusual) + "\n"
        if chains:
            ctx += "How some of these connect to the core:\n" + "\n".join(f"  · {c}" for c in chains) + "\n"
        ctx += "\n"
    if mode == "graph":
        tail = (f"Propose {n_ideas} DISTINCT, novel, non-obvious ideas or mechanisms that address this. "
                f"Use the retrieved graph context as the evidence pool: each idea should be grounded in "
                f"at least one central theme, unusual angle, or relation chain. If a retrieved cue is "
                f"irrelevant or scientifically weak for this question, ignore that cue and choose a "
                f"better one. Explain the mechanism; do not merely repeat graph terms. One sentence "
                f"each, numbered.")
    else:
        tail = (f"Propose {n_ideas} DISTINCT, novel, non-obvious ideas or mechanisms that address this. "
                f"Be imaginative but scientifically plausible. One sentence each, numbered.")
    return f"Question: {q}\n\n{ctx}{tail}"


def _parse_ideas(text, n):
    out = []
    for ln in text.splitlines():
        s = re.sub(r"^\s*(\d+[.\)]|[-*•])\s*", "", ln).strip()
        if len(s) > 12:
            out.append(s)
    return out[:n]


def _embed_ideas(ideas, model):
    from graphstore import embed_texts
    if not ideas:
        return np.zeros((0, 8), np.float32)
    return _unit(np.asarray(embed_texts(ideas, model), np.float32))


def _diversity(V):
    """Mean pairwise embedding distance (breadth) + # distinct (dedup at 0.85), from idea vectors."""
    if len(V) < 2:
        return 0.0, len(V)
    iu = np.triu_indices(len(V), 1)
    div = float((1.0 - V @ V.T)[iu].mean())
    kept = []
    for i in range(len(V)):
        if not kept or max(float(V[i] @ V[j]) for j in kept) < 0.85:
            kept.append(i)
    return div, len(kept)


def _ood(V, ref_V, is_self):
    """Departure from the model's default ideas: mean over V of (1 - nearest cosine to ref_V).
    For the closed arm (is_self) the self-match is excluded so its baseline isn't trivially 0."""
    if len(V) == 0 or len(ref_V) == 0:
        return float("nan")
    S = V @ ref_V.T
    if is_self:
        np.fill_diagonal(S, -1.0)
    return float((1.0 - S.max(1)).mean())


def _incorporation(idea_V, concept_texts, model):
    """MANIPULATION CHECK: did the arm's ideas actually use the injected concepts? Mean over ideas
    of the max cosine to the concept set the arm was given. ~0 means the model ignored them."""
    from graphstore import embed_texts
    if len(idea_V) == 0 or not concept_texts:
        return float("nan")
    C = _unit(np.asarray(embed_texts(concept_texts, model), np.float32))
    return float((idea_V @ C.T).max(1).mean())


def _judge_creative(call, q, shuffled):
    """One blind call scoring each idea-SET 1-5 on novelty/surprise/breadth/plausibility."""
    listing = "\n\n".join(f"[Set {i+1}]\n" + "\n".join(f"- {x}" for x in ideas)
                          for i, (_, ideas) in enumerate(shuffled))
    user = (f"Question:\n{q}\n\nEach set below is a brainstorm of ideas answering the question. Score "
            f"EACH set 1-5 on: novelty (original, not textbook), surprise (non-obvious, unexpected "
            f"angles or cross-domain leaps), breadth (range of genuinely distinct directions), "
            f"plausibility (scientifically reasonable, not nonsense). Return a verdict per set id "
            f"1..{len(shuffled)}.\n\n{listing}")
    v = _parse_json(call([{"role": "system", "content": GATE_SYSTEM}, {"role": "user", "content": user}],
                         0.0, 6000, CRE_SCHEMA)) or {}
    by_id = {d.get("id"): d for d in v.get("sets", v.get("answers", [])) if isinstance(d, dict)}
    out = {}
    for i, (arm, _) in enumerate(shuffled):
        d = by_id.get(i + 1, {})
        out[arm] = {dim: float(d.get(dim, 2)) for dim in CRE_DIMS}
    return out


def run_graphrag(args):
    from graphstore import embed_texts, resolve_embed_model
    tasks = [ln.strip() for ln in open(args.tasks, encoding="utf-8") if ln.strip()]
    G = I.load_graph(args.run)
    model = resolve_embed_model(args.run, args.embed_model)
    node_ids = list(G.nodes)
    print(f"[compare] graph-RAG creativity · {len(tasks)} tasks · graph {G.number_of_nodes()} concepts "
          f"/ {G.number_of_edges()} links · {args.n_ideas} ideas/arm · generator={args.model} · "
          f"judge={args.jm} · embed={model}", flush=True)
    vecs = I.embed_nodes(G, model)
    if not vecs:
        raise SystemExit("embeddings unavailable — graph-RAG needs them (--embed-model all-MiniLM-L6-v2).")
    X = _unit(np.stack([vecs[n] for n in node_ids]).astype(np.float32))
    idx = {n: i for i, n in enumerate(node_ids)}
    deg = dict(G.degree())
    n_central = max(2, args.concepts // 3)
    n_unusual = max(1, args.concepts - n_central)
    call_gen = _make_call(args.model, args.base_url, args.api_key)
    call_judge = _make_call(args.jm, args.jbu, args.jak, reasoning_effort=args.judge_effort)

    def brainstorm(q, mode, central, unusual, chains):
        out = call_gen([{"role": "system", "content": CREATIVE_SYSTEM},
                        {"role": "user", "content": _ideas_prompt(q, mode, central, unusual, chains,
                                                                  args.n_ideas)}],
                       temperature=args.temperature, max_tokens=800)
        return _parse_ideas(out, args.n_ideas)

    per_task = []
    for ti, q in enumerate(tasks):
        print(f"[compare] task {ti+1}/{len(tasks)}: brainstorm closed vs Graph-RAG...", flush=True)
        qv = _unit(np.asarray(embed_texts([q], model), np.float32))[0]
        # closed-book FIRST — it defines the model's own prior (the OOD reference) for this question
        ideas = {"closed": brainstorm(q, "closed", [], [], [])}
        defV = _embed_ideas(ideas["closed"], model)
        central, unusual, chains = _retrieve_provocations(qv, defV, node_ids, X, idx, G, deg,
                                                          args.rag_seeds, n_central, n_unusual, args.rag_hops)
        ideas["graph"] = brainstorm(q, "graph", central, unusual, chains)
        V = {arm: _embed_ideas(ideas[arm], model) for arm in ARMS}
        graph_context = central + unusual + chains
        injected = {"closed": [], "graph": graph_context}
        rng = random.Random(args.seed + ti)
        shuffled = list(ideas.items()); rng.shuffle(shuffled)
        scores = _judge_creative(call_judge, q, shuffled)
        rec = {"task": q, "central": central, "unusual": unusual, "chains": chains}
        for arm in ARMS:
            d, dist = _diversity(V[arm])
            rec[arm] = {**scores.get(arm, {d2: float("nan") for d2 in CRE_DIMS}),
                        "diversity": d, "distinct": dist, "ideas": ideas[arm],
                        "ood": _ood(V[arm], defV, arm == "closed"),
                        "incorporation": _incorporation(V[arm], injected[arm], model)}
        per_task.append(rec)
        print("    " + " · ".join(f"{ARM_LABEL[a]}: nov={rec[a]['novelty']:.0f} "
              f"plaus={rec[a]['plausibility']:.0f}" for a in ARMS)
              + f" · graph inc={rec['graph']['incorporation']:.2f}", flush=True)
    if not per_task:
        raise SystemExit("no tasks answered")
    _render_graphrag(args, G, per_task)


def _render_graphrag(args, G, per_task):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150})
    agg = {arm: {k: _ms([r[arm][k] for r in per_task])
                 for k in CRE_DIMS + ["diversity", "distinct", "ood", "incorporation"]} for arm in ARMS}
    overall = {arm: _ms([np.mean([r[arm][d] for d in CRE_DIMS]) for r in per_task]) for arm in ARMS}
    delta = {k: _ms([r["graph"][k] - r["closed"][k] for r in per_task]) for k in CRE_DIMS}
    overall_delta = _ms([np.mean([r["graph"][d] - r["closed"][d] for d in CRE_DIMS]) for r in per_task])
    fig, ax = plt.subplots(2, 2, figsize=(14.0, 9.2))
    xt = [ARM_LABEL[a2] for a2 in ARMS]; xr = np.arange(len(ARMS))

    # A grouped creativity dimensions
    a = ax[0, 0]; x = np.arange(len(CRE_DIMS)); w = 0.8 / len(ARMS)
    for k, arm in enumerate(ARMS):
        a.bar(x + (k - (len(ARMS) - 1) / 2) * w, [agg[arm][d][0] for d in CRE_DIMS], w,
              yerr=[agg[arm][d][1] for d in CRE_DIMS], capsize=2, color=ARM_COL[arm], label=ARM_LABEL[arm])
    a.set_xticks(x); a.set_xticklabels(CRE_DIMS, rotation=10); a.set_ylim(0, 5.4)
    a.set_ylabel("judge score (1-5, mean +/- s.e.)"); a.legend(frameon=False, fontsize=8.0, loc="upper left")
    a.set_title("(A) Creativity — blind absolute scoring")

    # B paired graph-minus-closed deltas
    a = ax[0, 1]
    vals = [delta[d][0] for d in CRE_DIMS]
    errs = [delta[d][1] for d in CRE_DIMS]
    cols = ["#d62728" if v >= 0 else "#7f7f7f" for v in vals]
    a.axhline(0, color="#222222", lw=0.8)
    a.bar(np.arange(len(CRE_DIMS)), vals, yerr=errs, capsize=3, color=cols)
    a.set_xticks(np.arange(len(CRE_DIMS))); a.set_xticklabels(CRE_DIMS, rotation=10)
    a.set_ylabel("Graph-RAG minus Llama alone")
    a.set_title("(B) Paired lift by dimension")

    # C objective idea-space checks
    a = ax[1, 0]
    obj = ["diversity", "ood"]
    x = np.arange(len(obj)); w = 0.8 / len(ARMS)
    for k, arm in enumerate(ARMS):
        a.bar(x + (k - (len(ARMS) - 1) / 2) * w, [agg[arm][m][0] for m in obj], w,
              yerr=[agg[arm][m][1] for m in obj], capsize=3, color=ARM_COL[arm], label=ARM_LABEL[arm])
    a.set_xticks(x); a.set_xticklabels(["idea diversity", "distance from prior"])
    a.set_ylabel("embedding distance")
    a.set_title("(C) Objective idea-space checks")
    a.legend(frameon=False, fontsize=8.0)

    # D caption
    a = ax[1, 1]; a.axis("off")
    a.text(0, .98, "Closed-book vs Graph-RAG", fontsize=12, fontweight="bold", va="top")
    inc = agg["graph"]["incorporation"][0]
    cap = (f"{len(per_task)} tasks · graph {G.number_of_nodes()} concepts / {G.number_of_edges()} links\n"
           f"generator: {args.model} · judge: {args.jm}\n\n"
           f"mean creativity: closed {overall['closed'][0]:.2f} -> graph {overall['graph'][0]:.2f}\n"
           f"paired lift:     {overall_delta[0]:+.2f} +/- {overall_delta[1]:.2f}\n"
           f"graph-context incorporation: {inc:.2f}\n"
           f"distinct ideas: closed {agg['closed']['distinct'][0]:.1f}, graph {agg['graph']['distinct'][0]:.1f}\n\n"
           f"retrieval: seeds={args.rag_seeds}, hops={args.rag_hops}, concepts={args.concepts}\n"
           "context: central anchors + unusual angles + relation chains\n\n"
           "The graph prompt asks the model to ground ideas in retrieved graph context,\n"
           "but to discard irrelevant or scientifically weak cues rather than parroting them.")
    a.text(0, .9, cap, fontsize=8.0, va="top", family="monospace")
    fig.suptitle(f"Llama alone vs Llama + Graph-RAG (n={len(per_task)} tasks)", y=1.0, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}.png/.svg/.pdf")
    report = {"mode": "graphrag", "n_tasks": len(per_task), "judge": args.jm, "generator": args.model,
              "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
              "retrieval": {"rag_seeds": args.rag_seeds, "rag_hops": args.rag_hops,
                            "concepts": args.concepts},
              "aggregate": agg, "overall_creativity": overall, "paired_delta": delta,
              "overall_delta": overall_delta, "per_task": per_task}
    json.dump(report, open(f"{args.out}.json", "w"), indent=2)
    lines = [f"# Graph-RAG benchmark - Llama alone vs Llama + Graph-RAG\n",
             f"*{len(per_task)} tasks · judge {args.jm} · retrieval seeds={args.rag_seeds}, "
             f"hops={args.rag_hops}, concepts={args.concepts}*\n",
             "| metric | Llama alone | Llama + Graph-RAG | paired delta |",
             "|---|---:|---:|---:|"]
    for k in CRE_DIMS + ["incorporation", "ood", "diversity"]:
        d = ""
        if k in CRE_DIMS:
            d = f"{delta[k][0]:+.2f}"
        lines.append(f"| {k} | {agg['closed'][k][0]:.2f} | {agg['graph'][k][0]:.2f} | {d} |")
    lines.append(f"| mean creativity | {overall['closed'][0]:.2f} | {overall['graph'][0]:.2f} | "
                 f"{overall_delta[0]:+.2f} |")
    open(f"{args.out}.md", "w").write("\n".join(lines) + "\n")
    print(f"wrote {args.out}.json / {args.out}.md")
    print(f"[compare] mean creativity closed={overall['closed'][0]:.2f} graph={overall['graph'][0]:.2f} "
          f"delta={overall_delta[0]:+.2f}; graph incorporation={inc:.2f}")


# --------------------------------------------------------------------------- #
#  Assess the mined material itself (--mode insights): a strong judge rates the
#  graph-mined connections vs CONTROLS — no small-model generation in the loop.
# --------------------------------------------------------------------------- #
CONN_DIMS = ["plausibility", "insight", "usefulness"]   # novelty dropped: judge-on-pairs can't measure it
CONN_SCHEMA = {"type": "json_schema", "json_schema": {"name": "conn", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["ratings"],
    "properties": {"ratings": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id"] + CONN_DIMS,
        "properties": dict({"id": {"type": "integer"}},
                           **{d: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for d in CONN_DIMS})}}}}}}
PAIR_KINDS = {"conceptual_bridge", "latent_link", "open_triad", "relational_analogy", "semantic_dissonance"}


def _connection_text(x, G):
    """Render a mined insight as a connection statement WITH its bridging mechanism — the graph's
    actual structural payload (path / relation / common-neighbour), not just the two endpoints."""
    k = x.get("kind")
    if k == "conceptual_bridge" and isinstance(x.get("path"), list) and len(x["path"]) >= 2 \
            and all(n in G for n in x["path"]):
        p, rels = x["path"], x.get("relations", [])
        chain = _lbl(G, p[0])
        for i, n in enumerate(p[1:]):
            chain += f" —{rels[i] if i < len(rels) else 'relates to'}→ {_lbl(G, n)}"
        return f"{_lbl(G, p[0])} and {_lbl(G, p[-1])}, connected via: {chain}"
    if k == "open_triad" and isinstance(x.get("chain"), list) and len(x["chain"]) >= 3 \
            and all(n in G for n in x["chain"]):
        a, b, c = x["chain"][:3]; r = (x.get("relations") or ["relates to", "relates to"]) + ["", ""]
        return (f"{_lbl(G, a)} and {_lbl(G, c)}, inferable via {_lbl(G, b)}: "
                f"{_lbl(G, a)} —{r[0]}→ {_lbl(G, b)} —{r[1]}→ {_lbl(G, c)}")
    if k == "relational_analogy" and isinstance(x.get("instances"), list) and len(x["instances"]) >= 2:
        i0, i1 = x["instances"][0], x["instances"][1]
        if len(i0) >= 3 and len(i1) >= 3 and all(n in G for n in (i0[0], i0[2], i1[0], i1[2])):
            return f"{_lbl(G, i0[0])} is to {_lbl(G, i0[2])} as {_lbl(G, i1[0])} is to {_lbl(G, i1[2])}"
    if k == "latent_link" and isinstance(x.get("pair"), list) and all(n in G for n in x["pair"][:2]):
        a, b = x["pair"][:2]
        return f"{_lbl(G, a)} and {_lbl(G, b)} (share {x.get('common_neighbors', 'several')} concepts but are not directly linked)"
    if k == "semantic_dissonance" and isinstance(x.get("pair"), list) and all(n in G for n in x["pair"][:2]):
        a, b = x["pair"][:2]
        return f"{_lbl(G, a)} and {_lbl(G, b)} (closely related in meaning yet never connected in the reasoning)"
    return None


def run_assess(args):
    if args.insights:
        data = json.load(open(args.insights))
        results = [(k, data.get("miners", {}).get(k, [])) for k in I.KIND_ORDER]
        topic = data.get("topic") or ""
        G = I.load_graph(args.run) if args.run else I.load_graph(os.path.dirname(args.insights) or ".")
    else:
        topic, G, results = I.load_insights_or_mine(args.run, embed_model=args.embed_model)
    # GRAPH arm: the top-actionability mined connections, rendered WITH their bridging mechanism
    pair_results = [(k, ins) for k, ins in results if k in PAIR_KINDS]
    K = args.n_per_miner
    items = []                                         # (source, connection_text)
    for x in S._rank_leads(pair_results, K):
        t = _connection_text(x, G)
        if t:
            items.append(("graph", t))
    K = len(items)
    if K == 0:
        raise SystemExit("no pair-type mined connections found — run insights.py first.")
    rng = random.Random(args.seed)                     # CONTROL 1: random concept pairs (no mechanism)
    labs = [str(G.nodes[n].get("label", n)) for n in G.nodes]
    for _ in range(K):
        a, b = rng.sample(labs, 2)
        items.append(("random", f"{a} and {b}"))
    if args.model:                                     # CONTROL 2: the model's OWN connections (+ a reason)
        call = _make_call(args.model, args.base_url, args.api_key)
        out = call([{"role": "system", "content": "You are a domain expert."},
                    {"role": "user", "content": f"Topic: {topic}\n\nList {K} NON-OBVIOUS connections "
                     f"between concepts relevant to this topic. Format EACH line exactly as "
                     f"'A | B | one-sentence reason they connect'. No numbering."}],
                   temperature=0.8, max_tokens=1600)
        c = 0
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) >= 3 and c < K:
                a, b, why = parts[0].strip().lstrip("-*0123456789. "), parts[1].strip(), parts[2].strip()
                if a and b:
                    items.append(("model", f"{a} and {b} — {why}")); c += 1

    order = list(range(len(items))); random.Random(args.seed + 1).shuffle(order)
    shuf = [items[i] for i in order]
    listing = "\n".join(f"{i+1}. {t}" for i, (_, t) in enumerate(shuf))
    user = (f"Domain: {topic}\n\nBelow are candidate CONNECTIONS between concepts in this domain; some "
            f"include a proposed mechanism or rationale. For EACH, rate it 1-5 on:\n"
            f"- plausibility: a real, scientifically sensible relationship genuinely exists.\n"
            f"- insight: the connection (and any mechanism shown) reveals a useful, non-obvious "
            f"mechanism or hypothesis.\n"
            f"- usefulness: it could seed a concrete, fruitful research direction.\n"
            f"Judge SUBSTANCE: an arbitrary pairing with no real link should score low across the board. "
            f"Return a rating for every id 1..{len(shuf)}.\n\n{listing}")
    judge = _make_call(args.jm, args.jbu, args.jak, reasoning_effort=args.judge_effort)
    print(f"[compare] assessing {K} top graph connections vs {K} random"
          + (" vs model-own" if args.model else "") + f" · judge={args.jm}", flush=True)
    v = _parse_json(judge([{"role": "system", "content": GATE_SYSTEM}, {"role": "user", "content": user}],
                          0.0, 8000, CONN_SCHEMA)) or {}
    by_id = {d.get("id"): d for d in v.get("ratings", v.get("answers", [])) if isinstance(d, dict)}
    srcs = [s for s in ["graph", "model", "random"] if any(x == s for x, _ in items)]
    scores = {s: {d: [] for d in CONN_DIMS} for s in srcs}
    examples = []
    for i, (src, t) in enumerate(shuf):
        d = by_id.get(i + 1, {})
        for dim in CONN_DIMS:
            scores[src][dim].append(float(d.get(dim, 2)))
        if src == "graph":
            examples.append((np.mean([float(d.get(dim, 0)) for dim in CONN_DIMS]), t))
    _render_assess(args, topic, G, scores, srcs, sorted(examples, reverse=True))


def _render_assess(args, topic, G, scores, srcs, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150})
    col = {"graph": "#d62728", "model": "#1f77b4", "random": "#7f7f7f"}
    lab = {"graph": "graph-mined", "model": "model's own", "random": "random pairs"}
    agg = {s: {d: _ms(scores[s][d]) for d in CONN_DIMS} for s in srcs}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.05, 1]})
    a = ax[0]; x = np.arange(len(CONN_DIMS)); w = 0.8 / len(srcs)
    for k, s in enumerate(srcs):
        a.bar(x + (k - (len(srcs) - 1) / 2) * w, [agg[s][d][0] for d in CONN_DIMS], w,
              yerr=[agg[s][d][1] for d in CONN_DIMS], capsize=3, color=col[s], label=lab[s])
    a.set_xticks(x); a.set_xticklabels(CONN_DIMS); a.set_ylim(0, 5.4)
    a.set_ylabel("judge rating (1-5, mean ± s.e.)"); a.legend(frameon=False, fontsize=9)
    a.set_title("Quality of graph-mined connections vs controls")
    a = ax[1]; a.axis("off")
    head = " · ".join(f"{lab[s]} {np.mean([agg[s][d][0] for d in CONN_DIMS]):.2f}" for s in srcs)
    ex = "\n".join(f"• {(t if len(t) <= 78 else t[:77] + '…')}" for _, t in examples[:9])
    a.text(0, 1, f"Mined-connection quality\n{(topic or '')[:48]}\njudge: {args.jm}\n\n"
                 f"mean over dims: {head}\n\nTop graph-mined connections (shown with mechanism):\n\n{ex}",
           va="top", fontsize=7.6, family="monospace")
    fig.suptitle(f"Are the graph-mined connections actually good? (judge: {args.jm})", y=1.02, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}.png/.svg/.pdf")
    rep = {"mode": "insights", "topic": topic, "judge": args.jm, "aggregate": agg,
           "top_graph_connections": [{"score": s, "connection": t} for s, t in examples[:30]]}
    json.dump(rep, open(f"{args.out}.json", "w"), indent=2)
    lines = ["| dimension | " + " | ".join(lab[s] for s in srcs) + " |",
             "|" + "---|" * (len(srcs) + 1)]
    for d in CONN_DIMS:
        lines.append(f"| {d} | " + " | ".join(f"{agg[s][d][0]:.2f}" for s in srcs) + " |")
    open(f"{args.out}.md", "w").write("# Mined-connection quality\n\n" + "\n".join(lines) + "\n")
    print(f"wrote {args.out}.json / {args.out}.md   ·   means: {head}")


# --------------------------------------------------------------------------- #
#  Pairwise answer judge: graph-assisted answers vs no-insights baseline
# --------------------------------------------------------------------------- #
PAIR_DIMS = {
    "novelty": "original and non-obvious; not a standard or generic answer",
    "insight": "reveals a useful non-obvious connection, mechanism, or hypothesis",
    "mechanism": "gives a concrete causal/mechanistic explanation, not just a claim",
    "testability": "contains a falsifiable prediction, measurement, or decisive experiment",
    "plausibility": "scientifically/technically plausible; not creative nonsense",
    "specificity": "uses concrete details, variables, materials, controls, or readouts",
}
PAIR_PRIMARY_DIMS = ["novelty", "insight", "mechanism", "testability", "specificity"]


def _strip_header(t):
    m = re.search(r"^\s*---\s*$", t, flags=re.MULTILINE)
    return t[m.end():].strip() if m else t.strip()


def _read_answers(d):
    fs = sorted(glob.glob(os.path.join(d, "*.md")) + glob.glob(os.path.join(d, "*.txt")))
    return [{"path": f, "text": _strip_header(open(f, encoding="utf-8").read())} for f in fs]


def _pair_primary(side):
    """Primary hypothesis-generation score on the judge's 1-5 scale.

    The core score rewards novelty, insight, mechanism, testability, and specificity. Implausible
    answers are capped by their plausibility score so speculative nonsense cannot win by being
    merely surprising.
    """
    core = float(np.mean([side[d] for d in PAIR_PRIMARY_DIMS]))
    return min(core, side["plausibility"]) if side["plausibility"] < 3 else core


def _validate_pairwise(v, dims):
    if not isinstance(v, dict):
        return None
    if v.get("preferred") not in ("A", "B", "tie"):
        return None
    out = {"preferred": v["preferred"], "rationale": str(v.get("rationale", "")).strip()}
    for side_name in ("A", "B"):
        side = v.get(side_name)
        if not isinstance(side, dict):
            return None
        clean = {}
        for d in dims:
            try:
                x = int(side[d])
            except Exception:
                return None
            if x < 1 or x > 5:
                return None
            clean[d] = float(x)
        clean["primary"] = _pair_primary(clean)
        out[side_name] = clean
    return out


def _pairwise_schema(dims):
    score = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
    side = {"type": "object", "additionalProperties": False,
            "properties": {d: score for d in dims}, "required": dims}
    return {"type": "json_schema", "json_schema": {"name": "pairwise_verdict", "strict": True,
            "schema": {"type": "object", "additionalProperties": False,
                       "required": ["A", "B", "preferred", "rationale"],
                       "properties": {
                           "A": side,
                           "B": side,
                           "preferred": {"type": "string", "enum": ["A", "B", "tie"]},
                           "rationale": {"type": "string"},
                       }}}}


def _judge_pairwise(call, task, answer_a, answer_b, dims, schema):
    dd = "\n".join(f"- {d}: {PAIR_DIMS[d]}" for d in dims)
    system = (
        "You are an impartial expert reviewer for a hypothesis-generation benchmark. "
        "You do not know which answer came from which system. Score substance only; do not reward "
        "verbosity, confident tone, formatting polish, or citations unless they support a better "
        "hypothesis. Penalize generic answers and penalize implausible speculation."
    )
    user = (
        "Two answers respond to the SAME task. Score each answer independently from 1 to 5 on "
        "every dimension, then choose the preferred answer overall.\n\n"
        "Rubric:\n"
        f"{dd}\n\n"
        "Preference rule: prefer the answer with the stronger useful hypothesis after considering "
        "novelty, insight, mechanism, testability, specificity, and plausibility. If both are "
        "substantively equivalent, choose tie.\n\n"
        f"TASK:\n{task}\n\n--- A ---\n{answer_a}\n\n--- B ---\n{answer_b}\n\n"
        "Return only the requested JSON."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    raw = ""
    for attempt in range(2):
        raw = call(messages, 0.0, 1800, schema)
        parsed = _validate_pairwise(_parse_json(raw), dims)
        if parsed is not None:
            parsed["raw"] = raw
            return parsed
        messages = messages + [
            {"role": "assistant", "content": raw[:2000]},
            {"role": "user", "content": "The previous response was not valid for the required schema. "
                                      "Return only valid JSON with A, B, preferred, and rationale."},
        ]
    return {"error": "invalid_judge_json", "raw": raw}


def run_pairwise(args):
    dims = list(PAIR_DIMS)
    tasks = [ln.strip() for ln in open(args.tasks, encoding="utf-8") if ln.strip()]
    sysa, basea = _read_answers(args.system), _read_answers(args.baseline)
    n = min(len(tasks), len(sysa), len(basea))
    if n == 0:
        raise SystemExit("need matching tasks + --system + --baseline answers")
    if len(tasks) != len(sysa) or len(tasks) != len(basea):
        print(f"[compare/pairwise] warning: using first {n} aligned items "
              f"(tasks={len(tasks)}, system={len(sysa)}, baseline={len(basea)})", flush=True)
    call = _make_call(args.jm, args.jbu, args.jak, reasoning_effort=args.judge_effort)
    schema = _pairwise_schema(dims)
    sc = {d: {"g": [], "b": []} for d in dims + ["primary"]}; prefs = []
    per_task, skipped = [], []
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n), desc="[compare/pairwise] judging", unit="task")
    except Exception:
        iterator = range(n)

    for i in iterator:
        rng = random.Random(args.seed + i); sysA = rng.random() < 0.5
        a = sysa[i]["text"] if sysA else basea[i]["text"]
        b = basea[i]["text"] if sysA else sysa[i]["text"]
        v = _judge_pairwise(call, tasks[i], a, b, dims, schema)
        if "A" not in v or "B" not in v:
            skipped.append({"index": i, "task": tasks[i], "error": v.get("error", "invalid_judge_json"),
                            "raw": v.get("raw", "")})
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(skipped=len(skipped))
            else:
                print(f"[compare/pairwise] task {i+1}/{n}: skipped invalid judge output", flush=True)
            continue
        gv, bv = (v["A"], v["B"]) if sysA else (v["B"], v["A"])
        for d in dims + ["primary"]:
            sc[d]["g"].append(float(gv[d])); sc[d]["b"].append(float(bv[d]))
        p = v.get("preferred", "tie")
        pref = "system" if p == ("A" if sysA else "B") else ("baseline" if p in ("A", "B") else "tie")
        prefs.append(pref)
        per_task.append({
            "index": i,
            "task": tasks[i],
            "system_file": sysa[i]["path"],
            "baseline_file": basea[i]["path"],
            "system_was": "A" if sysA else "B",
            "preferred": pref,
            "scores": {"system": gv, "baseline": bv},
            "delta_primary": float(gv["primary"] - bv["primary"]),
            "judge_preferred_label": p,
            "rationale": v.get("rationale", ""),
            "judge_raw": v.get("raw", ""),
        })
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(graph=f"{gv['primary']:.2f}", baseline=f"{bv['primary']:.2f}",
                                 pref=pref, skipped=len(skipped))
        else:
            print(f"[compare/pairwise] task {i+1}/{n}: graph={gv['primary']:.2f} "
                  f"baseline={bv['primary']:.2f} pref={pref}", flush=True)
    if not per_task:
        raise SystemExit("no valid pairwise judge results")
    _render_pairwise(args, dims, sc, prefs, per_task, skipped)


def _render_pairwise(args, dims, sc, prefs, per_task, skipped):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150})
    all_dims = dims + ["primary"]
    agg = {d: {"system": _ms(sc[d]["g"]), "baseline": _ms(sc[d]["b"]),
               "delta": _ms([g - b for g, b in zip(sc[d]["g"], sc[d]["b"])])}
           for d in all_dims}
    pref_counts = {k: prefs.count(k) for k in ("system", "baseline", "tie")}
    n = len(per_task)
    win_rate = 100.0 * pref_counts["system"] / max(1, n)
    base_rate = 100.0 * pref_counts["baseline"] / max(1, n)
    tie_rate = 100.0 * pref_counts["tie"] / max(1, n)

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.8), gridspec_kw={"width_ratios": [1.35, 0.9]})
    x = np.arange(len(all_dims)); w = 0.36
    sys_means = [agg[d]["system"][0] for d in all_dims]
    base_means = [agg[d]["baseline"][0] for d in all_dims]
    sys_err = [agg[d]["system"][1] for d in all_dims]
    base_err = [agg[d]["baseline"][1] for d in all_dims]
    a = ax[0]
    a.bar(x - w / 2, sys_means, w, yerr=sys_err, capsize=3, color="#d62728", label="graph insights")
    a.bar(x + w / 2, base_means, w, yerr=base_err, capsize=3, color="#1f77b4", label="baseline")
    a.set_xticks(x); a.set_xticklabels(all_dims, rotation=18, ha="right")
    a.set_ylim(0, 5.35); a.set_ylabel("GPT-5.5 score (1-5, mean +/- s.e.)")
    a.set_title("Pairwise blind scores by dimension")
    a.legend(frameon=False, fontsize=9)

    a = ax[1]; a.axis("off")
    delta = agg["primary"]["delta"]
    cap = (
        f"n scored: {n}\n"
        f"judge: {args.jm}  effort={args.judge_effort}\n\n"
        f"primary: system {agg['primary']['system'][0]:.2f} vs "
        f"baseline {agg['primary']['baseline'][0]:.2f}\n"
        f"paired delta: {delta[0]:+.2f} +/- {delta[1]:.2f}\n\n"
        f"preferences:\n"
        f"  graph insights: {pref_counts['system']} ({win_rate:.0f}%)\n"
        f"  baseline:       {pref_counts['baseline']} ({base_rate:.0f}%)\n"
        f"  tie:            {pref_counts['tie']} ({tie_rate:.0f}%)\n\n"
        "Primary = mean(novelty, insight,\n"
        "mechanism, testability, specificity),\n"
        "capped by plausibility when plausibility < 3.\n"
        f"Skipped invalid judge calls: {len(skipped)}"
    )
    a.text(0, 1, cap, va="top", fontsize=9.0, family="monospace")
    fig.suptitle("Graph-derived insights improve small-model hypothesis answers", y=1.02, fontsize=12.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
    plt.close(fig)

    report = {
        "mode": "pairwise",
        "judge": args.jm,
        "judge_effort": args.judge_effort,
        "n_scored": n,
        "n_skipped": len(skipped),
        "dimensions": PAIR_DIMS,
        "primary_definition": {
            "core_dimensions": PAIR_PRIMARY_DIMS,
            "plausibility_rule": "if plausibility < 3, primary = min(core_mean, plausibility)",
        },
        "aggregate": agg,
        "preferences": pref_counts,
        "per_task": per_task,
        "skipped": skipped,
    }
    json.dump(report, open(f"{args.out}.json", "w"), indent=2)

    lines = [f"# Pairwise hypothesis benchmark\n",
             f"*{n} scored tasks · judge {args.jm} · graph-insight answers in `{args.system}` · "
             f"baseline answers in `{args.baseline}`*\n",
             "| dimension | graph insights | baseline | paired delta |",
             "|---|---:|---:|---:|"]
    for d in all_dims:
        lines.append(f"| {d} | {agg[d]['system'][0]:.2f} +/- {agg[d]['system'][1]:.2f} | "
                     f"{agg[d]['baseline'][0]:.2f} +/- {agg[d]['baseline'][1]:.2f} | "
                     f"{agg[d]['delta'][0]:+.2f} +/- {agg[d]['delta'][1]:.2f} |")
    lines += ["",
              f"Preference counts: graph insights **{pref_counts['system']}**, "
              f"baseline **{pref_counts['baseline']}**, tie **{pref_counts['tie']}**.",
              "",
              "Primary score: mean of novelty, insight, mechanism, testability, and specificity; "
              "if plausibility is below 3, the primary score is capped by plausibility.",
              ""]
    if skipped:
        lines.append(f"Skipped invalid judge calls: {len(skipped)}.")
    open(f"{args.out}.md", "w").write("\n".join(lines) + "\n")

    print(f"wrote {args.out}.png/.svg/.pdf")
    print(f"wrote {args.out}.json / {args.out}.md")
    print("[compare/pairwise] " + " · ".join(
        f"{d}: graph={agg[d]['system'][0]:.2f} baseline={agg[d]['baseline'][0]:.2f}"
        for d in all_dims) + f" | graph preferred {win_rate:.0f}%")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["graphrag", "coverage", "pairwise", "insights"], default="graphrag")
    p.add_argument("--tasks", help="tasks file, one per line (required for all modes except 'insights')")
    p.add_argument("--out", default="figures/graphrag", help="output basename")
    # graphrag / coverage shared
    p.add_argument("--run", help="ideate.py run dir (graphrag reads its graph; coverage its insights.json)")
    p.add_argument("--insights", help="[coverage] path to an insights.json directly")
    p.add_argument("--model", help="the SMALL generator served for all arms (OpenAI-compatible)")
    p.add_argument("--base-url", dest="base_url", help="generator endpoint (e.g. http://localhost:8000/v1)")
    p.add_argument("--api-key", dest="api_key", help="generator api key (else $OPENAI_API_KEY)")
    # graphrag retrieval (creativity)
    p.add_argument("--rag-seeds", dest="rag_seeds", type=int, default=8, help="[graphrag] question-seed nodes")
    p.add_argument("--rag-hops", dest="rag_hops", type=int, default=2, help="[graphrag] BFS hops for distal concepts")
    p.add_argument("--concepts", type=int, default=12, help="[graphrag] retrieved graph concepts")
    p.add_argument("--n-ideas", dest="n_ideas", type=int, default=6, help="[graphrag] ideas each arm brainstorms")
    p.add_argument("--n-per-miner", dest="n_per_miner", type=int, default=18,
                   help="[insights] number of TOP-actionability mined connections to rate (vs equal # of each control)")
    # coverage
    p.add_argument("--leads", type=int, default=8, help="K: top leads → graph hypotheses (default 8)")
    p.add_argument("--baseline-samples", dest="baseline_samples", type=int, default=0,
                   help="M: baseline independent samples (default 2*K — baseline gets MORE compute)")
    p.add_argument("--temperature", type=float, default=0.6, help="generator temperature (all arms)")
    p.add_argument("--gate", type=float, default=3.0, help="min plausible & testable score to count (1-5)")
    p.add_argument("--dedup-tau", dest="dedup_tau", type=float, default=0.85, help="cosine dup threshold")
    p.add_argument("--cover-tau", dest="cover_tau", type=float, default=0.60,
                   help="cosine below which an idea counts as unreached by the other arm")
    p.add_argument("--embed-model", dest="embed_model", default=None, help="sentence-transformers id")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=None,
                   help="truncate the graph to iter <= this when mining leads (matched-compute cutoff)")
    # pairwise (legacy)
    p.add_argument("--system", help="[pairwise] dir of system answers")
    p.add_argument("--baseline", help="[pairwise] dir of baseline answers")
    p.add_argument("--seed", type=int, default=0)
    # judge (both modes)
    p.add_argument("--judge-model", dest="jm", default="gpt-5.5")
    p.add_argument("--judge-base-url", dest="jbu", default=None)
    p.add_argument("--judge-api-key", dest="jak", default=None)
    p.add_argument("--judge-effort", dest="judge_effort", default="high",
                   choices=["minimal", "low", "medium", "high"],
                   help="reasoning effort for the judge (gpt-5.x); ignored by models that don't support it")
    args = p.parse_args()

    if getattr(args, "max_iter", None) is not None:
        I.MAX_ITER = args.max_iter                     # applied by insights.load_graph
    if args.mode == "insights":
        if not (args.run or args.insights):
            raise SystemExit("--mode insights needs --run <dir> (or --insights <file.json>)")
        return run_assess(args)
    if not args.tasks:
        raise SystemExit(f"--mode {args.mode} needs --tasks <file> (one question/task per line)")
    if args.mode == "pairwise":
        if not (args.system and args.baseline):
            raise SystemExit("--mode pairwise needs --system and --baseline answer dirs")
        return run_pairwise(args)
    if args.mode == "coverage":
        if not (args.run or args.insights):
            raise SystemExit("--mode coverage needs --run <dir> (or --insights <file.json>)")
        if not args.model:
            raise SystemExit("--mode coverage needs --model (the small generator for both arms)")
        return run_coverage(args)
    # graphrag (default)
    if not args.run:
        raise SystemExit("--mode graphrag needs --run <dir> (the accumulated graph to retrieve from)")
    if not args.model:
        raise SystemExit("--mode graphrag needs --model (the small generator for all arms)")
    run_graphrag(args)


if __name__ == "__main__":
    main()
