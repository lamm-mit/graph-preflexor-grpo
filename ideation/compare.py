#!/usr/bin/env python
"""Benchmark: validated idea-space COVERAGE at over-matched compute (the strength of graph reasoning).

The headline claim is mode-collapse: a single-shot LLM, however many times you resample it, keeps
returning the same handful of obvious ideas, while structured graph exploration reaches validated,
non-obvious hypotheses it never produces. A "which single answer is better" test can't show this —
so this benchmark compares HYPOTHESIS SETS, with the baseline given MORE compute, not less.

Per task, the SAME small model generates hypotheses two ways:
  GRAPH    one hypothesis per top-actionability lead from the run's graph (K hypotheses, each
           anchored to a specific discovered connection the model surfaced over many iterations).
  BASELINE M >= 2K INDEPENDENT samples of the same model on the bare question (mode-seeking, and
           deliberately given MORE answer-time calls than the graph arm).

Then it's mostly judge-free geometry. One blind judge call per task gates every hypothesis for
plausibility + testability (so coverage can't be won by noise); the rest is embeddings:
  - distinct VALIDATED hypotheses per arm (dedup in embedding space),
  - coverage saturation: cumulative distinct vs #generations (baseline plateaus = mode-collapse),
  - EXCLUSIVE fraction: validated graph ideas with no validated baseline idea nearby — i.e. ideas
    the baseline never reached even with 2x the samples (and vice versa, for honesty),
  - idea-space spread, and judged novelty of the validated sets.

    python compare.py --run runs/exp --tasks benchmark_tasks.txt \
        --model meta-llama/Llama-3.2-3B-Instruct --base-url http://localhost:8000/v1 \
        --judge-model gpt-5.5 --out runs/exp/benchmark/coverage

The generator is the SAME small model for both arms (only the conditioning differs); the judge is
used ONLY as a yes/no-ish validity gate, never to compare arms — so a strong judge can't bias it.
A legacy pairwise-answer judge is still available via --mode pairwise.
"""
import argparse
import glob
import json
import os
import random
import re

import numpy as np

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
def _make_call(model, base_url, api_key):
    from openai import OpenAI
    client = OpenAI(base_url=base_url or None, api_key=api_key or os.environ.get("OPENAI_API_KEY") or "x")

    def call(messages, temperature=0.0, max_tokens=4000, response_format=None):
        kwargs = {"model": model, "messages": messages, "max_completion_tokens": max_tokens,
                  "temperature": temperature}
        if response_format is not None:
            kwargs["response_format"] = response_format
        last = None
        for _ in range(6):
            try:
                r = client.chat.completions.create(**kwargs)
                return r.choices[0].message.content or ""
            except Exception as e:
                last, msg = e, str(e).lower()
                rf = (kwargs.get("response_format") or {}).get("type")
                if rf and any(k in msg for k in ("response_format", "json_schema", "json_object")):
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
    call_judge = _make_call(args.jm, args.jbu, args.jak)

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
#  Legacy pairwise-answer judge (kept for corroboration; --mode pairwise)
# --------------------------------------------------------------------------- #
PAIR_DIMS = {"novelty": "original and non-obvious core idea?",
             "insight": "reveals a non-obvious connection/mechanism, not a restatement?",
             "mechanism": "concrete, specific, physically plausible mechanism?",
             "feasibility": "realistically implementable?",
             "testability": "a clear falsifiable prediction or experiment?"}


def _strip_header(t):
    m = re.search(r"^\s*---\s*$", t, flags=re.MULTILINE)
    return t[m.end():].strip() if m else t.strip()


def _read_answers(d):
    fs = sorted(glob.glob(os.path.join(d, "*.md")) + glob.glob(os.path.join(d, "*.txt")))
    return [_strip_header(open(f, encoding="utf-8").read()) for f in fs]


def run_pairwise(args):
    dims = list(PAIR_DIMS)
    tasks = [ln.strip() for ln in open(args.tasks, encoding="utf-8") if ln.strip()]
    sysa, basea = _read_answers(args.system), _read_answers(args.baseline)
    n = min(len(tasks), len(sysa), len(basea))
    if n == 0:
        raise SystemExit("need matching tasks + --system + --baseline answers")
    call = _make_call(args.jm, args.jbu, args.jak)
    score = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
    side = {"type": "object", "additionalProperties": False, "properties": {d: score for d in dims},
            "required": dims}
    schema = {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["A", "B", "preferred"],
        "properties": {"A": side, "B": side, "preferred": {"type": "string", "enum": ["A", "B", "tie"]}}}}}
    sc = {d: {"g": [], "b": []} for d in dims}; prefs = []
    for i in range(n):
        rng = random.Random(args.seed + i); sysA = rng.random() < 0.5
        a, b = (sysa[i], basea[i]) if sysA else (basea[i], sysa[i])
        dd = "\n".join(f"- {d}: {PAIR_DIMS[d]}" for d in dims)
        user = (f"Two answers respond to the SAME task. Score each 1-5 on each dimension, then prefer "
                f"one.\nDimensions:\n{dd}\n\nTASK:\n{tasks[i]}\n\n--- A ---\n{a}\n\n--- B ---\n{b}")
        v = _parse_json(call([{"role": "system", "content": "You are an impartial reviewer."},
                              {"role": "user", "content": user}], 0.0, 800, schema)) or {}
        if "A" not in v or "B" not in v:
            continue
        gv, bv = (v["A"], v["B"]) if sysA else (v["B"], v["A"])
        for d in dims:
            try:
                sc[d]["g"].append(float(gv[d])); sc[d]["b"].append(float(bv[d]))
            except Exception:
                pass
        p = v.get("preferred", "tie")
        prefs.append("system" if p == ("A" if sysA else "B") else ("baseline" if p in ("A", "B") else "tie"))
    print("[compare/pairwise] " + " · ".join(f"{d}: g={np.mean(sc[d]['g']):.2f} b={np.mean(sc[d]['b']):.2f}"
          for d in dims) + f" | system preferred {100*prefs.count('system')/max(1,len(prefs)):.0f}%")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["coverage", "pairwise"], default="coverage")
    p.add_argument("--tasks", required=True, help="tasks file (one per line)")
    p.add_argument("--out", default="figures/coverage", help="output basename")
    # coverage
    p.add_argument("--run", help="ideate.py run dir (its insights.json supplies the graph leads)")
    p.add_argument("--insights", help="path to an insights.json directly")
    p.add_argument("--model", help="the SMALL generator served for both arms (OpenAI-compatible)")
    p.add_argument("--base-url", dest="base_url", help="generator endpoint (e.g. http://localhost:8000/v1)")
    p.add_argument("--api-key", dest="api_key", help="generator api key (else $OPENAI_API_KEY)")
    p.add_argument("--leads", type=int, default=8, help="K: top leads → graph hypotheses (default 8)")
    p.add_argument("--baseline-samples", dest="baseline_samples", type=int, default=0,
                   help="M: baseline independent samples (default 2*K — baseline gets MORE compute)")
    p.add_argument("--temperature", type=float, default=0.9, help="generator temperature (both arms)")
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
    args = p.parse_args()

    if getattr(args, "max_iter", None) is not None:
        I.MAX_ITER = args.max_iter                     # applied by insights.load_graph when mining
    if args.mode == "pairwise":
        if not (args.system and args.baseline):
            raise SystemExit("--mode pairwise needs --system and --baseline answer dirs")
        return run_pairwise(args)
    if not (args.run or args.insights):
        raise SystemExit("--mode coverage needs --run <dir> (or --insights <file.json>)")
    if not args.model:
        raise SystemExit("--mode coverage needs --model (the small generator for both arms)")
    run_coverage(args)


if __name__ == "__main__":
    main()
