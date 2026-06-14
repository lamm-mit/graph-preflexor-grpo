#!/usr/bin/env python
"""Novelty-yield benchmark: direct Llama ideas vs graph-derived idea leads.

This is the deliberately simple paper figure:

  novelty yield = fraction of generated idea cards with judged novelty >= 4

The benchmark ignores feasibility and plausibility by design. The judge only scores
whether each one-sentence idea is clearly novel/non-obvious, with a lightweight
coherence score kept for audit. By default coherence is NOT part of the pass rule.

Example:

  python novelty_yield_benchmark.py --run runs/exp_leap \
      --out runs/exp_leap/benchmark/novelty_yield \
      --n 30 \
      --model meta-llama/Llama-3.2-3B-Instruct --base-url http://localhost:8000/v1 \
      --judge-model gpt-5.5 --judge-effort high
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import compare
import insights as I
import synthesize as S


GEN_SYSTEM = (
    "You are a bold research ideator. You produce short, concrete, unconventional "
    "scientific research ideas. Optimize for novelty and non-obviousness, not for "
    "feasibility. Return only the requested idea sentence."
)

JUDGE_SYSTEM = (
    "You are an impartial novelty judge for scientific research ideas. Score novelty "
    "strictly. Ignore feasibility, cost, and experimental plausibility unless the idea "
    "is incoherent."
)

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "novelty_score",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["novelty", "coherent", "reason"],
            "properties": {
                "novelty": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "coherent": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "reason": {"type": "string"},
            },
        },
    },
}


def _progress(it, desc, total=None):
    try:
        from tqdm import tqdm
        return tqdm(it, desc=desc, total=total, unit="idea", dynamic_ncols=True)
    except Exception:
        return it


def _read_json(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def _write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _clean_one_sentence(text):
    text = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", str(text).strip())
    text = re.sub(r"^idea\s*\d+\s*[:.-]\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip()
    text = " ".join(text.split())
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        text = lines[0]
    return text[:900].strip()


def _make_generator(args):
    if args.backend == "openai":
        return compare._make_call(args.model, args.base_url, args.api_key)

    def call(messages, temperature=0.0, max_tokens=300, response_format=None):
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        user = "\n\n".join(m["content"] for m in messages if m.get("role") != "system")
        return S.answer_hf(
            system,
            user,
            model=args.model,
            temperature=temperature,
            max_tokens=max_tokens,
            device=args.device,
            dtype=args.dtype or "auto",
        )

    return call


def _load_results(args):
    if args.max_iter is not None:
        I.MAX_ITER = args.max_iter
    if args.insights:
        data = json.load(open(args.insights, encoding="utf-8"))
        run_dir = Path(args.insights).resolve().parent
        G = I.load_graph(str(run_dir)) if (run_dir / "graph.graphml").exists() else None
        miners = data.get("miners", {})
        results = [(k, miners.get(k, [])) for k in I.KIND_ORDER]
        topic = args.topic or data.get("topic") or I.read_topic(str(run_dir), G)
        return topic, G, results
    if not args.run:
        raise SystemExit("provide --run <dir> or --insights <file.json>")
    topic, G, results = I.load_insights_or_mine(
        args.run,
        top=args.top,
        want_mine=args.mine,
        embed_model=args.embed_model,
    )
    return args.topic or topic, G, results


def _lead_texts(results, n, lead_kinds):
    allowed = {x.strip() for x in lead_kinds.split(",") if x.strip()}
    ranked = S._rank_leads(results, max(n * 3, n))
    leads = [x for x in ranked if not allowed or x.get("kind") in allowed]
    if len(leads) < n:
        seen = {id(x) for x in leads}
        leads.extend(x for x in ranked if id(x) not in seen)
    leads = leads[:n]
    out = []
    for x in leads:
        kind = str(x.get("kind", "lead")).replace("_", " ")
        title = S._humanize(x.get("title", ""))
        raw_detail = str(x.get("detail", ""))
        detail = S._humanize(raw_detail)
        detail = re.sub(
            r"\b(score|distance|hops|cos|aa|betweenness|constraint|structural|semantic)="
            r"[^,;\s]+[,;\s]*",
            "",
            detail,
            flags=re.I,
        )
        detail = re.sub(r"\brelations=\[[^\]]+\]\s*", "", detail, flags=re.I)
        detail = re.sub(r"\(AA[^)]*\)", "", detail, flags=re.I)
        detail = re.sub(r"\[structural only\]", "", detail, flags=re.I)
        detail = " ".join(detail.split())
        lead = f"{kind}: {title}"
        detail_has_path = any(tok in raw_detail for tok in ("--", "->", "\u2192", "via ", "["))
        if detail and detail_has_path:
            lead += f". {detail[:360]}"
        out.append(lead)
    return out


def _baseline_prompt(topic, i, n, previous):
    avoid = "\n".join(f"- {x}" for x in previous[-12:])
    avoid_block = f"\nAvoid repeating these earlier ideas:\n{avoid}\n" if avoid else ""
    return (
        f"Topic: {topic}\n\n"
        f"Generate idea {i} of {n}: ONE clearly novel, unconventional research idea direction "
        "for this topic.\n"
        "Rules: one sentence only; optimize for novelty and non-obviousness; ignore feasibility; "
        "do not write a list; do not explain.\n"
        f"{avoid_block}"
        "Return only the idea sentence."
    )


def _graph_prompt(topic, i, n, lead, previous):
    avoid = "\n".join(f"- {x}" for x in previous[-12:])
    avoid_block = f"\nAvoid repeating these earlier ideas:\n{avoid}\n" if avoid else ""
    return (
        f"Topic: {topic}\n\n"
        "Graph-derived exploratory lead from a long Graph-PRefLexOR reasoning run:\n"
        f"{lead}\n\n"
        f"Generate idea {i} of {n}: convert this lead into ONE clearly novel, unconventional "
        "research idea direction.\n"
        "Rules: use the lead as a springboard, but do not mention the graph or provenance; "
        "one sentence only; optimize for novelty and non-obviousness; ignore feasibility; "
        "do not write a list; do not explain.\n"
        f"{avoid_block}"
        "Return only the idea sentence."
    )


def _generate_arm(args, arm, topic, leads, out_dir):
    path = out_dir / "ideas" / f"{arm}.json"
    prompts = out_dir / "prompts" / arm
    ideas = _read_json(path, [])
    if args.force:
        ideas = []
    previous = [x.get("idea", "") for x in ideas if x.get("idea")]
    start = len(ideas) + 1
    if start > args.n:
        print(f"[novelty-yield] reusing {len(ideas[:args.n])} {arm} ideas", flush=True)
        return ideas[:args.n]

    call = _make_generator(args)
    lead_order = list(leads)
    if args.shuffle_leads:
        random.Random(args.seed).shuffle(lead_order)
    if not lead_order and arm == "graph":
        raise SystemExit("no graph leads found; run insights.py first or pass --mine")

    for i in _progress(range(start, args.n + 1), f"[novelty-yield] generating {arm}", total=args.n - start + 1):
        if arm == "baseline":
            prompt = _baseline_prompt(topic, i, args.n, previous)
            lead = None
        else:
            lead = lead_order[(i - 1) % len(lead_order)]
            prompt = _graph_prompt(topic, i, args.n, lead, previous)
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / f"{i:03d}.txt").write_text(GEN_SYSTEM + "\n\n" + prompt, encoding="utf-8")
        if args.dry_run:
            idea = f"(dry-run {arm} idea {i})"
        else:
            raw = call(
                [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            idea = _clean_one_sentence(raw)
        rec = {"id": i, "arm": arm, "idea": idea}
        if lead:
            rec["lead"] = lead
        ideas.append(rec)
        previous.append(idea)
        _write_json(path, ideas)
    return ideas[:args.n]


def _judge_prompt(topic, idea):
    return (
        f"Topic: {topic}\n\n"
        f"Idea: {idea}\n\n"
        "Score novelty from 1 to 5.\n\n"
        "Novelty rubric:\n"
        "1 = obvious, generic, textbook, or common extension\n"
        "2 = mildly different but predictable\n"
        "3 = somewhat non-obvious\n"
        "4 = clearly novel or surprising recombination\n"
        "5 = highly original, unusual, potentially field-opening\n\n"
        "Also score coherence from 1 to 5, where 1 is incoherent word salad and 5 is clear. "
        "Ignore feasibility and experimental plausibility unless the idea is incoherent.\n\n"
        "Return JSON with novelty, coherent, and a short reason."
    )


def _as_score(x, default=0):
    try:
        y = int(float(x))
    except Exception:
        return default
    return max(0, min(5, y))


def _judge_one(call, topic, rec, args, prompt_dir):
    prompt = _judge_prompt(topic, rec.get("idea", ""))
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / f"{rec['arm']}_{rec['id']:03d}.txt").write_text(
        JUDGE_SYSTEM + "\n\n" + prompt, encoding="utf-8")
    if args.dry_run:
        return {"novelty": 0, "coherent": 0, "reason": "dry run", "valid_json": True}
    try:
        raw = call(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
            response_format=JUDGE_SCHEMA,
        )
        data = compare._parse_json(raw) or {}
        return {
            "novelty": _as_score(data.get("novelty")),
            "coherent": _as_score(data.get("coherent")),
            "reason": str(data.get("reason", ""))[:600],
            "valid_json": bool(data),
        }
    except Exception as e:
        return {"novelty": 0, "coherent": 0, "reason": f"judge failed: {e}", "valid_json": False}


def _score_ideas(args, topic, ideas, out_dir):
    score_path = out_dir / "scores.json"
    existing = _read_json(score_path, {})
    if args.force_judge or args.force:
        existing = {}
    valid_keys = {f"{rec['arm']}:{rec['id']}" for rec in ideas}
    existing = {k: v for k, v in existing.items() if k in valid_keys}
    call = compare._make_call(args.judge_model, args.judge_base_url, args.judge_api_key,
                              reasoning_effort=args.judge_effort)
    prompt_dir = out_dir / "prompts" / "judge"
    for rec in _progress(ideas, "[novelty-yield] judging novelty", total=len(ideas)):
        key = f"{rec['arm']}:{rec['id']}"
        if key in existing:
            continue
        existing[key] = {**rec, **_judge_one(call, topic, rec, args, prompt_dir)}
        existing[key]["clearly_novel"] = (
            existing[key]["novelty"] >= args.novelty_threshold
            and existing[key]["coherent"] >= args.coherence_gate
        )
        _write_json(score_path, existing)
    return existing


def _arm_stats(scores, arm, threshold, coherence_gate):
    vals = [v for v in scores.values() if v.get("arm") == arm]
    n = len(vals)
    passes = [
        v for v in vals
        if v.get("novelty", 0) >= threshold and v.get("coherent", 0) >= coherence_gate
    ]
    p = len(passes) / max(1, n)
    se = math.sqrt(p * (1.0 - p) / n) if n else 0.0
    mean_nov = sum(v.get("novelty", 0) for v in vals) / max(1, n)
    return {
        "n": n,
        "passes": len(passes),
        "yield": p,
        "se": se,
        "mean_novelty": mean_nov,
    }


def _render(args, topic, scores, out_dir, leads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = {
        "baseline": _arm_stats(scores, "baseline", args.novelty_threshold, args.coherence_gate),
        "graph": _arm_stats(scores, "graph", args.novelty_threshold, args.coherence_gate),
    }
    labels = ["Llama baseline", "Graph-PRefLexOR"]
    vals = [100 * stats["baseline"]["yield"], 100 * stats["graph"]["yield"]]
    errs = [100 * stats["baseline"]["se"], 100 * stats["graph"]["se"]]
    colors = ["#1f77b4", "#d62728"]

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    bars = ax.bar([0, 1], vals, yerr=errs, capsize=5, color=colors, width=0.58)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"% ideas with novelty >= {args.novelty_threshold}")
    ax.set_title("Novelty yield: graph-derived leads vs direct Llama")
    for bar, arm in zip(bars, ["baseline", "graph"]):
        s = stats[arm]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{s['passes']}/{s['n']}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    note = (
        f"topic: {topic[:54]}\n"
        f"judge: {args.judge_model} effort={args.judge_effort}\n"
        f"coherence gate: {args.coherence_gate or 'off'}\n"
        f"mean novelty: baseline {stats['baseline']['mean_novelty']:.2f}, "
        f"graph {stats['graph']['mean_novelty']:.2f}\n"
        f"graph leads loaded: {len(leads)}"
    )
    ax.text(1.04, 0.98, note, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"novelty_yield.{ext}", bbox_inches="tight")
    plt.close(fig)

    report = {
        "mode": "novelty_yield",
        "topic": topic,
        "threshold": args.novelty_threshold,
        "coherence_gate": args.coherence_gate,
        "generator": {"backend": args.backend, "model": args.model, "base_url": args.base_url},
        "judge": {"model": args.judge_model, "base_url": args.judge_base_url,
                  "effort": args.judge_effort},
        "stats": stats,
        "scores": list(scores.values()),
        "leads": leads,
    }
    _write_json(out_dir / "results.json", report)
    lines = [
        f"# Novelty-yield benchmark - {topic}",
        "",
        f"*Novelty yield = fraction of ideas with novelty >= {args.novelty_threshold}.*",
        "",
        f"- Baseline Llama: **{stats['baseline']['passes']}/{stats['baseline']['n']}** "
        f"({100 * stats['baseline']['yield']:.1f}%)",
        f"- Graph-PRefLexOR: **{stats['graph']['passes']}/{stats['graph']['n']}** "
        f"({100 * stats['graph']['yield']:.1f}%)",
        f"- Mean novelty: baseline {stats['baseline']['mean_novelty']:.2f}, "
        f"graph {stats['graph']['mean_novelty']:.2f}",
        "",
        "Feasibility/plausibility is intentionally ignored. Coherence is scored for audit only"
        + (f" and gated at >= {args.coherence_gate}." if args.coherence_gate else "."),
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'novelty_yield.png'} / .svg / .pdf")
    print(f"wrote {out_dir / 'results.json'} / {out_dir / 'report.md'}")
    print(
        "[novelty-yield] "
        f"baseline {stats['baseline']['passes']}/{stats['baseline']['n']} "
        f"({100 * stats['baseline']['yield']:.1f}%) | "
        f"graph {stats['graph']['passes']}/{stats['graph']['n']} "
        f"({100 * stats['graph']['yield']:.1f}%)"
    )


def run(args):
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    topic, G, results = _load_results(args)
    if G is None:
        import networkx as nx
        G = nx.DiGraph()
    n_leads = args.leads or max(args.n, 30)
    leads = _lead_texts(results, n_leads, args.lead_kinds)
    if not leads:
        raise SystemExit("no mined graph leads found; run insights.py --run <dir> or pass --mine")

    manifest = {
        "mode": "novelty_yield",
        "topic": topic,
        "n": args.n,
        "leads": n_leads,
        "run": str(Path(args.run).resolve()) if args.run else None,
        "insights": str(Path(args.insights).resolve()) if args.insights else None,
        "novelty_threshold": args.novelty_threshold,
        "coherence_gate": args.coherence_gate,
        "lead_kinds": args.lead_kinds,
    }
    _write_json(out_dir / "manifest.json", manifest)
    (out_dir / "graph_leads.txt").write_text(
        "\n".join(f"{i+1}. {lead}" for i, lead in enumerate(leads)) + "\n",
        encoding="utf-8",
    )

    baseline = _generate_arm(args, "baseline", topic, leads, out_dir)
    graph = _generate_arm(args, "graph", topic, leads, out_dir)
    scores = _score_ideas(args, topic, baseline + graph, out_dir)
    _render(args, topic, scores, out_dir, leads)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="ideate.py run dir containing graph.graphml and/or insights.json")
    p.add_argument("--insights", help="path to insights.json; overrides --run insight lookup")
    p.add_argument("--topic", help="override recovered topic")
    p.add_argument("--out", default="runs/novelty_yield", help="output directory")
    p.add_argument("--n", type=int, default=20, help="idea cards per arm")
    p.add_argument("--leads", type=int, default=0,
                   help="graph leads loaded for graph arm; default max(n, 30)")
    p.add_argument("--lead-kinds", default="conceptual_bridge,relational_analogy,semantic_dissonance,feedback_loop",
                   help="comma-separated insight kinds preferred for graph ideas; broker_idea is excluded by default")
    p.add_argument("--shuffle-leads", action="store_true",
                   help="shuffle graph leads before assigning them to graph-arm ideas")
    p.add_argument("--top", type=int, default=12,
                   help="candidates per miner if insights must be mined")
    p.add_argument("--mine", action="store_true", help="remine insights instead of using cached insights.json")
    p.add_argument("--embed-model", dest="embed_model", default=None,
                   help="sentence-transformers id if mining insights")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=None,
                   help="truncate graph to iter <= this when mining")
    p.add_argument("--force", action="store_true", help="regenerate ideas and scores")
    p.add_argument("--force-judge", action="store_true", help="rescore existing ideas")
    p.add_argument("--dry-run", action="store_true", help="write prompts/manifest without model calls")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--backend", choices=["openai", "hf"], default="openai")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", dest="base_url", default=None)
    p.add_argument("--api-key", dest="api_key", default=None)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=180)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"])

    p.add_argument("--judge-model", dest="judge_model", default="gpt-5.5")
    p.add_argument("--judge-base-url", dest="judge_base_url", default=None)
    p.add_argument("--judge-api-key", dest="judge_api_key", default=None)
    p.add_argument("--judge-effort", dest="judge_effort", default="high",
                   choices=["minimal", "low", "medium", "high"])
    p.add_argument("--novelty-threshold", type=int, default=4,
                   help="idea counts as clearly novel when novelty >= this")
    p.add_argument("--coherence-gate", type=int, default=0,
                   help="optional coherence threshold; 0 disables coherence gating")
    args = p.parse_args()
    args.n = max(1, args.n)
    args.novelty_threshold = max(1, min(5, args.novelty_threshold))
    args.coherence_gate = max(0, min(5, args.coherence_gate))
    if args.force and args.force_judge:
        args.force_judge = True
    run(args)


if __name__ == "__main__":
    main()
