#!/usr/bin/env python
"""Journal-quality figures for ideation runs.

Reads one or more run directories (each produced by ideate.py: growth.csv +
summary.json) and renders publication figures comparing them. Pass multiple runs
(e.g. Graph-PRefLexOR vs a frontier baseline) to overlay them.

    python plot_ideation.py --runs runs/graphpreflexor runs/gpt4o \
        --labels "Graph-PRefLexOR-3B" "GPT-4o" --out figures/ideation

Outputs (PNG + SVG + PDF each):
    <out>_curves.*   2x2: ideas vs compute, diversity vs compute, edges/idea, components
    <out>_bars.*     final-metric comparison bars
    <out>_graph_<label>.*   spring-layout snapshot of each accumulated graph
"""
import argparse
import csv
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import metrics as M

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def load_run(path):
    """summary.json is optional (written only when a run finishes); growth.csv is enough."""
    sp = os.path.join(path, "summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else None
    gp = os.path.join(path, "growth.csv")
    if not os.path.exists(gp):
        raise SystemExit(f"{gp} not found — has the run produced any steps yet?")
    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(gp))]
    return summary, rows


def final_metrics(summary, rows):
    """Use summary.json metrics if present, else derive what we can from growth.csv."""
    if summary and "metrics" in summary:
        return summary["metrics"]
    if not rows:
        return {}
    last = rows[-1]
    n, e, calls = last["n_nodes"], last["n_edges"], last["iter"] + 1
    return {"nodes": n, "edges": e, "ideas_per_call": n / max(1, calls),
            "mean_pairwise_distance": last.get("diversity", 0.0),
            "elaboration": e / max(1, n), "density": None, "flexibility_clusters": None}


def _save(fig, base):
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    print(f"wrote {base}.png/.svg/.pdf")


def curves(runs, labels, base):
    fig, ax = plt.subplots(2, 2, figsize=(8.0, 6.0))
    for (summary, rows), lab, col in zip(runs, labels, PALETTE):
        x = [r["iter"] + 1 for r in rows]                 # generator calls
        ax[0, 0].plot(x, [r["n_nodes"] for r in rows], color=col, lw=2, label=lab)
        ax[0, 1].plot(x, [r["diversity"] for r in rows], color=col, lw=2, label=lab)
        epn = [r["n_edges"] / max(1, r["n_nodes"]) for r in rows]
        ax[1, 0].plot(x, epn, color=col, lw=2, label=lab)
        ax[1, 1].plot(x, [r["n_edges"] for r in rows], color=col, lw=2, label=lab)
    ax[0, 0].set_title("(a) Ideas vs compute"); ax[0, 0].set_ylabel("distinct ideas (nodes)")
    ax[0, 1].set_title("(b) Semantic diversity vs compute"); ax[0, 1].set_ylabel("mean pairwise distance")
    ax[1, 0].set_title("(c) Elaboration"); ax[1, 0].set_ylabel("edges / idea")
    ax[1, 1].set_title("(d) Graph connectivity"); ax[1, 1].set_ylabel("edges")
    for a in ax.flat:
        a.set_xlabel("generator calls"); a.grid(True, color="0.9", lw=0.6); a.set_axisbelow(True)
    ax[0, 0].legend(frameon=False)
    fig.tight_layout(); _save(fig, f"{base}_curves")


def bars(metrics_list, labels, base):
    keys = [("nodes", "ideas (fluency)"), ("ideas_per_call", "ideas / call"),
            ("mean_pairwise_distance", "diversity"), ("flexibility_clusters", "flexibility"),
            ("elaboration", "edges / idea"), ("density", "density")]
    fig, ax = plt.subplots(2, 3, figsize=(9.0, 5.0))
    for a, (k, title) in zip(ax.flat, keys):
        vals = [(m.get(k) or 0) for m in metrics_list]
        a.bar(labels, vals, color=PALETTE[:len(labels)])
        a.set_title(title); a.grid(True, axis="y", color="0.9", lw=0.6); a.set_axisbelow(True)
        a.tick_params(axis="x", rotation=20)
    fig.suptitle("Final ideation metrics", y=1.02)
    fig.tight_layout(); _save(fig, f"{base}_bars")


def graph_snapshot(run_dir, label, base):
    gp = os.path.join(run_dir, "graph.graphml")
    if not os.path.exists(gp):
        print(f"  (skip graph snapshot for {label}: {gp} not found — run not finished?)")
        return
    G = nx.read_graphml(gp)
    fig, a = plt.subplots(figsize=(6.0, 6.0))
    pos = nx.spring_layout(G, seed=0, k=0.6)
    deg = dict(G.degree())
    nx.draw_networkx_edges(G, pos, ax=a, alpha=0.25, arrows=False, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=a, node_size=[40 + 40 * deg[n] for n in G],
                           node_color="#1f77b4", alpha=0.85, linewidths=0)
    a.set_title(f"Accumulated idea graph — {label}  ({G.number_of_nodes()}n/{G.number_of_edges()}e)")
    a.axis("off"); fig.tight_layout(); _save(fig, f"{base}_graph_{label.replace(' ', '_')}")


def graph_growth(run_dir, label, base, frames=6):
    """Montage of the idea-graph at increasing iterations, reconstructed from the
    per-node/edge `iter` provenance in graph.graphml (fixed layout across frames)."""
    gp = os.path.join(run_dir, "graph.graphml")
    if not os.path.exists(gp):
        print(f"  (skip graph figure for {label}: {gp} not found — run not finished?)")
        return
    G = nx.read_graphml(gp)
    if G.number_of_nodes() == 0:
        return

    def it(attrs):
        try:
            return int(float(attrs.get("iter", 0)))
        except Exception:
            return 0

    max_it = max([it(G.nodes[n]) for n in G] + [0])
    checkpoints = sorted(set(int(round(x)) for x in np.linspace(0, max_it, frames)))
    pos = nx.spring_layout(G, seed=0, k=0.6)              # fixed layout (final graph)
    deg = dict(G.degree())
    ncol = min(len(checkpoints), 3)
    nrow = int(np.ceil(len(checkpoints) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, t in zip(axes, checkpoints):
        H = G.subgraph([n for n in G if it(G.nodes[n]) <= t])
        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.2, arrows=False, width=0.6)
        nx.draw_networkx_nodes(H, pos, ax=ax, node_color="#1f77b4", alpha=0.85, linewidths=0,
                               node_size=[18 + 22 * deg[n] for n in H])
        ax.set_title(f"iter ≤ {t}  ({H.number_of_nodes()}n/{H.number_of_edges()}e)", fontsize=9)
        ax.axis("off")
    for ax in axes[len(checkpoints):]:
        ax.axis("off")
    fig.suptitle(f"Idea-graph growth — {label}", y=1.0)
    fig.tight_layout()
    _save(fig, f"{base}_growth_{label.replace(' ', '_')}")


def graph_movie(run_dir, label, base, fps=2):
    """Animated GIF of the idea-graph accreting over iterations (fixed layout)."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    gp = os.path.join(run_dir, "graph.graphml")
    if not os.path.exists(gp):
        print(f"  (skip graph figure for {label}: {gp} not found — run not finished?)")
        return
    G = nx.read_graphml(gp)
    if G.number_of_nodes() == 0:
        return

    def it(attrs):
        try:
            return int(float(attrs.get("iter", 0)))
        except Exception:
            return 0

    max_it = max([it(G.nodes[n]) for n in G] + [0])
    pos = nx.spring_layout(G, seed=0, k=0.6)
    deg = dict(G.degree())
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(t):
        ax.clear(); ax.axis("off")
        H = G.subgraph([n for n in G if it(G.nodes[n]) <= t])
        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.2, arrows=False, width=0.6)
        nx.draw_networkx_nodes(H, pos, ax=ax, node_color="#1f77b4", alpha=0.85, linewidths=0,
                               node_size=[18 + 22 * deg[n] for n in H])
        ax.set_title(f"{label} — iter {t}  ({H.number_of_nodes()}n/{H.number_of_edges()}e)",
                     fontsize=10)

    anim = FuncAnimation(fig, draw, frames=range(max_it + 1), interval=1000 / max(1, fps))
    out = f"{base}_movie_{label.replace(' ', '_')}.gif"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out}")


def _short(s, n=22):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def graph_analytics(run_dir, label, base):
    """Rich journal-quality 2x3 panel of graph properties + a graph_analysis JSON."""
    gp = os.path.join(run_dir, "graph.graphml")
    if not os.path.exists(gp):
        print(f"  (skip analytics for {label}: {gp} not found)")
        return
    G = nx.read_graphml(gp)
    if G.number_of_nodes() < 3:
        print(f"  (skip analytics for {label}: graph too small)")
        return
    cents = M.centralities(G)
    adv = M.advanced_metrics(G)
    with open(f"{base}_analysis_{label.replace(' ', '_')}.json", "w") as f:
        json.dump(adv, f, indent=2)

    fig, ax = plt.subplots(2, 3, figsize=(11.5, 7.2))
    blue = "#1f77b4"

    # (a) degree distribution (hist + log-log inset)
    degs = [d for _, d in G.degree()]
    ax[0, 0].hist(degs, bins=range(0, max(degs) + 2), color=blue, alpha=0.85, align="left")
    ax[0, 0].set_title("(a) Degree distribution"); ax[0, 0].set_xlabel("degree"); ax[0, 0].set_ylabel("count")
    ins = ax[0, 0].inset_axes([0.55, 0.55, 0.42, 0.42])
    hist = Counter(degs); xs = sorted(k for k in hist if k > 0)
    if len(xs) > 1:
        ins.loglog([k for k in xs], [hist[k] for k in xs], "o", ms=3, color="#d62728")
        ins.set_title("log-log", fontsize=7); ins.tick_params(labelsize=6)

    # (b) centrality distributions (boxplots)
    bet, clo, pr = (list(cents[k].values()) for k in ("betweenness", "closeness", "pagerank"))
    ax[0, 1].boxplot([bet, clo, pr], tick_labels=["betw.", "close.", "pagerank"], showfliers=False)
    ax[0, 1].set_title("(b) Centrality distributions"); ax[0, 1].set_ylabel("centrality")

    # (c) top hub ideas by PageRank
    top = sorted(cents["pagerank"].items(), key=lambda x: x[1], reverse=True)[:12][::-1]
    names = [_short(G.nodes[n].get("label", n)) for n, _ in top]
    ax[0, 2].barh(range(len(top)), [v for _, v in top], color=blue)
    ax[0, 2].set_yticks(range(len(top))); ax[0, 2].set_yticklabels(names, fontsize=7)
    ax[0, 2].set_title("(c) Top hub ideas (PageRank)"); ax[0, 2].set_xlabel("PageRank")

    # (d) relation-type frequency
    rels = Counter(d.get("relation", "?") for _, _, d in G.edges(data=True)).most_common(12)[::-1]
    ax[1, 0].barh(range(len(rels)), [c for _, c in rels], color="#2ca02c")
    ax[1, 0].set_yticks(range(len(rels))); ax[1, 0].set_yticklabels([_short(r, 18) for r, _ in rels], fontsize=7)
    ax[1, 0].set_title("(d) Relation types"); ax[1, 0].set_xlabel("count")

    # (e) community sizes
    try:
        comms = list(nx.community.greedy_modularity_communities(G.to_undirected()))
        sizes = sorted((len(c) for c in comms), reverse=True)
        ax[1, 1].bar(range(len(sizes)), sizes, color="#9467bd")
        ax[1, 1].set_title(f"(e) Communities (Q={adv.get('modularity', float('nan')):.2f})")
        ax[1, 1].set_xlabel("community"); ax[1, 1].set_ylabel("size")
    except Exception:
        ax[1, 1].axis("off")

    # (f) global-metrics card
    ax[1, 2].axis("off")
    order = [("nodes", G.number_of_nodes()), ("edges", G.number_of_edges()),
             ("density", adv.get("density")), ("clustering C", adv.get("clustering")),
             ("transitivity", adv.get("transitivity")), ("avg path L", adv.get("avg_path_length")),
             ("diameter", adv.get("diameter")), ("small-world σ", adv.get("small_world_sigma")),
             ("small-world ω", adv.get("small_world_omega")), ("modularity Q", adv.get("modularity")),
             ("# communities", adv.get("n_communities")), ("assortativity", adv.get("degree_assortativity")),
             ("reciprocity", adv.get("reciprocity"))]
    lines = [f"{k:>15s} : {v:.3f}" if isinstance(v, float) else f"{k:>15s} : {v}"
             for k, v in order if v is not None]
    ax[1, 2].text(0.0, 1.0, "Global metrics\n" + "\n".join(lines), va="top", ha="left",
                  family="monospace", fontsize=8.5, transform=ax[1, 2].transAxes)

    for a in ax.flat:
        a.grid(True, color="0.92", lw=0.5); a.set_axisbelow(True)
    fig.suptitle(f"Graph analytics — {label}  ({G.number_of_nodes()}n / {G.number_of_edges()}e)", y=1.0)
    fig.tight_layout()
    _save(fig, f"{base}_analytics_{label.replace(' ', '_')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="one or more run dirs")
    p.add_argument("--labels", nargs="+", help="legend labels (default: dir names)")
    p.add_argument("--out", default="figures/ideation")
    p.add_argument("--no-graph", action="store_true", help="skip graph snapshots")
    p.add_argument("--growth-frames", type=int, default=6,
                   help="frames in the graph-growth montage (0 = skip)")
    p.add_argument("--movie", action="store_true", help="also render an animated GIF of growth")
    p.add_argument("--movie-fps", type=int, default=2)
    args = p.parse_args()
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    loaded = [load_run(r) for r in args.runs]
    curves(loaded, labels, args.out)
    bars([final_metrics(s, rows) for s, rows in loaded], labels, args.out)
    if not args.no_graph:
        for rd, lab in zip(args.runs, labels):
            graph_snapshot(rd, lab, args.out)
            graph_analytics(rd, lab, args.out)
            if args.growth_frames > 0:
                graph_growth(rd, lab, args.out, frames=args.growth_frames)
            if args.movie:
                graph_movie(rd, lab, args.out, fps=args.movie_fps)


if __name__ == "__main__":
    main()
