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

import matplotlib.pyplot as plt
import networkx as nx

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def load_run(path):
    with open(os.path.join(path, "summary.json")) as f:
        summary = json.load(f)
    rows = []
    with open(os.path.join(path, "growth.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return summary, rows


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


def bars(runs, labels, base):
    keys = [("nodes", "ideas (fluency)"), ("ideas_per_call", "ideas / call"),
            ("mean_pairwise_distance", "diversity"), ("flexibility_clusters", "flexibility"),
            ("elaboration", "edges / idea"), ("density", "density")]
    fig, ax = plt.subplots(2, 3, figsize=(9.0, 5.0))
    for a, (k, title) in zip(ax.flat, keys):
        vals = [(s["metrics"].get(k) or 0) for s, _ in runs]
        a.bar(labels, vals, color=PALETTE[:len(labels)])
        a.set_title(title); a.grid(True, axis="y", color="0.9", lw=0.6); a.set_axisbelow(True)
        a.tick_params(axis="x", rotation=20)
    fig.suptitle("Final ideation metrics", y=1.02)
    fig.tight_layout(); _save(fig, f"{base}_bars")


def graph_snapshot(run_dir, label, base):
    G = nx.read_graphml(os.path.join(run_dir, "graph.graphml"))
    fig, a = plt.subplots(figsize=(6.0, 6.0))
    pos = nx.spring_layout(G, seed=0, k=0.6)
    deg = dict(G.degree())
    nx.draw_networkx_edges(G, pos, ax=a, alpha=0.25, arrows=False, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=a, node_size=[40 + 40 * deg[n] for n in G],
                           node_color="#1f77b4", alpha=0.85, linewidths=0)
    a.set_title(f"Accumulated idea graph — {label}  ({G.number_of_nodes()}n/{G.number_of_edges()}e)")
    a.axis("off"); fig.tight_layout(); _save(fig, f"{base}_graph_{label.replace(' ', '_')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="one or more run dirs")
    p.add_argument("--labels", nargs="+", help="legend labels (default: dir names)")
    p.add_argument("--out", default="figures/ideation")
    p.add_argument("--no-graph", action="store_true", help="skip graph snapshots")
    args = p.parse_args()
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    runs = [load_run(r) for r in args.runs]
    curves(runs, labels, args.out)
    bars(runs, labels, args.out)
    if not args.no_graph:
        for rd, lab in zip(args.runs, labels):
            graph_snapshot(rd, lab, args.out)


if __name__ == "__main__":
    main()
