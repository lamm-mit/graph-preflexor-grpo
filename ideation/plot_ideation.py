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
    if MAX_ITER is not None:                           # truncate to a common length
        rows = [r for r in rows if r.get("iter", 0) <= MAX_ITER]
    return summary, rows


def final_metrics(summary, rows):
    """Use summary.json metrics if present, else derive what we can from growth.csv. When a
    --max-iter cap is active, ignore the (full-run) summary metrics and derive from capped rows."""
    if MAX_ITER is None and summary and "metrics" in summary:
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


# Module-level iteration cap (set from --max-iter); applied to every graph/rows read so all
# analysis functions truncate consistently for fair cross-run figures. None = no cap.
MAX_ITER = None


def read_graph(run_dir):
    """Read <run>/graph.graphml, tolerating a still-running ideate.py (which truncates and
    rewrites that file every step). On an empty/corrupt read, fall back to the newest stable
    per-iteration snapshot in graphml/ (written once, never rewritten). Truncates to MAX_ITER.
    Returns G or None."""
    import glob
    from graphstore import cap_graph
    candidates = []
    gp = os.path.join(run_dir, "graph.graphml")
    if os.path.exists(gp) and os.path.getsize(gp) > 0:
        candidates.append(gp)
    candidates += sorted(glob.glob(os.path.join(run_dir, "graphml", "iter_*.graphml")))[::-1]
    for path in candidates:
        try:
            if os.path.getsize(path) > 0:
                return cap_graph(nx.read_graphml(path), MAX_ITER)
        except Exception:
            continue
    return None


def _series(run_dir, rows, mode, embed_model=None):
    """Return (x, xlabel, nodes, edges, elab, div) for the curves, by 'iteration' or
    'depth' (reasoning depth = hops from seed, aggregated from graph.graphml provenance)."""
    G = read_graph(run_dir) if mode == "depth" else None
    if mode == "depth" and G is not None:

        def dep(a):
            try:
                return int(float(a.get("depth", 0)))
            except Exception:
                return 0
        ndep = {n: dep(G.nodes[n]) for n in G}
        edep = [(u, v, dep(d)) for u, v, d in G.edges(data=True)]
        maxd = max(list(ndep.values()) + [e[2] for e in edep] + [0])
        try:                                              # embed once (batched) for diversity-by-depth
            from graphstore import embed_texts, resolve_embed_model, DEFAULT_EMBED_MODEL
            model = resolve_embed_model(run_dir, embed_model)
            ns = list(G)
            print(f"  embedding {len(ns)} concepts for depth-diversity (batched)…", flush=True)
            V = embed_texts([str(G.nodes[n].get("label", n)) for n in ns], model or DEFAULT_EMBED_MODEL)
            vec = {n: V[i] for i, n in enumerate(ns)}
        except Exception:
            vec = None
        xs, nn, ee, el, dv = [], [], [], [], []
        for d in range(maxd + 1):
            Nd = [n for n in G if ndep[n] <= d]
            Ed = sum(1 for _, _, dd in edep if dd <= d)
            xs.append(d); nn.append(len(Nd)); ee.append(Ed); el.append(Ed / max(1, len(Nd)))
            if vec and len(Nd) > 1:
                X = np.stack([vec[n] for n in Nd]); S = X @ X.T
                dv.append(float(1.0 - S[np.triu_indices(len(X), 1)].mean()))
            else:
                dv.append(float("nan"))
        return xs, "reasoning depth (hops from seed)", nn, ee, el, dv
    # reasoning-index (per-step iteration) variant
    x = [r["iter"] + 1 for r in rows]
    return (x, "reasoning index", [r["n_nodes"] for r in rows], [r["n_edges"] for r in rows],
            [r["n_edges"] / max(1, r["n_nodes"]) for r in rows], [r["diversity"] for r in rows])


def curves(loaded, run_dirs, labels, base, mode="depth", embed_model=None):
    fig, ax = plt.subplots(2, 2, figsize=(8.0, 6.0))
    xlabel = "iteration"
    for (summary, rows), rd, lab, col in zip(loaded, run_dirs, labels, PALETTE):
        x, xlabel, nn, ee, el, dv = _series(rd, rows, mode, embed_model)
        ax[0, 0].plot(x, nn, color=col, lw=2, marker="o", ms=3, label=lab)
        ax[0, 1].plot(x, dv, color=col, lw=2, marker="o", ms=3, label=lab)
        ax[1, 0].plot(x, el, color=col, lw=2, marker="o", ms=3, label=lab)
        ax[1, 1].plot(x, ee, color=col, lw=2, marker="o", ms=3, label=lab)
    ax[0, 0].set_title("(a) Ideas"); ax[0, 0].set_ylabel("distinct ideas (nodes)")
    ax[0, 1].set_title("(b) Semantic diversity"); ax[0, 1].set_ylabel("mean pairwise distance")
    ax[1, 0].set_title("(c) Elaboration"); ax[1, 0].set_ylabel("edges / idea")
    ax[1, 1].set_title("(d) Graph connectivity"); ax[1, 1].set_ylabel("edges")
    for a in ax.flat:
        a.set_xlabel(xlabel); a.grid(True, color="0.9", lw=0.6); a.set_axisbelow(True)
    ax[0, 0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, f"{base}_curves" + ("_index" if mode == "iteration" else ""))


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
    G = read_graph(run_dir)
    if G is None:
        print(f"  (skip graph snapshot for {label}: no readable graph yet — run still writing?)")
        return
    fig, a = plt.subplots(figsize=(6.0, 6.0))
    print(f"  graph snapshot [{label}]: spring_layout on {G.number_of_nodes()} nodes "
          f"(slow on big graphs)…", flush=True)
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
    G = read_graph(run_dir)
    if G is None:
        print(f"  (skip graph growth for {label}: no readable graph yet — run still writing?)")
        return
    if G.number_of_nodes() == 0:
        return

    def it(attrs):
        try:
            return int(float(attrs.get("iter", 0)))
        except Exception:
            return 0

    max_it = max([it(G.nodes[n]) for n in G] + [0])
    checkpoints = sorted(set(int(round(x)) for x in np.linspace(0, max_it, frames)))
    print(f"  graph growth [{label}]: spring_layout on {G.number_of_nodes()} nodes "
          f"(slow on big graphs)…", flush=True)
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
    G = read_graph(run_dir)
    if G is None:
        print(f"  (skip movie for {label}: no readable graph yet — run still writing?)")
        return
    if G.number_of_nodes() == 0:
        return

    def it(attrs):
        try:
            return int(float(attrs.get("iter", 0)))
        except Exception:
            return 0

    max_it = max([it(G.nodes[n]) for n in G] + [0])
    n_frames = max_it + 1
    print(f"  movie [{label}]: computing layout for {G.number_of_nodes()} nodes "
          f"(slow on big graphs)…", flush=True)
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

    anim = FuncAnimation(fig, draw, frames=range(n_frames), interval=1000 / max(1, fps))
    out = f"{base}_movie_{label.replace(' ', '_')}.gif"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    try:                                                  # progress bar over rendered frames
        from tqdm import tqdm
        bar = tqdm(total=n_frames, desc=f"movie {label}", unit="frame", leave=False,
                   dynamic_ncols=True)
        cb = lambda i, n: bar.update(1)
    except Exception:
        bar, cb = None, None
    print(f"  movie [{label}]: rendering {n_frames} frames → GIF…", flush=True)
    anim.save(out, writer=PillowWriter(fps=fps), progress_callback=cb)
    if bar is not None:
        bar.close()
    plt.close(fig)
    print(f"wrote {out}")


def _short(s, n=22):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def graph_analytics(run_dir, label, base):
    """Rich journal-quality 2x3 panel of graph properties + a graph_analysis JSON."""
    G = read_graph(run_dir)
    if G is None:
        print(f"  (skip analytics for {label}: no readable graph yet — run still writing?)")
        return
    if G.number_of_nodes() < 3:
        print(f"  (skip analytics for {label}: graph too small)")
        return
    print(f"  analytics [{label}]: centralities + advanced metrics on "
          f"{G.number_of_nodes()}n/{G.number_of_edges()}e…", flush=True)
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


def _embed_nodes(G, model=None):
    """Re-embed node labels offline (embeddings aren't stored in graphml). Batched (single model
    load + batched forward passes, with a tqdm progress bar for big graphs).
    Returns {node: unit-vec} or None if sentence-transformers unavailable."""
    try:
        from graphstore import embed_texts, DEFAULT_EMBED_MODEL
        ns = list(G)
        print(f"  embedding {len(ns)} concepts for semantic panels (batched)…", flush=True)
        V = embed_texts([str(G.nodes[n].get("label", n)) for n in ns], model or DEFAULT_EMBED_MODEL)
        return {n: V[i] for i, n in enumerate(ns)}
    except Exception as e:
        print(f"  (semantic panels skipped: embedder unavailable — {e})")
        return None


def _key_below(ax, handles):
    """Put a numbered key BELOW the axes (full names, 2 columns), nothing overlapping the plot."""
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False,
              fontsize=8, ncol=2, handlelength=1.2, columnspacing=1.4,
              title="numbers mark the points above (full names below)", title_fontsize=9)


def broker_detail(G, deg, bet, label, base, topk=15):
    """Standalone, larger broker scatter (degree vs betweenness) with the top-`topk` brokers
    numbered on the plot and a full-name key BELOW it — no labels overlapping the cloud."""
    fig, ax = plt.subplots(figsize=(8.5, 8.8))
    ax.scatter([deg[n] for n in G], [bet[n] for n in G], s=16, color="#1f77b4", alpha=0.45,
               linewidths=0, zorder=1)
    handles = []
    for i, n in enumerate(sorted(G, key=lambda n: bet[n], reverse=True)[:topk], 1):
        h = ax.scatter([deg[n]], [bet[n]], s=140, color="#d62728", edgecolor="white",
                       linewidths=1.0, zorder=3, label=f"{i}.  {G.nodes[n].get('label', n)}")
        ax.annotate(str(i), (deg[n], bet[n]), color="white", fontsize=7.5, fontweight="bold",
                    ha="center", va="center", zorder=4)
        handles.append(h)
    ax.set_title(f"Broker ideas — {label}  (degree vs betweenness)")
    ax.set_xlabel("degree  (# ideas it connects to)")
    ax.set_ylabel("betweenness  (how often it bridges others)")
    ax.grid(True, color="0.92", lw=0.5); ax.set_axisbelow(True)
    _key_below(ax, handles)
    _save(fig, f"{base}_brokers_{label.replace(' ', '_')}"); plt.close(fig)


def semantic_detail(G, P, cmap_dict, pr, label, base, topk=15):
    """Standalone, larger semantic map (PCA, color=community, size=PageRank) with the top-`topk`
    hubs numbered and a full-name key BELOW it."""
    nodes = list(G.nodes)
    pos = {n: P[i] for i, n in enumerate(nodes)}
    fig, ax = plt.subplots(figsize=(9.0, 9.2))
    ax.scatter(P[:, 0], P[:, 1], c=[cmap_dict.get(n, 0) for n in nodes], cmap="tab10",
               s=[30 + 4000 * pr[n] for n in nodes], alpha=0.6, linewidths=0, zorder=1)
    handles = []
    for i, n in enumerate(sorted(G, key=lambda x: pr[x], reverse=True)[:topk], 1):
        h = ax.scatter([pos[n][0]], [pos[n][1]], s=150, color="black", edgecolor="white",
                       linewidths=1.0, zorder=3, label=f"{i}.  {G.nodes[n].get('label', n)}")
        ax.annotate(str(i), pos[n], color="white", fontsize=7.5, fontweight="bold",
                    ha="center", va="center", zorder=4)
        handles.append(h)
    ax.set_title(f"Semantic map — {label}  (PCA; color = community, size = PageRank)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, color="0.92", lw=0.5); ax.set_axisbelow(True)
    _key_below(ax, handles)
    _save(fig, f"{base}_semantic_{label.replace(' ', '_')}"); plt.close(fig)


def graph_structure(run_dir, label, base, embed_model=None):
    """Second analytics panel: structural 'shape of reasoning' properties that the
    first panel doesn't cover — k-core, brokers, articulation points, depth profile,
    a semantic embedding map, and structural-vs-semantic homophily.
    Also emits standalone, label-below detail plots of panels (b) and (e)."""
    G = read_graph(run_dir)
    if G is None:
        print(f"  (skip structure panel for {label}: no readable graph yet — run still writing?)")
        return
    if G.number_of_nodes() < 4:
        print(f"  (skip structure panel for {label}: graph too small)")
        return
    U = G.to_undirected()
    from graphstore import resolve_embed_model
    vecs = _embed_nodes(G, resolve_embed_model(run_dir, embed_model))

    fig, ax = plt.subplots(2, 3, figsize=(11.5, 7.2))
    blue, red, green, purple = "#1f77b4", "#d62728", "#2ca02c", "#9467bd"

    # (a) k-core decomposition — dense conceptual nucleus vs speculative periphery
    Uc = nx.Graph(U); Uc.remove_edges_from(nx.selfloop_edges(Uc))
    core = nx.core_number(Uc)
    csizes = Counter(core.values())
    ks = sorted(csizes)
    ax[0, 0].bar(ks, [csizes[k] for k in ks], color=blue, alpha=0.85)
    ax[0, 0].set_title(f"(a) k-core decomposition (max k={max(ks) if ks else 0})")
    ax[0, 0].set_xlabel("coreness k"); ax[0, 0].set_ylabel("# nodes")

    # (b) brokers — degree vs betweenness; top brokers connect separate clusters
    deg = dict(G.degree())
    nN = G.number_of_nodes()
    if nN <= 3:
        bet = {n: 0.0 for n in G}
    elif nN > 1500:                                       # approx (viz only) — exact is minutes
        print(f"  structure [{label}]: approx betweenness (k=400) on {nN} nodes…", flush=True)
        bet = nx.betweenness_centrality(G, k=min(400, nN), seed=0)
    else:
        bet = nx.betweenness_centrality(G)
    xs = [deg[n] for n in G]; ys = [bet[n] for n in G]
    ax[0, 1].scatter(xs, ys, s=18, color=blue, alpha=0.6, linewidths=0)
    brokers = sorted(G, key=lambda n: bet[n], reverse=True)[:5]
    for n in brokers:
        ax[0, 1].annotate(_short(G.nodes[n].get("label", n), 16), (deg[n], bet[n]),
                          fontsize=6.5, color=red,
                          xytext=(3, 3), textcoords="offset points")
    ax[0, 1].set_title("(b) Broker ideas (degree vs betweenness)")
    ax[0, 1].set_xlabel("degree"); ax[0, 1].set_ylabel("betweenness")

    # (c) articulation points — ideas whose removal fragments the knowledge graph
    Ulcc = U.subgraph(max(nx.connected_components(U), key=len)).copy() if U.number_of_nodes() else U
    base_comp = nx.number_connected_components(U)
    arts = list(nx.articulation_points(Ulcc)) if Ulcc.number_of_nodes() > 2 else []
    split = []
    for n in arts:
        H = U.copy(); H.remove_node(n)
        split.append((n, nx.number_connected_components(H) - base_comp))
    split.sort(key=lambda x: x[1], reverse=True)
    split = split[:12][::-1]
    if split:
        ax[1, 0].barh(range(len(split)), [s for _, s in split], color=red, alpha=0.8)
        ax[1, 0].set_yticks(range(len(split)))
        ax[1, 0].set_yticklabels([_short(G.nodes[n].get("label", n), 18) for n, _ in split], fontsize=7)
        ax[1, 0].set_xlabel("extra fragments if removed")
    else:
        ax[1, 0].text(0.5, 0.5, "no articulation points\n(2-connected)", ha="center", va="center")
    ax[1, 0].set_title(f"(c) Critical connector ideas  ({len(arts)} cut vertices)")

    # (d) reasoning-depth profile — ideas + semantic diversity per hop from seed
    def dep(n):
        try:
            return int(float(G.nodes[n].get("depth", 0)))
        except Exception:
            return 0
    by_depth = Counter(dep(n) for n in G)
    ds = sorted(by_depth)
    ax[1, 1].bar(ds, [by_depth[d] for d in ds], color=green, alpha=0.8, label="ideas")
    ax[1, 1].set_xlabel("reasoning depth (hops from seed)"); ax[1, 1].set_ylabel("# new ideas", color=green)
    if vecs:
        a2 = ax[1, 1].twinx()
        dv = []
        for d in ds:
            Nd = [n for n in G if dep(n) <= d]
            if len(Nd) > 1:
                X = np.stack([vecs[n] for n in Nd]); S = X @ X.T
                dv.append(float(1.0 - S[np.triu_indices(len(X), 1)].mean()))
            else:
                dv.append(float("nan"))
        a2.plot(ds, dv, color=purple, marker="o", ms=3, lw=2)
        a2.set_ylabel("cumulative diversity", color=purple)
    ax[1, 1].set_title("(d) Reasoning-depth profile")

    # (e) semantic map — 2D PCA of idea embeddings, colored by community, sized by PageRank
    if vecs:
        nodes = list(G.nodes)
        X = np.stack([vecs[n] for n in nodes])
        Xc = X - X.mean(0)
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            P = Xc @ Vt[:2].T
        except Exception:
            P = Xc[:, :2]
        try:
            comms = list(nx.community.greedy_modularity_communities(U))
            cmap = {n: i for i, c in enumerate(comms) for n in c}
        except Exception:
            cmap = {n: 0 for n in nodes}
        try:
            pr = nx.pagerank(G) if G.number_of_edges() else {n: 1.0 for n in G}
        except Exception:
            pr = {n: 1.0 / max(1, len(nodes)) for n in G}
        cols = [cmap.get(n, 0) for n in nodes]
        ax[0, 2].scatter(P[:, 0], P[:, 1], c=cols, cmap="tab10",
                         s=[40 + 4000 * pr[n] for n in nodes], alpha=0.75, linewidths=0)
        for n in sorted(G, key=lambda x: pr[x], reverse=True)[:6]:
            i = nodes.index(n)
            ax[0, 2].annotate(_short(G.nodes[n].get("label", n), 16), (P[i, 0], P[i, 1]),
                              fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        ax[0, 2].set_title("(e) Semantic map (PCA, color=community)")
        ax[0, 2].set_xlabel("PC1"); ax[0, 2].set_ylabel("PC2")
    else:
        ax[0, 2].axis("off")

    # (f) structural-vs-semantic homophily — are linked ideas embedding-similar?
    if vecs:
        nodes = list(G.nodes); idx = {n: i for i, n in enumerate(nodes)}
        X = np.stack([vecs[n] for n in nodes])
        conn = [float(np.dot(X[idx[u]], X[idx[v]])) for u, v in U.edges]
        rng = np.random.default_rng(0)
        non = []
        eset = set(map(frozenset, U.edges))
        tries = 0
        while len(non) < max(200, len(conn)) and tries < 20000:
            i, j = rng.integers(0, len(nodes), 2)
            tries += 1
            if i != j and frozenset((nodes[i], nodes[j])) not in eset:
                non.append(float(np.dot(X[i], X[j])))
        ax[1, 2].hist([conn, non], bins=20, color=[green, "0.6"],
                      label=["linked", "random pair"], density=True)
        mc = np.mean(conn) if conn else float("nan")
        mn = np.mean(non) if non else float("nan")
        ax[1, 2].axvline(mc, color=green, lw=1.5, ls="--")
        ax[1, 2].axvline(mn, color="0.4", lw=1.5, ls="--")
        ax[1, 2].set_title(f"(f) Link homophily  (Δ={mc - mn:+.2f})")
        ax[1, 2].set_xlabel("cosine similarity"); ax[1, 2].set_ylabel("density")
        ax[1, 2].legend(fontsize=7, frameon=False)
    else:
        ax[1, 2].axis("off")

    for a in ax.flat:
        a.grid(True, color="0.92", lw=0.5); a.set_axisbelow(True)
    fig.suptitle(f"Reasoning structure — {label}  "
                 f"({G.number_of_nodes()}n / {G.number_of_edges()}e)", y=1.0)
    fig.tight_layout()
    _save(fig, f"{base}_structure_{label.replace(' ', '_')}")
    plt.close(fig)

    # standalone, label-below detail versions of the two crowded panels (b) and (e),
    # reusing the already-computed degree/betweenness/PCA/community/PageRank (no recompute)
    try:
        broker_detail(G, deg, bet, label, base)
        if vecs:
            semantic_detail(G, P, cmap, pr, label, base)
    except Exception as e:
        print(f"  (detail plots skipped for {label}: {e})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="one or more run dirs")
    p.add_argument("--labels", nargs="+", help="legend labels (default: dir names)")
    p.add_argument("--out", default=None,
                   help="figure basename (default: <first-run-dir>/figures/ideation)")
    p.add_argument("--no-graph", action="store_true",
                   help="skip the per-run graph figures entirely (analytics + structure too)")
    p.add_argument("--no-structure", action="store_true",
                   help="skip the second (reasoning-structure) analytics panel")
    p.add_argument("--graph-snapshot", action="store_true",
                   help="also draw the spring-layout node-link snapshot (OFF by default — slow "
                        "on big graphs; open the graphml files in Gephi instead)")
    p.add_argument("--growth-frames", type=int, default=0,
                   help="frames in the spring-layout graph-growth montage (default 0 = skip; "
                        "slow on big graphs)")
    p.add_argument("--movie", action="store_true", help="also render an animated GIF of growth")
    p.add_argument("--movie-fps", type=int, default=2)
    p.add_argument("--embed-model", dest="embed_model", default=None,
                   help="sentence-transformers id for re-embedding the semantic panels "
                        "(default: each run's recorded model, else embeddinggemma-300m)")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=None,
                   help="truncate every run to iter <= this for fair cross-run figures "
                        "(applies to all panels/metrics)")
    args = p.parse_args()
    global MAX_ITER
    MAX_ITER = args.max_iter
    if MAX_ITER is not None:
        print(f"[plot] truncating all runs to iter <= {MAX_ITER}")
    if args.out is None:                                  # default: inside the (first) run dir
        args.out = os.path.join(args.runs[0].rstrip("/"), "figures", "ideation")
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    loaded = [load_run(r) for r in args.runs]
    curves(loaded, args.runs, labels, args.out, mode="depth", embed_model=args.embed_model)
    curves(loaded, args.runs, labels, args.out, mode="iteration", embed_model=args.embed_model)
    bars([final_metrics(s, rows) for s, rows in loaded], labels, args.out)
    if not args.no_graph:
        for rd, lab in zip(args.runs, labels):
            if args.graph_snapshot:                       # opt-in: spring-layout node-link snapshot
                graph_snapshot(rd, lab, args.out)
            graph_analytics(rd, lab, args.out)
            if not args.no_structure:
                graph_structure(rd, lab, args.out, embed_model=args.embed_model)
            if args.growth_frames > 0:                     # opt-in: spring-layout growth montage
                graph_growth(rd, lab, args.out, frames=args.growth_frames)
            if args.movie:
                graph_movie(rd, lab, args.out, fps=args.movie_fps)


if __name__ == "__main__":
    main()
