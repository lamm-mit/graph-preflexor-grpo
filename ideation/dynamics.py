#!/usr/bin/env python
"""Reasoning *dynamics*: HOW the idea graph grows with test-time compute (not just how much).

`scaling.py` answers "more compute -> more ideas/reach/recombinations" with cumulative-total
curves. This companion answers "*how* does it grow?" by replaying the single final graph in
`iter` order (every node and edge carries its birth iteration) and measuring rates and structural
events. The two are complementary: scaling = the totals story, dynamics = the mechanism story.

Five panels (each a distinct dynamic; all combine graph structure with the embedding geometry):

  D1  Explore -> consolidate.  Per iteration-bin, new concepts split into *novel* (embedding-far
      from the running idea centroid at arrival) vs *consolidating* (in-fill near what's known).
      The novel rate falls and the consolidating rate rises — the mechanism BEHIND coverage
      saturation: the model shifts from naming new territory to wiring up what's there.

  D2  Fragments -> one fabric.  Number of connected components and giant-component fraction vs
      iteration. Early reasoning is many islands; they merge into one. Merge events = "the model
      connected two sub-fields"; the largest is annotated.

  D3  Leap size over compute.  Each new edge that joins concepts which were previously far apart
      collapses a graph distance. Scatter (iteration, prior-distance-collapsed); point size = the
      endpoints' embedding distance. Shows whether the big creative leaps come early or are
      unlocked late. Edges that join two *separate* components are the largest leaps (top band).

  D4  Concept incubation.  Degree(t) for the highest-final-degree concepts — reveals ideas
      introduced early that lie dormant, then ignite late as the model returns to build on them.

  D5  Exploration trajectory.  On the final PCA of idea-space, the centroid of each iteration-bin's
      *new* concepts, traced as a path colored by time over the cloud of all concepts — does the
      search wander out and back (explore then consolidate) or drift monotonically?

    python dynamics.py --run runs/exp_leap --out runs/exp_leap/figures/dynamics
    python dynamics.py --run runs/exp --embed-model google/embeddinggemma-300m --max-iter 1500
"""
import argparse
import os
from collections import defaultdict, deque

import numpy as np
import networkx as nx

import insights as I

POS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]


def _iv(G, n, k):
    try:
        return int(float(G.nodes[n].get(k, 0)))
    except Exception:
        return 0


def _edge_iter(d):
    try:
        return int(float(d.get("iter", 0)))
    except Exception:
        return 0


def _bfs_dist(adj, src, dst, cutoff):
    """Unweighted shortest-path length src->dst over adjacency `adj` (dict[node]->set), bounded by
    `cutoff`. Returns the distance, or None if unreachable within cutoff (treated as a new bridge)."""
    if src == dst:
        return 0
    seen = {src}
    q = deque([(src, 0)])
    while q:
        n, d = q.popleft()
        if d >= cutoff:
            continue
        for m in adj[n]:
            if m == dst:
                return d + 1
            if m not in seen:
                seen.add(m)
                q.append((m, d + 1))
    return None


def _components(present, adj):
    """(n_components, giant_fraction) over the currently-present node set."""
    if not present:
        return 0, 0.0
    seen, sizes = set(), []
    for s in present:
        if s in seen:
            continue
        sz, q = 0, deque([s]); seen.add(s)
        while q:
            n = q.popleft(); sz += 1
            for m in adj[n]:
                if m in present and m not in seen:
                    seen.add(m); q.append(m)
        sizes.append(sz)
    return len(sizes), (max(sizes) / len(present))


def compute(run_dir, embed_model=None, n_ckpt=30, n_bins=25, top_incubate=6, bfs_cutoff=10):
    from graphstore import resolve_embed_model
    G = I.load_graph(run_dir)
    model = resolve_embed_model(run_dir, embed_model)
    nodes = list(G.nodes)
    print(f"[dynamics] {run_dir}: embedding {len(nodes)} concepts with '{model}'…", flush=True)
    vecs = I.embed_nodes(G, model)
    if not vecs:
        raise SystemExit(f"{run_dir}: embeddings unavailable with '{model}' — dynamics needs them "
                         f"(huggingface-cli login, or --embed-model all-MiniLM-L6-v2).")
    X = np.stack([vecs[n] for n in nodes]).astype(np.float32)
    idx = {n: i for i, n in enumerate(nodes)}
    node_iter = {n: _iv(G, n, "iter") for n in nodes}
    max_it = max(node_iter.values()) if node_iter else 0
    seed = min(nodes, key=lambda n: (_iv(G, n, "depth"), _iv(G, n, "iter")))

    # ---- D1 + D5: arrival novelty & per-bin new-node centroids (node replay in iter order) ----
    order = sorted(nodes, key=lambda n: (node_iter[n], _iv(G, n, "depth")))
    csum = np.zeros(X.shape[1], dtype=np.float64); cnt = 0
    arr_nov = {}                                          # node -> novelty at arrival (1 - cos to centroid)
    for n in order:
        v = X[idx[n]]
        if cnt:
            c = csum / cnt; nrm = np.linalg.norm(c)
            arr_nov[n] = float(1.0 - (np.dot(v, c) / nrm if nrm > 1e-9 else 0.0))
        else:
            arr_nov[n] = 0.0
        csum += v; cnt += 1
    nov_vals = np.array([arr_nov[n] for n in order[1:]]) if len(order) > 1 else np.array([0.0])
    nov_med = float(np.median(nov_vals))
    edges_bins = np.linspace(0, max_it, n_bins + 1)
    bin_of = lambda it: min(n_bins - 1, int(np.searchsorted(edges_bins, it, side="right") - 1))
    novel_pb = np.zeros(n_bins); cons_pb = np.zeros(n_bins)
    for n in order[1:]:
        b = bin_of(node_iter[n])
        (novel_pb if arr_nov[n] > nov_med else cons_pb)[b] += 1

    # PCA(2) of the whole idea space for the trajectory (D5)
    Xc = X - X.mean(0)
    try:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False); P = Xc @ Vt[:2].T
    except Exception:
        P = Xc[:, :2]
    bin_centroid, bin_mid = [], []
    for b in range(n_bins):
        members = [idx[n] for n in order if bin_of(node_iter[n]) == b]
        if members:
            bin_centroid.append(P[members].mean(0)); bin_mid.append(0.5 * (edges_bins[b] + edges_bins[b + 1]))
    bin_centroid = np.array(bin_centroid) if bin_centroid else np.zeros((0, 2))

    # ---- D2 + D3 + D4: edge replay in iter order ----
    udeg = dict(G.to_undirected().degree())
    track = [n for n, _ in sorted(udeg.items(), key=lambda kv: kv[1], reverse=True)[:top_incubate]]
    events = [(node_iter[n], 0, n, None) for n in nodes]          # 0 = node birth
    for u, v, d in G.to_undirected().edges(data=True):
        if u != v:
            events.append((_edge_iter(d), 1, u, v))               # 1 = edge birth
    events.sort(key=lambda e: (e[0], e[1]))
    ts = sorted(set(int(round(t)) for t in np.linspace(0, max_it, n_ckpt)))

    adj = defaultdict(set); present = set()
    leaps = []                                                    # (iter, prior_distance_or_inf, embed_dist)
    comp_curve, giant_curve = [], []
    deg_curve = {n: [] for n in track}
    deg_now = defaultdict(int)
    ti, big_merge = 0, (0, 0.0, None)                             # (iter, giant_jump, label)
    prev_giant = 0.0
    ev_i = 0
    for t in ts:
        while ev_i < len(events) and events[ev_i][0] <= t:
            _, kind, a, b = events[ev_i]; ev_i += 1
            if kind == 0:
                present.add(a)
            else:
                # a genuine recombination/leap joins two concepts ALREADY woven into the graph
                # (both already have a link) that were far apart — NOT a fresh leaf attaching.
                if a in present and b in present and deg_now[a] >= 1 and deg_now[b] >= 1:
                    d = _bfs_dist(adj, a, b, bfs_cutoff)
                    if d is None or d >= 2:                       # collapses a real distance
                        ed = float(1.0 - np.dot(X[idx[a]], X[idx[b]]))
                        leaps.append((events[ev_i - 1][0],
                                      (bfs_cutoff + 1) if d is None else d, ed))
                adj[a].add(b); adj[b].add(a)
                deg_now[a] += 1; deg_now[b] += 1
        nc, gf = _components(present, adj)
        comp_curve.append(nc); giant_curve.append(gf)
        if gf - prev_giant > big_merge[1]:
            big_merge = (t, gf - prev_giant, None)
        prev_giant = gf
        for n in track:
            deg_curve[n].append(deg_now[n])

    return {"model": model, "max_it": max_it, "seed": seed,
            "bins": (edges_bins, novel_pb, cons_pb, nov_med),
            "traj": (P, bin_centroid, bin_mid),
            "comp": (ts, comp_curve, giant_curve, big_merge),
            "leaps": np.array(leaps) if leaps else np.zeros((0, 3)),
            "incub": (ts, deg_curve, track, {n: str(G.nodes[n].get("label", n)) for n in track}),
            "bfs_cutoff": bfs_cutoff,
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}


def make_figure(run_dir, out, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
                                "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    D = compute(run_dir, **kw)
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.0))

    # D1 — explore -> consolidate
    edges_bins, novel_pb, cons_pb, nov_med = D["bins"]
    mids = 0.5 * (edges_bins[:-1] + edges_bins[1:])
    a = ax[0, 0]
    a.stackplot(mids, novel_pb, cons_pb, labels=["novel (frontier)", "consolidating (in-fill)"],
                colors=["#d62728", "#1f77b4"], alpha=0.85)
    a.set_title("(D1) Explore → consolidate"); a.set_xlabel("reasoning iteration")
    a.set_ylabel("new concepts per bin"); a.legend(frameon=False, fontsize=8, loc="upper right")

    # D2 — fragments -> one fabric
    ts, comp_curve, giant_curve, big_merge = D["comp"]
    a = ax[0, 1]; a2 = a.twinx(); a2.spines["top"].set_visible(False)
    a.plot(ts, comp_curve, color="#1f77b4", lw=2, marker="o", ms=3, label="# components")
    a2.plot(ts, giant_curve, color="#d62728", lw=2, marker="s", ms=3, label="giant fraction")
    a.set_title("(D2) Fragments → one fabric"); a.set_xlabel("reasoning iteration")
    a.set_ylabel("# connected components", color="#1f77b4")
    a2.set_ylabel("giant-component fraction", color="#d62728"); a2.set_ylim(0, 1.02)
    if big_merge[2] is None and big_merge[0]:
        a2.axvline(big_merge[0], color="0.6", ls="--", lw=1)
        a2.annotate("largest merge", (big_merge[0], 0.5), fontsize=7.5, color="0.4",
                    rotation=90, va="center", ha="right")

    # D3 — leap size over compute
    a = ax[0, 2]; L = D["leaps"]; cutoff = D["bfs_cutoff"]
    if len(L):
        finite = L[:, 1] <= cutoff
        a.scatter(L[finite, 0], L[finite, 1], s=8 + 120 * L[finite, 2], alpha=0.5,
                  color="#2ca02c", edgecolor="none", label="distance collapsed")
        if (~finite).any():
            a.scatter(L[~finite, 0], np.full((~finite).sum(), cutoff + 1),
                      s=8 + 120 * L[~finite, 2], alpha=0.6, color="#d62728", marker="^",
                      edgecolor="none", label="joined separate clusters")
        a.axhline(cutoff + 1, color="0.85", lw=0.8)
        a.legend(frameon=False, fontsize=8, loc="upper right")
    a.set_title("(D3) Leap size over compute"); a.set_xlabel("reasoning iteration")
    a.set_ylabel("prior graph distance collapsed\n(point size = embedding distance)")

    # D4 — concept incubation
    ts4, deg_curve, track, labmap = D["incub"]
    a = ax[1, 0]
    for n, col in zip(track, POS):
        lab = labmap[n]; lab = lab if len(lab) <= 28 else lab[:27] + "…"
        a.plot(ts4, deg_curve[n], lw=1.8, color=col, marker="o", ms=2.5, label=lab)
    a.set_title("(D4) Concept incubation (degree over time)"); a.set_xlabel("reasoning iteration")
    a.set_ylabel("degree of top concepts")
    a.legend(frameon=False, fontsize=7.2, loc="upper left", ncol=1)

    # D5 — exploration trajectory
    a = ax[1, 1]; P, bc, mid = D["traj"]
    a.scatter(P[:, 0], P[:, 1], s=8, color="0.8", alpha=0.5, edgecolor="none", zorder=1)
    if len(bc) > 1:
        for k in range(len(bc) - 1):
            a.annotate("", xy=bc[k + 1], xytext=bc[k],
                       arrowprops=dict(arrowstyle="->", color=plt.cm.viridis(k / (len(bc) - 1)), lw=2),
                       zorder=3)
        sc = a.scatter(bc[:, 0], bc[:, 1], c=mid, cmap="viridis", s=34, zorder=4, edgecolor="white",
                       linewidths=0.6)
        cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02); cb.set_label("iteration", fontsize=8)
    a.set_title("(D5) Exploration trajectory of new ideas"); a.set_xlabel("PC1"); a.set_ylabel("PC2")

    # legend / caption cell
    a = ax[1, 2]; a.axis("off")
    a.text(0.0, 0.98, "Reasoning dynamics", fontsize=12, fontweight="bold", va="top")
    cap = (f"run: {os.path.basename(run_dir.rstrip('/'))}\n"
           f"{D['n_nodes']} concepts · {D['n_edges']} links · embed: {D['model']}\n\n"
           "D1  the novel→consolidating shift is the mechanism behind coverage saturation;\n"
           "     recombination (scaling panel d) keeps rising because pairs, not nodes, grow.\n"
           "D2  components collapsing to one = sub-fields being connected.\n"
           "D3  late high points = big creative leaps unlocked by test-time compute.\n"
           "D4  flat-then-rising curves = dormant ideas the model returns to and builds on.\n"
           "D5  a path that wanders out then back = explore-then-consolidate.")
    a.text(0.0, 0.86, cap, fontsize=8.4, va="top", family="monospace")
    fig.legend(handles=[Line2D([0], [0], color="#d62728", lw=6, alpha=.85),
                        Line2D([0], [0], color="#1f77b4", lw=6, alpha=.85)],
               labels=["novel / frontier", "consolidating / in-fill"],
               loc="lower right", bbox_to_anchor=(0.985, 0.04), frameon=False, fontsize=8)

    fig.suptitle("How the reasoning graph grows with test-time compute (dynamics)", y=1.0, fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png/.svg/.pdf")
    nc_last = D["comp"][1][-1] if D["comp"][1] else 0
    print(f"[dynamics] {D['n_nodes']} concepts, {len(D['leaps'])} recombination edges, "
          f"giant fraction {D['comp'][2][-1]:.2f}, {nc_last} final components")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run dir produced by ideate.py")
    p.add_argument("--out", default=None, help="figure basename (default: <run>/figures/dynamics)")
    p.add_argument("--embed-model", dest="embed_model", default=None,
                   help="override the sentence-transformers id used for re-embedding")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=None,
                   help="truncate the graph to iter <= this (matched-compute cutoff)")
    p.add_argument("--checkpoints", type=int, default=30, help="checkpoints for D2/D3/D4 time series")
    p.add_argument("--bins", type=int, default=25, help="iteration bins for D1/D5")
    p.add_argument("--top-incubate", dest="top_incubate", type=int, default=6,
                   help="how many top-degree concepts to trace in D4")
    p.add_argument("--bfs-cutoff", dest="bfs_cutoff", type=int, default=10,
                   help="max prior graph distance probed per new edge in D3")
    args = p.parse_args()

    if args.max_iter is not None:
        I.MAX_ITER = args.max_iter
        print(f"[dynamics] truncating to iter <= {args.max_iter}")
    out = args.out or os.path.join(args.run.rstrip("/"), "figures", "dynamics")
    make_figure(args.run, out, embed_model=args.embed_model, n_ckpt=args.checkpoints,
                n_bins=args.bins, top_incubate=args.top_incubate, bfs_cutoff=args.bfs_cutoff)


if __name__ == "__main__":
    main()
