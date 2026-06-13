#!/usr/bin/env python
"""Reasoning *dynamics*: HOW the idea graph grows with test-time compute (not just how much).

`scaling.py` answers "more compute -> more ideas/reach/recombinations" with cumulative-total
curves. This companion answers "*how* does it grow?" by replaying the single final graph in
`iter` order (every node and edge carries its birth iteration) and measuring rates and structural
events. The two are complementary: scaling = the totals story, dynamics = the mechanism story.

Every panel is **size-robust** — it uses embedding geometry and mesoscale structure, NOT raw graph
distance/diameter (which shrink mechanically as the graph densifies, the same confound `scaling.py`
avoids). Five panels, each combining graph structure with the embedding geometry:

  D1  Explore vs consolidate.  The total new-concept *rate* falls (the saturation signal); within it,
      each new concept is split into *novel* (embedding-far from the running idea centroid at arrival)
      vs *consolidating* (in-fill). A line tracks the **novel fraction** — read the trend, don't
      assume one (some runs consolidate the core first, then explore the periphery).

  D2  Theme structure over compute.  Number of **modularity communities** and modularity **Q** vs
      iteration — the mesoscale "sub-fields forming / fusing" story. This is the right lens once the
      graph is one connected component (raw connected-components would just read ~1 the whole time).

  D3  Recombination distance (size-robust).  Each new edge that links two concepts ALREADY woven into
      the graph (both have a prior link) is a recombination; plot its endpoints' **embedding distance**
      vs iteration (hexbin density + per-bin median). A rising median = later edges bridge genuinely
      *more distant* concepts — creative reach unlocked by compute, with no graph-size confound.

  D4  Late bloomers.  Degree(t) for the concepts that gained most of their degree **late** (selected
      by late-half growth fraction, not final degree) — ideas introduced early, dormant, then built on.

  D5  Exploration radius.  Mean embedding distance-from-seed of each bin's *new* concepts (± IQR) vs
      iteration. A rise-then-fall is the explore-then-consolidate signature, size-robustly.

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


def _bfs_reachable(adj, src, dst, cutoff):
    """True if dst is reachable from src within `cutoff` hops (used only to gate recombinations)."""
    if src == dst:
        return True
    seen = {src}
    q = deque([(src, 0)])
    while q:
        n, d = q.popleft()
        if d >= cutoff:
            continue
        for m in adj[n]:
            if m == dst:
                return True
            if m not in seen:
                seen.add(m); q.append((m, d + 1))
    return False


def _components(present, adj):
    if not present:
        return 0
    seen, k = set(), 0
    for s in present:
        if s in seen:
            continue
        k += 1; q = deque([s]); seen.add(s)
        while q:
            n = q.popleft()
            for m in adj[n]:
                if m in present and m not in seen:
                    seen.add(m); q.append(m)
    return k


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
    sv = X[idx[seed]]

    edges_bins = np.linspace(0, max_it, n_bins + 1)
    mids = 0.5 * (edges_bins[:-1] + edges_bins[1:])
    bin_of = lambda it: min(n_bins - 1, int(np.searchsorted(edges_bins, it, side="right") - 1))

    # ---- D1 (arrival novelty) + D5 (radius from seed), via node replay in iter order ----
    order = sorted(nodes, key=lambda n: (node_iter[n], _iv(G, n, "depth")))
    csum = np.zeros(X.shape[1], dtype=np.float64); cnt = 0
    arr_nov = {}
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
    novel_pb = np.zeros(n_bins); cons_pb = np.zeros(n_bins)
    for n in order[1:]:
        b = bin_of(node_iter[n]); (novel_pb if arr_nov[n] > nov_med else cons_pb)[b] += 1
    total_pb = novel_pb + cons_pb
    frac_novel = np.where(total_pb > 0, novel_pb / np.maximum(total_pb, 1), np.nan)

    rad = {n: float(1.0 - np.dot(X[idx[n]], sv)) for n in nodes}   # embedding distance from seed
    rad_mean = np.full(n_bins, np.nan); rad_lo = np.full(n_bins, np.nan); rad_hi = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = [rad[n] for n in order if bin_of(node_iter[n]) == b]
        if vals:
            rad_mean[b] = np.mean(vals); rad_lo[b] = np.percentile(vals, 25); rad_hi[b] = np.percentile(vals, 75)

    # ---- edge replay: D2 (communities), D3 (recombination embedding distance), D4 (degree traj) ----
    events = [(node_iter[n], 0, n, None) for n in nodes]
    for u, v, d in G.to_undirected().edges(data=True):
        if u != v:
            events.append((_edge_iter(d), 1, u, v))
    events.sort(key=lambda e: (e[0], e[1]))
    ts = sorted(set(int(round(t)) for t in np.linspace(0, max_it, n_ckpt)))

    adj = defaultdict(set); present = set()
    leaps = []                                            # (iter, embedding distance) per recombination
    deg_now = defaultdict(int)
    deg_series = {n: [] for n in nodes}
    comm_count, modQ = [], []
    ev_i = 0
    print(f"[dynamics] {run_dir}: replaying {len(events)} events over {len(ts)} checkpoints "
          f"(community detection per checkpoint)…", flush=True)
    for t in ts:
        while ev_i < len(events) and events[ev_i][0] <= t:
            ei, kind, a, b = events[ev_i]; ev_i += 1
            if kind == 0:
                present.add(a)
            else:
                # recombination = both endpoints already in the fabric (degree >= 1); the new edge is
                # by construction non-adjacent, so its embedding distance is the size-robust "reach".
                if a in present and b in present and deg_now[a] >= 1 and deg_now[b] >= 1:
                    leaps.append((ei, float(1.0 - np.dot(X[idx[a]], X[idx[b]]))))
                adj[a].add(b); adj[b].add(a); deg_now[a] += 1; deg_now[b] += 1
        for n in nodes:
            deg_series[n].append(deg_now[n])
        H = nx.Graph(); H.add_nodes_from(present)          # community structure on the present subgraph
        for n in present:
            for m in adj[n]:
                if m in present and n < m:
                    H.add_edge(n, m)
        try:
            comms = list(nx.community.greedy_modularity_communities(H)) if H.number_of_edges() else \
                [{n} for n in present]
            comm_count.append(len(comms))
            modQ.append(float(nx.community.modularity(H, comms)) if H.number_of_edges() else 0.0)
        except Exception:
            comm_count.append(_components(present, adj)); modQ.append(float("nan"))

    # ---- D4: pick "late bloomers" by fraction of degree gained in the late half of compute ----
    half = min(max(int(np.searchsorted(ts, max_it / 2.0)), 1), len(ts) - 1)
    final_deg = {n: deg_series[n][-1] for n in nodes}
    pos = [d for d in final_deg.values() if d > 0]
    floor = max(2.0, float(np.median(pos))) if pos else 2.0      # ignore low-degree noise
    cand = [n for n in nodes if final_deg[n] >= floor]
    takeoff = lambda n: (final_deg[n] - deg_series[n][half]) / (final_deg[n] + 1e-9)
    track = sorted(cand, key=takeoff, reverse=True)[:top_incubate]

    return {"model": model, "max_it": max_it, "edges_bins": edges_bins, "mids": mids,
            "d1": (novel_pb, cons_pb, frac_novel),
            "d2": (ts, comm_count, modQ),
            "d3": np.array(leaps) if leaps else np.zeros((0, 2)),
            "d4": (ts, deg_series, track, {n: str(G.nodes[n].get("label", n)) for n in track}),
            "d5": (rad_mean, rad_lo, rad_hi),
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}


def make_figure(run_dir, out, n_bins=25, show_text=True, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
                                "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    D = compute(run_dir, n_bins=n_bins, **kw)
    mids, ebins = D["mids"], D["edges_bins"]
    fig, axg = plt.subplots(3, 2, figsize=(10.5, 11.8)); ax = axg.ravel()

    # D1 — explore vs consolidate (rate + novel-fraction line)
    novel_pb, cons_pb, frac = D["d1"]
    a = ax[0]
    a.stackplot(mids, novel_pb, cons_pb, labels=["novel", "consolidating (in-fill)"],
                colors=["#d62728", "#1f77b4"], alpha=0.85)
    a.set_title("(a) Explore vs consolidate"); a.set_xlabel("reasoning iteration")
    a.set_ylabel("new concepts per bin"); a.legend(frameon=False, fontsize=8, loc="upper right")
    a2 = a.twinx(); a2.spines["top"].set_visible(False)
    a2.plot(mids, frac, color="0.15", lw=1.8, marker="o", ms=2.5)
    a2.set_ylabel("novel fraction", color="0.15"); a2.set_ylim(0, 1.02)

    # D2 — theme structure (communities + modularity Q)
    ts, comm_count, modQ = D["d2"]
    a = ax[1]; a2 = a.twinx(); a2.spines["top"].set_visible(False)
    a.plot(ts, comm_count, color="#1f77b4", lw=2, marker="o", ms=3)
    a2.plot(ts, modQ, color="#d62728", lw=2, marker="s", ms=3)
    a.set_title("(b) Theme structure over compute"); a.set_xlabel("reasoning iteration")
    a.set_ylabel("# communities", color="#1f77b4")
    a2.set_ylabel("modularity Q", color="#d62728"); a2.set_ylim(0, 1.02)

    # D3 — recombination embedding distance (size-robust), hexbin + per-bin median
    a = ax[2]; L = D["d3"]
    if len(L):
        hb = a.hexbin(L[:, 0], L[:, 1], gridsize=28, cmap="Greens", mincnt=1, linewidths=0)
        med_x, med_y = [], []
        for b in range(n_bins):
            m = (L[:, 0] >= ebins[b]) & (L[:, 0] < ebins[b + 1])
            if m.any():
                med_x.append(mids[b]); med_y.append(float(np.median(L[m, 1])))
        a.plot(med_x, med_y, color="#d62728", lw=2.2, label="per-bin median")
        a.legend(frameon=False, fontsize=8, loc="lower right")
        cb = fig.colorbar(hb, ax=a, fraction=0.046, pad=0.02); cb.set_label("# recombination edges", fontsize=8)
    a.set_title("(c) Recombination distance over compute"); a.set_xlabel("reasoning iteration")
    a.set_ylabel("embedding distance of linked concepts")

    # D4 — late bloomers (degree over time). FULL labels, never truncated — the legend goes BELOW
    # the panel (one per row) so long concept names are shown complete instead of being clipped.
    ts4, deg_series, track, labmap = D["d4"]
    a = ax[3]
    for n, col in zip(track, POS):
        a.plot(ts4, deg_series[n], lw=1.8, color=col, marker="o", ms=2.5, label=labmap[n])
    a.set_title("(d) Late bloomers (degree over time)"); a.set_xlabel("reasoning iteration")
    a.set_ylabel("degree")
    if track:
        a.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                 ncol=1, handlelength=1.6, title="late-blooming concepts (full names)", title_fontsize=8)

    # D5 — exploration radius (embedding distance from seed of new concepts)
    rad_mean, rad_lo, rad_hi = D["d5"]
    a = ax[4]
    a.fill_between(mids, rad_lo, rad_hi, color="#2ca02c", alpha=0.18, label="IQR")
    a.plot(mids, rad_mean, color="#2ca02c", lw=2, marker="o", ms=2.5, label="mean")
    a.set_title("(e) Exploration radius of new ideas"); a.set_xlabel("reasoning iteration")
    a.set_ylabel("embedding distance from seed"); a.legend(frameon=False, fontsize=8, loc="lower right")

    # text panel (omit with --no-text / show_text=False)
    qlast = D["d2"][2][-1]
    a = ax[5]; a.axis("off")
    if show_text:
        cap = (f"{D['n_nodes']} concepts · {D['n_edges']} links · embed: {D['model']}\n"
               f"final: {D['d2'][1][-1]} communities · Q={qlast:.2f}\n\n"
               "All panels are SIZE-ROBUST (embedding geometry + mesoscale\n"
               "structure), not raw graph distance — which shrinks as the\n"
               "graph densifies (the confound scaling.py also avoids).\n\n"
               "(a)  total new-concept rate falls = saturation; the line is the\n"
               "     novel fraction — read its trend, don't assume one.\n"
               "(b)  communities/Q = sub-fields forming & fusing (right lens once\n"
               "     the graph is one connected component).\n"
               "(c)  rising median = later edges bridge MORE distant concepts.\n"
               "(d)  flat-then-rising = dormant ideas the model returns to.\n"
               "(e)  rise-then-fall = explore then consolidate.")
        a.text(0.0, 0.98, "Reasoning dynamics", fontsize=12, fontweight="bold", va="top")
        a.text(0.0, 0.88, cap, fontsize=8.0, va="top", family="monospace")

    fig.suptitle("How the reasoning graph grows with test-time compute (dynamics)", y=1.0, fontsize=13)
    fig.tight_layout(h_pad=0.4)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png/.svg/.pdf")
    print(f"[dynamics] {D['n_nodes']} concepts, {len(D['d3'])} recombination edges, "
          f"{D['d2'][1][-1]} final communities (Q={qlast:.2f})")


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
    p.add_argument("--bins", type=int, default=25, help="iteration bins for D1/D3-median/D5")
    p.add_argument("--top-incubate", dest="top_incubate", type=int, default=6,
                   help="how many late-blooming concepts to trace in D4")
    p.add_argument("--bfs-cutoff", dest="bfs_cutoff", type=int, default=10,
                   help="(reserved) max hops probed when gating recombinations")
    p.add_argument("--no-text", dest="no_text", action="store_true",
                   help="omit the in-figure text panel (use the paper caption instead)")
    args = p.parse_args()

    if args.max_iter is not None:
        I.MAX_ITER = args.max_iter
        print(f"[dynamics] truncating to iter <= {args.max_iter}")
    out = args.out or os.path.join(args.run.rstrip("/"), "figures", "dynamics")
    make_figure(args.run, out, embed_model=args.embed_model, n_ckpt=args.checkpoints,
                n_bins=args.bins, top_incubate=args.top_incubate, bfs_cutoff=args.bfs_cutoff,
                show_text=not args.no_text)


if __name__ == "__main__":
    main()
