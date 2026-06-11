#!/usr/bin/env python
"""Scaling of surprising-insight yield with test-time compute.

Headline result for "more test-time compute -> more surprising insights": as the reasoning
model spends more compute (cumulative tokens / reasoning iterations), it keeps uncovering MORE,
FARTHER-reaching, and genuinely ATYPICAL connections — i.e. novelty/creativity grows with
compute. Uses *size-robust* metrics measured against the global embedding-similarity null, and
deliberately AVOIDS nearest-prior novelty (which is confounded by graph size — it mechanically
declines as priors accumulate, so it can't support a "novelty increases" claim).

For each compute checkpoint t it reconstructs the graph-so-far (nodes/edges with iter <= t, read
straight off the single final graph's provenance) and computes:
  (a) distinct ideas (fluency)
  (b) idea-space expansion   — total embedding variance (the explored space growing; note this
      typically SATURATES — a real finding: the topic's concept space converges)
  (c) frontier reach         — max embedding distance from the seed (also tends to saturate)
  (d) surprising recombinations — cumulative # of ATYPICAL concept pairs (combination z < --z-thr
      vs the GLOBAL pairwise-similarity distribution) that the model BRIDGED via a shared
      intermediate concept (graph distance 2, NOT a direct edge — direct edges are homophilic,
      so they are the wrong object). This is the recombination/insight signal and can keep
      growing even when (b)/(c) plateau. **This is the headline panel.**

x-axis defaults to cumulative tokens (the real test-time compute, from growth.csv); use
`--x iter` for reasoning iterations. Pass several runs to overlay strategies.

    python scaling.py --run runs/exp_leap --out runs/exp_leap/figures/scaling
    python scaling.py --runs runs/exp2 runs/exp_novelty_2 runs/exp_leap \
        --labels frontier novelty leap --out figures/scaling_compare
"""
import argparse
import csv
import json
import os

import numpy as np
import networkx as nx

import insights as I

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _iv(G, n, k):
    try:
        return int(float(G.nodes[n].get(k, 0)))
    except Exception:
        return 0


def _token_map(run_dir):
    """(iters, cum_tokens) arrays from growth.csv, or None if unavailable."""
    gp = os.path.join(run_dir, "growth.csv")
    if not os.path.exists(gp):
        return None
    its, tok = [], []
    try:
        for r in csv.DictReader(open(gp)):
            its.append(int(float(r["iter"])))
            tok.append(float(r.get("cum_tokens", 0)))
    except Exception:
        return None
    return (np.array(its), np.array(tok)) if its else None


def scaling_curves(run_dir, embed_model=None, n_ckpt=40, z_thr=-1.0, x="tokens"):
    """Compute the size-robust surprise metrics at a grid of compute checkpoints for one run."""
    from graphstore import resolve_embed_model
    G = I.load_graph(run_dir)
    model = resolve_embed_model(run_dir, embed_model)
    nodes = list(G.nodes)
    print(f"[scaling] {run_dir}: embedding {len(nodes)} concepts with '{model}'…", flush=True)
    vecs = I.embed_nodes(G, model)
    if not vecs:
        raise SystemExit(f"{run_dir}: embeddings unavailable with '{model}' — scaling needs them "
                         f"(huggingface-cli login, or --embed-model all-MiniLM-L6-v2).")
    X = np.stack([vecs[n] for n in nodes]).astype(np.float32)
    idx = {n: i for i, n in enumerate(nodes)}
    node_iter = np.array([_iv(G, n, "iter") for n in nodes])
    seed = min(nodes, key=lambda n: (_iv(G, n, "depth"), _iv(G, n, "iter")))
    sv = X[idx[seed]]

    # global pairwise-cosine null (exact mean/std of all i<j cosines, chunked — size-robust
    # reference so "atypical" is relative to the whole idea space, not to graph size)
    n = len(nodes); ssum = ssq = 0.0; cnt = 0; CH = 2048
    for s in range(0, n, CH):
        B = X[s:min(n, s + CH)] @ X.T
        for r in range(s, min(n, s + CH)):
            row = B[r - s, r + 1:]
            ssum += float(row.sum()); ssq += float((row * row).sum()); cnt += row.size
    mu = ssum / cnt if cnt else 0.0
    sd = (max(ssq / cnt - mu * mu, 0.0) ** 0.5) if cnt else 1.0
    sd = sd or 1.0

    # SURPRISING RECOMBINATIONS: atypical concept pairs (combination z < z_thr) that the model
    # bridged via a shared intermediate concept (graph distance 2, NOT a direct edge — direct
    # edges are homophilic, so they're the wrong object). For each such pair we record the
    # iteration it was first realized = max(both node iters, both connecting-edge iters), and
    # count how many have accumulated by each compute checkpoint. This is the recombination
    # ("connected two unrelated ideas through reasoning") yield, not local linking.
    U = G.to_undirected()
    adj = {n: {} for n in nodes}
    for u, v, d in U.edges(data=True):
        if u == v:
            continue
        try:
            ei = int(float(d.get("iter", 0)))
        except Exception:
            ei = 0
        if v not in adj[u] or ei < adj[u][v]:
            adj[u][v] = ei; adj[v][u] = ei
    node_it = {n: int(node_iter[idx[n]]) for n in nodes}
    direct = set(frozenset((u, v)) for u, v in U.edges() if u != v)

    HUB, CAP = 200, 400000
    realize = {}                                          # frozenset(pair) -> earliest realization iter
    print(f"[scaling] {run_dir}: finding surprising recombinations (distance-2 atypical bridges)…",
          flush=True)
    for w in nodes:
        nb = list(adj[w])
        if len(nb) > HUB:
            nb = nb[:HUB]
        for ii in range(len(nb)):
            a = nb[ii]
            for jj in range(ii + 1, len(nb)):
                b = nb[jj]
                fs = frozenset((a, b))
                if fs in direct:
                    continue
                rr = max(node_it[a], node_it[b], adj[w][a], adj[w][b])
                if fs in realize:
                    if rr < realize[fs]:
                        realize[fs] = rr
                    continue
                if (float(X[idx[a]] @ X[idx[b]]) - mu) / sd < z_thr:   # atypical pair
                    realize[fs] = rr
        if len(realize) > CAP:
            break
    real_iters = np.array(sorted(realize.values())) if realize else np.array([])

    max_it = int(node_iter.max()) if len(node_iter) else 0
    ts = sorted(set(int(round(t)) for t in np.linspace(0, max_it, n_ckpt)))

    ideas, spread, reach, recomb = [], [], [], []
    for t in ts:
        m = node_iter <= t
        nt = int(m.sum())
        ideas.append(nt)
        if nt >= 2:
            Xt = X[m]
            spread.append(float(Xt.var(axis=0).sum()))
            reach.append(float(1.0 - float((Xt @ sv).min())))
        else:
            spread.append(0.0); reach.append(0.0)
        recomb.append(int(np.searchsorted(real_iters, t, side="right")) if real_iters.size else 0)

    # x-axis: cumulative tokens (real compute) or iteration
    tm = _token_map(run_dir)
    if x == "tokens" and tm is not None:
        xs = np.interp(ts, tm[0], tm[1])
        xlabel = "test-time compute (cumulative tokens)"
    else:
        xs = np.array(ts, dtype=float)
        xlabel = "reasoning iteration"

    return {"x": xs.tolist(), "xlabel": xlabel, "iters": ts,
            "ideas": ideas, "spread": spread, "reach": reach, "recomb": recomb,
            "n_recomb": len(realize), "z_thr": z_thr, "model": model,
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}


def make_figure(runs, labels, out, embed_model=None, n_ckpt=40, z_thr=-1.0, x="tokens"):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    curves = [scaling_curves(r, embed_model=embed_model, n_ckpt=n_ckpt, z_thr=z_thr, x=x)
              for r in runs]
    xlabel = curves[0]["xlabel"]

    fig, ax = plt.subplots(2, 2, figsize=(10.0, 7.6))
    for c, lab, col in zip(curves, labels, PALETTE):
        xs = c["x"]
        ax[0, 0].plot(xs, c["ideas"], color=col, lw=2, marker="o", ms=3, label=lab)
        ax[0, 1].plot(xs, c["spread"], color=col, lw=2, marker="o", ms=3, label=lab)
        ax[1, 0].plot(xs, c["reach"], color=col, lw=2, marker="o", ms=3, label=lab)
        ax[1, 1].plot(xs, c["recomb"], color=col, lw=2, marker="o", ms=3, label=lab)

    ax[0, 0].set_title("(a) Distinct ideas (fluency)"); ax[0, 0].set_ylabel("# ideas")
    ax[0, 1].set_title("(b) Idea-space expansion"); ax[0, 1].set_ylabel("embedding spread (total variance)")
    ax[1, 0].set_title("(c) Frontier reach"); ax[1, 0].set_ylabel("max distance from seed")
    ax[1, 1].set_title(f"(d) Surprising recombinations  (atypical pairs bridged, z < {z_thr})")
    ax[1, 1].set_ylabel("cumulative count")
    for a in ax.flat:
        a.set_xlabel(xlabel); a.grid(True, color="0.92", lw=0.5); a.set_axisbelow(True)
    ax[0, 0].legend(frameon=False, fontsize=9, ncol=min(3, len(runs)))
    fig.suptitle("Surprising-insight yield scales with test-time compute", y=1.0, fontsize=12.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out}_scaling.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}_scaling.png/.svg/.pdf")

    report = {lab: c for lab, c in zip(labels, curves)}
    with open(f"{out}_scaling.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}_scaling.json")
    for lab, c in zip(labels, curves):
        print(f"[scaling] {lab}: {c['n_nodes']} ideas, {c['recomb'][-1]} surprising recombinations "
              f"(atypical pairs bridged via a shared concept)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="a single run dir")
    p.add_argument("--runs", nargs="+", help="one or more run dirs (overlaid)")
    p.add_argument("--labels", nargs="+", help="legend labels (default: dir names)")
    p.add_argument("--out", default=None, help="figure basename (default: <first-run>/figures/ideation)")
    p.add_argument("--x", choices=["tokens", "iter"], default="tokens",
                   help="x-axis: cumulative tokens (test-time compute) or reasoning iteration")
    p.add_argument("--checkpoints", type=int, default=40, help="compute checkpoints along the run")
    p.add_argument("--z-thr", dest="z_thr", type=float, default=-1.0,
                   help="atypical-combination threshold (edge endpoint z below this = surprising)")
    p.add_argument("--embed-model", dest="embed_model", default=None,
                   help="override the sentence-transformers id used for re-embedding")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=None,
                   help="truncate every run to iter <= this for a fair (matched-compute) comparison")
    args = p.parse_args()

    if args.max_iter is not None:                      # shared cap: insights.load_graph applies it
        I.MAX_ITER = args.max_iter
        print(f"[scaling] truncating all runs to iter <= {args.max_iter}")

    runs = args.runs or ([args.run] if args.run else None)
    if not runs:
        raise SystemExit("provide --run <dir> or --runs <dir> [<dir> ...]")
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in runs]
    out = args.out or os.path.join(runs[0].rstrip("/"), "figures", "ideation")
    make_figure(runs, labels, out, embed_model=args.embed_model, n_ckpt=args.checkpoints,
                z_thr=args.z_thr, x=args.x)


if __name__ == "__main__":
    main()
