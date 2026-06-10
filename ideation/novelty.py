#!/usr/bin/env python
"""Quantify and visualize how *novel* an ideation run's concepts and mined insights are.

Built for a publication-grade figure: every panel is a defensible, citeable method, and
every number is also dumped to `<out>_novelty.json` so it can be quoted in the text.

Panels
------
(A) Concept-space map — UMAP (or PCA) of every concept. The seed is marked; the
    "established region" (earliest-introduced concepts) is shaded via a KDE contour; each
    concept is colored by its **novelty-when-introduced** (distance to the nearest concept
    that already existed when it appeared). New ideas land *outside* the known region.
(B) Novelty expands with reasoning — mean nearest-prior-neighbor novelty per iteration
    (the open-ended **novelty-search** metric, Lehman & Stanley 2011), with a bootstrap CI
    band. Pass several runs to overlay strategies (frontier vs novelty vs leap).
(C) Relational-motif significance — z-scores of relation-typed 2-step motifs against a
    relation-label-shuffled null (network-motif significance, Milo et al. Science 2002).
    Over-represented motifs (z>1.96) are the structural basis of the *analogy* insights.
    Community **modularity** z and edge **heterophily** z (degree/label-permutation nulls)
    are annotated — "structure beyond chance".
(D) Novel combinations — combination **typicality** z of each linked concept pair vs the
    global pairwise-similarity distribution (adapting Uzzi et al. Science 2013): unusually
    *dissimilar* pairs that are nonetheless connected are novel recombinations. The mined
    latent links skew into the atypical tail vs existing edges and random pairs
    (Mann-Whitney p reported).

Usage
-----
    python novelty.py --run runs/exp2 --out runs/exp2/figures/novelty
    python novelty.py --runs runs/exp2 runs/exp_novelty runs/exp_leap \
        --labels frontier novelty leap --out figures/novelty_compare
    python novelty.py --run runs/exp2 --n-null 500          # tighter null distributions

Needs embeddings (the whole point); structural-only runs are not enough. Reuses the run's
recorded embed model unless overridden with --embed-model.
"""
import argparse
import json
import os

import numpy as np
import networkx as nx

import insights as I


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def _intattr(G, n, key, default=0):
    try:
        return int(float(G.nodes[n].get(key, default)))
    except Exception:
        return default


def load_run(run_dir, embed_model=None):
    """Return (G, vecs, topic, seed_node). vecs is {node: unit-vec}."""
    G = I.load_graph(run_dir)
    from graphstore import resolve_embed_model
    model = resolve_embed_model(run_dir, embed_model)
    vecs = I.embed_nodes(G, model)
    if not vecs:
        raise SystemExit(f"{run_dir}: embeddings unavailable — novelty analysis needs them "
                         f"(install sentence-transformers / set --embed-model).")
    topic = ""
    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        try:
            topic = json.load(open(sp)).get("topic", "")
        except Exception:
            pass
    # seed = the depth-0 / lowest-iter node (falls back to the topic-labelled node)
    seed = min(G.nodes, key=lambda n: (_intattr(G, n, "depth", 0), _intattr(G, n, "iter", 0)))
    if topic:
        for n in G.nodes:
            if str(G.nodes[n].get("label", n)).strip().lower() == topic.strip().lower():
                seed = n
                break
    return G, vecs, topic, seed


# --------------------------------------------------------------------------- #
#  Per-concept novelty
# --------------------------------------------------------------------------- #
def concept_novelty(G, vecs, seed):
    """For each concept: novelty_when_introduced = 1 - max cosine to concepts that existed at a
    STRICTLY earlier iteration; novelty_vs_seed = 1 - cosine to the seed. Returns dict-of-dicts."""
    nodes = list(vecs)
    it = {n: _intattr(G, n, "iter", 0) for n in nodes}
    dep = {n: _intattr(G, n, "depth", 0) for n in nodes}
    out = {}
    for n in nodes:
        prior = [m for m in nodes if it[m] < it[n]]
        if prior:
            nv = 1.0 - max(float(np.dot(vecs[n], vecs[m])) for m in prior)
        else:
            nv = float("nan")                         # nothing existed before it (seed batch)
        out[n] = {"novelty_intro": nv, "iter": it[n], "depth": dep[n],
                  "novelty_seed": 1.0 - float(np.dot(vecs[n], vecs[seed]))}
    return out


def novelty_trajectory(cn):
    """Mean (+ sem) novelty-when-introduced per iteration, over concepts born that iteration."""
    by_it = {}
    for n, d in cn.items():
        if d["novelty_intro"] == d["novelty_intro"]:     # not nan
            by_it.setdefault(d["iter"], []).append(d["novelty_intro"])
    its = sorted(by_it)
    mean = [float(np.mean(by_it[i])) for i in its]
    sem = [float(np.std(by_it[i]) / max(1, np.sqrt(len(by_it[i])))) for i in its]
    return its, mean, sem


# --------------------------------------------------------------------------- #
#  Null-model significance ("structure beyond chance")
# --------------------------------------------------------------------------- #
def _simple_undirected(G):
    U = nx.Graph()
    U.add_nodes_from(G.nodes())
    for u, v in G.to_undirected().edges():
        if u != v:
            U.add_edge(u, v)
    return U


def _emp_p(obs, null, greater=True):
    null = np.asarray(null, float)
    if null.size == 0:
        return float("nan")
    k = np.sum(null >= obs) if greater else np.sum(null <= obs)
    return float((1 + k) / (1 + null.size))


def modularity_significance(G, n_null=100):
    """Observed community modularity vs a degree-preserving rewired null (double-edge-swap)."""
    U = _simple_undirected(G)
    if U.number_of_edges() < 5:
        return {}
    comms = nx.community.greedy_modularity_communities(U)
    Q = float(nx.community.modularity(U, comms))
    m = U.number_of_edges()
    nulls = []
    for s in range(n_null):
        H = U.copy()
        try:
            nx.double_edge_swap(H, nswap=2 * m, max_tries=30 * m, seed=s)
        except Exception:
            continue
        c = nx.community.greedy_modularity_communities(H)
        nulls.append(float(nx.community.modularity(H, c)))
    if not nulls:
        return {"Q": Q}
    mu, sd = float(np.mean(nulls)), float(np.std(nulls) or 1.0)
    return {"Q": Q, "null_mean": mu, "null_std": sd, "z": (Q - mu) / sd,
            "p": _emp_p(Q, nulls, greater=True), "n_null": len(nulls),
            "n_communities": len(comms)}


def heterophily_significance(G, vecs, n_null=500):
    """Mean cosine of LINKED concept pairs vs a node-label-permutation null. z<0 => the model
    links more *dissimilar* concepts than chance (creative/heterophilic bridging)."""
    U = _simple_undirected(G)
    edges = list(U.edges())
    if not edges:
        return {}
    nodes = list(vecs)
    X = np.stack([vecs[n] for n in nodes])
    idx = {n: i for i, n in enumerate(nodes)}
    obs = float(np.mean([float(np.dot(X[idx[u]], X[idx[v]])) for u, v in edges]))
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(n_null):
        perm = rng.permutation(len(nodes))
        pos = {nodes[i]: X[perm[i]] for i in range(len(nodes))}
        nulls.append(float(np.mean([float(np.dot(pos[u], pos[v])) for u, v in edges])))
    mu, sd = float(np.mean(nulls)), float(np.std(nulls) or 1.0)
    return {"mean_edge_cosine": obs, "null_mean": mu, "null_std": sd, "z": (obs - mu) / sd,
            "p_two_sided": 2 * min(_emp_p(obs, nulls, True), _emp_p(obs, nulls, False)),
            "n_null": n_null}


def motif_significance(G, n_null=200, top=10):
    """Relation-typed 2-step motifs (A--r1-->B--r2-->C) vs a relation-label-shuffled null.
    Returns per-signature {obs, z, p}; over-represented (z>1.96) motifs underpin the analogies."""
    def count(g):
        sig = {}
        for b in g.nodes:
            for a in g.predecessors(b):
                r1 = g[a][b].get("relation", "related_to")
                for c in g.successors(b):
                    if c == a:
                        continue
                    r2 = g[b][c].get("relation", "related_to")
                    sig[(r1, r2)] = sig.get((r1, r2), 0) + 1
        return sig

    obs = count(G)
    if not obs:
        return []
    edges = list(G.edges())
    rels = [G[u][v].get("relation", "related_to") for u, v in edges]
    nulls = {k: [] for k in obs}
    rng = np.random.default_rng(0)
    for _ in range(n_null):
        perm = rng.permutation(len(rels))
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        for (u, v), pi in zip(edges, perm):
            H.add_edge(u, v, relation=rels[pi])
        c = count(H)
        for k in obs:
            nulls[k].append(c.get(k, 0))
    rows = []
    for k, o in obs.items():
        arr = np.asarray(nulls[k], float)
        mu, sd = float(arr.mean()), float(arr.std() or 1.0)
        rows.append({"signature": list(k), "obs": int(o), "null_mean": mu,
                     "z": (o - mu) / sd, "p": _emp_p(o, arr, greater=True)})
    rows.sort(key=lambda r: r["z"], reverse=True)
    return rows[:top]


# --------------------------------------------------------------------------- #
#  Combination novelty (Uzzi-style)
# --------------------------------------------------------------------------- #
def combination_typicality(vecs, pairs):
    """z of each pair's cosine vs the global pairwise-cosine distribution. Lower (more
    negative) = a more atypical / novel combination of concepts."""
    nodes = list(vecs)
    X = np.stack([vecs[n] for n in nodes])
    S = X @ X.T
    iu = np.triu_indices(len(X), 1)
    mu, sd = float(S[iu].mean()), float(S[iu].std() or 1.0)
    idx = {n: i for i, n in enumerate(nodes)}
    out = []
    for a, b in pairs:
        if a in idx and b in idx and a != b:
            out.append((float(np.dot(X[idx[a]], X[idx[b]])) - mu) / sd)
    return out


def _bridge_pairs(run_dir, G, vecs, top=12):
    """Endpoint pairs of the mined *conceptual bridges* — the long-range connections that span
    distant regions of the idea space (the system's genuinely novel combinations). From
    insights.json if present, else computed on the fly."""
    jp = os.path.join(run_dir, "insights.json")
    if os.path.exists(jp):
        try:
            miners = json.load(open(jp)).get("miners", {})
            pairs = [tuple(x["endpoints"]) for x in miners.get("conceptual_bridge", [])
                     if "endpoints" in x]
            if pairs:
                return pairs
        except Exception:
            pass
    return [tuple(x["endpoints"]) for x in I.conceptual_bridges(G, vecs, top=top)
            if "endpoints" in x]


def _random_pairs(G, n, seed=0):
    nodes = list(G.nodes)
    U = G.to_undirected()
    eset = {frozenset(e) for e in U.edges}
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    while len(out) < n and tries < 50 * n + 1000:
        i, j = rng.integers(0, len(nodes), 2)
        tries += 1
        if i != j and frozenset((nodes[i], nodes[j])) not in eset:
            out.append((nodes[i], nodes[j]))
    return out


# --------------------------------------------------------------------------- #
#  Figure
# --------------------------------------------------------------------------- #
def _project(X):
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X), "UMAP"
    except Exception:
        Xc = X - X.mean(0)
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            return Xc @ Vt[:2].T, "PCA"
        except Exception:
            return Xc[:, :2], "PCA"


def make_figure(runs, labels, out, n_null=200, embed_model=None, top=12):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    PAL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    report = {"runs": {}, "trajectories": {}}

    # load every run (for the trajectory overlay); panels A/C/D use the FIRST (primary) run
    loaded = [load_run(r, embed_model) for r in runs]
    G0, vecs0, topic0, seed0 = loaded[0]

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.32, 1.0], height_ratios=[1, 1, 1],
                           wspace=0.24, hspace=0.5)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[2, 1])

    # ---------- (A) concept-space map -------------------------------------- #
    nodes = list(vecs0)
    X = np.stack([vecs0[n] for n in nodes])
    P, proj = _project(X)
    pos = {n: P[i] for i, n in enumerate(nodes)}
    cn0 = concept_novelty(G0, vecs0, seed0)
    nov = np.array([cn0[n]["novelty_intro"] if cn0[n]["novelty_intro"] == cn0[n]["novelty_intro"]
                    else 0.0 for n in nodes])
    its = np.array([cn0[n]["iter"] for n in nodes])
    try:
        pr = nx.pagerank(G0) if G0.number_of_edges() else {n: 1.0 for n in G0}
    except Exception:
        pr = {n: 1.0 / max(1, len(nodes)) for n in nodes}
    # shade the "established region": KDE of the earliest tercile of concepts
    try:
        from scipy.stats import gaussian_kde
        early = P[its <= np.quantile(its, 0.34)]
        if len(early) > 5:
            kde = gaussian_kde(early.T)
            xg = np.linspace(P[:, 0].min(), P[:, 0].max(), 120)
            yg = np.linspace(P[:, 1].min(), P[:, 1].max(), 120)
            XX, YY = np.meshgrid(xg, yg)
            ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            axA.contourf(XX, YY, ZZ, levels=6, cmap="Greys", alpha=0.35, zorder=0)
            axA.contour(XX, YY, ZZ, levels=[ZZ.max() * 0.2], colors="0.5", linewidths=0.8, zorder=1)
    except Exception:
        pass
    for u, v in G0.to_undirected().edges():
        axA.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="0.85", lw=0.4, zorder=1)
    sc = axA.scatter(P[:, 0], P[:, 1], c=nov, cmap="viridis", s=[18 + 2600 * pr[n] for n in nodes],
                     alpha=0.85, linewidths=0, zorder=3)
    axA.scatter([pos[seed0][0]], [pos[seed0][1]], marker="*", s=320, color="#d62728",
                edgecolor="white", linewidths=1.0, zorder=5, label="seed")
    for n in sorted(nodes, key=lambda n: nov[nodes.index(n)], reverse=True)[:8]:
        axA.annotate(I.lbl(G0, n, 22), pos[n], fontsize=6.6, zorder=6,
                     xytext=(3, 3), textcoords="offset points")
    cb = fig.colorbar(sc, ax=axA, fraction=0.045, pad=0.02)
    cb.set_label("novelty when introduced\n(1 − cosine to nearest prior concept)", fontsize=8)
    axA.set_title(f"(A) Concept space — {labels[0]}  "
                  f"({G0.number_of_nodes()} ideas, {proj}); ★ seed, shaded = established region")
    axA.set_xlabel(f"{proj}-1"); axA.set_ylabel(f"{proj}-2")
    axA.legend(loc="upper right", frameon=False, fontsize=8)

    # ---------- (B) novelty expands with reasoning ------------------------- #
    for (G, vecs, _, seed), lab, col in zip(loaded, labels, PAL):
        cn = concept_novelty(G, vecs, seed)
        its_b, mean_b, sem_b = novelty_trajectory(cn)
        if not its_b:
            continue
        mean_b, sem_b = np.array(mean_b), np.array(sem_b)
        axB.plot(its_b, mean_b, color=col, lw=2, marker="o", ms=3, label=lab)
        axB.fill_between(its_b, mean_b - sem_b, mean_b + sem_b, color=col, alpha=0.18, lw=0)
        report["trajectories"][lab] = {"iter": list(its_b), "mean_novelty": [float(x) for x in mean_b]}
    axB.set_title("(B) Novelty expands with reasoning")
    axB.set_xlabel("reasoning iteration"); axB.set_ylabel("nearest-prior\nnovelty")
    axB.grid(True, color="0.92", lw=0.5); axB.set_axisbelow(True)
    if len(loaded) > 1:
        axB.legend(frameon=False, fontsize=7.5, ncol=min(3, len(loaded)))

    # ---------- (C) relational-motif significance -------------------------- #
    motifs = motif_significance(G0, n_null=n_null)
    mod = modularity_significance(G0, n_null=max(50, n_null // 2))
    het = heterophily_significance(G0, vecs0, n_null=max(200, n_null))
    if motifs:
        ys = range(len(motifs))[::-1]
        zs = [m["z"] for m in motifs][::-1]
        names = [f"{a}·{b}" for (a, b) in [m["signature"] for m in motifs]][::-1]
        cols = ["#2ca02c" if z > 1.96 else "0.6" for z in zs]
        axC.barh(list(ys), zs, color=cols)
        axC.set_yticks(list(ys)); axC.set_yticklabels([n[:20] for n in names], fontsize=6.6)
        axC.axvline(1.96, color="#d62728", ls="--", lw=1)
        axC.set_xlabel("motif z-score (vs label-shuffled null)")
    axC.set_title("(C) Relational-motif significance")
    txt = []
    if mod.get("z") is not None:
        txt.append(f"modularity Q={mod['Q']:.2f}  z={mod['z']:+.1f}")
    if het.get("z") is not None:
        txt.append(f"edge heterophily z={het['z']:+.1f}")
    if txt:
        axC.text(0.98, 0.04, "\n".join(txt), transform=axC.transAxes, ha="right", va="bottom",
                 fontsize=7.2, family="monospace",
                 bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    axC.grid(True, axis="x", color="0.92", lw=0.5); axC.set_axisbelow(True)

    # ---------- (D) novel combinations (Uzzi-style) ------------------------ #
    # Knowledge-graph edges are normally homophilic (link similar concepts); the system's
    # novelty lives in the long-range conceptual bridges that connect atypically dissimilar
    # ideas. Show all three distributions; test bridges vs edges.
    U0 = _simple_undirected(G0)
    edge_z = combination_typicality(vecs0, list(U0.edges()))
    bridge_pairs = _bridge_pairs(runs[0], G0, vecs0, top=top)
    bridge_z = combination_typicality(vecs0, bridge_pairs)
    rand_z = combination_typicality(vecs0, _random_pairs(G0, max(200, len(edge_z))))
    series = [("random pairs", rand_z, "0.6"), ("existing edges", edge_z, "#1f77b4"),
              ("conceptual bridges", bridge_z, "#d62728")]
    series = [(nm, z, c) for nm, z, c in series if len(z) > 1]
    if series:
        lo = min(min(z) for _, z, _ in series); hi = max(max(z) for _, z, _ in series)
        bins = np.linspace(lo, hi, 24)
        for nm, z, c in series:
            axD.hist(z, bins=bins, density=True, histtype="step", lw=2, color=c, label=nm)
            axD.axvline(np.median(z), color=c, ls="--", lw=1)
    axD.set_title("(D) Novel combinations (typicality)")
    axD.set_xlabel("combination z  (← more novel / atypical)"); axD.set_ylabel("density")
    axD.legend(frameon=False, fontsize=7)
    axD.grid(True, color="0.92", lw=0.5); axD.set_axisbelow(True)
    mw = {}
    try:
        from scipy.stats import mannwhitneyu
        if len(bridge_z) > 1 and len(edge_z) > 1:
            u, p = mannwhitneyu(bridge_z, edge_z, alternative="less")
            mw = {"U": float(u), "p_bridges_more_novel_than_edges": float(p)}
            axD.text(0.02, 0.95, f"bridges vs edges\nMann–Whitney p={p:.1e}", transform=axD.transAxes,
                     va="top", ha="left", fontsize=7, family="monospace")
    except Exception:
        pass

    fig.suptitle(f"Novelty of generated concepts & mined insights — {labels[0]}"
                 + (f"  (topic: {topic0})" if topic0 else ""), y=0.995, fontsize=12)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out}_novelty.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}_novelty.png/.svg/.pdf")

    # ---------- numeric report -------------------------------------------- #
    valid = [d["novelty_intro"] for d in cn0.values() if d["novelty_intro"] == d["novelty_intro"]]
    report["runs"][labels[0]] = {
        "n_concepts": G0.number_of_nodes(), "n_links": G0.number_of_edges(), "projection": proj,
        "mean_novelty_when_introduced": float(np.mean(valid)) if valid else None,
        "mean_novelty_vs_seed": float(np.mean([d["novelty_seed"] for d in cn0.values()])),
        "modularity": mod, "heterophily": het,
        "top_motifs": motifs,
        "combination_typicality_median": {
            "existing_edges": float(np.median(edge_z)) if edge_z else None,
            "conceptual_bridges": float(np.median(bridge_z)) if bridge_z else None,
            "random_pairs": float(np.median(rand_z)) if rand_z else None},
        "bridges_vs_edges_mannwhitney": mw,
    }
    with open(f"{out}_novelty.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}_novelty.json")
    # headline lines for the console
    r = report["runs"][labels[0]]
    print(f"[novelty] mean novelty-when-introduced = {r['mean_novelty_when_introduced']}")
    if mod.get("z") is not None:
        print(f"[novelty] modularity z = {mod['z']:+.2f} (p={mod.get('p')})")
    if het.get("z") is not None:
        print(f"[novelty] edge-heterophily z = {het['z']:+.2f}")
    if mw:
        print(f"[novelty] conceptual bridges more atypical than edges: "
              f"p = {mw['p_bridges_more_novel_than_edges']:.2e}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="a single run dir")
    p.add_argument("--runs", nargs="+", help="one or more run dirs (first = primary for A/C/D; "
                                             "all overlaid in B)")
    p.add_argument("--labels", nargs="+", help="legend labels (default: dir names)")
    p.add_argument("--out", default=None, help="figure basename (default: <first-run>/figures/ideation)")
    p.add_argument("--n-null", dest="n_null", type=int, default=200,
                   help="null-model resamples for the significance tests (higher = tighter)")
    p.add_argument("--top", type=int, default=12, help="mined latent links to test in (D)")
    p.add_argument("--embed-model", dest="embed_model", default=None,
                   help="override the sentence-transformers id used for re-embedding")
    args = p.parse_args()

    runs = args.runs or ([args.run] if args.run else None)
    if not runs:
        raise SystemExit("provide --run <dir> or --runs <dir> [<dir> ...]")
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in runs]
    out = args.out or os.path.join(runs[0].rstrip("/"), "figures", "ideation")
    make_figure(runs, labels, out, n_null=args.n_null, embed_model=args.embed_model, top=args.top)


if __name__ == "__main__":
    main()
