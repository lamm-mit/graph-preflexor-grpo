#!/usr/bin/env python
"""Quantify and visualize how *novel* an ideation run's concepts and mined insights are.

Built for publication-grade figures: every panel is a defensible, citeable method, and every
number is also dumped to `<out>_novelty.json` so it can be quoted in the text. Two images are
written so each can stand alone in a paper:
  * `<out>_novelty_map.{png,svg,pdf}`   — the wide concept-space map, panel (A).
  * `<out>_novelty_stats.{png,svg,pdf}` — panels (B), (C), (D) stacked, full-width.

Panels (map image: A, B; stats image: C, D, E)
-----------------------------------------------
(A) Concept-space map — UMAP (or PCA) of every concept. The seed is marked; the
    "established region" (earliest-introduced concepts) is shaded via a KDE contour; each
    concept is colored by its **novelty-when-introduced** (distance to the nearest concept
    that already existed when it appeared). New ideas land *outside* the known region. Kept
    label-free so the cloud stays readable — the names live in panel (B).
(B) Ideation dynamics — a horizontal bar of the most novel concepts (names on the y-axis, so
    they're always legible), bar length = novelty-when-introduced, colored by the iteration
    each first appeared: *which* ideas were the novel ones, and *when* they emerged.
(C) Novelty expands with reasoning — mean nearest-prior-neighbor novelty per iteration
    (the open-ended **novelty-search** metric, Lehman & Stanley 2011), with a bootstrap CI
    band. Pass several runs to overlay strategies (frontier vs novelty vs leap).
(D) Relational-motif significance — z-scores of relation-typed 2-step motifs against a
    relation-label-shuffled null (network-motif significance, Milo et al. Science 2002).
    Over-represented motifs (z>1.96) are the structural basis of the *analogy* insights.
    Community **modularity** z and edge **heterophily** z (degree/label-permutation nulls)
    are annotated — "structure beyond chance".
(E) Novel combinations — combination **typicality** z of each linked concept pair vs the
    global pairwise-similarity distribution (adapting Uzzi et al. Science 2013): unusually
    *dissimilar* pairs that are nonetheless connected are novel recombinations. The mined
    conceptual bridges skew into the atypical tail vs existing edges and random pairs
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


def _pbar(it, desc, total=None):
    """tqdm progress bar if available, else the plain iterable (no hard dependency)."""
    try:
        from tqdm import tqdm
        return tqdm(it, desc=desc, total=total, leave=False, dynamic_ncols=True)
    except Exception:
        return it


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def _intattr(G, n, key, default=0):
    try:
        return int(float(G.nodes[n].get(key, default)))
    except Exception:
        return default


def load_run(run_dir, embed_model=None):
    """Return (G, vecs, topic, seed_node, model). vecs is {node: unit-vec}; model is the
    sentence-transformers id actually used (for reproducibility)."""
    G = I.load_graph(run_dir)
    from graphstore import resolve_embed_model, embed_texts
    model = resolve_embed_model(run_dir, embed_model)
    nodes = list(G.nodes)
    print(f"[novelty] {run_dir}: embedding {len(nodes)} concepts with '{model}' (batched)…",
          flush=True)
    try:                                              # batched: one model load, batched forward
        V = embed_texts([str(G.nodes[n].get("label", n)) for n in nodes], model)
        vecs = {n: V[i] for i, n in enumerate(nodes)}
    except Exception as e:
        raise SystemExit(f"{run_dir}: embeddings unavailable with '{model}' ({e}) — novelty "
                         f"analysis needs them. Authenticate the gated model "
                         f"(huggingface-cli login) or pass --embed-model all-MiniLM-L6-v2.")
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
    return G, vecs, topic, seed, model


# --------------------------------------------------------------------------- #
#  Per-concept novelty
# --------------------------------------------------------------------------- #
def concept_novelty(G, vecs, seed):
    """For each concept: novelty_when_introduced = 1 - max cosine to concepts that existed at a
    STRICTLY earlier iteration; novelty_vs_seed = 1 - cosine to the seed. Chunked so it stays
    memory-safe on large graphs (never materializes the full n x n similarity matrix)."""
    nodes = list(vecs)
    X = np.stack([vecs[n] for n in nodes]).astype(np.float32)
    it = np.array([_intattr(G, n, "iter", 0) for n in nodes])
    dep = np.array([_intattr(G, n, "depth", 0) for n in nodes])
    sidx = nodes.index(seed)
    nov_seed = 1.0 - (X @ X[sidx])
    n = len(nodes)
    nov_intro = np.full(n, np.nan, dtype=np.float32)
    CH = 2048
    for s in _pbar(range(0, n, CH), "novelty", total=(n + CH - 1) // CH):  # (chunk x n) blocks
        e = min(n, s + CH)
        Sb = X[s:e] @ X.T
        for k in range(s, e):
            prior = it < it[k]
            if prior.any():
                nov_intro[k] = 1.0 - float(Sb[k - s][prior].max())
    return {nodes[i]: {"novelty_intro": float(nov_intro[i]), "iter": int(it[i]),
                       "depth": int(dep[i]), "novelty_seed": float(nov_seed[i])}
            for i in range(n)}


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
    for s in _pbar(range(n_null), "modularity null", total=n_null):
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
    X = np.stack([vecs[n] for n in nodes]).astype(np.float32)
    idx = {n: i for i, n in enumerate(nodes)}
    ei = np.array([idx[u] for u, v in edges])
    ej = np.array([idx[v] for u, v in edges])
    obs = float(np.einsum("ij,ij->i", X[ei], X[ej]).mean())
    rng = np.random.default_rng(0)
    n = len(nodes)
    nulls = np.empty(n_null)
    for t in _pbar(range(n_null), "heterophily null", total=n_null):   # vectorized permutation null
        Xp = X[rng.permutation(n)]
        nulls[t] = np.einsum("ij,ij->i", Xp[ei], Xp[ej]).mean()
    mu, sd = float(nulls.mean()), float(nulls.std() or 1.0)
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
    for _ in _pbar(range(n_null), "motif null", total=n_null):
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
def global_cosine_stats(vecs):
    """EXACT mean/std of all i<j pairwise cosines, computed by chunked streaming (no full n x n
    matrix, no sampling) — memory-safe and numerically identical to the brute-force statistic."""
    nodes = list(vecs)
    X = np.stack([vecs[n] for n in nodes]).astype(np.float32)
    n = len(nodes)
    ssum = ssq = 0.0
    cnt = 0
    CH = 2048
    for s in _pbar(range(0, n, CH), "pairwise stats", total=(n + CH - 1) // CH):
        e = min(n, s + CH)
        B = X[s:e] @ X.T                              # (chunk x n)
        for r in range(s, e):
            row = B[r - s, r + 1:]                    # only j > r → exact upper triangle
            ssum += float(row.sum()); ssq += float((row * row).sum()); cnt += row.size
    mu = ssum / cnt if cnt else 0.0
    sd = (max(ssq / cnt - mu * mu, 0.0) ** 0.5) if cnt else 0.0
    return nodes, X, mu, (sd or 1.0)


def combination_typicality(vecs, pairs, stats=None):
    """z of each pair's cosine vs the EXACT global pairwise-cosine distribution. Lower (more
    negative) = a more atypical / novel combination. Pass `stats` from global_cosine_stats() to
    reuse the (single) O(n^2) pass across several pair sets."""
    nodes, X, mu, sd = stats if stats is not None else global_cosine_stats(vecs)
    idx = {nm: k for k, nm in enumerate(nodes)}
    return [(float(X[idx[a]] @ X[idx[b]]) - mu) / sd
            for a, b in pairs if a in idx and b in idx and a != b]


def _bridge_pairs(run_dir, G, vecs, top=12):
    """Endpoint pairs of the *conceptual bridges* — the long-range connections that span distant
    regions of the idea space (the system's genuinely novel combinations). From insights.json if
    present, else computed fast: the top-K most embedding-distant pairs that are still connected."""
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
    print("[novelty] note: no insights.json found — Panel D uses an on-the-fly bridge "
          "approximation (most-distant connected peripheral pairs). Run `insights.py` first to "
          "use the canonical mined conceptual bridges.", flush=True)
    # bound cost on large graphs: distant bridges live among the most peripheral concepts, so
    # restrict the all-pairs search to the M least-central nodes (never the full n x n).
    nodes = list(vecs)
    X = np.stack([vecs[n] for n in nodes])
    centroid = X.mean(0)
    M = min(len(nodes), 400)
    periph = np.argsort(X @ centroid)[:M]             # least central first
    sub = [nodes[i] for i in periph]
    Xs = X[periph]
    Ss = Xs @ Xs.T
    iu = np.triu_indices(M, 1)
    order = np.argsort(Ss[iu])                         # ascending cosine = most distant first
    U = G.to_undirected()
    out = []
    for idx in order[: top * 50]:
        a, b = sub[iu[0][idx]], sub[iu[1][idx]]
        if nx.has_path(U, a, b):
            out.append((a, b))
        if len(out) >= top:
            break
    return out


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
    G0, vecs0, topic0, seed0, model0 = loaded[0]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # ================= Figure 1: concept-space map (wide) ================= #
    print("[novelty] panel A: projecting concept space (UMAP/PCA)…", flush=True)
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

    import matplotlib as mpl
    figM = plt.figure(figsize=(13.0, 11.4))
    gsM = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1.0], hspace=0.30)
    axA = figM.add_subplot(gsM[0])
    axN = figM.add_subplot(gsM[1])

    # ---- (A) PCA/UMAP concept-space map (kept clean — no in-cloud labels) ----
    try:                                              # shade the "established region"
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
    cb = figM.colorbar(sc, ax=axA, fraction=0.038, pad=0.015)
    cb.set_label("novelty when introduced\n(1 − cosine to nearest prior concept)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    xr = P[:, 0].max() - P[:, 0].min(); yr = P[:, 1].max() - P[:, 1].min()
    axA.set_xlim(P[:, 0].min() - 0.04 * xr, P[:, 0].max() + 0.06 * xr)
    axA.set_ylim(P[:, 1].min() - 0.05 * yr, P[:, 1].max() + 0.06 * yr)
    axA.set_title(f"(A) Concept space — {labels[0]}  ({G0.number_of_nodes()} ideas, {proj})"
                  + (f"  ·  topic: {topic0}" if topic0 else "")
                  + f"  ·  embeddings: {model0}", fontsize=12)
    axA.set_xlabel(f"{proj}-1"); axA.set_ylabel(f"{proj}-2")
    axA.legend(loc="upper left", frameon=False, fontsize=9,
               title="★ seed · shaded = established region", title_fontsize=8)

    # ---- (B) ideation dynamics: the most novel concepts, and when they emerged ----
    valid = [i for i in range(len(nodes))
             if cn0[nodes[i]]["novelty_intro"] == cn0[nodes[i]]["novelty_intro"]]  # not nan
    K = min(18, len(valid))
    order = sorted(valid, key=lambda i: nov[i], reverse=True)[:K][::-1]  # highest at top
    sel_names = [I.lbl(G0, nodes[i], 46) for i in order]
    sel_val = [float(nov[i]) for i in order]
    sel_it = [int(its[i]) for i in order]
    cmap2 = plt.cm.plasma
    vmin, vmax = float(its.min()), float(its.max())
    norm2 = mpl.colors.Normalize(vmin=vmin, vmax=(vmax if vmax > vmin else vmin + 1))
    axN.barh(range(K), sel_val, color=[cmap2(norm2(t)) for t in sel_it], zorder=2)
    axN.set_yticks(range(K)); axN.set_yticklabels(sel_names, fontsize=8)
    axN.set_ylim(-0.6, K - 0.4)
    axN.set_xlabel("novelty when introduced  (1 − cosine to nearest prior concept)")
    axN.set_title(f"(B) Ideation dynamics — the {K} most novel concepts and when they emerged")
    axN.grid(True, axis="x", color="0.92", lw=0.5); axN.set_axisbelow(True)
    sm = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2); sm.set_array([])
    cbn = figM.colorbar(sm, ax=axN, fraction=0.030, pad=0.015)
    cbn.set_label("iteration introduced\n(reasoning order)", fontsize=9)
    cbn.ax.tick_params(labelsize=8)

    for ext in ("png", "svg", "pdf"):
        figM.savefig(f"{out}_novelty_map.{ext}", bbox_inches="tight")
    plt.close(figM)
    print(f"wrote {out}_novelty_map.png/.svg/.pdf")

    # ============ Figure 2: statistics — B, C, D stacked (wide) =========== #
    figS = plt.figure(figsize=(11.0, 12.8))
    gsS = gridspec.GridSpec(3, 1, hspace=0.42)
    axB = figS.add_subplot(gsS[0])
    axC = figS.add_subplot(gsS[1])
    axD = figS.add_subplot(gsS[2])

    # ---------- (B) novelty expands with reasoning ------------------------- #
    print("[novelty] panel B: novelty trajectory per run…", flush=True)
    for (G, vecs, _, seed, _m), lab, col in zip(loaded, labels, PAL):
        cn = concept_novelty(G, vecs, seed)
        its_b, mean_b, sem_b = novelty_trajectory(cn)
        if not its_b:
            continue
        mean_b, sem_b = np.array(mean_b), np.array(sem_b)
        axB.plot(its_b, mean_b, color=col, lw=1.6, marker="o", ms=2.5, label=lab)
        axB.fill_between(its_b, mean_b - sem_b, mean_b + sem_b, color=col, alpha=0.18, lw=0)
        report["trajectories"][lab] = {"iter": list(its_b), "mean_novelty": [float(x) for x in mean_b]}
    axB.set_title("(C) Novelty expands with reasoning")
    axB.set_xlabel("reasoning iteration"); axB.set_ylabel("nearest-prior novelty")
    axB.grid(True, color="0.92", lw=0.5); axB.set_axisbelow(True)
    if len(loaded) > 1:
        axB.legend(frameon=False, fontsize=8, ncol=min(4, len(loaded)))

    # ---------- (C) relational-motif significance -------------------------- #
    # auto-scale the two heaviest nulls down on big graphs (greedy modularity / motif recount
    # are superlinear); tight on small graphs where it's cheap.
    nE, nN = G0.number_of_edges(), G0.number_of_nodes()
    nn_motif = n_null if nE < 3000 else max(50, n_null // 4)
    nn_mod = (max(50, n_null // 2) if nN < 1500 else max(20, n_null // 8))
    print(f"[novelty] panel C: null-model significance "
          f"(motifs×{nn_motif}, modularity×{nn_mod}, heterophily)… this is the slow part",
          flush=True)
    motifs = motif_significance(G0, n_null=nn_motif)
    print("[novelty]   … motifs done; modularity null (rewiring)…", flush=True)
    mod = modularity_significance(G0, n_null=nn_mod)
    print("[novelty]   … modularity done; heterophily null…", flush=True)
    het = heterophily_significance(G0, vecs0, n_null=max(200, n_null))
    if motifs:
        ys = range(len(motifs))[::-1]
        zs = [m["z"] for m in motifs][::-1]
        names = [f"{a}·{b}" for (a, b) in [m["signature"] for m in motifs]][::-1]
        cols = ["#2ca02c" if z > 1.96 else "0.6" for z in zs]
        axC.barh(list(ys), zs, color=cols)
        axC.set_yticks(list(ys)); axC.set_yticklabels([n[:26] for n in names], fontsize=8)
        axC.axvline(1.96, color="#d62728", ls="--", lw=1)
        axC.set_xlabel("motif z-score (vs label-shuffled null)")
        axC.set_xlim(0, max(zs) * 1.28)               # headroom so the stats box clears the bars
    axC.set_title("(D) Relational-motif significance")
    txt = []
    if mod.get("z") is not None:
        txt.append(f"modularity Q={mod['Q']:.2f}  z={mod['z']:+.1f}")
    if het.get("z") is not None:
        txt.append(f"edge heterophily z={het['z']:+.1f}")
    if txt:                                            # top-right, where the (shortest) bars leave space
        axC.text(0.985, 0.94, "\n".join(txt), transform=axC.transAxes, ha="right", va="top",
                 fontsize=8.5, family="monospace",
                 bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.95))
    axC.grid(True, axis="x", color="0.92", lw=0.5); axC.set_axisbelow(True)

    # ---------- (D) novel combinations (Uzzi-style) ------------------------ #
    print("[novelty] panel D: combination typicality (bridges vs edges vs random)…", flush=True)
    # Knowledge-graph edges are normally homophilic (link similar concepts); the system's
    # novelty lives in the long-range conceptual bridges that connect atypically dissimilar
    # ideas. Show all three distributions; test bridges vs edges.
    U0 = _simple_undirected(G0)
    stats0 = global_cosine_stats(vecs0)               # one exact O(n^2) pass, reused below
    edge_z = combination_typicality(vecs0, list(U0.edges()), stats=stats0)
    bridge_pairs = _bridge_pairs(runs[0], G0, vecs0, top=top)
    bridge_z = combination_typicality(vecs0, bridge_pairs, stats=stats0)
    rand_z = combination_typicality(vecs0, _random_pairs(G0, max(200, len(edge_z))), stats=stats0)
    series = [("random pairs", rand_z, "0.6"), ("existing edges", edge_z, "#1f77b4"),
              ("conceptual bridges", bridge_z, "#d62728")]
    series = [(nm, z, c) for nm, z, c in series if len(z) > 1]
    if series:
        lo = min(min(z) for _, z, _ in series); hi = max(max(z) for _, z, _ in series)
        bins = np.linspace(lo, hi, 30)
        for nm, z, c in series:
            axD.hist(z, bins=bins, density=True, histtype="step", lw=2, color=c, label=nm)
            axD.axvline(np.median(z), color=c, ls="--", lw=1)
    axD.set_title("(E) Novel combinations (typicality)")
    axD.set_xlabel("combination z  (← more novel / atypical)"); axD.set_ylabel("density")
    axD.legend(frameon=False, fontsize=8, loc="upper right")
    axD.grid(True, color="0.92", lw=0.5); axD.set_axisbelow(True)
    mw = {}
    try:
        from scipy.stats import mannwhitneyu
        if len(bridge_z) > 1 and len(edge_z) > 1:
            u, p = mannwhitneyu(bridge_z, edge_z, alternative="less")
            mw = {"U": float(u), "p_bridges_more_novel_than_edges": float(p)}
            axD.text(0.985, 0.62, f"bridges vs edges\nMann–Whitney p={p:.1e}",
                     transform=axD.transAxes, va="top", ha="right", fontsize=8,
                     family="monospace",
                     bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.95))
    except Exception:
        pass

    figS.suptitle(f"Novelty statistics — {labels[0]}"
                  + (f"  (topic: {topic0})" if topic0 else ""), y=0.995, fontsize=12)
    for ext in ("png", "svg", "pdf"):
        figS.savefig(f"{out}_novelty_stats.{ext}", bbox_inches="tight")
    plt.close(figS)
    print(f"wrote {out}_novelty_stats.png/.svg/.pdf")

    # ---------- numeric report -------------------------------------------- #
    valid = [d["novelty_intro"] for d in cn0.values() if d["novelty_intro"] == d["novelty_intro"]]
    report["runs"][labels[0]] = {
        "embed_model": model0,
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
    p.add_argument("--max-iter", dest="max_iter", type=int, default=None,
                   help="truncate every run to iter <= this for fair cross-run figures")
    args = p.parse_args()

    if args.max_iter is not None:                      # shared cap: insights.load_graph applies it
        I.MAX_ITER = args.max_iter
        print(f"[novelty] truncating all runs to iter <= {args.max_iter}")

    runs = args.runs or ([args.run] if args.run else None)
    if not runs:
        raise SystemExit("provide --run <dir> or --runs <dir> [<dir> ...]")
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in runs]
    out = args.out or os.path.join(runs[0].rstrip("/"), "figures", "ideation")
    make_figure(runs, labels, out, n_null=args.n_null, embed_model=args.embed_model, top=args.top)


if __name__ == "__main__":
    main()
