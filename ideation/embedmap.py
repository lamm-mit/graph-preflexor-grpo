#!/usr/bin/env python
"""Joint idea-space coverage map: project SEVERAL runs' concepts into ONE shared PCA so their
embedding-space coverage is directly comparable — the visual companion to scaling.py panel (b).

Two separate per-run semantic maps are NOT comparable (each is fit with its own PCA, so the axes
differ). This fits a SINGLE PCA on the pooled concepts of all runs, plots each run on the shared
axes with a per-run density contour, and reports each run's idea-space **spread** (total embedding
variance — the same quantity as scaling panel (b)) and 2D **coverage** (convex-hull area). Use it to
show whether one strategy (e.g. converse) spreads WIDER than another (e.g. novelty) — its points and
contour extend beyond the other's region.

    python embedmap.py --runs runs/exp_novelty_2 runs/exp_converse \
        --labels novelty converse --out figures/embedmap_converse
    python embedmap.py --runs runs/exp2 runs/exp_novelty_2 runs/exp_leap runs/exp_converse \
        --labels frontier novelty leap converse --out figures/embedmap_compare

Needs embeddings (reuses each run's recorded embed_model; override with --embed-model). --max-iter
truncates all runs to a common length for a fair comparison.
"""
import argparse
import json
import os

import numpy as np

import insights as I

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _spread(X):
    """Total embedding variance (sum of per-dim variances) — same metric as scaling panel (b)."""
    return float(X.var(axis=0).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run dirs to overlay (shared PCA)")
    p.add_argument("--labels", nargs="+", help="legend labels (default: dir names)")
    p.add_argument("--out", default=None, help="figure basename (default: <first-run>/figures/embedmap)")
    p.add_argument("--embed-model", dest="embed_model", default=None,
                   help="override the sentence-transformers id used for re-embedding")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=None,
                   help="truncate every run to iter <= this for a fair comparison")
    p.add_argument("--max-points", type=int, default=4000,
                   help="subsample points per run for the scatter/KDE (default 4000)")
    args = p.parse_args()

    import matplotlib.pyplot as plt
    from graphstore import resolve_embed_model
    if args.max_iter is not None:
        I.MAX_ITER = args.max_iter
        print(f"[embedmap] truncating all runs to iter <= {args.max_iter}")
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    out = args.out or os.path.join(args.runs[0].rstrip("/"), "figures", "embedmap")

    # embed each run
    Xs = []
    for r in args.runs:
        G = I.load_graph(r)
        model = resolve_embed_model(r, args.embed_model)
        vecs = I.embed_nodes(G, model)
        if not vecs:
            raise SystemExit(f"{r}: embeddings unavailable with '{model}' — needs them "
                             f"(huggingface-cli login, or --embed-model all-MiniLM-L6-v2).")
        X = np.stack([vecs[n] for n in G.nodes]).astype(np.float32)
        Xs.append(X)
        print(f"[embedmap] {r}: {len(X)} concepts  (spread={_spread(X):.4f})", flush=True)

    # ONE shared PCA on the pooled concepts → comparable axes for every run
    Xall = np.vstack(Xs)
    mean = Xall.mean(0)
    try:
        _, _, Vt = np.linalg.svd(Xall - mean, full_matrices=False)
        W = Vt[:2].T
    except Exception:
        W = np.eye(Xall.shape[1])[:, :2]

    fig, ax = plt.subplots(figsize=(9.0, 8.2))
    rng = np.random.default_rng(0)
    report = {}
    for X, lab, col in zip(Xs, labels, PALETTE):
        P = (X - mean) @ W
        idx = rng.choice(len(P), min(len(P), args.max_points), replace=False)
        Ps = P[idx]
        ax.scatter(Ps[:, 0], Ps[:, 1], s=8, color=col, alpha=0.22, linewidths=0, zorder=2)
        # per-run density contour (where the mass is) — robust to a few outliers
        try:
            from scipy.stats import gaussian_kde
            k = gaussian_kde(Ps.T)
            xg = np.linspace(P[:, 0].min(), P[:, 0].max(), 120)
            yg = np.linspace(P[:, 1].min(), P[:, 1].max(), 120)
            XX, YY = np.meshgrid(xg, yg)
            ZZ = k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            ax.contour(XX, YY, ZZ, levels=[ZZ.max() * 0.1], colors=col, linewidths=2.2, zorder=3)
        except Exception:
            pass
        # convex-hull area (2D extent)
        hull_area = None
        try:
            from scipy.spatial import ConvexHull
            hull_area = float(ConvexHull(P).volume)        # 'volume' is area in 2D
        except Exception:
            pass
        ax.plot([], [], color=col, lw=2.5,
                label=f"{lab}   spread={_spread(X):.3f}"
                      + (f",  area={hull_area:.2f}" if hull_area else "")
                      + f"  (n={len(X)})")
        report[lab] = {"n": int(len(X)), "spread": _spread(X), "hull_area_2d": hull_area}

    ax.set_title("Idea-space coverage (shared PCA) — wider spread = more of the space explored")
    ax.set_xlabel("PC1 (shared across runs)"); ax.set_ylabel("PC2 (shared across runs)")
    ax.legend(frameon=False, fontsize=9, loc="best", title="run  ·  spread = total embedding variance")
    ax.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out}_embedmap.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}_embedmap.png/.svg/.pdf")
    with open(f"{out}_embedmap.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}_embedmap.json")
    for lab, d in report.items():
        print(f"[embedmap] {lab}: spread={d['spread']:.4f}, 2D hull area={d['hull_area_2d']}")


if __name__ == "__main__":
    main()
