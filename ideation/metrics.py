"""Ideation / creativity metrics over the accumulated graph.

Graph dynamics + semantic diversity (embeddings) + simple creativity proxies
(fluency / flexibility / elaboration). Originality vs a reference set and the
LLM-judge eval live in compare.py.
"""
import numpy as np
import networkx as nx


def graph_metrics(store):
    G = store.G
    n, m = G.number_of_nodes(), G.number_of_edges()
    U = G.to_undirected()
    comps = list(nx.connected_components(U)) if n else []
    largest = max((len(c) for c in comps), default=0)
    out = {
        "nodes": n, "edges": m,
        "density": nx.density(G) if n > 1 else 0.0,
        "avg_degree": (2 * m / n) if n else 0.0,
        "n_components": len(comps),
        "largest_component_frac": (largest / n) if n else 0.0,
        "avg_clustering": nx.average_clustering(U) if n > 2 else 0.0,
    }
    try:
        out["longest_path"] = nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) else None
    except Exception:
        out["longest_path"] = None
    return out


def semantic_metrics(store):
    vecs = store.node_vectors()
    if len(vecs) < 2:
        return {"mean_pairwise_distance": 0.0, "embedding_spread": 0.0, "n_embedded": len(vecs)}
    X = np.stack(list(vecs.values()))
    sims = X @ X.T
    iu = np.triu_indices(len(X), k=1)
    mean_dist = float(1.0 - sims[iu].mean())          # higher => more diverse ideas
    spread = float(np.trace(np.cov(X.T)))             # variance over embedding dims
    return {"mean_pairwise_distance": mean_dist, "embedding_spread": spread, "n_embedded": len(X)}


def creativity_proxies(store, run_stats):
    """Torrance-style proxies mapped to the graph."""
    G = store.G
    n = G.number_of_nodes()
    calls = max(1, run_stats.get("calls", 1))
    # flexibility: # semantic clusters (cheap k via sqrt heuristic + kmeans if available)
    flexibility = None
    vecs = store.node_vectors()
    if len(vecs) >= 4:
        try:
            from sklearn.cluster import KMeans
            X = np.stack(list(vecs.values()))
            k = max(2, int(np.sqrt(len(X))))
            flexibility = int(len(set(KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X))))
        except Exception:
            flexibility = None
    return {
        "fluency": n,                                  # distinct ideas
        "ideas_per_call": n / calls,                   # ideation efficiency
        "elaboration": (G.number_of_edges() / n) if n else 0.0,   # edges per idea
        "flexibility_clusters": flexibility,
    }


def all_metrics(store, run_stats):
    return {**graph_metrics(store), **semantic_metrics(store),
            **creativity_proxies(store, run_stats)}
