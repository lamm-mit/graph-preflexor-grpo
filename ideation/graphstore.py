"""Accumulating knowledge graph with embedding-based node de-duplication.

Nodes emitted across many traces ("collagen", "Collagen fibrils", ...) are merged
into one canonical node when their embeddings are cosine-similar above `tau`.
Every node/edge carries provenance (which question/iteration/response added it).
"""
import functools
import numpy as np
import networkx as nx


def make_embedder(model_name="all-MiniLM-L6-v2"):
    """Return a cached embed(text)->unit-norm np.array using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    @functools.lru_cache(maxsize=20000)
    def embed(text):
        v = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return v.astype(np.float32)
    return embed


class GraphStore:
    def __init__(self, embed, tau=0.85):
        self.G = nx.DiGraph()
        self.embed = embed
        self.tau = tau
        self.vecs = {}        # canonical_id -> embedding
        self.origin = {}      # canonical_id -> response_id that introduced it

    def canonical(self, label):
        v = self.embed(label)
        best, best_sim = None, -1.0
        for cid, cv in self.vecs.items():
            sim = float(np.dot(v, cv))            # both unit-norm => cosine
            if sim > best_sim:
                best, best_sim = cid, sim
        if best is not None and best_sim >= self.tau:
            return best
        self.vecs[label] = v
        return label

    def merge(self, g_local, prov):
        """Merge a per-trace graph into the global graph. Returns list of NEW canonical ids."""
        mapping = {n: self.canonical(n) for n in g_local.nodes}
        new = []
        for raw, cid in mapping.items():
            if cid not in self.G:
                self.G.add_node(cid, label=cid, **prov)
                self.origin.setdefault(cid, prov.get("response_id"))
                new.append(cid)
        for u, v, d in g_local.edges(data=True):
            cu, cv = mapping[u], mapping[v]
            if cu != cv and not self.G.has_edge(cu, cv):
                self.G.add_edge(cu, cv, relation=d.get("relation", "related_to"), **prov)
        return new

    def node_vectors(self):
        return {cid: self.vecs[cid] for cid in self.G.nodes if cid in self.vecs}
