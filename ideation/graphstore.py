"""Accumulating knowledge graph with embedding-based node de-duplication.

Nodes emitted across many traces ("collagen", "Collagen fibrils", ...) are merged
into one canonical node when their embeddings are cosine-similar above `tau`.
Every node/edge carries provenance (which question/iteration/response added it).
"""
import functools
import json
import os
import numpy as np
import networkx as nx

# Default node-embedding model. EmbeddingGemma (Google, 2025) is newer and stronger than the
# classic MiniLM; it is the default. MiniLM stays available as a lighter, ungated fallback.
#   NOTE: embeddinggemma-300m is a *gated* HF model — `huggingface-cli login` + accept its
#   license once, and use a recent `sentence-transformers` (>= 5). If unavailable, set
#   embed_model (config / --embed-model) back to the legacy id below.
DEFAULT_EMBED_MODEL = "google/embeddinggemma-300m"
LEGACY_EMBED_MODEL = "all-MiniLM-L6-v2"


def make_embedder(model_name=DEFAULT_EMBED_MODEL, prompt_name=None):
    """Return a cached embed(text)->unit-norm np.array using sentence-transformers.

    EmbeddingGemma ships task-specific prompts; "STS" yields *symmetric* sentence-similarity
    embeddings, which is exactly what node dedup / diversity / link proposals need (we compare
    concept labels against each other). It is auto-selected for embeddinggemma when present and
    silently ignored for models (like MiniLM) that don't define it."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    if prompt_name is None and "embeddinggemma" in model_name.lower():
        prompt_name = "STS"
    prompts = getattr(model, "prompts", None) or {}
    use_prompt = prompt_name if prompt_name in prompts else None

    @functools.lru_cache(maxsize=20000)
    def embed(text):
        kw = dict(convert_to_numpy=True, normalize_embeddings=True)
        if use_prompt:
            kw["prompt_name"] = use_prompt
        v = model.encode([text], **kw)[0]
        return v.astype(np.float32)
    return embed


def cap_graph(G, max_iter):
    """Return the subgraph of nodes/edges introduced at `iter <= max_iter` (provenance from the
    'iter' attribute). Lets every analysis truncate runs to a common length for fair cross-run
    journal figures. `max_iter=None` returns G unchanged."""
    if max_iter is None:
        return G

    def _it(d):
        try:
            return int(float(d.get("iter", 0)))
        except Exception:
            return 0
    H = G.__class__()
    H.add_nodes_from((n, d) for n, d in G.nodes(data=True) if _it(d) <= max_iter)
    H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True)
                     if _it(d) <= max_iter and u in H and v in H)
    return H


def embed_texts(texts, model_name=DEFAULT_EMBED_MODEL, prompt_name=None, batch_size=64):
    """Batched embedding of many texts -> (n, d) unit-norm float32 array. Far faster than
    calling make_embedder() once per text for large node sets (single load, batched forward
    passes). Same STS-prompt auto-selection as make_embedder."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    if prompt_name is None and "embeddinggemma" in model_name.lower():
        prompt_name = "STS"
    prompts = getattr(model, "prompts", None) or {}
    kw = {"prompt_name": prompt_name} if prompt_name in prompts else {}
    texts = list(texts)
    V = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                     batch_size=batch_size, show_progress_bar=len(texts) > 256, **kw)
    return V.astype(np.float32)


def resolve_embed_model(run_dir=None, cli=None, default=DEFAULT_EMBED_MODEL):
    """Pick the embedding model for offline re-embedding: explicit CLI value wins, else the
    model the run recorded in summary.json (so plots/insights match the run's dedup), else
    the library default."""
    if cli:
        return cli
    if run_dir:
        try:
            s = json.load(open(os.path.join(run_dir, "summary.json")))
            m = (s.get("config") or {}).get("embed_model")
            if m:
                return m
        except Exception:
            pass
    return default


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
