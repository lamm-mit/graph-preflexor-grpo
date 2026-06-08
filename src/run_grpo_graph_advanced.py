#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_grpo_graph_advanced.py

GRPO training on top of ORPO graph-native model with ADVANCED typed schema.

This script supports the domain-agnostic typed graph schema with:
- Node types: entity, attribute, process, event, outcome, law, claim
- Node levels: micro, meso, macro (optional)
- Constrained relations: causes, enables, inhibits, modulates, part_of, instance_of,
                        supports, challenges, represents, promotes, violates, constrains

Requires dataset with columns:
- prompt: question
- answer: gold answer
- teacher_graph_json: (optional) teacher's graph JSON with typed nodes
- chosen, rejected: not used here

Reward components:
- format_score      (graph DSL correctness & JSON validity)
- correctness       (final answer vs gold)
- graph_utility     (can a judge, using ONLY graph_json, reconstruct the answer?)
- graph_schema      (compliance with typed schema: valid types, relations, levels)
- graph_networkx    (NetworkX validity: edge consistency, connectivity)
- graph_diversity   (semantic diversity using sentence embeddings)
- graph_structure   (topology quality: depth, internal nodes, size)

Usage example:

python run_grpo_graph_advanced.py \
  --base_model_dir ./orpo_graph_llama32_3b \
  --dataset lamm-mit/graph_reasoning_advanced \
  --output_dir ./grpo_graph_llama32_3b_advanced \
  --judge_model gpt-5-mini \
  --judge_api_key $OPENAI_API_KEY \
  --judge_base_url http://localhost:8000/v1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_generations 4 \
  --learning_rate 5e-6 \
  --epochs 1 \
  --weight_graph_schema 0.1 \
  --push_to_hub \
  --hub_model_id lamm-mit/llama32-3b-grpo-graph-reasoning-advanced \
  --hf_token $HF_TOKEN

"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import wandb

# Optional imports for graph analysis (lazy loaded)
_networkx = None
_sentence_transformer = None

# Set up reward logger (configured later based on --debug_rewards flag)
reward_logger = logging.getLogger('rewards')
reward_logger.setLevel(logging.WARNING)  # Default: off (only warnings)
from datasets import load_dataset
from huggingface_hub import HfFolder
from openai import OpenAI
from pydantic import BaseModel, Field
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import GRPOConfig, GRPOTrainer

# Sentinels
THINK_START = "<think>"
THINK_END = "</think>"
BRAINSTORM_START = "<brainstorm>"
BRAINSTORM_END = "</brainstorm>"
GRAPH_START = "<graph>"
GRAPH_END = "</graph>"
GRAPH_JSON_START = "<graph_json>"
GRAPH_JSON_END = "</graph_json>"
PATTERNS_START = "<patterns>"
PATTERNS_END = "</patterns>"
SYNTHESIS_START = "<synthesis>"
SYNTHESIS_END = "</synthesis>"

SPECIAL_TOKENS = [
    THINK_START, THINK_END,
    BRAINSTORM_START, BRAINSTORM_END,
    GRAPH_START, GRAPH_END,
    GRAPH_JSON_START, GRAPH_JSON_END,
    PATTERNS_START, PATTERNS_END,
    SYNTHESIS_START, SYNTHESIS_END,
]

# ----- Graph schema (advanced, domain-agnostic) -----

# Valid node types (domain-general semantic roles)
NODE_TYPES = ["entity", "attribute", "process", "event", "outcome", "law", "claim"]

# Valid scale levels
SCALE_LEVELS = ["micro", "meso", "macro"]

# Valid relation vocabulary (constrained to 12 verbs)
VALID_RELATIONS = [
    "causes", "enables", "inhibits", "modulates",
    "part_of", "instance_of",
    "supports", "challenges",
    "represents", "promotes", "violates", "constrains",
]


class Node(BaseModel):
    """
    Node in the reasoning graph.

    - id:    short CamelCase identifier, no spaces (e.g., "LaserPulse", "ThemeAlienation")
    - type:  semantic role of the node (domain-general)
    - level: optional scale (micro/meso/macro)
    """
    id: str = Field(...)
    type: Literal[
        "entity", "attribute", "process", "event", "outcome", "law", "claim"
    ]
    level: Optional[Literal["micro", "meso", "macro"]] = None


class Edge(BaseModel):
    """
    Edge in the reasoning graph.

    - relation: chosen from a small, reusable vocabulary that works across domains.
    """
    source: str
    relation: Literal[
        "causes", "enables", "inhibits", "modulates",
        "part_of", "instance_of",
        "supports", "challenges",
        "represents", "promotes", "violates", "constrains",
    ]
    target: str


class GraphJSON(BaseModel):
    nodes: List[Node]
    edges: List[Edge] = Field(default_factory=list)


def validate_graph_semantics(obj: Dict[str, Any], strict: bool = True) -> Optional[Dict[str, Any]]:
    """
    Validate graph JSON beyond basic Pydantic structure.

    Checks:
    - Node ids must be unique
    - Node ids should have no spaces (CamelCase)
    - All edge endpoints must reference existing node ids
    - If strict=True: all types/relations must be from constrained vocabulary
    - If strict=False: accept any type/relation (backward compatible)

    Returns cleaned object if valid, else None.
    """
    try:
        nodes = obj.get("nodes", [])
        edges = obj.get("edges", [])

        if not nodes:
            return None

        # Collect and validate node ids
        ids = []
        for n in nodes:
            nid = n.get("id", "")
            if not isinstance(nid, str) or not nid:
                return None
            if " " in nid:
                return None  # No spaces allowed in ids
            ids.append(nid)

        # Check for unique ids
        if len(ids) != len(set(ids)):
            return None

        id_set = set(ids)

        # Validate edges reference existing nodes
        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            if src not in id_set or tgt not in id_set:
                return None

        # If strict mode, validate against Pydantic schema (checks types and relations)
        if strict:
            try:
                _ = GraphJSON(**obj)
            except Exception:
                return None

        return obj

    except Exception:
        return None


def extract_graph_json_advanced(full_output: str, strict: bool = True) -> Optional[str]:
    """
    Extract graph_json from output and validate against advanced schema.

    Args:
        full_output: Model's full output string
        strict: If True, require typed schema compliance. If False, accept simpler schema.

    Returns:
        JSON string if valid, None otherwise
    """
    # Find thinking block
    th_start = full_output.find(THINK_START)
    th_end = full_output.find(THINK_END)
    if th_start == -1 or th_end == -1 or th_end <= th_start:
        return None

    region = full_output[th_start:th_end + len(THINK_END)]

    # Find graph_json block
    gj_start = region.find(GRAPH_JSON_START)
    gj_end = region.find(GRAPH_JSON_END)
    if gj_start == -1 or gj_end == -1 or gj_end <= gj_start:
        return None

    inner = region[gj_start + len(GRAPH_JSON_START):gj_end].strip()

    try:
        obj = json.loads(inner)
        validated = validate_graph_semantics(obj, strict=strict)
        if validated is None:
            return None
        return json.dumps(validated, ensure_ascii=False)
    except Exception:
        return None


# ----- LLM client / judge -----

def get_llm_client(api_key: str, base_url: Optional[str] = None, timeout: float = 120.0) -> OpenAI:
    params: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        params["base_url"] = base_url
    if timeout:
        params["timeout"] = timeout
    return OpenAI(**params)


def judge_json_object(api_client: OpenAI, model: str, system_prompt: str, user_prompt: str, debug_label: str = "") -> Optional[Dict[str, Any]]:
    try:
        resp = api_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
        )
        txt = (resp.output_text or "").strip()
    except Exception:
        # fall back to chat completions
        try:
            chat = api_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            txt = (chat.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[judge_json_object] error: {e}")
            return None

    # extract JSON
    idx1 = txt.find("{")
    idx2 = txt.rfind("}")
    if idx1 == -1 or idx2 == -1 or idx2 <= idx1:
        reward_logger.warning(f"[Judge:{debug_label}] No JSON found in response: {txt[:200]}...")
        return None
    try:
        obj = json.loads(txt[idx1 : idx2 + 1])
        if debug_label:
            reward_logger.info(f"[Judge:{debug_label}] {obj}")
        return obj
    except Exception as e:
        reward_logger.warning(f"[Judge:{debug_label}] JSON parse error: {e}, raw: {txt[:200]}...")
        return None


# ----- Parsing helpers -----

def find_once_span(haystack: str, start_tag: str, end_tag: str) -> Optional[Tuple[int, int]]:
    s = haystack.find(start_tag)
    if s == -1:
        return None
    e = haystack.find(end_tag, s + len(start_tag))
    if e == -1:
        return None
    # end index inclusive of close tag
    return (s, e + len(end_tag))


def extract_inner_from_span(s: str, span: Tuple[int, int], open_tag: str, close_tag: str) -> str:
    block = s[span[0] : span[1]]
    if not (block.startswith(open_tag) and block.endswith(close_tag)):
        oi = block.find(open_tag)
        cj = block.rfind(close_tag)
        if oi == -1 or cj == -1 or cj <= oi + len(open_tag):
            raise ValueError("Malformed tag block.")
        return block[oi + len(open_tag) : cj].strip()
    return block[len(open_tag) : -len(close_tag)].strip()


def extract_post_thinking_answer(full_output: str) -> Optional[str]:
    idx = full_output.find(THINK_END)
    if idx == -1:
        return None
    return full_output[idx + len(THINK_END) :].strip()


def extract_graph_json_model(full_output: str, strict: bool = False) -> Optional[str]:
    """
    Extract graph_json from model output.

    Args:
        full_output: Model's full output string
        strict: If True, validate against typed schema (types, relations).
                If False (default), accept any valid JSON with nodes/edges.

    Returns:
        JSON string if valid, None otherwise.
    """
    th_span = find_once_span(full_output, THINK_START, THINK_END)
    if not th_span:
        return None
    region = full_output[th_span[0] : th_span[1]]
    gj_span = find_once_span(region, GRAPH_JSON_START, GRAPH_JSON_END)
    if not gj_span:
        return None
    try:
        inner = extract_inner_from_span(region, gj_span, GRAPH_JSON_START, GRAPH_JSON_END)
        obj = json.loads(inner)

        # Basic validation: must have nodes array
        if "nodes" not in obj or not isinstance(obj["nodes"], list):
            return None

        if strict:
            # Full typed schema validation
            _ = GraphJSON(**obj)
        else:
            # Lenient: just check basic structure
            for n in obj.get("nodes", []):
                if "id" not in n:
                    return None

        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return None


# ----- Reward components -----

def score_format(full_output: str) -> float:
    """
    Format score [0,1]:
    - 0.15 thinking tags present
    - 0.10 brainstorm tags present
    - 0.15 graph tags present
    - 0.20 graph_json tags present and JSON-parseable
    - 0.15 patterns tags present
    - 0.15 synthesis tags present
    - 0.10 non-empty graph_json.nodes
    """
    score = 0.0

    if THINK_START in full_output and THINK_END in full_output:
        score += 0.15

    if BRAINSTORM_START in full_output and BRAINSTORM_END in full_output:
        score += 0.10

    if GRAPH_START in full_output and GRAPH_END in full_output:
        score += 0.15

    gj_str = extract_graph_json_model(full_output)
    if gj_str is not None:
        score += 0.20
    else:
        return max(0.0, min(1.0, score))  # no more credit possible if graph_json invalid

    if PATTERNS_START in full_output and PATTERNS_END in full_output:
        score += 0.15

    if SYNTHESIS_START in full_output and SYNTHESIS_END in full_output:
        score += 0.15

    try:
        obj = json.loads(gj_str)
        g = GraphJSON(**obj)
        if len(g.nodes) > 0:
            score += 0.10
    except Exception:
        pass

    return max(0.0, min(1.0, score))


def score_correctness(
    api_client: OpenAI,
    judge_model: str,
    question: str,
    gold_answer: str,
    full_output: str,
) -> float:
    """
    Use judge LLM on the post-thinking answer.
    """
    ans = extract_post_thinking_answer(full_output)
    if not ans:
        return 0.0

    system_prompt = (
        "You are a precise grader. Score answers on a continuous scale from 0.0 to 1.0. "
        "Use the FULL range - avoid defaulting to 0.4 or 0.7. Be specific. "
        'Return JSON: {"score": <float>, "justification": "<reason>", "error_flags": []}'
    )

    user_prompt = f"""
Question:
{question}

Gold Answer:
{gold_answer}

Candidate Answer:
{ans}

Score on a continuous 0.0-1.0 scale based on:
- Factual accuracy (does it match the gold answer?)
- Completeness (are key points covered?)
- Specificity (concrete details vs vague statements?)
- Errors (any incorrect claims?)

Scoring guide (use values BETWEEN these, not just these exact numbers):
- 0.95-1.0: Fully correct, complete, specific
- 0.80-0.94: Correct with minor omissions
- 0.60-0.79: Mostly correct, some gaps or vagueness
- 0.40-0.59: Partially correct, significant gaps
- 0.20-0.39: Few correct elements, mostly incomplete
- 0.01-0.19: Mostly wrong or irrelevant
- 0.0: Completely wrong or no answer

Return only JSON.
"""

    obj = judge_json_object(api_client, judge_model, system_prompt, user_prompt, debug_label="correctness")
    if not obj:
        return 0.0
    try:
        s = float(obj.get("score", 0.0))
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.0


def score_graph_utility(
    api_client: OpenAI,
    judge_model: str,
    question: str,
    gold_answer: str,
    full_output: str,
) -> float:
    """
    Graph utility: can the judge reconstruct a good answer using ONLY the model's graph_json?
    """
    gj_str = extract_graph_json_model(full_output)
    if gj_str is None:
        return 0.0

    system_prompt = (
        "You see ONLY a knowledge graph in JSON form and a question. "
        "Use ONLY the graph nodes and edges to infer an answer - no outside knowledge. "
        "If the graph lacks information, say so."
    )
    user_prompt = f"""
Question:
{question}

Graph JSON:
{gj_str}

Using ONLY the information in this graph (nodes and their relationships via edges), answer the question.
- If the graph contains relevant information, synthesize an answer from it
- If the graph is missing key information, note what's missing
- Do not use any knowledge beyond what's in the graph

Return JSON: {{"answer": "your answer based solely on the graph", "graph_coverage": "complete|partial|insufficient"}}
"""

    # First: let judge produce a graph-based answer
    obj = judge_json_object(api_client, judge_model, system_prompt, user_prompt, debug_label="graph_reconstruct")
    if not obj or "answer" not in obj:
        return 0.0

    graph_based_answer = str(obj["answer"])

    # Now grade that graph-based answer vs the gold answer
    system_prompt2 = (
        "You are a precise grader evaluating if a knowledge graph captured enough information to answer a question. "
        "Score on a continuous 0.0-1.0 scale. Use the FULL range."
    )
    user_prompt2 = f"""
Question:
{question}

Gold Answer (what a complete answer should contain):
{gold_answer}

Answer derived from graph only:
{graph_based_answer}

How well does the graph-derived answer capture the gold answer's key information?

Score continuously from 0.0 to 1.0:
- 0.90-1.0: Graph captured all key concepts and relationships
- 0.70-0.89: Graph captured most information, minor gaps
- 0.50-0.69: Graph captured core idea but missing important details
- 0.30-0.49: Graph captured some relevant info but major gaps
- 0.10-0.29: Graph barely useful, most information missing
- 0.0-0.09: Graph not useful for answering this question

Return JSON: {{"score": <float>, "missing_from_graph": "<what key info the graph lacked>"}}
"""

    obj2 = judge_json_object(api_client, judge_model, system_prompt2, user_prompt2, debug_label="graph_grade")
    if not obj2:
        return 0.0
    try:
        s = float(obj2.get("score", 0.0))
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.0


# ----- Optional reward components (enabled via non-zero weights) -----

def _get_networkx():
    """Lazy load networkx."""
    global _networkx
    if _networkx is None:
        import networkx as nx
        _networkx = nx
    return _networkx


def _get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy load sentence transformer."""
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer(model_name)
    return _sentence_transformer


def score_graph_schema(full_output: str) -> float:
    """
    Score [0,1]: How well does the graph comply with the advanced typed schema?

    Evaluates:
    - 0.20: Valid JSON with nodes array
    - 0.20: All node IDs are unique, no spaces (CamelCase)
    - 0.20: All nodes have valid 'type' from vocabulary
    - 0.20: All edges have valid 'relation' from vocabulary
    - 0.10: All edge endpoints reference existing node IDs
    - 0.10: Nodes have optional 'level' field when present

    This is a soft scoring function that gives partial credit for partial compliance.
    """
    # Try to extract raw JSON first (non-strict)
    th_start = full_output.find(THINK_START)
    th_end = full_output.find(THINK_END)
    if th_start == -1 or th_end == -1 or th_end <= th_start:
        return 0.0

    region = full_output[th_start:th_end + len(THINK_END)]

    gj_start = region.find(GRAPH_JSON_START)
    gj_end = region.find(GRAPH_JSON_END)
    if gj_start == -1 or gj_end == -1 or gj_end <= gj_start:
        return 0.0

    inner = region[gj_start + len(GRAPH_JSON_START):gj_end].strip()

    try:
        obj = json.loads(inner)
    except Exception:
        return 0.0  # Not valid JSON

    nodes = obj.get("nodes", [])
    edges = obj.get("edges", [])

    if not nodes:
        return 0.1  # Has JSON but empty nodes

    score = 0.0

    # Base: valid JSON with nodes (0.20)
    score += 0.20

    # Check node IDs: unique and no spaces (0.20)
    ids = []
    valid_ids = 0
    for n in nodes:
        nid = n.get("id", "")
        if isinstance(nid, str) and nid and " " not in nid:
            valid_ids += 1
            ids.append(nid)

    if len(ids) == len(set(ids)) and valid_ids == len(nodes):
        score += 0.20
    else:
        # Partial credit
        if len(nodes) > 0:
            score += 0.20 * (valid_ids / len(nodes)) * 0.5

    id_set = set(ids)

    # Check node types (0.20)
    valid_types = 0
    for n in nodes:
        ntype = n.get("type", "")
        if ntype in NODE_TYPES:
            valid_types += 1

    if len(nodes) > 0:
        type_ratio = valid_types / len(nodes)
        score += 0.20 * type_ratio

    # Check edge relations (0.20)
    if edges:
        valid_relations = 0
        for e in edges:
            rel = e.get("relation", "")
            if rel in VALID_RELATIONS:
                valid_relations += 1
        rel_ratio = valid_relations / len(edges)
        score += 0.20 * rel_ratio
    else:
        # No edges to check, give partial credit
        score += 0.10

    # Check edge endpoints reference existing nodes (0.10)
    if edges:
        valid_endpoints = 0
        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            if src in id_set and tgt in id_set:
                valid_endpoints += 1
        endpoint_ratio = valid_endpoints / len(edges)
        score += 0.10 * endpoint_ratio
    else:
        score += 0.05

    # Check level field when present (0.10)
    levels_present = 0
    levels_valid = 0
    for n in nodes:
        if "level" in n:
            levels_present += 1
            if n["level"] in SCALE_LEVELS or n["level"] is None:
                levels_valid += 1

    if levels_present > 0:
        score += 0.10 * (levels_valid / levels_present)
    else:
        # No levels specified, that's OK (field is optional)
        score += 0.05

    return max(0.0, min(1.0, score))


def score_graph_networkx(full_output: str) -> float:
    """
    Score [0,1]: Can the graph_json be parsed into a valid NetworkX graph?
    Checks:
    - Valid JSON (already done in extract, but re-verify)
    - All edge sources/targets exist as nodes
    - No self-loops (optional penalty)
    - Graph is non-empty
    """
    nx = _get_networkx()

    gj_str = extract_graph_json_model(full_output)
    if gj_str is None:
        return 0.0

    try:
        obj = json.loads(gj_str)
        nodes = obj.get("nodes", [])
        edges = obj.get("edges", [])

        if not nodes:
            return 0.1  # Empty graph

        # Build NetworkX graph
        G = nx.DiGraph()
        node_ids = set()
        for n in nodes:
            nid = n.get("id", "")
            if nid:
                G.add_node(nid)
                node_ids.add(nid)

        score = 0.3  # Base score for having nodes

        # Check edges
        valid_edges = 0
        invalid_edges = 0
        self_loops = 0
        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            rel = e.get("relation", "related_to")

            if src in node_ids and tgt in node_ids:
                G.add_edge(src, tgt, relation=rel)
                valid_edges += 1
                if src == tgt:
                    self_loops += 1
            else:
                invalid_edges += 1

        # Score components
        if valid_edges > 0:
            score += 0.3  # Has valid edges

        if invalid_edges == 0:
            score += 0.2  # All edges reference valid nodes
        else:
            score += 0.2 * max(0, 1 - invalid_edges / max(1, len(edges)))

        if self_loops == 0:
            score += 0.1  # No self-loops

        # Connectivity bonus
        if len(node_ids) > 1:
            if nx.is_weakly_connected(G):
                score += 0.1  # Graph is connected
        else:
            score += 0.1  # Single node is trivially connected

        return max(0.0, min(1.0, score))

    except Exception as e:
        reward_logger.warning(f"[score_graph_networkx] Error: {e}")
        return 0.0


def score_graph_diversity(full_output: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Score [0,1]: Semantic diversity of ideas in the graph.
    Uses sentence embeddings to measure spread of concepts.
    Higher diversity (lower avg pairwise similarity) = better score.
    """
    gj_str = extract_graph_json_model(full_output)
    if gj_str is None:
        return 0.0

    try:
        obj = json.loads(gj_str)
        nodes = obj.get("nodes", [])
        edges = obj.get("edges", [])

        # Collect text elements
        texts = []
        for n in nodes:
            nid = n.get("id", "")
            if nid:
                texts.append(nid)
        for e in edges:
            rel = e.get("relation", "")
            if rel:
                # Include edge as "source relation target"
                src = e.get("source", "")
                tgt = e.get("target", "")
                texts.append(f"{src} {rel} {tgt}")

        if len(texts) < 2:
            return 0.3  # Not enough elements to measure diversity

        # Get embeddings
        model = _get_sentence_transformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)

        # Compute pairwise cosine similarities
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-8)

        # Pairwise similarity matrix
        sim_matrix = np.dot(embeddings_norm, embeddings_norm.T)

        # Average pairwise similarity (excluding diagonal)
        n = len(texts)
        mask = ~np.eye(n, dtype=bool)
        avg_sim = sim_matrix[mask].mean()

        # Convert to diversity score: low similarity = high diversity
        # Knowledge graph nodes typically have similarity 0.15-0.50
        # Tighter range for better discrimination among good graphs
        # Map: 0.15 sim -> 1.0, 0.50 sim -> 0.0
        diversity_score = max(0.0, min(1.0, 1.0 - (avg_sim - 0.15) / 0.35))

        # Small bonus for having more elements (richer graph)
        size_bonus = min(0.1, len(texts) / 100)  # Up to 0.1 bonus for 10+ elements

        return max(0.0, min(1.0, diversity_score * 0.9 + size_bonus))

    except Exception as e:
        reward_logger.warning(f"[score_graph_diversity] Error: {e}")
        return 0.0


def score_graph_structure(full_output: str) -> float:
    """
    Score [0,1]: Graph topology/structure quality.
    Rewards:
    - Depth (longer reasoning chains)
    - Internal nodes (nodes with both in and out edges)
    - Balanced structure (not just star or linear)
    """
    nx = _get_networkx()

    gj_str = extract_graph_json_model(full_output)
    if gj_str is None:
        return 0.0

    try:
        obj = json.loads(gj_str)
        nodes = obj.get("nodes", [])
        edges = obj.get("edges", [])

        if not nodes:
            return 0.0

        # Build NetworkX graph
        G = nx.DiGraph()
        node_ids = set()
        for n in nodes:
            nid = n.get("id", "")
            if nid:
                G.add_node(nid)
                node_ids.add(nid)

        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            if src in node_ids and tgt in node_ids:
                G.add_edge(src, tgt)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_nodes == 0:
            return 0.0

        score = 0.0

        # 1. Size score (0.2 max) - reward 5-20 nodes
        if n_nodes >= 5:
            size_score = min(0.2, 0.2 * min(n_nodes, 20) / 20)
        else:
            size_score = 0.1 * n_nodes / 5
        score += size_score

        # 2. Edge density (0.2 max) - reward having edges
        if n_nodes > 1:
            density = n_edges / (n_nodes * (n_nodes - 1))  # Max possible directed edges
            score += min(0.2, density * 2)  # Reward up to 10% density

        # 3. Internal node ratio (0.3 max) - nodes with both in and out edges
        internal_nodes = sum(1 for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0)
        if n_nodes > 2:
            internal_ratio = internal_nodes / n_nodes
            score += 0.3 * internal_ratio

        # 4. Depth score (0.2 max) - longest path in DAG
        try:
            if nx.is_directed_acyclic_graph(G):
                longest_path = nx.dag_longest_path_length(G)
                # Reward paths of length 3-6
                depth_score = min(0.2, 0.2 * min(longest_path, 6) / 6)
            else:
                # Has cycles - use diameter of underlying undirected graph
                UG = G.to_undirected()
                if nx.is_connected(UG):
                    diameter = nx.diameter(UG)
                    depth_score = min(0.2, 0.2 * min(diameter, 6) / 6)
                else:
                    depth_score = 0.1
            score += depth_score
        except Exception:
            score += 0.1  # Default if path calculation fails

        # 5. Connectivity (0.1 max)
        if n_nodes > 1:
            if nx.is_weakly_connected(G):
                score += 0.1
        else:
            score += 0.1

        return max(0.0, min(1.0, score))

    except Exception as e:
        reward_logger.warning(f"[score_graph_structure] Error: {e}")
        return 0.0


def combined_reward(
    api_client: OpenAI,
    judge_model: str,
    question: str,
    gold_answer: str,
    full_output: str,
    weight_correctness: float = 0.4,
    weight_format: float = 0.3,
    weight_graph_utility: float = 0.3,
    # Optional components (disabled by default)
    weight_graph_schema: float = 0.0,  # NEW: typed schema compliance
    weight_graph_networkx: float = 0.0,
    weight_graph_diversity: float = 0.0,
    weight_graph_structure: float = 0.0,
) -> float:
    # Core rewards (always computed if weight > 0)
    fmt = score_format(full_output) if weight_format > 0 else 0.0
    corr = score_correctness(api_client, judge_model, question, gold_answer, full_output) if weight_correctness > 0 else 0.0
    gut = score_graph_utility(api_client, judge_model, question, gold_answer, full_output) if weight_graph_utility > 0 else 0.0

    # Optional rewards (only computed if weight > 0)
    gschema = score_graph_schema(full_output) if weight_graph_schema > 0 else 0.0  # NEW
    gnx = score_graph_networkx(full_output) if weight_graph_networkx > 0 else 0.0
    gdiv = score_graph_diversity(full_output) if weight_graph_diversity > 0 else 0.0
    gstruct = score_graph_structure(full_output) if weight_graph_structure > 0 else 0.0

    total = float(
        weight_format * fmt +
        weight_correctness * corr +
        weight_graph_utility * gut +
        weight_graph_schema * gschema +  # NEW
        weight_graph_networkx * gnx +
        weight_graph_diversity * gdiv +
        weight_graph_structure * gstruct
    )

    # Build log message (only show non-zero components)
    log_parts = []
    if weight_format > 0:
        log_parts.append(f"format={fmt:.3f}")
    if weight_correctness > 0:
        log_parts.append(f"correctness={corr:.3f}")
    if weight_graph_utility > 0:
        log_parts.append(f"graph_utility={gut:.3f}")
    if weight_graph_schema > 0:
        log_parts.append(f"graph_schema={gschema:.3f}")  # NEW
    if weight_graph_networkx > 0:
        log_parts.append(f"graph_nx={gnx:.3f}")
    if weight_graph_diversity > 0:
        log_parts.append(f"graph_div={gdiv:.3f}")
    if weight_graph_structure > 0:
        log_parts.append(f"graph_struct={gstruct:.3f}")
    log_parts.append(f"total={total:.3f}")
    reward_logger.info(f"[Reward] {' '.join(log_parts)}")

    # Log to wandb if active
    if wandb.run is not None:
        wandb_log = {"reward/total": total}
        if weight_format > 0:
            wandb_log["reward/format"] = fmt
        if weight_correctness > 0:
            wandb_log["reward/correctness"] = corr
        if weight_graph_utility > 0:
            wandb_log["reward/graph_utility"] = gut
        if weight_graph_schema > 0:
            wandb_log["reward/graph_schema"] = gschema  # NEW
        if weight_graph_networkx > 0:
            wandb_log["reward/graph_networkx"] = gnx
        if weight_graph_diversity > 0:
            wandb_log["reward/graph_diversity"] = gdiv
        if weight_graph_structure > 0:
            wandb_log["reward/graph_structure"] = gstruct
        wandb.log(wandb_log, commit=False)

    return total


# ----- GRPO reward shim -----
def make_reward_function(
    api_client: OpenAI,
    judge_model: str,
    weight_correctness: float = 0.4,
    weight_format: float = 0.3,
    weight_graph_utility: float = 0.3,
    # Optional components
    weight_graph_schema: float = 0.0,  # NEW: typed schema compliance
    weight_graph_networkx: float = 0.0,
    weight_graph_diversity: float = 0.0,
    weight_graph_structure: float = 0.0,
):
    def reward_function(
        completions: List[str],
        prompts: Optional[List[str]] = None,
        question: Optional[List[str]] = None,
        gold_answer: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        n = len(completions)
        # question & gold_answer come from dataset columns
        q_list = question if question is not None else prompts
        if q_list is None:
            q_list = [""] * n
        ga_list = gold_answer if gold_answer is not None else ["" for _ in range(n)]

        rewards: List[float] = []
        for i, out in enumerate(completions):
            q = q_list[i] if i < len(q_list) else ""
            ga = ga_list[i] if i < len(ga_list) else ""
            if not ga:
                rewards.append(0.0)
                continue
            r = combined_reward(
                api_client, judge_model, q, ga, out,
                weight_correctness=weight_correctness,
                weight_format=weight_format,
                weight_graph_utility=weight_graph_utility,
                weight_graph_schema=weight_graph_schema,  # NEW
                weight_graph_networkx=weight_graph_networkx,
                weight_graph_diversity=weight_graph_diversity,
                weight_graph_structure=weight_graph_structure,
            )
            rewards.append(r)
        return rewards

    return reward_function


# ----- Main -----

def main():
    parser = argparse.ArgumentParser(
        description="Run GRPO for graph-native reasoning with ADVANCED typed schema. "
                    "Supports typed nodes (entity/attribute/process/event/outcome/law/claim), "
                    "scale levels (micro/meso/macro), and constrained relation vocabulary."
    )
    parser.add_argument("--base_model_dir", type=str, required=True, help="Directory of ORPO-trained model")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./grpo_graph_model")

    parser.add_argument("--judge_model", type=str, required=True)
    parser.add_argument("--judge_api_key", type=str, required=True)
    parser.add_argument("--judge_base_url", type=str, default=None)

    # Reward weights (must sum to 1.0)
    parser.add_argument("--weight_correctness", type=float, default=0.4, help="Weight for answer correctness (default: 0.4)")
    parser.add_argument("--weight_format", type=float, default=0.3, help="Weight for format compliance (default: 0.3)")
    parser.add_argument("--weight_graph_utility", type=float, default=0.3, help="Weight for graph utility (default: 0.3)")
    # Optional reward components (disabled by default, set weight > 0 to enable)
    parser.add_argument("--weight_graph_schema", type=float, default=0.0,
                        help="Weight for typed schema compliance (node types, relations, levels) (default: 0.0 = disabled)")
    parser.add_argument("--weight_graph_networkx", type=float, default=0.0,
                        help="Weight for NetworkX graph validity check (default: 0.0 = disabled)")
    parser.add_argument("--weight_graph_diversity", type=float, default=0.0,
                        help="Weight for semantic diversity of graph concepts (default: 0.0 = disabled)")
    parser.add_argument("--weight_graph_structure", type=float, default=0.0,
                        help="Weight for graph topology/structure quality (default: 0.0 = disabled)")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=4096, help="Max completion length (reasoning + answer)")
    parser.add_argument("--temperature", type=float, default=0.4)

    # GRPO algorithm settings
    parser.add_argument("--scale_rewards", type=str, default="batch",
                        choices=["batch", "group", "none"],
                        help="Reward scaling method: 'batch' (normalize across batch), 'group' (normalize per prompt group), 'none' (default: batch)")
    parser.add_argument("--loss_type", type=str, default="dapo",
                        choices=["grpo", "dapo", "dr_grpo", "rloo"],
                        help="Loss function type: 'grpo' (standard), 'dapo' (dynamic), 'dr_grpo' (dropout regularized), 'rloo' (leave-one-out) (default: dapo)")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_lora", action="store_true", help="Train full model instead of LoRA")
    parser.add_argument("--add_new_special_tokens", action="store_true",
                        help="Add custom special tokens and resize embeddings (default: False, uses existing tokenizer)")

    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)

    # vLLM options
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for faster generation")
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"],
                        help="vLLM mode: 'colocate' (same process) or 'server' (external server) (default: colocate)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.6, help="vLLM GPU memory utilization (default: 0.6)")
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0", help="vLLM server host (default: 0.0.0.0)")
    parser.add_argument("--vllm_server_port", type=int, default=8000, help="vLLM server port (default: 8000)")

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_public", action="store_true", help="Make Hub repo public (default: private)")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--debug_rewards", action="store_true", help="Enable verbose reward/judge logging to grpo_rewards.log")
    parser.add_argument("--save_merged_orpo", action="store_true", default=True,
                        help="Save and push the merged ORPO model before adding GRPO LoRA (default: True)")
    parser.add_argument("--no_save_merged_orpo", action="store_false", dest="save_merged_orpo",
                        help="Disable saving the merged ORPO model")
    parser.add_argument("--resume_grpo_checkpoint", type=str, default=None,
                        help="Path to GRPO checkpoint to resume training from (loads existing LoRA adapter instead of creating new one)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to trainer checkpoint for crash recovery (restores optimizer, scheduler, step count)")

    args = parser.parse_args()

    # Validate resume options (mutually exclusive)
    if args.resume_grpo_checkpoint and args.resume_from_checkpoint:
        raise ValueError(
            "Cannot use both --resume_grpo_checkpoint and --resume_from_checkpoint.\n"
            "  --resume_grpo_checkpoint: Load LoRA weights, start training fresh (new optimizer/LR)\n"
            "  --resume_from_checkpoint: Full crash recovery (restore optimizer, scheduler, step count)"
        )

    # Configure reward debugging
    if args.debug_rewards:
        reward_logger.setLevel(logging.INFO)
        handler = logging.FileHandler('grpo_rewards.log')
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        reward_logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(message)s'))
        reward_logger.addHandler(console)
        print("Reward debugging enabled → grpo_rewards.log")

    if args.hf_token:
        HfFolder.save_token(args.hf_token)

    # Load dataset
    ds = load_dataset(args.dataset, split="train")
    required_cols = {"prompt", "answer"}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset {args.dataset} missing required columns: {missing}")

    # Load tokenizer first (needed for chat template)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for generation

    # For GRPO we want columns:
    #   - prompt   (with chat template applied)
    #   - question (raw question for reward function)
    #   - gold_answer
    def map_for_grpo(ex):
        q = (ex.get("prompt") or "").strip()
        ga = (ex.get("answer") or "").strip()

        if not q:
            return {"prompt": None, "gold_answer": None, "question": None}

        # Apply chat template with generation prompt
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback if chat template fails
            prompt = q

        return {
            "prompt": prompt,      # chat-templated prompt for model
            "question": q,         # raw question for reward function
            "gold_answer": ga,     # gold answer for reward function
        }

    ds_mapped = ds.map(map_for_grpo)
    # Filter out any None rows
    ds_mapped = ds_mapped.filter(lambda x: x["prompt"] is not None)

    # small eval split
    split = ds_mapped.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Dataset: {len(train_ds)} train, {len(eval_ds)} eval")

    # Load ORPO model (with automatic PEFT adapter detection)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading ORPO model from: {args.base_model_dir}")
    if args.use_vllm:
        if args.vllm_mode == "server":
            print(f"vLLM enabled (mode=server, host={args.vllm_server_host}:{args.vllm_server_port})")
        else:
            print(f"vLLM enabled (mode={args.vllm_mode}, gpu_memory_utilization={args.vllm_gpu_memory_utilization})")

    # Check if this is a PEFT adapter (has adapter_config.json)
    adapter_config_path = os.path.join(args.base_model_dir, "adapter_config.json")
    is_peft_adapter = os.path.exists(adapter_config_path)

    # Also check on Hub if it's not a local path
    if not is_peft_adapter and not os.path.isdir(args.base_model_dir):
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(args.base_model_dir, "adapter_config.json")
            is_peft_adapter = True
        except Exception:
            pass

    if is_peft_adapter:
        # Load as PEFT adapter: first load base model, then adapter, then merge
        print("Detected PEFT adapter, loading base model + adapter and merging...")
        try:
            # Try local first
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
        except FileNotFoundError:
            # Load from Hub
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(args.base_model_dir, "adapter_config.json")
            with open(config_path, "r") as f:
                adapter_config = json.load(f)

        base_model_name = adapter_config.get("base_model_name_or_path")
        print(f"  Base model: {base_model_name}")
        print(f"  Adapter: {args.base_model_dir}")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        # Load and merge adapter
        model = PeftModel.from_pretrained(model, args.base_model_dir)
        model = model.merge_and_unload()
        print("  Adapter merged into base model")
        # Clean up residual PEFT attributes to avoid warnings when loading new adapter
        if hasattr(model, 'peft_config'):
            delattr(model, 'peft_config')

        # Save and push merged ORPO model
        if args.save_merged_orpo:
            # Derive merged model name from input adapter
            if "/" in args.base_model_dir:
                # Hub path like "lamm-mit/orpo-graph_v2" -> "lamm-mit/orpo-graph_v2_merged"
                merged_hub_id = f"{args.base_model_dir}_merged"
                merged_local_dir = os.path.join(args.output_dir, os.path.basename(args.base_model_dir) + "_merged")
            else:
                # Local path
                merged_hub_id = f"{args.hub_model_id}_orpo_merged" if args.hub_model_id else None
                merged_local_dir = f"{args.base_model_dir}_merged"

            # Save locally
            print(f"  Saving merged ORPO model to: {merged_local_dir}")
            model.save_pretrained(merged_local_dir)
            tokenizer.save_pretrained(merged_local_dir)

            # Push to Hub
            if args.push_to_hub and merged_hub_id:
                print(f"  Pushing merged ORPO model to Hub: {merged_hub_id}")
                model.push_to_hub(merged_hub_id, private=(not args.hub_public))
                tokenizer.push_to_hub(merged_hub_id, private=(not args.hub_public))

    else:
        # Load as regular model (already merged or full fine-tune)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_dir,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    model.config.use_cache = False

    # Optionally add special tokens (usually not needed if using existing tokenizer vocabulary)
    if args.add_new_special_tokens:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            print(f"Added {num_added} special tokens, resized embeddings to {len(tokenizer)}")
    else:
        print("Using existing tokenizer vocabulary (no special tokens added)")

    # Attach GRPO LoRA: either resume from checkpoint or create new
    if not args.no_lora:
        if args.resume_grpo_checkpoint:
            # ===== RESUME: Load existing LoRA adapter from checkpoint =====
            print("=" * 60)
            print("RESUMING GRPO TRAINING FROM CHECKPOINT")
            print("=" * 60)
            print(f"  Checkpoint path: {args.resume_grpo_checkpoint}")

            # Check if checkpoint exists (local or Hub)
            is_local = os.path.exists(args.resume_grpo_checkpoint)
            is_hub = False

            if is_local:
                print(f"  Found local checkpoint ✓")
                adapter_config_check = os.path.join(args.resume_grpo_checkpoint, "adapter_config.json")
                if not os.path.exists(adapter_config_check):
                    raise ValueError(f"No adapter_config.json found in {args.resume_grpo_checkpoint}")
                print(f"  Found adapter_config.json ✓")
            else:
                # Try to verify it exists on Hub
                print(f"  Not found locally, checking Hub...")
                try:
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(args.resume_grpo_checkpoint, "adapter_config.json")
                    is_hub = True
                    print(f"  Found on Hub ✓")
                except Exception as e:
                    raise ValueError(f"Checkpoint not found locally or on Hub: {args.resume_grpo_checkpoint}\n  Error: {e}")

            # Load the existing LoRA adapter (works for both local and Hub paths)
            print(f"  Loading LoRA adapter from {'Hub' if is_hub else 'local'}...")
            model = PeftModel.from_pretrained(model, args.resume_grpo_checkpoint, is_trainable=True)
            print(f"  LoRA adapter loaded successfully ✓")
            model.print_trainable_parameters()
            print("=" * 60)
        else:
            # ===== NEW: Create fresh LoRA adapter =====
            #target_modules = [ "q_proj", "k_proj", "v_proj", "o_proj",   "gate_proj", "up_proj", "down_proj", ]
            #target_modules=["embed_tokens", "q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj", "gate_proj"]
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

            peft_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=target_modules,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_cfg)
            model.print_trainable_parameters()
    else:
        print("Training full model (no LoRA)")
        # After merge_and_unload(), parameters have requires_grad=False
        # Enable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # GRPO config
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="no",
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        bf16=use_bf16,
        fp16=(not use_bf16),
        # vLLM options
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode if args.use_vllm else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_server_host=args.vllm_server_host if args.use_vllm else None,
        vllm_server_port=args.vllm_server_port if args.use_vllm else None,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_private_repo=(not args.hub_public) if args.push_to_hub else None,
        remove_unused_columns=False,
        report_to=["wandb"],
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
    )
    print(f"GRPO settings: scale_rewards={args.scale_rewards}, loss_type={args.loss_type}")

    api_client = get_llm_client(api_key=args.judge_api_key, base_url=args.judge_base_url)

    # Validate and print reward weights
    total_weight = (
        args.weight_correctness + args.weight_format + args.weight_graph_utility +
        args.weight_graph_schema +  # NEW
        args.weight_graph_networkx + args.weight_graph_diversity + args.weight_graph_structure
    )
    print(f"Reward weights: correctness={args.weight_correctness}, format={args.weight_format}, graph_utility={args.weight_graph_utility} (sum={total_weight:.2f})")
    # Print optional components if enabled
    optional_parts = []
    if args.weight_graph_schema > 0:
        optional_parts.append(f"graph_schema={args.weight_graph_schema}")
    if args.weight_graph_networkx > 0:
        optional_parts.append(f"graph_networkx={args.weight_graph_networkx}")
    if args.weight_graph_diversity > 0:
        optional_parts.append(f"graph_diversity={args.weight_graph_diversity}")
    if args.weight_graph_structure > 0:
        optional_parts.append(f"graph_structure={args.weight_graph_structure}")
    if optional_parts:
        print(f"  Optional rewards enabled: {', '.join(optional_parts)}")
    if abs(total_weight - 1.0) > 0.01:
        print(f"  Warning: weights sum to {total_weight:.2f}, not 1.0")

    reward_fn = make_reward_function(
        api_client,
        args.judge_model,
        weight_correctness=args.weight_correctness,
        weight_format=args.weight_format,
        weight_graph_utility=args.weight_graph_utility,
        weight_graph_schema=args.weight_graph_schema,  # NEW
        weight_graph_networkx=args.weight_graph_networkx,
        weight_graph_diversity=args.weight_graph_diversity,
        weight_graph_structure=args.weight_graph_structure,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=[reward_fn],
        peft_config=None,  # already PEFT-wrapped
    )

    # Start training
    print("=" * 60)
    if args.resume_from_checkpoint:
        # CRASH RECOVERY: Restore everything (optimizer, scheduler, step count)
        print("CRASH RECOVERY: Resuming from trainer checkpoint")
        print(f"  Checkpoint: {args.resume_from_checkpoint}")
        print(f"  Restoring: optimizer state, scheduler state, step count")
        print("=" * 60)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elif args.resume_grpo_checkpoint:
        # CONTINUE TRAINING: Load LoRA weights, fresh optimizer/scheduler
        print("STARTING TRAINING (with loaded LoRA weights, fresh optimizer/scheduler)")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Epochs: {args.epochs}")
        print("=" * 60)
        trainer.train()
    else:
        # NEW TRAINING: Fresh LoRA, fresh optimizer/scheduler
        print("STARTING TRAINING (new LoRA, fresh optimizer/scheduler)")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Epochs: {args.epochs}")
        print("=" * 60)
        trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing GRPO model to Hub: {args.hub_model_id}")
        final_model = trainer.model
        final_model.push_to_hub(args.hub_model_id, private=(not args.hub_public))
        tokenizer.push_to_hub(args.hub_model_id, private=(not args.hub_public))


if __name__ == "__main__":
    main()
