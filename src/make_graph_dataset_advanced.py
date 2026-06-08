#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_graph_dataset_advanced.py

Build a graph-native reasoning dataset with a *structured, domain-agnostic ontology*.

Schema (per example):

- prompt:            user question (string)
- answer:            gold final answer (string)
- chosen:            full assistant completion with <think>...<graph_json>...</graph_json>...</think> + final answer
- rejected:          weaker/shorter direct answer (no thinking block)
- teacher_graph_json: JSON string from inside <graph_json>...</graph_json> in chosen

Graph JSON schema (domain-general):

{
  "nodes": [
    {
      "id": "SilkFiber",              # short CamelCase identifier, no spaces
      "type": "entity",               # one of: entity, attribute, process, event, outcome, law, claim
      "level": "micro"                # optional, one of: micro, meso, macro
    },
    ...
  ],
  "edges": [
    {
      "source": "SilkFiber",
      "relation": "enables",          # constrained vocabulary (see Relation below)
      "target": "FundamentalFrequency"
    },
    ...
  ]
}

This ontology is intended to work across science, tech, art, humanities, etc.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Literal

from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfFolder
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

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

TRUNC_LEN = 32000

# ====== Graph Schema for validation (advanced, domain-agnostic) ======


class Node(BaseModel):
    """
    Node in the reasoning graph.

    - id:    short CamelCase identifier, no spaces (e.g., "LaserPulse", "ThemeAlienation")
    - type:  semantic role of the node (domain-general)
    - level: optional scale (micro/meso/macro)
    """

    id: str = Field(...)
    type: Literal[
        "entity",
        "attribute",
        "process",
        "event",
        "outcome",
        "law",
        "claim",
    ]
    level: Optional[Literal["micro", "meso", "macro"]] = None


class Edge(BaseModel):
    """
    Edge in the reasoning graph.

    - relation: chosen from a small, reusable vocabulary that works across domains.
    """

    source: str
    relation: Literal[
        "causes",
        "enables",
        "inhibits",
        "modulates",
        "part_of",
        "instance_of",
        "supports",
        "challenges",
        "represents",
        "promotes",
        "violates",
        "constrains",
    ]
    target: str


class GraphJSON(BaseModel):
    nodes: List[Node]
    edges: List[Edge] = Field(default_factory=list)


# ====== Pydantic Response Models for Structured Parsing ======


class QuestionResponse(BaseModel):
    """Structured response for question generation."""
    question: str = Field(..., description="A challenging, standalone question based on the context")


class RejectedAnswerResponse(BaseModel):
    """Structured response for rejected (short) answers."""
    answer: str = Field(..., description="A brief 1-3 sentence answer without reasoning")


class GraphRepairResponse(BaseModel):
    """Response model for graph validation/repair."""
    nodes: List[Node]
    edges: List[Edge] = Field(default_factory=list)
    repair_notes: Optional[str] = Field(None, description="Notes about any repairs made")


def validate_graph_semantics(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extra validation beyond Pydantic structure:
    - Node ids must be unique
    - Edge endpoints must refer to existing node ids
    - Node ids should be "simple" (no spaces)
    Returns cleaned object if valid, else None.
    """
    try:
        graph = GraphJSON(**obj)
    except Exception:
        return None

    ids = [n.id for n in graph.nodes]
    if len(ids) != len(set(ids)):
        return None
    for _id in ids:
        if not isinstance(_id, str) or not _id:
            return None
        if " " in _id:
            return None

    id_set = set(ids)
    for e in graph.edges:
        if e.source not in id_set or e.target not in id_set:
            return None

    # If all checks pass, return original obj
    return obj


# ====== LLM Client ======


def get_llm_client(api_key: str, base_url: Optional[str] = None, timeout: float = 120.0) -> OpenAI:
    params: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        params["base_url"] = base_url
    if timeout:
        params["timeout"] = timeout
    return OpenAI(**params)


# ====== Dataset parsing ======


def parse_dataset_spec(spec: str) -> Tuple[str, Optional[int]]:
    """
    Parse a single dataset spec like "org/dataset[:1000]"
    Returns (dataset_name, num_samples) where num_samples is None for unlimited.
    """
    spec = spec.strip()
    if "[::" in spec or spec.count("[") > 1:
        raise ValueError(f"Invalid dataset spec: {spec}")

    if "[:" in spec and spec.endswith("]"):
        bracket_idx = spec.rfind("[:")
        dataset_name = spec[:bracket_idx]
        num_str = spec[bracket_idx + 2 : -1]
        try:
            num_samples = int(num_str)
        except ValueError:
            raise ValueError(f"Invalid sample count in spec: {spec}")
        return dataset_name, num_samples
    else:
        return spec, None


def parse_datasets_string(datasets_str: str) -> List[Tuple[str, Optional[int]]]:
    """
    Parse the full datasets string like "dataset_a[:1000]|dataset_b[:500]|dataset_c"
    Returns list of (dataset_name, num_samples) tuples.
    """
    if not datasets_str or not datasets_str.strip():
        raise ValueError("--datasets string cannot be empty")

    specs = datasets_str.split("|")
    parsed = []
    for spec in specs:
        spec = spec.strip()
        if spec:
            parsed.append(parse_dataset_spec(spec))

    if not parsed:
        raise ValueError("No valid datasets found in --datasets string")

    return parsed


# ====== Dataset sampling ======


def sample_streamed_dataset(
    dataset_name: str,
    split: str = "train",
    sample_size: Optional[int] = None,
    streaming: bool = True,
    field: str = "text",
) -> Dataset:
    limit_str = f"n={sample_size}" if sample_size else "all"
    print(f"Streaming dataset: {dataset_name} (split={split}, {limit_str})")
    ds_stream = load_dataset(dataset_name, split=split, streaming=streaming)
    it = iter(ds_stream)
    rows = []
    count = 0
    for row in it:
        if sample_size is not None and count >= sample_size:
            break
        if field in row and isinstance(row[field], str):
            rows.append(row)
            count += 1
    ds = Dataset.from_list(rows)
    print(f"  -> Collected {len(ds)} rows")
    return ds


def build_corpus(
    dataset_specs: List[Tuple[str, Optional[int]]],
    text_field: str = "text",
) -> Dataset:
    """
    Build corpus from list of (dataset_name, num_samples) tuples.
    """
    datasets_list = []
    for dataset_name, num_samples in dataset_specs:
        ds = sample_streamed_dataset(dataset_name, sample_size=num_samples, field=text_field)
        if len(ds) > 0:
            datasets_list.append(ds)

    if not datasets_list:
        raise ValueError("No data collected from any dataset")

    if len(datasets_list) == 1:
        combined = datasets_list[0]
    else:
        combined = concatenate_datasets(datasets_list)

    combined = combined.shuffle(seed=42)
    print(f"Final combined corpus size: {len(combined)}")
    return combined


# ====== LLM call with fallback ======


def llm_call(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Optional[str]:
    """
    Try responses.create first, fall back to chat.completions.create.
    Returns the text output or None on failure.
    """
    # Try responses API first
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
        )
        return (resp.output_text or "").strip()
    except Exception:
        pass

    # Fallback to chat completions
    try:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (chat.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[llm_call] both APIs failed: {e}")
        return None


def llm_parse(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_format: type,
) -> Optional[Any]:
    """
    Use OpenAI's responses.parse() for structured Pydantic output.
    Falls back to chat.completions with response_format for older APIs.

    Args:
        client: OpenAI client
        model: Model name
        system_prompt: System prompt
        user_prompt: User prompt
        response_format: Pydantic model class for structured output

    Returns:
        Parsed Pydantic object or None on failure
    """
    # Try responses.parse first (newer API)
    try:
        resp = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=response_format,
        )
        return resp.output_parsed
    except Exception:
        pass

    # Fallback to chat completions with response_format (beta)
    try:
        chat = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
        )
        return chat.choices[0].message.parsed
    except Exception:
        pass

    # Final fallback: manual JSON extraction from regular chat completion
    try:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = chat.choices[0].message.content or ""
        obj = json.loads(content)
        return response_format(**obj)
    except Exception as e:
        print(f"[llm_parse] all methods failed: {e}")
        return None


# ====== Teacher calls ======


def teacher_generate_question(
    client: OpenAI,
    model: str,
    context: str,
    system_prompt: str = "You are a scientist who designs deep, self-contained questions.",
) -> Optional[str]:
    """
    Generate a challenging question using Pydantic structured parsing.
    """
    user_prompt = (
        "Using ONLY the information and style implicit in the following context, write a single challenging, "
        "standalone question. The question should:\n"
        "- not mention the word 'context', 'paper', 'authors', or references,\n"
        "- be answerable by a highly trained expert (in that domain),\n"
        "- be specific, not vague.\n\n"
        f"Context:\n{context}\n\n"
        "Return a JSON object with a 'question' field containing your question."
    )

    # Try structured parsing first
    result = llm_parse(client, model, system_prompt, user_prompt, QuestionResponse)
    if result is not None:
        q = result.question.strip()
        if not q.endswith("?"):
            q += "?"
        return q

    # Fallback to unstructured call
    txt = llm_call(client, model, system_prompt, user_prompt)
    if not txt:
        return None
    # Take first line as question
    q = txt.split("\n")[0].strip()
    if not q.endswith("?"):
        q += "?"
    return q


def teacher_generate_structured_answer(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
) -> Optional[str]:
    """
    Ask teacher to produce:
    <think>
      <brainstorm>...</brainstorm>
      <graph>...</graph>
      <graph_json>{nodes:[...],edges:[...]}</graph_json>
      <patterns>...</patterns>
      <synthesis>...</synthesis>
    </think>
    Final answer text...
    """
    sys = (
        "You are an expert in mechanistic, graph-based reasoning across ALL domains: "
        "science, engineering, software, art, humanities, social sciences, etc. "
        "You always reason using a typed, multi-level graph as a latent world model."
    )

    user = f"""
Given the following context and question, answer in TWO phases.

FIRST, produce a structured internal reasoning trace using EXACTLY this template:

{THINK_START}
{BRAINSTORM_START}
Freely explore the problem space. Identify relevant concepts, mechanisms, actors, events, themes, and outcomes.
Organize your thinking across three scales:
- micro: concrete components, individuals, local actions or details (e.g., cells, functions, characters, transactions),
- meso: interactions, subsystems, scenes, workflows, social groups (e.g., pathways, services, rituals, committees),
- macro: large-scale systems, institutions, themes, goals, long-term outcomes (e.g., ecosystems, markets, historical trends, narrative themes).
Think broadly and creatively before structuring anything. Note key variables and how they might interact, in any domain.
{BRAINSTORM_END}

{GRAPH_START}
Now, create a domain-agnostic conceptual graph *in words* (NOT JSON yet):

1. List 8–20 core nodes. For each node, specify:
   - a short CamelCase id with no spaces (e.g., SilkFiber, LaserPulse, ThemeAlienation, UserSignupFlow),
   - a type chosen from: entity, attribute, process, event, outcome, law, claim,
   - an optional level chosen from: micro, meso, macro.

   Example lines:
   - SilkFiber [entity, micro]
   - FundamentalFrequency [outcome, meso]
   - ThemeAlienation [attribute, macro]
   - UserSignupFlow [process, meso]

2. Then, describe the directed relationships among these nodes using ONLY the following relation verbs:
   causes, enables, inhibits, modulates, part_of, instance_of, supports, challenges, represents, promotes, violates, constrains.

   Write them as simple statements like:
   - SilkFiber constrains Tension
   - Tension increases FundamentalFrequency
   - ThemeAlienation represents SocialFragmentation
   - CounterArgument challenges ThesisClaim

These relationships should be fundamentally causal, structural, argumentative, or symbolic – not just loose associations.
{GRAPH_END}

{GRAPH_JSON_START}
Now provide a STRICT JSON object with keys "nodes" and "edges" ONLY, following this schema:

- Each node must have:
  - "id": a short CamelCase string without spaces, e.g. "SilkFiber", "LaserPulse", "ThemeAlienation".
  - "type": one of ["entity", "attribute", "process", "event", "outcome", "law", "claim"].
  - "level": optional, one of ["micro", "meso", "macro"].

- Each edge must have:
  - "source": a node id (string),
  - "relation": one of ["causes", "enables", "inhibits", "modulates", "part_of", "instance_of",
                       "supports", "challenges", "represents", "promotes", "violates", "constrains"],
  - "target": a node id (string).

- Node ids must be unique. All edges must refer only to existing node ids.
- Do NOT include any extra keys or comments.

Example (just an illustration, DO NOT reuse these exact ids unless appropriate):
{{
  "nodes": [
    {{"id": "SilkFiber", "type": "entity", "level": "micro"}},
    {{"id": "Tension", "type": "attribute", "level": "micro"}},
    {{"id": "FundamentalFrequency", "type": "outcome", "level": "meso"}}
  ],
  "edges": [
    {{"source": "SilkFiber", "relation": "constrains", "target": "Tension"}},
    {{"source": "Tension", "relation": "enables", "target": "FundamentalFrequency"}}
  ]
}}
{GRAPH_JSON_END}

{PATTERNS_START}
Based on the graph, describe 2–4 abstract patterns or laws that summarize its structure.

Guidelines:
- Use node ids and the allowed relation verbs only.
- Prefer multi-step chains like:
  - A causes B causes C
  - A enables B, which inhibits C
  - MicroProcessX modulates MacroOutcomeY
- When relevant, express simple quantitative or comparative relations (e.g., TempRes ∝ 1/PulseDuration),
  but keep them tied to the graph nodes.

Avoid storytelling here; treat this as compressed graph structure and causal/argument patterns.
{PATTERNS_END}

{SYNTHESIS_START}
Integrate everything above into a coherent picture that answers the question.

Explain, in a domain-agnostic but concrete way:
- How micro-level elements (components, individuals, local events) interact at the meso level (processes, systems, scenes)
  to produce macro-level outcomes (themes, performance, social effects, historical trajectories).
- How the key causal / structural / argumentative / symbolic relationships in the graph jointly explain the phenomenon
  or support an answer to the question.

This should feel like a mechanistic explanation tailored to the domain of the question, whether that's physics, software,
art, ethics, history, or something else.
{SYNTHESIS_END}
{THINK_END}

SECOND, on a NEW line after {THINK_END}, write a comprehensive, detailed final answer to the question.

Your final answer should:
- Be thorough and well-structured (multiple paragraphs if appropriate),
- Explain the underlying mechanisms, structures, or arguments,
- Reference concepts from your graph but ALWAYS use natural language (e.g., write "silk fiber" NOT "SilkFiber", write "fundamental frequency" NOT "FundamentalFrequency"),
- Provide specific details, examples, or quantitative insights where relevant,
- Be suitable for an expert audience seeking deep understanding.

IMPORTANT: The final answer must be written in proper English prose. Do NOT use CamelCase identifiers, graph notation, or any technical markup from the thinking section. The answer should read naturally as if written for a scientific paper or textbook.

Do NOT be brief. Provide a complete, educational response that fully addresses the question.

Question:
{question}

Context:
{context}
"""
    return llm_call(client, model, sys, user)


def teacher_generate_rejected(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
) -> Optional[str]:
    """
    Short, direct answer with no thinking or graph structure.
    Uses Pydantic structured parsing for reliable extraction.
    """
    sys = "You are a hurried scientist who gives very short, shallow answers."
    user = f"""
Answer the following question in 1–3 sentences WITHOUT showing your reasoning, and WITHOUT any special tokens.

Question:
{question}

Context:
{context}

Return a JSON object with an 'answer' field containing your brief answer.
"""

    # Try structured parsing first
    result = llm_parse(client, model, sys, user, RejectedAnswerResponse)
    if result is not None:
        return result.answer.strip()

    # Fallback to unstructured call
    return llm_call(client, model, sys, user)


def validate_and_repair_graph(
    client: OpenAI,
    model: str,
    raw_json_str: str,
) -> Optional[GraphJSON]:
    """
    Validate raw JSON against GraphJSON schema and attempt repair if invalid.
    Uses Pydantic structured parsing for reliable output.

    Args:
        client: OpenAI client
        model: Model name for repair
        raw_json_str: Raw JSON string extracted from model output

    Returns:
        Validated GraphJSON object or None if unrecoverable
    """
    # First, try direct Pydantic validation
    try:
        obj = json.loads(raw_json_str)
        graph = GraphJSON(**obj)

        # Additional semantic validation
        ids = [n.id for n in graph.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate node IDs")
        for _id in ids:
            if " " in _id:
                raise ValueError(f"Space in node ID: {_id}")

        id_set = set(ids)
        for e in graph.edges:
            if e.source not in id_set or e.target not in id_set:
                raise ValueError(f"Edge references non-existent node: {e.source} -> {e.target}")

        return graph
    except Exception:
        pass

    # Attempt repair using LLM with structured output
    sys = (
        "You are a graph schema validator. Fix the provided graph JSON to comply with the schema. "
        "Node types must be one of: entity, attribute, process, event, outcome, law, claim. "
        "Edge relations must be one of: causes, enables, inhibits, modulates, part_of, instance_of, "
        "supports, challenges, represents, promotes, violates, constrains. "
        "Node IDs must be CamelCase with no spaces. All edge endpoints must reference existing nodes."
    )
    user = f"""
Fix this graph JSON to comply with the schema. Preserve the semantic content but correct any schema violations.

Invalid JSON:
{raw_json_str}

Return a valid graph with 'nodes' and 'edges' arrays following the schema.
"""

    # Use structured parsing for repair
    result = llm_parse(client, model, sys, user, GraphRepairResponse)
    if result is not None:
        try:
            # Convert repair response to GraphJSON
            graph = GraphJSON(nodes=result.nodes, edges=result.edges)

            # Validate semantics
            ids = [n.id for n in graph.nodes]
            if len(ids) != len(set(ids)):
                return None
            id_set = set(ids)
            for e in graph.edges:
                if e.source not in id_set or e.target not in id_set:
                    return None

            return graph
        except Exception:
            return None

    return None


# ====== Parsing helpers ======


def find_tag_block(s: str, start_tag: str, end_tag: str) -> Optional[str]:
    i = s.find(start_tag)
    if i == -1:
        return None
    j = s.find(end_tag, i + len(start_tag))
    if j == -1:
        return None
    return s[i + len(start_tag) : j].strip()


def extract_graph_json_block(
    chosen: str,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """
    Extract inner JSON from <graph_json>...</graph_json>, validate it against
    the advanced schema + semantic checks, and return a canonical JSON string.

    If client and model are provided, attempts to repair invalid graphs using
    Pydantic structured parsing.

    Args:
        chosen: Full model output containing graph_json block
        client: Optional OpenAI client for graph repair
        model: Optional model name for graph repair

    Returns:
        Canonical JSON string if valid (or repaired), None otherwise
    """
    inner = find_tag_block(chosen, GRAPH_JSON_START, GRAPH_JSON_END)
    if not inner:
        return None

    # If we have client and model, use the repair-capable validation
    if client is not None and model is not None:
        graph = validate_and_repair_graph(client, model, inner)
        if graph is not None:
            return graph.model_dump_json()
        return None

    # Fallback: simple validation without repair
    try:
        obj = json.loads(inner)
    except Exception:
        return None

    validated = validate_graph_semantics(obj)
    if not validated:
        return None

    # Return normalized JSON string
    try:
        return json.dumps(validated, ensure_ascii=False)
    except Exception:
        return None


def extract_final_answer(chosen: str) -> str:
    """
    Everything after </think> is the final answer.
    """
    idx = chosen.find(THINK_END)
    if idx == -1:
        return chosen.strip()
    return chosen[idx + len(THINK_END) :].strip()


# ====== Dataset builder ======


def build_graph_reasoning_dataset(
    corpus: Dataset,
    teacher_client: OpenAI,
    reject_client: OpenAI,
    teacher_model: str,
    reject_model: str,
    num_examples: int,
    output_path: str,
    save_steps: int = 100,
    resume: bool = False,
    push_to_hub: bool = False,
    output_repo: Optional[str] = None,
    hub_public: bool = False,
    max_context_tokens: int = 2048,  # currently not used for hard truncation, only approx by chars
) -> Dataset:
    rows: List[Dict[str, Any]] = []

    # Resume from existing data if requested
    if resume:
        # Priority 1: Local file
        if os.path.exists(output_path):
            try:
                existing_ds = load_dataset("json", data_files=output_path, split="train")
                rows = [dict(row) for row in existing_ds]
                print(f"Resuming from local file {output_path} with {len(rows)} existing examples")
            except Exception as e:
                print(f"Could not load local file: {e}")

        # Priority 2: If no local file, try to download from Hub
        if len(rows) == 0 and output_repo:
            try:
                print(f"Local file not found, checking Hub for {output_repo}...")
                existing_ds = load_dataset(output_repo, split="train")
                rows = [dict(row) for row in existing_ds]
                print(f"Downloaded {len(rows)} existing examples from Hub: {output_repo}")
                # Save to local file for faster future resumes
                if len(rows) > 0:
                    Dataset.from_list(rows).to_json(output_path, lines=True)
                    print(f"Saved Hub data to local file: {output_path}")
            except Exception as e:
                print(f"Could not load from Hub: {e}")
                print("Starting fresh...")

    # Check if we already have enough examples
    if len(rows) >= num_examples:
        print(f"Already have {len(rows)} examples (target: {num_examples}). Nothing to generate.")
        return Dataset.from_list(rows)

    remaining = num_examples - len(rows)
    print(f"Building up to {num_examples} examples (have {len(rows)}, need {remaining} more)...")
    print(f"  Teacher model (questions + chosen): {teacher_model}")
    print(f"  Reject model (rejected answers): {reject_model}")
    print(f"  Saving every {save_steps} examples to {output_path}")
    if push_to_hub and output_repo:
        print(f"  Pushing to Hub: {output_repo}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # sample random indices to avoid ordering bias
    indices = list(range(len(corpus)))
    random.shuffle(indices)

    pbar = tqdm(total=num_examples, initial=len(rows), desc="Building examples", unit="ex")
    skipped = 0
    last_save_count = len(rows)

    for idx in indices:
        if len(rows) >= num_examples:
            break
        row = corpus[idx]
        context = row.get("text", "")
        if not isinstance(context, str) or len(context.strip()) < 200:
            skipped += 1
            pbar.set_postfix({"skipped": skipped})
            continue
        if len(context) > TRUNC_LEN:
            context = context[:TRUNC_LEN]  # rough trunc

        q = teacher_generate_question(teacher_client, teacher_model, context)
        if not q:
            skipped += 1
            pbar.set_postfix({"skipped": skipped})
            continue

        structured = teacher_generate_structured_answer(teacher_client, teacher_model, q, context)
        if not structured:
            skipped += 1
            pbar.set_postfix({"skipped": skipped})
            continue

        # Extract and validate graph JSON with automatic repair if needed
        graph_json_str = extract_graph_json_block(
            structured,
            client=teacher_client,
            model=teacher_model,
        )
        if not graph_json_str:
            # skip examples where teacher couldn't produce valid graph_json
            skipped += 1
            pbar.set_postfix({"skipped": skipped})
            continue

        final_answer = extract_final_answer(structured)
        if not final_answer:
            # skip if answer missing
            skipped += 1
            pbar.set_postfix({"skipped": skipped})
            continue

        rejected = teacher_generate_rejected(reject_client, reject_model, q, context)
        if not rejected:
            skipped += 1
            pbar.set_postfix({"skipped": skipped})
            continue

        rows.append(
            {
                "prompt": q.strip(),
                "answer": final_answer.strip(),
                "chosen": structured.strip(),
                "rejected": rejected.strip(),
                "teacher_graph_json": graph_json_str,
            }
        )
        pbar.update(1)
        pbar.set_postfix({"skipped": skipped})

        # Periodic save
        if len(rows) - last_save_count >= save_steps:
            last_save_count = len(rows)
            ds_checkpoint = Dataset.from_list(rows)
            ds_checkpoint.to_json(output_path, lines=True)
            tqdm.write(f"  [checkpoint] Saved {len(rows)} examples to {output_path}")
            if push_to_hub and output_repo:
                try:
                    ds_checkpoint.push_to_hub(output_repo, private=(not hub_public))
                    tqdm.write(f"  [checkpoint] Pushed {len(rows)} examples to {output_repo}")
                except Exception as e:
                    tqdm.write(f"  [checkpoint] Push failed: {e}")

    pbar.close()
    print(f"Done. Created {len(rows)} examples (skipped {skipped}).")
    return Dataset.from_list(rows)


# ====== Main ======


def main():
    parser = argparse.ArgumentParser(description="Build advanced graph-native reasoning dataset.")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help='Dataset specs: "dataset_a[:1000]|dataset_b[:500]|dataset_c" (| separated, [:N] for sample limit)',
    )
    parser.add_argument("--num_examples", type=int, default=500)

    parser.add_argument("--teacher_model", type=str, required=True, help="Model for questions and chosen answers")
    parser.add_argument(
        "--reject_model",
        type=str,
        default=None,
        help="Model for rejected answers (default: same as teacher_model)",
    )
    parser.add_argument("--teacher_api_key", type=str, required=True)
    parser.add_argument("--teacher_base_url", type=str, default=None)
    parser.add_argument(
        "--reject_api_key",
        type=str,
        default=None,
        help="API key for reject model (default: same as teacher_api_key)",
    )
    parser.add_argument(
        "--reject_base_url",
        type=str,
        default=None,
        help="Base URL for reject model (default: same as teacher_base_url)",
    )

    parser.add_argument("--output_path", type=str, default="./graph_reasoning_advanced.jsonl")
    parser.add_argument("--save_steps", type=int, default=100, help="Save dataset every N examples (default: 100)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output_path if it exists")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--output_repo", type=str, default=None)
    parser.add_argument("--hub_public", action="store_true", help="Make Hub repo public (default: private)")
    parser.add_argument("--hf_token", type=str, default=None)

    args = parser.parse_args()

    if args.hf_token:
        HfFolder.save_token(args.hf_token)

    # Parse datasets string
    dataset_specs = parse_datasets_string(args.datasets)
    print(f"Parsed {len(dataset_specs)} dataset(s):")
    for name, n in dataset_specs:
        print(f"  - {name} [{n if n else 'all'}]")

    corpus = build_corpus(dataset_specs=dataset_specs)

    teacher_client = get_llm_client(api_key=args.teacher_api_key, base_url=args.teacher_base_url)

    # Use reject_model if specified, otherwise fall back to teacher_model
    reject_model = args.reject_model if args.reject_model else args.teacher_model

    # Create separate client for reject model if different credentials provided
    reject_api_key = args.reject_api_key if args.reject_api_key else args.teacher_api_key
    reject_base_url = args.reject_base_url if args.reject_base_url else args.teacher_base_url

    if reject_api_key != args.teacher_api_key or reject_base_url != args.teacher_base_url:
        reject_client = get_llm_client(api_key=reject_api_key, base_url=reject_base_url)
    else:
        reject_client = teacher_client

    # Validate push_to_hub requirements
    if args.push_to_hub and not args.output_repo:
        raise ValueError("--output_repo must be set if --push_to_hub is used")

    ds = build_graph_reasoning_dataset(
        corpus=corpus,
        teacher_client=teacher_client,
        reject_client=reject_client,
        teacher_model=args.teacher_model,
        reject_model=reject_model,
        num_examples=args.num_examples,
        output_path=args.output_path,
        save_steps=args.save_steps,
        resume=args.resume,
        push_to_hub=args.push_to_hub,
        output_repo=args.output_repo,
        hub_public=args.hub_public,
    )

    # Final save
    ds.to_json(args.output_path, lines=True)
    print(f"Final dataset saved to {args.output_path}")

    # Final push to Hub
    if args.push_to_hub:
        print(f"Final push to Hub: {args.output_repo}")
        ds.push_to_hub(args.output_repo, private=(not args.hub_public))


if __name__ == "__main__":
    main()
