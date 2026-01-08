#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_graph_dataset.py

Build a graph-native reasoning dataset with this schema:

- prompt:            user question (string)
- answer:            gold final answer (string)
- chosen:            full assistant completion with <|thinking|> ... graph ... patterns ... </|thinking|> + final answer
- rejected:          weaker/shorter direct answer (no thinking block)
- teacher_graph_json: JSON string from inside <|graph_json|>...</|graph_json|> in chosen

Dataset format: "dataset_name[:num_samples]|dataset_name[:num_samples]|..."
  - Each dataset separated by |
  - Optional [:N] suffix to limit samples (default: all)
  - Examples:
    - "my-org/dataset-a[:5000]|my-org/dataset-b[:200]"
    - "huggingface/dataset"  (no limit)

Model options:
  --teacher_model       Model for generating questions and chosen (structured) answers
  --teacher_api_key     API key for teacher model
  --teacher_base_url    Optional base URL for teacher model API
  --reject_model        Model for generating rejected (shallow) answers (default: teacher_model)
  --reject_api_key      API key for reject model (default: teacher_api_key)
  --reject_base_url     Optional base URL for reject model API (default: teacher_base_url)

"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

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

# ----- Graph Schema for validation -----

class Node(BaseModel):
    id: str = Field(...)


class Edge(BaseModel):
    source: str
    relation: Optional[str] = None
    target: str


class GraphJSON(BaseModel):
    nodes: List[Node]
    edges: List[Edge] = Field(default_factory=list)


# ----- LLM Client -----

def get_llm_client(api_key: str, base_url: Optional[str] = None, timeout: float = 120.0) -> OpenAI:
    params: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        params["base_url"] = base_url
    if timeout:
        params["timeout"] = timeout
    return OpenAI(**params)


# ----- Dataset parsing -----

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


# ----- Dataset sampling -----

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


# ----- LLM call with fallback -----

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


# ----- Teacher calls -----

def teacher_generate_question(
    client: OpenAI,
    model: str,
    context: str,
    system_prompt: str = "You are a scientist who designs deep, self-contained questions.",
) -> Optional[str]:
    user_prompt = (
        "Using ONLY the information and style implicit in the following context, write a single challenging, "
        "standalone scientific or engineering question. The question should:\n"
        "- not mention the word 'context', 'paper', 'authors', or references,\n"
        "- be answerable by a highly trained expert,\n"
        "- be specific, not vague.\n\n"
        f"Context:\n{context}\n\nQuestion:"
    )
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
    <brainstorm>...free exploration...</brainstorm>
    <graph>...text edges...</graph>
    <graph_json>{"nodes":[...],"edges":[...]}</graph_json>
    <patterns>...abstract patterns...</patterns>
    <synthesis>...integrate all insights...</synthesis>
    </think>
    Final answer text...
    """
    sys = (
        "You are a materials and mechanistic reasoning expert. You always reason using a graph-based latent structure."
    )
    user = f"""
Given the following context and question, answer in TWO phases.

FIRST, produce a structured internal reasoning trace using EXACTLY this template:

{THINK_START}
{BRAINSTORM_START}
Freely explore the problem space. What concepts, mechanisms, or phenomena are relevant?
What hypotheses come to mind? What are the key variables and how might they interact?
Think broadly and creatively before structuring.
{BRAINSTORM_END}

{GRAPH_START}
Based on your brainstorming, list the core entities (concepts, variables, materials, phenomena).
Informally describe directed relationships between them in text.
{GRAPH_END}

{GRAPH_JSON_START}
Provide a STRICT JSON object with keys "nodes" and "edges" ONLY. For example:
{{
  "nodes": [{{"id": "A"}}, {{"id": "B"}}],
  "edges": [{{"source": "A", "relation": "influences", "target": "B"}}]
}}
Nodes should have simple string ids. Edges must refer to nodes by id. Do not add extra keys.
{GRAPH_JSON_END}

{PATTERNS_START}
Describe one or two abstract patterns or laws that summarize the graph structure (use symbols like →, ↑, ↓, ∝ if helpful).
{PATTERNS_END}

{SYNTHESIS_START}
Integrate everything above: the brainstormed ideas, the graph relationships, and the patterns.
What is the coherent picture that emerges? What is the key insight that answers the question?
{SYNTHESIS_END}
{THINK_END}

SECOND, on a NEW line after {THINK_END}, write a comprehensive, detailed final answer to the question.

Your answer should:
- Be thorough and well-structured (multiple paragraphs if appropriate)
- Explain the underlying mechanisms and principles
- Reference the key relationships you identified in your reasoning
- Provide specific details, examples, or quantitative insights where relevant
- Be suitable for an expert audience seeking deep understanding

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
    """
    sys = "You are a hurried scientist who gives very short, shallow answers."
    user = f"""
Answer the following question in 1–3 sentences WITHOUT showing your reasoning, and WITHOUT any special tokens.

Question:
{question}

Context:
{context}
"""
    return llm_call(client, model, sys, user)


# ----- Parsing helpers -----

def find_tag_block(s: str, start_tag: str, end_tag: str) -> Optional[str]:
    i = s.find(start_tag)
    if i == -1:
        return None
    j = s.find(end_tag, i + len(start_tag))
    if j == -1:
        return None
    return s[i + len(start_tag) : j].strip()


def extract_graph_json_block(chosen: str) -> Optional[str]:
    inner = find_tag_block(chosen, GRAPH_JSON_START, GRAPH_JSON_END)
    if not inner:
        return None
    try:
        # Validate via GraphJSON schema; keep original string if valid
        obj = json.loads(inner)
        _ = GraphJSON(**obj)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return None


def extract_final_answer(chosen: str) -> str:
    """
    Everything after </thinking> is the final answer.
    """
    idx = chosen.find(THINK_END)
    if idx == -1:
        return chosen.strip()
    return chosen[idx + len(THINK_END) :].strip()


# ----- Dataset builder -----

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
    max_context_tokens: int = 2048,
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

    # We'll just use raw text (no token truncation in this script),
    # but we can limit context length by character count approx.
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

        graph_json_str = extract_graph_json_block(structured)
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


# ----- Main -----

def main():
    parser = argparse.ArgumentParser(description="Build graph-native reasoning dataset.")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help='Dataset specs: "dataset_a[:1000]|dataset_b[:500]|dataset_c" (| separated, [:N] for sample limit)',
    )
    parser.add_argument("--num_examples", type=int, default=500)

    parser.add_argument("--teacher_model", type=str, required=True, help="Model for questions and chosen answers")
    parser.add_argument("--reject_model", type=str, default=None, help="Model for rejected answers (default: same as teacher_model)")
    parser.add_argument("--teacher_api_key", type=str, required=True)
    parser.add_argument("--teacher_base_url", type=str, default=None)
    parser.add_argument("--reject_api_key", type=str, default=None, help="API key for reject model (default: same as teacher_api_key)")
    parser.add_argument("--reject_base_url", type=str, default=None, help="Base URL for reject model (default: same as teacher_base_url)")

    parser.add_argument("--output_path", type=str, default="./graph_reasoning_v3.jsonl")
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
