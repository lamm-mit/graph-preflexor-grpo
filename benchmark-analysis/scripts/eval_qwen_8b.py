"""
eval_qwen_8b.py
Qwen3-8B — Benchmark Inference Script

Paper: Graph-Structured Reinforcement Learning for Scientific Hypothesis Generation in Materials Design
Authors: Subhadeep Pal, Markus J. Buehler (MIT)

Model  : Qwen/Qwen3-8B
Scale  : 8B
Base   : —

Description:
    Runs Qwen3-8B on the 100-question open-ended benchmark
    (data/benchmark/benchmark_questions.jsonl) using a locally hosted vLLM server.
    Outputs one JSONL record per question to data/results/.

Usage:
    # Start vLLM server first:
    #   vllm serve Qwen/Qwen3-8B --max-model-len 32768 --port 8000
    python eval_qwen_8b.py \
        --dataset ../data/benchmark/benchmark_questions.jsonl \
        --outfile ../data/results/qwen_8b_results.jsonl

Requirements:
    pip install openai transformers tqdm vllm
"""

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time
from tqdm import tqdm
import re

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
MODEL       = "Qwen/Qwen3-8B"
DATASET     = "../data/benchmark/benchmark_questions.jsonl"
OUTFILE     = "qwen_8b_results.jsonl"
MAX_TOKENS  = 32768
TEMPERATURE = 0.6
TOP_P       = 0.95
TOP_K       = 20
MIN_P       = 0.0
MAX_WORKERS = 16

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy", timeout=None)

# ------------------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------------------
dataset = []
with open(DATASET) as f:
    for line in f:
        line = line.strip()
        if line:
            dataset.append(json.loads(line))

print(f"Loaded {len(dataset)} questions")

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def get_thinking(full_output):
    if "<think>" in full_output and "</think>" in full_output:
        return full_output.split("<think>")[1].split("</think>")[0].strip()
    return ""

def get_output(full_output):
    if "</think>" in full_output:
        return full_output.split("</think>")[1].strip()
    return full_output.strip()


# ------------------------------------------------------------------------------
# Prompt formatter
# ------------------------------------------------------------------------------
def format_prompt(row):
    context  = row.get("context", "")
    question = row["question"]
    prompt   = f"{question}"
    return prompt

# ------------------------------------------------------------------------------
# Per-question worker
# ------------------------------------------------------------------------------
def evaluate_row(args):
    i, row = args
    prompt = format_prompt(row)

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt},   # no system message
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            extra_body={
                "top_p": TOP_P,
                "top_k": TOP_K,
                # "min_p": MIN_P,
            },
            stream=False,
        )
        raw_output        = response.choices[0].message.content
        completion_tokens = response.usage.completion_tokens
        error             = ""
    except Exception as e:
        raw_output        = ""
        completion_tokens = 0
        error             = f"{type(e).__name__}: {e}"

    elapsed = time.time() - start

    return {
        "id":                row.get("id", i),
        "doi":               row.get("doi", ""),
        "title":             row.get("title", ""),
        "question_type":     row.get("question_type", ""),
        "question":          row.get("question", ""),
        "raw_output":        raw_output,
        "thinking":          get_thinking(raw_output),
        "llm_output":        get_thinking(raw_output),
        "error":             error,
        "time_sec":          round(elapsed, 2),
        "completion_tokens": completion_tokens,
        "tokens_per_sec":    round(completion_tokens / elapsed, 1) if elapsed > 0 else 0,
    }

# ------------------------------------------------------------------------------
# Parallel eval loop
# ------------------------------------------------------------------------------
results = []
n    = len(dataset)
rows = [(i, row) for i, row in enumerate(dataset)]

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(evaluate_row, args): args[0] for args in rows}

    for future in tqdm(as_completed(futures), total=n):
        result = future.result()
        results.append(result)

        print(f"[{result['id']+1}/{n}] {result['question_type']}")
        print(f"  DOI      : {result['doi']}")
        print(f"  Question : {result['question'][:80]}...")
        print(f"  Output   : {result['llm_output'][:100]}...")
        print(f"  Time     : {result['time_sec']}s | Tokens: {result['completion_tokens']}")
        print("-" * 60)

        with open(OUTFILE, "a") as f:
            f.write(json.dumps(result) + "\n")

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
results.sort(key=lambda x: x["id"])
total_time = sum(r["time_sec"] for r in results)
errors     = [r for r in results if r["error"]]

import pandas as pd
df = pd.DataFrame(results)

print(f"\n{'='*60}")
print(f"Total Qs         : {len(results)}")
print(f"Errors           : {len(errors)}")
print(f"Total time       : {round(total_time, 2)}s")
print(f"Avg / question   : {round(total_time / len(results), 2)}s")

if "tag" in df.columns:
    print(f"\n--- By Tag ---")
    print(df.groupby("tag").size().to_string())
print(f"\nResults saved : {OUTFILE}")