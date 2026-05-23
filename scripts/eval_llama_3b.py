"""
eval_llama_3b.py
Llama-3.2-3B-Instruct — Benchmark Inference Script

Paper: Graph-Structured Reinforcement Learning for Scientific Hypothesis Generation in Materials Design
Authors: Subhadeep Pal, Markus J. Buehler (MIT)

Model  : meta-llama/Llama-3.2-3B-Instruct
Scale  : 3B
Base   : —

Description:
    Runs Llama-3.2-3B-Instruct on the 100-question open-ended benchmark
    (data/benchmark/benchmark_questions.json) using a locally hosted vLLM server.
    Outputs one JSONL record per question to data/results/.

Usage:
    # Start vLLM server first:
    #   vllm serve meta-llama/Llama-3.2-3B-Instruct --max-model-len 32768 --port 8000
    python eval_llama_3b.py \
        --dataset ../data/benchmark/benchmark_questions.json \
        --outfile ../data/results/llama_3b_results.jsonl

Requirements:
    pip install openai transformers tqdm vllm
"""

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
MODEL       = "meta-llama/Llama-3.2-3B-Instruct"
DATASET     = "../data/results/question_p2.jsonl"
OUTFILE     = "llama_3b_results_p2.jsonl"
MAX_TOKENS  = 32768
TEMPERATURE = 0.6
TOP_P       = 0.9
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
        "paper_title":       row.get("paper_title", ""),
        "question_type":     row.get("question_type", ""),
        "question":          row.get("question", ""),
        "raw_output":        raw_output,
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
        print(f"  Output   : {result['raw_output'][:100]}...")
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