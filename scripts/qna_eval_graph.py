from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, re
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
MODEL       = "lamm-mit/Graph-Preflexor-8b_12292025"
DATASET     = "/home/palsubhadeep/Github/Reasoning_comparison/datasets/research_paper/final_dataset_with_answers.json"
OUTFILE     = "QnA_graph_results.jsonl"
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

dataset = dataset[:100]
print(f"Loaded {len(dataset)} questions")

# ------------------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert scientist and materials researcher. "
    "You will be given a scientific context and a question about it. "
    "Reason carefully and provide a detailed, accurate answer grounded in the provided context. "
    "Support your reasoning with specific evidence from the text."
)

# ------------------------------------------------------------------------------
# Helpers — extract Graph-PRefLexOR reasoning tags
# ------------------------------------------------------------------------------
def extract_tag(text, tag):
    """Extract content between <tag>...</tag>, returns '' if not found."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match   = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def get_thinking(output):
    """Concatenate all Graph-PRefLexOR reasoning stages."""
    parts = []
    for tag in ["brainstorm", "graph", "graph_json", "patterns"]:
        content = extract_tag(output, tag)
        if content:
            parts.append(f"<{tag}>\n{content}\n</{tag}>")
    return "\n\n".join(parts)

def get_final_output(output):
    """Extract the synthesis/final answer, falling back to full output."""
    synthesis = extract_tag(output, "synthesis")
    if synthesis:
        return synthesis
    # if no <synthesis> tag, strip all known thinking tags and return remainder
    cleaned = re.sub(
        r"<(brainstorm|graph|graph_json|patterns)>.*?</(brainstorm|graph|graph_json|patterns)>",
        "", output, flags=re.DOTALL
    ).strip()
    return cleaned if cleaned else output

# ------------------------------------------------------------------------------
# Prompt formatter
# ------------------------------------------------------------------------------
def format_prompt(row):
    context  = row.get("context", "")
    question = row["question"]
    tag      = row.get("Tag", "")
    doi      = row.get("doi", "")
    prompt   = f"{context}\n\n{question}"
    return prompt, question, tag, doi

# ------------------------------------------------------------------------------
# Per-question worker
# ------------------------------------------------------------------------------
def evaluate_row(args):
    i, row = args
    prompt, question, tag, doi = format_prompt(row)

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            extra_body={
                "top_p": TOP_P,
                "top_k": TOP_K,
                "min_p": MIN_P,
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

    thinking   = get_thinking(raw_output)
    llm_output = get_final_output(raw_output)

    return {
        "id":                i,
        "doi":               doi,
        "tag":               tag,
        "title":             row.get("title", ""),
        "question":          question,
        "accepted_answer":   row.get("accepted_answer", ""),
        "rejected_answer":   row.get("rejected_answer", ""),
        "graph_json":        row.get("graph_json", ""),
        "raw_output":        raw_output,
        "llm_output":        llm_output,
        "thinking":          thinking,
        "has_brainstorm":    bool(extract_tag(raw_output, "brainstorm")),
        "has_graph":         bool(extract_tag(raw_output, "graph")),
        "has_graph_json":    bool(extract_tag(raw_output, "graph_json")),
        "has_synthesis":     bool(extract_tag(raw_output, "synthesis")),
        "error":             error,
        "time_sec":          round(elapsed, 2),
        "completion_tokens": completion_tokens,
        "tokens_per_sec":    round(completion_tokens / elapsed, 1) if elapsed > 0 else 0,
    }

# ------------------------------------------------------------------------------
# Parallel eval loop
# ------------------------------------------------------------------------------
results = []
n       = len(dataset)
rows    = [(i, row) for i, row in enumerate(dataset)]

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(evaluate_row, args): args[0] for args in rows}

    for future in tqdm(as_completed(futures), total=n):
        result = future.result()
        results.append(result)

        tags_present = " | ".join(
            t for t, v in [
                ("<brainstorm>", result["has_brainstorm"]),
                ("<graph>",      result["has_graph"]),
                ("<graph_json>", result["has_graph_json"]),
                ("<synthesis>",  result["has_synthesis"]),
            ] if v
        )
        print(f"[{result['id']+1}/{n}] {result['tag']}")
        print(f"  DOI      : {result['doi']}")
        print(f"  Question : {result['question'][:80]}...")
        print(f"  Output   : {result['llm_output'][:100]}...")
        print(f"  Tags     : {tags_present or 'none'}")
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
print(f"Has brainstorm   : {df['has_brainstorm'].sum()}/{len(df)}")
print(f"Has graph        : {df['has_graph'].sum()}/{len(df)}")
print(f"Has graph_json   : {df['has_graph_json'].sum()}/{len(df)}")
print(f"Has synthesis    : {df['has_synthesis'].sum()}/{len(df)}")
if "tag" in df.columns:
    print(f"\n--- By Tag ---")
    print(df.groupby("tag").size().to_string())
print(f"\nResults saved : {OUTFILE}")