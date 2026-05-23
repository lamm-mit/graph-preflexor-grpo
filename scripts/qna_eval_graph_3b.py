from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, re
from tqdm import tqdm
from transformers import AutoTokenizer

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
MODEL       = "lamm-mit/Graph-Preflexor-3b_08012026"
DATASET     = "/home/palsubhadeep/Github/Reasoning_comparison/datasets/research_paper/final_dataset_with_answers_bluff_question.json"
OUTFILE     = "qna_bluff_graph_3B_results.jsonl"
MAX_TOKENS  = 32768
TEMPERATURE = 0.2
# TOP_P       = 0.9
MAX_WORKERS = 16

tokenizer = AutoTokenizer.from_pretrained(MODEL)

REASONING_TAGS = ["brainstorm", "graph", "graph_json", "patterns", "synthesis"]

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
def extract_tag(text, tag):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match   = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def get_thinking(output):
    parts = []
    for tag in ["brainstorm", "graph", "graph_json", "patterns"]:
        content = extract_tag(output, tag)
        if content:
            parts.append(f"<{tag}>\n{content}\n</{tag}>")
    return "\n\n".join(parts)

def get_final_output(output):
    synthesis = extract_tag(output, "synthesis")
    if synthesis:
        return synthesis
    cleaned = re.sub(
        rf"<({'|'.join(REASONING_TAGS)})>.*?</({'|'.join(REASONING_TAGS)})>",
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

    messages = [
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return formatted, question, tag, doi

# ------------------------------------------------------------------------------
# Per-question worker
# ------------------------------------------------------------------------------
def evaluate_row(args):
    i, row = args
    prompt, question, tag, doi = format_prompt(row)

    start = time.time()
    try:
        response = client.completions.create(
            model=MODEL,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            # extra_body={"top_p": TOP_P},
            stream=False,
        )
        raw_output        = response.choices[0].text
        completion_tokens = response.usage.completion_tokens
        error             = ""
    except Exception as e:
        raw_output        = ""
        completion_tokens = 0
        error             = f"{type(e).__name__}: {e}"

    elapsed    = time.time() - start
    thinking   = get_thinking(raw_output)
    llm_output = get_final_output(raw_output)
    extracted  = {t: extract_tag(raw_output, t) for t in REASONING_TAGS}

    return {
        "id":                i,
        "doi":               doi,
        "tag":               tag,
        "title":             row.get("title", ""),
        "question":          question,
        "accepted_answer":   row.get("accepted_answer", ""),
        "raw_output":        raw_output,
        "llm_output":        llm_output,
        "thinking":          thinking,
        "brainstorm":        extracted["brainstorm"],
        "graph":             extracted["graph"],
        "graph_json":        extracted["graph_json"],
        "patterns":          extracted["patterns"],
        "synthesis":         extracted["synthesis"],
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

        tags_present = " | ".join(f"<{t}>" for t in REASONING_TAGS if result.get(t))
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

print(f"\n--- Reasoning Tag Coverage ---")
for t in REASONING_TAGS:
    count = sum(1 for r in results if r.get(t))
    print(f"  <{t}>: {count}/{len(results)} ({100*count//len(results)}%)")

if "tag" in df.columns:
    print(f"\n--- By Tag ---")
    print(df.groupby("tag").size().to_string())
print(f"\nResults saved : {OUTFILE}")