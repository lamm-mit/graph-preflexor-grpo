"""
step0_graph_preflexor_100.py
============================
Run Graph-PRefLexOR on all 100 questions in question_all.jsonl.
Generates brainstorm → graph → patterns → synthesis → final answer
using Qwen3-8B in non-thinking mode (same pipeline as graph_8b_data_eval.jsonl).

Output: /projects/bfir/ssourav/graph_8b_data_eval_100.jsonl
"""

import json, re, time, gc
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "/projects/bfir/ssourav/question_all.jsonl"
OUTPUT_FILE = "/projects/bfir/ssourav/graph_8b_data_eval_100.jsonl"
MODEL_NAME  = "Qwen/Qwen3-8B"

GEN_CFG = dict(
    max_new_tokens=1200,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    do_sample=True,
)

# ── Stage prompts ─────────────────────────────────────────────────────────────
BRAINSTORM_PROMPT = """You are a scientific reasoning assistant.

Question:
{question}

Step 1 — Brainstorm:
List key concepts, hypotheses, and mechanisms relevant to this question.
Be specific. Identify what is known and what needs to be explained.
Write 3-5 sentences."""

GRAPH_PROMPT = """You are a scientific reasoning assistant.

Question:
{question}

Brainstorm:
{brainstorm}

Step 2 — Causal Graph:
Based on the brainstorm, identify core entities and their causal relationships.
Format: Entity1 → Entity2 (relationship); Entity2 → Entity3 (relationship); ...
Be specific about mechanisms. Write 4-8 relationships."""

PATTERNS_PROMPT = """You are a scientific reasoning assistant.

Question:
{question}

Brainstorm:
{brainstorm}

Causal Graph:
{graph}

Step 3 — Patterns:
Extract the key causal patterns and abstractions from the graph.
Use compact notation: A → B ↑ C (meaning A causes B which increases C).
Identify the main causal chain and any feedback loops or tradeoffs.
Write 2-4 compact pattern statements."""

SYNTHESIS_PROMPT = """You are a scientific reasoning assistant.

Question:
{question}

Brainstorm:
{brainstorm}

Causal Graph:
{graph}

Patterns:
{patterns}

Step 4 — Synthesis:
Synthesize the brainstorm, graph, and patterns into a coherent scientific explanation.
Explain the mechanism, identify tradeoffs or failure modes, and draw conclusions.
Write 4-6 sentences."""

FINAL_ANSWER_PROMPT = """You are a scientific reasoning assistant.

Question:
{question}

You have developed the following structured reasoning:

Brainstorm:
{brainstorm}

Causal Graph:
{graph}

Patterns:
{patterns}

Synthesis:
{synthesis}

Using all of the above, write the final answer to the question.
Be specific, explain the mechanism, and address tradeoffs or failure modes.
Do not mention the reasoning stages. Give only the final answer."""

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def generate(tokenizer, model, prompt, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **GEN_CFG,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    # Strip any accidental <think> blocks
    decoded = re.sub(r"<think>[\s\S]*?</think>", "", decoded).strip()
    return decoded

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    questions = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(questions)} questions from {INPUT_FILE}")

    # Load already-done ids for resumability
    done_ids = set()
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resuming — {len(done_ids)} already done.")

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    out_f = open(OUTPUT_FILE, "a", buffering=1)
    total = len(questions)
    start = time.time()

    for i, row in enumerate(questions):
        if row["id"] in done_ids:
            continue

        q = row["question"]
        t0 = time.time()

        try:
            brainstorm = generate(tokenizer, model,
                BRAINSTORM_PROMPT.format(question=q))

            graph = generate(tokenizer, model,
                GRAPH_PROMPT.format(question=q, brainstorm=brainstorm))

            patterns = generate(tokenizer, model,
                PATTERNS_PROMPT.format(question=q, brainstorm=brainstorm, graph=graph))

            synthesis = generate(tokenizer, model,
                SYNTHESIS_PROMPT.format(question=q, brainstorm=brainstorm,
                                        graph=graph, patterns=patterns))

            llm_output = generate(tokenizer, model,
                FINAL_ANSWER_PROMPT.format(question=q, brainstorm=brainstorm,
                                           graph=graph, patterns=patterns,
                                           synthesis=synthesis))

            elapsed = time.time() - t0
            done_count = i + 1
            eta = (time.time() - start) / done_count * (total - done_count)

            record = {
                "id":           row["id"],
                "paper_title":  row.get("paper_title", ""),
                "doi":          row.get("doi", ""),
                "question_type":row["question_type"],
                "question":     q,
                "brainstorm":   brainstorm,
                "graph":        graph,
                "graph_json":   "",
                "patterns":     patterns,
                "synthesis":    synthesis,
                "llm_output":   llm_output,
                "error":        "",
                "time_sec":     round(elapsed, 2),
            }
            out_f.write(json.dumps(record) + "\n")

            print(f"[{done_count}/{total}] id={row['id']} "
                  f"type={row['question_type']} "
                  f"time={elapsed:.1f}s ETA={eta/60:.1f}min")

        except Exception as e:
            print(f"  ERROR id={row['id']}: {e}")
            record = {
                "id": row["id"], "question_type": row["question_type"],
                "question": q, "brainstorm": "", "graph": "", "graph_json": "",
                "patterns": "", "synthesis": "", "llm_output": "",
                "error": str(e), "time_sec": 0,
            }
            out_f.write(json.dumps(record) + "\n")

    out_f.close()
    print(f"\nDone. Output: {OUTPUT_FILE}")
    with open(OUTPUT_FILE) as f:
        n = sum(1 for l in f if l.strip())
    print(f"Total rows written: {n}")

if __name__ == "__main__":
    main()
