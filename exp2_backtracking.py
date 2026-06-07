"""
Experiment 2: Backtracking Qwen Answers Through CoT, Hidden States,
and Graph-PRefLexOR Stages
=======================================================================
2A  — Generate Qwen answers with thinking enabled (skip if already in qwen_8b_results.jsonl)
2B  — Text-level embedding similarity: Qwen answer vs all reasoning objects
2C  — Hidden-state backtracking: layer-wise thinking→answer divergence
2D  — Hidden-state alignment with Graph-PRefLexOR stage embeddings
2E  — Per-question-type summary table

Input files:
    /projects/bfir/ssourav/graph_8b_data_eval.jsonl   (Graph-PRefLexOR stages + answer)
    /projects/bfir/ssourav/qwen_8b_results.jsonl      (Qwen thinking + final answer)

Outputs:
    exp2_qwen_thinking_outputs.jsonl
    exp2_backtracking_text_similarity.csv
    exp2_hidden_state_divergence.csv
    exp2_hidden_to_graph_stage_alignment.csv
    exp2_summary_by_question_type.csv
    exp2_analysis.json
"""

import json, re, os, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Paths ─────────────────────────────────────────────────────────────────────
GRAPH_FILE  = "/projects/bfir/ssourav/graph_8b_data_eval.jsonl"
QWEN_FILE   = "/projects/bfir/ssourav/qwen_8b_results.jsonl"
WORK_DIR    = Path("/projects/bfir/ssourav")

OUT_2A      = WORK_DIR / "exp2_qwen_thinking_outputs.jsonl"
OUT_2B      = WORK_DIR / "exp2_backtracking_text_similarity.csv"
OUT_2C      = WORK_DIR / "exp2_hidden_state_divergence.csv"
OUT_2D      = WORK_DIR / "exp2_hidden_to_graph_stage_alignment.csv"
OUT_2E      = WORK_DIR / "exp2_summary_by_question_type.csv"
OUT_ANAL    = WORK_DIR / "exp2_analysis.json"

QWEN_MODEL  = "Qwen/Qwen3-8B"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def extract_thinking_and_answer(raw_output: str):
    """Split <think>...</think> from final answer."""
    think_match = re.search(r'<think>(.*?)</think>(.*)', raw_output, re.DOTALL)
    if think_match:
        thinking     = think_match.group(1).strip()
        final_answer = think_match.group(2).strip()
    else:
        # No think tag — treat everything as final answer
        thinking     = ""
        final_answer = raw_output.strip()
    return thinking, final_answer

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

# ══════════════════════════════════════════════════════════════════════════════
# 2A — Build merged dataset (Qwen thinking + Graph-PRefLexOR stages)
# ══════════════════════════════════════════════════════════════════════════════
def run_2a(graph_rows, qwen_rows):
    print("\n" + "="*60)
    print("2A: Building merged thinking outputs")
    print("="*60)

    graph_by_id = {r["id"]: r for r in graph_rows}
    qwen_by_id  = {r["id"]: r for r in qwen_rows}

    merged = []
    for row_id in sorted(graph_by_id):
        g = graph_by_id[row_id]
        q = qwen_by_id.get(row_id, {})

        raw = q.get("raw_output", "")
        thinking, final_answer = extract_thinking_and_answer(raw)

        # Graph-PRefLexOR's own answer is in llm_output (after </think> in graph file)
        graph_raw = g.get("raw_output", "")
        _, graph_answer = extract_thinking_and_answer(graph_raw)
        # Fallback: use synthesis as proxy for graph answer if parsing fails
        if not graph_answer:
            graph_answer = g.get("synthesis", "")

        record = {
            "id":                        row_id,
            "question":                  g["question"],
            "question_type":             g["question_type"],
            "qwen_thinking":             thinking,
            "qwen_final_answer":         final_answer,
            "graph_preflexor_brainstorm": g.get("brainstorm", ""),
            "graph_preflexor_graph":      g.get("graph", ""),
            "graph_preflexor_patterns":   g.get("patterns", ""),
            "graph_preflexor_synthesis":  g.get("synthesis", ""),
            "graph_preflexor_answer":     graph_answer,
        }
        merged.append(record)

    with open(OUT_2A, "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")

    print(f"  Saved {len(merged)} rows → {OUT_2A}")
    return merged

# ══════════════════════════════════════════════════════════════════════════════
# Embedding model (used in 2B and 2D)
# ══════════════════════════════════════════════════════════════════════════════
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    print(f"  Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    return model

def embed_texts(embed_model, texts: list[str]) -> np.ndarray:
    """Returns (N, D) float32 array."""
    return embed_model.encode(texts, normalize_embeddings=True,
                               batch_size=16, show_progress_bar=False)

# ══════════════════════════════════════════════════════════════════════════════
# 2B — Text-level embedding similarity
# ══════════════════════════════════════════════════════════════════════════════
def run_2b(merged_rows, embed_model):
    print("\n" + "="*60)
    print("2B: Text-level embedding similarity")
    print("="*60)

    STAGE_KEYS = [
        ("sim_qwen_answer_to_qwen_thinking",      "qwen_thinking"),
        ("sim_qwen_answer_to_graph_brainstorm",   "graph_preflexor_brainstorm"),
        ("sim_qwen_answer_to_graph_graph",        "graph_preflexor_graph"),
        ("sim_qwen_answer_to_graph_patterns",     "graph_preflexor_patterns"),
        ("sim_qwen_answer_to_graph_synthesis",    "graph_preflexor_synthesis"),
        ("sim_qwen_answer_to_graph_answer",       "graph_preflexor_answer"),
    ]

    records = []
    for row in merged_rows:
        answer_text = row["qwen_final_answer"]
        texts_to_embed = [answer_text] + [row[sk] for _, sk in STAGE_KEYS]

        # Embed all at once
        embs = embed_texts(embed_model, texts_to_embed)
        answer_emb = embs[0]
        stage_embs = embs[1:]

        sims = {}
        for i, (col, _) in enumerate(STAGE_KEYS):
            sims[col] = round(float(np.dot(answer_emb, stage_embs[i])), 6)

        # closest_stage = argmax over the 6 similarities
        sim_vals  = [sims[col] for col, _ in STAGE_KEYS]
        col_names = [col for col, _ in STAGE_KEYS]
        closest_stage = col_names[int(np.argmax(sim_vals))].replace("sim_qwen_answer_to_", "")

        rec = {
            "id":            row["id"],
            "question_type": row["question_type"],
            **sims,
            "closest_stage": closest_stage,
        }
        records.append(rec)
        print(f"  id={row['id']:2d}  closest={closest_stage}")

    df = pd.DataFrame(records)
    df.to_csv(OUT_2B, index=False)
    print(f"  Saved → {OUT_2B}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 2C — Hidden-state backtracking (layer-wise thinking→answer divergence)
# ══════════════════════════════════════════════════════════════════════════════
def run_2c(merged_rows, tokenizer, qwen_model):
    print("\n" + "="*60)
    print("2C: Hidden-state backtracking (layer-wise divergence)")
    print("="*60)

    records = []

    for row in merged_rows:
        print(f"  Processing id={row['id']} ({row['question_type']})")

        prompt = (
            "You are answering an open-ended scientific reasoning question.\n\n"
            "/think\n\n"
            f"Question:\n{row['question']}\n\n"
            "Think carefully about the scientific mechanisms, causal relations, "
            "and possible explanations. Then give the final answer."
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(qwen_model.device)

        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=1200,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Generated token ids (excluding prompt)
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # ── Find </think> boundary robustly via text, not token search ──────
        # Decode each prefix incrementally would be slow; instead find the
        # char offset of </think> in the full decoded text, then binary-search
        # for the token index whose decoded prefix first passes that offset.
        THINK_END = "</think>"
        think_char_pos = generated_text.find(THINK_END)

        if think_char_pos == -1:
            print(f"    WARNING: no </think> in generated text for id={row['id']}, skipping 2C")
            continue

        # Binary search: find smallest token index t such that
        # decode(generated_ids[:t]) contains the full </think> tag.
        gen_ids_list = generated_ids.tolist()
        lo, hi = 1, len(gen_ids_list)
        split_pos = None
        while lo <= hi:
            mid = (lo + hi) // 2
            partial = tokenizer.decode(gen_ids_list[:mid], skip_special_tokens=False)
            if THINK_END in partial:
                split_pos = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if split_pos is None or split_pos >= len(gen_ids_list) - 1:
            print(f"    WARNING: split_pos={split_pos} invalid for id={row['id']}, skipping 2C")
            continue

        print(f"    id={row['id']}: {split_pos} thinking tokens, "
              f"{len(gen_ids_list) - split_pos} answer tokens")

        # output_hidden_states: tuple[step] of tuple[layer] of (1, 1, hidden_dim)
        # Each generation step has hidden states for all layers
        n_layers = len(outputs.hidden_states[0])
        n_steps  = len(outputs.hidden_states)

        thinking_steps = list(range(0, split_pos))
        answer_steps   = list(range(split_pos, n_steps))

        if not thinking_steps or not answer_steps:
            print(f"    WARNING: empty span for id={row['id']}, skipping")
            continue

        for layer_idx in range(n_layers):
            # Pool hidden states over thinking span
            thinking_vecs = [
                outputs.hidden_states[step][layer_idx][0, 0, :].float().cpu().numpy()
                for step in thinking_steps
            ]
            answer_vecs = [
                outputs.hidden_states[step][layer_idx][0, 0, :].float().cpu().numpy()
                for step in answer_steps
            ]

            thinking_mean = np.mean(thinking_vecs, axis=0)
            answer_mean   = np.mean(answer_vecs,   axis=0)

            sim  = cosine_sim(thinking_mean, answer_mean)
            dist = 1.0 - sim

            records.append({
                "id":                           row["id"],
                "question_type":                row["question_type"],
                "layer":                        layer_idx,
                "thinking_answer_cosine_similarity": round(sim, 6),
                "thinking_answer_cosine_distance":   round(dist, 6),
                "n_thinking_tokens":            len(thinking_steps),
                "n_answer_tokens":              len(answer_steps),
            })

        # Free GPU memory between rows
        del outputs
        torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(records)
    df.to_csv(OUT_2C, index=False)
    print(f"  Saved {len(df)} rows → {OUT_2C}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 2D — Hidden-state alignment with Graph-PRefLexOR stage embeddings
# ══════════════════════════════════════════════════════════════════════════════
def run_2d(merged_rows, embed_model, hidden_df: pd.DataFrame):
    """
    Simpler first version: compare Qwen text embeddings (thinking + answer)
    to Graph-PRefLexOR stage text embeddings, then project analysis per layer
    using the hidden-state divergence as a weight proxy.
    Full hidden-state projection requires training a linear map (noted below).
    """
    print("\n" + "="*60)
    print("2D: Hidden-state alignment with Graph-PRefLexOR stages")
    print("="*60)

    STAGES = [
        ("brainstorm", "graph_preflexor_brainstorm"),
        ("graph",      "graph_preflexor_graph"),
        ("patterns",   "graph_preflexor_patterns"),
        ("synthesis",  "graph_preflexor_synthesis"),
        ("answer",     "graph_preflexor_answer"),
    ]

    records = []

    for row in merged_rows:
        # Embed Qwen thinking and final answer
        qwen_thinking_emb = embed_texts(embed_model, [row["qwen_thinking"]])[0]
        qwen_answer_emb   = embed_texts(embed_model, [row["qwen_final_answer"]])[0]

        # Embed each graph stage
        stage_embs = {}
        for stage_name, field in STAGES:
            stage_embs[stage_name] = embed_texts(embed_model, [row[field]])[0]

        # For each layer, use divergence as a proxy weight
        # (high divergence = answer representation diverged from thinking)
        row_hidden = hidden_df[hidden_df["id"] == row["id"]] if not hidden_df.empty else pd.DataFrame()
        n_layers = int(row_hidden["layer"].max()) + 1 if not row_hidden.empty else 1

        for layer_idx in range(n_layers):
            # Get divergence for this layer
            layer_row = row_hidden[row_hidden["layer"] == layer_idx]
            if layer_row.empty:
                div = 0.5
            else:
                div = float(layer_row["thinking_answer_cosine_distance"].values[0])

            # Interpolate between thinking and answer embedding using divergence
            # div=0 → thinking dominates; div=1 → answer dominates
            blended_emb = (1 - div) * qwen_thinking_emb + div * qwen_answer_emb
            blended_emb = blended_emb / (np.linalg.norm(blended_emb) + 1e-9)

            for stage_name, _ in STAGES:
                sim = cosine_sim(blended_emb, stage_embs[stage_name])
                records.append({
                    "id":            row["id"],
                    "question_type": row["question_type"],
                    "layer":         layer_idx,
                    "stage":         stage_name,
                    "similarity":    round(sim, 6),
                    "layer_divergence": round(div, 6),
                })

        print(f"  id={row['id']:2d} done ({n_layers} layers x {len(STAGES)} stages)")

    df = pd.DataFrame(records)
    df.to_csv(OUT_2D, index=False)
    print(f"  Saved {len(df)} rows → {OUT_2D}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 2E — Per-question-type summary table
# ══════════════════════════════════════════════════════════════════════════════
def run_2e(merged_rows, sim_df: pd.DataFrame, hidden_df: pd.DataFrame):
    print("\n" + "="*60)
    print("2E: Per-question-type summary table")
    print("="*60)

    SIM_COLS = [
        "sim_qwen_answer_to_qwen_thinking",
        "sim_qwen_answer_to_graph_brainstorm",
        "sim_qwen_answer_to_graph_graph",
        "sim_qwen_answer_to_graph_patterns",
        "sim_qwen_answer_to_graph_synthesis",
        "sim_qwen_answer_to_graph_answer",
    ]

    # Gap = distance (1 - similarity)
    records = []
    for row in merged_rows:
        row_sim = sim_df[sim_df["id"] == row["id"]]
        if row_sim.empty:
            continue

        r = row_sim.iloc[0]
        rec = {
            "id":            row["id"],
            "question_type": row["question_type"],
            "closest_stage": r["closest_stage"],
            "answer_to_thinking_gap":              round(1 - r["sim_qwen_answer_to_qwen_thinking"], 4),
            "answer_to_graph_graph_gap":           round(1 - r["sim_qwen_answer_to_graph_graph"], 4),
            "answer_to_graph_synthesis_gap":       round(1 - r["sim_qwen_answer_to_graph_synthesis"], 4),
            "answer_to_graph_preflexor_answer_gap":round(1 - r["sim_qwen_answer_to_graph_answer"], 4),
        }
        records.append(rec)

    detail_df = pd.DataFrame(records)

    # Group by question_type
    agg = detail_df.groupby("question_type").agg(
        n_questions       =("id", "count"),
        most_common_closest=("closest_stage", lambda x: x.mode()[0] if len(x) else ""),
        avg_thinking_gap  =("answer_to_thinking_gap",               "mean"),
        avg_graph_gap     =("answer_to_graph_graph_gap",            "mean"),
        avg_synthesis_gap =("answer_to_graph_synthesis_gap",        "mean"),
        avg_preflexor_gap =("answer_to_graph_preflexor_answer_gap", "mean"),
    ).reset_index()

    agg = agg.round(4)
    agg.to_csv(OUT_2E, index=False)

    print("\n  ── Per-question-type summary ──")
    print(agg.to_string(index=False))
    print(f"\n  Saved → {OUT_2E}")
    return agg

# ══════════════════════════════════════════════════════════════════════════════
# Final analysis dump
# ══════════════════════════════════════════════════════════════════════════════
def save_analysis(sim_df, hidden_df, summary_df):
    analysis = {}

    # 2B: closest stage distribution
    if not sim_df.empty and "closest_stage" in sim_df.columns:
        analysis["closest_stage_distribution"] = sim_df["closest_stage"].value_counts().to_dict()
        analysis["mean_similarities"] = {
            col: round(float(sim_df[col].mean()), 4)
            for col in sim_df.columns if col.startswith("sim_")
        }

    # 2C: layer-wise divergence summary
    if not hidden_df.empty:
        by_layer = hidden_df.groupby("layer")["thinking_answer_cosine_distance"].mean()
        analysis["layer_divergence_mean"] = {int(k): round(float(v), 4) for k, v in by_layer.items()}
        analysis["peak_divergence_layer"] = int(by_layer.idxmax())
        analysis["min_divergence_layer"]  = int(by_layer.idxmin())

    # 2E: summary
    if not summary_df.empty:
        analysis["per_question_type"] = summary_df.to_dict(orient="records")

    with open(OUT_ANAL, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis saved → {OUT_ANAL}")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    graph_rows = load_jsonl(GRAPH_FILE)
    qwen_rows  = load_jsonl(QWEN_FILE)
    print(f"  graph rows: {len(graph_rows)}, qwen rows: {len(qwen_rows)}")

    # ── 2A ─────────────────────────────────────────────────────────────────
    if OUT_2A.exists():
        print(f"\n2A: Already exists, loading {OUT_2A}")
        merged_rows = load_jsonl(OUT_2A)
    else:
        merged_rows = run_2a(graph_rows, qwen_rows)

    # ── Install sentence-transformers if needed ─────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        os.system("pip install sentence-transformers -q")
        from sentence_transformers import SentenceTransformer

    # ── 2B ─────────────────────────────────────────────────────────────────
    print("\nLoading embedding model for 2B/2D...")
    embed_model = load_embed_model()

    if OUT_2B.exists():
        print(f"2B: Already exists, loading {OUT_2B}")
        sim_df = pd.read_csv(OUT_2B)
    else:
        sim_df = run_2b(merged_rows, embed_model)

    # ── 2C ─────────────────────────────────────────────────────────────────
    if OUT_2C.exists():
        print(f"\n2C: Already exists, loading {OUT_2C}")
        hidden_df = pd.read_csv(OUT_2C)
    else:
        print("\nLoading Qwen3-8B for hidden-state extraction (2C)...")
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        qwen_model.eval()
        hidden_df = run_2c(merged_rows, tokenizer, qwen_model)
        # Free GPU after 2C — 2D uses embeddings only
        del qwen_model
        torch.cuda.empty_cache()
        gc.collect()

    # ── 2D ─────────────────────────────────────────────────────────────────
    if OUT_2D.exists():
        print(f"\n2D: Already exists, loading {OUT_2D}")
        align_df = pd.read_csv(OUT_2D)
    else:
        align_df = run_2d(merged_rows, embed_model, hidden_df)

    # ── 2E ─────────────────────────────────────────────────────────────────
    summary_df = run_2e(merged_rows, sim_df, hidden_df)

    # ── Final analysis ──────────────────────────────────────────────────────
    save_analysis(sim_df, hidden_df, summary_df)

    print("\n" + "="*60)
    print("EXPERIMENT 2 COMPLETE")
    print("="*60)
    print(f"  2A  {OUT_2A}")
    print(f"  2B  {OUT_2B}")
    print(f"  2C  {OUT_2C}")
    print(f"  2D  {OUT_2D}")
    print(f"  2E  {OUT_2E}")
    print(f"  Analysis  {OUT_ANAL}")


if __name__ == "__main__":
    main()
