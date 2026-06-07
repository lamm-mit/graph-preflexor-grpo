"""
exp2_patch_2c.py — Rerun 2C for ids that were skipped due to max_new_tokens truncation.
Appends to existing exp2_hidden_state_divergence.csv and regenerates 2D/2E/analysis.
"""

import json, gc, re
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

WORK_DIR   = Path("/projects/bfir/ssourav")
GRAPH_FILE = WORK_DIR / "graph_8b_data_eval.jsonl"
OUT_2A     = WORK_DIR / "exp2_qwen_thinking_outputs.jsonl"
OUT_2C     = WORK_DIR / "exp2_hidden_state_divergence.csv"
OUT_2D     = WORK_DIR / "exp2_hidden_to_graph_stage_alignment.csv"
OUT_2E     = WORK_DIR / "exp2_summary_by_question_type.csv"
OUT_ANAL   = WORK_DIR / "exp2_analysis.json"

QWEN_MODEL     = "Qwen/Qwen3-8B"
EMBED_MODEL    = "BAAI/bge-base-en-v1.5"
SKIPPED_IDS    = {2, 3, 4, 5, 6}
MAX_NEW_TOKENS = 5000          # enough for the longest CoT (~3900 tokens)

# ── helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

# ── 2C patch ──────────────────────────────────────────────────────────────────
def patch_2c(merged_rows, tokenizer, qwen_model):
    print("=" * 60)
    print("2C PATCH: reprocessing skipped ids", sorted(SKIPPED_IDS))
    print("=" * 60)

    new_records = []

    for row in merged_rows:
        if row["id"] not in SKIPPED_IDS:
            continue

        print(f"\n  Processing id={row['id']} ({row['question_type']})")

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
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids  = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        print(f"    Generated {len(generated_ids)} tokens")
        print(f"    </think> present: {'</think>' in generated_text}")

        if "</think>" not in generated_text:
            print(f"    WARNING: still no </think> even at {MAX_NEW_TOKENS} tokens — skipping")
            del outputs
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # Binary search for split position
        gen_ids_list = generated_ids.tolist()
        lo, hi = 1, len(gen_ids_list)
        split_pos = None
        while lo <= hi:
            mid = (lo + hi) // 2
            partial = tokenizer.decode(gen_ids_list[:mid], skip_special_tokens=False)
            if "</think>" in partial:
                split_pos = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if split_pos is None or split_pos >= len(gen_ids_list) - 1:
            print(f"    WARNING: split_pos={split_pos} invalid — skipping")
            del outputs
            torch.cuda.empty_cache()
            gc.collect()
            continue

        thinking_steps = list(range(0, split_pos))
        answer_steps   = list(range(split_pos, len(outputs.hidden_states)))

        print(f"    Split: {len(thinking_steps)} thinking steps, {len(answer_steps)} answer steps")

        if not thinking_steps or not answer_steps:
            print(f"    WARNING: empty span — skipping")
            continue

        n_layers = len(outputs.hidden_states[0])

        for layer_idx in range(n_layers):
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

            new_records.append({
                "id":                                row["id"],
                "question_type":                     row["question_type"],
                "layer":                             layer_idx,
                "thinking_answer_cosine_similarity": round(sim, 6),
                "thinking_answer_cosine_distance":   round(dist, 6),
                "n_thinking_tokens":                 len(thinking_steps),
                "n_answer_tokens":                   len(answer_steps),
            })

        print(f"    Saved {n_layers} layer records for id={row['id']}")

        del outputs
        torch.cuda.empty_cache()
        gc.collect()

    # Append to existing CSV
    if new_records:
        new_df  = pd.DataFrame(new_records)
        old_df  = pd.read_csv(OUT_2C)
        full_df = pd.concat([old_df, new_df], ignore_index=True)
        full_df.sort_values(["id", "layer"], inplace=True)
        full_df.to_csv(OUT_2C, index=False)
        print(f"\n  Appended {len(new_records)} rows → {OUT_2C} (total: {len(full_df)})")
        return full_df
    else:
        print("  No new records — returning existing CSV")
        return pd.read_csv(OUT_2C)

# ── Rerun 2D with full hidden_df ──────────────────────────────────────────────
def rerun_2d(merged_rows, embed_model, hidden_df):
    print("\n" + "=" * 60)
    print("2D RERUN: full dataset alignment")
    print("=" * 60)

    STAGES = [
        ("brainstorm", "graph_preflexor_brainstorm"),
        ("graph",      "graph_preflexor_graph"),
        ("patterns",   "graph_preflexor_patterns"),
        ("synthesis",  "graph_preflexor_synthesis"),
        ("answer",     "graph_preflexor_answer"),
    ]

    records = []
    for row in merged_rows:
        qwen_thinking_emb = embed_model.encode([row["qwen_thinking"]], normalize_embeddings=True)[0]
        qwen_answer_emb   = embed_model.encode([row["qwen_final_answer"]], normalize_embeddings=True)[0]

        stage_embs = {}
        for stage_name, field in STAGES:
            stage_embs[stage_name] = embed_model.encode([row[field]], normalize_embeddings=True)[0]

        row_hidden = hidden_df[hidden_df["id"] == row["id"]]
        n_layers   = int(row_hidden["layer"].max()) + 1 if not row_hidden.empty else 1

        for layer_idx in range(n_layers):
            layer_row = row_hidden[row_hidden["layer"] == layer_idx]
            div = float(layer_row["thinking_answer_cosine_distance"].values[0]) if not layer_row.empty else 0.5

            blended = (1 - div) * qwen_thinking_emb + div * qwen_answer_emb
            blended = blended / (np.linalg.norm(blended) + 1e-9)

            for stage_name, _ in STAGES:
                sim = cosine_sim(blended, stage_embs[stage_name])
                records.append({
                    "id":               row["id"],
                    "question_type":    row["question_type"],
                    "layer":            layer_idx,
                    "stage":            stage_name,
                    "similarity":       round(sim, 6),
                    "layer_divergence": round(div, 6),
                })

        print(f"  id={row['id']:2d} done ({n_layers} layers x {len(STAGES)} stages)")

    df = pd.DataFrame(records)
    df.to_csv(OUT_2D, index=False)
    print(f"  Saved {len(df)} rows → {OUT_2D}")
    return df

# ── Rerun 2E ──────────────────────────────────────────────────────────────────
def rerun_2e(merged_rows, hidden_df):
    print("\n" + "=" * 60)
    print("2E RERUN: per-question-type summary")
    print("=" * 60)

    sim_df = pd.read_csv(WORK_DIR / "exp2_backtracking_text_similarity.csv")

    records = []
    for row in merged_rows:
        row_sim = sim_df[sim_df["id"] == row["id"]]
        if row_sim.empty:
            continue
        r = row_sim.iloc[0]
        records.append({
            "id":            row["id"],
            "question_type": row["question_type"],
            "closest_stage": r["closest_stage"],
            "answer_to_thinking_gap":               round(1 - r["sim_qwen_answer_to_qwen_thinking"], 4),
            "answer_to_graph_graph_gap":            round(1 - r["sim_qwen_answer_to_graph_graph"], 4),
            "answer_to_graph_synthesis_gap":        round(1 - r["sim_qwen_answer_to_graph_synthesis"], 4),
            "answer_to_graph_preflexor_answer_gap": round(1 - r["sim_qwen_answer_to_graph_answer"], 4),
        })

    detail_df = pd.DataFrame(records)
    agg = detail_df.groupby("question_type").agg(
        n_questions        =("id", "count"),
        most_common_closest=("closest_stage", lambda x: x.mode()[0]),
        avg_thinking_gap   =("answer_to_thinking_gap", "mean"),
        avg_graph_gap      =("answer_to_graph_graph_gap", "mean"),
        avg_synthesis_gap  =("answer_to_graph_synthesis_gap", "mean"),
        avg_preflexor_gap  =("answer_to_graph_preflexor_answer_gap", "mean"),
    ).reset_index().round(4)

    agg.to_csv(OUT_2E, index=False)
    print(agg.to_string(index=False))

    # Update analysis JSON
    by_layer = hidden_df.groupby("layer")["thinking_answer_cosine_distance"].mean()
    analysis = {
        "closest_stage_distribution": sim_df["closest_stage"].value_counts().to_dict(),
        "mean_similarities": {
            col: round(float(sim_df[col].mean()), 4)
            for col in sim_df.columns if col.startswith("sim_")
        },
        "layer_divergence_mean":  {int(k): round(float(v), 4) for k, v in by_layer.items()},
        "peak_divergence_layer":  int(by_layer.idxmax()),
        "min_divergence_layer":   int(by_layer.idxmin()),
        "ids_with_full_2c":       sorted(hidden_df["id"].unique().tolist()),
        "per_question_type":      agg.to_dict(orient="records"),
    }
    with open(OUT_ANAL, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis saved → {OUT_ANAL}")
    return agg

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    merged_rows = load_jsonl(OUT_2A)
    print(f"Loaded {len(merged_rows)} merged rows")

    # Load Qwen for 2C patch
    print("\nLoading Qwen3-8B ...")
    tokenizer  = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    qwen_model.eval()

    hidden_df = patch_2c(merged_rows, tokenizer, qwen_model)

    del qwen_model
    torch.cuda.empty_cache()
    gc.collect()

    # Load embedding model for 2D rerun
    print("\nLoading embedding model ...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL)

    rerun_2d(merged_rows, embed_model, hidden_df)
    rerun_2e(merged_rows, hidden_df)

    print("\nPatch complete.")
    covered = sorted(hidden_df["id"].unique().tolist())
    print(f"  2C now covers ids: {covered}")

if __name__ == "__main__":
    main()
