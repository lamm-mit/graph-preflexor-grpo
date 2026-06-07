"""
exp2_100q_full_pipeline.py
==========================
Full Experiment 2 pipeline for 100 questions.
Runs sequentially: 2A → 2B → 2C → 2D → 2E → 2F → 2G.1 → 2G.2

Inputs:
  /projects/bfir/ssourav/graph_8b_data_eval_100.jsonl
  (Qwen thinking generated fresh in 2A)

Outputs:
  exp2_qwen_thinking_outputs_100.jsonl
  exp2_backtracking_text_similarity_100.csv
  exp2_hidden_state_divergence_100.csv
  exp2_question_level_divergence_summary_100.csv
  exp2_summary_by_question_type_100.csv
  exp2_layer_probe_thinking_vs_answer_100.csv
  exp2_logit_lens_token_categories_100.csv
  plots_100/plot*.png
"""

import json, re, gc, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK   = Path("/projects/bfir/ssourav")
PLOTS  = WORK / "plots_100"
PLOTS.mkdir(exist_ok=True)

GRAPH_FILE  = WORK / "graph_8b_data_eval_100.jsonl"
OUT_2A      = WORK / "exp2_qwen_thinking_outputs_100.jsonl"
OUT_2B      = WORK / "exp2_backtracking_text_similarity_100.csv"
OUT_2C      = WORK / "exp2_hidden_state_divergence_100.csv"
OUT_2E      = WORK / "exp2_question_level_divergence_summary_100.csv"
OUT_2F      = WORK / "exp2_summary_by_question_type_100.csv"
OUT_PROBE   = WORK / "exp2_layer_probe_thinking_vs_answer_100.csv"
OUT_LOGIT   = WORK / "exp2_logit_lens_token_categories_100.csv"

QWEN_MODEL  = "Qwen/Qwen3-8B"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

GEN_CFG_THINKING = dict(max_new_tokens=5000, do_sample=False)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})
BLUE="#378ADD"; CORAL="#D85A30"; TEAL="#1D9E75"
PURPLE="#7F77DD"; AMBER="#BA7517"; GRAY="#888780"; DPURPLE="#3C3489"

SIM_COLS = [
    ("sim_qwen_answer_to_qwen_thinking",    "Qwen thinking"),
    ("sim_qwen_answer_to_graph_brainstorm", "Graph brainstorm"),
    ("sim_qwen_answer_to_graph_graph",      "Graph graph"),
    ("sim_qwen_answer_to_graph_patterns",   "Graph patterns"),
    ("sim_qwen_answer_to_graph_synthesis",  "Graph synthesis"),
    ("sim_qwen_answer_to_graph_answer",     "Graph answer"),
]
SIM_KEYS   = [c for c, _ in SIM_COLS]
SIM_LABELS = [l for _, l in SIM_COLS]
SIM_COLORS = [PURPLE, TEAL, CORAL, DPURPLE, AMBER, BLUE]

# ── Helpers ───────────────────────────────────────────────────────────────────
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

def find_think_split(tokenizer, generated_ids):
    """Binary search for </think> split position in generated token ids."""
    gen_list = generated_ids.tolist()
    lo, hi = 1, len(gen_list)
    split_pos = None
    while lo <= hi:
        mid = (lo + hi) // 2
        partial = tokenizer.decode(gen_list[:mid], skip_special_tokens=False)
        if "</think>" in partial:
            split_pos = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return split_pos

def qwen_generate(tokenizer, model, question, return_hidden=False):
    prompt = (
        "You are answering an open-ended scientific reasoning question.\n\n"
        "/think\n\n"
        f"Question:\n{question}\n\n"
        "Think carefully about the scientific mechanisms, causal relations, "
        "and possible explanations. Then give the final answer."
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        **inputs,
        **GEN_CFG_THINKING,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_hidden_states=return_hidden,
    )
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    generated_ids   = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    generated_text  = tokenizer.decode(generated_ids, skip_special_tokens=False)
    split_pos       = find_think_split(tokenizer, generated_ids)

    if split_pos and split_pos < len(generated_ids) - 1:
        thinking     = tokenizer.decode(generated_ids[:split_pos], skip_special_tokens=True).strip()
        final_answer = tokenizer.decode(generated_ids[split_pos:], skip_special_tokens=True).strip()
        has_split    = True
    else:
        thinking     = ""
        final_answer = generated_text.strip()
        has_split    = False
        split_pos    = None

    result = dict(
        thinking=thinking,
        final_answer=final_answer,
        has_split=has_split,
        split_pos=split_pos,
        generated_ids=generated_ids,
        n_generated=len(generated_ids),
    )
    if return_hidden:
        result["hidden_states"] = outputs.hidden_states
    return result

# ══════════════════════════════════════════════════════════════════════════════
# 2A — Generate Qwen thinking for all 100 questions
# ══════════════════════════════════════════════════════════════════════════════
def run_2a(graph_rows, tokenizer, model):
    print("\n" + "="*60)
    print("2A: Generating Qwen thinking outputs (100 questions)")
    print("="*60)

    done_ids = set()
    if OUT_2A.exists():
        with open(OUT_2A) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except: pass
        print(f"  Resuming — {len(done_ids)} already done.")

    out_f = open(OUT_2A, "a", buffering=1)
    total = len(graph_rows)
    start = time.time()

    for i, row in enumerate(graph_rows):
        if row["id"] in done_ids:
            continue

        t0  = time.time()
        res = qwen_generate(tokenizer, model, row["question"], return_hidden=False)
        elapsed = time.time() - t0

        record = {
            "id":                  row["id"],
            "question_type":       row["question_type"],
            "question":            row["question"],
            "qwen_thinking":       res["thinking"],
            "qwen_final_answer":   res["final_answer"],
            "graph_brainstorm":    row.get("brainstorm", ""),
            "graph_graph":         row.get("graph", ""),
            "graph_patterns":      row.get("patterns", ""),
            "graph_synthesis":     row.get("synthesis", ""),
            "graph_answer":        row.get("llm_output", ""),
            "thinking_token_count":res["split_pos"] if res["split_pos"] else 0,
            "answer_token_count":  res["n_generated"] - (res["split_pos"] or 0),
            "has_think_split":     res["has_split"],
        }
        out_f.write(json.dumps(record) + "\n")

        done_count = i + 1
        eta = (time.time() - start) / done_count * (total - done_count)
        print(f"  [{done_count}/{total}] id={row['id']} "
              f"split={res['has_split']} "
              f"think_tok={record['thinking_token_count']} "
              f"ans_tok={record['answer_token_count']} "
              f"time={elapsed:.1f}s ETA={eta/60:.1f}min")

        del res
        torch.cuda.empty_cache()
        gc.collect()

    out_f.close()
    merged = load_jsonl(OUT_2A)
    print(f"  2A complete. {len(merged)} rows saved.")
    return merged

# ══════════════════════════════════════════════════════════════════════════════
# 2B — Text-level embedding similarity
# ══════════════════════════════════════════════════════════════════════════════
def run_2b(merged_rows, embed_model):
    print("\n" + "="*60)
    print("2B: Text-level embedding similarity")
    print("="*60)

    records = []
    for row in merged_rows:
        texts = [row["qwen_final_answer"],
                 row["qwen_thinking"],
                 row["graph_brainstorm"],
                 row["graph_graph"],
                 row["graph_patterns"],
                 row["graph_synthesis"],
                 row["graph_answer"]]
        embs = embed_model.encode(texts, normalize_embeddings=True,
                                   batch_size=16, show_progress_bar=False)
        ans_emb   = embs[0]
        stage_embs= embs[1:]

        sims = {col: round(float(np.dot(ans_emb, stage_embs[i])), 6)
                for i, (col, _) in enumerate(SIM_COLS)}

        vals = [sims[c] for c in SIM_KEYS]
        closest = SIM_KEYS[int(np.argmax(vals))].replace("sim_qwen_answer_to_", "")

        records.append({
            "id":            row["id"],
            "question_type": row["question_type"],
            "question":      row["question"],
            **sims,
            "closest_source": closest,
        })

    df = pd.DataFrame(records)
    df.to_csv(OUT_2B, index=False)
    print(f"  Saved {len(df)} rows → {OUT_2B}")
    print("  Closest source distribution:")
    print(df["closest_source"].value_counts().to_string())
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 2C — Hidden-state divergence (layer-wise)
# ══════════════════════════════════════════════════════════════════════════════
def run_2c(merged_rows, sim_df, tokenizer, model):
    print("\n" + "="*60)
    print("2C: Hidden-state divergence")
    print("="*60)

    done_ids = set()
    if OUT_2C.exists():
        existing = pd.read_csv(OUT_2C)
        done_ids = set(existing["id"].unique())
        print(f"  Resuming — {len(done_ids)} ids already done.")

    out_f = open(OUT_2C, "a", buffering=1)
    first_write = not Path(OUT_2C).exists() or os.path.getsize(OUT_2C) == 0
    header_written = not first_write

    closest_map = dict(zip(sim_df["id"], sim_df["closest_source"]))
    records_new = []

    for i, row in enumerate(merged_rows):
        if row["id"] in done_ids:
            continue
        if not row.get("has_think_split", False):
            print(f"  id={row['id']} skipping — no think split")
            continue

        print(f"  [{i+1}/{len(merged_rows)}] id={row['id']} ({row['question_type']})")

        res = qwen_generate(tokenizer, model, row["question"], return_hidden=True)

        if not res["has_split"]:
            print(f"    WARNING: no </think> split — skipping")
            del res; torch.cuda.empty_cache(); gc.collect()
            continue

        split_pos    = res["split_pos"]
        hidden_states= res["hidden_states"]
        n_steps      = len(hidden_states)
        n_layers     = len(hidden_states[0])

        thinking_steps = list(range(0, min(split_pos, n_steps)))
        answer_steps   = list(range(split_pos, n_steps))

        if not thinking_steps or not answer_steps:
            print(f"    WARNING: empty span — skipping")
            del res; torch.cuda.empty_cache(); gc.collect()
            continue

        print(f"    {len(thinking_steps)} thinking steps, {len(answer_steps)} answer steps, {n_layers} layers")

        for layer_idx in range(n_layers):
            t_vecs = [hidden_states[s][layer_idx][0, 0, :].float().cpu().numpy()
                      for s in thinking_steps]
            a_vecs = [hidden_states[s][layer_idx][0, 0, :].float().cpu().numpy()
                      for s in answer_steps]
            t_mean = np.mean(t_vecs, axis=0)
            a_mean = np.mean(a_vecs, axis=0)
            sim  = cosine_sim(t_mean, a_mean)
            dist = 1.0 - sim

            records_new.append({
                "id":                              row["id"],
                "question_type":                   row["question_type"],
                "question":                        row["question"][:100],
                "closest_source":                  closest_map.get(row["id"], ""),
                "layer":                           layer_idx,
                "thinking_answer_cosine_distance": round(dist, 6),
                "thinking_answer_cosine_similarity":round(sim, 6),
                "thinking_token_count":            len(thinking_steps),
                "answer_token_count":              len(answer_steps),
            })

        del res; torch.cuda.empty_cache(); gc.collect()

    if records_new:
        new_df = pd.DataFrame(records_new)
        if OUT_2C.exists() and os.path.getsize(OUT_2C) > 0:
            old_df = pd.read_csv(OUT_2C)
            full_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            full_df = new_df
        full_df.sort_values(["id", "layer"], inplace=True)
        full_df.to_csv(OUT_2C, index=False)
        print(f"  Saved {len(full_df)} rows → {OUT_2C}")
    else:
        full_df = pd.read_csv(OUT_2C) if OUT_2C.exists() else pd.DataFrame()

    return full_df

# ══════════════════════════════════════════════════════════════════════════════
# 2E — Question-level divergence summary
# ══════════════════════════════════════════════════════════════════════════════
def run_2e(hidden_df, sim_df, merged_rows):
    print("\n" + "="*60)
    print("2E: Question-level divergence summary")
    print("="*60)

    merged_map = {r["id"]: r for r in merged_rows}
    records = []

    for qid, grp in hidden_df.groupby("id"):
        grp = grp.sort_values("layer")
        dist = grp["thinking_answer_cosine_distance"].values
        layers = grp["layer"].values

        peak_idx   = int(np.argmax(dist))
        peak_layer = int(layers[peak_idx])
        peak_value = round(float(dist[peak_idx]), 6)

        early_mask = np.isin(layers, [7, 8, 9, 10])
        early_peak = round(float(dist[early_mask].max()) if early_mask.any() else 0.0, 6)

        l36_mask = layers == 36
        l36_val  = round(float(dist[l36_mask][0]) if l36_mask.any() else 0.0, 6)

        mean_div = round(float(dist.mean()), 6)
        closest  = grp["closest_source"].iloc[0]
        qt       = grp["question_type"].iloc[0]

        row_data = merged_map.get(qid, {})
        records.append({
            "id":                  qid,
            "question_type":       qt,
            "closest_source":      closest,
            "peak_layer":          peak_layer,
            "peak_value":          peak_value,
            "early_peak_7_10":     early_peak,
            "layer36_value":       l36_val,
            "mean_divergence":     mean_div,
            "thinking_token_count":int(grp["thinking_token_count"].iloc[0]),
            "answer_token_count":  int(grp["answer_token_count"].iloc[0]),
            "question":            row_data.get("question", "")[:200],
            "qwen_thinking_short": row_data.get("qwen_thinking", "")[:150],
            "qwen_answer_short":   row_data.get("qwen_final_answer", "")[:150],
        })

    df = pd.DataFrame(records).sort_values("layer36_value", ascending=False)
    df.to_csv(OUT_2E, index=False)
    print(f"  Saved {len(df)} rows → {OUT_2E}")
    print("\n  Top 5 by layer36_value:")
    print(df[["id","question_type","closest_source","peak_layer","layer36_value","early_peak_7_10"]].head().to_string(index=False))
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 2F — Question-type summary
# ══════════════════════════════════════════════════════════════════════════════
def run_2f(sim_df, summary_df):
    print("\n" + "="*60)
    print("2F: Question-type summary")
    print("="*60)

    merged = sim_df.merge(
        summary_df[["id","peak_layer","layer36_value","early_peak_7_10","mean_divergence"]],
        on="id", how="left"
    )
    merged["thinking_gap"]  = 1 - merged["sim_qwen_answer_to_qwen_thinking"]
    merged["graph_gap"]     = 1 - merged["sim_qwen_answer_to_graph_graph"]
    merged["synthesis_gap"] = 1 - merged["sim_qwen_answer_to_graph_synthesis"]
    merged["graph_ans_gap"] = 1 - merged["sim_qwen_answer_to_graph_answer"]

    agg = merged.groupby("question_type").agg(
        n_questions             =("id", "count"),
        most_common_closest     =("closest_source", lambda x: x.mode()[0]),
        avg_thinking_gap        =("thinking_gap",  "mean"),
        avg_graph_gap           =("graph_gap",     "mean"),
        avg_synthesis_gap       =("synthesis_gap", "mean"),
        avg_graph_answer_gap    =("graph_ans_gap", "mean"),
        avg_layer36_divergence  =("layer36_value", "mean"),
        avg_early_peak_7_10     =("early_peak_7_10","mean"),
    ).reset_index().round(4)

    agg.to_csv(OUT_2F, index=False)
    print(f"  Saved → {OUT_2F}")
    print(agg.to_string(index=False))
    return agg

# ══════════════════════════════════════════════════════════════════════════════
# 2D — Split divergence by backtracking group
# (uses hidden_df + closest_source — no extra model needed)
# ══════════════════════════════════════════════════════════════════════════════
def run_2d_plot(hidden_df):
    print("\n" + "="*60)
    print("2D: Split divergence by backtracking group")
    print("="*60)

    groups = hidden_df.groupby(["closest_source", "layer"])["thinking_answer_cosine_distance"]
    mean_g = groups.mean().reset_index()
    std_g  = groups.std().reset_index()
    merged = mean_g.merge(std_g, on=["closest_source","layer"], suffixes=("_mean","_std"))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors_map = {"qwen_thinking": PURPLE, "graph_answer": BLUE,
                  "graph_brainstorm": TEAL, "graph_synthesis": AMBER,
                  "graph_graph": CORAL, "graph_patterns": DPURPLE}

    for src, grp in merged.groupby("closest_source"):
        grp = grp.sort_values("layer")
        c = colors_map.get(src, GRAY)
        ax.plot(grp["layer"], grp["thinking_answer_cosine_distance_mean"],
                label=src, color=c, linewidth=2)
        ax.fill_between(grp["layer"],
                        grp["thinking_answer_cosine_distance_mean"] - grp["thinking_answer_cosine_distance_std"],
                        grp["thinking_answer_cosine_distance_mean"] + grp["thinking_answer_cosine_distance_std"],
                        alpha=0.12, color=c)

    ax.set_xlabel("transformer layer")
    ax.set_ylabel("cosine distance (thinking vs answer)")
    ax.set_title("2D — Layer-wise divergence by backtracking group")
    ax.legend(fontsize=9, framealpha=0.4, loc="upper right")
    out = PLOTS / "plot_2d_divergence_by_group.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 2G.1 — Linear probe: thinking vs answer token decodability
# ══════════════════════════════════════════════════════════════════════════════
def run_2g1(merged_rows, tokenizer, model):
    print("\n" + "="*60)
    print("2G.1: Linear probe — thinking vs answer token decodability")
    print("="*60)

    MAX_SAMPLES_PER_Q = 50   # balanced per question
    layer_data = defaultdict(lambda: {"X": [], "y": []})

    for i, row in enumerate(merged_rows[:30]):  # use first 30 for speed
        if not row.get("has_think_split"): continue
        print(f"  [{i+1}] id={row['id']}")

        res = qwen_generate(tokenizer, model, row["question"], return_hidden=True)
        if not res["has_split"]:
            del res; torch.cuda.empty_cache(); gc.collect()
            continue

        split_pos    = res["split_pos"]
        hidden_states= res["hidden_states"]
        n_steps      = len(hidden_states)
        n_layers     = len(hidden_states[0])

        t_steps = list(range(0, min(split_pos, n_steps)))
        a_steps = list(range(split_pos, n_steps))

        # Balance
        n_sample = min(MAX_SAMPLES_PER_Q, len(t_steps), len(a_steps))
        t_idx = np.random.choice(len(t_steps), n_sample, replace=False)
        a_idx = np.random.choice(len(a_steps), n_sample, replace=False)

        for layer_idx in range(n_layers):
            for si in t_idx:
                v = hidden_states[t_steps[si]][layer_idx][0, 0, :].float().cpu().numpy()
                layer_data[layer_idx]["X"].append(v)
                layer_data[layer_idx]["y"].append(0)
            for si in a_idx:
                v = hidden_states[a_steps[si]][layer_idx][0, 0, :].float().cpu().numpy()
                layer_data[layer_idx]["X"].append(v)
                layer_data[layer_idx]["y"].append(1)

        del res; torch.cuda.empty_cache(); gc.collect()

    print("  Training probes...")
    records = []
    for layer_idx in sorted(layer_data.keys()):
        X = np.array(layer_data[layer_idx]["X"])
        y = np.array(layer_data[layer_idx]["y"])
        if len(np.unique(y)) < 2: continue

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
        clf.fit(X_s, y)

        y_pred = clf.predict(X_s)
        y_prob = clf.predict_proba(X_s)[:, 1]
        acc  = round(float((y_pred == y).mean()), 4)
        auroc= round(float(roc_auc_score(y, y_prob)), 4)
        f1   = round(float(f1_score(y, y_pred)), 4)

        records.append({"layer": layer_idx, "accuracy": acc, "auroc": auroc, "f1": f1})
        if layer_idx % 5 == 0:
            print(f"    layer {layer_idx:2d}: acc={acc:.3f} auroc={auroc:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(OUT_PROBE, index=False)
    print(f"  Saved → {OUT_PROBE}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 2G.2 — Logit lens: token category probability mass
# ══════════════════════════════════════════════════════════════════════════════
def run_2g2(merged_rows, tokenizer, model):
    print("\n" + "="*60)
    print("2G.2: Logit lens — token category probability mass")
    print("="*60)

    TOKEN_CATEGORIES = {
        "reasoning":    ["because", "therefore", "maybe", "consider", "if", "however"],
        "answer":       ["thus", "summary", "answer", "suggests", "conclusion"],
        "causal":       ["causes", "leads", "mechanism", "due", "results"],
        "uncertainty":  ["may", "could", "might", "possible", "perhaps"],
        "contrast":     ["however", "although", "whereas", "tradeoff", "but"],
    }

    # Pre-compute token ids for each category word
    cat_ids = {}
    for cat, words in TOKEN_CATEGORIES.items():
        ids = set()
        for w in words:
            for variant in [w, " " + w, w.capitalize(), " " + w.capitalize()]:
                tids = tokenizer.encode(variant, add_special_tokens=False)
                ids.update(tids)
        cat_ids[cat] = list(ids)

    INSPECT_LAYERS = [6, 7, 8, 9, 10, 35, 36]
    lm_head = model.lm_head
    records = []

    for i, row in enumerate(merged_rows[:20]):  # first 20
        if not row.get("has_think_split"): continue
        print(f"  [{i+1}] id={row['id']}")

        res = qwen_generate(tokenizer, model, row["question"], return_hidden=True)
        if not res["has_split"]:
            del res; torch.cuda.empty_cache(); gc.collect()
            continue

        split_pos    = res["split_pos"]
        hidden_states= res["hidden_states"]
        n_steps      = len(hidden_states)

        t_steps = list(range(0, min(split_pos, n_steps)))
        a_steps = list(range(split_pos, n_steps))
        if not t_steps or not a_steps:
            del res; torch.cuda.empty_cache(); gc.collect()
            continue

        closest = row.get("closest_source", "")

        for layer_idx in INSPECT_LAYERS:
            if layer_idx >= len(hidden_states[0]): continue

            for span_name, steps in [("thinking", t_steps), ("answer", a_steps)]:
                # Pool hidden states over span
                vecs = [hidden_states[s][layer_idx][0, 0, :].float()
                        for s in steps[:50]]  # cap at 50 steps
                mean_h = torch.stack(vecs).mean(0).to(model.device)

                with torch.no_grad():
                    logits = lm_head(mean_h.unsqueeze(0))  # (1, vocab)
                    probs  = torch.softmax(logits[0], dim=-1).cpu().numpy()

                for cat, ids in cat_ids.items():
                    valid_ids = [tid for tid in ids if tid < len(probs)]
                    mass = float(probs[valid_ids].sum()) if valid_ids else 0.0
                    records.append({
                        "id":              row["id"],
                        "question_type":   row["question_type"],
                        "closest_source":  closest,
                        "layer":           layer_idx,
                        "span_type":       span_name,
                        "category":        cat,
                        "probability_mass":round(mass, 8),
                    })

        del res; torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(records)
    df.to_csv(OUT_LOGIT, index=False)
    print(f"  Saved {len(df)} rows → {OUT_LOGIT}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Plots (2B, 2C, 2D, 2E, 2F, 2G.1, 2G.2)
# ══════════════════════════════════════════════════════════════════════════════
def make_all_plots(sim_df, hidden_df, summary_df, qtype_df, probe_df, logit_df):
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)

    # ── Plot 1: Backtracking distribution ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    vc = sim_df["closest_source"].value_counts()
    colors = [{"qwen_thinking": PURPLE, "graph_answer": BLUE,
               "graph_brainstorm": TEAL, "graph_synthesis": AMBER,
               "graph_graph": CORAL, "graph_patterns": DPURPLE}.get(k, GRAY)
              for k in vc.index]
    bars = ax.barh(vc.index, vc.values, color=colors, height=0.55, zorder=3)
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("number of questions (n=100)")
    ax.set_title("Plot 1 — Where does Qwen's answer backtrack? (n=100)")
    ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", visible=False)
    plt.tight_layout()
    plt.savefig(PLOTS / "plot1_backtrack_distribution.png"); plt.close()

    # ── Plot 2: Mean ± std similarity ─────────────────────────────────────────
    means = [sim_df[c].mean() for c in SIM_KEYS]
    stds  = [sim_df[c].std()  for c in SIM_KEYS]
    order = sorted(range(len(means)), key=lambda i: -means[i])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = range(len(order))
    ax.barh([SIM_LABELS[i] for i in order],
            [means[i] for i in order],
            xerr=[stds[i]  for i in order],
            color=[SIM_COLORS[i] for i in order],
            height=0.55, capsize=4, zorder=3)
    ax.set_xlabel("mean cosine similarity ± std")
    ax.set_title("Plot 2 — Mean similarity: Qwen answer vs reasoning sources (n=100)")
    ax.set_xlim(0.65, 1.0)
    ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", visible=False)
    plt.tight_layout()
    plt.savefig(PLOTS / "plot2_mean_similarities.png"); plt.close()

    # ── Plot 3: Per-question heatmap ───────────────────────────────────────────
    df_sorted = sim_df.sort_values(["question_type", "id"]).reset_index(drop=True)
    matrix = df_sorted[SIM_KEYS].values
    cmap = LinearSegmentedColormap.from_list("teal", ["#E1F5EE","#1D9E75","#04342C"])
    fig, ax = plt.subplots(figsize=(10, max(8, len(df_sorted)*0.18)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.65, vmax=0.97)
    ax.set_xticks(range(len(SIM_LABELS)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in SIM_LABELS], fontsize=9)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([f"{r.id} {r.question_type[:18]}" for _, r in df_sorted.iterrows()],
                       fontsize=7)
    # Bold marker for closest source
    for i, row in df_sorted.iterrows():
        j = SIM_KEYS.index(f"sim_qwen_answer_to_{row['closest_source']}")
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                     fill=False, edgecolor="white", linewidth=2))
    plt.colorbar(im, ax=ax, shrink=0.5, label="cosine similarity")
    ax.set_title("Plot 3 — Per-question similarity heatmap (white box = closest source)")
    plt.tight_layout()
    plt.savefig(PLOTS / "plot3_similarity_heatmap.png", dpi=120); plt.close()

    # ── Plot 4: Question-type backtracking gaps ────────────────────────────────
    metrics = [("avg_thinking_gap","thinking gap",BLUE),
               ("avg_graph_gap","graph gap",CORAL),
               ("avg_synthesis_gap","synthesis gap",TEAL),
               ("avg_graph_answer_gap","graph answer gap",PURPLE)]
    qt_labels = [s.replace("_"," ").replace("and","&") for s in qtype_df["question_type"]]
    x = np.arange(len(qtype_df)); w = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (col, label, color) in enumerate(metrics):
        if col not in qtype_df.columns: continue
        ax.bar(x + (i - 1.5)*w, qtype_df[col], width=w,
               label=label, color=color, zorder=3, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(qt_labels, fontsize=9)
    ax.set_ylabel("gap (1 − similarity)"); ax.legend(fontsize=10)
    ax.set_title("Plot 4 — Question-type backtracking gaps (n=100)")
    ax.grid(axis="y", alpha=0.25); ax.grid(axis="x", visible=False)
    plt.tight_layout()
    plt.savefig(PLOTS / "plot4_qtype_gaps.png"); plt.close()

    # ── Plot 5: Layer-wise divergence mean ± std ───────────────────────────────
    layer_stats = hidden_df.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for qid, grp in hidden_df.groupby("id"):
        g = grp.sort_values("layer")
        ax.plot(g["layer"], g["thinking_answer_cosine_distance"],
                color=BLUE, alpha=0.07, linewidth=0.8)
    ax.plot(layer_stats["layer"], layer_stats["mean"],
            color=BLUE, linewidth=2.5, label="mean", zorder=4)
    ax.fill_between(layer_stats["layer"],
                    layer_stats["mean"] - layer_stats["std"],
                    layer_stats["mean"] + layer_stats["std"],
                    alpha=0.15, color=BLUE, label="± 1 std")
    peak_row = layer_stats.loc[layer_stats["mean"].idxmax()]
    ax.annotate(f"layer {int(peak_row['layer'])} (peak)",
                xy=(peak_row["layer"], peak_row["mean"]),
                xytext=(peak_row["layer"]-5, peak_row["mean"]+0.02),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5),
                fontsize=9, color=CORAL, fontweight="bold")
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("cosine distance (thinking vs answer)")
    ax.set_title("Plot 5 — Layer-wise thinking–answer divergence (n=100, mean ± std)")
    ax.legend(fontsize=10, framealpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOTS / "plot5_layer_divergence.png"); plt.close()

    # ── Plot 6: Layer36 value distribution ────────────────────────────────────
    if not summary_df.empty and "layer36_value" in summary_df.columns:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(summary_df["layer36_value"], bins=20, color=BLUE, edgecolor="white", zorder=3)
        ax.axvline(summary_df["layer36_value"].mean(), color=CORAL, linestyle="--",
                   label=f"mean={summary_df['layer36_value'].mean():.3f}")
        ax.set_xlabel("layer 36 cosine distance")
        ax.set_ylabel("count"); ax.legend()
        ax.set_title("Plot 6 — Distribution of layer-36 divergence across questions")
        plt.tight_layout()
        plt.savefig(PLOTS / "plot6_layer36_distribution.png"); plt.close()

    # ── Plot 7: Linear probe AUROC ─────────────────────────────────────────────
    if not probe_df.empty:
        # Get mean divergence per layer to overlay
        layer_mean_div = hidden_df.groupby("layer")["thinking_answer_cosine_distance"].mean()

        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        ax2 = ax1.twinx()
        ax1.plot(probe_df["layer"], probe_df["auroc"],
                 color=PURPLE, linewidth=2.5, label="probe AUROC")
        ax1.fill_between(probe_df["layer"], 0.5, probe_df["auroc"],
                         alpha=0.12, color=PURPLE)
        ax2.plot(layer_mean_div.index, layer_mean_div.values,
                 color=BLUE, linewidth=1.5, linestyle="--", alpha=0.7, label="mean divergence")
        ax1.set_xlabel("transformer layer")
        ax1.set_ylabel("probe AUROC", color=PURPLE)
        ax2.set_ylabel("mean cosine distance", color=BLUE)
        ax1.set_title("Plot 7 — Probe AUROC vs divergence by layer (2G.1)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, fontsize=10, framealpha=0.4)
        plt.tight_layout()
        plt.savefig(PLOTS / "plot7_probe_auroc.png"); plt.close()

    # ── Plot 8: Logit lens heatmap ─────────────────────────────────────────────
    if not logit_df.empty:
        pivot = logit_df.groupby(["layer","category","span_type"])["probability_mass"].mean().unstack("span_type")
        if "thinking" in pivot.columns and "answer" in pivot.columns:
            pivot["diff"] = pivot["answer"] - pivot["thinking"]
            diff_mat = pivot["diff"].unstack("category").fillna(0)

            fig, ax = plt.subplots(figsize=(9, 4))
            cmap2 = LinearSegmentedColormap.from_list("div", ["#D85A30","white","#378ADD"])
            vmax = max(abs(diff_mat.values.min()), abs(diff_mat.values.max()))
            im = ax.imshow(diff_mat.values, aspect="auto", cmap=cmap2, vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(diff_mat.columns)))
            ax.set_xticklabels(diff_mat.columns, fontsize=10)
            ax.set_yticks(range(len(diff_mat.index)))
            ax.set_yticklabels([f"layer {l}" for l in diff_mat.index], fontsize=10)
            plt.colorbar(im, ax=ax, label="answer − thinking probability mass")
            ax.set_title("Plot 8 — Logit lens: answer vs thinking token category shift (2G.2)")
            plt.tight_layout()
            plt.savefig(PLOTS / "plot8_logit_lens_heatmap.png"); plt.close()

    print(f"\n  All plots saved to {PLOTS}")
    for f in sorted(PLOTS.iterdir()):
        if f.suffix == ".png":
            print(f"    {f.name}  ({f.stat().st_size//1024} KB)")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load graph data ────────────────────────────────────────────────────────
    print(f"Loading {GRAPH_FILE}")
    graph_rows = load_jsonl(GRAPH_FILE)
    print(f"  {len(graph_rows)} rows loaded.")

    # ── Load Qwen model ────────────────────────────────────────────────────────
    print(f"\nLoading {QWEN_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded.")

    # ── 2A ────────────────────────────────────────────────────────────────────
    merged_rows = run_2a(graph_rows, tokenizer, model)

    # ── 2B (CPU — no GPU needed) ───────────────────────────────────────────────
    print("\nLoading embedding model ...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    sim_df = run_2b(merged_rows, embed_model) if not OUT_2B.exists() else pd.read_csv(OUT_2B)

    # ── 2C ────────────────────────────────────────────────────────────────────
    hidden_df = run_2c(merged_rows, sim_df, tokenizer, model)

    # Free GPU after hidden-state work
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 2E, 2F ────────────────────────────────────────────────────────────────
    summary_df = run_2e(hidden_df, sim_df, merged_rows) if not hidden_df.empty else pd.DataFrame()
    qtype_df   = run_2f(sim_df, summary_df) if not summary_df.empty else pd.DataFrame()

    # ── 2D plot (no extra model needed) ───────────────────────────────────────
    if not hidden_df.empty:
        run_2d_plot(hidden_df)

    # ── Reload model for 2G.1 and 2G.2 ───────────────────────────────────────
    print("\nReloading model for 2G...")
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # Attach closest_source to merged_rows for logit lens
    cs_map = dict(zip(sim_df["id"], sim_df["closest_source"]))
    for r in merged_rows:
        r["closest_source"] = cs_map.get(r["id"], "")

    probe_df = run_2g1(merged_rows, tokenizer, model)
    logit_df = run_2g2(merged_rows, tokenizer, model)

    del model; torch.cuda.empty_cache(); gc.collect()

    # ── All plots ──────────────────────────────────────────────────────────────
    make_all_plots(sim_df, hidden_df, summary_df, qtype_df, probe_df, logit_df)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    for f in [OUT_2A, OUT_2B, OUT_2C, OUT_2E, OUT_2F, OUT_PROBE, OUT_LOGIT]:
        exists = "✓" if Path(f).exists() else "✗"
        print(f"  {exists}  {f}")

if __name__ == "__main__":
    main()
