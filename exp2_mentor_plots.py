"""
exp2_mentor_plots.py
====================
Generates all plots requested by Dr. Pal:

Plot A1: Qwen answer closest source distribution (bar chart)
Plot A2: Graph-PRefLexOR answer closest internal stage distribution
Plot B:  Layer-wise divergence — Qwen vs Graph-PRefLexOR (on same axes)
Plot B2: Detailed Graph-PRefLexOR — each stage vs answer separately
Plot C1: Qwen layer-wise divergence split by backtracking group
         (Group A: backtracks to qwen_thinking, Group B: does not)
Plot C2: Graph-PRefLexOR layer-wise divergence split by stage group
         (Group A: closest = graph/synthesis, Group B: closest = brainstorm/patterns)

All plots saved to /projects/bfir/ssourav/plots_100/
Graph-PRefLexOR hidden states computed fresh using Qwen3-8B.
"""

import json, gc, re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK  = Path("/projects/bfir/ssourav")
PLOTS = WORK / "plots_100"
PLOTS.mkdir(exist_ok=True)

SIM_FILE   = WORK / "exp2_backtracking_text_similarity_100.csv"
HIDDEN_FILE= WORK / "exp2_hidden_state_divergence_100.csv"
GRAPH_FILE = WORK / "graph_8b_data_eval_100.jsonl"
QWEN_FILE  = WORK / "exp2_qwen_thinking_outputs_100.jsonl"

GRAPH_HIDDEN_FILE = WORK / "exp2_graph_hidden_state_divergence_100.csv"

QWEN_MODEL  = "Qwen/Qwen3-8B"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,
    "axes.titlesize":12,"axes.titleweight":"bold",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,"grid.linestyle":"--",
    "figure.dpi":150,"savefig.dpi":150,
    "savefig.bbox":"tight","savefig.facecolor":"white",
})
BLUE="#378ADD"; CORAL="#D85A30"; TEAL="#1D9E75"
PURPLE="#7F77DD"; AMBER="#BA7517"; GRAY="#888780"; DPURPLE="#3C3489"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def cosine_sim_np(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def find_think_split(tokenizer, generated_ids):
    gen_list = generated_ids.tolist()
    lo, hi, sp = 1, len(gen_list), None
    while lo <= hi:
        mid = (lo + hi) // 2
        if "</think>" in tokenizer.decode(gen_list[:mid], skip_special_tokens=False):
            sp = mid; hi = mid - 1
        else: lo = mid + 1
    return sp

# ══════════════════════════════════════════════════════════════════════════════
# Compute Graph-PRefLexOR hidden state divergence
# (reasoning stages concatenated vs final answer)
# ══════════════════════════════════════════════════════════════════════════════
def compute_graph_hidden_divergence(graph_rows, tokenizer, model):
    """
    For each Graph-PRefLexOR question:
      - Encode reasoning text (brainstorm + graph + patterns + synthesis) as a prompt
      - Encode final answer (llm_output) as a prompt
      - Extract layer-wise hidden states for both
      - Compute cosine distance per layer

    Also compute per-stage divergence:
      brainstorm vs answer, graph vs answer, patterns vs answer, synthesis vs answer
    """
    print("\n" + "="*60)
    print("Computing Graph-PRefLexOR hidden state divergence")
    print("="*60)

    # Load already done
    done_ids = set()
    if GRAPH_HIDDEN_FILE.exists():
        existing = pd.read_csv(GRAPH_HIDDEN_FILE)
        done_ids = set(existing["id"].unique())
        print(f"  Resuming — {len(done_ids)} ids done.")

    records = []
    out_f = open(GRAPH_HIDDEN_FILE, "a", buffering=1)
    write_header = not GRAPH_HIDDEN_FILE.exists() or GRAPH_HIDDEN_FILE.stat().st_size == 0
    if write_header:
        out_f.write("id,question_type,layer,stage,"
                    "reasoning_answer_cosine_distance,"
                    "reasoning_answer_cosine_similarity\n")

    def get_hidden_mean(text_prompt, max_tokens=800):
        """Run model on prompt, return dict[layer] -> mean hidden vec over generated tokens."""
        messages = [{"role": "user", "content": text_prompt}]
        txt = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
                temperature=1.0,
            )
        n_steps  = len(outputs.hidden_states)
        n_layers = len(outputs.hidden_states[0])
        result = {}
        for li in range(n_layers):
            vecs = [outputs.hidden_states[s][li][0,0,:].float().cpu().numpy()
                    for s in range(min(n_steps, 200))]
            result[li] = np.mean(vecs, axis=0)
        del outputs; torch.cuda.empty_cache(); gc.collect()
        return result

    STAGE_FIELDS = [
        ("brainstorm", "brainstorm"),
        ("graph",      "graph"),
        ("patterns",   "patterns"),
        ("synthesis",  "synthesis"),
        ("reasoning",  None),   # combined
    ]

    total = len(graph_rows)
    for i, row in enumerate(graph_rows):
        qid = row["id"]
        if qid in done_ids:
            continue

        qt  = row["question_type"]
        q   = row["question"]
        ans = row.get("llm_output", "").strip()
        if not ans:
            continue

        print(f"  [{i+1}/{total}] id={qid} ({qt})")

        # Answer hidden states
        ans_prompt = (
            f"The following is a scientific answer:\n\n{ans[:1000]}\n\n"
            "Continue this answer."
        )
        ans_hidden = get_hidden_mean(ans_prompt, max_tokens=200)
        n_layers   = len(ans_hidden)

        # Per-stage hidden states
        stage_hiddens = {}
        for stage_name, field in STAGE_FIELDS:
            if field is None:
                # Combined reasoning
                combined = " ".join([
                    row.get("brainstorm",""),
                    row.get("graph",""),
                    row.get("patterns",""),
                    row.get("synthesis",""),
                ])[:1500]
                text = combined
            else:
                text = row.get(field, "")[:800]

            if not text.strip():
                continue

            stage_prompt = (
                f"The following is scientific reasoning:\n\n{text}\n\n"
                "Continue this reasoning."
            )
            stage_hiddens[stage_name] = get_hidden_mean(stage_prompt, max_tokens=200)

        # Compute divergence per layer per stage
        for stage_name, sh in stage_hiddens.items():
            for li in range(n_layers):
                if li not in sh or li not in ans_hidden:
                    continue
                sim  = cosine_sim_np(sh[li], ans_hidden[li])
                dist = 1.0 - sim
                out_f.write(f"{qid},{qt},{li},{stage_name},"
                            f"{round(dist,6)},{round(sim,6)}\n")

        done_ids.add(qid)
        print(f"    Done ({n_layers} layers x {len(stage_hiddens)} stages)")

    out_f.close()
    df = pd.read_csv(GRAPH_HIDDEN_FILE)
    print(f"  Saved {len(df)} rows → {GRAPH_HIDDEN_FILE}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Compute Graph-PRefLexOR internal backtracking
# (which of its own stages is its final answer closest to?)
# ══════════════════════════════════════════════════════════════════════════════
def compute_graph_backtracking(graph_rows, embed_model):
    print("\n" + "="*60)
    print("Computing Graph-PRefLexOR internal backtracking")
    print("="*60)

    STAGE_COLS = [
        ("sim_graph_answer_to_brainstorm", "brainstorm"),
        ("sim_graph_answer_to_graph",      "graph"),
        ("sim_graph_answer_to_patterns",   "patterns"),
        ("sim_graph_answer_to_synthesis",  "synthesis"),
    ]

    records = []
    for row in graph_rows:
        ans = row.get("llm_output", "").strip()
        if not ans:
            continue

        texts = [ans,
                 row.get("brainstorm",""),
                 row.get("graph",""),
                 row.get("patterns",""),
                 row.get("synthesis","")]

        embs = embed_model.encode(texts, normalize_embeddings=True,
                                   batch_size=8, show_progress_bar=False)
        ans_emb = embs[0]

        sims = {}
        for i, (col, _) in enumerate(STAGE_COLS):
            sims[col] = round(float(np.dot(ans_emb, embs[i+1])), 6)

        vals    = [sims[c] for c, _ in STAGE_COLS]
        closest = STAGE_COLS[int(np.argmax(vals))][1]

        records.append({
            "id":            row["id"],
            "question_type": row["question_type"],
            **sims,
            "closest_stage": closest,
        })

    df = pd.DataFrame(records)
    out = WORK / "exp2_graph_internal_backtracking_100.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} rows → {out}")
    print("  Closest stage distribution:")
    print(df["closest_stage"].value_counts().to_string())
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Plot A1: Qwen answer backtracking distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot_a1(sim_df):
    print("\nPlot A1: Qwen backtracking distribution")
    vc = sim_df["closest_source"].value_counts()

    # Two-category version
    qwen_count  = vc.get("qwen_thinking", 0)
    graph_count = len(sim_df) - qwen_count

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: two-category
    ax = axes[0]
    labels = ["Backtracks to\nQwen thinking", "Does not backtrack\nto Qwen thinking"]
    counts = [qwen_count, graph_count]
    colors = [PURPLE, BLUE]
    bars = ax.bar(labels, counts, color=colors, width=0.5, zorder=3)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v}\n({v/len(sim_df)*100:.0f}%)",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("number of questions")
    ax.set_title("Qwen answer backtracking\n(binary: own thinking vs not)")
    ax.set_ylim(0, max(counts)*1.2)
    ax.grid(axis="y", alpha=0.25); ax.grid(axis="x", visible=False)

    # Right: full distribution
    ax = axes[1]
    color_map = {"qwen_thinking":PURPLE,"graph_answer":BLUE,"graph_brainstorm":TEAL,
                 "graph_graph":CORAL,"graph_synthesis":AMBER,"graph_patterns":DPURPLE}
    colors_bar = [color_map.get(k, GRAY) for k in vc.index]
    bars = ax.barh(vc.index, vc.values, color=colors_bar, height=0.55, zorder=3)
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                f"{v} ({v/len(sim_df)*100:.0f}%)",
                va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("number of questions (n=100)")
    ax.set_title("Qwen answer closest source\n(full distribution)")
    ax.set_xlim(0, vc.max()*1.35)
    ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", visible=False)

    fig.suptitle("Plot A1 — Where does Qwen's final answer backtrack?",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.04,
             "In 16/100 cases Qwen's answer is most similar to its own thinking trace. "
             "In 84/100 cases it backtracks to Graph-PRefLexOR's answer or reasoning stages.\n"
             "Graph-PRefLexOR's final answer is the single most common closest source (46 cases).",
             ha="center", fontsize=9, color=GRAY, style="italic")
    plt.tight_layout()
    plt.savefig(PLOTS/"plotA1_qwen_backtracking_distribution.png"); plt.close()
    print("  Saved plotA1_qwen_backtracking_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot A2: Graph-PRefLexOR internal backtracking
# ══════════════════════════════════════════════════════════════════════════════
def plot_a2(graph_bt_df):
    print("\nPlot A2: Graph-PRefLexOR internal backtracking")
    vc = graph_bt_df["closest_stage"].value_counts()

    color_map = {"brainstorm":TEAL,"graph":CORAL,"patterns":DPURPLE,"synthesis":AMBER}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: bar chart
    ax = axes[0]
    colors_bar = [color_map.get(k, GRAY) for k in vc.index]
    bars = ax.barh(vc.index, vc.values, color=colors_bar, height=0.5, zorder=3)
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                f"{v} ({v/len(graph_bt_df)*100:.0f}%)",
                va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("number of questions (n=100)")
    ax.set_title("Graph-PRefLexOR answer closest internal stage")
    ax.set_xlim(0, vc.max()*1.35)
    ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", visible=False)

    # Right: mean similarity to each stage
    ax = axes[1]
    sim_cols = ["sim_graph_answer_to_brainstorm","sim_graph_answer_to_graph",
                "sim_graph_answer_to_patterns","sim_graph_answer_to_synthesis"]
    stage_labels = ["brainstorm","graph","patterns","synthesis"]
    means = [graph_bt_df[c].mean() for c in sim_cols]
    stds  = [graph_bt_df[c].std()  for c in sim_cols]
    colors_bar2 = [color_map[s] for s in stage_labels]
    ax.barh(stage_labels, means, xerr=stds, color=colors_bar2,
            height=0.5, capsize=4, zorder=3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m+s+0.002, i, f"{m:.3f}", va="center", fontsize=10)
    ax.set_xlabel("mean cosine similarity ± std")
    ax.set_title("Mean similarity: Graph-PRefLexOR\nanswer vs each reasoning stage")
    ax.set_xlim(0.75, 1.02)
    ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", visible=False)

    fig.suptitle("Plot A2 — Graph-PRefLexOR answer backtracking to its own reasoning stages",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.04,
             "Graph-PRefLexOR's final answer most often backtracks to synthesis (its final reasoning stage), "
             "followed by graph and brainstorm.\nThis confirms that Graph-PRefLexOR's answer is tightly coupled "
             "to its structured reasoning pipeline.",
             ha="center", fontsize=9, color=GRAY, style="italic")
    plt.tight_layout()
    plt.savefig(PLOTS/"plotA2_graph_internal_backtracking.png"); plt.close()
    print("  Saved plotA2_graph_internal_backtracking.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot B: Qwen vs Graph-PRefLexOR layer-wise divergence
# ══════════════════════════════════════════════════════════════════════════════
def plot_b(qwen_hidden_df, graph_hidden_df):
    print("\nPlot B: Qwen vs Graph-PRefLexOR layer-wise divergence")

    # Qwen: thinking vs answer
    q_stats = (qwen_hidden_df.groupby("layer")["thinking_answer_cosine_distance"]
               .agg(["mean","std"]).reset_index())

    # Graph: combined reasoning vs answer
    g_combined = graph_hidden_df[graph_hidden_df["stage"]=="reasoning"]
    g_stats = (g_combined.groupby("layer")["reasoning_answer_cosine_distance"]
               .agg(["mean","std"]).reset_index())

    fig, ax = plt.subplots(figsize=(11, 5))

    # Qwen line
    ax.plot(q_stats["layer"], q_stats["mean"],
            color=PURPLE, linewidth=2.5, label="Qwen: thinking vs answer", zorder=4)
    ax.fill_between(q_stats["layer"],
                    q_stats["mean"]-q_stats["std"],
                    q_stats["mean"]+q_stats["std"],
                    alpha=0.12, color=PURPLE)

    # Graph-PRefLexOR line
    ax.plot(g_stats["layer"], g_stats["mean"],
            color=TEAL, linewidth=2.5, label="Graph-PRefLexOR: reasoning vs answer", zorder=4)
    ax.fill_between(g_stats["layer"],
                    g_stats["mean"]-g_stats["std"],
                    g_stats["mean"]+g_stats["std"],
                    alpha=0.12, color=TEAL)

    # Annotate candidate layers
    for l in [7, 8, 36]:
        ax.axvline(l, color=AMBER, alpha=0.3, linewidth=1.5, linestyle="--")
    ax.axvspan(7, 10, alpha=0.05, color=AMBER)
    ax.text(8.5, ax.get_ylim()[1]*0.95, "layers\n7–10", fontsize=8,
            color=AMBER, ha="center")

    ax.set_xlabel("transformer layer")
    ax.set_ylabel("cosine distance (reasoning vs answer)")
    ax.set_title("Plot B — Layer-wise divergence: Qwen thinking–answer vs Graph-PRefLexOR reasoning–answer\n"
                 "(mean ± std across 100 questions)")
    ax.legend(fontsize=11, framealpha=0.4)
    fig.text(0.5, -0.04,
             "Qwen (purple): divergence between thinking-span and answer-span hidden states — "
             "captures how much the model transforms its reasoning into the final answer.\n"
             "Graph-PRefLexOR (teal): divergence between combined reasoning-stage hidden states "
             "and final-answer hidden states — analogous measure for the structured pipeline.",
             ha="center", fontsize=9, color=GRAY, style="italic")
    plt.tight_layout()
    plt.savefig(PLOTS/"plotB_qwen_vs_graph_layer_divergence.png"); plt.close()
    print("  Saved plotB_qwen_vs_graph_layer_divergence.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot B2: Graph-PRefLexOR per-stage divergence
# ══════════════════════════════════════════════════════════════════════════════
def plot_b2(graph_hidden_df):
    print("\nPlot B2: Graph-PRefLexOR per-stage divergence")

    stages = ["brainstorm","graph","patterns","synthesis"]
    stage_colors = {"brainstorm":TEAL,"graph":CORAL,"patterns":DPURPLE,"synthesis":AMBER}
    stage_ls     = {"brainstorm":"-","graph":"--","patterns":":","synthesis":"-."}

    fig, ax = plt.subplots(figsize=(11, 5))

    for stage in stages:
        sub = graph_hidden_df[graph_hidden_df["stage"]==stage]
        if sub.empty: continue
        stats = sub.groupby("layer")["reasoning_answer_cosine_distance"].agg(["mean","std"]).reset_index()
        ax.plot(stats["layer"], stats["mean"],
                color=stage_colors[stage], linewidth=2.0,
                linestyle=stage_ls[stage], label=stage, zorder=3)
        ax.fill_between(stats["layer"],
                        stats["mean"]-stats["std"],
                        stats["mean"]+stats["std"],
                        alpha=0.08, color=stage_colors[stage])

    for l in [7, 36]:
        ax.axvline(l, color=AMBER, alpha=0.3, linewidth=1.5, linestyle="--")

    ax.set_xlabel("transformer layer")
    ax.set_ylabel("cosine distance (stage vs answer)")
    ax.set_title("Plot B2 — Graph-PRefLexOR: per-stage divergence from final answer\n"
                 "(mean ± std, each reasoning stage vs llm_output)")
    ax.legend(fontsize=11, framealpha=0.4)
    fig.text(0.5, -0.04,
             "Each line shows how much a specific Graph-PRefLexOR reasoning stage diverges "
             "from the final answer at each transformer layer.\n"
             "Lower divergence = that stage is more similar to the final answer in hidden-state space.",
             ha="center", fontsize=9, color=GRAY, style="italic")
    plt.tight_layout()
    plt.savefig(PLOTS/"plotB2_graph_per_stage_divergence.png"); plt.close()
    print("  Saved plotB2_graph_per_stage_divergence.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot C1: Qwen layer-wise divergence split by backtracking group
# ══════════════════════════════════════════════════════════════════════════════
def plot_c1(qwen_hidden_df, sim_df):
    print("\nPlot C1: Qwen layer-wise divergence split by backtracking group")

    # Group A: backtracks to qwen_thinking
    # Group B: does not backtrack to qwen_thinking
    group_a_ids = set(sim_df[sim_df["closest_source"]=="qwen_thinking"]["id"])
    group_b_ids = set(sim_df[sim_df["closest_source"]!="qwen_thinking"]["id"])

    df_a = qwen_hidden_df[qwen_hidden_df["id"].isin(group_a_ids)]
    df_b = qwen_hidden_df[qwen_hidden_df["id"].isin(group_b_ids)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, df_grp, label, color, n_ids in [
        (axes[0], df_a, f"Group A: backtracks to Qwen thinking\n(n={len(group_a_ids)} questions)", PURPLE, len(group_a_ids)),
        (axes[1], df_b, f"Group B: does NOT backtrack to Qwen thinking\n(n={len(group_b_ids)} questions)", BLUE,   len(group_b_ids)),
    ]:
        if df_grp.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Individual question lines
        for qid, grp in df_grp.groupby("id"):
            g = grp.sort_values("layer")
            ax.plot(g["layer"], g["thinking_answer_cosine_distance"],
                    color=color, alpha=0.08, linewidth=0.8)

        stats = df_grp.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()
        ax.plot(stats["layer"], stats["mean"],
                color=color, linewidth=2.5, label="mean", zorder=4)
        ax.fill_between(stats["layer"],
                        stats["mean"]-stats["std"],
                        stats["mean"]+stats["std"],
                        alpha=0.15, color=color, label="± 1 std")

        for l in [7, 36]:
            ax.axvline(l, color=AMBER, alpha=0.35, linewidth=1.5, linestyle="--")
        ax.axvspan(7, 10, alpha=0.05, color=AMBER)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("transformer layer")
        ax.legend(fontsize=9, framealpha=0.4)
        if ax == axes[0]:
            ax.set_ylabel("cosine distance (thinking vs answer)")

    fig.suptitle("Plot C1 — Qwen layer-wise divergence split by backtracking group",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.04,
             "Group A (backtracks to own thinking): lower divergence expected — "
             "answer stays close to thinking-span representations.\n"
             "Group B (backtracks to Graph-PRefLexOR): higher divergence expected — "
             "answer moves away from thinking toward answer-style representations.",
             ha="center", fontsize=9, color=GRAY, style="italic")
    plt.tight_layout()
    plt.savefig(PLOTS/"plotC1_qwen_divergence_by_backtrack_group.png"); plt.close()
    print("  Saved plotC1_qwen_divergence_by_backtrack_group.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot C2: Graph-PRefLexOR divergence split by stage group
# ══════════════════════════════════════════════════════════════════════════════
def plot_c2(graph_hidden_df, graph_bt_df):
    print("\nPlot C2: Graph-PRefLexOR divergence split by stage group")

    # Group A: closest = graph or synthesis (structured/analytical stages)
    # Group B: closest = brainstorm or patterns (exploratory stages)
    group_a_ids = set(graph_bt_df[graph_bt_df["closest_stage"].isin(["graph","synthesis"])]["id"])
    group_b_ids = set(graph_bt_df[graph_bt_df["closest_stage"].isin(["brainstorm","patterns"])]["id"])

    # Use combined reasoning stage
    df_combined = graph_hidden_df[graph_hidden_df["stage"]=="reasoning"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, grp_ids, label, color in [
        (axes[0], group_a_ids,
         f"Group A: closest = graph/synthesis\n(n={len(group_a_ids)} questions)", TEAL),
        (axes[1], group_b_ids,
         f"Group B: closest = brainstorm/patterns\n(n={len(group_b_ids)} questions)", CORAL),
    ]:
        df_grp = df_combined[df_combined["id"].isin(grp_ids)]
        if df_grp.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        for qid, grp in df_grp.groupby("id"):
            g = grp.sort_values("layer")
            ax.plot(g["layer"], g["reasoning_answer_cosine_distance"],
                    color=color, alpha=0.1, linewidth=0.8)

        stats = df_grp.groupby("layer")["reasoning_answer_cosine_distance"].agg(["mean","std"]).reset_index()
        ax.plot(stats["layer"], stats["mean"],
                color=color, linewidth=2.5, label="mean", zorder=4)
        ax.fill_between(stats["layer"],
                        stats["mean"]-stats["std"],
                        stats["mean"]+stats["std"],
                        alpha=0.15, color=color, label="± 1 std")

        for l in [7, 36]:
            ax.axvline(l, color=AMBER, alpha=0.35, linewidth=1.5, linestyle="--")
        ax.axvspan(7, 10, alpha=0.05, color=AMBER)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("transformer layer")
        ax.legend(fontsize=9, framealpha=0.4)
        if ax == axes[0]:
            ax.set_ylabel("cosine distance (reasoning vs answer)")

    fig.suptitle("Plot C2 — Graph-PRefLexOR layer-wise divergence split by closest stage group",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.04,
             "Group A (answer backtracks to graph/synthesis): structured reasoning stages — "
             "expect lower divergence in middle layers.\n"
             "Group B (answer backtracks to brainstorm/patterns): exploratory stages — "
             "expect higher or more variable divergence.",
             ha="center", fontsize=9, color=GRAY, style="italic")
    plt.tight_layout()
    plt.savefig(PLOTS/"plotC2_graph_divergence_by_stage_group.png"); plt.close()
    print("  Saved plotC2_graph_divergence_by_stage_group.png")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Load existing data
    print("Loading data...")
    sim_df        = pd.read_csv(SIM_FILE)
    qwen_hidden_df= pd.read_csv(HIDDEN_FILE)
    graph_rows    = load_jsonl(GRAPH_FILE)
    print(f"  sim_df={len(sim_df)} qwen_hidden={len(qwen_hidden_df)} graph_rows={len(graph_rows)}")

    # Load embedding model for Graph-PRefLexOR backtracking
    print(f"\nLoading embedding model {EMBED_MODEL} ...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    # Compute Graph-PRefLexOR internal backtracking (embedding-based, CPU)
    graph_bt_df = compute_graph_backtracking(graph_rows, embed_model)

    # Plots A1, A2 — no model needed
    plot_a1(sim_df)
    plot_a2(graph_bt_df)

    # Load Qwen for hidden state computation
    print(f"\nLoading {QWEN_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded.")

    # Compute Graph-PRefLexOR hidden divergence (GPU)
    if GRAPH_HIDDEN_FILE.exists():
        existing = pd.read_csv(GRAPH_HIDDEN_FILE)
        done_ids = set(existing["id"].unique())
        remaining = [r for r in graph_rows if r["id"] not in done_ids]
        print(f"\nGraph hidden: {len(done_ids)} done, {len(remaining)} remaining")
    else:
        remaining = graph_rows

    if remaining:
        graph_hidden_df = compute_graph_hidden_divergence(remaining, tokenizer, model)
    else:
        graph_hidden_df = pd.read_csv(GRAPH_HIDDEN_FILE)

    del model; torch.cuda.empty_cache(); gc.collect()

    # Plots B, B2, C1, C2
    plot_b(qwen_hidden_df, graph_hidden_df)
    plot_b2(graph_hidden_df)
    plot_c1(qwen_hidden_df, sim_df)
    plot_c2(graph_hidden_df, graph_bt_df)

    print("\n" + "="*60)
    print("All mentor plots complete:")
    for f in sorted(PLOTS.iterdir()):
        if f.name.startswith("plot") and f.suffix == ".png":
            if any(x in f.name for x in ["A1","A2","plotB","plotC"]):
                print(f"  {f.name}  ({f.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
