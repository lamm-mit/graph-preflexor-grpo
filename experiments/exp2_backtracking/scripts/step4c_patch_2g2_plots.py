"""
exp2_patch_2g2_and_plots.py
============================
1. Reruns 2G.2 (logit lens) with dtype fix — cast hidden state to bfloat16
   before passing to lm_head.
2. Generates all plots (1-8) from saved CSV files.
   Skips any plot whose PNG already exists.

Run on a GPU node (needs Qwen for 2G.2).
"""

import json, re, gc, os
import numpy as np
import pandas as pd
from pathlib import Path

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

OUT_2A    = WORK / "exp2_qwen_thinking_outputs_100.jsonl"
OUT_2B    = WORK / "exp2_backtracking_text_similarity_100.csv"
OUT_2C    = WORK / "exp2_hidden_state_divergence_100.csv"
OUT_2E    = WORK / "exp2_question_level_divergence_summary_100.csv"
OUT_2F    = WORK / "exp2_summary_by_question_type_100.csv"
OUT_PROBE = WORK / "exp2_layer_probe_thinking_vs_answer_100.csv"
OUT_LOGIT = WORK / "exp2_logit_lens_token_categories_100.csv"

QWEN_MODEL  = "Qwen/Qwen3-8B"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

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

SIM_COLS   = ["sim_qwen_answer_to_qwen_thinking",
              "sim_qwen_answer_to_graph_brainstorm",
              "sim_qwen_answer_to_graph_graph",
              "sim_qwen_answer_to_graph_patterns",
              "sim_qwen_answer_to_graph_synthesis",
              "sim_qwen_answer_to_graph_answer"]
SIM_LABELS = ["Qwen thinking","Graph brainstorm","Graph graph",
              "Graph patterns","Graph synthesis","Graph answer"]
SIM_COLORS = [PURPLE, TEAL, CORAL, DPURPLE, AMBER, BLUE]
CANDIDATE_LAYERS = [7, 8, 9, 10, 36]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def find_think_split(tokenizer, generated_ids):
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

# ══════════════════════════════════════════════════════════════════════════════
# 2G.2 — Logit lens (fixed dtype)
# ══════════════════════════════════════════════════════════════════════════════
def run_2g2(merged_rows, tokenizer, model):
    print("\n" + "="*60)
    print("2G.2: Logit lens (dtype fixed)")
    print("="*60)

    TOKEN_CATEGORIES = {
        "reasoning":   ["because", "therefore", "maybe", "consider", "if", "however"],
        "answer":      ["thus", "summary", "answer", "suggests", "conclusion"],
        "causal":      ["causes", "leads", "mechanism", "due", "results"],
        "uncertainty": ["may", "could", "might", "possible", "perhaps"],
        "contrast":    ["however", "although", "whereas", "tradeoff", "but"],
    }
    cat_ids = {}
    for cat, words in TOKEN_CATEGORIES.items():
        ids = set()
        for w in words:
            for variant in [w, " "+w, w.capitalize(), " "+w.capitalize()]:
                tids = tokenizer.encode(variant, add_special_tokens=False)
                ids.update(tids)
        cat_ids[cat] = list(ids)

    INSPECT_LAYERS = [6, 7, 8, 9, 10, 35, 36]
    lm_head = model.lm_head
    records = []

    # Load closest_source map from 2B
    sim_df = pd.read_csv(OUT_2B)
    cs_map = dict(zip(sim_df["id"], sim_df["closest_source"]))

    for i, row in enumerate(merged_rows[:20]):
        if not row.get("has_think_split", False):
            continue
        print(f"  [{i+1}] id={row['id']}")

        question = row["question"]
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

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5000,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        split_pos     = find_think_split(tokenizer, generated_ids)

        if split_pos is None or split_pos >= len(generated_ids) - 1:
            del outputs; torch.cuda.empty_cache(); gc.collect()
            continue

        n_steps      = len(outputs.hidden_states)
        t_steps      = list(range(0, min(split_pos, n_steps)))
        a_steps      = list(range(split_pos, n_steps))
        closest      = cs_map.get(row["id"], "")

        if not t_steps or not a_steps:
            del outputs; torch.cuda.empty_cache(); gc.collect()
            continue

        for layer_idx in INSPECT_LAYERS:
            if layer_idx >= len(outputs.hidden_states[0]):
                continue

            for span_name, steps in [("thinking", t_steps), ("answer", a_steps)]:
                vecs = [outputs.hidden_states[s][layer_idx][0, 0, :].float()
                        for s in steps[:50]]
                mean_h = torch.stack(vecs).mean(0)

                # ── DTYPE FIX: cast to model dtype before lm_head ──────────
                mean_h_bf16 = mean_h.to(dtype=model.dtype, device=model.device)

                with torch.no_grad():
                    logits = lm_head(mean_h_bf16.unsqueeze(0))
                    probs  = torch.softmax(logits[0].float(), dim=-1).cpu().numpy()

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

        del outputs; torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(records)
    df.to_csv(OUT_LOGIT, index=False)
    print(f"  Saved {len(df)} rows → {OUT_LOGIT}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# All plots
# ══════════════════════════════════════════════════════════════════════════════
def make_all_plots(sim_df, hidden_df, summary_df, qtype_df, probe_df, logit_df):
    print("\n" + "="*60)
    print("Generating all plots")
    print("="*60)

    # ── Plot 1: Backtracking distribution ─────────────────────────────────────
    p = PLOTS / "plot1_backtrack_distribution.png"
    if not p.exists():
        fig, ax = plt.subplots(figsize=(7, 4))
        vc = sim_df["closest_source"].value_counts()
        colors_map = {"qwen_thinking":PURPLE,"graph_answer":BLUE,
                      "graph_brainstorm":TEAL,"graph_synthesis":AMBER,
                      "graph_graph":CORAL,"graph_patterns":DPURPLE}
        colors = [colors_map.get(k, GRAY) for k in vc.index]
        bars = ax.barh(vc.index, vc.values, color=colors, height=0.55, zorder=3)
        for bar, v in zip(bars, vc.values):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    str(v), va="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("number of questions (n=100)")
        ax.set_title("Plot 1 — Where does Qwen's answer backtrack? (n=100)")
        ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", visible=False)
        plt.tight_layout(); plt.savefig(p); plt.close()
        print(f"  Saved {p.name}")

    # ── Plot 2: Mean ± std similarity ─────────────────────────────────────────
    p = PLOTS / "plot2_mean_similarities.png"
    if not p.exists():
        means = [sim_df[c].mean() for c in SIM_COLS]
        stds  = [sim_df[c].std()  for c in SIM_COLS]
        order = sorted(range(len(means)), key=lambda i: -means[i])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh([SIM_LABELS[i] for i in order],
                [means[i] for i in order],
                xerr=[stds[i] for i in order],
                color=[SIM_COLORS[i] for i in order],
                height=0.55, capsize=4, zorder=3)
        ax.set_xlabel("mean cosine similarity ± std")
        ax.set_title("Plot 2 — Mean similarity: Qwen answer vs sources (n=100)")
        ax.set_xlim(0.65, 1.02)
        ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", visible=False)
        plt.tight_layout(); plt.savefig(p); plt.close()
        print(f"  Saved {p.name}")

    # ── Plot 3: Per-question heatmap ───────────────────────────────────────────
    p = PLOTS / "plot3_similarity_heatmap.png"
    if not p.exists():
        df_s = sim_df.sort_values(["question_type","id"]).reset_index(drop=True)
        matrix = df_s[SIM_COLS].values
        cmap = LinearSegmentedColormap.from_list("teal",["#E1F5EE","#1D9E75","#04342C"])
        fig, ax = plt.subplots(figsize=(10, max(8, len(df_s)*0.18)))
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.65, vmax=0.97)
        ax.set_xticks(range(len(SIM_LABELS)))
        ax.set_xticklabels([l.replace(" ","\n") for l in SIM_LABELS], fontsize=9)
        ax.set_yticks(range(len(df_s)))
        ax.set_yticklabels([f"{r.id} {r.question_type[:18]}"
                            for _, r in df_s.iterrows()], fontsize=7)
        for i, row in df_s.iterrows():
            try:
                j = SIM_COLS.index(f"sim_qwen_answer_to_{row['closest_source']}")
                ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,
                             fill=False, edgecolor="white", linewidth=2))
            except ValueError:
                pass
        plt.colorbar(im, ax=ax, shrink=0.4, label="cosine similarity")
        ax.set_title("Plot 3 — Per-question heatmap (white box = closest source)")
        plt.tight_layout(); plt.savefig(p, dpi=120); plt.close()
        print(f"  Saved {p.name}")

    # ── Plot 4: Question-type gaps ─────────────────────────────────────────────
    p = PLOTS / "plot4_qtype_gaps.png"
    if not p.exists() and not qtype_df.empty:
        metrics = [("avg_thinking_gap","thinking gap",BLUE),
                   ("avg_graph_gap","graph gap",CORAL),
                   ("avg_synthesis_gap","synthesis gap",TEAL),
                   ("avg_graph_answer_gap","graph answer gap",PURPLE)]
        qt_labels = [s.replace("_"," ") for s in qtype_df["question_type"]]
        x = np.arange(len(qtype_df)); w = 0.18
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, (col, label, color) in enumerate(metrics):
            if col not in qtype_df.columns: continue
            ax.bar(x+(i-1.5)*w, qtype_df[col], width=w,
                   label=label, color=color, zorder=3, alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(qt_labels, fontsize=9)
        ax.set_ylabel("gap (1 − similarity)"); ax.legend(fontsize=10)
        ax.set_title("Plot 4 — Question-type backtracking gaps (n=100)")
        ax.grid(axis="y", alpha=0.25); ax.grid(axis="x", visible=False)
        plt.tight_layout(); plt.savefig(p); plt.close()
        print(f"  Saved {p.name}")

    # ── Plot 5: Layer divergence mean ± std ────────────────────────────────────
    p = PLOTS / "plot5_layer_divergence.png"
    if not p.exists() and not hidden_df.empty:
        ls = hidden_df.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for qid, grp in hidden_df.groupby("id"):
            g = grp.sort_values("layer")
            ax.plot(g["layer"], g["thinking_answer_cosine_distance"],
                    color=BLUE, alpha=0.06, linewidth=0.8)
        ax.plot(ls["layer"], ls["mean"], color=BLUE, linewidth=2.5,
                label="mean", zorder=4)
        ax.fill_between(ls["layer"], ls["mean"]-ls["std"], ls["mean"]+ls["std"],
                        alpha=0.15, color=BLUE, label="± 1 std")
        peak = ls.loc[ls["mean"].idxmax()]
        ax.annotate(f"layer {int(peak['layer'])} (peak)",
                    xy=(peak["layer"], peak["mean"]),
                    xytext=(peak["layer"]-5, peak["mean"]+0.02),
                    arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5),
                    fontsize=9, color=CORAL, fontweight="bold")
        ax.set_xlabel("transformer layer")
        ax.set_ylabel("cosine distance (thinking vs answer)")
        ax.set_title("Plot 5 — Layer-wise thinking–answer divergence (n=100, mean ± std)")
        ax.legend(fontsize=10, framealpha=0.4)
        plt.tight_layout(); plt.savefig(p); plt.close()
        print(f"  Saved {p.name}")

    # ── Plot 6: Layer-36 distribution ─────────────────────────────────────────
    p = PLOTS / "plot6_layer36_distribution.png"
    if not p.exists() and not summary_df.empty and "layer36_value" in summary_df.columns:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(summary_df["layer36_value"], bins=20,
                color=BLUE, edgecolor="white", zorder=3)
        ax.axvline(summary_df["layer36_value"].mean(), color=CORAL,
                   linestyle="--",
                   label=f"mean={summary_df['layer36_value'].mean():.3f}")
        ax.set_xlabel("layer 36 cosine distance")
        ax.set_ylabel("count"); ax.legend()
        ax.set_title("Plot 6 — Distribution of layer-36 divergence (n=100)")
        plt.tight_layout(); plt.savefig(p); plt.close()
        print(f"  Saved {p.name}")

    # ── Plot 7: Probe AUROC ────────────────────────────────────────────────────
    p = PLOTS / "plot7_probe_auroc.png"
    if not p.exists() and not probe_df.empty and not hidden_df.empty:
        layer_mean = hidden_df.groupby("layer")["thinking_answer_cosine_distance"].mean()
        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        ax2 = ax1.twinx()
        ax1.plot(probe_df["layer"], probe_df["auroc"],
                 color=PURPLE, linewidth=2.5, label="probe AUROC")
        ax1.fill_between(probe_df["layer"], 0.5, probe_df["auroc"],
                         alpha=0.12, color=PURPLE)
        ax2.plot(layer_mean.index, layer_mean.values,
                 color=BLUE, linewidth=1.5, linestyle="--",
                 alpha=0.7, label="mean divergence")
        ax1.set_xlabel("transformer layer")
        ax1.set_ylabel("probe AUROC", color=PURPLE)
        ax2.set_ylabel("mean cosine distance", color=BLUE)
        ax1.set_ylim(0.45, 1.05)
        ax1.set_title("Plot 7 — Probe AUROC vs divergence by layer (2G.1)")
        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1+l2, lab1+lab2, fontsize=10, framealpha=0.4)
        plt.tight_layout(); plt.savefig(p); plt.close()
        print(f"  Saved {p.name}")

    # ── Plot 8: Logit lens heatmap ─────────────────────────────────────────────
    p = PLOTS / "plot8_logit_lens_heatmap.png"
    if not p.exists() and not logit_df.empty:
        pivot = (logit_df.groupby(["layer","category","span_type"])
                 ["probability_mass"].mean().unstack("span_type"))
        if "thinking" in pivot.columns and "answer" in pivot.columns:
            pivot["diff"] = pivot["answer"] - pivot["thinking"]
            diff_mat = pivot["diff"].unstack("category").fillna(0)
            cmap2 = LinearSegmentedColormap.from_list(
                "div", [CORAL, "white", BLUE])
            vmax = max(abs(diff_mat.values.min()), abs(diff_mat.values.max()))
            fig, ax = plt.subplots(figsize=(9, 4))
            im = ax.imshow(diff_mat.values, aspect="auto",
                           cmap=cmap2, vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(diff_mat.columns)))
            ax.set_xticklabels(diff_mat.columns, fontsize=10)
            ax.set_yticks(range(len(diff_mat.index)))
            ax.set_yticklabels([f"layer {l}" for l in diff_mat.index], fontsize=10)
            plt.colorbar(im, ax=ax, label="answer − thinking probability mass")
            ax.set_title("Plot 8 — Logit lens: answer vs thinking token category shift")
            plt.tight_layout(); plt.savefig(p); plt.close()
            print(f"  Saved {p.name}")

    print(f"\nAll plots in {PLOTS}:")
    for f in sorted(PLOTS.iterdir()):
        if f.suffix == ".png":
            print(f"  {f.name}  ({f.stat().st_size//1024} KB)")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading saved CSVs...")
    sim_df    = pd.read_csv(OUT_2B)
    hidden_df = pd.read_csv(OUT_2C)
    summary_df= pd.read_csv(OUT_2E) if OUT_2E.exists() else pd.DataFrame()
    qtype_df  = pd.read_csv(OUT_2F) if OUT_2F.exists() else pd.DataFrame()
    probe_df  = pd.read_csv(OUT_PROBE) if OUT_PROBE.exists() else pd.DataFrame()
    print(f"  sim_df={len(sim_df)} hidden_df={len(hidden_df)} "
          f"summary_df={len(summary_df)} probe_df={len(probe_df)}")

    # ── 2G.2 ──────────────────────────────────────────────────────────────────
    if not OUT_LOGIT.exists():
        print(f"\nLoading {QWEN_MODEL} for 2G.2 ...")
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        model.eval()
        merged_rows = load_jsonl(OUT_2A)
        logit_df = run_2g2(merged_rows, tokenizer, model)
        del model; torch.cuda.empty_cache(); gc.collect()
    else:
        print(f"2G.2 already done, loading {OUT_LOGIT}")
        logit_df = pd.read_csv(OUT_LOGIT)

    # ── All plots ──────────────────────────────────────────────────────────────
    make_all_plots(sim_df, hidden_df, summary_df, qtype_df,
                   probe_df, logit_df)

    print("\nDone.")

if __name__ == "__main__":
    main()
