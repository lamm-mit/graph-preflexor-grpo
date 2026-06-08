"""
exp2_fix_plots_7_8.py
=====================
1. Fix Plot 7: divergence curve is invisible because right-axis scale is inverted.
   Fix by flipping the right y-axis and adding proper annotation.

2. Fix Plot 8: add top-token analysis showing which specific tokens shift most
   between answer and thinking states at layers 7-10 and 36.
   This produces:
     - plot8_logit_lens_heatmap.png  (fixed with better colorscale annotation)
     - plot8b_top_tokens_layer7_10.png  (top tokens more likely in answer vs thinking)
     - plot8b_top_tokens_layer36.png    (same for layer 36)
"""

import json, re, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

WORK  = Path("/projects/bfir/ssourav")
PLOTS = WORK / "plots_100"

OUT_2A    = WORK / "exp2_qwen_thinking_outputs_100.jsonl"
OUT_2C    = WORK / "exp2_hidden_state_divergence_100.csv"
OUT_PROBE = WORK / "exp2_layer_probe_thinking_vs_answer_100.csv"
OUT_LOGIT = WORK / "exp2_logit_lens_token_categories_100.csv"

QWEN_MODEL = "Qwen/Qwen3-8B"

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,
    "axes.titlesize":12,"axes.titleweight":"bold",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,"grid.linestyle":"--",
    "figure.dpi":150,"savefig.dpi":150,
    "savefig.bbox":"tight","savefig.facecolor":"white",
})
BLUE="#378ADD"; CORAL="#D85A30"; PURPLE="#7F77DD"; AMBER="#BA7517"; GRAY="#888780"

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

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
# Fix Plot 7
# ══════════════════════════════════════════════════════════════════════════════
def fix_plot7():
    print("Fixing Plot 7...")
    hidden_df = pd.read_csv(OUT_2C)
    probe_df  = pd.read_csv(OUT_PROBE)

    layer_stats = hidden_df.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    # Remove spines from ax2 manually
    ax2.spines["top"].set_visible(False)

    # Left axis: probe AUROC
    ax1.plot(probe_df["layer"], probe_df["auroc"],
             color=PURPLE, linewidth=2.5, label="probe AUROC", zorder=4)
    ax1.fill_between(probe_df["layer"], 0.5, probe_df["auroc"],
                     alpha=0.15, color=PURPLE)
    ax1.set_xlabel("transformer layer")
    ax1.set_ylabel("probe AUROC", color=PURPLE, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=PURPLE)
    ax1.set_ylim(0.45, 1.08)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    # Right axis: mean divergence — NOT inverted, proper scale
    ax2.plot(layer_stats["layer"], layer_stats["mean"],
             color=BLUE, linewidth=2.0, linestyle="--",
             alpha=0.85, label="mean cosine distance", zorder=3)
    ax2.fill_between(layer_stats["layer"],
                     layer_stats["mean"] - layer_stats["std"],
                     layer_stats["mean"] + layer_stats["std"],
                     alpha=0.10, color=BLUE)
    ax2.set_ylabel("mean cosine distance (thinking vs answer)", color=BLUE, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.set_ylim(0, layer_stats["mean"].max() * 1.5)

    # Annotate key layers
    for l, label, color in [(7, "layers\n7–10", AMBER), (36, "layer 36", CORAL)]:
        ax1.axvline(l, color=color, alpha=0.4, linewidth=1.5, linestyle=":")
    ax1.axvspan(7, 10, alpha=0.06, color=AMBER, label="layers 7–10 zone")

    # Key observation annotations
    ax1.annotate("probe saturates\nat layer 5\n(AUROC=1.0)",
                 xy=(5, 1.0), xytext=(8, 0.88),
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2),
                 fontsize=9, color=PURPLE)
    ax1.annotate("divergence\npeak (layers 7–8)",
                 xy=(7, layer_stats.loc[layer_stats["layer"]==7,"mean"].values[0]),
                 xytext=(12, layer_stats.loc[layer_stats["layer"]==7,"mean"].values[0]+0.01),
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
                 fontsize=9, color=BLUE, transform=ax1.transData)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2,
               fontsize=9, framealpha=0.4, loc="lower right")

    ax1.set_title("Plot 7 — Probe AUROC vs thinking–answer divergence by layer\n"
                  "Probe saturates at layer 5; divergence peaks at layers 7–8 and spikes at layer 36",
                  fontsize=11)

    # Caption
    fig.text(0.5, -0.04,
             "Probe AUROC = 1.0 from layer 5 onward: thinking vs answer tokens are linearly separable\n"
             "across almost all layers. Divergence peaks at layers 7–8 (early reasoning separation)\n"
             "and spikes at layer 36 (final projection). The two metrics are complementary:\n"
             "AUROC measures decodability, divergence measures geometric distance.",
             ha="center", fontsize=9, color=GRAY, style="italic")

    plt.tight_layout()
    out = PLOTS / "plot7_probe_auroc.png"
    plt.savefig(out); plt.close()
    print(f"  Saved {out.name}")

# ══════════════════════════════════════════════════════════════════════════════
# Fix Plot 8 + add top-token analysis (plot8b)
# ══════════════════════════════════════════════════════════════════════════════
def fix_plot8_and_add_topk(tokenizer, model):
    print("\nFixing Plot 8 + generating top-token analysis...")

    # ── Reload existing logit lens CSV ────────────────────────────────────────
    logit_df = pd.read_csv(OUT_LOGIT)

    # Fix Plot 8: same heatmap but with better annotations and note on magnitude
    pivot = (logit_df.groupby(["layer","category","span_type"])
             ["probability_mass"].mean().unstack("span_type"))
    if "thinking" in pivot.columns and "answer" in pivot.columns:
        pivot["diff"] = pivot["answer"] - pivot["thinking"]
        diff_mat = pivot["diff"].unstack("category").fillna(0)

        cmap = LinearSegmentedColormap.from_list("div", [CORAL, "white", BLUE])
        vmax = max(abs(diff_mat.values.min()), abs(diff_mat.values.max()))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(diff_mat.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(diff_mat.columns)))
        ax.set_xticklabels(diff_mat.columns, fontsize=11)
        ax.set_yticks(range(len(diff_mat.index)))
        ax.set_yticklabels([f"layer {l}" for l in diff_mat.index], fontsize=11)

        # Annotate each cell with value
        for i in range(diff_mat.values.shape[0]):
            for j in range(diff_mat.values.shape[1]):
                v = diff_mat.values[i, j]
                ax.text(j, i, f"{v:.1e}", ha="center", va="center",
                        fontsize=8, color="black")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("answer − thinking probability mass\n(blue = answer higher, red = thinking higher)",
                       fontsize=9)

        # Box layers 7-10 as zone
        ax.add_patch(plt.Rectangle((-0.5, 0.5), len(diff_mat.columns), 4,
                     fill=False, edgecolor=AMBER, linewidth=2.5, linestyle="--"))
        ax.text(len(diff_mat.columns)-0.1, 2.5, "layers 7–10\nzone",
                fontsize=8, color=AMBER, ha="right", va="center")

        ax.set_title("Plot 8 — Logit lens: answer vs thinking token category shift\n"
                     "(magnitude ~1e-4: preliminary diagnostic only — see plot 8b for top-token analysis)",
                     fontsize=10)

        fig.text(0.5, -0.04,
                 "Layers 7–10: answer states carry slightly more probability mass for answer/causal/reasoning categories.\n"
                 "Layer 36: answer/causal categories increase but contrast/reasoning categories decrease,\n"
                 "suggesting the final layer sharpens toward answer-style tokens while suppressing exploratory reasoning markers.\n"
                 "Values are ~1e-4; treat as a directional signal, not a strong effect.",
                 ha="center", fontsize=9, color=GRAY, style="italic")

        plt.tight_layout()
        plt.savefig(PLOTS / "plot8_logit_lens_heatmap.png"); plt.close()
        print("  Saved plot8_logit_lens_heatmap.png")

    # ── Top-token analysis ────────────────────────────────────────────────────
    print("  Running top-token analysis (this needs the model)...")

    merged_rows = load_jsonl(OUT_2A)
    # Use first 15 rows with valid split
    rows = [r for r in merged_rows if r.get("has_think_split", False)][:15]

    INSPECT_LAYERS = [7, 8, 9, 10, 36]
    TOP_K = 20  # top tokens per layer per direction

    # layer -> list of (token, prob_diff) where prob_diff = answer_prob - thinking_prob
    layer_token_diffs = defaultdict(list)

    for i, row in enumerate(rows):
        print(f"  Top-token [{i+1}/{len(rows)}] id={row['id']}")
        q = row["question"]
        prompt = (
            "You are answering an open-ended scientific reasoning question.\n\n"
            "/think\n\n"
            f"Question:\n{q}\n\n"
            "Think carefully about scientific mechanisms and explanations. "
            "Then give the final answer."
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

        gen_ids   = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        split_pos = find_think_split(tokenizer, gen_ids)
        if not split_pos or split_pos >= len(outputs.hidden_states) - 1:
            del outputs; torch.cuda.empty_cache(); gc.collect()
            continue

        n_steps = len(outputs.hidden_states)
        t_steps = list(range(0, min(split_pos, n_steps)))
        a_steps = list(range(split_pos, n_steps))
        if not t_steps or not a_steps:
            del outputs; torch.cuda.empty_cache(); gc.collect()
            continue

        for layer_idx in INSPECT_LAYERS:
            if layer_idx >= len(outputs.hidden_states[0]):
                continue

            # Pool hidden states per span
            t_vecs = [outputs.hidden_states[s][layer_idx][0,0,:].float()
                      for s in t_steps[:80]]
            a_vecs = [outputs.hidden_states[s][layer_idx][0,0,:].float()
                      for s in a_steps[:80]]
            t_mean = torch.stack(t_vecs).mean(0).to(dtype=model.dtype, device=model.device)
            a_mean = torch.stack(a_vecs).mean(0).to(dtype=model.dtype, device=model.device)

            with torch.no_grad():
                t_logits = model.lm_head(t_mean.unsqueeze(0))[0].float()
                a_logits = model.lm_head(a_mean.unsqueeze(0))[0].float()
                t_probs  = torch.softmax(t_logits, dim=-1).cpu().numpy()
                a_probs  = torch.softmax(a_logits, dim=-1).cpu().numpy()

            diff = a_probs - t_probs  # positive = more likely in answer
            layer_token_diffs[layer_idx].append(diff)

        del outputs; torch.cuda.empty_cache(); gc.collect()

    # Average diffs across questions
    print("  Generating top-token plots...")
    for layer_group_name, layer_list in [("layers_7_to_10", [7,8,9,10]), ("layer_36", [36])]:
        # Average across layers in group and questions
        all_diffs = []
        for li in layer_list:
            if li in layer_token_diffs:
                all_diffs.extend(layer_token_diffs[li])
        if not all_diffs:
            continue

        mean_diff = np.mean(all_diffs, axis=0)  # (vocab_size,)

        # Top K tokens more likely in answer (positive diff)
        top_answer_idx = np.argsort(-mean_diff)[:TOP_K]
        top_answer_toks = [(tokenizer.decode([idx]).strip(), float(mean_diff[idx]))
                           for idx in top_answer_idx
                           if tokenizer.decode([idx]).strip()]

        # Top K tokens more likely in thinking (negative diff)
        top_think_idx = np.argsort(mean_diff)[:TOP_K]
        top_think_toks = [(tokenizer.decode([idx]).strip(), float(-mean_diff[idx]))
                          for idx in top_think_idx
                          if tokenizer.decode([idx]).strip()]

        # Filter to printable tokens
        top_answer_toks = [(t, v) for t, v in top_answer_toks
                           if t.isprintable() and len(t) > 0 and len(t) < 20][:15]
        top_think_toks  = [(t, v) for t, v in top_think_toks
                           if t.isprintable() and len(t) > 0 and len(t) < 20][:15]

        # Plot
        fig = plt.figure(figsize=(14, 5))
        gs  = GridSpec(1, 2, figure=fig, wspace=0.4)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Answer-biased tokens
        if top_answer_toks:
            toks, vals = zip(*top_answer_toks)
            y = range(len(toks))
            ax1.barh(y, vals, color=BLUE, height=0.6, zorder=3)
            ax1.set_yticks(y); ax1.set_yticklabels(toks, fontsize=10)
            ax1.set_xlabel("mean prob. increase (answer − thinking)")
            ax1.set_title("More likely in ANSWER states", color=BLUE, fontsize=10)
            ax1.invert_yaxis()
            ax1.grid(axis="x", alpha=0.25); ax1.grid(axis="y", visible=False)

        # Thinking-biased tokens
        if top_think_toks:
            toks, vals = zip(*top_think_toks)
            y = range(len(toks))
            ax2.barh(y, vals, color=CORAL, height=0.6, zorder=3)
            ax2.set_yticks(y); ax2.set_yticklabels(toks, fontsize=10)
            ax2.set_xlabel("mean prob. increase (thinking − answer)")
            ax2.set_title("More likely in THINKING states", color=CORAL, fontsize=10)
            ax2.invert_yaxis()
            ax2.grid(axis="x", alpha=0.25); ax2.grid(axis="y", visible=False)

        layer_label = "Layers 7–10" if "7" in layer_group_name else "Layer 36"
        fig.suptitle(f"Plot 8b — Top token shifts at {layer_label}\n"
                     f"(logit lens: answer vs thinking hidden states)",
                     fontsize=12, fontweight="bold")

        fig.text(0.5, -0.03,
                 f"Blue = tokens the model is more likely to predict from answer-span hidden states at {layer_label}.\n"
                 "Red = tokens more likely from thinking-span states. Averaged over 15 questions.",
                 ha="center", fontsize=9, color=GRAY, style="italic")

        plt.tight_layout()
        out = PLOTS / f"plot8b_top_tokens_{layer_group_name}.png"
        plt.savefig(out); plt.close()
        print(f"  Saved {out.name}")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Fix Plot 7 (no model needed)
    fix_plot7()

    # Fix Plot 8 + top-token analysis (needs model)
    print(f"\nLoading {QWEN_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded.")

    fix_plot8_and_add_topk(tokenizer, model)

    del model; torch.cuda.empty_cache(); gc.collect()

    print("\nAll done. Updated plots:")
    for f in sorted(PLOTS.iterdir()):
        if "plot7" in f.name or "plot8" in f.name:
            print(f"  {f.name}  ({f.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
