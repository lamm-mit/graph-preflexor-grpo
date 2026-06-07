"""
exp2_publication_plots_v2.py
============================
All Experiment 2 plots in the user's exact figure style:
  - figsize=(8,6), dpi=300
  - white background (ax + fig)
  - fontsize=20 everywhere
  - black edgecolor on bars, linewidth=1.2
  - ax.grid(axis="y", alpha=0.3)
  - bold titles
  - value labels above bars
  - saved as PDF + PNG to plots_pub/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK  = Path("/projects/bfir/ssourav")
PLOTS = WORK / "plots_pub"
PLOTS.mkdir(exist_ok=True)

SIM_FILE     = WORK / "exp2_backtracking_text_similarity_100.csv"
HIDDEN_FILE  = WORK / "exp2_hidden_state_divergence_100.csv"
GRAPH_HIDDEN = WORK / "exp2_graph_hidden_state_divergence_100.csv"
GRAPH_BT     = WORK / "exp2_graph_internal_backtracking_100.csv"
SUMMARY_FILE = WORK / "exp2_summary_by_question_type_100.csv"
PROBE_FILE   = WORK / "exp2_layer_probe_thinking_vs_answer_100.csv"
LOGIT_FILE   = WORK / "exp2_logit_lens_token_categories_100.csv"
QDIV_FILE    = WORK / "exp2_question_level_divergence_summary_100.csv"

# ── User's exact palette ──────────────────────────────────────────────────────
C_RED    = "#E63946"   # Graph-PRefLexOR primary
C_BLUE   = "#457B9D"   # Qwen primary
C_GREEN  = "#2A9D8F"   # third category
C_ORANGE = "#E9C46A"   # fourth
C_PURPLE = "#6A4C93"   # fifth
C_BROWN  = "#A8763E"   # sixth
C_GRAY   = "#8C8C8C"

SOURCE_COLORS = {
    "qwen_thinking":    C_BLUE,
    "graph_answer":     C_RED,
    "graph_brainstorm": C_GREEN,
    "graph_graph":      C_PURPLE,
    "graph_synthesis":  C_ORANGE,
    "graph_patterns":   C_BROWN,
}
STAGE_COLORS = {
    "brainstorm": C_GREEN,
    "graph":      C_PURPLE,
    "patterns":   C_BROWN,
    "synthesis":  C_ORANGE,
    "reasoning":  C_BLUE,
}

FS = 20   # global fontsize matching user style

def base_fig(w=8, h=6):
    """Create fig/ax with user's exact base settings."""
    fig, ax = plt.subplots(figsize=(w, h), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax

def base_fig2(ncols=2, w=16, h=6):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), dpi=300)
    for ax in axes:
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, axes

def style_ax(ax):
    """Apply user's axis style."""
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FS)
    ax.xaxis.label.set_size(FS)
    ax.yaxis.label.set_size(FS)
    ax.title.set_size(FS)
    ax.title.set_fontweight("bold")

def bar_label(ax, bars, fmt="{:.0f}", fontsize=FS):
    ymax = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h): continue
        ax.text(bar.get_x() + bar.get_width()/2,
                h + ymax * 0.015,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize-2, fontweight="bold")

def save(fig, name):
    plt.rcParams.update({"font.size": FS})
    fig.tight_layout()
    fig.savefig(PLOTS / f"{name}.pdf", format="pdf", bbox_inches="tight",
                facecolor="white")
    fig.savefig(PLOTS / f"{name}.png", format="png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}.pdf + .png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Backtracking distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot1(sim_df):
    vc = sim_df["closest_source"].value_counts()
    nice = {"graph_answer":"Graph\nAnswer","qwen_thinking":"Qwen\nThinking",
            "graph_brainstorm":"Graph\nBrainstorm","graph_graph":"Graph\nGraph",
            "graph_synthesis":"Graph\nSynthesis","graph_patterns":"Graph\nPatterns"}
    labels = [nice.get(k, k) for k in vc.index]
    colors = [SOURCE_COLORS.get(k, C_GRAY) for k in vc.index]

    fig, ax = base_fig(10, 6)
    bars = ax.bar(labels, vc.values, color=colors, width=0.6, zorder=3,
                  edgecolor="black", linewidth=1.2)
    ax.set_ylim(0, vc.max() * 1.25)
    bar_label(ax, bars)
    ax.set_ylabel("Number of Questions", fontsize=FS)
    ax.set_title("Qwen Answer Backtracking: Closest Reasoning Source  (n = 100)")
    style_ax(ax)
    save(fig, "plot1_backtrack_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Mean similarities
# ══════════════════════════════════════════════════════════════════════════════
def plot2(sim_df):
    entries = [
        ("sim_qwen_answer_to_graph_answer",    "Graph\nAnswer",    C_RED),
        ("sim_qwen_answer_to_qwen_thinking",   "Qwen\nThinking",   C_BLUE),
        ("sim_qwen_answer_to_graph_brainstorm","Graph\nBrainstorm",C_GREEN),
        ("sim_qwen_answer_to_graph_synthesis", "Graph\nSynthesis", C_ORANGE),
        ("sim_qwen_answer_to_graph_graph",     "Graph\nGraph",     C_PURPLE),
        ("sim_qwen_answer_to_graph_patterns",  "Graph\nPatterns",  C_BROWN),
    ]
    data = sorted([(l, sim_df[c].mean(), sim_df[c].std(), col)
                   for c,l,col in entries], key=lambda x: -x[1])
    labels, means, stds, colors = zip(*data)

    fig, ax = base_fig(10, 6)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.6, zorder=3,
                  edgecolor="black", linewidth=1.2, capsize=6,
                  error_kw=dict(elinewidth=1.8, capthick=1.8, ecolor="black"))
    ax.set_ylim(0.6, 1.06)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS)
    ax.set_ylabel("Mean Cosine Similarity (± std)", fontsize=FS)
    ax.set_title("Mean Similarity: Qwen Final Answer vs Reasoning Sources  (n = 100)")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2,
                m + 0.006, f"{m:.3f}",
                ha="center", va="bottom", fontsize=FS-4, fontweight="bold")
    style_ax(ax)
    save(fig, "plot2_mean_similarities")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Per-question heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot3(sim_df):
    SIM_COLS = ["sim_qwen_answer_to_qwen_thinking",
                "sim_qwen_answer_to_graph_brainstorm",
                "sim_qwen_answer_to_graph_graph",
                "sim_qwen_answer_to_graph_patterns",
                "sim_qwen_answer_to_graph_synthesis",
                "sim_qwen_answer_to_graph_answer"]
    COL_LABELS = ["Qwen\nThinking","Graph\nBrainstorm","Graph\nGraph",
                  "Graph\nPatterns","Graph\nSynthesis","Graph\nAnswer"]
    df_s   = sim_df.sort_values(["question_type","id"]).reset_index(drop=True)
    matrix = df_s[SIM_COLS].values
    cmap   = LinearSegmentedColormap.from_list("teal",["#f0f9f6","#1a7a5e","#0a3d30"])

    h = max(8, len(df_s)*0.18)
    fig, ax = plt.subplots(figsize=(10, h), dpi=300)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.65, vmax=0.98)
    ax.set_xticks(range(len(COL_LABELS)))
    ax.set_xticklabels(COL_LABELS, fontsize=12)
    ax.set_yticks(range(len(df_s)))
    ax.set_yticklabels([f"id {r.id} · {r.question_type[:22]}"
                        for _, r in df_s.iterrows()], fontsize=7)
    ax.tick_params(axis="both", length=0)
    for i, row in df_s.iterrows():
        try:
            j = SIM_COLS.index(f"sim_qwen_answer_to_{row['closest_source']}")
            ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,
                         fill=False, edgecolor="white", linewidth=2.5))
        except ValueError:
            pass
    cbar = plt.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=12)
    ax.set_title("Per-Question Similarity Heatmap\n(white box = closest source)",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS/"plot3_similarity_heatmap.pdf", format="pdf",
                bbox_inches="tight", facecolor="white")
    fig.savefig(PLOTS/"plot3_similarity_heatmap.png", format="png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved plot3_similarity_heatmap.pdf + .png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Question-type gaps
# ══════════════════════════════════════════════════════════════════════════════
def plot4(qtype_df):
    metrics = [("avg_thinking_gap","Thinking Gap",C_BLUE),
               ("avg_graph_gap","Graph Gap",C_PURPLE),
               ("avg_synthesis_gap","Synthesis Gap",C_ORANGE),
               ("avg_graph_answer_gap","Graph Answer Gap",C_RED)]
    metrics = [(c,l,col) for c,l,col in metrics if c in qtype_df.columns]
    qt_labels = [s.replace("_"," ").title() for s in qtype_df["question_type"]]
    x = np.arange(len(qtype_df)); w = 0.18; n = len(metrics)

    fig, ax = base_fig(14, 6)
    for i,(col,label,color) in enumerate(metrics):
        offset = (i-(n-1)/2)*w
        ax.bar(x+offset, qtype_df[col], width=w, label=label,
               color=color, zorder=3, edgecolor="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(qt_labels, fontsize=FS-4, rotation=12, ha="right")
    ax.set_ylabel("Gap  (1 − Similarity)", fontsize=FS)
    ax.set_title("Backtracking Gaps by Question Type  (n = 100)")
    ax.legend(fontsize=FS-4, framealpha=0.9)
    style_ax(ax)
    save(fig, "plot4_qtype_gaps")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Layer-wise divergence
# ══════════════════════════════════════════════════════════════════════════════
def plot5(hidden_df):
    ls = hidden_df.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()
    fig, ax = base_fig(10, 6)
    for qid, grp in hidden_df.groupby("id"):
        g = grp.sort_values("layer")
        ax.plot(g["layer"], g["thinking_answer_cosine_distance"],
                color=C_BLUE, alpha=0.05, linewidth=0.7)
    ax.plot(ls["layer"], ls["mean"], color=C_BLUE, linewidth=2.5,
            label="Mean", zorder=4)
    ax.fill_between(ls["layer"], ls["mean"]-ls["std"], ls["mean"]+ls["std"],
                    alpha=0.18, color=C_BLUE, label="± 1 Std Dev")
    ax.axvspan(7, 10, alpha=0.08, color=C_ORANGE)
    ax.axvline(36, color=C_RED, alpha=0.45, linewidth=2, linestyle="--")
    peak = ls.loc[ls["mean"].idxmax()]
    ax.annotate(f"Peak: Layer {int(peak['layer'])}",
                xy=(peak["layer"], peak["mean"]),
                xytext=(peak["layer"]+3, peak["mean"]+0.02),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.8),
                fontsize=FS-4, color=C_RED, fontweight="bold")
    ax.text(8.5, ls["mean"].max()*0.55, "Layers\n7–10",
            fontsize=FS-6, color=C_ORANGE, ha="center", fontweight="bold")
    ax.set_xlabel("Transformer Layer", fontsize=FS)
    ax.set_ylabel("Cosine Distance", fontsize=FS)
    ax.set_title("Layer-wise Thinking–Answer Divergence\n(Qwen3-8B, n = 100, mean ± std)")
    ax.legend(fontsize=FS-4)
    style_ax(ax)
    save(fig, "plot5_layer_divergence")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Layer-36 distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot6(qdiv_df):
    if qdiv_df.empty or "layer36_value" not in qdiv_df.columns: return
    fig, ax = base_fig(8, 6)
    ax.hist(qdiv_df["layer36_value"], bins=20, color=C_BLUE,
            edgecolor="black", linewidth=1.0, zorder=3)
    mean_val = qdiv_df["layer36_value"].mean()
    ax.axvline(mean_val, color=C_RED, linewidth=2.5, linestyle="--",
               label=f"Mean = {mean_val:.3f}")
    ax.set_xlabel("Layer 36 Cosine Distance", fontsize=FS)
    ax.set_ylabel("Count", fontsize=FS)
    ax.set_title("Distribution of Layer-36 Divergence  (n = 100)")
    ax.legend(fontsize=FS-2)
    style_ax(ax)
    save(fig, "plot6_layer36_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 7 — Probe AUROC
# ══════════════════════════════════════════════════════════════════════════════
def plot7(probe_df, hidden_df):
    ls = hidden_df.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()
    fig, ax1 = base_fig(10, 6)
    ax2 = ax1.twinx()
    ax2.set_facecolor("white")
    ax2.spines["top"].set_visible(False)

    ax1.plot(probe_df["layer"], probe_df["auroc"], color=C_PURPLE,
             linewidth=2.5, label="Probe AUROC", zorder=4)
    ax1.fill_between(probe_df["layer"], 0.5, probe_df["auroc"],
                     alpha=0.15, color=C_PURPLE)
    ax2.plot(ls["layer"], ls["mean"], color=C_BLUE, linewidth=2.0,
             linestyle="--", alpha=0.85, label="Mean Divergence")
    ax2.fill_between(ls["layer"], ls["mean"]-ls["std"], ls["mean"]+ls["std"],
                     alpha=0.10, color=C_BLUE)

    ax1.axvspan(7, 10, alpha=0.07, color=C_ORANGE)
    ax1.axvline(36, color=C_RED, alpha=0.35, linewidth=2, linestyle=":")

    ax1.set_xlabel("Transformer Layer", fontsize=FS)
    ax1.set_ylabel("Probe AUROC", color=C_PURPLE, fontsize=FS)
    ax2.set_ylabel("Mean Cosine Distance", color=C_BLUE, fontsize=FS)
    ax1.tick_params(axis="y", labelcolor=C_PURPLE, labelsize=FS-2)
    ax2.tick_params(axis="y", labelcolor=C_BLUE, labelsize=FS-2)
    ax1.tick_params(axis="x", labelsize=FS-2)
    ax1.set_ylim(0.45, 1.10)
    ax2.set_ylim(0, ls["mean"].max()*1.6)
    ax1.set_title("Probe AUROC vs Thinking–Answer Divergence by Layer\n(n = 100)")
    l1,lab1 = ax1.get_legend_handles_labels()
    l2,lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lab1+lab2, fontsize=FS-4, loc="lower right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    save(fig, "plot7_probe_auroc")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 8 — Logit lens heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot8(logit_df):
    pivot = (logit_df.groupby(["layer","category","span_type"])
             ["probability_mass"].mean().unstack("span_type"))
    if "thinking" not in pivot.columns or "answer" not in pivot.columns: return
    pivot["diff"] = pivot["answer"] - pivot["thinking"]
    diff_mat = pivot["diff"].unstack("category").fillna(0)
    cmap = LinearSegmentedColormap.from_list("div",[C_RED,"white",C_BLUE])
    vmax = max(abs(diff_mat.values.min()), abs(diff_mat.values.max()))

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    im = ax.imshow(diff_mat.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(diff_mat.columns)))
    ax.set_xticklabels([c.capitalize() for c in diff_mat.columns], fontsize=FS-2)
    ax.set_yticks(range(len(diff_mat.index)))
    ax.set_yticklabels([f"Layer {l}" for l in diff_mat.index], fontsize=FS-2)
    ax.tick_params(axis="both", length=0)
    for i in range(diff_mat.values.shape[0]):
        for j in range(diff_mat.values.shape[1]):
            ax.text(j, i, f"{diff_mat.values[i,j]:.1e}",
                    ha="center", va="center", fontsize=10, color="black")
    ax.add_patch(plt.Rectangle((-0.5,0.5), len(diff_mat.columns), 4,
                 fill=False, edgecolor=C_ORANGE, linewidth=2.5, linestyle="--"))
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Answer − Thinking Probability Mass", fontsize=FS-4)
    cbar.ax.tick_params(labelsize=FS-6)
    ax.set_title("Logit Lens: Token Category Shift  (Answer vs Thinking)",
                 fontsize=FS, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS/"plot8_logit_lens_heatmap.pdf", format="pdf",
                bbox_inches="tight", facecolor="white")
    fig.savefig(PLOTS/"plot8_logit_lens_heatmap.png", format="png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved plot8_logit_lens_heatmap.pdf + .png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot A1 — Qwen backtracking distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot_a1(sim_df):
    vc = sim_df["closest_source"].value_counts()
    n  = len(sim_df)
    qwen_n  = vc.get("qwen_thinking", 0)
    graph_n = n - qwen_n

    fig, axes = base_fig2(2, 16, 6)

    # Left: binary
    ax = axes[0]
    labels = ["Backtracks to\nQwen Thinking", "Backtracks to\nGraph-PRefLexOR"]
    counts = [qwen_n, graph_n]
    colors = [C_BLUE, C_RED]
    bars = ax.bar(labels, counts, color=colors, width=0.45, zorder=3,
                  edgecolor="black", linewidth=1.2)
    ax.set_ylim(0, max(counts)*1.28)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + max(counts)*0.02,
                f"{v}  ({v/n*100:.0f}%)",
                ha="center", fontsize=FS-2, fontweight="bold")
    ax.set_ylabel("Number of Questions", fontsize=FS)
    ax.set_title("Binary Backtracking Split")
    style_ax(ax)

    # Right: full distribution
    ax = axes[1]
    nice = {"qwen_thinking":"Qwen\nThinking","graph_answer":"Graph\nAnswer",
            "graph_brainstorm":"Graph\nBrainstorm","graph_graph":"Graph\nGraph",
            "graph_synthesis":"Graph\nSynthesis","graph_patterns":"Graph\nPatterns"}
    labels2 = [nice.get(k,k) for k in vc.index]
    colors2 = [SOURCE_COLORS.get(k, C_GRAY) for k in vc.index]
    bars2 = ax.bar(labels2, vc.values, color=colors2, width=0.6, zorder=3,
                   edgecolor="black", linewidth=1.2)
    ax.set_ylim(0, vc.max()*1.28)
    for bar, v in zip(bars2, vc.values):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + vc.max()*0.02,
                str(v), ha="center", fontsize=FS-2, fontweight="bold")
    ax.set_ylabel("Number of Questions", fontsize=FS)
    ax.set_title("Full Source Distribution")
    ax.set_xticklabels(labels2, fontsize=FS-4)
    style_ax(ax)

    fig.suptitle("Qwen Final Answer Backtracking  (n = 100)",
                 fontsize=FS, fontweight="bold", y=1.02)
    save(fig, "plotA1_qwen_backtracking_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# Plot A2 — Graph-PRefLexOR internal backtracking
# ══════════════════════════════════════════════════════════════════════════════
def plot_a2(graph_bt_df):
    vc = graph_bt_df["closest_stage"].value_counts()
    n  = len(graph_bt_df)

    fig, axes = base_fig2(2, 16, 6)

    # Left: count
    ax = axes[0]
    nice = {"brainstorm":"Brainstorm","graph":"Graph",
            "patterns":"Patterns","synthesis":"Synthesis"}
    labels = [nice.get(k,k) for k in vc.index]
    colors = [STAGE_COLORS.get(k, C_GRAY) for k in vc.index]
    bars = ax.bar(labels, vc.values, color=colors, width=0.5, zorder=3,
                  edgecolor="black", linewidth=1.2)
    ax.set_ylim(0, vc.max()*1.28)
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + vc.max()*0.02,
                f"{v}  ({v/n*100:.0f}%)",
                ha="center", fontsize=FS-2, fontweight="bold")
    ax.set_ylabel("Number of Questions", fontsize=FS)
    ax.set_title("Closest Internal Stage  (n = 100)")
    style_ax(ax)

    # Right: mean similarity
    ax = axes[1]
    entries = [("sim_graph_answer_to_brainstorm","Brainstorm",C_GREEN),
               ("sim_graph_answer_to_graph","Graph",C_PURPLE),
               ("sim_graph_answer_to_patterns","Patterns",C_BROWN),
               ("sim_graph_answer_to_synthesis","Synthesis",C_ORANGE)]
    entries = [(c,l,col) for c,l,col in entries if c in graph_bt_df.columns]
    labels2  = [l for _,l,_ in entries]
    means    = [graph_bt_df[c].mean() for c,_,_ in entries]
    stds     = [graph_bt_df[c].std()  for c,_,_ in entries]
    colors2  = [col for _,_,col in entries]
    x = np.arange(len(labels2))
    bars2 = ax.bar(x, means, yerr=stds, color=colors2, width=0.5, zorder=3,
                   edgecolor="black", linewidth=1.2, capsize=6,
                   error_kw=dict(elinewidth=1.8, capthick=1.8, ecolor="black"))
    ax.set_ylim(0.7, 1.06)
    ax.set_xticks(x); ax.set_xticklabels(labels2, fontsize=FS-2)
    ax.set_ylabel("Mean Cosine Similarity (± std)", fontsize=FS)
    ax.set_title("Mean Similarity: Answer vs Stage")
    for bar, m in zip(bars2, means):
        ax.text(bar.get_x()+bar.get_width()/2,
                m+0.005, f"{m:.3f}",
                ha="center", va="bottom", fontsize=FS-4, fontweight="bold")
    style_ax(ax)

    fig.suptitle("Graph-PRefLexOR Internal Backtracking  (n = 100)",
                 fontsize=FS, fontweight="bold", y=1.02)
    save(fig, "plotA2_graph_internal_backtracking")

# ══════════════════════════════════════════════════════════════════════════════
# Plot B — Qwen vs Graph-PRefLexOR layer divergence
# ══════════════════════════════════════════════════════════════════════════════
def plot_b(qwen_hidden, graph_hidden):
    q_st = qwen_hidden.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()
    g_comb = graph_hidden[graph_hidden["stage"]=="reasoning"]
    g_st = g_comb.groupby("layer")["reasoning_answer_cosine_distance"].agg(["mean","std"]).reset_index()

    fig, ax = base_fig(10, 6)
    ax.plot(q_st["layer"], q_st["mean"], color=C_BLUE, linewidth=2.5,
            label="Qwen: Thinking vs Answer", zorder=4)
    ax.fill_between(q_st["layer"], q_st["mean"]-q_st["std"],
                    q_st["mean"]+q_st["std"], alpha=0.16, color=C_BLUE)
    ax.plot(g_st["layer"], g_st["mean"], color=C_RED, linewidth=2.5,
            linestyle="--", label="Graph-PRefLexOR: Reasoning vs Answer", zorder=4)
    ax.fill_between(g_st["layer"], g_st["mean"]-g_st["std"],
                    g_st["mean"]+g_st["std"], alpha=0.16, color=C_RED)
    ax.axvspan(7, 10, alpha=0.08, color=C_ORANGE)
    ax.axvline(36, color=C_GRAY, alpha=0.5, linewidth=1.5, linestyle=":")
    ax.text(8.5, max(q_st["mean"].max(), g_st["mean"].max())*0.55,
            "Layers\n7–10", fontsize=FS-6, color=C_ORANGE, ha="center", fontweight="bold")
    ax.set_xlabel("Transformer Layer", fontsize=FS)
    ax.set_ylabel("Cosine Distance", fontsize=FS)
    ax.set_title("Layer-wise Divergence: Qwen vs Graph-PRefLexOR  (n = 100, mean ± std)")
    ax.legend(fontsize=FS-4)
    style_ax(ax)
    save(fig, "plotB_qwen_vs_graph_layer_divergence")

# ══════════════════════════════════════════════════════════════════════════════
# Plot B2 — Graph-PRefLexOR per-stage divergence
# ══════════════════════════════════════════════════════════════════════════════
def plot_b2(graph_hidden):
    stages = ["brainstorm","graph","patterns","synthesis"]
    ls_map = {"brainstorm":"-","graph":"--","patterns":":","synthesis":"-."}

    fig, ax = base_fig(10, 6)
    for stage in stages:
        sub = graph_hidden[graph_hidden["stage"]==stage]
        if sub.empty: continue
        st = sub.groupby("layer")["reasoning_answer_cosine_distance"].agg(["mean","std"]).reset_index()
        c  = STAGE_COLORS[stage]
        ax.plot(st["layer"], st["mean"], color=c, linewidth=2.2,
                linestyle=ls_map[stage], label=stage.capitalize(), zorder=3)
        ax.fill_between(st["layer"], st["mean"]-st["std"], st["mean"]+st["std"],
                        alpha=0.10, color=c)
    ax.axvspan(7, 10, alpha=0.08, color=C_ORANGE)
    ax.axvline(36, color=C_GRAY, alpha=0.4, linewidth=1.5, linestyle=":")
    ax.set_xlabel("Transformer Layer", fontsize=FS)
    ax.set_ylabel("Cosine Distance (Stage vs Answer)", fontsize=FS)
    ax.set_title("Graph-PRefLexOR: Per-Stage Divergence from Final Answer  (n = 100)")
    ax.legend(fontsize=FS-4)
    style_ax(ax)
    save(fig, "plotB2_graph_per_stage_divergence")

# ══════════════════════════════════════════════════════════════════════════════
# Plot C1 — Qwen divergence by backtracking group
# ══════════════════════════════════════════════════════════════════════════════
def plot_c1(qwen_hidden, sim_df):
    group_a = set(sim_df[sim_df["closest_source"]=="qwen_thinking"]["id"])
    group_b = set(sim_df[sim_df["closest_source"]!="qwen_thinking"]["id"])

    fig, axes = base_fig2(2, 16, 6)

    for ax, grp_ids, label, color in [
        (axes[0], group_a, f"Group A: Backtracks to Qwen Thinking  (n={len(group_a)})", C_RED),
        (axes[1], group_b, f"Group B: Does NOT Backtrack to Qwen Thinking  (n={len(group_b)})", C_BLUE),
    ]:
        df_g = qwen_hidden[qwen_hidden["id"].isin(grp_ids)]
        if df_g.empty:
            ax.text(0.5,0.5,"No data",ha="center",transform=ax.transAxes); continue
        for _, grp in df_g.groupby("id"):
            g = grp.sort_values("layer")
            ax.plot(g["layer"], g["thinking_answer_cosine_distance"],
                    color=color, alpha=0.07, linewidth=0.7)
        st = df_g.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()
        ax.plot(st["layer"], st["mean"], color=color, linewidth=2.5, label="Mean", zorder=4)
        ax.fill_between(st["layer"], st["mean"]-st["std"], st["mean"]+st["std"],
                        alpha=0.18, color=color, label="± 1 Std Dev")
        ax.axvspan(7, 10, alpha=0.08, color=C_ORANGE)
        ax.axvline(36, color=C_GRAY, alpha=0.4, linewidth=1.5, linestyle=":")
        ax.set_title(label, fontsize=FS-4)
        ax.set_xlabel("Transformer Layer", fontsize=FS-2)
        ax.set_ylabel("Cosine Distance", fontsize=FS-2)
        ax.legend(fontsize=FS-6)
        style_ax(ax)

    fig.suptitle("Qwen Layer-wise Divergence by Backtracking Group",
                 fontsize=FS, fontweight="bold", y=1.02)
    save(fig, "plotC1_qwen_divergence_by_backtrack_group")

# ══════════════════════════════════════════════════════════════════════════════
# Plot C2 — Graph-PRefLexOR divergence by stage group
# ══════════════════════════════════════════════════════════════════════════════
def plot_c2(graph_hidden, graph_bt_df):
    group_a = set(graph_bt_df[graph_bt_df["closest_stage"].isin(["graph","synthesis"])]["id"])
    group_b = set(graph_bt_df[graph_bt_df["closest_stage"].isin(["brainstorm","patterns"])]["id"])
    df_comb = graph_hidden[graph_hidden["stage"]=="reasoning"]

    fig, axes = base_fig2(2, 16, 6)

    for ax, grp_ids, label, color in [
        (axes[0], group_a, f"Group A: Closest = Graph / Synthesis  (n={len(group_a)})", C_GREEN),
        (axes[1], group_b, f"Group B: Closest = Brainstorm / Patterns  (n={len(group_b)})", C_RED),
    ]:
        df_g = df_comb[df_comb["id"].isin(grp_ids)]
        if df_g.empty:
            ax.text(0.5,0.5,"No data",ha="center",transform=ax.transAxes); continue
        for _, grp in df_g.groupby("id"):
            g = grp.sort_values("layer")
            ax.plot(g["layer"], g["reasoning_answer_cosine_distance"],
                    color=color, alpha=0.09, linewidth=0.7)
        st = df_g.groupby("layer")["reasoning_answer_cosine_distance"].agg(["mean","std"]).reset_index()
        ax.plot(st["layer"], st["mean"], color=color, linewidth=2.5, label="Mean", zorder=4)
        ax.fill_between(st["layer"], st["mean"]-st["std"], st["mean"]+st["std"],
                        alpha=0.18, color=color, label="± 1 Std Dev")
        ax.axvspan(7, 10, alpha=0.08, color=C_ORANGE)
        ax.axvline(36, color=C_GRAY, alpha=0.4, linewidth=1.5, linestyle=":")
        ax.set_title(label, fontsize=FS-4)
        ax.set_xlabel("Transformer Layer", fontsize=FS-2)
        ax.set_ylabel("Cosine Distance", fontsize=FS-2)
        ax.legend(fontsize=FS-6)
        style_ax(ax)

    fig.suptitle("Graph-PRefLexOR Layer-wise Divergence by Stage Group",
                 fontsize=FS, fontweight="bold", y=1.02)
    save(fig, "plotC2_graph_divergence_by_stage_group")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    plt.rcParams.update({"font.size": FS})

    print("Loading data...")
    sim_df       = pd.read_csv(SIM_FILE)
    hidden_df    = pd.read_csv(HIDDEN_FILE)
    graph_hidden = pd.read_csv(GRAPH_HIDDEN)
    graph_bt_df  = pd.read_csv(GRAPH_BT)
    qtype_df     = pd.read_csv(SUMMARY_FILE) if SUMMARY_FILE.exists() else pd.DataFrame()
    probe_df     = pd.read_csv(PROBE_FILE)   if PROBE_FILE.exists()   else pd.DataFrame()
    logit_df     = pd.read_csv(LOGIT_FILE)   if LOGIT_FILE.exists()   else pd.DataFrame()
    qdiv_df      = pd.read_csv(QDIV_FILE)    if QDIV_FILE.exists()    else pd.DataFrame()

    print(f"  sim={len(sim_df)} hidden={len(hidden_df)} "
          f"graph_hidden={len(graph_hidden)} graph_bt={len(graph_bt_df)}")

    print("\nGenerating all publication plots...")
    plot1(sim_df)
    plot2(sim_df)
    plot3(sim_df)
    if not qtype_df.empty: plot4(qtype_df)
    plot5(hidden_df)
    if not qdiv_df.empty:  plot6(qdiv_df)
    if not probe_df.empty: plot7(probe_df, hidden_df)
    if not logit_df.empty: plot8(logit_df)
    plot_a1(sim_df)
    plot_a2(graph_bt_df)
    plot_b(hidden_df, graph_hidden)
    plot_b2(graph_hidden)
    plot_c1(hidden_df, sim_df)
    plot_c2(graph_hidden, graph_bt_df)

    pdfs = sorted(PLOTS.glob("*.pdf"))
    print(f"\nDone. {len(pdfs)} PDFs saved to {PLOTS}")
    for f in pdfs:
        print(f"  {f.name}  ({f.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
