"""
exp2_publication_plots.py
=========================
Regenerates all Experiment 2 plots (A1, A2, B, B2, C1, C2 + plots 1-8)
in publication style matching the reference figure:
  - White background, clean axes (left + bottom spines only)
  - Bold titles, value labels above bars
  - Steelblue / coral / sage color palette
  - Dashed grid lines on y-axis only
  - Saved as PDF (vector) + PNG (raster)

No GPU needed — reads from saved CSV files only.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK  = Path("/projects/bfir/ssourav")
PLOTS = WORK / "plots_pub"
PLOTS.mkdir(exist_ok=True)

# Input files
SIM_FILE        = WORK / "exp2_backtracking_text_similarity_100.csv"
HIDDEN_FILE     = WORK / "exp2_hidden_state_divergence_100.csv"
GRAPH_HIDDEN    = WORK / "exp2_graph_hidden_state_divergence_100.csv"
GRAPH_BT_FILE   = WORK / "exp2_graph_internal_backtracking_100.csv"
GRAPH_FULL_BT_FILE = WORK / "exp2_graph_full_backtracking_100.csv"
SUMMARY_FILE    = WORK / "exp2_summary_by_question_type_100.csv"
PROBE_FILE      = WORK / "exp2_layer_probe_thinking_vs_answer_100.csv"
LOGIT_FILE      = WORK / "exp2_logit_lens_token_categories_100.csv"
QDIV_FILE       = WORK / "exp2_question_level_divergence_summary_100.csv"

# ── Publication style ─────────────────────────────────────────────────────────
# Match reference: clean, minimal, bold titles, value labels
STYLE = {
    "font.family":        "DejaVu Sans",
    "font.size":          13,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.labelsize":     13,
    "axes.labelweight":   "normal",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.linewidth":     1.2,
    "axes.grid":          True,
    "grid.alpha":         0.4,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.8,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    11,
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
    "savefig.format":     "pdf",
}
plt.rcParams.update(STYLE)

# ── Color palette (matches reference steelblue/coral/sage) ───────────────────
C_BLUE   = "#4878CF"   # steelblue — primary
C_CORAL  = "#D65F5F"   # coral/salmon — secondary
C_GREEN  = "#6ACC65"   # sage green — tertiary
C_PURPLE = "#956CB4"   # purple
C_AMBER  = "#D5BB21"   # amber/yellow
C_TEAL   = "#82C6E2"   # light teal
C_BROWN  = "#B47846"   # brown/orange
C_GRAY   = "#8C8C8C"   # neutral gray

SOURCE_COLORS = {
    "qwen_thinking":    C_CORAL,
    "graph_answer":     C_BLUE,
    "graph_brainstorm": C_GREEN,
    "graph_graph":      C_PURPLE,
    "graph_synthesis":  C_AMBER,
    "graph_patterns":   C_BROWN,
}
STAGE_COLORS = {
    "brainstorm": C_GREEN,
    "graph":      C_PURPLE,
    "patterns":   C_BROWN,
    "synthesis":  C_AMBER,
    "reasoning":  C_TEAL,
}

def save(fig, name):
    """Save as both PDF and PNG."""
    pdf_path = PLOTS / f"{name}.pdf"
    png_path = PLOTS / f"{name}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=200)
    plt.close(fig)
    print(f"  Saved {name}.pdf + .png")

def bar_labels(ax, bars, fmt="{:.0f}", offset_frac=0.01, fontsize=11):
    """Add value labels above each bar."""
    ymax = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h): continue
        ax.text(bar.get_x() + bar.get_width()/2,
                h + ymax * offset_frac,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold")

def hbar_labels(ax, bars, fmt="{:.0f}", offset=0.3, fontsize=11):
    """Add value labels beside horizontal bars."""
    for bar in bars:
        w = bar.get_width()
        if np.isnan(w): continue
        ax.text(w + offset,
                bar.get_y() + bar.get_height()/2,
                fmt.format(w),
                ha="left", va="center",
                fontsize=fontsize, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Backtracking distribution (replaces old plot1)
# ══════════════════════════════════════════════════════════════════════════════
def plot1(sim_df):
    vc = sim_df["closest_source"].value_counts()
    nice = {
        "graph_answer":    "Graph-PRefLexOR\nAnswer",
        "qwen_thinking":   "Qwen\nThinking",
        "graph_brainstorm":"Graph\nBrainstorm",
        "graph_graph":     "Graph\nGraph",
        "graph_synthesis": "Graph\nSynthesis",
        "graph_patterns":  "Graph\nPatterns",
    }
    labels = [nice.get(k, k) for k in vc.index]
    colors = [SOURCE_COLORS.get(k, C_GRAY) for k in vc.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vc.values, color=colors, width=0.6, zorder=3,
                  edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, vc.max() * 1.22)
    bar_labels(ax, bars, fmt="{:.0f}")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Qwen Answer Backtracking: Closest Reasoning Source\n(n = 100)")
    ax.tick_params(axis="x", which="both", length=0)
    fig.tight_layout()
    save(fig, "plot1_backtrack_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Mean similarities (replaces old plot2)
# ══════════════════════════════════════════════════════════════════════════════
def plot2(sim_df):
    SIM_COLS = [
        ("sim_qwen_answer_to_graph_answer",    "Graph-PRefLexOR Answer", C_BLUE),
        ("sim_qwen_answer_to_qwen_thinking",   "Qwen Thinking",          C_CORAL),
        ("sim_qwen_answer_to_graph_brainstorm","Graph Brainstorm",       C_GREEN),
        ("sim_qwen_answer_to_graph_synthesis", "Graph Synthesis",        C_AMBER),
        ("sim_qwen_answer_to_graph_graph",     "Graph Graph",            C_PURPLE),
        ("sim_qwen_answer_to_graph_patterns",  "Graph Patterns",         C_BROWN),
    ]
    # Sort by mean
    data = sorted([(lbl, sim_df[col].mean(), sim_df[col].std(), c)
                   for col, lbl, c in SIM_COLS], key=lambda x: -x[1])
    labels, means, stds, colors = zip(*data)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.6,
                  capsize=5, zorder=3, edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="#555555"))
    ax.set_ylim(0.6, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Mean Similarity: Qwen Final Answer vs Reasoning Sources\n(± std, n = 100)")
    ax.tick_params(axis="x", which="both", length=0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2,
                m + ax.get_ylim()[1]*0.01,
                f"{m:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "plot2_mean_similarities")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Per-question heatmap
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

    df_s = sim_df.sort_values(["question_type","id"]).reset_index(drop=True)
    matrix = df_s[SIM_COLS].values

    cmap = LinearSegmentedColormap.from_list("teal",["#f0f9f6","#1a7a5e","#0a3d30"])
    fig, ax = plt.subplots(figsize=(10, max(7, len(df_s)*0.16)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.65, vmax=0.98)

    ax.set_xticks(range(len(COL_LABELS)))
    ax.set_xticklabels(COL_LABELS, fontsize=10)
    ax.set_yticks(range(len(df_s)))
    ax.set_yticklabels([f"id {r.id} · {r.question_type[:20]}"
                        for _, r in df_s.iterrows()], fontsize=7)
    ax.tick_params(axis="both", length=0)

    # White box for closest source
    for i, row in df_s.iterrows():
        try:
            j = SIM_COLS.index(f"sim_qwen_answer_to_{row['closest_source']}")
            ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,
                         fill=False,edgecolor="white",linewidth=2.5))
        except ValueError:
            pass

    cbar = plt.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title("Per-Question Similarity Heatmap\n(white box = closest source per row)", pad=10)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    save(fig, "plot3_similarity_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Question-type backtracking gaps
# ══════════════════════════════════════════════════════════════════════════════
def plot4(qtype_df):
    metrics = [
        ("avg_thinking_gap",    "Thinking Gap",    C_CORAL),
        ("avg_graph_gap",       "Graph Gap",       C_PURPLE),
        ("avg_synthesis_gap",   "Synthesis Gap",   C_AMBER),
        ("avg_graph_answer_gap","Graph Answer Gap",C_BLUE),
    ]
    metrics = [(c,l,col) for c,l,col in metrics if c in qtype_df.columns]

    qt_labels = [s.replace("_"," ").title() for s in qtype_df["question_type"]]
    x = np.arange(len(qtype_df)); w = 0.18
    n = len(metrics)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (col, label, color) in enumerate(metrics):
        offset = (i - (n-1)/2) * w
        bars = ax.bar(x+offset, qtype_df[col], width=w, label=label,
                      color=color, zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(qt_labels, fontsize=11)
    ax.set_ylabel("Gap  (1 − Similarity)")
    ax.set_title("Backtracking Gaps by Question Type\n(lower = closer to that source, n = 100)")
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    save(fig, "plot4_qtype_gaps")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 5: Layer-wise divergence mean ± std
# ══════════════════════════════════════════════════════════════════════════════
def plot5(hidden_df):
    ls = hidden_df.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))

    # Individual faint lines
    for qid, grp in hidden_df.groupby("id"):
        g = grp.sort_values("layer")
        ax.plot(g["layer"], g["thinking_answer_cosine_distance"],
                color=C_BLUE, alpha=0.06, linewidth=0.7)

    ax.plot(ls["layer"], ls["mean"], color=C_BLUE, linewidth=2.5,
            label="Mean (all questions)", zorder=4)
    ax.fill_between(ls["layer"],
                    ls["mean"]-ls["std"],
                    ls["mean"]+ls["std"],
                    alpha=0.18, color=C_BLUE, label="± 1 Std Dev")

    # Annotate key zones
    ax.axvspan(7, 10, alpha=0.08, color=C_AMBER)
    ax.axvline(36, color=C_CORAL, alpha=0.4, linewidth=1.5, linestyle="--")

    peak = ls.loc[ls["mean"].idxmax()]
    ax.annotate(f"Peak: layer {int(peak['layer'])}\n({peak['mean']:.3f})",
                xy=(peak["layer"], peak["mean"]),
                xytext=(peak["layer"]+3, peak["mean"]+0.025),
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=1.5),
                fontsize=10, color=C_CORAL, fontweight="bold")
    ax.text(8.5, ls["mean"].max()*0.6, "Layers\n7–10", fontsize=9,
            color=C_AMBER, ha="center", fontweight="bold")

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Cosine Distance")
    ax.set_title("Layer-wise Thinking–Answer Divergence\n(Qwen3-8B, n = 100, mean ± std)")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "plot5_layer_divergence")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 6: Layer-36 distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot6(qdiv_df):
    if qdiv_df.empty or "layer36_value" not in qdiv_df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(qdiv_df["layer36_value"], bins=20, color=C_BLUE,
            edgecolor="white", linewidth=0.8, zorder=3)
    mean_val = qdiv_df["layer36_value"].mean()
    ax.axvline(mean_val, color=C_CORAL, linewidth=2, linestyle="--",
               label=f"Mean = {mean_val:.3f}")
    ax.set_xlabel("Layer 36 Cosine Distance")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Layer-36 Divergence\n(Qwen3-8B, n = 100)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "plot6_layer36_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 7: Probe AUROC
# ══════════════════════════════════════════════════════════════════════════════
def plot7(probe_df, hidden_df):
    layer_stats = hidden_df.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    ax1.plot(probe_df["layer"], probe_df["auroc"],
             color=C_PURPLE, linewidth=2.5, label="Probe AUROC", zorder=4)
    ax1.fill_between(probe_df["layer"], 0.5, probe_df["auroc"],
                     alpha=0.15, color=C_PURPLE)

    ax2.plot(layer_stats["layer"], layer_stats["mean"],
             color=C_BLUE, linewidth=2.0, linestyle="--",
             alpha=0.8, label="Mean Divergence")
    ax2.fill_between(layer_stats["layer"],
                     layer_stats["mean"]-layer_stats["std"],
                     layer_stats["mean"]+layer_stats["std"],
                     alpha=0.10, color=C_BLUE)

    ax1.axvspan(7, 10, alpha=0.06, color=C_AMBER)
    ax1.axvline(36, color=C_CORAL, alpha=0.35, linewidth=1.5, linestyle=":")

    ax1.set_xlabel("Transformer Layer")
    ax1.set_ylabel("Probe AUROC", color=C_PURPLE)
    ax2.set_ylabel("Mean Cosine Distance", color=C_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_PURPLE)
    ax2.tick_params(axis="y", labelcolor=C_BLUE)
    ax1.set_ylim(0.45, 1.08)
    ax2.set_ylim(0, layer_stats["mean"].max()*1.6)

    ax1.set_title("Linear Probe AUROC vs Thinking–Answer Divergence by Layer\n"
                  "(probe: thinking=0 vs answer=1; n = 100)")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lab1+lab2, fontsize=10, loc="lower right")
    ax1.grid(axis="y", alpha=0.4, linestyle="--")
    ax1.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "plot7_probe_auroc")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 8: Logit lens heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot8(logit_df):
    pivot = (logit_df.groupby(["layer","category","span_type"])
             ["probability_mass"].mean().unstack("span_type"))
    if "thinking" not in pivot.columns or "answer" not in pivot.columns:
        return
    pivot["diff"] = pivot["answer"] - pivot["thinking"]
    diff_mat = pivot["diff"].unstack("category").fillna(0)

    cmap = LinearSegmentedColormap.from_list("div",[C_CORAL,"white",C_BLUE])
    vmax = max(abs(diff_mat.values.min()), abs(diff_mat.values.max()))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(diff_mat.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(diff_mat.columns)))
    ax.set_xticklabels([c.capitalize() for c in diff_mat.columns], fontsize=11)
    ax.set_yticks(range(len(diff_mat.index)))
    ax.set_yticklabels([f"Layer {l}" for l in diff_mat.index], fontsize=11)
    ax.tick_params(axis="both", length=0)

    for i in range(diff_mat.values.shape[0]):
        for j in range(diff_mat.values.shape[1]):
            v = diff_mat.values[i,j]
            ax.text(j, i, f"{v:.1e}", ha="center", va="center",
                    fontsize=9, color="black")

    # Box layers 7-10
    ax.add_patch(plt.Rectangle((-0.5, 0.5), len(diff_mat.columns), 4,
                 fill=False, edgecolor=C_AMBER, linewidth=2.5, linestyle="--"))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Answer − Thinking Probability Mass", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title("Logit Lens: Token Category Shift (Answer vs Thinking)\n"
                 "(blue = higher in answer states; orange box = layers 7–10)")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    save(fig, "plot8_logit_lens_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# Plot A1: Qwen backtracking distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot_a1(sim_df):
    vc = sim_df["closest_source"].value_counts()
    qwen_n  = vc.get("qwen_thinking", 0)
    graph_n = len(sim_df) - qwen_n
    n = len(sim_df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: binary
    ax = axes[0]
    labels  = ["Backtracks to\nQwen Thinking", "Backtracks to\nGraph-PRefLexOR"]
    counts  = [qwen_n, graph_n]
    colors  = [C_CORAL, C_BLUE]
    bars = ax.bar(labels, counts, color=colors, width=0.45, zorder=3,
                  edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, max(counts)*1.25)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + max(counts)*0.02,
                f"{v}  ({v/n*100:.0f}%)",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Qwen Answer Backtracking\n(Binary Split)")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.grid(axis="x", visible=False)

    # Right: full distribution
    ax = axes[1]
    nice = {"qwen_thinking":"Qwen Thinking","graph_answer":"Graph Answer",
            "graph_brainstorm":"Graph Brainstorm","graph_graph":"Graph Graph",
            "graph_synthesis":"Graph Synthesis","graph_patterns":"Graph Patterns"}
    labels_r = [nice.get(k,k) for k in vc.index]
    colors_r = [SOURCE_COLORS.get(k, C_GRAY) for k in vc.index]
    bars = ax.bar(labels_r, vc.values, color=colors_r, width=0.6, zorder=3,
                  edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, vc.max()*1.25)
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + vc.max()*0.02,
                f"{v}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Qwen Answer Closest Source\n(Full Distribution)")
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xticklabels(labels_r, fontsize=10, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Plot A1 — Qwen Final Answer Backtracking (n = 100)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "plotA1_qwen_backtracking_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# Plot A2: Graph-PRefLexOR internal backtracking
# ══════════════════════════════════════════════════════════════════════════════
def plot_a2(graph_bt_df):
    vc = graph_bt_df["closest_stage"].value_counts()
    n  = len(graph_bt_df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: count bar
    ax = axes[0]
    nice = {"brainstorm":"Brainstorm","graph":"Graph",
            "patterns":"Patterns","synthesis":"Synthesis"}
    labels  = [nice.get(k,k) for k in vc.index]
    colors  = [STAGE_COLORS.get(k, C_GRAY) for k in vc.index]
    bars = ax.bar(labels, vc.values, color=colors, width=0.5, zorder=3,
                  edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, vc.max()*1.25)
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + vc.max()*0.02,
                f"{v}  ({v/n*100:.0f}%)",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Graph-PRefLexOR Answer\nClosest Internal Stage")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.grid(axis="x", visible=False)

    # Right: mean similarity
    ax = axes[1]
    sim_cols = [("sim_graph_answer_to_brainstorm","Brainstorm",C_GREEN),
                ("sim_graph_answer_to_graph","Graph",C_PURPLE),
                ("sim_graph_answer_to_patterns","Patterns",C_BROWN),
                ("sim_graph_answer_to_synthesis","Synthesis",C_AMBER)]
    sim_cols = [(c,l,col) for c,l,col in sim_cols if c in graph_bt_df.columns]
    labels2  = [l for _,l,_ in sim_cols]
    means    = [graph_bt_df[c].mean() for c,_,_ in sim_cols]
    stds     = [graph_bt_df[c].std()  for c,_,_ in sim_cols]
    colors2  = [col for _,_,col in sim_cols]
    x = np.arange(len(labels2))
    bars2 = ax.bar(x, means, yerr=stds, color=colors2, width=0.5,
                   capsize=5, zorder=3, edgecolor="white", linewidth=0.8,
                   error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="#555"))
    ax.set_ylim(0.7, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(labels2)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Mean Similarity: Graph-PRefLexOR\nAnswer vs Reasoning Stages (± std)")
    ax.tick_params(axis="x", length=0)
    for bar, m in zip(bars2, means):
        ax.text(bar.get_x()+bar.get_width()/2,
                m + 0.005, f"{m:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Plot A2 — Graph-PRefLexOR Internal Backtracking (n = 100)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "plotA2_graph_internal_backtracking")

# ══════════════════════════════════════════════════════════════════════════════
# Plot B: Qwen vs Graph-PRefLexOR layer divergence
# ══════════════════════════════════════════════════════════════════════════════
def plot_b(qwen_hidden, graph_hidden):
    q_stats = qwen_hidden.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()
    g_combined = graph_hidden[graph_hidden["stage"]=="reasoning"]
    g_stats = g_combined.groupby("layer")["reasoning_answer_cosine_distance"].agg(["mean","std"]).reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(q_stats["layer"], q_stats["mean"], color=C_BLUE, linewidth=2.5,
            label="Qwen: Thinking vs Answer", zorder=4)
    ax.fill_between(q_stats["layer"],
                    q_stats["mean"]-q_stats["std"],
                    q_stats["mean"]+q_stats["std"],
                    alpha=0.14, color=C_BLUE)

    ax.plot(g_stats["layer"], g_stats["mean"], color=C_GREEN, linewidth=2.5,
            linestyle="--", label="Graph-PRefLexOR: Reasoning vs Answer", zorder=4)
    ax.fill_between(g_stats["layer"],
                    g_stats["mean"]-g_stats["std"],
                    g_stats["mean"]+g_stats["std"],
                    alpha=0.14, color=C_GREEN)

    ax.axvspan(7, 10, alpha=0.07, color=C_AMBER, label="Layers 7–10 zone")
    ax.axvline(36, color=C_CORAL, alpha=0.35, linewidth=1.5,
               linestyle=":", label="Layer 36")

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Cosine Distance")
    ax.set_title("Layer-wise Divergence: Qwen vs Graph-PRefLexOR\n"
                 "(Reasoning→Answer, mean ± std, n = 100)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "plotB_qwen_vs_graph_layer_divergence")

# ══════════════════════════════════════════════════════════════════════════════
# Plot B2: Graph-PRefLexOR per-stage divergence
# ══════════════════════════════════════════════════════════════════════════════
def plot_b2(graph_hidden):
    stages = ["brainstorm","graph","patterns","synthesis"]
    stage_ls = {"brainstorm":"-","graph":"--","patterns":":","synthesis":"-."}

    fig, ax = plt.subplots(figsize=(11, 5))
    for stage in stages:
        sub = graph_hidden[graph_hidden["stage"]==stage]
        if sub.empty: continue
        st = sub.groupby("layer")["reasoning_answer_cosine_distance"].agg(["mean","std"]).reset_index()
        c  = STAGE_COLORS[stage]
        ax.plot(st["layer"], st["mean"], color=c, linewidth=2.2,
                linestyle=stage_ls[stage], label=stage.capitalize(), zorder=3)
        ax.fill_between(st["layer"], st["mean"]-st["std"], st["mean"]+st["std"],
                        alpha=0.09, color=c)

    ax.axvspan(7, 10, alpha=0.07, color=C_AMBER)
    ax.axvline(36, color=C_CORAL, alpha=0.3, linewidth=1.5, linestyle=":")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Cosine Distance (Stage vs Answer)")
    ax.set_title("Graph-PRefLexOR: Per-Stage Divergence from Final Answer\n"
                 "(mean ± std, n = 100)")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "plotB2_graph_per_stage_divergence")

# ══════════════════════════════════════════════════════════════════════════════
# Plot C1: Qwen divergence split by backtracking group
# ══════════════════════════════════════════════════════════════════════════════
def plot_c1(qwen_hidden, sim_df):
    group_a = set(sim_df[sim_df["closest_source"]=="qwen_thinking"]["id"])
    group_b = set(sim_df[sim_df["closest_source"]!="qwen_thinking"]["id"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, grp_ids, label, color in [
        (axes[0], group_a, f"Group A: Backtracks to Qwen Thinking\n(n = {len(group_a)})", C_CORAL),
        (axes[1], group_b, f"Group B: Does NOT Backtrack to Qwen Thinking\n(n = {len(group_b)})", C_BLUE),
    ]:
        df_g = qwen_hidden[qwen_hidden["id"].isin(grp_ids)]
        if df_g.empty:
            ax.text(0.5,0.5,"No data",ha="center",transform=ax.transAxes); continue

        for qid, grp in df_g.groupby("id"):
            g = grp.sort_values("layer")
            ax.plot(g["layer"], g["thinking_answer_cosine_distance"],
                    color=color, alpha=0.08, linewidth=0.7)

        st = df_g.groupby("layer")["thinking_answer_cosine_distance"].agg(["mean","std"]).reset_index()
        ax.plot(st["layer"], st["mean"], color=color, linewidth=2.5,
                label="Mean", zorder=4)
        ax.fill_between(st["layer"], st["mean"]-st["std"], st["mean"]+st["std"],
                        alpha=0.16, color=color, label="± 1 Std Dev")

        ax.axvspan(7, 10, alpha=0.07, color=C_AMBER)
        ax.axvline(36, color=C_CORAL if color!=C_CORAL else C_BLUE,
                   alpha=0.3, linewidth=1.5, linestyle=":")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Transformer Layer")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        ax.grid(axis="x", visible=False)
        if ax == axes[0]:
            ax.set_ylabel("Cosine Distance (Thinking vs Answer)")

    fig.suptitle("Plot C1 — Qwen Layer-wise Divergence by Backtracking Group",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "plotC1_qwen_divergence_by_backtrack_group")

# ══════════════════════════════════════════════════════════════════════════════
# Plot C2: Graph-PRefLexOR divergence split by stage group
# ══════════════════════════════════════════════════════════════════════════════
def plot_c2(graph_hidden, graph_full_bt_df):
    # Split by the FULL cross-model backtracking (own reasoning stages vs Qwen outputs),
    # mirroring plot_c1's Qwen own-thinking-vs-other split. This matches the 92/8 own-vs-Qwen
    # split reported in Fig 8c / Fig 9 (left) and Section 2.5. The internal-only file cannot
    # represent "does NOT backtrack to own reasoning" (every example there is own-reasoning).
    own_stages = {"brainstorm", "graph", "patterns", "synthesis"}
    group_a = set(graph_full_bt_df[graph_full_bt_df["closest_source"].isin(own_stages)]["id"])
    group_b = set(graph_full_bt_df[~graph_full_bt_df["closest_source"].isin(own_stages)]["id"])
    df_comb = graph_hidden[graph_hidden["stage"]=="reasoning"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, grp_ids, label, color in [
        (axes[0], group_a, f"Backtracks to Own Reasoning (N = {len(group_a)})", C_GREEN),
        (axes[1], group_b, f"Does NOT Backtrack (N = {len(group_b)})", C_CORAL),
    ]:
        df_g = df_comb[df_comb["id"].isin(grp_ids)]
        if df_g.empty:
            ax.text(0.5,0.5,"No data",ha="center",transform=ax.transAxes); continue

        for qid, grp in df_g.groupby("id"):
            g = grp.sort_values("layer")
            ax.plot(g["layer"], g["reasoning_answer_cosine_distance"],
                    color=color, alpha=0.09, linewidth=0.7)

        st = df_g.groupby("layer")["reasoning_answer_cosine_distance"].agg(["mean","std"]).reset_index()
        ax.plot(st["layer"], st["mean"], color=color, linewidth=2.5,
                label="Mean", zorder=4)
        ax.fill_between(st["layer"], st["mean"]-st["std"], st["mean"]+st["std"],
                        alpha=0.16, color=color, label="± 1 Std Dev")

        ax.axvspan(7, 10, alpha=0.07, color=C_AMBER)
        ax.axvline(36, color=C_BLUE, alpha=0.3, linewidth=1.5, linestyle=":")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Transformer Layer")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        ax.grid(axis="x", visible=False)
        if ax == axes[0]:
            ax.set_ylabel("Cosine Distance (Reasoning vs Answer)")

    fig.suptitle("Plot C2 — Graph-PRefLexOR Layer-wise Divergence by Stage Group",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "plotC2_graph_divergence_by_stage_group")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data files...")
    sim_df      = pd.read_csv(SIM_FILE)
    hidden_df   = pd.read_csv(HIDDEN_FILE)
    graph_hidden= pd.read_csv(GRAPH_HIDDEN)
    graph_bt_df = pd.read_csv(GRAPH_BT_FILE)
    graph_full_bt_df = pd.read_csv(GRAPH_FULL_BT_FILE)
    qtype_df    = pd.read_csv(SUMMARY_FILE)   if SUMMARY_FILE.exists()  else pd.DataFrame()
    probe_df    = pd.read_csv(PROBE_FILE)     if PROBE_FILE.exists()    else pd.DataFrame()
    logit_df    = pd.read_csv(LOGIT_FILE)     if LOGIT_FILE.exists()    else pd.DataFrame()
    qdiv_df     = pd.read_csv(QDIV_FILE)      if QDIV_FILE.exists()     else pd.DataFrame()
    print(f"  sim={len(sim_df)} hidden={len(hidden_df)} "
          f"graph_hidden={len(graph_hidden)} graph_bt={len(graph_bt_df)}")

    print("\nGenerating publication plots (PDF + PNG)...")

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
    plot_c2(graph_hidden, graph_full_bt_df)

    print(f"\nAll plots saved to {PLOTS}")
    total_pdf = list(PLOTS.glob("*.pdf"))
    total_png = list(PLOTS.glob("*.png"))
    print(f"  {len(total_pdf)} PDF files, {len(total_png)} PNG files")
    for f in sorted(total_pdf):
        print(f"  {f.name}  ({f.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
