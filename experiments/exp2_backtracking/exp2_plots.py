"""
exp2_plots.py — Generate all 6 Experiment 2 plots from output files.
Saves PNG files to /projects/bfir/ssourav/plots/
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

WORK_DIR  = Path("/projects/bfir/ssourav")
PLOT_DIR  = WORK_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

BLUE    = "#378ADD"
CORAL   = "#D85A30"
TEAL    = "#1D9E75"
PURPLE  = "#7F77DD"
AMBER   = "#BA7517"
DPURPLE = "#3C3489"
GRAY    = "#888780"

# ── Load files ────────────────────────────────────────────────────────────────
with open(WORK_DIR / "exp2_analysis.json") as f:
    analysis = json.load(f)

sim_df    = pd.read_csv(WORK_DIR / "exp2_backtracking_text_similarity.csv")
hidden_df = pd.read_csv(WORK_DIR / "exp2_hidden_state_divergence.csv")
align_df  = pd.read_csv(WORK_DIR / "exp2_hidden_to_graph_stage_alignment.csv")
summary_df= pd.read_csv(WORK_DIR / "exp2_summary_by_question_type.csv")

print("Files loaded.")
print(f"  sim_df     : {len(sim_df)} rows")
print(f"  hidden_df  : {len(hidden_df)} rows")
print(f"  align_df   : {len(align_df)} rows")
print(f"  summary_df : {len(summary_df)} rows")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Where does Qwen's answer backtrack?
# ══════════════════════════════════════════════════════════════════════════════
def plot1():
    dist = analysis["closest_stage_distribution"]
    labels = list(dist.keys())
    counts = list(dist.values())

    nice = {"graph_answer": "Graph-PRefLexOR\nanswer", "qwen_thinking": "Qwen\nthinking"}
    labels_nice = [nice.get(l, l) for l in labels]
    colors = [BLUE if "graph" in l else PURPLE for l in labels]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(labels_nice, counts, color=colors, height=0.5, zorder=3)

    for bar, v in zip(bars, counts):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0, max(counts) + 1.2)
    ax.set_xlabel("number of questions (n=10)")
    ax.set_title("Plot 1 — Where does Qwen's answer backtrack?")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.grid(axis="y", visible=False)

    caption = ("Across the 10-question pilot set, Qwen's final answer is closest to\n"
               "Graph-PRefLexOR's final answer in 6 cases and to its own thinking trace in 4.")
    fig.text(0.5, -0.08, caption, ha="center", fontsize=9, color=GRAY, style="italic")

    plt.tight_layout()
    out = PLOT_DIR / "plot1_backtrack_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Mean similarity of Qwen answer to each source
# ══════════════════════════════════════════════════════════════════════════════
def plot2():
    sims = analysis["mean_similarities"]

    key_map = {
        "sim_qwen_answer_to_graph_answer":    ("Graph-PRefLexOR answer",  BLUE),
        "sim_qwen_answer_to_qwen_thinking":   ("Qwen thinking",           PURPLE),
        "sim_qwen_answer_to_graph_brainstorm":("Graph brainstorm",        TEAL),
        "sim_qwen_answer_to_graph_synthesis": ("Graph synthesis",         CORAL),
        "sim_qwen_answer_to_graph_graph":     ("Graph graph",             AMBER),
        "sim_qwen_answer_to_graph_patterns":  ("Graph patterns",          DPURPLE),
    }

    # Sort descending by value
    items = sorted([(key_map[k][0], sims[k], key_map[k][1]) for k in key_map], key=lambda x: x[1], reverse=True)
    labels, values, colors = zip(*items)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=colors, height=0.55, zorder=3)

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10)

    ax.set_xlim(0.68, 0.975)
    ax.set_xlabel("mean cosine similarity")
    ax.set_title("Plot 2 — Mean similarity: Qwen answer vs reasoning sources")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.grid(axis="y", visible=False)

    caption = ("Qwen's final answer is most similar to final-answer style text, and also highly aligned with its own\n"
               "thinking and Graph-PRefLexOR's brainstorm/synthesis stages. Raw graph and pattern stages score lower\n"
               "because they are structurally different from final prose.")
    fig.text(0.5, -0.08, caption, ha="center", fontsize=9, color=GRAY, style="italic")

    plt.tight_layout()
    out = PLOT_DIR / "plot2_mean_similarities.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Per-question similarity heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot3():
    cols = [
        "sim_qwen_answer_to_qwen_thinking",
        "sim_qwen_answer_to_graph_brainstorm",
        "sim_qwen_answer_to_graph_graph",
        "sim_qwen_answer_to_graph_patterns",
        "sim_qwen_answer_to_graph_synthesis",
        "sim_qwen_answer_to_graph_answer",
    ]
    col_labels = [
        "Qwen\nthinking",
        "Graph\nbrainstorm",
        "Graph\ngraph",
        "Graph\npatterns",
        "Graph\nsynthesis",
        "Graph\nanswer",
    ]

    df = sim_df.sort_values("id").reset_index(drop=True)
    matrix = df[cols].values
    row_labels = [f"id {row.id}  {row.question_type}" for _, row in df.iterrows()]

    cmap = LinearSegmentedColormap.from_list("teal_ramp", ["#E1F5EE", "#1D9E75", "#04342C"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.70, vmax=0.97)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, ha="center")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.tick_params(axis="y", which="both", left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt_col = "white" if v > 0.87 else "#04342C"
            # Bold the max in each row
            is_max = (v == matrix[i].max())
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color=txt_col,
                    fontweight="bold" if is_max else "normal")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("cosine similarity", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("Plot 3 — Per-question similarity heatmap (bold = closest source per row)")

    caption = ("Similarity patterns vary by question. Bold values mark the closest source for each question.\n"
               "Some answers backtrack to Qwen's own thinking; others align more with Graph-PRefLexOR's final answer.")
    fig.text(0.5, -0.03, caption, ha="center", fontsize=9, color=GRAY, style="italic")

    plt.tight_layout()
    out = PLOT_DIR / "plot3_similarity_heatmap.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Question-type backtracking gaps
# ══════════════════════════════════════════════════════════════════════════════
def plot4():
    df = summary_df.copy()

    qt_short = {
        "causal_multiscale_reasoning":     "causal\nmultiscale",
        "cross_domain_mapping":            "cross\ndomain",
        "hidden_variable_identification":  "hidden\nvariable",
        "model_abstraction_and_breakdown": "model\nabstraction",
        "tradeoff_and_non_monotonicity":   "tradeoff\nnon-mono",
    }
    df["qt_label"] = df["question_type"].map(qt_short)

    metrics = [
        ("avg_thinking_gap",   "thinking gap",         BLUE),
        ("avg_graph_gap",      "graph gap",            CORAL),
        ("avg_synthesis_gap",  "synthesis gap",        TEAL),
        ("avg_preflexor_gap",  "preflexor answer gap", PURPLE),
    ]

    n_groups = len(df)
    n_bars   = len(metrics)
    x        = np.arange(n_groups)
    width    = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (col, label, color) in enumerate(metrics):
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, df[col], width=width, label=label,
                      color=color, zorder=3, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["qt_label"], fontsize=10)
    ax.set_ylabel("gap  (1 − similarity)")
    ax.set_title("Plot 4 — Question-type backtracking gaps  (lower = closer)")
    ax.legend(fontsize=10, framealpha=0.4, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.grid(axis="x", visible=False)

    # Annotate the min gap per group (= closest source)
    for idx, row in df.iterrows():
        gaps = {m[1]: row[m[0]] for m in metrics}
        closest = min(gaps, key=gaps.get)
        ax.text(idx, max(row[m[0]] for m in metrics) + 0.008,
                f"↑ {closest}", ha="center", fontsize=8, color=GRAY)

    caption = ("Tradeoff and non-monotonicity questions backtrack more to Qwen's own thinking;\n"
               "other types are consistently closer to Graph-PRefLexOR's answer.\n"
               "Pilot of 10 questions — trends are suggestive, not final.")
    fig.text(0.5, -0.04, caption, ha="center", fontsize=9, color=GRAY, style="italic")

    plt.tight_layout()
    out = PLOT_DIR / "plot4_qtype_gaps.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Layer-wise thinking–answer divergence
# ══════════════════════════════════════════════════════════════════════════════
def plot5():
    mean_div = (
        hidden_df.groupby("layer")["thinking_answer_cosine_distance"]
        .mean()
        .reset_index()
        .sort_values("layer")
    )

    # Per-id lines for background
    id_lines = {}
    for qid, grp in hidden_df.groupby("id"):
        g = grp.sort_values("layer")
        id_lines[qid] = (g["layer"].values, g["thinking_answer_cosine_distance"].values)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Faint individual lines
    for qid, (lx, ly) in id_lines.items():
        ax.plot(lx, ly, color=BLUE, alpha=0.12, linewidth=1.0, zorder=1)

    # Mean line
    ax.plot(mean_div["layer"], mean_div["thinking_answer_cosine_distance"],
            color=BLUE, linewidth=2.5, zorder=3, label="mean (all ids)")

    # Fill under mean
    ax.fill_between(mean_div["layer"], mean_div["thinking_answer_cosine_distance"],
                    alpha=0.12, color=BLUE, zorder=2)

    # Annotate peak zones
    peak_layer = int(mean_div.loc[mean_div["thinking_answer_cosine_distance"].idxmax(), "layer"])
    peak_val   = mean_div.loc[mean_div["layer"] == peak_layer, "thinking_answer_cosine_distance"].values[0]
    ax.annotate(f"layer {peak_layer}\n(peak)",
                xy=(peak_layer, peak_val),
                xytext=(peak_layer - 4, peak_val + 0.02),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5),
                fontsize=9, color=CORAL, fontweight="bold")

    # Annotate layer 7 bump
    bump_val = mean_div.loc[mean_div["layer"] == 7, "thinking_answer_cosine_distance"].values[0]
    ax.annotate("early peak\n(layers 7–10)",
                xy=(7, bump_val),
                xytext=(10, bump_val + 0.025),
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5),
                fontsize=9, color=AMBER, fontweight="bold")

    ax.set_xlabel("transformer layer")
    ax.set_ylabel("cosine distance (thinking vs answer)")
    ax.set_title("Plot 5 — Layer-wise thinking–answer divergence")
    ax.legend(fontsize=10, framealpha=0.4)
    ax.grid(alpha=0.25, linestyle="--")

    caption = ("Qwen's thinking and answer representations are nearly aligned in early layers (1–6),\n"
               "diverge sharply around layers 7–10, converge through mid-layers, then spike at the final\n"
               "projection layer (36). Faint lines = individual questions; bold = mean.")
    fig.text(0.5, -0.07, caption, ha="center", fontsize=9, color=GRAY, style="italic")

    plt.tight_layout()
    out = PLOT_DIR / "plot5_layer_divergence.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Hidden-state alignment to Graph-PRefLexOR stages by layer
# ══════════════════════════════════════════════════════════════════════════════
def plot6():
    mean_align = (
        align_df.groupby(["layer", "stage"])["similarity"]
        .mean()
        .reset_index()
        .sort_values(["stage", "layer"])
    )

    stage_styles = {
        "brainstorm": (BLUE,   "-",  "brainstorm"),
        "graph":      (CORAL,  "--", "graph"),
        "patterns":   (TEAL,   ":",  "patterns"),
        "synthesis":  (PURPLE, "-.", "synthesis"),
        "answer":     (AMBER,  "-",  "answer"),
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for stage, (color, ls, label) in stage_styles.items():
        sub = mean_align[mean_align["stage"] == stage]
        ax.plot(sub["layer"], sub["similarity"],
                color=color, linestyle=ls, linewidth=2.0,
                label=label, zorder=3)

    ax.set_xlabel("transformer layer")
    ax.set_ylabel("similarity (blended hidden state vs stage embedding)")
    ax.set_title("Plot 6 — Hidden-state alignment to Graph-PRefLexOR stages by layer")
    ax.legend(title="Stage", fontsize=10, framealpha=0.4, loc="lower right")
    ax.grid(alpha=0.25, linestyle="--")

    caption = ("Layer-wise alignment of Qwen's blended hidden-state representation to each Graph-PRefLexOR stage.\n"
               "Answer and brainstorm stages maintain the highest alignment throughout;\n"
               "raw graph and patterns stages remain consistently lower.")
    fig.text(0.5, -0.07, caption, ha="center", fontsize=9, color=GRAY, style="italic")

    plt.tight_layout()
    out = PLOT_DIR / "plot6_stage_alignment.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating plots...")
    plot1(); print("  Plot 1 done")
    plot2(); print("  Plot 2 done")
    plot3(); print("  Plot 3 done")
    plot4(); print("  Plot 4 done")
    plot5(); print("  Plot 5 done")
    plot6(); print("  Plot 6 done")
    print(f"\nAll plots saved to {PLOT_DIR}")
    import os
    for f in sorted(os.listdir(PLOT_DIR)):
        if f.endswith(".png"):
            size = os.path.getsize(PLOT_DIR / f) // 1024
            print(f"  {f}  ({size} KB)")
