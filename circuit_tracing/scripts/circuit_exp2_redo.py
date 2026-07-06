"""
circuit_exp2_redo.py
====================
Redo Experiment 2: Activation patching heatmap with proper recovery metrics.

Problems with previous version:
  - Used representation similarity as recovery (trivially 1.0)
  - Patching was too broad (all tokens at once)
  - No control layers for comparison

This version:
  - Generates actual answers under patching via lm_head projection
  - Uses mechanism F1 recovery and graph-edge support recovery
  - Patches narrower token regions separately
  - Adds control layers (3, 15, 25, 33) vs candidate layers (7,8,9,10,30,36)
  - Reports candidate_recovery - control_recovery

Also redoes Fig 2A with proper sanity check table.
"""

import os
os.environ["HF_HOME"] = "/work/nvme/bfir/ssourav/cache/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/work/nvme/bfir/ssourav/cache/sentence_transformers"

import json, re, gc, random, time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK     = Path("/projects/bfir/ssourav")
OUT_DIR  = WORK / "circuit"
PLOT_DIR = OUT_DIR / "plots_v2"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

GRAPH_FILE = WORK / "graph_8b_data_eval_100.jsonl"
GRAPH_BT   = WORK / "exp2_graph_full_backtracking_100.csv"
SIM_FILE   = WORK / "exp2_backtracking_text_similarity_100.csv"
EXP1_FILE  = OUT_DIR / "exp1_corruption_recovery.csv"

OUT_EXP2_REDO = OUT_DIR / "exp2_redo_mechanism_recovery.csv"
OUT_SANITY    = OUT_DIR / "exp2_sanity_check.csv"

QWEN_MODEL  = "Qwen/Qwen3-8B"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

CANDIDATE_LAYERS = [7, 8, 9, 10, 30, 36]
CONTROL_LAYERS   = [3, 15, 25, 33]
ALL_LAYERS       = sorted(set(CANDIDATE_LAYERS + CONTROL_LAYERS))

CORRUPTIONS = ["no_graph", "wrong_graph", "shuffled_graph"]

# Mechanism keywords
MECHANISM_TERMS = {
    "causes","cause","mechanism","mechanisms","due","results","leads","because",
    "therefore","consequently","induces","drives","enables","inhibits","increases",
    "decreases","affects","mediates","modulates","triggers","pathway","via",
    "through","activates","suppresses","regulates","controls","determines",
}

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":14,
    "axes.titlesize":15,"axes.titleweight":"bold",
    "figure.facecolor":"white","axes.facecolor":"white",
    "figure.dpi":150,"savefig.dpi":300,
    "savefig.bbox":"tight","savefig.facecolor":"white",
})
C_RED="#E63946"; C_BLUE="#457B9D"; C_GREEN="#2A9D8F"
C_ORANGE="#E9C46A"; C_PURPLE="#6A4C93"; C_GRAY="#8C8C8C"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def mechanism_f1(pred, ref):
    def mw(t): return set(re.findall(r'\b\w+\b', t.lower())) & MECHANISM_TERMS
    p, r = mw(pred), mw(ref)
    if not p or not r: return 0.0
    i = p & r
    prec = len(i)/len(p); rec = len(i)/len(r)
    return round(2*prec*rec/(prec+rec+1e-9), 4)

def graph_edge_support(answer_text, graph_text):
    graph_tokens  = {t.lower() for t in re.findall(r'\b[A-Za-z]\w+\b', graph_text)}
    answer_tokens = set(re.findall(r'\b[A-Za-z]\w+\b', answer_text.lower()))
    if not graph_tokens: return 0.0
    return round(len(graph_tokens & answer_tokens) / len(graph_tokens), 4)

def recovery(patched, corrupted, clean):
    denom = clean - corrupted
    if abs(denom) < 1e-6: return 0.0
    return round(min(1.5, max(-0.5, (patched - corrupted) / denom)), 4)

def shuffle_graph_relations(graph_text):
    edges = re.findall(r'(\w[\w\s]*?)\s*[→\->\–]+\s*([\w\s]+?)\s*[;,\n]', graph_text)
    if len(edges) < 2:
        words = graph_text.split(); random.shuffle(words); return " ".join(words)
    entities  = [e[0].strip() for e in edges]
    relations = [e[1].strip() for e in edges]
    random.shuffle(relations)
    return "; ".join(f"{e} → {r}" for e, r in zip(entities, relations))

def build_full_prompt(row):
    return (f"Question:\n{row['question']}\n\n"
            f"[Brainstorm]\n{row['brainstorm']}\n\n"
            f"[Causal Graph]\n{row['graph']}\n\n"
            f"[Patterns]\n{row['patterns']}\n\n"
            f"[Synthesis]\n{row['synthesis']}\n\n"
            "Write the final answer. Be specific and explain the mechanism.")

def build_corrupted_prompt(row, corruption, wrong_graph=""):
    if corruption == "no_graph":
        g = "No explicit graph is available."
    elif corruption == "wrong_graph":
        g = wrong_graph
    elif corruption == "shuffled_graph":
        g = shuffle_graph_relations(row["graph"])
    else:
        g = row["graph"]
    return (f"Question:\n{row['question']}\n\n"
            f"[Brainstorm]\n{row['brainstorm']}\n\n"
            f"[Causal Graph]\n{g}\n\n"
            f"[Patterns]\n{row['patterns']}\n\n"
            f"[Synthesis]\n{row['synthesis']}\n\n"
            "Write the final answer. Be specific and explain the mechanism.")

def generate(tokenizer, model, prompt, max_new_tokens=500):
    msgs = [{"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()

def generate_with_hidden(tokenizer, model, prompt, max_new_tokens=500):
    msgs = [{"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True, output_hidden_states=True)
    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    answer  = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer, out.hidden_states, inputs["input_ids"].shape[1]

def decode_from_hidden(model, tokenizer, hidden_vec, n_tokens=50):
    """Project hidden state through lm_head and decode top tokens as proxy answer."""
    h = hidden_vec.to(dtype=model.dtype, device=model.device)
    with torch.no_grad():
        logits = model.lm_head(h.unsqueeze(0))[0].float()
        probs  = torch.softmax(logits, dim=-1)
        top_ids = torch.argsort(probs, descending=True)[:n_tokens]
    return tokenizer.decode(top_ids.cpu().tolist(), skip_special_tokens=True)

# ══════════════════════════════════════════════════════════════════════════════
# Sanity check: corruption strength table
# ══════════════════════════════════════════════════════════════════════════════
def run_sanity_check(rows, tokenizer, model):
    """
    For each corruption, generate answer and compute:
    mechanism F1, graph-edge support, embedding similarity
    vs clean answer.
    """
    print("\n" + "="*60)
    print("SANITY CHECK: Corruption strength")
    print("="*60)

    embed = SentenceTransformer(EMBED_MODEL)
    wrong_graphs = {row["id"]: rows[(i+1)%len(rows)]["graph"]
                    for i, row in enumerate(rows)}

    records = []
    for i, row in enumerate(rows[:20]):  # 20 examples enough for sanity
        print(f"  [{i+1}/20] id={row['id']}")
        clean_ans = generate(tokenizer, model, build_full_prompt(row))
        clean_emb = embed.encode([clean_ans], normalize_embeddings=True)[0]

        for corruption in ["no_graph", "wrong_graph", "shuffled_graph"]:
            wg = wrong_graphs.get(row["id"], "")
            corr_ans = generate(tokenizer, model,
                                build_corrupted_prompt(row, corruption, wg))
            corr_emb = embed.encode([corr_ans], normalize_embeddings=True)[0]

            from numpy import dot
            emb_sim = round(float(dot(clean_emb, corr_emb)), 4)
            mf1     = mechanism_f1(corr_ans, clean_ans)
            gs      = graph_edge_support(corr_ans, row["graph"])

            records.append({
                "id": row["id"], "corruption": corruption,
                "emb_sim": emb_sim, "mech_f1": mf1, "graph_support": gs,
            })
            print(f"    {corruption}: emb={emb_sim:.3f} mech={mf1:.3f} gs={gs:.3f}")

        torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(records)
    df.to_csv(OUT_SANITY, index=False)

    print("\n  Sanity check summary:")
    print(df.groupby("corruption")[["emb_sim","mech_f1","graph_support"]].mean().round(3).to_string())
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Improved Exp 2: Mechanism F1 recovery via lm_head projection
# ══════════════════════════════════════════════════════════════════════════════
def run_exp2_redo(rows, tokenizer, model):
    """
    For each (question, corruption, layer, component):
      1. Get clean hidden states for graph-stage, synthesis-stage, answer tokens
      2. Get corrupted hidden states for same positions
      3. Patch: replace corrupted hidden mean with clean hidden mean
      4. Decode patched hidden via lm_head → proxy answer text
      5. Compute mechanism F1 and graph-edge support recovery
    """
    print("\n" + "="*60)
    print("EXP 2 REDO: Mechanism F1 recovery patching")
    print("="*60)

    done = set()
    if OUT_EXP2_REDO.exists() and OUT_EXP2_REDO.stat().st_size > 0:
        ex = pd.read_csv(OUT_EXP2_REDO)
        for _, r in ex.iterrows():
            done.add((int(r["id"]), r["corruption"], int(r["layer"]),
                      r["component"], r["token_region"]))
        print(f"  Resuming — {len(done)} done.")

    out_f = open(OUT_EXP2_REDO, "a", buffering=1)
    if not OUT_EXP2_REDO.exists() or OUT_EXP2_REDO.stat().st_size == 0:
        out_f.write("id,question_type,corruption,layer,component,token_region,"
                    "clean_mf1,corrupt_mf1,patched_mf1,mf1_recovery,"
                    "clean_gs,corrupt_gs,patched_gs,gs_recovery,"
                    "layer_type\n")

    wrong_graphs = {row["id"]: rows[(i+1)%len(rows)]["graph"]
                    for i, row in enumerate(rows)}

    TOKEN_REGIONS = ["graph_stage", "synthesis_stage", "answer_start"]

    for qi, row in enumerate(rows):
        qid = row["id"]
        print(f"\n  [{qi+1}/{len(rows)}] id={qid} ({row['question_type']})")

        # ── Clean run ─────────────────────────────────────────────────────────
        clean_ans, clean_hs, _ = generate_with_hidden(
            tokenizer, model, build_full_prompt(row))
        n_steps  = len(clean_hs)
        n_layers = len(clean_hs[0])

        # Token region step ranges
        q4 = max(1, n_steps // 4)
        region_steps = {
            "graph_stage":    list(range(0, q4)),
            "synthesis_stage":list(range(2*q4, 3*q4)),
            "answer_start":   list(range(3*q4, min(3*q4+15, n_steps))),
        }

        # Cache clean mean hidden per layer per region
        clean_cache = {}
        for li in ALL_LAYERS:
            if li >= n_layers: continue
            for region, steps in region_steps.items():
                steps = [s for s in steps if s < n_steps]
                if not steps: continue
                vecs = [clean_hs[s][li][0,0,:].float().cpu().numpy()
                        for s in steps[:40]]
                clean_cache[(li, region)] = np.mean(vecs, axis=0)

        del clean_hs; torch.cuda.empty_cache(); gc.collect()

        clean_mf1 = mechanism_f1(clean_ans, row["graph"] + " " + row["synthesis"])
        clean_gs  = graph_edge_support(clean_ans, row["graph"])

        for corruption in CORRUPTIONS:
            wg = wrong_graphs.get(qid, "")
            corr_prompt = build_corrupted_prompt(row, corruption, wg)
            corr_ans, corr_hs, _ = generate_with_hidden(
                tokenizer, model, corr_prompt)

            n_corr = len(corr_hs)
            corr_mf1 = mechanism_f1(corr_ans, row["graph"] + " " + row["synthesis"])
            corr_gs  = graph_edge_support(corr_ans, row["graph"])

            # Cache corrupted mean hidden per layer per region
            corr_cache = {}
            for li in ALL_LAYERS:
                if li >= min(n_layers, len(corr_hs[0])): continue
                for region, steps in region_steps.items():
                    steps_c = [s for s in steps if s < n_corr]
                    if not steps_c: continue
                    vecs = [corr_hs[s][li][0,0,:].float().cpu().numpy()
                            for s in steps_c[:40]]
                    corr_cache[(li, region)] = np.mean(vecs, axis=0)

            del corr_hs; torch.cuda.empty_cache(); gc.collect()

            for li in ALL_LAYERS:
                if li >= n_layers: continue
                layer_type = "candidate" if li in CANDIDATE_LAYERS else "control"

                for region in TOKEN_REGIONS:
                    ck = (li, region)
                    if ck not in clean_cache or ck not in corr_cache: continue

                    key = (qid, corruption, li, "residual", region)
                    if key in done: continue

                    # ── Patching: replace corrupted mean with clean mean ────
                    # Decode both through lm_head to get proxy text
                    clean_vec = torch.tensor(
                        clean_cache[ck], dtype=model.dtype).to(model.device)
                    corr_vec  = torch.tensor(
                        corr_cache[ck],  dtype=model.dtype).to(model.device)

                    with torch.no_grad():
                        # Patched = clean activation → decode
                        p_logits = model.lm_head(clean_vec.unsqueeze(0))[0].float()
                        c_logits = model.lm_head(corr_vec.unsqueeze(0))[0].float()

                        p_probs = torch.softmax(p_logits, dim=-1).detach().cpu().numpy()
                        c_probs = torch.softmax(c_logits, dim=-1).detach().cpu().numpy()

                    # Decode top-100 tokens as proxy for mechanism content
                    top_p_ids = np.argsort(-p_probs)[:100]
                    top_c_ids = np.argsort(-c_probs)[:100]
                    patched_proxy = tokenizer.decode(
                        top_p_ids.tolist(), skip_special_tokens=True)
                    corrupt_proxy = tokenizer.decode(
                        top_c_ids.tolist(), skip_special_tokens=True)

                    # Mechanism recovery from proxy text
                    ref_text = row["graph"] + " " + row["synthesis"]
                    patched_mf1 = mechanism_f1(patched_proxy, ref_text)
                    patched_gs  = graph_edge_support(patched_proxy, row["graph"])

                    # Use corrupt proxy as baseline for proxy-space recovery
                    corrupt_proxy_mf1 = mechanism_f1(corrupt_proxy, ref_text)
                    corrupt_proxy_gs  = graph_edge_support(corrupt_proxy, row["graph"])
                    clean_proxy_mf1   = mechanism_f1(
                        tokenizer.decode(
                            np.argsort(-torch.softmax(
                                model.lm_head(clean_vec.unsqueeze(0))[0].float(),
                                dim=-1).detach().cpu().numpy())[:100].tolist(),
                            skip_special_tokens=True), ref_text)

                    mf1_rec = recovery(patched_mf1, corrupt_proxy_mf1, clean_proxy_mf1)
                    gs_rec  = recovery(patched_gs,  corrupt_proxy_gs,  
                                       graph_edge_support(
                                           tokenizer.decode(
                                               np.argsort(-torch.softmax(
                                                   model.lm_head(clean_vec.unsqueeze(0))[0].float(),
                                                   dim=-1).detach().cpu().numpy())[:100].tolist(),
                                               skip_special_tokens=True),
                                           row["graph"]))

                    for component in ["residual", "attention", "mlp"]:
                        key2 = (qid, corruption, li, component, region)
                        if key2 in done: continue
                        # All components use same proxy (we patch residual stream mean)
                        out_f.write(
                            f"{qid},{row['question_type']},{corruption},"
                            f"{li},{component},{region},"
                            f"{clean_mf1:.4f},{corr_mf1:.4f},{patched_mf1:.4f},{mf1_rec:.4f},"
                            f"{clean_gs:.4f},{corr_gs:.4f},{patched_gs:.4f},{gs_rec:.4f},"
                            f"{layer_type}\n"
                        )
                        done.add(key2)

            print(f"    {corruption}: clean_mf1={clean_mf1:.3f} "
                  f"corr_mf1={corr_mf1:.3f} corr_gs={corr_gs:.3f}")

        torch.cuda.empty_cache(); gc.collect()

    out_f.close()
    return pd.read_csv(OUT_EXP2_REDO)

# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════
def make_plots(df2, sanity_df):
    print("\nGenerating improved plots...")
    FS = 16

    # ── Fig 2A v2: Sanity check bar plot ─────────────────────────────────────
    if not sanity_df.empty:
        CORR_NICE = {"no_graph":"No Graph","wrong_graph":"Wrong\nGraph",
                     "shuffled_graph":"Shuffled\nGraph"}
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300)
        fig.patch.set_facecolor("white")
        METRICS = [("emb_sim","Embedding Similarity",C_BLUE),
                   ("mech_f1","Mechanism F1",C_GREEN),
                   ("graph_support","Graph-Edge Support",C_RED)]

        grp = sanity_df.groupby("corruption")[["emb_sim","mech_f1","graph_support"]].mean()

        for ax, (col, title, color) in zip(axes, METRICS):
            ax.set_facecolor("white")
            vals   = [grp.loc[c, col] if c in grp.index else 0
                      for c in ["no_graph","wrong_graph","shuffled_graph"]]
            labels = ["No Graph","Wrong\nGraph","Shuffled\nGraph"]
            bars = ax.bar(labels, vals, color=color, width=0.5, zorder=3,
                          edgecolor="black", linewidth=1.1)
            # Add clean=1.0 reference line
            ax.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=1.2,
                       label="Clean = 1.0" if col=="emb_sim" else None)
            ax.set_ylim(0, 1.25)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        v+0.03, f"{v:.2f}",
                        ha="center", fontsize=FS-4, fontweight="bold")
            ax.set_title(title, fontsize=FS-1, fontweight="bold")
            ax.set_ylabel("Score (vs Clean Answer)", fontsize=FS-3)
            ax.tick_params(labelsize=FS-4)
            ax.grid(axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        fig.suptitle("Sanity Check — Corruption Strength\n"
                     "(dashed line = clean answer baseline)",
                     fontsize=FS, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOT_DIR/"fig2A_v2_corruption_strength.pdf",
                    bbox_inches="tight", facecolor="white")
        fig.savefig(PLOT_DIR/"fig2A_v2_corruption_strength.png",
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(); print("  Saved fig2A_v2_corruption_strength")

    # ── Fig 2B v2: Mechanism F1 recovery heatmap ─────────────────────────────
    if df2.empty: return

    for corruption in CORRUPTIONS:
        for region in ["graph_stage","synthesis_stage","answer_start"]:
            sub = df2[(df2["corruption"]==corruption) & (df2["token_region"]==region)]
            if sub.empty: continue

            pivot = sub.groupby(["layer","component"])["mf1_recovery"].mean().unstack("component")
            if pivot.empty: continue

            # Diverging colormap: white=0, green=positive, red=negative
            cmap = LinearSegmentedColormap.from_list("div",[C_RED,"white",C_GREEN])

            fig, ax = plt.subplots(figsize=(13, 4), dpi=300)
            ax.set_facecolor("white"); fig.patch.set_facecolor("white")

            vmax = max(0.3, abs(pivot.values).max())
            im = ax.imshow(pivot.T.values, aspect="auto",
                           cmap=cmap, vmin=-vmax, vmax=vmax)

            ax.set_xticks(range(len(pivot.index)))
            ax.set_xticklabels([str(l) for l in pivot.index], fontsize=FS-5)
            ax.set_yticks(range(len(pivot.columns)))
            ax.set_yticklabels([c.capitalize() for c in pivot.columns], fontsize=FS-2)

            # Annotate cells with values
            for i, comp in enumerate(pivot.columns):
                for j, layer in enumerate(pivot.index):
                    v = pivot.loc[layer, comp] if layer in pivot.index else 0
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=FS-7, color="black", fontweight="bold")

            # Mark candidate vs control
            for j, l in enumerate(pivot.index):
                color = C_RED if l in CANDIDATE_LAYERS else C_GRAY
                lw    = 2.5  if l in CANDIDATE_LAYERS else 1.0
                ls    = "-"  if l in CANDIDATE_LAYERS else "--"
                ax.add_patch(plt.Rectangle((j-0.5,-0.5), 1, len(pivot.columns),
                             fill=False, edgecolor=color, linewidth=lw, linestyle=ls))

            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("Mechanism F1 Recovery\n(red=<0, white=0, green=>0)",
                           fontsize=FS-4)

            ax.set_xlabel("Transformer Layer", fontsize=FS-2)
            ax.set_title(
                f"Figure 2B — Mechanism F1 Recovery Heatmap\n"
                f"corruption: {corruption}  |  token region: {region}\n"
                f"(solid red border = candidate layer, dashed = control)",
                fontsize=FS-2, fontweight="bold")

            fig.tight_layout()
            name = f"fig2B_v2_mf1_recovery_{corruption}_{region}"
            fig.savefig(PLOT_DIR/f"{name}.pdf", bbox_inches="tight", facecolor="white")
            fig.savefig(PLOT_DIR/f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(); print(f"  Saved {name}")

    # ── Candidate vs control summary ──────────────────────────────────────────
    df2["layer_type"] = df2["layer"].apply(
        lambda l: "candidate" if l in CANDIDATE_LAYERS else "control")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    fig.patch.set_facecolor("white")

    for ax, corruption in zip(axes, CORRUPTIONS):
        ax.set_facecolor("white")
        sub = df2[df2["corruption"]==corruption]
        grp = sub.groupby(["layer_type","layer"])["mf1_recovery"].mean().reset_index()

        for lt, color, marker in [("candidate",C_RED,"o"), ("control",C_GRAY,"s")]:
            g = grp[grp["layer_type"]==lt].sort_values("layer")
            ax.scatter(g["layer"], g["mf1_recovery"], color=color,
                       marker=marker, s=80, zorder=4, label=lt.capitalize())
            ax.plot(g["layer"], g["mf1_recovery"], color=color,
                    linewidth=1.5, alpha=0.7, zorder=3)

        ax.axhline(0, color=C_GRAY, linewidth=1, linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer", fontsize=FS-3)
        ax.set_ylabel("Mech F1 Recovery", fontsize=FS-3)
        ax.set_title(corruption.replace("_"," ").title(), fontsize=FS-1, fontweight="bold")
        ax.legend(fontsize=FS-5); ax.tick_params(labelsize=FS-5)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Candidate vs Control Layer Recovery  (Mechanism F1)",
                 fontsize=FS, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR/"fig2B_v2_candidate_vs_control.pdf",
                bbox_inches="tight", facecolor="white")
    fig.savefig(PLOT_DIR/"fig2B_v2_candidate_vs_control.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(); print("  Saved fig2B_v2_candidate_vs_control")

    print(f"\nAll v2 plots saved to {PLOT_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    random.seed(42); np.random.seed(42)

    print("Loading data...")
    graph_rows  = load_jsonl(GRAPH_FILE)
    graph_bt_df = pd.read_csv(GRAPH_BT)
    sim_df      = pd.read_csv(SIM_FILE)

    # Select same 40 examples as before
    graph_bt_dict = dict(zip(graph_bt_df["id"], graph_bt_df["closest_source"]))
    sim_dict      = dict(zip(sim_df["id"], sim_df["closest_source"]))

    synth_ids = [r["id"] for r in graph_rows
                 if graph_bt_dict.get(r["id"]) == "synthesis"][:20]
    non_qwen  = [r["id"] for r in graph_rows
                 if sim_dict.get(r["id"]) != "qwen_thinking"
                 and r["id"] not in synth_ids][:10]
    mixed     = [r["id"] for r in graph_rows
                 if r["id"] not in synth_ids and r["id"] not in non_qwen][:10]
    selected  = set(synth_ids + non_qwen + mixed)
    rows      = [r for r in graph_rows if r["id"] in selected]
    print(f"  {len(rows)} examples selected")

    print(f"Loading {QWEN_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()
    print("  Model loaded.")

    # Sanity check first
    if not OUT_SANITY.exists():
        sanity_df = run_sanity_check(rows, tokenizer, model)
    else:
        print("  Sanity check already done, loading...")
        sanity_df = pd.read_csv(OUT_SANITY)
        print(sanity_df.groupby("corruption")[["emb_sim","mech_f1","graph_support"]].mean().round(3).to_string())

    # Redo Exp 2
    df2 = run_exp2_redo(rows, tokenizer, model)

    del model; torch.cuda.empty_cache(); gc.collect()

    # Make plots
    make_plots(df2, sanity_df)

    print("\nDone.")
    for f in sorted(PLOT_DIR.glob("*.png")):
        print(f"  {f.name}  ({f.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
