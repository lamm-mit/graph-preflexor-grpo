"""
exp2_activation_patching_100.py  (v2 — representation-space patching)
======================================================================
Qwen3's rotary embeddings make hook-based activation injection infeasible.
Instead we use representation-space patching:

  For each question and each candidate/control layer:
    1. Clean run  : full thinking → collect layer-wise mean hidden states over answer span
    2. Corrupted  : no_thinking / shuffled_thinking → collect same
    3. rep_sim[layer] = cosine_sim(clean_hidden[layer], corrupted_hidden[layer])
       Low rep_sim → layer is corrupted → patching here could recover behavior
       High rep_sim → layer already looks clean → patching would help little
    4. patch_potential = 1 - rep_sim (higher = more patchable)

Also measures corrupted answer quality vs clean answer.

Output: /projects/bfir/ssourav/exp2_activation_patching_100.csv
Plots:  /projects/bfir/ssourav/plots_100/plot_patching_*.png
"""

import json, re, gc, os, time
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

WORK  = Path("/projects/bfir/ssourav")
PLOTS = WORK / "plots_100"
PLOTS.mkdir(exist_ok=True)

THINKING_FILE = WORK / "exp2_qwen_thinking_outputs_100.jsonl"
OUT_PATCH     = WORK / "exp2_activation_patching_100.csv"

QWEN_MODEL  = "Qwen/Qwen3-8B"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

CANDIDATE_LAYERS = [7, 8, 9, 10, 36]
CONTROL_LAYERS   = [1, 20, 30]
ALL_PATCH_LAYERS = CANDIDATE_LAYERS + CONTROL_LAYERS
MAX_QUESTIONS    = 50

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,
    "axes.titlesize":12,"axes.titleweight":"bold",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,"grid.linestyle":"--",
    "figure.dpi":150,"savefig.dpi":150,
    "savefig.bbox":"tight","savefig.facecolor":"white",
})
BLUE="#378ADD"; CORAL="#D85A30"; TEAL="#1D9E75"
PURPLE="#7F77DD"; AMBER="#BA7517"; GRAY="#888780"

MECHANISM_TERMS = {
    "causes","cause","mechanism","due","results","leads","because",
    "therefore","consequently","induces","drives","enables","inhibits",
    "increases","decreases","affects","mediates","modulates","triggers",
}

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

def token_overlap(a, b):
    def tok(t): return set(re.findall(r'\b\w+\b', t.lower()))
    sa, sb = tok(a), tok(b)
    if not sa or not sb: return 0.0
    i = sa & sb
    p = len(i)/len(sb); r = len(i)/len(sa)
    return round(2*p*r/(p+r+1e-9), 4)

def mechanism_overlap(a, b):
    def mw(t): return set(re.findall(r'\b\w+\b', t.lower())) & MECHANISM_TERMS
    sa, sb = mw(a), mw(b)
    if not sa or not sb: return 0.0
    return round(len(sa&sb)/(len(sa|sb)+1e-9), 4)

def shuffle_thinking(text):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sents) <= 1:
        words = text.split(); np.random.shuffle(words); return " ".join(words)
    np.random.shuffle(sents); return " ".join(sents)

def find_think_split(tokenizer, generated_ids):
    gen_list = generated_ids.tolist()
    lo, hi, split_pos = 1, len(gen_list), None
    while lo <= hi:
        mid = (lo + hi) // 2
        if "</think>" in tokenizer.decode(gen_list[:mid], skip_special_tokens=False):
            split_pos = mid; hi = mid - 1
        else:
            lo = mid + 1
    return split_pos

def generate_with_hidden(tokenizer, model, prompt, enable_thinking, max_new_tokens=5000):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
    gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    gen_text = re.sub(r"<think>[\s\S]*?</think>", "", gen_text).strip()
    return gen_ids, gen_text, outputs.hidden_states

def extract_answer_hidden(hidden_states, split_pos, layers):
    n_steps  = len(hidden_states)
    n_layers = len(hidden_states[0])
    steps    = list(range(split_pos, n_steps)) if split_pos else list(range(n_steps))
    if not steps: steps = list(range(n_steps))
    result = {}
    for li in layers:
        if li >= n_layers: continue
        vecs = [hidden_states[s][li][0, 0, :].float().cpu().numpy()
                for s in steps[:100]]
        result[li] = np.mean(vecs, axis=0)
    return result

def main():
    print("Loading data...")
    rows = [r for r in load_jsonl(THINKING_FILE) if r.get("has_think_split", False)]
    rows = rows[:MAX_QUESTIONS]
    print(f"  {len(rows)} questions")

    print(f"Loading {QWEN_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    print("  Model loaded.")

    embed_model = SentenceTransformer(EMBED_MODEL)
    print("  Embed model loaded.")

    done = set()
    if OUT_PATCH.exists() and os.path.getsize(OUT_PATCH) > 0:
        ex = pd.read_csv(OUT_PATCH)
        for _, r in ex.iterrows():
            done.add((int(r["id"]), r["corruption_type"], int(r["patched_layer"])))
        print(f"  Resuming — {len(done)} done.")

    out_f = open(OUT_PATCH, "a", buffering=1)
    if not OUT_PATCH.exists() or os.path.getsize(OUT_PATCH) == 0:
        out_f.write("id,question_type,closest_source,corruption_type,patched_layer,"
                    "layer_type,corrupted_answer_similarity,patch_potential,"
                    "rep_sim_corrupted_to_clean,entity_overlap,mechanism_overlap\n")

    start = time.time()
    for qi, row in enumerate(rows):
        q       = row["question"]
        think   = row["qwen_thinking"]
        closest = row.get("closest_source", "")
        qt      = row["question_type"]
        qid     = int(row["id"])
        print(f"\n[{qi+1}/{len(rows)}] id={qid} ({qt})")

        # Clean run
        clean_prompt = (
            "You are answering an open-ended scientific reasoning question.\n\n"
            "/think\n\n"
            f"Question:\n{q}\n\n"
            "Think carefully about mechanisms, causal relations, and explanations. "
            "Then give the final answer."
        )
        print("  Clean run...")
        c_ids, c_ans, c_hs = generate_with_hidden(
            tokenizer, model, clean_prompt, enable_thinking=True, max_new_tokens=5000)
        c_split  = find_think_split(tokenizer, c_ids)
        c_hidden = extract_answer_hidden(c_hs, c_split, ALL_PATCH_LAYERS)
        c_emb    = embed_model.encode([c_ans], normalize_embeddings=True)[0]
        del c_hs; torch.cuda.empty_cache(); gc.collect()
        print(f"  Clean ({len(c_ans.split())} words): {c_ans[:80]}...")

        for ct in ["no_thinking", "shuffled_thinking"]:
            if ct == "no_thinking":
                corr_prompt = (
                    "You are answering an open-ended scientific reasoning question.\n\n"
                    f"Question:\n{q}\n\nGive a clear final answer. Explain the mechanism."
                )
                enab = False
            else:
                sh = shuffle_thinking(think)
                corr_prompt = (
                    "You are answering an open-ended scientific reasoning question.\n\n"
                    f"Question:\n{q}\n\n"
                    f"Here are some initial thoughts (disordered):\n{sh[:800]}\n\n"
                    "Now give the final answer."
                )
                enab = False

            print(f"  Corrupted: {ct}...")
            _, corr_ans, corr_hs = generate_with_hidden(
                tokenizer, model, corr_prompt, enable_thinking=enab, max_new_tokens=800)
            corr_hidden = extract_answer_hidden(corr_hs, None, ALL_PATCH_LAYERS)
            corr_emb    = embed_model.encode([corr_ans], normalize_embeddings=True)[0]
            del corr_hs; torch.cuda.empty_cache(); gc.collect()

            corr_sim  = cosine_sim_np(c_emb, corr_emb)
            ent_ov    = token_overlap(c_ans, corr_ans)
            mech_ov   = mechanism_overlap(c_ans, corr_ans)

            for li in ALL_PATCH_LAYERS:
                key = (qid, ct, li)
                if key in done: continue
                lt = "candidate" if li in CANDIDATE_LAYERS else "control"
                rep_sim = (cosine_sim_np(c_hidden[li], corr_hidden[li])
                           if li in c_hidden and li in corr_hidden else float("nan"))
                patch_pot = round(1.0 - rep_sim, 4) if not np.isnan(rep_sim) else float("nan")
                out_f.write(f"{qid},{qt},{closest},{ct},{li},{lt},"
                            f"{round(corr_sim,4)},{patch_pot},"
                            f"{round(rep_sim,4) if not np.isnan(rep_sim) else ''},"
                            f"{ent_ov},{mech_ov}\n")
                done.add(key)

            cand_str = " ".join(
                f"L{l}:{round(1-cosine_sim_np(c_hidden.get(l,np.zeros(1)),corr_hidden.get(l,np.zeros(1))),3)}"
                for l in CANDIDATE_LAYERS if l in c_hidden and l in corr_hidden)
            print(f"    corr_sim={corr_sim:.3f}  patch_potential: {cand_str}")

        del c_ids, c_ans, c_emb, c_hidden
        torch.cuda.empty_cache(); gc.collect()
        eta = (time.time()-start)/(qi+1)*(len(rows)-qi-1)
        print(f"  ETA: {eta/60:.1f} min")

    out_f.close()
    df = pd.read_csv(OUT_PATCH)
    print(f"\nDone. {len(df)} rows → {OUT_PATCH}")
    make_plots(df)

def make_plots(df):
    print("\nGenerating plots...")
    LAYER_ORDER = sorted(df["patched_layer"].unique())

    # Plot A: patch potential by layer
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, ct in zip(axes, ["no_thinking","shuffled_thinking"]):
        sub  = df[df["corruption_type"]==ct]
        grp  = sub.groupby("patched_layer")["patch_potential"]
        means= grp.mean().reindex(LAYER_ORDER)
        stds = grp.std().reindex(LAYER_ORDER).fillna(0)
        ax.plot(LAYER_ORDER, means, color=BLUE, linewidth=2.5)
        ax.fill_between(LAYER_ORDER, means-stds, means+stds, alpha=0.15, color=BLUE)
        for l in CANDIDATE_LAYERS:
            ax.axvline(l, color=AMBER, alpha=0.4, linewidth=1.5, linestyle="--")
        ax.set_title(f"corruption: {ct}"); ax.set_xlabel("layer")
        if ax==axes[0]: ax.set_ylabel("patch potential (1 − rep similarity)")
    fig.suptitle("Patch potential by layer — high = more patchable",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS/"plot_patching_potential_by_layer.png"); plt.close()

    # Plot B: rep sim heatmap by question type
    for ct in ["no_thinking","shuffled_thinking"]:
        sub   = df[df["corruption_type"]==ct]
        pivot = sub.groupby(["question_type","patched_layer"])["rep_sim_corrupted_to_clean"].mean().unstack()
        if pivot.empty: continue
        cmap  = LinearSegmentedColormap.from_list("rep",[CORAL,"white",BLUE])
        fig, ax = plt.subplots(figsize=(10, 3.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(l) for l in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([s.replace("_"," ") for s in pivot.index], fontsize=9)
        plt.colorbar(im, ax=ax, label="rep similarity (corrupted vs clean)")
        for j,l in enumerate(pivot.columns):
            if l in CANDIDATE_LAYERS:
                ax.add_patch(plt.Rectangle((j-0.5,-0.5),1,len(pivot.index),
                             fill=False,edgecolor=AMBER,linewidth=2))
        ax.set_title(f"Rep similarity heatmap — {ct}")
        plt.tight_layout()
        plt.savefig(PLOTS/f"plot_patching_rep_sim_{ct}.png"); plt.close()

    # Plot C: candidate vs control
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ct in zip(axes, ["no_thinking","shuffled_thinking"]):
        sub = df[df["corruption_type"]==ct]
        grp = sub.groupby("layer_type")["patch_potential"].agg(["mean","std"])
        for i,(lt,row) in enumerate(grp.iterrows()):
            c = BLUE if lt=="candidate" else GRAY
            ax.bar(lt, row["mean"], yerr=row["std"], color=c, capsize=5, zorder=3, width=0.4)
            ax.text(i, row["mean"]+row["std"]+0.005, f"{row['mean']:.3f}",
                    ha="center", fontsize=10)
        ax.set_ylabel("mean patch potential"); ax.set_title(f"corruption: {ct}")
        ax.grid(axis="y",alpha=0.25); ax.grid(axis="x",visible=False)
    fig.suptitle("Candidate vs control layer patch potential", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS/"plot_patching_candidate_vs_control.png"); plt.close()

    print(f"  Plots saved to {PLOTS}")
    print("\n── Mean patch potential by layer ──")
    print(df.groupby(["corruption_type","patched_layer"])["patch_potential"]
          .mean().unstack("patched_layer").round(3).to_string())

if __name__ == "__main__":
    main()
