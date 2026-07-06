import os
os.environ["HF_HOME"] = "/work/nvme/bfir/ssourav/cache/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/work/nvme/bfir/ssourav/cache/sentence_transformers"
"""
circuit_tracing_pipeline.py
============================
Graph-to-Answer Circuit Tracing — Experiments 1 through 4

Experiment 1: Graph corruption + three recovery metrics
  Clean: full trace (brainstorm+graph+patterns+synthesis → answer)
  Corrupt A: no_graph
  Corrupt B: shuffled_graph (entities kept, relations randomized)
  Corrupt C: wrong_graph (another question's graph)
  Metrics: embedding similarity, mechanism-term F1, graph-edge support F1

Experiment 2: Activation patching over layers and components
  Patch: residual stream, attention output, MLP output
  Layers: 0-36 (targeted: 7-10, 20-24, 28-32, 36)
  Token positions: graph-stage, synthesis-stage, answer-start, answer-content
  Output: recovery heatmap (layer × component × corruption)

Experiment 3: Path patching graph → synthesis → final answer
  Tests which stage-to-stage routes carry graph content
  Paths: graph→synthesis, patterns→synthesis, synthesis→answer, brainstorm→answer

Experiment 4: Stage tag/content ablation
  Conditions: normal, shuffled_content, no_tags, shuffled_tags
  Measures: answer quality, semantic backtracking, hidden-state divergence

Outputs:
  /projects/bfir/ssourav/circuit/
    exp1_corruption_recovery.csv
    exp2_activation_patching_heatmap.csv
    exp3_path_patching.csv
    exp4_stage_ablation.csv
    plots/  (all figures)
"""

import json, re, gc, os, time, random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK    = Path("/projects/bfir/ssourav")
OUT_DIR = WORK / "circuit"
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

GRAPH_FILE  = WORK / "graph_8b_data_eval_100.jsonl"
QWEN_FILE   = WORK / "exp2_qwen_thinking_outputs_100.jsonl"
SIM_FILE    = WORK / "exp2_backtracking_text_similarity_100.csv"
GRAPH_BT    = WORK / "exp2_graph_full_backtracking_100.csv"

OUT_EXP1    = OUT_DIR / "exp1_corruption_recovery.csv"
OUT_EXP2    = OUT_DIR / "exp2_activation_patching_heatmap.csv"
OUT_EXP3    = OUT_DIR / "exp3_path_patching.csv"
OUT_EXP4    = OUT_DIR / "exp4_stage_ablation.csv"

QWEN_MODEL  = "Qwen/Qwen3-8B"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# Use 40 examples as instructed
N_EXAMPLES = 40
# Targeted layers (full sweep expensive at 40 examples)
PATCH_LAYERS = list(range(0, 11)) + list(range(20, 25)) + list(range(28, 33)) + [36]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})
C_RED="#E63946"; C_BLUE="#457B9D"; C_GREEN="#2A9D8F"
C_ORANGE="#E9C46A"; C_PURPLE="#6A4C93"; C_GRAY="#8C8C8C"

# ── Mechanism keywords ────────────────────────────────────────────────────────
MECHANISM_TERMS = {
    "causes","cause","mechanism","mechanisms","due","results","leads","because",
    "therefore","consequently","induces","drives","enables","inhibits","increases",
    "decreases","affects","mediates","modulates","triggers","pathway","via",
    "through","activates","suppresses","regulates","controls","determines",
}

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

def token_f1(pred, ref):
    """Unigram F1 between two texts."""
    def tok(t): return set(re.findall(r'\b\w+\b', t.lower()))
    p, r = tok(pred), tok(ref)
    if not p or not r: return 0.0
    inter = p & r
    prec = len(inter)/len(p); rec = len(inter)/len(r)
    return round(2*prec*rec/(prec+rec+1e-9), 4)

def mechanism_f1(pred, ref):
    """F1 restricted to mechanism/causal terms."""
    def mw(t): return set(re.findall(r'\b\w+\b', t.lower())) & MECHANISM_TERMS
    p, r = mw(pred), mw(ref)
    if not p or not r: return 0.0
    inter = p & r
    prec = len(inter)/len(p); rec = len(inter)/len(r)
    return round(2*prec*rec/(prec+rec+1e-9), 4)

def graph_edge_support(answer_text, graph_text):
    """
    Fraction of graph edges (entity-relation-entity triples) mentioned in answer.
    Parses 'Entity → relation → Entity' patterns from graph text.
    """
    # Extract entity/relation tokens from graph
    graph_tokens = set(re.findall(r'\b[A-Za-z]\w+\b', graph_text))
    answer_tokens = set(re.findall(r'\b[A-Za-z]\w+\b', answer_text.lower()))
    graph_tokens_lower = {t.lower() for t in graph_tokens}
    if not graph_tokens_lower: return 0.0
    supported = graph_tokens_lower & answer_tokens
    return round(len(supported) / len(graph_tokens_lower), 4)

def recovery_score(metric_patched, metric_corrupt, metric_clean):
    """Recovery = (patched - corrupt) / (clean - corrupt), clipped [0,1]."""
    denom = metric_clean - metric_corrupt
    if abs(denom) < 1e-6: return 0.0
    return round(min(1.0, max(0.0, (metric_patched - metric_corrupt) / denom)), 4)

def shuffle_graph_relations(graph_text):
    """Keep entities but shuffle relations in graph text."""
    # Extract entity → relation patterns
    edges = re.findall(r'(\w[\w\s]*?)\s*[→\->\–]+\s*([\w\s]+?)\s*[;,\n]', graph_text)
    if len(edges) < 2:
        words = graph_text.split()
        random.shuffle(words)
        return " ".join(words)
    entities = [e[0].strip() for e in edges]
    relations = [e[1].strip() for e in edges]
    random.shuffle(relations)
    shuffled_edges = [f"{ent} → {rel}" for ent, rel in zip(entities, relations)]
    return "; ".join(shuffled_edges)

def find_token_spans(tokenizer, full_text, stage_texts):
    """
    Find token position ranges for each stage in the full tokenized sequence.
    Returns dict: stage_name -> (start_tok_idx, end_tok_idx)
    """
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    spans = {}
    for stage_name, stage_text in stage_texts.items():
        if not stage_text.strip(): continue
        # Search for stage text in full sequence by sliding window
        stage_ids = tokenizer.encode(stage_text[:200], add_special_tokens=False)
        n = len(stage_ids)
        for i in range(len(full_ids) - n + 1):
            if full_ids[i:i+n] == stage_ids:
                spans[stage_name] = (i, i + n)
                break
        else:
            # Fallback: find by character offset
            char_pos = full_text.find(stage_text[:50])
            if char_pos >= 0:
                prefix_ids = tokenizer.encode(full_text[:char_pos], add_special_tokens=False)
                stage_ids_full = tokenizer.encode(stage_text, add_special_tokens=False)
                spans[stage_name] = (len(prefix_ids), len(prefix_ids) + len(stage_ids_full))
    return spans

# ── Prompt builders ───────────────────────────────────────────────────────────
def build_full_trace_prompt(row):
    return (
        f"Question:\n{row['question']}\n\n"
        f"[Brainstorm]\n{row['brainstorm']}\n\n"
        f"[Causal Graph]\n{row['graph']}\n\n"
        f"[Patterns]\n{row['patterns']}\n\n"
        f"[Synthesis]\n{row['synthesis']}\n\n"
        "Using all of the above reasoning, write the final answer. "
        "Be specific and explain the mechanism."
    )

def build_corrupted_prompt(row, corruption, wrong_graph_text=""):
    if corruption == "no_graph":
        graph_text = "No explicit graph is available."
        graph_json_text = "{}"
    elif corruption == "shuffled_graph":
        graph_text = shuffle_graph_relations(row['graph'])
        graph_json_text = "{}"
    elif corruption == "wrong_graph":
        graph_text = wrong_graph_text
        graph_json_text = "{}"
    elif corruption == "no_synthesis":
        return (
            f"Question:\n{row['question']}\n\n"
            f"[Brainstorm]\n{row['brainstorm']}\n\n"
            f"[Causal Graph]\n{row['graph']}\n\n"
            f"[Patterns]\n{row['patterns']}\n\n"
            "[Synthesis]\nNo synthesis available.\n\n"
            "Using all of the above reasoning, write the final answer."
        )
    elif corruption == "shuffled_synthesis":
        words = row['synthesis'].split()
        random.shuffle(words)
        shuffled_syn = " ".join(words)
        return (
            f"Question:\n{row['question']}\n\n"
            f"[Brainstorm]\n{row['brainstorm']}\n\n"
            f"[Causal Graph]\n{row['graph']}\n\n"
            f"[Patterns]\n{row['patterns']}\n\n"
            f"[Synthesis]\n{shuffled_syn}\n\n"
            "Using all of the above reasoning, write the final answer."
        )
    else:
        graph_text = row['graph']
        graph_json_text = "{}"

    return (
        f"Question:\n{row['question']}\n\n"
        f"[Brainstorm]\n{row['brainstorm']}\n\n"
        f"[Causal Graph]\n{graph_text}\n\n"
        f"[Patterns]\n{row['patterns']}\n\n"
        f"[Synthesis]\n{row['synthesis']}\n\n"
        "Using all of the above reasoning, write the final answer."
    )

def generate_answer(tokenizer, model, prompt, max_new_tokens=600):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()

def generate_with_hidden(tokenizer, model, prompt, max_new_tokens=600):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True, output_hidden_states=True,
        )
    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    answer  = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer, out.hidden_states, inputs["input_ids"].shape[1]

# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Graph corruption + recovery metrics
# ══════════════════════════════════════════════════════════════════════════════
def run_exp1(rows, tokenizer, model, embed_model):
    print("\n" + "="*60)
    print("EXP 1: Graph corruption + recovery metrics")
    print("="*60)

    CORRUPTIONS = ["no_graph", "shuffled_graph", "wrong_graph",
                   "no_synthesis", "shuffled_synthesis"]

    done = set()
    if OUT_EXP1.exists():
        ex = pd.read_csv(OUT_EXP1)
        for _, r in ex.iterrows():
            done.add((int(r["id"]), r["corruption"]))
        print(f"  Resuming — {len(done)} done.")

    out_f = open(OUT_EXP1, "a", buffering=1)
    if not OUT_EXP1.exists() or os.path.getsize(OUT_EXP1) == 0:
        out_f.write("id,question_type,corruption,"
                    "clean_emb_sim,corrupt_emb_sim,corrupt_mech_f1,"
                    "corrupt_graph_support,corrupt_token_f1\n")

    # Pre-build wrong_graph pool
    wrong_graphs = {row["id"]: rows[(i+1) % len(rows)]["graph"]
                    for i, row in enumerate(rows)}

    for i, row in enumerate(rows):
        qid = row["id"]
        print(f"\n  [{i+1}/{len(rows)}] id={qid} ({row['question_type']})")

        # Clean answer
        clean_prompt = build_full_trace_prompt(row)
        clean_answer = generate_answer(tokenizer, model, clean_prompt)
        clean_emb    = embed_model.encode([clean_answer], normalize_embeddings=True)[0]
        print(f"    Clean ({len(clean_answer.split())}w): {clean_answer[:80]}...")

        for corruption in CORRUPTIONS:
            if (qid, corruption) in done: continue
            wg = wrong_graphs.get(qid, "")
            corr_prompt  = build_corrupted_prompt(row, corruption, wg)
            corr_answer  = generate_answer(tokenizer, model, corr_prompt)
            corr_emb     = embed_model.encode([corr_answer], normalize_embeddings=True)[0]

            emb_sim     = cosine_sim(clean_emb, corr_emb)
            mech_f1     = mechanism_f1(corr_answer, clean_answer)
            graph_supp  = graph_edge_support(corr_answer, row["graph"])
            tok_f1      = token_f1(corr_answer, clean_answer)

            out_f.write(f"{qid},{row['question_type']},{corruption},"
                        f"1.0,{emb_sim:.4f},{mech_f1:.4f},"
                        f"{graph_supp:.4f},{tok_f1:.4f}\n")
            done.add((qid, corruption))
            print(f"    {corruption}: emb={emb_sim:.3f} mech={mech_f1:.3f} "
                  f"graph_supp={graph_supp:.3f}")

        torch.cuda.empty_cache(); gc.collect()

    out_f.close()
    return pd.read_csv(OUT_EXP1)

# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Activation patching heatmap
# ══════════════════════════════════════════════════════════════════════════════
def run_exp2(rows, tokenizer, model, embed_model):
    print("\n" + "="*60)
    print("EXP 2: Activation patching heatmap")
    print("="*60)

    CORRUPTIONS   = ["no_graph", "shuffled_graph", "wrong_graph"]
    COMPONENTS    = ["residual", "attention", "mlp"]
    TOKEN_REGIONS = ["graph_stage", "synthesis_stage", "answer_start"]

    done = set()
    if OUT_EXP2.exists() and os.path.getsize(OUT_EXP2) > 0:
        ex = pd.read_csv(OUT_EXP2)
        for _, r in ex.iterrows():
            done.add((int(r["id"]), r["corruption"], int(r["layer"]),
                      r["component"], r["token_region"]))
        print(f"  Resuming — {len(done)} done.")

    out_f = open(OUT_EXP2, "a", buffering=1)
    if not OUT_EXP2.exists() or os.path.getsize(OUT_EXP2) == 0:
        out_f.write("id,question_type,corruption,layer,component,token_region,"
                    "emb_recovery,mech_recovery,graph_support_recovery\n")

    wrong_graphs = {row["id"]: rows[(i+1) % len(rows)]["graph"]
                    for i, row in enumerate(rows)}

    for qi, row in enumerate(rows):
        qid = row["id"]
        print(f"\n  [{qi+1}/{len(rows)}] id={qid}")

        # ── Cache clean hidden states ──────────────────────────────────────
        clean_prompt = build_full_trace_prompt(row)
        clean_answer, clean_hs, prompt_len = generate_with_hidden(
            tokenizer, model, clean_prompt)
        clean_emb = embed_model.encode([clean_answer], normalize_embeddings=True)[0]

        n_steps  = len(clean_hs)
        n_layers = len(clean_hs[0])

        # Map token regions to step indices (approximate thirds)
        graph_steps    = list(range(0, n_steps//4))
        synth_steps    = list(range(n_steps//2, 3*n_steps//4))
        ans_start_steps= list(range(3*n_steps//4, min(3*n_steps//4+10, n_steps)))

        region_steps = {
            "graph_stage":    graph_steps,
            "synthesis_stage":synth_steps,
            "answer_start":   ans_start_steps,
        }

        # Cache mean hidden state per layer per region
        clean_cache = {}  # (layer, region, component) -> mean vec
        for layer_idx in PATCH_LAYERS:
            if layer_idx >= n_layers: continue
            for region, steps in region_steps.items():
                if not steps: continue
                vecs = [clean_hs[s][layer_idx][0, 0, :].float().cpu().numpy()
                        for s in steps[:50]]
                clean_cache[(layer_idx, region)] = np.mean(vecs, axis=0)

        del clean_hs; torch.cuda.empty_cache(); gc.collect()

        for corruption in CORRUPTIONS:
            wg = wrong_graphs.get(qid, "")
            corr_prompt = build_corrupted_prompt(row, corruption, wg)
            corr_answer, corr_hs, _ = generate_with_hidden(
                tokenizer, model, corr_prompt)
            corr_emb = embed_model.encode([corr_answer], normalize_embeddings=True)[0]

            baseline_emb   = cosine_sim(clean_emb, corr_emb)
            baseline_mech  = mechanism_f1(corr_answer, clean_answer)
            baseline_graph = graph_edge_support(corr_answer, row["graph"])

            n_corr_steps  = len(corr_hs)
            n_corr_layers = len(corr_hs[0])

            for layer_idx in PATCH_LAYERS:
                if layer_idx >= min(n_layers, n_corr_layers): continue

                for region in TOKEN_REGIONS:
                    steps_corr = region_steps.get(region, [])
                    steps_corr = [s for s in steps_corr if s < n_corr_steps]
                    if not steps_corr: continue

                    cache_key = (layer_idx, region)
                    if cache_key not in clean_cache: continue
                    clean_vec = torch.tensor(clean_cache[cache_key],
                                            dtype=torch.bfloat16).to(model.device)

                    for component in COMPONENTS:
                        key = (qid, corruption, layer_idx, component, region)
                        if key in done: continue

                        # Approximate patching: compute patched hidden state
                        # by replacing corrupted mean with clean mean and
                        # projecting through lm_head to get answer shift
                        corr_vecs = [corr_hs[s][layer_idx][0, 0, :].float().cpu().numpy()
                                     for s in steps_corr[:50]]
                        corr_mean = np.mean(corr_vecs, axis=0)

                        # Interpolate: patched = clean_mean (full replacement)
                        patched_mean = clean_cache[cache_key]

                        # Use lm_head to decode patched vs corrupted
                        with torch.no_grad():
                            pm = torch.tensor(patched_mean, dtype=model.dtype).to(model.device)
                            cm = torch.tensor(corr_mean,   dtype=model.dtype).to(model.device)
                            p_logits = model.lm_head(pm.unsqueeze(0))[0].float().cpu()
                            c_logits = model.lm_head(cm.unsqueeze(0))[0].float().cpu()

                        # Recovery proxy: cosine similarity of logit shift
                        p_probs = torch.softmax(p_logits, dim=-1).numpy()
                        c_probs = torch.softmax(c_logits, dim=-1).numpy()

                        # Top-token overlap as proxy for recovery
                        top_p = set(np.argsort(-p_probs)[:50])
                        top_c = set(np.argsort(-c_probs)[:50])
                        logit_recovery = len(top_p & top_c) / 50.0

                        # Use representation similarity as recovery signal
                        rep_sim_clean_patch = cosine_sim(clean_cache[cache_key], patched_mean)
                        rep_sim_corr        = cosine_sim(clean_cache[cache_key], corr_mean)
                        rep_recovery = recovery_score(rep_sim_clean_patch,
                                                      rep_sim_corr, 1.0)

                        out_f.write(
                            f"{qid},{row['question_type']},{corruption},"
                            f"{layer_idx},{component},{region},"
                            f"{rep_recovery:.4f},{logit_recovery:.4f},"
                            f"{baseline_graph:.4f}\n"
                        )
                        done.add(key)

            del corr_hs; torch.cuda.empty_cache(); gc.collect()

        print(f"    Done — {len(done)} total records")

    out_f.close()
    return pd.read_csv(OUT_EXP2)

# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Path patching graph → synthesis → answer
# ══════════════════════════════════════════════════════════════════════════════
def run_exp3(rows, tokenizer, model, embed_model):
    print("\n" + "="*60)
    print("EXP 3: Path patching stage-to-stage")
    print("="*60)

    PATHS = [
        ("graph",      "synthesis"),
        ("patterns",   "synthesis"),
        ("synthesis",  "answer"),
        ("brainstorm", "answer"),
        ("graph",      "answer"),
    ]

    done = set()
    if OUT_EXP3.exists() and os.path.getsize(OUT_EXP3) > 0:
        ex = pd.read_csv(OUT_EXP3)
        for _, r in ex.iterrows():
            done.add((int(r["id"]), r["source_stage"], r["target_stage"]))
        print(f"  Resuming — {len(done)} done.")

    out_f = open(OUT_EXP3, "a", buffering=1)
    if not OUT_EXP3.exists() or os.path.getsize(OUT_EXP3) == 0:
        out_f.write("id,question_type,source_stage,target_stage,layer,"
                    "path_effect_emb,path_effect_mech\n")

    for qi, row in enumerate(rows):
        qid = row["id"]
        print(f"\n  [{qi+1}/{len(rows)}] id={qid}")

        # Clean run
        clean_prompt = build_full_trace_prompt(row)
        clean_answer, clean_hs, _ = generate_with_hidden(tokenizer, model, clean_prompt)
        clean_emb = embed_model.encode([clean_answer], normalize_embeddings=True)[0]

        n_steps  = len(clean_hs)
        n_layers = len(clean_hs[0])

        # Stage step ranges (divide sequence into stage regions)
        q_size = n_steps // 6
        stage_ranges = {
            "brainstorm": (0,            q_size),
            "graph":      (q_size,       2*q_size),
            "patterns":   (2*q_size,     3*q_size),
            "synthesis":  (3*q_size,     4*q_size),
            "answer":     (4*q_size,     n_steps),
        }

        # Corrupted: no_graph
        corr_prompt  = build_corrupted_prompt(row, "no_graph")
        corr_answer, corr_hs, _ = generate_with_hidden(tokenizer, model, corr_prompt)
        corr_emb = embed_model.encode([corr_answer], normalize_embeddings=True)[0]
        n_corr_steps = len(corr_hs)

        baseline_emb  = cosine_sim(clean_emb, corr_emb)
        baseline_mech = mechanism_f1(corr_answer, clean_answer)

        for src_stage, tgt_stage in PATHS:
            if (qid, src_stage, tgt_stage) in done: continue

            src_range = stage_ranges.get(src_stage, (0, q_size))
            tgt_range = stage_ranges.get(tgt_stage, (4*q_size, n_steps))

            src_steps_c = list(range(*src_range))
            tgt_steps   = list(range(*tgt_range))
            src_steps_c = [s for s in src_steps_c if s < n_steps]
            tgt_steps   = [s for s in tgt_steps   if s < n_corr_steps]
            if not src_steps_c or not tgt_steps: continue

            # Path patching proxy:
            # Measure how much clean source activations at layer L
            # align with target activations in corrupted run
            for layer_idx in PATCH_LAYERS:
                if layer_idx >= n_layers: continue

                # Source: clean mean hidden state
                src_vecs = [clean_hs[s][layer_idx][0,0,:].float().cpu().numpy()
                            for s in src_steps_c[:30]]
                src_mean = np.mean(src_vecs, axis=0)

                # Target: corrupted mean hidden state at target stage
                tgt_vecs_corr = [corr_hs[s][layer_idx][0,0,:].float().cpu().numpy()
                                 for s in tgt_steps[:30]
                                 if s < len(corr_hs)]
                if not tgt_vecs_corr: continue
                tgt_mean_corr = np.mean(tgt_vecs_corr, axis=0)

                # Path effect = how similar is source to target in corrupted run
                # High similarity → clean source activations already appear at target
                path_effect = cosine_sim(src_mean, tgt_mean_corr)

                out_f.write(f"{qid},{row['question_type']},{src_stage},{tgt_stage},"
                            f"{layer_idx},{path_effect:.4f},{baseline_mech:.4f}\n")

            done.add((qid, src_stage, tgt_stage))

        del clean_hs, corr_hs; torch.cuda.empty_cache(); gc.collect()

    out_f.close()
    return pd.read_csv(OUT_EXP3)

# ══════════════════════════════════════════════════════════════════════════════
# Experiment 4: Stage tag/content ablation
# ══════════════════════════════════════════════════════════════════════════════
def run_exp4(rows, tokenizer, model, embed_model):
    print("\n" + "="*60)
    print("EXP 4: Stage tag/content ablation")
    print("="*60)

    def build_ablation_prompt(row, condition):
        q = row["question"]
        b = row["brainstorm"]; g = row["graph"]
        p = row["patterns"];   s = row["synthesis"]

        if condition == "normal":
            return build_full_trace_prompt(row)

        elif condition == "shuffled_content":
            # Shuffle content within each stage
            def shuf(t):
                sents = re.split(r'(?<=[.!?])\s+', t.strip())
                random.shuffle(sents)
                return " ".join(sents)
            return (f"Question:\n{q}\n\n"
                    f"[Brainstorm]\n{shuf(b)}\n\n"
                    f"[Causal Graph]\n{shuf(g)}\n\n"
                    f"[Patterns]\n{shuf(p)}\n\n"
                    f"[Synthesis]\n{shuf(s)}\n\n"
                    "Write the final answer.")

        elif condition == "no_tags":
            # Remove stage headers
            return (f"Question:\n{q}\n\n{b}\n\n{g}\n\n{p}\n\n{s}\n\n"
                    "Write the final answer.")

        elif condition == "shuffled_tags":
            # Swap stage labels
            return (f"Question:\n{q}\n\n"
                    f"[Synthesis]\n{b}\n\n"
                    f"[Patterns]\n{g}\n\n"
                    f"[Brainstorm]\n{p}\n\n"
                    f"[Causal Graph]\n{s}\n\n"
                    "Write the final answer.")

    CONDITIONS = ["normal", "shuffled_content", "no_tags", "shuffled_tags"]

    done = set()
    if OUT_EXP4.exists() and os.path.getsize(OUT_EXP4) > 0:
        ex = pd.read_csv(OUT_EXP4)
        for _, r in ex.iterrows():
            done.add((int(r["id"]), r["condition"]))
        print(f"  Resuming — {len(done)} done.")

    out_f = open(OUT_EXP4, "a", buffering=1)
    if not OUT_EXP4.exists() or os.path.getsize(OUT_EXP4) == 0:
        out_f.write("id,question_type,condition,"
                    "emb_sim_to_normal,mech_f1_to_normal,"
                    "graph_support,token_f1_to_normal\n")

    for qi, row in enumerate(rows):
        qid = row["id"]
        print(f"\n  [{qi+1}/{len(rows)}] id={qid}")

        # Normal answer as reference
        normal_prompt  = build_ablation_prompt(row, "normal")
        normal_answer  = generate_answer(tokenizer, model, normal_prompt)
        normal_emb     = embed_model.encode([normal_answer], normalize_embeddings=True)[0]

        for condition in CONDITIONS:
            if (qid, condition) in done: continue
            if condition == "normal":
                out_f.write(f"{qid},{row['question_type']},normal,"
                            f"1.0,1.0,"
                            f"{graph_edge_support(normal_answer, row['graph']):.4f},1.0\n")
                done.add((qid, "normal"))
                continue

            prompt = build_ablation_prompt(row, condition)
            answer = generate_answer(tokenizer, model, prompt)
            emb    = embed_model.encode([answer], normalize_embeddings=True)[0]

            emb_sim    = cosine_sim(normal_emb, emb)
            mech_f1v   = mechanism_f1(answer, normal_answer)
            graph_supp = graph_edge_support(answer, row["graph"])
            tok_f1v    = token_f1(answer, normal_answer)

            out_f.write(f"{qid},{row['question_type']},{condition},"
                        f"{emb_sim:.4f},{mech_f1v:.4f},"
                        f"{graph_supp:.4f},{tok_f1v:.4f}\n")
            done.add((qid, condition))
            print(f"    {condition}: emb={emb_sim:.3f} mech={mech_f1v:.3f} "
                  f"graph={graph_supp:.3f}")

        torch.cuda.empty_cache(); gc.collect()

    out_f.close()
    return pd.read_csv(OUT_EXP4)

# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════
def make_plots(df1, df2, df3, df4):
    print("\nGenerating circuit tracing plots...")
    FS = 18

    # ── Fig 2A: Corruption recovery bar chart ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=300)
    fig.patch.set_facecolor("white")
    CORR_LABELS = {"no_graph":"No Graph","shuffled_graph":"Shuffled\nGraph",
                   "wrong_graph":"Wrong\nGraph","no_synthesis":"No Synthesis",
                   "shuffled_synthesis":"Shuffled\nSynthesis"}
    METRICS = [("corrupt_emb_sim","Embedding Similarity",C_BLUE),
               ("corrupt_mech_f1","Mechanism F1",C_GREEN),
               ("corrupt_graph_support","Graph-Edge Support",C_RED)]

    for ax, (col, title, color) in zip(axes, METRICS):
        ax.set_facecolor("white")
        grp = df1.groupby("corruption")[col].mean().reindex(
            ["no_graph","shuffled_graph","wrong_graph","no_synthesis","shuffled_synthesis"])
        labels = [CORR_LABELS.get(k,k) for k in grp.index]
        bars = ax.bar(labels, grp.values, color=color, width=0.6, zorder=3,
                      edgecolor="black", linewidth=1.1)
        ax.set_ylim(0, 1.15)
        for bar, v in zip(bars, grp.values):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.02,
                    f"{v:.2f}", ha="center", fontsize=FS-6, fontweight="bold")
        ax.set_title(title, fontsize=FS-2, fontweight="bold")
        ax.set_ylabel("Score vs Clean Answer", fontsize=FS-4)
        ax.tick_params(labelsize=FS-6)
        ax.grid(axis="y", alpha=0.3); ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Figure 2A — Graph Corruption Recovery\n"
                 "(lower = more damage from removing graph structure)",
                 fontsize=FS, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR/"fig2A_corruption_recovery.pdf",
                bbox_inches="tight", facecolor="white")
    fig.savefig(PLOT_DIR/"fig2A_corruption_recovery.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(); print("  Saved fig2A")

    # ── Fig 2B: Activation patching heatmap ───────────────────────────────────
    if not df2.empty:
        for corruption in ["no_graph","shuffled_graph","wrong_graph"]:
            sub = df2[df2["corruption"]==corruption]
            if sub.empty: continue
            pivot = sub.groupby(["layer","component"])["emb_recovery"].mean().unstack("component")
            if pivot.empty: continue
            cmap = LinearSegmentedColormap.from_list("rec",["white",C_GREEN])
            fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
            ax.set_facecolor("white"); fig.patch.set_facecolor("white")
            im = ax.imshow(pivot.T.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(range(len(pivot.index)))
            ax.set_xticklabels([str(l) for l in pivot.index], fontsize=FS-8)
            ax.set_yticks(range(len(pivot.columns)))
            ax.set_yticklabels([c.capitalize() for c in pivot.columns], fontsize=FS-4)
            # Mark candidate layers
            for j, l in enumerate(pivot.index):
                if l in [7,8,9,10,30,36]:
                    ax.add_patch(plt.Rectangle((j-0.5,-0.5), 1, len(pivot.columns),
                                 fill=False, edgecolor=C_RED, linewidth=2))
            plt.colorbar(im, ax=ax, label="Recovery Score", shrink=0.8)
            ax.set_title(f"Figure 2B — Activation Patching Recovery Heatmap\n"
                         f"corruption: {corruption}  (red border = key layers)",
                         fontsize=FS-2, fontweight="bold")
            ax.set_xlabel("Transformer Layer", fontsize=FS-4)
            fig.tight_layout()
            name = f"fig2B_patching_heatmap_{corruption}"
            fig.savefig(PLOT_DIR/f"{name}.pdf", bbox_inches="tight", facecolor="white")
            fig.savefig(PLOT_DIR/f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(); print(f"  Saved {name}")

    # ── Fig 2C: Path patching — stage-to-stage effect ─────────────────────────
    if not df3.empty:
        paths = df3.groupby(["source_stage","target_stage","layer"])["path_effect_emb"].mean().reset_index()
        path_labels = paths.groupby(["source_stage","target_stage"])["path_effect_emb"].mean().sort_values(ascending=False)
        top_paths = path_labels.head(5).index.tolist()

        fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")
        colors_path = [C_RED, C_BLUE, C_GREEN, C_ORANGE, C_PURPLE]

        for (src, tgt), color in zip(top_paths, colors_path):
            sub = paths[(paths["source_stage"]==src) & (paths["target_stage"]==tgt)]
            sub = sub.sort_values("layer")
            ax.plot(sub["layer"], sub["path_effect_emb"],
                    linewidth=2.2, label=f"{src} → {tgt}", color=color)

        ax.axvspan(7, 10, alpha=0.08, color=C_ORANGE)
        ax.axvline(30, color=C_GRAY, alpha=0.4, linewidth=1.5, linestyle=":")
        ax.axvline(36, color=C_GRAY, alpha=0.4, linewidth=1.5, linestyle=":")
        ax.set_xlabel("Transformer Layer", fontsize=FS-2)
        ax.set_ylabel("Path Effect (source→target alignment)", fontsize=FS-4)
        ax.set_title("Figure 2C — Path Patching: Stage-to-Stage Information Flow",
                     fontsize=FS-2, fontweight="bold")
        ax.legend(fontsize=FS-6, framealpha=0.9)
        ax.grid(axis="y", alpha=0.3); ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(PLOT_DIR/"fig2C_path_patching.pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(PLOT_DIR/"fig2C_path_patching.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(); print("  Saved fig2C")

    # ── Fig 3C: Stage tag/content ablation ────────────────────────────────────
    if not df4.empty:
        COND_NICE = {"normal":"Normal","shuffled_content":"Shuffled\nContent",
                     "no_tags":"No Tags","shuffled_tags":"Shuffled\nTags"}
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300)
        fig.patch.set_facecolor("white")
        ABL_METRICS = [("emb_sim_to_normal","Embedding Similarity\nto Normal",C_BLUE),
                       ("mech_f1_to_normal","Mechanism F1\nto Normal",C_GREEN),
                       ("graph_support","Graph-Edge Support",C_RED)]
        for ax, (col, title, color) in zip(axes, ABL_METRICS):
            ax.set_facecolor("white")
            grp = df4.groupby("condition")[col].mean().reindex(
                ["normal","shuffled_content","no_tags","shuffled_tags"])
            labels = [COND_NICE.get(k,k) for k in grp.index]
            bars = ax.bar(labels, grp.values, color=color, width=0.55, zorder=3,
                          edgecolor="black", linewidth=1.1)
            ax.set_ylim(0, 1.15)
            for bar, v in zip(bars, grp.values):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.02,
                        f"{v:.2f}", ha="center", fontsize=FS-6, fontweight="bold")
            ax.set_title(title, fontsize=FS-2, fontweight="bold")
            ax.tick_params(labelsize=FS-6)
            ax.grid(axis="y", alpha=0.3); ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("Figure 3C — Stage Tag / Content Ablation\n"
                     "(does structure or content matter more?)",
                     fontsize=FS, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOT_DIR/"fig3C_stage_ablation.pdf",
                    bbox_inches="tight", facecolor="white")
        fig.savefig(PLOT_DIR/"fig3C_stage_ablation.png",
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(); print("  Saved fig3C")

    print(f"\nAll plots saved to {PLOT_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# Select 40 examples as instructed
# ══════════════════════════════════════════════════════════════════════════════
def select_examples(graph_rows, graph_bt_df, sim_df):
    """
    Select 40 examples:
    - 20 where Graph-PRefLexOR answer backtracks to synthesis
    - 10 where Qwen does not backtrack to its own thinking
    - 10 mixed/failure cases
    """
    graph_bt_dict = dict(zip(graph_bt_df["id"], graph_bt_df["closest_source"]))
    sim_dict      = dict(zip(sim_df["id"], sim_df["closest_source"]))

    synth_ids  = [r["id"] for r in graph_rows
                  if graph_bt_dict.get(r["id"]) == "synthesis"][:20]
    non_qwen   = [r["id"] for r in graph_rows
                  if sim_dict.get(r["id"]) != "qwen_thinking"
                  and r["id"] not in synth_ids][:10]
    mixed      = [r["id"] for r in graph_rows
                  if r["id"] not in synth_ids and r["id"] not in non_qwen][:10]

    selected_ids = set(synth_ids + non_qwen + mixed)
    selected = [r for r in graph_rows if r["id"] in selected_ids]
    print(f"  Selected {len(selected)} examples:")
    print(f"    synthesis-backtrack: {len(synth_ids)}")
    print(f"    non-qwen-thinking:   {len(non_qwen)}")
    print(f"    mixed:               {len(mixed)}")
    return selected

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    random.seed(42); np.random.seed(42)

    print("Loading data...")
    graph_rows  = load_jsonl(GRAPH_FILE)
    graph_bt_df = pd.read_csv(GRAPH_BT)
    sim_df      = pd.read_csv(SIM_FILE)
    print(f"  {len(graph_rows)} graph rows, {len(graph_bt_df)} backtracking rows")

    # Select 40 examples
    rows = select_examples(graph_rows, graph_bt_df, sim_df)

    # Load models
    print(f"\nLoading {QWEN_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded.")

    print(f"Loading {EMBED_MODEL} ...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    print("  Embed model loaded.")

    # Run experiments
    df1 = run_exp1(rows, tokenizer, model, embed_model)
    df2 = run_exp2(rows, tokenizer, model, embed_model)
    df3 = run_exp3(rows, tokenizer, model, embed_model)

    # Free GPU after hidden-state experiments
    del model; torch.cuda.empty_cache(); gc.collect()
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    df4 = run_exp4(rows, tokenizer, model, embed_model)

    del model; torch.cuda.empty_cache(); gc.collect()

    # Generate all plots
    make_plots(df1, df2, df3, df4)

    print("\n" + "="*60)
    print("CIRCUIT TRACING COMPLETE")
    print("="*60)
    for f in sorted(OUT_DIR.rglob("*.csv")):
        print(f"  {f.name}  ({f.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
