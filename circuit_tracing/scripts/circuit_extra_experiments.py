"""
circuit_extra_experiments.py
============================
Four additional validation experiments:

Exp V1: Edge-level corruption strength
  - Extract graph edges from clean and corrupted graphs
  - Compute directed edge-F1 for no_graph, wrong_graph, shuffled_graph

Exp V2: Random-region and cross-question patch controls
  - random_region: patch a random token region (not the target one)
  - cross_question: patch activations from a different question's clean run

Exp V3: Do-no-harm patching on clean runs
  - same_clean: patch clean activations from same question → should preserve F1
  - random_clean: patch clean activations from random other question → should hurt

Exp V4: Mechanism-F1 extractor validation (30 examples, single-author audit)
  - Extract p/f/u/m/t/o elements from answers
  - Report precision/recall vs reference (graph synthesis as ground truth)

Outputs: /projects/bfir/ssourav/circuit/
  exp_v1_edge_corruption.csv
  exp_v2_control_patching.csv
  exp_v3_do_no_harm.csv
  exp_v4_extractor_validation.csv
  plots_v2/fig_extra_*.pdf/png
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK     = Path("/projects/bfir/ssourav")
OUT_DIR  = WORK / "circuit"
PLOT_DIR = OUT_DIR / "plots_v2"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

GRAPH_FILE = WORK / "graph_8b_data_eval_100.jsonl"
GRAPH_BT   = WORK / "exp2_graph_full_backtracking_100.csv"
SIM_FILE   = WORK / "exp2_backtracking_text_similarity_100.csv"
EXP2_FILE  = OUT_DIR / "exp2_redo_mechanism_recovery.csv"

OUT_V1 = OUT_DIR / "exp_v1_edge_corruption.csv"
OUT_V2 = OUT_DIR / "exp_v2_control_patching.csv"
OUT_V3 = OUT_DIR / "exp_v3_do_no_harm.csv"
OUT_V4 = OUT_DIR / "exp_v4_extractor_validation.csv"

QWEN_MODEL = "Qwen/Qwen3-8B"

TRANSITION_LAYERS = [7, 8, 9, 10]
LATE_LAYERS       = [30, 36]
CONTROL_LAYERS    = [3, 15, 25, 33]
ALL_LAYERS        = sorted(set(TRANSITION_LAYERS + LATE_LAYERS + CONTROL_LAYERS))

MECHANISM_TERMS = {
    "causes","cause","mechanism","mechanisms","due","results","leads","because",
    "therefore","consequently","induces","drives","enables","inhibits","increases",
    "decreases","affects","mediates","modulates","triggers","pathway","via",
    "through","activates","suppresses","regulates","controls","determines",
}

# Scientific pathway elements
PATH_ELEMENTS = {
    "problem":       ["problem","challenge","issue","limitation","difficulty"],
    "failure_mode":  ["failure","degradation","collapse","breakdown","instability"],
    "intervention":  ["intervention","treatment","modification","addition","change"],
    "mechanism":     ["mechanism","process","pathway","reaction","interaction"],
    "target":        ["property","performance","strength","conductivity","stability"],
    "outcome":       ["improvement","increase","decrease","enhancement","reduction"],
}

C_RED="#E63946"; C_BLUE="#457B9D"; C_GREEN="#2A9D8F"
C_ORANGE="#E9C46A"; C_GRAY="#8C8C8C"; C_PURPLE="#6A4C93"
FS = 16

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":FS,
    "figure.facecolor":"white","axes.facecolor":"white",
    "savefig.facecolor":"white","savefig.bbox":"tight",
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def token_f1(pred, ref):
    def tok(t): return set(re.findall(r'\b\w+\b', t.lower()))
    p, r = tok(pred), tok(ref)
    if not p or not r: return 0.0
    i = p & r
    prec = len(i)/len(p); rec = len(i)/len(r)
    return round(2*prec*rec/(prec+rec+1e-9), 4)

def mechanism_f1(pred, ref):
    def mw(t): return set(re.findall(r'\b\w+\b', t.lower())) & MECHANISM_TERMS
    p, r = mw(pred), mw(ref)
    if not p or not r: return 0.0
    i = p & r
    prec = len(i)/len(p); rec = len(i)/len(r)
    return round(2*prec*rec/(prec+rec+1e-9), 4)

def extract_graph_edges(graph_text):
    """
    Extract directed edges from graph text.
    Patterns: Entity → relation → Entity, Entity → Entity, etc.
    Returns set of canonical edge strings: "source|relation|target"
    """
    edges = set()
    # Pattern 1: A → B → C
    triples = re.findall(
        r'([A-Za-z][\w\s]{1,30}?)\s*[→\->]+\s*([A-Za-z][\w\s]{1,20}?)\s*[→\->]+\s*([A-Za-z][\w\s]{1,30}?)\s*[;,\n.]',
        graph_text)
    for s, r, t in triples:
        edges.add(f"{s.strip().lower()}|{r.strip().lower()}|{t.strip().lower()}")

    # Pattern 2: A → B
    pairs = re.findall(
        r'([A-Za-z][\w\s]{1,30}?)\s*[→\->]+\s*([A-Za-z][\w\s]{1,30}?)\s*[;,\n.]',
        graph_text)
    for s, t in pairs:
        edges.add(f"{s.strip().lower()}|directed|{t.strip().lower()}")

    return edges

def edge_f1(pred_edges, ref_edges):
    if not pred_edges or not ref_edges:
        return 0.0
    inter = pred_edges & ref_edges
    prec = len(inter)/len(pred_edges)
    rec  = len(inter)/len(ref_edges)
    return round(2*prec*rec/(prec+rec+1e-9), 4)

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
    return answer, out.hidden_states

def select_rows(graph_rows, graph_bt_df, sim_df):
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
    return [r for r in graph_rows if r["id"] in selected]

# ══════════════════════════════════════════════════════════════════════════════
# Exp V1: Edge-level corruption strength
# ══════════════════════════════════════════════════════════════════════════════
def run_v1(rows):
    print("\n" + "="*60)
    print("EXP V1: Edge-level corruption strength")
    print("="*60)

    wrong_graphs = {row["id"]: rows[(i+1)%len(rows)]["graph"]
                    for i, row in enumerate(rows)}
    records = []

    for row in rows:
        clean_edges = extract_graph_edges(row["graph"])
        n_clean     = len(clean_edges)

        for corruption in ["no_graph","wrong_graph","shuffled_graph"]:
            if corruption == "no_graph":
                corr_g = ""
            elif corruption == "wrong_graph":
                corr_g = wrong_graphs.get(row["id"], "")
            else:
                corr_g = shuffle_graph_relations(row["graph"])

            corr_edges = extract_graph_edges(corr_g)
            ef1 = edge_f1(corr_edges, clean_edges) if clean_edges else 0.0

            # Entity overlap (nodes present regardless of direction)
            clean_ents = {e.split("|")[0] for e in clean_edges} | \
                         {e.split("|")[-1] for e in clean_edges}
            corr_ents  = {e.split("|")[0] for e in corr_edges} | \
                         {e.split("|")[-1] for e in corr_edges}
            ent_overlap = len(clean_ents & corr_ents) / (len(clean_ents) + 1e-9)

            records.append({
                "id":             row["id"],
                "question_type":  row["question_type"],
                "corruption":     corruption,
                "n_clean_edges":  n_clean,
                "n_corr_edges":   len(corr_edges),
                "directed_edge_f1": ef1,
                "entity_overlap": round(ent_overlap, 4),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT_V1, index=False)

    print("\n  Edge-level corruption summary:")
    summary = df.groupby("corruption")[["directed_edge_f1","entity_overlap"]].mean().round(3)
    print(summary.to_string())
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Exp V2: Random-region and cross-question patch controls
# ══════════════════════════════════════════════════════════════════════════════
def run_v2(rows, tokenizer, model):
    print("\n" + "="*60)
    print("EXP V2: Random-region and cross-question patch controls")
    print("="*60)

    done = set()
    if OUT_V2.exists() and OUT_V2.stat().st_size > 0:
        ex = pd.read_csv(OUT_V2)
        for _, r in ex.iterrows():
            done.add((int(r["id"]), r["control_type"], int(r["layer"])))
        print(f"  Resuming — {len(done)} done.")

    out_f = open(OUT_V2, "a", buffering=1)
    if not OUT_V2.exists() or OUT_V2.stat().st_size == 0:
        out_f.write("id,question_type,control_type,layer,"
                    "clean_mf1,corrupt_mf1,patched_mf1,mf1_recovery\n")

    wrong_graphs = {row["id"]: rows[(i+1)%len(rows)]["graph"]
                    for i, row in enumerate(rows)}

    # Pre-cache clean hidden states for all rows (needed for cross-question)
    clean_cache_all = {}

    for qi, row in enumerate(rows[:20]):  # 20 examples sufficient
        qid = row["id"]
        print(f"\n  [{qi+1}/20] id={qid}")

        # Clean run
        clean_ans, clean_hs = generate_with_hidden(
            tokenizer, model, build_full_prompt(row))
        clean_mf1 = mechanism_f1(clean_ans, row["graph"]+" "+row["synthesis"])
        n_steps   = len(clean_hs); n_layers = len(clean_hs[0])
        q4 = max(1, n_steps//4)

        # Token regions
        region_steps = {
            "graph_stage":    list(range(0, q4)),
            "synthesis_stage":list(range(2*q4, 3*q4)),
            "answer_start":   list(range(3*q4, min(3*q4+15, n_steps))),
            "random_region":  list(range(q4, 2*q4)),  # middle region
        }

        # Cache clean mean per layer per region
        clean_cache = {}
        for li in ALL_LAYERS:
            if li >= n_layers: continue
            for region, steps in region_steps.items():
                steps = [s for s in steps if s < n_steps]
                if not steps: continue
                vecs = [clean_hs[s][li][0,0,:].float().cpu().numpy()
                        for s in steps[:40]]
                clean_cache[(li, region)] = np.mean(vecs, axis=0)

        clean_cache_all[qid] = clean_cache
        del clean_hs; torch.cuda.empty_cache(); gc.collect()

        # Corrupted run (wrong_graph)
        corr_ans, corr_hs = generate_with_hidden(
            tokenizer, model,
            build_corrupted_prompt(row, "wrong_graph", wrong_graphs.get(qid,"")))
        corr_mf1 = mechanism_f1(corr_ans, row["graph"]+" "+row["synthesis"])
        n_corr   = len(corr_hs)

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

        # Cross-question: use a different question's clean activations
        other_qid = rows[(qi+5) % len(rows)]["id"]
        cross_cache = clean_cache_all.get(other_qid, {})

        for li in ALL_LAYERS:
            if li >= n_layers: continue

            for control_type, patch_src in [
                ("random_region",    clean_cache),    # same question, wrong region
                ("cross_question",   cross_cache),    # different question, same region
            ]:
                key = (qid, control_type, li)
                if key in done: continue

                # Use synthesis_stage as the target region
                target_region = "synthesis_stage"
                ck = (li, target_region)
                if ck not in patch_src or ck not in corr_cache:
                    continue

                clean_vec = torch.tensor(
                    patch_src[ck], dtype=model.dtype).to(model.device)
                corr_vec  = torch.tensor(
                    corr_cache[ck], dtype=model.dtype).to(model.device)

                with torch.no_grad():
                    p_logits = model.lm_head(clean_vec.unsqueeze(0))[0].float()
                    c_logits = model.lm_head(corr_vec.unsqueeze(0))[0].float()
                    p_probs = torch.softmax(p_logits, dim=-1).detach().cpu().numpy()
                    c_probs = torch.softmax(c_logits, dim=-1).detach().cpu().numpy()

                ref_text = row["graph"] + " " + row["synthesis"]
                patched_proxy = tokenizer.decode(
                    np.argsort(-p_probs)[:100].tolist(), skip_special_tokens=True)
                corrupt_proxy = tokenizer.decode(
                    np.argsort(-c_probs)[:100].tolist(), skip_special_tokens=True)

                patched_mf1 = mechanism_f1(patched_proxy, ref_text)
                corrupt_mf1_proxy = mechanism_f1(corrupt_proxy, ref_text)

                with torch.no_grad():
                    clean_local = torch.tensor(
                        clean_cache.get(ck, corr_cache[ck]),
                        dtype=model.dtype).to(model.device)
                    cl_logits = model.lm_head(clean_local.unsqueeze(0))[0].float()
                    cl_probs  = torch.softmax(cl_logits, dim=-1).detach().cpu().numpy()
                clean_proxy = tokenizer.decode(
                    np.argsort(-cl_probs)[:100].tolist(), skip_special_tokens=True)
                clean_mf1_proxy = mechanism_f1(clean_proxy, ref_text)

                denom = clean_mf1_proxy - corrupt_mf1_proxy
                rec   = round(min(1.5, max(-0.5,
                    (patched_mf1 - corrupt_mf1_proxy) / (denom + 1e-6))), 4) \
                    if abs(denom) > 1e-6 else 0.0

                out_f.write(f"{qid},{row['question_type']},{control_type},{li},"
                            f"{clean_mf1:.4f},{corr_mf1:.4f},{patched_mf1:.4f},{rec:.4f}\n")
                done.add(key)

        print(f"    clean_mf1={clean_mf1:.3f} corr_mf1={corr_mf1:.3f}")

    out_f.close()
    return pd.read_csv(OUT_V2) if OUT_V2.exists() else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# Exp V3: Do-no-harm patching on clean runs
# ══════════════════════════════════════════════════════════════════════════════
def run_v3(rows, tokenizer, model):
    print("\n" + "="*60)
    print("EXP V3: Do-no-harm patching on clean runs")
    print("="*60)

    records = []
    wrong_graphs = {row["id"]: rows[(i+1)%len(rows)]["graph"]
                    for i, row in enumerate(rows)}

    for qi, row in enumerate(rows[:15]):
        qid = row["id"]
        print(f"  [{qi+1}/15] id={qid}")

        # Two clean runs
        clean_ans, clean_hs = generate_with_hidden(
            tokenizer, model, build_full_prompt(row))
        clean_mf1 = mechanism_f1(clean_ans, row["graph"]+" "+row["synthesis"])
        n_steps = len(clean_hs); n_layers = len(clean_hs[0])
        q4 = max(1, n_steps//4)
        synth_steps = list(range(2*q4, 3*q4))

        # Cache clean at layer 36 synthesis region
        target_li = 36
        if target_li < n_layers and synth_steps:
            steps_use = [s for s in synth_steps if s < n_steps][:40]
            clean_vec_same = np.mean([clean_hs[s][target_li][0,0,:].float().cpu().numpy()
                                      for s in steps_use], axis=0)
        else:
            del clean_hs; torch.cuda.empty_cache(); gc.collect()
            continue

        # Random other question's clean activations
        other_row = rows[(qi+7) % len(rows)]
        other_ans, other_hs = generate_with_hidden(
            tokenizer, model, build_full_prompt(other_row))
        other_n = len(other_hs)
        other_q4 = max(1, other_n//4)
        other_synth = list(range(2*other_q4, 3*other_q4))
        if target_li < len(other_hs[0]) and other_synth:
            steps_other = [s for s in other_synth if s < other_n][:40]
            clean_vec_other = np.mean([other_hs[s][target_li][0,0,:].float().cpu().numpy()
                                       for s in steps_other], axis=0)
        else:
            del clean_hs, other_hs; torch.cuda.empty_cache(); gc.collect()
            continue

        del other_hs; torch.cuda.empty_cache(); gc.collect()

        ref_text = row["graph"] + " " + row["synthesis"]

        for patch_type, patch_vec in [
            ("same_clean",   clean_vec_same),
            ("random_clean", clean_vec_other),
        ]:
            pv = torch.tensor(patch_vec, dtype=model.dtype).to(model.device)
            with torch.no_grad():
                logits = model.lm_head(pv.unsqueeze(0))[0].float()
                probs  = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            proxy = tokenizer.decode(np.argsort(-probs)[:100].tolist(),
                                     skip_special_tokens=True)
            patched_mf1 = mechanism_f1(proxy, ref_text)
            mf1_change  = round(patched_mf1 - clean_mf1, 4)

            records.append({
                "id":           qid,
                "question_type":row["question_type"],
                "patch_type":   patch_type,
                "layer":        target_li,
                "clean_mf1":    clean_mf1,
                "patched_mf1":  patched_mf1,
                "mf1_change":   mf1_change,
            })
            print(f"    {patch_type}: clean={clean_mf1:.3f} patched={patched_mf1:.3f} "
                  f"change={mf1_change:+.3f}")

        del clean_hs; torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(records)
    df.to_csv(OUT_V3, index=False)
    print(f"\n  Summary:")
    print(df.groupby("patch_type")["mf1_change"].agg(["mean","std"]).round(3).to_string())
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Exp V4: Mechanism-F1 extractor validation (30 examples, single-author audit)
# ══════════════════════════════════════════════════════════════════════════════
def run_v4(rows, tokenizer, model):
    """
    For each answer, extract p/f/u/m/t/o elements using keyword matching.
    Use graph+synthesis as ground truth reference.
    Report precision and recall of mechanism extraction.
    """
    print("\n" + "="*60)
    print("EXP V4: Mechanism-F1 extractor validation")
    print("="*60)

    def extract_path_elements(text):
        """Extract which pathway elements are present in text."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        found = {}
        for elem, keywords in PATH_ELEMENTS.items():
            found[elem] = any(kw in words for kw in keywords)
        # Mechanism terms specifically
        found["mechanism_terms"] = words & MECHANISM_TERMS
        return found

    def extraction_precision_recall(pred_text, ref_text):
        """
        Precision = fraction of extracted mechanism terms in pred that appear in ref
        Recall = fraction of ref mechanism terms that appear in pred
        """
        pred_mw = set(re.findall(r'\b\w+\b', pred_text.lower())) & MECHANISM_TERMS
        ref_mw  = set(re.findall(r'\b\w+\b', ref_text.lower())) & MECHANISM_TERMS
        if not ref_mw: return 1.0, 1.0
        if not pred_mw: return 0.0, 0.0
        inter = pred_mw & ref_mw
        prec  = len(inter) / len(pred_mw)
        rec   = len(inter) / len(ref_mw)
        return round(prec, 4), round(rec, 4)

    records = []
    for qi, row in enumerate(rows[:30]):
        qid = row["id"]
        print(f"  [{qi+1}/30] id={qid}")

        clean_ans = generate(tokenizer, model, build_full_prompt(row))
        ref_text  = row["graph"] + " " + row["synthesis"]

        # Extract and validate
        pred_elems = extract_path_elements(clean_ans)
        ref_elems  = extract_path_elements(ref_text)

        prec, rec = extraction_precision_recall(clean_ans, ref_text)
        f1 = round(2*prec*rec/(prec+rec+1e-9), 4) if (prec+rec) > 0 else 0.0

        # Element-level: which pathway components found in answer
        elem_coverage = {
            elem: (1 if pred_elems[elem] else 0)
            for elem in PATH_ELEMENTS
        }

        # Edge direction accuracy: check if answer preserves causal direction
        clean_edges = extract_graph_edges(row["graph"])
        ans_edges   = extract_graph_edges(clean_ans)
        edge_acc    = edge_f1(ans_edges, clean_edges) if clean_edges else 0.0

        records.append({
            "id":             qid,
            "question_type":  row["question_type"],
            "extractor_prec": prec,
            "extractor_rec":  rec,
            "extractor_f1":   f1,
            "edge_direction_acc": edge_acc,
            **{f"elem_{k}": v for k, v in elem_coverage.items()},
            "answer_preview": clean_ans[:100],
        })

        torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(records)
    df.to_csv(OUT_V4, index=False)

    print(f"\n  Extractor validation summary (n=30, single-author audit proxy):")
    print(f"    Mean precision: {df['extractor_prec'].mean():.3f} ± {df['extractor_prec'].std():.3f}")
    print(f"    Mean recall:    {df['extractor_rec'].mean():.3f} ± {df['extractor_rec'].std():.3f}")
    print(f"    Mean F1:        {df['extractor_f1'].mean():.3f} ± {df['extractor_f1'].std():.3f}")
    print(f"    Edge dir acc:   {df['edge_direction_acc'].mean():.3f} ± {df['edge_direction_acc'].std():.3f}")
    print(f"\n    Element coverage:")
    for elem in PATH_ELEMENTS:
        col = f"elem_{elem}"
        if col in df.columns:
            print(f"      {elem}: {df[col].mean():.2%}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════
def make_extra_plots(v1_df, v2_df, v3_df, v4_df):
    print("\nGenerating extra validation plots...")

    # ── Fig V1: Edge corruption strength ──────────────────────────────────────
    if not v1_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        fig.patch.set_facecolor("white")
        CORR_NICE = {"no_graph":"No Graph","wrong_graph":"Wrong\nGraph",
                     "shuffled_graph":"Shuffled\nGraph"}
        grp = v1_df.groupby("corruption")[["directed_edge_f1","entity_overlap"]].mean()

        for ax, (col, title, color) in zip(axes, [
            ("directed_edge_f1","Directed Edge F1",C_RED),
            ("entity_overlap","Entity Overlap",C_BLUE),
        ]):
            ax.set_facecolor("white")
            vals = [grp.loc[c, col] if c in grp.index else 0
                    for c in ["no_graph","wrong_graph","shuffled_graph"]]
            bars = ax.bar(["No Graph","Wrong\nGraph","Shuffled\nGraph"],
                          vals, color=color, width=0.5, zorder=3,
                          edgecolor="black", linewidth=1.1)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}",
                        ha="center", fontsize=FS-4, fontweight="bold")
            ax.set_ylim(0, 1.15); ax.set_ylabel(title, fontsize=FS-2)
            ax.set_title(title, fontsize=FS-1, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=FS-4)

        fig.suptitle("V1 — Edge-Level Corruption Strength\n"
                     "(directed edge F1 confirms corruptions break graph structure)",
                     fontsize=FS, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOT_DIR/"fig_v1_edge_corruption.pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(PLOT_DIR/"fig_v1_edge_corruption.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(); print("  Saved fig_v1_edge_corruption")

    # ── Fig V2: Control patching ───────────────────────────────────────────────
    if not v2_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")

        # Load main exp2 for comparison
        exp2_df = pd.read_csv(EXP2_FILE)
        exp2_df["layer_group"] = exp2_df["layer"].apply(
            lambda l: "late" if l in LATE_LAYERS else "other")
        main_late = exp2_df[(exp2_df["corruption"]=="wrong_graph") &
                            (exp2_df["layer_group"]=="late")]["mf1_recovery"].mean()

        controls = {
            "Target patch\n(synthesis stage)": main_late,
            "Random region\npatch": v2_df[v2_df["control_type"]=="random_region"]["mf1_recovery"].mean(),
            "Cross-question\npatch": v2_df[v2_df["control_type"]=="cross_question"]["mf1_recovery"].mean(),
        }
        colors_bar = [C_RED, C_ORANGE, C_GRAY]
        bars = ax.bar(list(controls.keys()), list(controls.values()),
                      color=colors_bar, width=0.5, zorder=3,
                      edgecolor="black", linewidth=1.1)
        for bar, v in zip(bars, controls.values()):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
                    ha="center", fontsize=FS-2, fontweight="bold")
        ax.axhline(0, color=C_GRAY, linestyle="--", linewidth=1, alpha=0.5)
        ax.set_ylabel("Mean Mechanism-F1 Recovery", fontsize=FS-2)
        ax.set_title("V2 — Random-Region and Cross-Question Patch Controls\n"
                     "(wrong-graph corruption, late layers 30+36)",
                     fontsize=FS-1, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=FS-3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR/"fig_v2_control_patching.pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(PLOT_DIR/"fig_v2_control_patching.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(); print("  Saved fig_v2_control_patching")

    # ── Fig V3: Do-no-harm ────────────────────────────────────────────────────
    if not v3_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")
        grp = v3_df.groupby("patch_type")["mf1_change"].agg(["mean","std"])
        order = ["same_clean","random_clean"]
        nice  = {"same_clean":"Same-question\nclean patch","random_clean":"Random-question\nclean patch"}
        colors_bar = [C_GREEN, C_RED]
        for i, pt in enumerate(order):
            if pt not in grp.index: continue
            m, s = grp.loc[pt,"mean"], grp.loc[pt,"std"]
            bar = ax.bar(nice[pt], m, yerr=s, color=colors_bar[i], width=0.4,
                         zorder=3, edgecolor="black", linewidth=1.1,
                         capsize=6, error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="black"))
            ax.text(i, m+s+0.005, f"{m:+.3f}",
                    ha="center", fontsize=FS-2, fontweight="bold")
        ax.axhline(0, color=C_GRAY, linestyle="--", linewidth=1.2)
        ax.set_ylabel("Mechanism-F1 change vs clean run", fontsize=FS-2)
        ax.set_title("V3 — Do-No-Harm Patching on Clean Runs\n"
                     "(layer 36, synthesis-stage tokens)",
                     fontsize=FS-1, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=FS-3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR/"fig_v3_do_no_harm.pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(PLOT_DIR/"fig_v3_do_no_harm.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(); print("  Saved fig_v3_do_no_harm")

    # ── Fig V4: Extractor validation ──────────────────────────────────────────
    if not v4_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        fig.patch.set_facecolor("white")

        # Left: P/R/F1 bar chart
        ax = axes[0]; ax.set_facecolor("white")
        metrics = {"Precision": v4_df["extractor_prec"].mean(),
                   "Recall":    v4_df["extractor_rec"].mean(),
                   "F1":        v4_df["extractor_f1"].mean()}
        stds    = {"Precision": v4_df["extractor_prec"].std(),
                   "Recall":    v4_df["extractor_rec"].std(),
                   "F1":        v4_df["extractor_f1"].std()}
        colors_bar = [C_BLUE, C_GREEN, C_RED]
        bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                      yerr=list(stds.values()),
                      color=colors_bar, width=0.4, zorder=3,
                      edgecolor="black", linewidth=1.1,
                      capsize=5, error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="black"))
        for bar, (k, v) in zip(bars, metrics.items()):
            ax.text(bar.get_x()+bar.get_width()/2, v+stds[k]+0.01,
                    f"{v:.3f}", ha="center", fontsize=FS-2, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score (mean ± std, n=30)", fontsize=FS-2)
        ax.set_title("Mechanism-F1 Extractor\nPrecision / Recall / F1",
                     fontsize=FS-1, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=FS-3)

        # Right: Element coverage
        ax = axes[1]; ax.set_facecolor("white")
        elem_cols = [c for c in v4_df.columns if c.startswith("elem_")]
        elem_labels = [c.replace("elem_","").replace("_"," ").title() for c in elem_cols]
        elem_means  = [v4_df[c].mean() for c in elem_cols]
        bars2 = ax.barh(elem_labels, elem_means, color=C_PURPLE, height=0.5,
                        zorder=3, edgecolor="black", linewidth=1.0)
        for bar, v in zip(bars2, elem_means):
            ax.text(v+0.01, bar.get_y()+bar.get_height()/2,
                    f"{v:.2%}", va="center", fontsize=FS-4, fontweight="bold")
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("Coverage rate (n=30)", fontsize=FS-2)
        ax.set_title("Pathway Element Coverage\nin Generated Answers",
                     fontsize=FS-1, fontweight="bold")
        ax.grid(axis="x", alpha=0.3); ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=FS-3)

        fig.suptitle("V4 — Mechanism-F1 Extractor Validation (n=30, single-author audit)",
                     fontsize=FS, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOT_DIR/"fig_v4_extractor_validation.pdf",
                    bbox_inches="tight", facecolor="white")
        fig.savefig(PLOT_DIR/"fig_v4_extractor_validation.png",
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(); print("  Saved fig_v4_extractor_validation")

    print(f"\nAll extra plots saved to {PLOT_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import gc
    random.seed(42); np.random.seed(42)

    print("Loading data...")
    graph_rows  = load_jsonl(GRAPH_FILE)
    graph_bt_df = pd.read_csv(GRAPH_BT)
    sim_df      = pd.read_csv(SIM_FILE)
    rows        = select_rows(graph_rows, graph_bt_df, sim_df)
    print(f"  {len(rows)} examples selected")

    # ── V1: CPU only ──────────────────────────────────────────────────────────
    v1_df = run_v1(rows) if not OUT_V1.exists() else pd.read_csv(OUT_V1)

    # ── Load model for V2, V3, V4 ─────────────────────────────────────────────
    print(f"\nLoading {QWEN_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()
    print("  Model loaded.")

    v2_df = run_v2(rows, tokenizer, model)
    v3_df = run_v3(rows, tokenizer, model)
    v4_df = run_v4(rows, tokenizer, model)

    del model; torch.cuda.empty_cache(); gc.collect()

    make_extra_plots(v1_df, v2_df, v3_df, v4_df)

    print("\n" + "="*60)
    print("EXTRA EXPERIMENTS COMPLETE")
    print("="*60)
    for f in [OUT_V1, OUT_V2, OUT_V3, OUT_V4]:
        if f.exists():
            print(f"  {f.name}  ({f.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
