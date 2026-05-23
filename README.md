# Graph-PRefLexOR: Graph-Structured Reinforcement Learning for Scientific Hypothesis Generation

**Paper:** *Graph-Structured Reinforcement Learning for Scientific Hypothesis Generation in Materials Design*  
**Authors:** Subhadeep Pal, Markus J. Buehler  
**Institution:** Department of Civil and Environmental Engineering, MIT  
**Contact:** spu8516@mit.edu, mbuehler@mit.edu

---

## Overview

Graph-PRefLexOR is a family of compact LLMs (1.7B–8B parameters) fine-tuned with **Group Relative Policy Optimization (GRPO)** to perform structured, multi-stage scientific reasoning. The models decompose open-ended materials design problems into explicit reasoning phases:

```
<think>
  <brainstorm>  divergent hypothesis exploration       </brainstorm>
  <graph>       text-form entity/relation map          </graph>
  <graph_json>  machine-readable knowledge graph       </graph_json>
  <patterns>    higher-order causal patterns           </patterns>
  <synthesis>   integrated, testable hypothesis        </synthesis>
</think>
[final answer]
```

Evaluated on a manually curated benchmark of **100 open-ended questions** from materials science and mechanics literature, Graph-PRefLexOR achieves **40–65% improvements** in reasoning quality over base models, with the largest gains in *Reasoning Traceability*.

## Model Weights (HuggingFace)

| Model | Scale | Base Model | HuggingFace ID |
|---|---|---|---|
| Graph-PRefLexOR-8B  | 8B   | Qwen3-8B             | `lamm-mit/Graph-Preflexor-8b_12292025`  |
| Graph-PRefLexOR-3B  | 3B   | Llama-3.2-3B-Instruct| `lamm-mit/Graph-Preflexor-3b_08012026`  |
| Graph-PRefLexOR-1.7B| 1.7B | Qwen3-1.7B           | `lamm-mit/Graph-Preflexor-1.7b_08012026`|

---

## Repository Structure

```
.
├── notebooks/                     # Jupyter notebooks (numbered in pipeline order)
│   ├── 01_paper_ingestion.ipynb   # PDF → Markdown → metadata extraction → question generation
│   ├── 02_dataset_construction.ipynb  # Graph-PRefLexOR training dataset generation
│   ├── 03_claude_evaluation.ipynb     # Claude-as-judge scoring → Figure 1
│   ├── 04_semantic_analysis.ipynb     # PCA, trajectories, diversity → Figures 2–4
│   └── 05_graph_analysis.ipynb        # Graph structure analysis (supplementary)
│
├── scripts/                       # Model inference scripts (run via vLLM)
│   ├── eval_graph_8b.py           # Graph-PRefLexOR-8B inference
│   ├── eval_graph_3b.py           # Graph-PRefLexOR-3B inference
│   ├── eval_graph_1_7b.py         # Graph-PRefLexOR-1.7B inference
│   ├── eval_qwen_8b.py            # Qwen3-8B inference (thinking enabled)
│   ├── eval_qwen_8b_no_thinking.py    # Qwen3-8B (thinking disabled, ablation)
│   ├── eval_qwen_1_7b.py          # Qwen3-1.7B inference
│   ├── eval_qwen_1_7b_no_thinking.py  # Qwen3-1.7B (thinking disabled, ablation)
│   └── eval_llama_3b.py           # Llama-3.2-3B-Instruct inference
│
├── data/
│   ├── benchmark/                 # The 100-question evaluation benchmark
│   │   ├── benchmark_questions.json   # Questions (id, context, question, category, doi)
│   │   ├── benchmark_questions.csv
│   │   └── benchmark_answers.json     # With ground-truth open-ended answers
│   ├── corpus/                    # Paper corpus used for benchmark construction
│   │   ├── consolidated_papers.json   # Merged metadata for all 100 source papers
│   │   ├── paper_text/                # Extracted text from each source paper
│   │   └── marker_runs/               # Per-paper Markdown + JSON (from Marker)
│   ├── results/                   # Model inference outputs and evaluation scores
│   │   ├── graph_8b_data_eval_all.jsonl   # Graph-PRefLexOR-8B inference results
│   │   ├── graph_8b_res_eval.jsonl        # Claude evaluation scores for above
│   │   ├── qwen_8b_data_eval_all.jsonl    # Qwen3-8B inference results
│   │   ├── qwen_8b_res_eval.jsonl         # Evaluation scores
│   │   └── ...                            # (analogous files for 3B and 1.7B variants)
│   └── training/                  # GRPO training dataset
│       ├── graph_reasoning_v3.jsonl   # Graph-native (question, chosen, rejected) triples
│       └── graph_reasoning_v3.csv
│
└── figures/                       # Publication figures and supplementary analysis plots
    ├── qna_reasoning_comparison.png    # Figure 1 — bar chart evaluation
    ├── reasoning_comparison.png
    ├── analysis1_complexity_vs_scores.png
    ├── analysis2_relation_distribution.png
    └── ...
```

---

## Reproducing the Results

### Prerequisites

```bash
pip install openai anthropic transformers datasets sentence-transformers \
            marker-pdf python-dotenv pandas matplotlib seaborn networkx \
            scikit-learn scipy tqdm pydantic vllm
```

Create a `.env` file at the repo root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Pipeline (in order)

#### Step 1 — Paper Ingestion (optional — corpus already provided)

Run **`notebooks/01_paper_ingestion.ipynb`** to convert your own PDF papers to structured metadata and generate benchmark questions.  
The pre-built benchmark is already in `data/benchmark/`.

#### Step 2 — Training Dataset Generation (optional)

Run **`notebooks/02_dataset_construction.ipynb`** to generate additional (question, chosen, rejected) training triples from the paper corpus.  
The pre-built dataset is in `data/training/graph_reasoning_v3.jsonl`.

#### Step 3 — Model Inference

Start a vLLM server for each model and run the corresponding script:

```bash
# Start vLLM server
vllm serve lamm-mit/Graph-Preflexor-8b_12292025 \
     --max-model-len 32768 --port 8000

# Run inference
python scripts/eval_graph_8b.py \
    --dataset data/benchmark/benchmark_questions.json \
    --outfile data/results/graph_8b_results.jsonl
```

Pre-computed inference outputs are in `data/results/`.

#### Step 4 — Claude-as-Judge Evaluation → Figure 1

Run **`notebooks/03_claude_evaluation.ipynb`**.  
Pre-computed scores are in `data/results/*_res_eval.jsonl`.

#### Step 5 — Semantic Analysis → Figures 2–4

Run **`notebooks/04_semantic_analysis.ipynb`**.  
Requires GPU for embedding computation (~15 min per model with `google/embeddinggemma-300m`).

#### Step 6 — Graph Structure Analysis (Supplementary)

Run **`notebooks/05_graph_analysis.ipynb`**.

---

## Key Results

| Model | Reasoning Quality | Intellectual Depth | Traceability | Overall |
|---|---|---|---|---|
| Graph-PRefLexOR-8B  | **5.91** | **5.85** | **6.91** | **6.22** |
| Qwen3-8B            | 4.18     | 4.58     | 4.00     | 4.25     |
| Qwen3-8B (no-think) | 3.56     | 3.58     | 2.85     | 3.33     |
| Graph-PRefLexOR-3B  | **4.25** | **3.94** | **4.78** | **4.32** |
| Llama-3.2-3B-Instruct | 2.83  | 2.80     | 2.49     | 2.71     |
| Graph-PRefLexOR-1.7B | **4.69** | **4.46** | **5.37** | **4.84** |
| Qwen3-1.7B          | 3.20     | 3.30     | 3.27     | 3.25     |
| Qwen3-1.7B (no-think)| 3.04   | 3.00     | 2.73     | 2.92     |

*Scored 0–10 by Claude Opus-4.7 on N=100 open-ended materials science questions.*

**Semantic diversity gains (Table 1):**

| Output Type | Scale | Graph-PRefLexOR | Base Model | Gain |
|---|---|---|---|---|
| Reasoning trace | 1.7B | 0.20 | 0.07 | 2.9× |
| Reasoning trace | 8B   | 0.21 | 0.08 | 2.6× |
| Final answer    | 8B   | 0.43 | 0.18 | 2.4× |

---

## Benchmark Description

The benchmark contains **100 open-ended questions** across five reasoning categories:

| Category | Count | Description |
|---|---|---|
| `causal_multiscale_reasoning`     | ~20 | Multi-scale cause-effect chains |
| `tradeoff_and_non_monotonicity`   | ~20 | Competing objectives |
| `hidden_variable_identification`  | ~20 | Latent confounders |
| `model_abstraction_and_breakdown` | ~20 | Model failure modes |
| `cross_domain_mapping`            | ~20 | Cross-field mechanism transfer |

Source papers span: large language models, spider silk, polymer nanocomposites, epoxy networks, and collagen-based protein materials.

---

## Citation

```bibtex
@article{pal2026graphpreflexor,
  title   = {Graph-Structured Reinforcement Learning for Scientific Hypothesis Generation in Materials Design},
  author  = {Pal, Subhadeep and Buehler, Markus J.},
  year    = {2026},
  institution = {Massachusetts Institute of Technology}
}
```

---

## Acknowledgments

This work was supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research and Office of Basic Energy Sciences, Scientific Discovery through Advanced Computing (SciDAC) program under the FORUM-AI project.
