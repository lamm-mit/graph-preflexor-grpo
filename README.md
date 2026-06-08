# Graph-PRefLexOR: Traceable Scientific Hypothesis Generation via Graph-Structured Reinforcement Learning

**Paper:** *Graph-PRefLexOR: Traceable Scientific Hypothesis Generation via Graph-Structured Reinforcement Learning*  
**Authors:** Subhadeep Pal, Shashwat Sourav, Tirthankar Ghosal, Markus J. Buehler  
**Institutions:** Massachusetts Institute of Technology; Washington University in St. Louis; Oak Ridge National Laboratory  
**Contact:** spu8516@mit.edu, s.shashwat@wustl.edu, ghosalt@ornl.gov, mbuehler@mit.edu

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
│   ├── 01_paper_ingestion.ipynb     # PDF → Markdown → metadata extraction → question generation
│   ├── 02_reasoning_evaluation.ipynb  # Claude-as-judge scoring → Figure 1
│   └── 03_semantic_analysis.ipynb     # PCA, trajectories, diversity → Figures 5–7
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
└── data/
    ├── benchmark/                 # The 100-question evaluation benchmark
    │   └── benchmark_questions.jsonl  # Questions (id, context, question, category, doi)
    ├── corpus/                    # Source-paper index for benchmark construction
    │   └── papers.xlsx                # Title + DOI of each source paper (no full text redistributed)
    └── results/                   # Model inference outputs (with embedded Claude scores)
        ├── graph_8b_data_eval_all.jsonl   # Graph-PRefLexOR-8B inference + evaluation results
        ├── graph_3b_data_eval_all.jsonl   # Graph-PRefLexOR-3B inference + evaluation results
        ├── graph_1_7b_data_eval_all.jsonl # Graph-PRefLexOR-1.7B inference + evaluation results
        ├── qwen_8b_data_eval_all.jsonl    # Qwen3-8B inference + evaluation results
        ├── qwen_8b_no_thinking_data_eval_all.jsonl
        ├── qwen_1_7b_data_eval_all.jsonl  # Qwen3-1.7B inference + evaluation results
        ├── qwen_1_7b_no_thinking_data_eval_all.jsonl
        ├── llama_3b_data_eval_all.jsonl   # Llama-3.2-3B inference + evaluation results
        └── embeddings_cache/              # Pre-computed sentence embeddings (.npy)
```

Figures are regenerated from the analysis sources when run (they are not vendored in this repo):

| Figure(s) | Source |
|---|---|
| Fig 1 (reasoning evaluation) | [notebooks/02_reasoning_evaluation.ipynb](notebooks/02_reasoning_evaluation.ipynb) |
| Figs 5–7 (PCA, trajectories, semantic diversity) | [notebooks/03_semantic_analysis.ipynb](notebooks/03_semantic_analysis.ipynb) |
| Figs 8–11, 13 (backtracking, layer-wise divergence, probe) | [experiments/exp2_backtracking/exp2_publication_plots.ipynb](experiments/exp2_backtracking/exp2_publication_plots.ipynb) (equivalently `exp2_publication_plots.py`) |
| Fig 12 (benchmark workflow) | [notebooks/01_paper_ingestion.ipynb](notebooks/01_paper_ingestion.ipynb) (schematic) |

Figs 2–4 are qualitative examples (a benchmark question and an extracted reasoning trace / knowledge graph) rather than computed plots.

---

## Reproducing the Results

### Prerequisites

```bash
pip install openai anthropic transformers datasets sentence-transformers \
            marker-pdf python-dotenv pandas matplotlib seaborn networkx \
            scikit-learn scipy tqdm pydantic nltk vllm
```

Create a `.env` file at the repo root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Pipeline (in order)

#### Step 1 — Paper Ingestion (optional)

Run **`notebooks/01_paper_ingestion.ipynb`** to convert your own PDF papers to structured metadata and generate benchmark questions.  
Source papers used for the published benchmark are listed by title and DOI in [data/corpus/papers.xlsx](data/corpus/papers.xlsx); paper PDFs and extracted text are not redistributed. The pre-built benchmark is already in `data/benchmark/`.

#### Step 2 — Model Inference

Start a vLLM server for each model and run the corresponding script:

```bash
# Start vLLM server
vllm serve lamm-mit/Graph-Preflexor-8b_12292025 \
     --max-model-len 32768 --port 8000

# Run inference (output written to data/results/ by default)
python scripts/eval_graph_8b.py
```

Pre-computed inference outputs are in `data/results/`.

#### Step 3 — Claude-as-Judge Evaluation → Figure 1

Run **`notebooks/02_reasoning_evaluation.ipynb`**.  
Pre-computed inference outputs (with embedded Claude scores) are in `data/results/*_data_eval_all.jsonl`.

#### Step 4 — Semantic Analysis → Figures 5–7

Run **`notebooks/03_semantic_analysis.ipynb`**.  
Requires GPU for embedding computation (~15 min per model with `google/embeddinggemma-300m`).

---

## Key Results

| Model | Reasoning Quality | Intellectual Depth | Traceability | Overall |
|---|---|---|---|---|
| Graph-PRefLexOR-8B  | **5.91** | **5.85** | **6.91** | **6.22** |
| Qwen3-8B            | 4.18     | 4.58     | 4.00     | 4.25     |
| Qwen3-8B (no-think) | 3.56     | 3.58     | 2.85     | 3.33     |
| Graph-PRefLexOR-3B  | **4.25** | **3.94** | **4.78** | **4.32** |
| Llama-3.2-3B-Instruct | 2.85  | 2.80     | 2.49     | 2.71     |
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
  title   = {Graph-PRefLexOR: Traceable Scientific Hypothesis Generation via Graph-Structured Reinforcement Learning},
  author  = {Pal, Subhadeep and Sourav, Shashwat and Ghosal, Tirthankar and Buehler, Markus J.},
  year    = {2026},
  institution = {Massachusetts Institute of Technology}
}
```

---

---

## Experiment 2: Backtracking Analysis

This analysis asks: *When Qwen3-8B produces a final answer, does it backtrack more to its own visible thinking, or to Graph-PRefLexOR's structured reasoning stages?*

### Key Results (n=100 questions)

| Finding | Result |
|---|---|
| Qwen answer closest to Graph-PRefLexOR answer | 46/100 cases |
| Qwen answer closest to its own thinking | 16/100 cases |
| Thinking–answer divergence peak | Layers 7–10 and layer 36 |
| Linear probe AUROC (thinking vs answer) | 1.0 from layer 1 onward |
| Graph-PRefLexOR answer closest to synthesis stage | Most common internal backtrack |

### Structure
### Scripts (run in order)

| Step | Script | Description |
|---|---|---|
| 0 | `step0_generate_graph_stages.py` | Run Graph-PRefLexOR pipeline on 100 questions |
| 1 | `step1_full_pipeline.py` | 2A–2G: Qwen thinking → similarities → hidden states → probes → logit lens |
| 2 | `step2_activation_patching.py` | Representation-space activation patching |
| 3 | `step3_mentor_plots.py` | Backtracking + layer-wise divergence plots |
| 4 | `step4_publication_plots.py` | All plots in publication style (PDF + PNG) |

### Models Used
- Generator: `Qwen/Qwen3-8B`
- Embeddings: `BAAI/bge-base-en-v1.5`
- Platform: NCSA DeltaAI GH200 GPUs
- Device: NVIDIA DGX Spark


## Acknowledgments

This work was supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research and Office of Basic Energy Sciences, Scientific Discovery through Advanced Computing (SciDAC) program under the FORUM-AI project.
