---
license: gemma
library_name: transformers
pipeline_tag: text-generation
base_model: google/gemma-4-E4B-it
datasets:
  - lamm-mit/graph-canvas-inpainting-121k
language:
  - en
tags:
  - gemma
  - graph-completion
  - graph-inpainting
  - scientific-reasoning
  - structured-generation
  - grpo
  - vllm
---

# Graph-PRefLexOR-4B-Inpainting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lamm-mit/graph-preflexor-grpo/blob/main/Notebooks/Graph_PRefLexOR_4B_Inpainting_Colab.ipynb)

`lamm-mit/Graph-PRefLexOR-4B-Inpainting` is trained to complete and repair scientific mechanism graphs. Given a natural
language condition and an empty, incomplete, or corrupted graph canvas, the
model returns the complete corrected graph as structured JSON.

## Links

- Model: <https://huggingface.co/lamm-mit/Graph-PRefLexOR-4B-Inpainting>
- Dataset: <https://huggingface.co/datasets/lamm-mit/graph-canvas-inpainting-121k>
- Code: <https://github.com/lamm-mit/graph-preflexor-grpo>
- Standalone six-mode Colab:
  <https://colab.research.google.com/github/lamm-mit/graph-preflexor-grpo/blob/main/Notebooks/Graph_PRefLexOR_4B_Inpainting_Colab.ipynb>
- Base model: <https://huggingface.co/google/gemma-4-E4B-it>
- Detailed training and evaluation guide:
  [`GRAPH_COMPLETION_GRPO_README.md`](https://github.com/lamm-mit/graph-preflexor-grpo/blob/main/GRAPH_COMPLETION_GRPO_README.md)

## What the model was trained to do

The training data contains six graph-inpainting and repair modes:

| Mode | Input canvas | Intended operation |
|---|---|---|
| `prior_empty` | No nodes or edges | Construct a complete graph from the condition |
| `fixed_nodes_only` | A fixed subset of nodes and no edges | Preserve supplied nodes and add the missing graph |
| `missing_edges` | Complete nodes and an incomplete edge set | Add missing edges while preserving supplied content |
| `partial_subgraph` | A correct but incomplete subgraph | Add missing nodes and edges |
| `wrong_relations` | Correct nodes/endpoints with corrupted relation labels | Repair incorrect relations |
| `extra_edges` | Correct graph plus spurious edges | Remove unsupported edges |

For additive completion, supplied objects should normally be marked fixed.
For repair or removal, the supplied graph must remain editable.

## Output contract

The model may use Gemma's native thinking channel. The scored final response is
the last complete answer block:

```text
<answer>
{
  "nodes": [...],
  "edges": [...]
}
</answer>
```

The graph object must contain:

- `nodes`: objects with at least an `id`;
- `edges`: objects with `source`, `relation`, and `target`;
- no duplicate nodes or edges;
- no edges whose endpoints are absent from `nodes`.

The model returns the complete graph, not a patch or edit list. Any optional
node or edge payload fields supplied in the canvas are part of the object
contract and should be preserved.

## Checkpoint-selection results

The following values were measured with deterministic decoding on the fixed
512-task held-out validation selection used for checkpoint comparison.
Metrics are source-macro means. These are validation results, not official-test
results.

| Model | Shaped reward | Valid completion | Exact graph | Node score | Edge score |
|---|---:|---:|---:|---:|---:|
| `google/gemma-4-E4B-it` | 0.3871 | 0.6436 | 0.0597 | 0.5871 | 0.3516 |
| `Graph-PRefLexOR-4B-Inpainting` | **0.5460** | **0.9869** | **0.1694** | **0.7124** | **0.4115** |

Use the reproducible official-test workflow below to generate test results for
the installed model and dataset revisions.

## Installation

The provided CLI uses vLLM for generation and the repository's deterministic
Python evaluator for scoring.

### 1. Clone the repository

```bash
git clone https://github.com/lamm-mit/graph-preflexor-grpo.git
cd graph-preflexor-grpo
```

### 2. Create an environment

The following is a practical GPU inference environment. Install the PyTorch
build appropriate for the CUDA driver on the target machine.

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade \
  torch torchvision torchaudio

python -m pip install --upgrade \
  "transformers>=5.10.1" \
  accelerate datasets huggingface_hub safetensors \
  sentencepiece protobuf tqdm numpy pandas matplotlib

python -m pip install --upgrade vllm
```

For training or adapter manipulation, also install:

```bash
python -m pip install --upgrade peft trl wandb
```

The inference CLI was written for the current vLLM generation interface,
including vLLM 0.23-style prompt truncation. If vLLM and Transformers are
already installed in a working training environment, reuse that environment.

### 3. Authenticate

The release is public, but Gemma-derived weights remain subject to the Gemma
license and Hugging Face access requirements.

```bash
huggingface-cli login
```

Verify the relevant interfaces without loading the model:

```bash
python -c "import torch,transformers,vllm,datasets; print(torch.__version__, transformers.__version__, vllm.__version__, datasets.__version__)"
python src/sample_graph_completion.py --help
python src/analyze_graph_completion_benchmark.py --help
```

## Quick dataset smoke test: all six modes

This selects one deterministic example from every trained corruption mode,
prints the raw model continuation, and then prints the reference graph and
score breakdown:

```bash
mkdir -p outputs/graph_completion

python -u src/sample_graph_completion.py \
  --model lamm-mit/Graph-PRefLexOR-4B-Inpainting \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split test \
  --modes prior_empty,fixed_nodes_only,missing_edges,partial_subgraph,wrong_relations,extra_edges \
  --num_tasks 6 \
  --num_generations 1 \
  --generation_batch_size 6 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 42 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view scored \
  --reward_stage shaped \
  --output_jsonl outputs/graph_completion/public-model-six-mode-smoke.jsonl
```

Everything between `RAW DECODED MODEL COMPLETION` and `END RAW COMPLETION` is
the untouched generated string. The CLI does not repair, extract, or rewrite
it before scoring.

## Manual graph completion

Manual inference takes:

1. a natural-language scientific condition;
2. an inline graph through `--partial_graph_json`, or a JSON file through
   `--partial_graph_file`;
3. the corruption mode;
4. a fixed-object policy.

Use:

- `--manual_fixed_policy all` for `prior_empty`, `fixed_nodes_only`,
  `missing_edges`, and `partial_subgraph`. Every supplied node and edge is
  rendered as `[FIXED]` and must be preserved exactly.
- `--manual_fixed_policy none` for `wrong_relations` and `extra_edges`, because
  the model must be allowed to edit or remove supplied edges.

Manual inputs do not have a gold reference, so they support `--view raw`, not
`--view scored`.

The six examples below use one common humidity–silk-film mechanism so the
difference between task modes is explicit.

### 1. Construct from an empty prior: `prior_empty`

The condition supplies all semantic context; the graph canvas is empty.

```bash
mkdir -p outputs/graph_completion/manual

python -u src/sample_graph_completion.py \
  --model lamm-mit/Graph-PRefLexOR-4B-Inpainting \
  --condition "Construct a mechanism graph explaining how increasing humidity reduces the stiffness of a silk fibroin film through water uptake, plasticization, and increased chain mobility." \
  --partial_graph_json '{"nodes":[],"edges":[]}' \
  --manual_mode prior_empty \
  --manual_fixed_policy all \
  --num_generations 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 11 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view raw \
  --output_jsonl outputs/graph_completion/manual/prior-empty.jsonl \
  --output_text_file outputs/graph_completion/manual/prior-empty.txt
```

Expected operation: introduce the scientifically appropriate nodes and edges
needed to represent the mechanism.

### 2. Connect fixed nodes: `fixed_nodes_only`

All supplied nodes are fixed and the edge set is empty.

```bash
python -u src/sample_graph_completion.py \
  --model lamm-mit/Graph-PRefLexOR-4B-Inpainting \
  --condition "Connect the supplied concepts into a mechanism graph explaining how increasing humidity reduces silk fibroin film stiffness through water uptake and chain mobility." \
  --partial_graph_json '{"nodes":[{"id":"Humidity"},{"id":"WaterUptake"},{"id":"ChainMobility"},{"id":"Stiffness"}],"edges":[]}' \
  --manual_mode fixed_nodes_only \
  --manual_fixed_policy all \
  --num_generations 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 11 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view raw \
  --output_jsonl outputs/graph_completion/manual/fixed-nodes-only.jsonl \
  --output_text_file outputs/graph_completion/manual/fixed-nodes-only.txt
```

Expected operation: preserve the four named nodes exactly and infer the
mechanistic edge structure. Additional scientifically necessary content may be
added.

### 3. Add omitted relations: `missing_edges`

The full node set and one correct edge are fixed; other mechanistic edges are
missing.

```bash
python -u src/sample_graph_completion.py \
  --model lamm-mit/Graph-PRefLexOR-4B-Inpainting \
  --condition "Complete the mechanism graph: humidity increases water uptake, water plasticizes the silk matrix and increases chain mobility, and increased mobility reduces stiffness." \
  --partial_graph_json '{"nodes":[{"id":"Humidity"},{"id":"WaterUptake"},{"id":"ChainMobility"},{"id":"Stiffness"}],"edges":[{"source":"Humidity","relation":"increases","target":"WaterUptake"}]}' \
  --manual_mode missing_edges \
  --manual_fixed_policy all \
  --num_generations 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 11 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view raw \
  --output_jsonl outputs/graph_completion/manual/missing-edges.jsonl \
  --output_text_file outputs/graph_completion/manual/missing-edges.txt
```

Expected operation: preserve all supplied objects and add the missing
relations.

### 4. Complete a partial subgraph: `partial_subgraph`

Only part of the mechanism is present. The supplied subgraph is correct and
fixed, but missing intermediate concepts and relations may be added.

```bash
python -u src/sample_graph_completion.py \
  --model lamm-mit/Graph-PRefLexOR-4B-Inpainting \
  --condition "Complete a mechanism graph explaining how humidity changes the stiffness of a silk fibroin film through water uptake, plasticization, microstructural disruption, and chain mobility." \
  --partial_graph_json '{"nodes":[{"id":"Humidity"},{"id":"WaterUptake"},{"id":"Stiffness"}],"edges":[{"source":"Humidity","relation":"increases","target":"WaterUptake"}]}' \
  --manual_mode partial_subgraph \
  --manual_fixed_policy all \
  --num_generations 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 11 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view raw \
  --output_jsonl outputs/graph_completion/manual/partial-subgraph.jsonl \
  --output_text_file outputs/graph_completion/manual/partial-subgraph.txt
```

Expected operation: retain the supplied subgraph and add missing nodes and
edges to produce a complete mechanism.

### 5. Repair corrupted relation labels: `wrong_relations`

The endpoints are plausible, but the supplied relation directions contradict
the condition. The graph is intentionally editable.

```bash
python -u src/sample_graph_completion.py \
  --model lamm-mit/Graph-PRefLexOR-4B-Inpainting \
  --condition "Repair the graph so it states that increasing humidity increases water uptake, while greater water uptake reduces the stiffness of the silk fibroin film." \
  --partial_graph_json '{"nodes":[{"id":"Humidity"},{"id":"WaterUptake"},{"id":"Stiffness"}],"edges":[{"source":"Humidity","relation":"decreases","target":"WaterUptake"},{"source":"WaterUptake","relation":"increases","target":"Stiffness"}]}' \
  --manual_mode wrong_relations \
  --manual_fixed_policy none \
  --num_generations 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 11 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view raw \
  --output_jsonl outputs/graph_completion/manual/wrong-relations.jsonl \
  --output_text_file outputs/graph_completion/manual/wrong-relations.txt
```

Expected operation: replace the incorrect relation labels while retaining the
scientifically relevant nodes and endpoint pairs.

### 6. Remove a spurious edge: `extra_edges`

The graph contains a reverse-causal edge from stiffness to humidity that is not
supported by the condition. The graph is intentionally editable.

```bash
python -u src/sample_graph_completion.py \
  --model lamm-mit/Graph-PRefLexOR-4B-Inpainting \
  --condition "Repair the mechanism graph so it represents humidity-driven water uptake, increased chain mobility, and reduced silk fibroin film stiffness. Remove unsupported causal edges." \
  --partial_graph_json '{"nodes":[{"id":"Humidity"},{"id":"WaterUptake"},{"id":"ChainMobility"},{"id":"Stiffness"}],"edges":[{"source":"Humidity","relation":"increases","target":"WaterUptake"},{"source":"WaterUptake","relation":"increases","target":"ChainMobility"},{"source":"ChainMobility","relation":"decreases","target":"Stiffness"},{"source":"Stiffness","relation":"increases","target":"Humidity"}]}' \
  --manual_mode extra_edges \
  --manual_fixed_policy none \
  --num_generations 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 11 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view raw \
  --output_jsonl outputs/graph_completion/manual/extra-edges.jsonl \
  --output_text_file outputs/graph_completion/manual/extra-edges.txt
```

Expected operation: retain supported content and remove the unsupported
`Stiffness -> Humidity` edge.

## Supplying larger graphs through a file

For a larger graph, save a JSON object containing `nodes` and `edges`:

```json
{
  "nodes": [
    {"id": "Humidity"},
    {"id": "WaterUptake"},
    {"id": "Stiffness"}
  ],
  "edges": [
    {
      "source": "Humidity",
      "relation": "increases",
      "target": "WaterUptake"
    }
  ]
}
```

Then replace the inline argument with:

```text
--partial_graph_file inputs/humidity-partial-graph.json
```

## Official-test benchmark

Sampling and analysis are deliberately separate:

```text
model + official test tasks
    -> raw prediction JSONL
    -> deterministic reference scoring
    -> metrics, tables, and publication figures
```

The current filtered official test split contains 3,641 unique tasks across all
six modes.

### 1. Generate predictions

```bash
cd graph-preflexor-grpo
source .venv/bin/activate

mkdir -p outputs/graph_completion/public-release-test

python -u src/sample_graph_completion.py \
  --model lamm-mit/Graph-PRefLexOR-4B-Inpainting \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split test \
  --invalid_pair_policy filter \
  --num_tasks 3641 \
  --num_generations 1 \
  --generation_batch_size 64 \
  --stream_output_jsonl \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 42 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view raw \
  --output_jsonl outputs/graph_completion/public-release-test/predictions.jsonl \
  > outputs/graph_completion/public-release-test/generation.log 2>&1
```

Monitor progress:

```bash
watch -n 10 'wc -l outputs/graph_completion/public-release-test/predictions.jsonl'
tail -f outputs/graph_completion/public-release-test/generation.log
```

Generation is complete when the JSONL contains 3,641 lines.

If generation was interrupted, preserve completed rows and generate only
missing task identities:

```bash
python -u src/sample_graph_completion.py \
  --model lamm-mit/Graph-PRefLexOR-4B-Inpainting \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split test \
  --invalid_pair_policy filter \
  --num_tasks 3641 \
  --num_generations 1 \
  --generation_batch_size 64 \
  --stream_output_jsonl \
  --resume_output_jsonl \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 42 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view raw \
  --output_jsonl outputs/graph_completion/public-release-test/predictions.jsonl \
  >> outputs/graph_completion/public-release-test/generation.log 2>&1
```

### 2. Analyze predictions

This stage does not load the model or run inference:

```bash
python -u src/analyze_graph_completion_benchmark.py \
  --predictions outputs/graph_completion/public-release-test/predictions.jsonl \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split test \
  --invalid_pair_policy filter \
  --seed 42 \
  --max_completion_length 4096 \
  --reward_stage shaped \
  --aggregation_unit source \
  --bootstrap_samples 2000 \
  --bootstrap_seed 42 \
  --confidence 0.95 \
  --png_dpi 450 \
  --label "Graph-PRefLexOR-4B-Inpainting" \
  --output_dir outputs/graph_completion/public-release-test/analysis
```

The analysis directory contains:

```text
scored_predictions.jsonl
benchmark_summary.json
end_to_end_metrics.csv
conditional_graph_metrics.csv
reward_components.csv
generation_metrics.csv
README.md
benchmark_overall.{svg,png}
benchmark_by_mode.{svg,png}
benchmark_conditional_precision_recall.{svg,png}
benchmark_reward_components.{svg,png}
benchmark_generation_diagnostics.{svg,png}
```

Inspect the main overall metrics:

```bash
jq '.end_to_end.overall |
  {
    reward,
    valid_completion,
    exact_match,
    fixed_contract,
    node,
    edge,
    mode_primary
  }' \
  outputs/graph_completion/public-release-test/analysis/benchmark_summary.json
```

The integrity section should report:

```text
expected_unique_tasks: 3641
predicted_unique_tasks: 3641
predictions_scored: 3641
predictions_without_reference: 0
predictions_with_ambiguous_reference: 0
missing_mode_rows: 0
duplicate_task_generation_rows: 0
missing_expected_tasks: 0
```

## How evaluation works

Evaluation is symbolic and deterministic; it does not use embedding similarity
or an LLM judge.

- Node identity is based on exact node IDs and payloads.
- Edge identity is based on exact `(source, relation, target)` content and
  payloads.
- Node and edge array order does not affect canonical graph matching.
- `exact_match` requires the full canonical graph to equal the reference.
- Fixed objects must be preserved exactly.
- Duplicate or dangling objects are structural errors.
- Mode-specific metrics separately measure additions, relation repair, and
  spurious-edge removal.
- Source-macro aggregation prevents sources with more corrupted variants from
  dominating the reported mean.

The default shaped reward combines:

```text
0.10 format and parsing
0.10 schema and structural validity
0.15 exact fixed-object preservation
0.10 node F1
0.15 edge F1
0.15 mode-specific primary score
0.10 improvement over the unchanged input
0.15 exact canonical graph match
```

## Training summary

- Base model: `google/gemma-4-E4B-it`
- Training dataset: `lamm-mit/graph-canvas-inpainting-121k`
- Objective: Group Relative Policy Optimization (GRPO)
- Selected release: checkpoint 1,350
- Rollouts per prompt: 8
- Rollout temperature: 0.8
- Rollout top-p: 0.95
- Reward: shaped graph-completion reward
- Loss: DAPO
- KL coefficient: 0
- LoRA rank: 64
- LoRA alpha: 128
- LoRA dropout: 0
- Numeric format: BF16
- Rollout backend: colocated vLLM
- Training attention: SDPA
- vLLM CUDA graphs: disabled during training
- Chat template: native thinking enabled
- Maximum prompt length: 4,096 tokens
- Maximum completion length: 4,096 tokens

The public release contains merged full-model weights for direct inference.

## Limitations and responsible use

- A schema-valid graph is not necessarily scientifically correct.
- The model can omit mechanisms, introduce unsupported concepts, or choose an
  incorrect causal relation.
- Exact symbolic metrics do not award semantic equivalence between differently
  named nodes or relations.
- Manual inference has no reference graph and therefore cannot automatically
  establish scientific correctness.
- The six corruption-mode labels are part of the task specification; an
  incorrect mode can encourage the wrong edit behavior.
- Use `--manual_fixed_policy all` when user-supplied facts must not be changed.
- Outputs should be reviewed by a domain expert before use in scientific
  analysis, education, or decision-making.
- The model is intended for research on structured scientific reasoning and
  graph completion, not as an autonomous scientific authority.

## License

This model is derived from Gemma and is distributed subject to the Gemma terms
and the licenses or terms associated with the training data and source code.
Users are responsible for reviewing the applicable terms before redistribution
or deployment.

## Citation

If you use the model, please cite the model, dataset, and repository:

```text
Graph-PRefLexOR-4B-Inpainting
https://huggingface.co/lamm-mit/Graph-PRefLexOR-4B-Inpainting

Graph Canvas Inpainting 121K
https://huggingface.co/datasets/lamm-mit/graph-canvas-inpainting-121k

Graph-PRefLexOR GRPO
https://github.com/lamm-mit/graph-preflexor-grpo
```
