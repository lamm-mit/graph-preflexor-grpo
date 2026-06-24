# Graph-GRPO: Graph-Native Reasoning Training Pipeline (Graph-PRefLexOR)

Train language models to reason through structured knowledge graphs before answering questions. This pipeline uses ORPO (preference learning) followed by Graph-GRPO (graph reinforcement learning) to teach models a graph-based reasoning process.

## Overview

The model learns to produce responses with this structure:

```
<think>
  <brainstorm>
    Freely explore the problem space, hypotheses, key variables...
  </brainstorm>

  <graph>
    Core entities and their relationships in text form...
  </graph>

  <graph_json>
    {"nodes": [{"id": "A"}, {"id": "B"}], "edges": [{"source": "A", "relation": "influences", "target": "B"}]}
  </graph_json>

  <patterns>
    Abstract patterns/laws (using →, ↑, ↓, ∝ symbols)...
  </patterns>

  <synthesis>
    Integrate all insights into coherent understanding...
  </synthesis>
</think>

Detailed final answer here...
```

## Ideation engine (`ideation/`)

Beyond training, the trained models can drive a **self-expanding knowledge-graph ideation
loop**: seed a topic, generate a graph-native answer, accumulate its `<graph_json>` into one
growing graph (with embedding de-duplication), and expand via follow-up questions under a
compute budget — then score the result for ideation/creativity and compare against a frontier
model. Served with an OpenAI-compatible endpoint (e.g. mistral.rs, Responses API).

See **[`ideation/README.md`](ideation/README.md)** for setup, the strategy/context-mode
options, metrics, and journal-quality plots.

## Pipeline

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  make_graph_dataset │ ──▶ │   run_orpo_graph    │ ──▶ │   run_grpo_graph    │
│    (Data Gen)       │     │  (SFT or ORPO)      │     │  (RL Refinement)    │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
     Teacher LLM              Base Model + LoRA         SFT/ORPO Model + LoRA
     generates                learns format             refines via reward:
     chosen/rejected                                    - correctness (0.4)
                                                        - format (0.3)
                                                        - graph utility (0.3)
```

**Two training paths:**
```
Path A (ORPO):  Dataset ──▶ ORPO (preference) ──▶ Graph-GRPO (RL)   # Uses chosen vs rejected
Path B (SFT):   Dataset ──▶ SFT (supervised)  ──▶ Graph-GRPO (RL)   # Uses chosen only, faster
```

## Released Models & Datasets

### Pre-trained Models

| Model | Base Model | Description |
|-------|------------|-------------|
| [lamm-mit/Graph-Preflexor-8b_12292025](https://huggingface.co/lamm-mit/Graph-Preflexor-8b_12292025) | Qwen-8B | 8B parameter graph reasoning model |
| [lamm-mit/Graph-Preflexor-3b_08012026](https://huggingface.co/lamm-mit/Graph-Preflexor-3b_08012026) | Llama-3.2-3B-Instruct | 3B parameter model (non-reasoning Instruct base) |
| [lamm-mit/Graph-Preflexor-1.7b_08012026](https://huggingface.co/lamm-mit/Graph-Preflexor-1.7b_08012026) | Qwen-1.7B | Lightweight 1.7B parameter graph reasoning model |

### Training Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| [lamm-mit/graph_reasoning_10K](https://huggingface.co/datasets/lamm-mit/graph_reasoning_10K) | 10K examples | Large training dataset |
| [lamm-mit/graph_reasoning_1K](https://huggingface.co/datasets/lamm-mit/graph_reasoning_1K) | 1K examples | Smaller dataset |

---

## Setup

### Installation

```bash
git clone https://github.com/lamm-mit/graph-preflexor-grpo.git
cd graph-preflexor-grpo
```

### Export Secrets

```bash
# OpenAI API (for teacher/judge models)
export OPENAI_API_KEY="sk-proj-..."

# Grok/xAI API (alternative teacher)
export GROK_API_KEY="xai-..."

# Hugging Face (for dataset/model upload)
export HF_TOKEN="hf_..."
```

### Install Dependencies

**Core dependencies:**
```bash
pip install torch transformers datasets peft trl openai pydantic tqdm huggingface_hub wandb
```

**Optional (for vLLM acceleration):**
```bash
pip install vllm
```

You may want to consult the vLLM repo for detailed installation instructions: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

**Optional (for advanced graph rewards):**
```bash
pip install networkx sentence-transformers numpy
```

These are only needed if you enable `--weight_graph_networkx`, `--weight_graph_diversity`, or `--weight_graph_structure`.

## Usage

### Step 1: Generate Dataset

Creates training data with structured reasoning (chosen) vs shallow answers (rejected).

```bash
python src/make_graph_dataset.py \
  --datasets "karpathy/fineweb-edu-100b-shuffle[:2000]|lamm-mit/bio-silk-mech-mix-80K[:1000]" \
  --num_examples 2048 \
  --teacher_model gpt-5.1 \
  --teacher_api_key $OPENAI_API_KEY \
  --reject_model gpt-5-nano \
  --reject_api_key $OPENAI_API_KEY \
  --output_path ./graph_reasoning_dataset.jsonl \
  --save_steps 100 \
  --resume \
  --push_to_hub \
  --output_repo lamm-mit/graph-reasoning-v1
```

### Advanced Dataset Generation (Typed Schema)

The advanced script (`src/make_graph_dataset_advanced.py`) uses a **domain-agnostic typed ontology** with:

- **Node types**: `entity`, `attribute`, `process`, `event`, `outcome`, `law`, `claim`
- **Scale levels**: `micro`, `meso`, `macro` (optional)
- **Constrained relations**: 12 verbs only (`causes`, `enables`, `inhibits`, `modulates`, `part_of`, `instance_of`, `supports`, `challenges`, `represents`, `promotes`, `violates`, `constrains`)

**Example graph_json with typed schema:**
```json
{
  "nodes": [
    {"id": "SilkFiber", "type": "entity", "level": "micro"},
    {"id": "Tension", "type": "attribute", "level": "micro"},
    {"id": "FundamentalFrequency", "type": "outcome", "level": "meso"}
  ],
  "edges": [
    {"source": "SilkFiber", "relation": "constrains", "target": "Tension"},
    {"source": "Tension", "relation": "enables", "target": "FundamentalFrequency"}
  ]
}
```

**Features:**
- **Pydantic structured parsing**: Uses `client.responses.parse()` for reliable JSON extraction
- **Automatic graph repair**: Invalid graphs are automatically fixed using LLM with structured output
- **Schema validation**: All nodes/edges validated against the typed schema

```bash
python src/make_graph_dataset_advanced.py \
  --datasets "karpathy/fineweb-edu-100b-shuffle[:2000]|lamm-mit/bio-silk-mech-mix-80K[:1000]" \
  --num_examples 2048 \
  --teacher_model gpt-4o \
  --teacher_api_key $OPENAI_API_KEY \
  --reject_model gpt-4o-mini \
  --reject_api_key $OPENAI_API_KEY \
  --output_path ./graph_reasoning_advanced.jsonl \
  --save_steps 100 \
  --resume \
  --push_to_hub \
  --output_repo lamm-mit/graph-reasoning-advanced
```
```bash
python src/make_graph_dataset_advanced.py \
  --datasets "karpathy/fineweb-edu-100b-shuffle[:1000]|lamm-mit/bio-silk-mech-mix-80K[:1000]" \
  --num_examples 1024 \
  --teacher_model gpt-4.1-mini \
  --teacher_api_key $OPENAI_API_KEY  \
  --reject_model gpt-4.1-nano \
  --reject_api_key $OPENAI_API_KEY \
  --output_repo lamm-mit/graph_reasoning_adv_v5 \
  --output_path ./graph_reasoning_advanced.jsonl \
  --push_to_hub
```

Download dataset and save as local:
```bash
python -c "from datasets import load_dataset; load_dataset('lamm-mit/graph_reasoning_v3', split='train').to_json('./graph_reasoning.jsonl', lines=True)"
```

Download dataset and push, e.g. for backup:
```bash
python -c "from datasets import load_dataset; ds=load_dataset('lamm-mit/graph_reasoning_v3', split='train'); ds.push_to_hub('lamm-mit/graph_reasoning_backup', private=True)"
```
```bash
python -c "from datasets import load_dataset; ds=load_dataset('lamm-mit/graph_reasoning_adv_v5', split='train'); ds.push_to_hub('lamm-mit/graph_reasoning_adv_v5_backup2', private=True)"
```

### Convert Existing Datasets to Messages

`src/convert_dataset_to_messages.py` rewrites any HF dataset into the `messages=[{role,user}, {role,assistant}]` format used by RL runs. You can merge multiple source repos before upload by separating them with a pipe:

```bash
python src/convert_dataset_to_messages.py \
  --source "lamm-mit/graph_reasoning_v3 | lamm-mit/graph_reasoning_adv_v5 | lamm-mit/graph_reasoning_v4" \
  --target lamm-mit/graph_reasoning_messages_merged \
  --prompt-col prompt \
  --chosen-col chosen
```

Each split with matching names (e.g., `train`, `validation`) is concatenated in order. Quote the `--source` argument when using spaces or multiple datasets.

### Step 2: ORPO Training (Preference Learning)

Teaches the model to prefer structured reasoning over shallow answers. Loads dataset from Hub, pushes model to Hub.

**Dataset format:** Uses `prompt`, `chosen`, and `rejected` columns (not `messages`). The script converts to messages format internally.

| Column | Description | Required for |
|--------|-------------|--------------|
| `prompt` | User question | ORPO & SFT |
| `chosen` | Full assistant completion with structured reasoning | ORPO & SFT |
| `rejected` | Weaker, shallow 1-3 sentence answer | ORPO only |

**Example datasets:**
- [lamm-mit/graph_reasoning_10K](https://huggingface.co/datasets/lamm-mit/graph_reasoning_10K) (10K examples)
- [lamm-mit/graph_reasoning_1K](https://huggingface.co/datasets/lamm-mit/graph_reasoning_1K) (1K examples)

```bash
python src/run_orpo_graph.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --dataset lamm-mit/graph-reasoning-v1 \
  --output_dir ./orpo_graph_model \
  --lora_r 16 \
  --lora_alpha 32 \
  --lr 5e-5 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 4 \
  --max_length 6144 \
  --save_steps 500 \
  --eval_steps 500 \
  --push_to_hub \
  --hub_model_id lamm-mit/llama-3b-orpo-graph \
  --hf_token $HF_TOKEN
```

### Step 3: Graph-GRPO Training (Reinforcement Learning)

Refines the model using reward signals. Loads ORPO model and dataset from Hub, pushes final model to Hub.

**Dataset format:** Graph-GRPO requires `prompt` and `answer` columns (not the `messages` format). The chat template is applied internally.

| Column | Description |
|--------|-------------|
| `prompt` | The question/input |
| `answer` | Gold/reference answer (used by judge for scoring) |

```bash
python src/run_grpo_graph.py \
  --base_model_dir lamm-mit/llama-3b-orpo-graph \
  --dataset lamm-mit/graph-reasoning-v1 \
  --output_dir ./grpo_graph_model \
  --judge_model gpt-5-mini \
  --judge_api_key $OPENAI_API_KEY \
  --weight_correctness 0.4 \
  --weight_format 0.3 \
  --weight_graph_utility 0.3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_generations 4 \
  --learning_rate 5e-6 \
  --epochs 1 \
  --max_completion_length 4096 \
  --save_steps 500 \
  --push_to_hub \
  --hub_model_id lamm-mit/llama-3b-grpo-graph \
  --hf_token $HF_TOKEN
```

### Step 3b: Advanced Graph-GRPO Training (Typed Schema)

For datasets created with `src/make_graph_dataset_advanced.py`, use the advanced Graph-GRPO script that enforces the typed schema:

```bash
python src/run_grpo_graph_advanced.py \
  --base_model_dir lamm-mit/llama-3b-orpo-graph \
  --dataset lamm-mit/graph-reasoning-advanced \
  --output_dir ./grpo_graph_advanced \
  --judge_model gpt-4o-mini \
  --judge_api_key $OPENAI_API_KEY \
  --weight_correctness 0.35 \
  --weight_format 0.25 \
  --weight_graph_utility 0.25 \
  --weight_graph_schema 0.15 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_generations 4 \
  --learning_rate 5e-6 \
  --epochs 1 \
  --max_completion_length 4096 \
  --save_steps 500 \
  --push_to_hub \
  --hub_model_id lamm-mit/llama-3b-grpo-graph-advanced \
  --hf_token $HF_TOKEN
```

**Key difference:** The `--weight_graph_schema` reward evaluates compliance with the typed schema:
- Valid node types (`entity`, `attribute`, `process`, `event`, `outcome`, `law`, `claim`)
- Valid relation verbs (12 constrained verbs)
- Valid scale levels (`micro`, `meso`, `macro`)
- Unique CamelCase node IDs without spaces
- Edge endpoints reference existing nodes

---

## Complete Example Workflows

### Example 1: Quick Test (Small Scale)

```bash
# Export secrets
export OPENAI_API_KEY="sk-proj-..."
export HF_TOKEN="hf_..."

# 1. Generate dataset → push to Hub
python src/make_graph_dataset.py \
  --datasets "lamm-mit/bio-silk-mech-mix-80K[:500]" \
  --num_examples 1024 \
  --teacher_model gpt-5.1 \
  --teacher_api_key $OPENAI_API_KEY \
  --reject_model gpt-5-nano \
  --output_path ./test_dataset.jsonl \
  --save_steps 50 \
  --push_to_hub \
  --output_repo lamm-mit/graph-reasoning-test \
  --hf_token $HF_TOKEN
```
```
# 2. ORPO: load dataset from Hub → push model to Hub
python src/run_orpo_graph.py \
  --base_model Qwen/Qwen3-4B \
  --dataset lamm-mit/graph_reasoning_v3 \
  --output_dir ./orpo-graph_v1 \
  --epochs 1 \
  --save_steps 100 \
  --eval_steps 100 \
  --push_to_hub \
  --hub_model_id  lamm-mit/orpo-graph_v1 \
  --hf_token $HF_TOKEN
```
```
# 3. Graph-GRPO: load ORPO model + dataset from Hub → push final model to Hub
python src/run_grpo_graph.py \
  --base_model_dir lamm-mit/llama-1b-orpo-test \
  --dataset lamm-mit/graph-reasoning-test \
  --output_dir ./orpo-grpo-graph_v1 \
  --judge_model gpt-5-mini \
  --judge_api_key $OPENAI_API_KEY \
  --epochs 1 \
  --num_generations 2 \
  --push_to_hub \
  --hub_model_id lamm-mit/orpo-grpo-graph_v1 \
  --hf_token $HF_TOKEN
```

### Example 2: Production Run with Grok

```bash
# Export secrets
export GROK_API_KEY="xai-..."
export HF_TOKEN="hf_..."

# 1. Generate dataset with Grok → push to Hub
python src/make_graph_dataset.py \
  --datasets "karpathy/fineweb-edu-100b-shuffle[:3000]|lamm-mit/bio-silk-mech-mix-80K[:2000]" \
  --num_examples 4096 \
  --teacher_model grok-3 \
  --teacher_api_key $GROK_API_KEY \
  --teacher_base_url https://api.x.ai/v1 \
  --reject_model grok-3-fast \
  --reject_api_key $GROK_API_KEY \
  --reject_base_url https://api.x.ai/v1 \
  --output_path ./grok_dataset.jsonl \
  --save_steps 100 \
  --resume \
  --push_to_hub \
  --output_repo lamm-mit/graph-reasoning-grok

# 2. ORPO: load dataset from Hub → push model to Hub
python src/run_orpo_graph.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --dataset lamm-mit/graph-reasoning-grok \
  --output_dir ./orpo_grok \
  --lora_r 32 \
  --lora_alpha 64 \
  --lr 3e-5 \
  --epochs 1 \
  --max_length 8192 \
  --push_to_hub \
  --hub_model_id lamm-mit/llama-3b-orpo-grok \
  --hf_token $HF_TOKEN

# 3. Graph-GRPO: load ORPO model from Hub, adjusted weights (emphasize graph utility)
python src/run_grpo_graph.py \
  --base_model_dir lamm-mit/llama-3b-orpo-grok \
  --dataset lamm-mit/graph-reasoning-grok \
  --output_dir ./grpo_grok \
  --judge_model grok-3-fast \
  --judge_api_key $GROK_API_KEY \
  --judge_base_url https://api.x.ai/v1 \
  --weight_correctness 0.35 \
  --weight_format 0.25 \
  --weight_graph_utility 0.40 \
  --num_generations 4 \
  --learning_rate 3e-6 \
  --max_completion_length 20000 \
  --push_to_hub \
  --hub_model_id lamm-mit/llama-3b-grpo-grok \
  --hf_token $HF_TOKEN
```

### Example 3: Full Model Training (No LoRA)

```bash
# Export secrets
export OPENAI_API_KEY="sk-proj-..."
export HF_TOKEN="hf_..."

# 1. Generate large dataset → push to Hub
python src/make_graph_dataset.py \
  --datasets "karpathy/fineweb-edu-100b-shuffle[:5000]|lamm-mit/bio-silk-mech-mix-80K[:3000]" \
  --num_examples 8192 \
  --teacher_model gpt-4o \
  --teacher_api_key $OPENAI_API_KEY \
  --reject_model gpt-4o-mini \
  --reject_api_key $OPENAI_API_KEY \
  --output_path ./large_dataset.jsonl \
  --save_steps 200 \
  --resume \
  --push_to_hub \
  --output_repo lamm-mit/graph-reasoning-large

# 2. ORPO (full model, no LoRA)
python src/run_orpo_graph.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --dataset lamm-mit/graph-reasoning-large \
  --output_dir ./orpo_full \
  --no_lora \
  --lr 1e-5 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 8 \
  --max_length 8192 \
  --save_steps 250 \
  --push_to_hub \
  --hub_model_id lamm-mit/llama-3b-orpo-full \
  --hf_token $HF_TOKEN

# 3. Graph-GRPO: load ORPO model from Hub (full model, no LoRA)
python src/run_grpo_graph.py \
  --base_model_dir lamm-mit/llama-3b-orpo-full \
  --dataset lamm-mit/graph-reasoning-large \
  --output_dir ./grpo_full \
  --judge_model gpt-4o-mini \
  --judge_api_key $OPENAI_API_KEY \
  --no_lora \
  --learning_rate 1e-6 \
  --num_generations 4 \
  --gradient_accumulation_steps 8 \
  --max_completion_length 6144 \
  --push_to_hub \
  --hub_model_id lamm-mit/llama-3b-grpo-full \
  --hf_token $HF_TOKEN
```

---

## Key Arguments Reference

### src/make_graph_dataset.py

| Argument | Description |
|----------|-------------|
| `--datasets` | Pipe-separated datasets with optional `[:N]` sample limits |
| `--num_examples` | Target total examples (with --resume, generates only what's needed) |
| `--teacher_model` | Model for questions + structured answers |
| `--reject_model` | Model for shallow rejected answers |
| `--save_steps` | Save checkpoint every N examples |
| `--resume` | Resume from local file OR download from Hub if local missing |
| `--hub_public` | Make Hub repo public (default: private) |

### src/run_orpo_graph.py

| Argument | Description |
|----------|-------------|
| `--base_model` | HuggingFace model ID |
| `--mode` | Training mode: `sft` or `orpo` (default: orpo) |
| `--no_lora` | Train full model instead of LoRA |
| `--lora_target_modules` | `default` for suffix-based q/k/v/o/gate/up/down, `language-default` for Gemma 4 text-only attention/MLP projections, `language-all-linear` for Gemma 4 text-only linear modules, `all-linear` for broad PEFT targeting, or a comma-separated module list |
| `--lora_modules_to_save` | `auto` saves `lm_head,embed_tokens` only when adding special tokens; use `none` or a comma-separated list to override |
| `--max_length` | Max sequence length (default: 6144) |
| `--save_steps` | Save checkpoint every N steps |
| `--eval_steps` | Evaluate every N steps |
| `--hub_public` | Make Hub repo public (default: private) |

**Mode comparison:**
- `--mode sft`: Supervised fine-tuning on `chosen` only (faster, simpler)
- `--mode orpo`: Preference learning using `chosen` vs `rejected` (default)

### src/run_grpo_graph.py

| Argument | Description |
|----------|-------------|
| `--base_model_dir` | Path to ORPO-trained model |
| `--tokenizer_model` | Optional tokenizer source to use when a merged checkpoint has stale tokenizer metadata |
| `--judge_model` | Model for reward evaluation |
| `--weight_correctness` | Reward weight for answer correctness (default: 0.4) |
| `--weight_format` | Reward weight for format compliance (default: 0.3) |
| `--weight_graph_utility` | Reward weight for graph utility (default: 0.3) |
| `--num_generations` | Completions per prompt for Graph-GRPO (default: 4) |
| `--no_lora` | Train full model instead of LoRA |
| `--lora_target_modules` | `default` for suffix-based q/k/v/o/gate/up/down, `language-default` for Gemma 4 text-only attention/MLP projections, `language-all-linear` for Gemma 4 text-only linear modules, `all-linear` for broad PEFT targeting, or a comma-separated module list |
| `--lora_modules_to_save` | `auto` saves `lm_head,embed_tokens` only when adding special tokens; use `none` or a comma-separated list to override |
| `--chat_template_enable_thinking` | `auto`, `true`, or `false`; pass `false` for Gemma 4 baseline runs that use the Graph-PRefLexOR template |
| `--debug_rewards` | Enable verbose reward/judge logging to `grpo_rewards.log` |
| `--hub_public` | Make Hub repo public (default: private) |

**vLLM Options:**

| Argument | Description |
|----------|-------------|
| `--use_vllm` | Enable vLLM for faster generation |
| `--vllm_mode` | `colocate` (same process, default) or `server` (external vLLM server) |
| `--vllm_gpu_memory_utilization` | GPU memory fraction for colocate mode (default: 0.6) |
| `--vllm_server_host` | vLLM server host for server mode (default: 0.0.0.0) |
| `--vllm_server_port` | vLLM server port for server mode (default: 8000) |

**vLLM Usage Examples:**

```bash
# Colocate mode (default) - vLLM runs in same process
python src/run_grpo_graph.py \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.7 \
  ...

# Server mode - connect to external vLLM server
python src/run_grpo_graph.py \
  --use_vllm \
  --vllm_mode server \
  --vllm_server_host 0.0.0.0 \
  --vllm_server_port 8000 \
  ...
```

**Graph-GRPO Algorithm Options:**

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--scale_rewards` | `batch` | `batch`, `group`, `none` | How to normalize rewards |
| `--loss_type` | `dapo` | `grpo`, `dapo`, `dr_grpo`, `rloo` | Loss function variant |

**Loss types explained:**
- `grpo`: Standard Group Relative Policy Optimization
- `dapo`: Dynamic Advantage Policy Optimization (default, more stable training)
- `dr_grpo`: Dropout-regularized GRPO (adds regularization)
- `rloo`: REINFORCE Leave-One-Out baseline

**Reward scaling explained:**
- `batch`: Normalize rewards across entire batch (default, recommended)
- `group`: Normalize rewards within each prompt's generation group
- `none`: No reward scaling (raw rewards)

**Optional Reward Components:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--weight_graph_schema` | 0.0 | Typed schema compliance (advanced script only) |
| `--weight_graph_networkx` | 0.0 | NetworkX graph validity (requires `networkx`) |
| `--weight_graph_diversity` | 0.0 | Semantic diversity of concepts (requires `sentence-transformers`) |
| `--weight_graph_structure` | 0.0 | Graph topology quality (requires `networkx`) |

Set weight > 0 to enable. See [Reward Components](#reward-components-grpo) for details.

**Note:** `--weight_graph_schema` is only available in `src/run_grpo_graph_advanced.py` and validates against the typed ontology (node types, relation verbs, scale levels).

**Resume Options:**

Two mutually exclusive options for resuming Graph-GRPO training:

| Argument | Use Case | What it does |
|----------|----------|--------------|
| `--resume_grpo_checkpoint` | Continue training with more epochs | Loads LoRA weights only, starts fresh (new optimizer, new LR schedule, step=0) |
| `--resume_from_checkpoint` | Crash recovery | Restores everything (LoRA weights + optimizer + scheduler + step count) |

**Continue training** (finished 1 epoch, want to add more with fresh LR schedule):
```bash
python src/run_grpo_graph.py \
  --base_model_dir lamm-mit/orpo-graph_v20 --no_save_merged_orpo \
  --resume_grpo_checkpoint lamm-mit/orpo-graph_v21 \
  --output_dir ./orpo-grpo-graph_v21_cont \
  --learning_rate 5e-6 \
  ...
```

**Crash recovery** (resume from checkpoint-100):
```bash
python src/run_grpo_graph.py \
  --base_model_dir lamm-mit/orpo-graph_v20 --no_save_merged_orpo \
  --resume_from_checkpoint ./orpo-grpo-graph_v21/checkpoint-100 \
  --output_dir ./orpo-grpo-graph_v21 \
  ...  # use same args as original run
```

**Notes:**
- `--resume_grpo_checkpoint` supports both local paths and Hub model IDs
- `--resume_from_checkpoint` requires a local checkpoint folder with `trainer_state.json`, `optimizer.pt`, etc.
- Using both options simultaneously will raise an error
- Both options are also available in `src/run_grpo_graph_advanced.py`

---

## Reward Components (Graph-GRPO)

Graph-GRPO uses a multi-component reward function with a focus on graph-theoretic analyses. The three core components are always available; three optional components can be enabled by setting their weights > 0.

### Core Reward Components

| Component | Default Weight | Requires API | Description |
|-----------|----------------|--------------|-------------|
| **Correctness** | 0.4 | Yes | LLM judge evaluates answer quality |
| **Format** | 0.3 | No | Structural compliance check |
| **Graph Utility** | 0.3 | Yes (2 calls) | Can a judge answer using only the graph? |

### Optional Reward Components

| Component | Default Weight | Requires | Description |
|-----------|----------------|----------|-------------|
| **Graph Schema** | 0.0 | Advanced script | Typed schema compliance (types, relations, levels) |
| **Graph NetworkX** | 0.0 | `networkx` | Graph validity and consistency |
| **Graph Diversity** | 0.0 | `sentence-transformers` | Semantic spread of concepts |
| **Graph Structure** | 0.0 | `networkx` | Topology quality (depth, internal nodes) |

---

### Detailed Reward Definitions

#### 1. Format Score (`--weight_format`)

Checks structural compliance of the model output. **No API calls required.**

| Sub-component | Points | Criterion |
|---------------|--------|-----------|
| `<think>` tags | 0.15 | Opening and closing tags present |
| `<brainstorm>` tags | 0.10 | Opening and closing tags present |
| `<graph>` tags | 0.15 | Opening and closing tags present |
| `<graph_json>` tags | 0.20 | Present, valid JSON, passes Pydantic schema |
| `<patterns>` tags | 0.15 | Opening and closing tags present |
| `<synthesis>` tags | 0.15 | Opening and closing tags present |
| Non-empty nodes | 0.10 | `graph_json.nodes` has at least 1 node |

**Total: 1.0**

---

#### 2. Correctness Score (`--weight_correctness`)

LLM judge evaluates the **post-thinking answer** against the gold answer.

**Process:**
1. Extract text after `</think>` tag
2. Send to judge with question + gold answer + candidate answer
3. Judge returns score 0.0-1.0

**Rubric (continuous scale):**
- **0.95-1.0**: Fully correct, complete, specific
- **0.80-0.94**: Correct with minor omissions
- **0.60-0.79**: Mostly correct, some gaps or vagueness
- **0.40-0.59**: Partially correct, significant gaps
- **0.20-0.39**: Few correct elements, mostly incomplete
- **0.01-0.19**: Mostly wrong or irrelevant
- **0.0**: Completely wrong or no answer

---

#### 3. Graph Utility Score (`--weight_graph_utility`)

Tests if the `<graph_json>` actually encodes useful information for answering the question.

**Two-step process:**

**Step 1 - Reconstruction:** Judge attempts to answer the question using **only** the graph_json (no other context):
```
Input:  Question + graph_json
Output: {"answer": "...", "graph_coverage": "complete|partial|insufficient"}
```

**Step 2 - Grading:** Judge compares the graph-derived answer to the gold answer:
```
Input:  Gold answer + Graph-derived answer
Output: {"score": 0.0-1.0, "missing_from_graph": "..."}
```

**Rubric:**
- **0.90-1.0**: Graph captured all key concepts and relationships
- **0.70-0.89**: Graph captured most information, minor gaps
- **0.50-0.69**: Graph captured core idea but missing important details
- **0.30-0.49**: Graph captured some relevant info but major gaps
- **0.10-0.29**: Graph barely useful, most information missing
- **0.0-0.09**: Graph not useful for answering this question

**Purpose:** Penalizes "decorative" graphs that look valid but don't actually support reasoning.

---

#### 4. Graph Schema Score (`--weight_graph_schema`)

**Advanced script only.** Evaluates compliance with the typed domain-agnostic ontology.

**Available in:** `src/run_grpo_graph_advanced.py`

| Sub-component | Points | Criterion |
|---------------|--------|-----------|
| Valid JSON with nodes | 0.20 | Has `nodes` array with at least one node |
| Unique CamelCase IDs | 0.20 | All node IDs unique, no spaces |
| Valid node types | 0.20 | All nodes have valid `type` field |
| Valid relations | 0.20 | All edges have valid `relation` verb |
| Edge consistency | 0.10 | All edge endpoints reference existing nodes |
| Valid levels | 0.10 | `level` field (if present) is valid |

**Total: 1.0**

**Valid Node Types:**
- `entity`: Physical or abstract things (e.g., SilkFiber, UserAccount)
- `attribute`: Properties or characteristics (e.g., Tension, Color)
- `process`: Ongoing activities (e.g., Crystallization, DataFlow)
- `event`: Discrete occurrences (e.g., Impact, LoginAttempt)
- `outcome`: Results or effects (e.g., Fracture, SystemFailure)
- `law`: General principles (e.g., HookesLaw, Conservation)
- `claim`: Assertions or hypotheses (e.g., ThesisClaim, Hypothesis)

**Valid Relations (12 constrained verbs):**
- Causal: `causes`, `enables`, `inhibits`, `modulates`
- Structural: `part_of`, `instance_of`
- Argumentative: `supports`, `challenges`
- Symbolic: `represents`, `promotes`, `violates`, `constrains`

**Valid Scale Levels:**
- `micro`: Components, individuals, local details
- `meso`: Interactions, subsystems, workflows
- `macro`: Systems, institutions, themes

**Use case:** Ensures models learn the structured ontology for cross-domain reasoning.

---

#### 5. Graph NetworkX Score (`--weight_graph_networkx`)

**Optional.** Validates that the graph can be parsed into a proper NetworkX DiGraph with consistent structure.

**Requires:** `pip install networkx`

**Note:** Available in both `src/run_grpo_graph.py` and `src/run_grpo_graph_advanced.py`.

| Sub-component | Points | Criterion |
|---------------|--------|-----------|
| Has nodes | 0.30 | At least one node with valid ID |
| Has valid edges | 0.30 | At least one edge between existing nodes |
| Edge consistency | 0.20 | All edge sources/targets exist as nodes |
| No self-loops | 0.10 | No edges where source == target |
| Connectivity | 0.10 | Graph is weakly connected (single component) |

**Total: 1.0**

**Use case:** Catches graphs where edges reference non-existent nodes, or graphs with structural inconsistencies.

---

#### 6. Graph Diversity Score (`--weight_graph_diversity`)

**Optional.** Measures semantic diversity of concepts in the graph using sentence embeddings.

**Requires:** `pip install sentence-transformers numpy`

**Process:**
1. Collect all node IDs and edge descriptions ("source relation target")
2. Encode using `all-MiniLM-L6-v2` sentence transformer
3. Compute average pairwise cosine similarity
4. Convert to diversity score (lower similarity = higher diversity)

**Scoring:**
- Low avg similarity (0.3) → High diversity → Score ~1.0
- High avg similarity (0.9) → Low diversity → Score ~0.2
- Bonus for richer graphs (more nodes/edges)

**Use case:** Penalizes graphs where all nodes are semantically similar (e.g., all variations of the same concept).

---

#### 7. Graph Structure Score (`--weight_graph_structure`)

**Optional.** Evaluates graph topology quality for reasoning.

**Requires:** `pip install networkx`

| Sub-component | Points | Criterion |
|---------------|--------|-----------|
| Size | 0.20 | Optimal: 5-20 nodes (penalize <4 or >30) |
| Edge density | 0.20 | Reward having edges (up to ~10% density) |
| Internal nodes | 0.30 | Ratio of nodes with both in-degree and out-degree > 0 |
| Depth | 0.20 | Longest path length (reward 3-6 hops) |
| Connectivity | 0.10 | Graph is weakly connected |

**Total: 1.0**

**Key metric - Internal Node Ratio:**
- **Source nodes** (in-degree=0): Premises, inputs
- **Sink nodes** (out-degree=0): Conclusions, outputs
- **Internal nodes** (both >0): Reasoning steps

Good reasoning graphs have multiple internal nodes, not just premises → conclusion.

**Use case:** Penalizes shallow graphs (star topology) or linear chains without branching.

---

### Example Weight Configurations

**Default (core only):**
```bash
--weight_correctness 0.4 --weight_format 0.3 --weight_graph_utility 0.3
```

**Emphasize graph quality:**
```bash
--weight_correctness 0.3 \
--weight_format 0.15 \
--weight_graph_utility 0.25 \
--weight_graph_networkx 0.10 \
--weight_graph_diversity 0.10 \
--weight_graph_structure 0.10
```

**Focus on structural validity:**
```bash
--weight_correctness 0.35 \
--weight_format 0.20 \
--weight_graph_utility 0.25 \
--weight_graph_networkx 0.10 \
--weight_graph_structure 0.10
```

**Advanced script with typed schema:**
```bash
# For src/run_grpo_graph_advanced.py
--weight_correctness 0.30 \
--weight_format 0.15 \
--weight_graph_utility 0.25 \
--weight_graph_schema 0.15 \
--weight_graph_networkx 0.05 \
--weight_graph_structure 0.10
```

**Note:** Weights should sum to 1.0. A warning is printed if they don't.

---

## Suggestions:

1. **Start small**: Test with 256-512 examples before scaling up (example datasets with 1k and 10K size are provided)
2. **Monitor eval loss**: Stop ORPO if eval loss diverges
3. **Quality > Quantity**: High-quality teacher outputs matter more than volume
4. **Resume on crash**: Use `--resume` for dataset generation
5. **Adjust weights**: If graphs are decorative, increase `--weight_graph_utility`

---

## Gemma 4 Baseline Runbook

This section records the current Gemma 4 decision and gives the end-to-end Lambda commands for an ORPO then Graph-GRPO run.

### Template Decision

Gemma 4 has a native thinking mode controlled by the official chat template. When enabled, the template inserts `<|think|>` and the model writes reasoning in a native thought channel such as `<|channel>thought ... <channel|>`. Graph-PRefLexOR currently trains and rewards a different visible structure:

```text
<think>
  <brainstorm>...</brainstorm>
  <graph>...</graph>
  <graph_json>...</graph_json>
  <patterns>...</patterns>
  <synthesis>...</synthesis>
</think>
final answer
```

**Discussion note:** for the first Gemma 4 baseline, preserve the existing Graph-PRefLexOR template and do not enable Gemma's native thought channel. A native Gemma thought-channel run is a separate ablation because it requires converting the ORPO data and updating GRPO reward parsing to read `<|channel>thought ... <channel|>` while keeping the graph sentinels inside that thought block.

Use these Gemma-specific flags for the baseline:

```bash
--lora_target_modules language-default \
--chat_template_enable_thinking false
```

`--lora_target_modules language-default` avoids Gemma 4 vision/audio towers while targeting text attention/MLP projections. Raw PEFT `all-linear` is too broad for the full Gemma 4 conditional-generation wrapper. `--chat_template_enable_thinking false` should be used in GRPO and `src/test_model.py` runs so native Gemma thought-channel markup does not compete with the graph tags. ORPO keeps the existing `prompt/chosen/rejected` data format; do not pass `--add_new_special_tokens` unless you intentionally want to resize embeddings and save `lm_head,embed_tokens` with the adapter.

Accept the Gemma model license on Hugging Face before running these commands. The baseline dataset below follows the Qwen 8B recipe and uses `lamm-mit/graph_reasoning_1K`.

References:

- Gemma 4 thinking guide: https://ai.google.dev/gemma/docs/capabilities/thinking
- Gemma QLoRA guide: https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

### Lambda Setup

Start from a fresh Lambda GPU instance. For `google/gemma-4-E4B-it`, prefer an H100 80GB for GRPO with colocated vLLM. For `google/gemma-4-12B-it`, prefer a larger GPU such as B200 180GB; H100 80GB may require the conservative GRPO settings shown below.

```bash
tmux new -s gemma4-grpo

nvidia-smi
python3 --version

git clone https://github.com/lamm-mit/graph-preflexor-grpo.git
cd graph-preflexor-grpo

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

python -m pip install -U \
  torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu129

python -m pip install -U \
  "transformers>=5.10.1" \
  accelerate datasets peft trl \
  openai pydantic tqdm huggingface_hub wandb \
  safetensors sentencepiece protobuf

python -m pip install -U \
  vllm \
  --extra-index-url https://download.pytorch.org/whl/cu129

python -m pip install -U networkx sentence-transformers numpy
```

Authenticate and set common environment variables:

```bash
export HF_TOKEN="hf_..."
export WANDB_API_KEY="..."
export OPENAI_API_KEY="sk-proj-..."
export HF_NAMESPACE="your-hf-user-or-org"

export DATASET="lamm-mit/graph_reasoning_1K"
export WANDB_PROJECT="graph-preflexor"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
wandb login "$WANDB_API_KEY"
```

### Gemma 4 E4B: ORPO then Graph-GRPO

Set the E4B model/output names:

```bash
export MODEL_ID="google/gemma-4-E4B-it"

export ORPO_OUT="./gemma4-e4b-orpo-graph-1k"
export ORPO_HUB="$HF_NAMESPACE/gemma4-e4b-orpo-graph-1k"

export ORPO_MERGED_OUT="./gemma4-e4b-orpo-graph-1k-merged"
export ORPO_MERGED_HUB="$HF_NAMESPACE/gemma4-e4b-orpo-graph-1k-merged"

export GRPO_OUT="./gemma4-e4b-grpo-graph-1k"
export GRPO_HUB="$HF_NAMESPACE/gemma4-e4b-grpo-graph-1k"

export WANDB_RUN_GROUP="gemma4-e4b-graph-1k"
```

Preflight the tokenizer/config:

```bash
python - <<'PY'
import torch, transformers
from transformers import AutoConfig, AutoTokenizer

model_id = "google/gemma-4-E4B-it"
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, extra_special_tokens={})

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("config:", type(cfg))
print("architectures:", getattr(cfg, "architectures", None))
print("tokenizer size:", len(tok))
PY
```

Run ORPO:

```bash
export WANDB_NAME="gemma4-e4b-orpo-graph_reasoning_1K"
export WANDB_TAGS="orpo,gemma4-e4b,graph_reasoning_1K"

python src/run_orpo_graph.py \
  --base_model "$MODEL_ID" \
  --dataset "$DATASET" \
  --output_dir "$ORPO_OUT" \
  --mode orpo \
  --lora_target_modules language-default \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lr 5e-5 \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 4 \
  --max_length 2048 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10 \
  --push_to_hub \
  --hub_model_id "$ORPO_HUB" \
  --hf_token "$HF_TOKEN"
```

Smoke-test the ORPO adapter:

```bash
python src/test_model.py \
  --model "$ORPO_OUT" \
  --test \
  --max_tokens 1024 \
  --temperature 0.7 \
  --chat_template_enable_thinking false
```

Smoke-test a Hub adapter at an earlier commit:

```bash
python src/test_model.py \
  --model lamm-mit/gemma4-e4b-sft-graph-10k-L \
  --revision 1bc18496096922b7a70b38d442ae8808fc6de7f3 \
  --tokenizer_model "$MODEL_ID" \
  --prompt "Explain how spider silk achieves high toughness using graph-based reasoning." \
  --max_tokens 4096 \
  --temperature 0.2 \
  --chat_template_enable_thinking false
```

Merge the `lamm-mit/gemma4-e4b-sft-graph-10k-L` adapter at commit `1bc18496096922b7a70b38d442ae8808fc6de7f3` into a full `_step_600` checkpoint:

```bash
export MODEL_ID="google/gemma-4-E4B-it"
export SFT_ADAPTER_HUB="lamm-mit/gemma4-e4b-sft-graph-10k-L"
export SFT_ADAPTER_REVISION="1bc18496096922b7a70b38d442ae8808fc6de7f3"
export SFT_STEP600_MERGED_OUT="./gemma4-e4b-sft-graph-10k-L_step_600"
export SFT_STEP600_MERGED_HUB="$HF_NAMESPACE/gemma4-e4b-sft-graph-10k-L_step_600"

python - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

base = os.environ["MODEL_ID"]
adapter = os.environ["SFT_ADAPTER_HUB"]
revision = os.environ["SFT_ADAPTER_REVISION"]
out = os.environ["SFT_STEP600_MERGED_OUT"]
hub = os.environ["SFT_STEP600_MERGED_HUB"]
token = os.environ["HF_TOKEN"]

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

print("Loading base:", base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading SFT adapter:", adapter, "revision:", revision)
model = PeftModel.from_pretrained(model, adapter, revision=revision)
model = model.merge_and_unload()

# Keep the clean base processor/tokenizer. Loading from the adapter can preserve
# stale Gemma tokenizer metadata.
processor = AutoProcessor.from_pretrained(base, trust_remote_code=True, extra_special_tokens={})
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, extra_special_tokens={})

print("Saving merged SFT step 600:", out)
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
processor.save_pretrained(out)
tok.save_pretrained(out)

print("Pushing merged SFT step 600:", hub)
model.push_to_hub(hub, private=True, token=token)
processor.push_to_hub(hub, private=True, token=token)
tok.push_to_hub(hub, private=True, token=token)
PY
```

Test the merged `_step_600` model:

```bash
python src/test_model.py \
  --model "$SFT_STEP600_MERGED_HUB" \
  --tokenizer_model "$MODEL_ID" \
  --prompt "Explain how spider silk achieves high toughness using graph-based reasoning." \
  --max_tokens 4096 \
  --temperature 0.2 \
  --chat_template_enable_thinking false
```

Merge the ORPO LoRA adapter into a full model for vLLM-backed GRPO:

```bash
python - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

base = os.environ["MODEL_ID"]
adapter = os.environ["ORPO_OUT"]
out = os.environ["ORPO_MERGED_OUT"]
hub = os.environ["ORPO_MERGED_HUB"]
token = os.environ["HF_TOKEN"]

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

print("Loading base:", base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading ORPO adapter:", adapter)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()

# This baseline does not add special tokens, so keep the clean base tokenizer/processor.
# Loading from the adapter can preserve stale Gemma tokenizer metadata.
processor = AutoProcessor.from_pretrained(base, trust_remote_code=True, extra_special_tokens={})
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, extra_special_tokens={})

print("Saving merged ORPO:", out)
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
processor.save_pretrained(out)
tok.save_pretrained(out)

print("Pushing merged ORPO:", hub)
model.push_to_hub(hub, private=True, token=token)
processor.push_to_hub(hub, private=True, token=token)
tok.push_to_hub(hub, private=True, token=token)
PY
```

Run Graph-GRPO:

```bash
export WANDB_NAME="gemma4-e4b-grpo-graph_reasoning_1K"
export WANDB_TAGS="grpo,gemma4-e4b,graph_reasoning_1K,vllm"

python src/run_grpo_graph.py \
  --base_model_dir "$ORPO_MERGED_HUB" \
  --tokenizer_model "$MODEL_ID" \
  --dataset "$DATASET" \
  --output_dir "$GRPO_OUT" \
  --judge_model gpt-5-mini \
  --judge_api_key "$OPENAI_API_KEY" \
  --weight_correctness 0.30 \
  --weight_format 0.15 \
  --weight_graph_utility 0.25 \
  --weight_graph_networkx 0.10 \
  --weight_graph_diversity 0.10 \
  --weight_graph_structure 0.10 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 8 \
  --learning_rate 5e-6 \
  --epochs 3 \
  --max_prompt_length 1536 \
  --max_completion_length 3500 \
  --temperature 1.0 \
  --scale_rewards batch \
  --loss_type dapo \
  --lora_target_modules language-default \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --save_steps 100 \
  --logging_steps 10 \
  --chat_template_enable_thinking false \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.4 \
  --push_to_hub \
  --hub_model_id "$GRPO_HUB" \
  --hf_token "$HF_TOKEN" \
  --debug_rewards
```

Final test:

```bash
python src/test_model.py \
  --model "$GRPO_OUT" \
  --test \
  --max_tokens 3072 \
  --temperature 1.0 \
  --chat_template_enable_thinking false
```

Merge a Gemma 4 GRPO adapter into a full model for mistral.rs. The `--mistralrs_compat_save` flag is intentionally off by default; it clones the merged state dict and fills missing raw tensor keys from matching full-model sources. If `--raw_tensor_source` is not set, the script tries `base_model`, then `processor_model`, then `tokenizer_model`.

Latest pushed adapter state:

```bash
python src/merge_lora_adapter.py \
  --base_model lamm-mit/gemma4-e4b-sft-graph-10k-L_step_600 \
  --adapter lamm-mit/gemma4-e4b-grpo-from-sftL-step600 \
  --tokenizer_model google/gemma-4-E4B-it \
  --processor_model google/gemma-4-E4B-it \
  --output_dir ./Graph-Preflexor-4b_06222026 \
  --hub_model_id lamm-mit/Graph-Preflexor-4b_06222026 \
  --mistralrs_compat_save \
  --shard_size 50GB \
  --hf_token "$HF_TOKEN"
```

Specific adapter commit, for example step 50:

```bash
python src/merge_lora_adapter.py \
  --base_model lamm-mit/gemma4-e4b-sft-graph-10k-L_step_600 \
  --adapter lamm-mit/gemma4-e4b-grpo-from-sftL-step600 \
  --adapter_commit "<step_50_commit_sha>" \
  --tokenizer_model google/gemma-4-E4B-it \
  --processor_model google/gemma-4-E4B-it \
  --output_dir ./Graph-Preflexor-4b_06222026 \
  --hub_model_id lamm-mit/Graph-Preflexor-4b_06222026 \
  --mistralrs_compat_save \
  --shard_size 50GB \
  --hf_token "$HF_TOKEN"
```

Verify the uploaded model has the raw Gemma keys expected by mistral.rs:

```bash
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open

repo = "lamm-mit/Graph-Preflexor-4b_06222026"
snap = Path(snapshot_download(repo, force_download=True, token=os.environ.get("HF_TOKEN")))
print("snapshot:", snap)

for key in [
    "model.language_model.layers.24.self_attn.k_proj.weight",
    "model.language_model.layers.24.self_attn.v_proj.weight",
    "model.language_model.layers.24.self_attn.k_norm.weight",
]:
    hits = []
    for path in snap.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            if key in handle.keys():
                hits.append(path.name)
    print(key, hits)
PY
```

Check whether Google’s original raw Gemma 4 checkpoint duplicates donor-layer shared-KV values. For `google/gemma-4-E4B-it`, this should report mismatches for `k_proj.weight` and `v_proj.weight`, which is why the merge helper copies missing raw keys from a full checkpoint source rather than reconstructing them from donor layers.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

repo = "google/gemma-4-E4B-it"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
config_path = Path(hf_hub_download(repo_id=repo, filename="config.json", token=token))
cfg = json.loads(config_path.read_text())
text_cfg = cfg.get("text_config", cfg)
first_shared = text_cfg["num_hidden_layers"] - text_cfg["num_kv_shared_layers"]
layer_types = text_cfg["layer_types"]
snap = Path(snapshot_download(repo, token=token, allow_patterns=["*.safetensors"]))
safetensors_files = sorted(snap.glob("*.safetensors"))

def get_tensor(key):
    for path in safetensors_files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if key in handle.keys():
                return handle.get_tensor(key)
    raise KeyError(key)

mismatches = 0
checked = 0
for layer_idx in range(first_shared, text_cfg["num_hidden_layers"]):
    attention_type = layer_types[layer_idx]
    donor_idx = max(
        idx
        for idx, candidate_type in enumerate(layer_types[:first_shared])
        if candidate_type == attention_type
    )
    for suffix in ("k_proj.weight", "k_proj.bias", "v_proj.weight", "v_proj.bias", "k_norm.weight", "k_norm.bias"):
        raw_key = f"model.language_model.layers.{layer_idx}.self_attn.{suffix}"
        donor_key = f"model.language_model.layers.{donor_idx}.self_attn.{suffix}"
        try:
            raw_tensor = get_tensor(raw_key)
            donor_tensor = get_tensor(donor_key)
        except KeyError:
            continue
        checked += 1
        same = torch.equal(raw_tensor, donor_tensor)
        if not same:
            mismatches += 1
            print("MISMATCH", raw_key, "donor", donor_key)

print("checked:", checked)
print("mismatches:", mismatches)
PY
```

Run the merged model directly with mistral.rs:

```bash
mistralrs serve \
  --port 1234 \
  --isq Q8_0 \
  --model-id lamm-mit/Graph-Preflexor-4b_06222026
```

Test the Responses endpoint and pretty-print the JSON:

```bash
curl -s http://localhost:1234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lamm-mit/Graph-Preflexor-4b_06222026",
    "input": "Why are materials often weak?",
    "temperature": 0.4,
    "max_output_tokens": 512,
    "reasoning": {
      "effort": "none"
    }
  }' | jq .
```

If colocated vLLM runs out of memory on E4B, reduce generation pressure:

```bash
--num_generations 4 \
--max_completion_length 3072 \
--vllm_gpu_memory_utilization 0.35
```

### Gemma 4 12B Notes and Commands

Use the 12B model as a separate run. The same Graph-PRefLexOR template decision applies. One important caveat: the 12B model card documents the multimodal loader path (`AutoProcessor` plus `AutoModelForMultimodalLM`). The current training scripts use `AutoTokenizer` and `AutoModelForCausalLM`, so run the preflight first. If 12B fails to load in the current scripts, patch the loader before starting a paid training job.

Set the 12B names:

```bash
export MODEL_ID="google/gemma-4-12B-it"

export ORPO_OUT="./gemma4-12b-orpo-graph-1k"
export ORPO_HUB="$HF_NAMESPACE/gemma4-12b-orpo-graph-1k"

export ORPO_MERGED_OUT="./gemma4-12b-orpo-graph-1k-merged"
export ORPO_MERGED_HUB="$HF_NAMESPACE/gemma4-12b-orpo-graph-1k-merged"

export GRPO_OUT="./gemma4-12b-grpo-graph-1k"
export GRPO_HUB="$HF_NAMESPACE/gemma4-12b-grpo-graph-1k"

export WANDB_RUN_GROUP="gemma4-12b-graph-1k"
```

Preflight the 12B loader and tokenizer/processor:

```bash
python - <<'PY'
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
import transformers

model_id = "google/gemma-4-12B-it"

cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print("transformers:", transformers.__version__)
print("config:", type(cfg))
print("architectures:", getattr(cfg, "architectures", None))

tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, extra_special_tokens={})
print("tokenizer ok:", len(tok))

proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print("processor ok:", type(proc))
PY
```

Run 12B ORPO with conservative batch settings:

```bash
export WANDB_NAME="gemma4-12b-orpo-graph_reasoning_1K"
export WANDB_TAGS="orpo,gemma4-12b,graph_reasoning_1K"

python src/run_orpo_graph.py \
  --base_model "$MODEL_ID" \
  --dataset "$DATASET" \
  --output_dir "$ORPO_OUT" \
  --mode orpo \
  --lora_target_modules language-default \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lr 5e-5 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 8 \
  --max_length 2048 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10 \
  --push_to_hub \
  --hub_model_id "$ORPO_HUB" \
  --hf_token "$HF_TOKEN"
```

Smoke-test the 12B ORPO adapter:

```bash
python src/test_model.py \
  --model "$ORPO_OUT" \
  --test \
  --max_tokens 1024 \
  --temperature 0.7 \
  --chat_template_enable_thinking false
```

Merge the 12B ORPO adapter. If this fails because `AutoModelForCausalLM` is unsupported for the 12B checkpoint in your installed Transformers version, stop and patch the loader before continuing.

```bash
python - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

base = os.environ["MODEL_ID"]
adapter = os.environ["ORPO_OUT"]
out = os.environ["ORPO_MERGED_OUT"]
hub = os.environ["ORPO_MERGED_HUB"]
token = os.environ["HF_TOKEN"]

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

print("Loading base:", base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading ORPO adapter:", adapter)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()

# This baseline does not add special tokens, so keep the clean base tokenizer/processor.
# Loading from the adapter can preserve stale Gemma tokenizer metadata.
processor = AutoProcessor.from_pretrained(base, trust_remote_code=True, extra_special_tokens={})
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, extra_special_tokens={})

print("Saving merged ORPO:", out)
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
processor.save_pretrained(out)
tok.save_pretrained(out)

print("Pushing merged ORPO:", hub)
model.push_to_hub(hub, private=True, token=token)
processor.push_to_hub(hub, private=True, token=token)
tok.push_to_hub(hub, private=True, token=token)
PY
```

Run conservative 12B Graph-GRPO:

```bash
export WANDB_NAME="gemma4-12b-grpo-graph_reasoning_1K"
export WANDB_TAGS="grpo,gemma4-12b,graph_reasoning_1K,vllm"

python src/run_grpo_graph.py \
  --base_model_dir "$ORPO_MERGED_HUB" \
  --tokenizer_model "$MODEL_ID" \
  --dataset "$DATASET" \
  --output_dir "$GRPO_OUT" \
  --judge_model gpt-5-mini \
  --judge_api_key "$OPENAI_API_KEY" \
  --weight_correctness 0.30 \
  --weight_format 0.15 \
  --weight_graph_utility 0.25 \
  --weight_graph_networkx 0.10 \
  --weight_graph_diversity 0.10 \
  --weight_graph_structure 0.10 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --learning_rate 5e-6 \
  --epochs 1 \
  --max_prompt_length 1536 \
  --max_completion_length 3072 \
  --temperature 1.0 \
  --scale_rewards batch \
  --loss_type dapo \
  --lora_target_modules language-default \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --save_steps 100 \
  --logging_steps 10 \
  --chat_template_enable_thinking false \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.3 \
  --push_to_hub \
  --hub_model_id "$GRPO_HUB" \
  --hf_token "$HF_TOKEN" \
  --debug_rewards
```

Test the final 12B checkpoint:

```bash
python src/test_model.py \
  --model "$GRPO_OUT" \
  --test \
  --max_tokens 3072 \
  --temperature 1.0 \
  --chat_template_enable_thinking false
```

---

# Full example `lamm-mit/Graph-Preflexor-8b_12292025`

## Training 

Training run ORPO, then Graph-GRPO:

```bash
python ./src/run_orpo_graph.py
--base_model Qwen/Qwen3-8B
--dataset lamm-mit/graph_reasoning_1K
--output_dir ./orpo-graph
--epochs 1 --lr 5e-5 --batch_size 2
--save_steps 100 --max_length 2048
--eval_steps 100
--push_to_hub
--hub_model_id lamm-mit/orpo-graph
--hf_token $HF_TOKEN
```

Test warm-start model:

```bash
python ./src/test_model.py --model ./orpo-graph
```

Graph-GRPO phase to ultimately produce `lamm-mit/Graph-Preflexor-8b_12292025`: 

```bash
python ./src/run_grpo_graph.py
--base_model_dir lamm-mit/orpo-graph
--dataset lamm-mit/graph_reasoning_1K
--output_dir ./model/Graph-Preflexor-8b_12292025
--judge_model grok-4-1-fast-non-reasoning
--judge_api_key $GROK_API_KEY
--judge_base_url https://api.x.ai/v1
--weight_correctness 0.30
--weight_format 0.15
--weight_graph_utility 0.25
--weight_graph_networkx 0.10
--weight_graph_diversity 0.10
--weight_graph_structure 0.10
--num_generations 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 8
--learning_rate 5e-6 --epochs 3
--max_completion_length 3500
--push_to_hub
--hub_model_id lamm-mit/lamm-mit/Graph-Preflexor-8b_12292025
--hf_token $HF_TOKEN --use_vllm --vllm_gpu_memory_utilization 0.4
```

## Inference / Example Usage for 

For a complete interactive example including graph visualization, see the [Colab notebook](Notebooks/Colab_graph_reasoning.ipynb).

### Basic Inference Example for `lamm-mit/Graph-Preflexor-8b_12292025` 

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

token = 'hf_...' #if needed

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
MODEL_NAME = "lamm-mit/Graph-Preflexor-8b_12292025"
PROMPT = "Give me a short introduction to materiomics."
MAX_NEW_TOKENS = 32_768
THINK_END_TOKEN_ID = 151668  # </think>

# ------------------------------------------------------------------------------
# Model & Tokenizer Loading
# ------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=token,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    token=token,
)
model.eval()

# ------------------------------------------------------------------------------
# Prompt Construction
# ------------------------------------------------------------------------------
messages = [
    {"role": "user", "content": PROMPT}
]

prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # toggles chain-of-thought mode
)

model_inputs = tokenizer(
    prompt_text,
    return_tensors="pt",
).to(model.device)

# ------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------
gen_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,  # sample
    temperature=0.2,
)

with torch.no_grad():
    generated = model.generate(
        **model_inputs,
        generation_config=gen_config,
    )

# Slice off the prompt tokens
output_ids = generated[0, model_inputs.input_ids.shape[1]:].tolist()

# ------------------------------------------------------------------------------
# Thinking / Content Parsing
# ------------------------------------------------------------------------------
def split_thinking(output_ids, tokenizer, think_end_id):
    """
    Split generated tokens into (thinking, final_content) based on </think>.
    Falls back gracefully if no thinking block is present.
    """
    try:
        split_idx = len(output_ids) - output_ids[::-1].index(think_end_id)
    except ValueError:
        split_idx = 0

    thinking = tokenizer.decode(
        output_ids[:split_idx],
        skip_special_tokens=True,
    ).strip()

    content = tokenizer.decode(
        output_ids[split_idx:],
        skip_special_tokens=True,
    ).strip()

    return thinking, content


thinking, content = split_thinking(
    output_ids,
    tokenizer,
    THINK_END_TOKEN_ID,
)

# ------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("THINKING")
print("=" * 80)
print(thinking or "[**no thinking content detected**]")

print("\n" + "=" * 80)
print("FINAL OUTPUT")
print("=" * 80)
print(content)
```

---

## Dataset Conversion (src/convert_dataset_to_messages.py)

Convert a graph reasoning dataset to the standard `messages` format for SFT training with other frameworks.

```bash
# Default (prompt/chosen columns) - private repo
python src/convert_dataset_to_messages.py \
  --source lamm-mit/graph_reasoning_v3 \
  --target lamm-mit/graph_reasoning_v3_messages

# Custom columns
python src/convert_dataset_to_messages.py \
  --source lamm-mit/graph_reasoning_v3 \
  --target lamm-mit/graph_reasoning_v3_messages \
  --prompt-col instruction \
  --chosen-col response

# Public repo
python src/convert_dataset_to_messages.py \
  --source lamm-mit/graph_reasoning_v3 \
  --target lamm-mit/graph_reasoning_v3_messages \
  --hub_public
```

| Argument | Description |
|----------|-------------|
| `--source` | Source dataset on Hub (default: lamm-mit/graph_reasoning_v3) |
| `--target` | Target dataset repo on Hub |
| `--prompt-col` | Column for user prompt (default: prompt) |
| `--chosen-col` | Column for assistant response (default: chosen) |
| `--hub_public` | Make Hub repo public (default: private) |

**Output format:**
```json
{
  "messages": [
    {"role": "user", "content": "<question>"},
    {"role": "assistant", "content": "<think>...</think>\n<answer>"}
  ]
}
```

---

## Testing Checkpoints (src/test_model.py)

Simple CLI to test any model checkpoint with chat template + generation.

```bash
# Interactive mode - type your own prompts
python src/test_model.py --model ./orpo-graph_v1

# Run built-in test prompts (3 bio-inspired questions)
python src/test_model.py --model lamm-mit/orpo-graph_v1 --test

# Single prompt from command line
python src/test_model.py --model ./checkpoint --prompt "How does spider silk achieve high toughness?"

# Load a Hub model or PEFT adapter at an earlier commit
python src/test_model.py --model lamm-mit/orpo-graph_v1 --revision <commit_sha> --test

# Adjust generation params
python src/test_model.py --model ./checkpoint --test --max_tokens 2048 --temperature 0.7

# Gemma 4 baseline: keep Graph-PRefLexOR tags, disable native thought channel
python src/test_model.py --model ./gemma4-e4b-grpo-graph-1k --test --chat_template_enable_thinking false
```

| Argument | Description |
|----------|-------------|
| `--model` | Model path (local directory or Hub ID) |
| `--revision` | Optional Hub branch, tag, or commit SHA to load for `--model`; ignored for local paths |
| `--tokenizer_model` | Optional tokenizer source when a checkpoint has stale tokenizer metadata |
| `--prompt` | Single prompt to run |
| `--test` | Run built-in test prompts |
| `--max_tokens` | Max new tokens (default: 4096) |
| `--temperature` | Sampling temperature (default: 0.6) |
| `--chat_template_enable_thinking` | `auto`, `true`, or `false`; pass `false` for Gemma 4 baseline testing |

---

## Model Merging (src/merge_models.py)

These tools allow you to create interpolated models between a base model and your fine-tuned model using various merging methods.

### Supported Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **linear** | Simple weighted average `(1-α)·base + α·finetuned` | Baseline comparison |
| **slerp** | Spherical interpolation on weight hypersphere | Smooth blends, default choice |
| **ties** | Trim small deltas, resolve sign conflicts | Reducing interference |
| **dare** | Randomly drop 90%+ of deltas, rescale remainder | Aggressive sparsification |
| **task_arithmetic** | Task vectors with optional outlier trimming | Simple task vector addition |

### Usage Examples

```bash
# SLERP merge (default) - create 5 interpolated models
python src/merge_models.py \
  --hf_token $HF_TOKEN \
  --hub_namespace lamm-mit \
  --method slerp \
  --fractions 0.0,0.25,0.5,0.75,1.0 \
  --base_model Qwen/Qwen3-1.7B \
  --grpo_model lamm-mit/Qwen3-1.7B-GRPO

# TIES merge - keep top 50% of weight changes by magnitude
python src/merge_models.py \
  --hf_token $HF_TOKEN \
  --hub_namespace lamm-mit \
  --method ties \
  --density 0.5 \
  --fractions 0.5,0.75,1.0 \
  --base_model Qwen/Qwen3-1.7B \
  --grpo_model lamm-mit/Qwen3-1.7B-GRPO

# DARE merge - drop 90% of weights, rescale remainder
python src/merge_models.py \
  --hf_token $HF_TOKEN \
  --hub_namespace lamm-mit \
  --method dare \
  --drop_rate 0.9 \
  --fractions 0.5,0.75,1.0 \
  --base_model Qwen/Qwen3-1.7B \
  --grpo_model lamm-mit/Qwen3-1.7B-GRPO
```

### src/merge_models.py Arguments

| Argument | Description |
|----------|-------------|
| `--method` | Merge method: linear, slerp, ties, dare, task_arithmetic (default: slerp) |
| `--fractions` | Comma-separated α values in [0,1] (default: 0.0,0.25,0.5,0.75,1.0) |
| `--density` | TIES: fraction of weights to keep (default: 0.5) |
| `--drop_rate` | DARE: fraction of weights to drop (default: 0.9) |
| `--trim_percentile` | Task Arithmetic: outlier percentile to trim (default: 0) |
| `--hub_public` | Make Hub repo public (default: private) |


## Training figures

Generate the ORPO/Graph-GRPO training figures from [Weights & Biases](https://wandb.ai/) logs. All scripts live in `plots/` and read run IDs from a YAML config (kept out of git).

```bash
pip install wandb pandas matplotlib pyyaml   # requirements (+ `wandb login`)
cd plots
cp plot_config.example.yaml plot_config.yaml # then add your wandb entity/project + run IDs
```

`plot_config.yaml` (see `plot_config.example.yaml` for the template):

```yaml
entity: your-wandb-entity
project: your-wandb-project
models:
  - {label: Graph-PRefLexOR-1.7B, color: "#1f77b4", orpo: <orpo_id>, grpo: [<grpo_id>]}
  - {label: Graph-PRefLexOR-3B,   color: "#ff7f0e", orpo: <orpo_id>, grpo: [<grpo_id>, <continuation_id>]}
  - {label: Graph-PRefLexOR-8B,   color: "#2ca02c", orpo: <orpo_id>, grpo: [<grpo_id>]}
max_step_b: 1000   # steps of a grpo continuation run appended after the main run
```

Generate the three paper figures (each writes PNG+SVG+PDF to `plots/figures/`):

```bash
python plot_orpo_panels.py     # ORPO loss + accuracy/margin, panels (a)(b)(c)
python plot_reward_panels.py   # GRPO total reward + 6 components, panels (a)(b)(c)
python plot_length_diag.py     # completion length + GRPO diagnostics, panels (a)(b)
```

Options: `--config PATH` (or `$PLOT_CONFIG`), `--smooth N`, `--out PATH`. `plotcfg.py` is the shared config loader.

### References

- [PRefLexOR](https://arxiv.org/abs/2410.12375) - PRefLexOR Model 
- [Graph-PRefLexOR](https://arxiv.org/abs/2501.08120) - Graph-PRefLexOR Model
- [TIES-Merging](https://arxiv.org/abs/2306.01708) - Trim, Elect Sign & Merge
- [DARE](https://arxiv.org/abs/2311.03099) - Drop And REscale
- [MergeKit](https://github.com/arcee-ai/mergekit) - Comprehensive merging toolkit

```bibtex
@article{Buehler2025PRefLexOR,
  author       = {Buehler, Markus J.},
  title        = {PRefLexOR: preference-based recursive language modeling for exploratory optimization of reasoning and agentic thinking},
  journal      = {npj Artificial Intelligence},
  volume       = {1},
  number       = {4},
  year         = {2025},
  publisher    = {Springer Nature},
  doi          = {10.1038/s44387-025-00003-z},
  url          = {https://doi.org/10.1038/s44387-025-00003-z},
  issn         = {2731-990X},
  received     = {2024-11-01},
  accepted     = {2025-03-22},
  published    = {2025-05-14},
  keywords     = {Complex networks, Computational biology and bioinformatics}
}

@article{Buehler2025GraphPRefLexOR,
  author       = {Buehler, Markus J.},
  title        = {In Situ Graph Reasoning and Knowledge Expansion Using Graph-PRefLexOR},
  journal      = {Advanced Intelligent Discovery},
  year         = {2025},
  publisher    = {Wiley},
  doi          = {10.1002/aidi.202500006},
  url          = {https://doi.org/10.1002/aidi.202500006},
  note         = {Research Article, Open Access},
  published    = {2025-06-09}
}
```
