# Gemma 4 Lambda Notes

Personal runbook for Gemma 4 E4B Graph-PRefLexOR warm-start training, then Graph-GRPO.

## Security

Do not store real tokens in this file. If a real `HF_TOKEN`, `WANDB_API_KEY`, or `OPENAI_API_KEY` was pasted into a terminal log, chat, or shared note, rotate it before continuing.

## Instance

Recommended first run:

- E4B SFT: `1x H100 80 GB PCIe` is enough.
- E4B GRPO with colocated vLLM: `1x H100 80 GB PCIe` should work with conservative settings; `1x H100 80 GB SXM5` is safer/faster.
- Avoid A10/A100 40GB/V100 for this GRPO path.

## Setup

```bash
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
  transformers \
  accelerate datasets peft trl \
  openai pydantic tqdm huggingface_hub wandb \
  safetensors sentencepiece protobuf

python -m pip install -U networkx sentence-transformers numpy
python -m pip check
```

Do not install vLLM before SFT/ORPO. Install it only before GRPO, because TRL imports can fail when vLLM is mismatched.

## Secrets and Common Variables

```bash
export HF_TOKEN="hf_..."
export WANDB_API_KEY="..."
export OPENAI_API_KEY="sk-proj-..."

export HF_NAMESPACE="lamm-mit"
export MODEL_ID="google/gemma-4-E4B-it"

export DATASET_SFT="lamm-mit/graph_reasoning_10K"
export DATASET_GRPO="lamm-mit/graph_reasoning_10K"

export WANDB_PROJECT="graph-preflexor"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRL_EXPERIMENTAL_SILENCE=1

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
wandb login "$WANDB_API_KEY"
```

## Pull Latest Fixes

```bash
git pull
```

The current repo includes fixes for:

- new `huggingface_hub` token login API
- current TRL ORPO import path
- current TRL `SFTConfig(max_length=...)`
- Gemma nested config vocab size in `test_model.py`
- configurable LoRA targets
- Gemma native thinking disabled for this baseline

## Version Guidance

Use a current Transformers release, then run the cached-vs-uncached Gemma check below before starting a long job. The Gemma 4 E4B/12B `use_cache=False` issue was fixed upstream, but the preflight is the practical guardrail: if cached and uncached logits disagree badly, stop and fix the install first.

## Import Checks

For SFT/ORPO without vLLM:

```bash
python - <<'PY'
from trl.experimental.orpo import ORPOConfig, ORPOTrainer
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login
print("SFT/ORPO imports ok")
PY
```

Gemma E4B config/tokenizer preflight:

```bash
python - <<'PY'
import torch, transformers
from transformers import AutoConfig, AutoTokenizer

model_id = "google/gemma-4-E4B-it"
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("config:", type(cfg))
print("architectures:", getattr(cfg, "architectures", None))
print("tokenizer size:", len(tok))
PY
```

Optional cached-vs-uncached Gemma check. If this fails badly, do not run long training until the Transformers install is fixed:

```bash
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-4-E4B-it"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()

messages = [{"role": "user", "content": "Say hello in a formal way."}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    cached = model(**inputs, use_cache=True).logits[:, -1, :].float()
    uncached = model(**inputs, use_cache=False).logits[:, -1, :].float()

print("max_abs_diff:", (cached - uncached).abs().max().item())
print("top cached:", tok.decode([cached.argmax(-1).item()]))
print("top uncached:", tok.decode([uncached.argmax(-1).item()]))
PY
```

## SFT Warm-Start

SFT is the preferred initial phase for Gemma 4 because the immediate goal is to teach the model to imitate the Graph-PRefLexOR output format. ORPO is a later ablation if needed.

The current SFT path trains:

- user: `prompt`
- assistant target: `chosen`

It does not use the dataset `answer` field.

```bash
export SFT_OUT="./gemma4-e4b-sft-graph-10k"
export SFT_HUB="$HF_NAMESPACE/gemma4-e4b-sft-graph-10k"
export WANDB_NAME="gemma4-e4b-sft-graph_reasoning_10K"
export WANDB_TAGS="sft,gemma4-e4b,graph_reasoning_10K"
export WANDB_RUN_GROUP="gemma4-e4b-graph-10k"

python src/run_orpo_graph.py \
  --base_model "$MODEL_ID" \
  --dataset "$DATASET_SFT" \
  --output_dir "$SFT_OUT" \
  --mode sft \
  --lora_target_modules language-default \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lr 1e-5 \
  --epochs 1 \
  --batch_size 3 \
  --grad_accum 8 \
  --max_length 3072 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10 \
  --push_to_hub \
  --hub_model_id "$SFT_HUB" \
  --hf_token "$HF_TOKEN"
```

Notes:

- Native Gemma thinking stays disabled for this baseline.

## SFT Structure Test

Plain prompt:

```bash
python src/test_model.py \
  --model "$SFT_OUT" \
  --prompt "Explain how spider silk achieves high toughness." \
  --max_tokens 4096 \
  --temperature 0.2 \
  --chat_template_enable_thinking false
```

Explicit format prompt:

```bash
python src/test_model.py \
  --model "$SFT_OUT" \
  --prompt "Use the Graph-PRefLexOR format with <think>, <brainstorm>, <graph>, <graph_json>, <patterns>, and <synthesis>. Explain how spider silk achieves high toughness." \
  --max_tokens 4096 \
  --temperature 0.2 \
  --chat_template_enable_thinking false
```

If explicit format works but plain prompt does not, the adapter learned the structure but still needs instruction conditioning.

## Merge SFT Adapter for GRPO

```bash
export SFT_MERGED_OUT="./gemma4-e4b-sft-graph-10k-merged"
export SFT_MERGED_HUB="$HF_NAMESPACE/gemma4-e4b-sft-graph-10k-merged"

python - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = os.environ["MODEL_ID"]
adapter = os.environ["SFT_OUT"]
out = os.environ["SFT_MERGED_OUT"]
hub = os.environ["SFT_MERGED_HUB"]
token = os.environ["HF_TOKEN"]

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

print("Loading base:", base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading SFT adapter:", adapter)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()

tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)

print("Saving merged SFT:", out)
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
tok.save_pretrained(out)

print("Pushing merged SFT:", hub)
model.push_to_hub(hub, private=True, token=token)
tok.push_to_hub(hub, private=True, token=token)
PY
```

## Install vLLM Only Before GRPO

```bash
python -m pip install --no-cache-dir "vllm==0.19.0" \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

Validate GRPO imports:

```bash
python - <<'PY'
from trl import GRPOConfig, GRPOTrainer
import vllm
print("GRPO imports ok")
print("vllm:", vllm.__version__)
PY
```

If the import fails with a CUDA runtime error, uninstall vLLM and resolve the CUDA/vLLM mismatch before GRPO:

```bash
python -m pip uninstall -y vllm
```

## Graph-GRPO

```bash
export GRPO_OUT="./gemma4-e4b-grpo-graph-10k"
export GRPO_HUB="$HF_NAMESPACE/gemma4-e4b-grpo-graph-10k"
export WANDB_NAME="gemma4-e4b-grpo-graph_reasoning_10K"
export WANDB_TAGS="grpo,gemma4-e4b,graph_reasoning_10K,vllm"

python src/run_grpo_graph.py \
  --base_model_dir "$SFT_MERGED_HUB" \
  --dataset "$DATASET_GRPO" \
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
  --max_completion_length 8000 \
  --temperature 1.0 \
  --scale_rewards batch \
  --loss_type dapo \
  --lora_target_modules language-default \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --save_steps 100 \
  --logging_steps 10 \
  --chat_template_enable_thinking false \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.35 \
  --push_to_hub \
  --hub_model_id "$GRPO_HUB" \
  --hf_token "$HF_TOKEN" \
  --debug_rewards
```

If memory is comfortable, increase later:

```bash
--num_generations 8 \
--max_completion_length 3500 \
--vllm_gpu_memory_utilization 0.4
```

## Final Test

```bash
python src/test_model.py \
  --model "$GRPO_OUT" \
  --test \
  --max_tokens 3072 \
  --temperature 1.0 \
  --chat_template_enable_thinking false
```

## Notes From Unsloth Gemma 4 Guide

- For text-only Gemma 4 tuning, train language layers, attention modules, and MLP modules; leave vision off.
- For this full Gemma 4 conditional-generation wrapper, raw PEFT `all-linear` also sees vision/audio towers. Use this repo's `language-default` target for text-only SFT/GRPO.
- For RL/GRPO, start around `5e-6`.
- Do not mix native Gemma thought channels with custom visible `<think>` XML-style blocks in the same baseline.
- We keep native Gemma thinking disabled for this Graph-PRefLexOR baseline.
