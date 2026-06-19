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
- Hub commit/revision loading in `test_model.py`
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
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, extra_special_tokens={})

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
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, extra_special_tokens={})
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
tmux new -s gemma4-grpo

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
Longer SFT:
```bash
tmux new -s gemma4-grpo

export SFT_OUT="./gemma4-e4b-sft-graph-10k-L"
export SFT_HUB="$HF_NAMESPACE/gemma4-e4b-sft-graph-10k-L"
export WANDB_NAME="gemma4-e4b-sft-L-graph_reasoning_10K"
export WANDB_TAGS="sft,gemma4-e4b,graph_reasoning_10K-L"
export WANDB_RUN_GROUP="gemma4-e4b-graph-10k-L"

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
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 8 \
  --max_length 3000 \
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

## Load and Merge Longer SFT Step 600

Use this to test or merge the earlier Hub commit for the longer SFT run:

- adapter: `lamm-mit/gemma4-e4b-sft-graph-10k-L`
- revision: `1bc18496096922b7a70b38d442ae8808fc6de7f3`
- merged target suffix: `_step_600`

Smoke-test the adapter at that exact commit:

```bash
export MODEL_ID="google/gemma-4-E4B-it"
export SFT_ADAPTER_HUB="lamm-mit/gemma4-e4b-sft-graph-10k-L"
export SFT_ADAPTER_REVISION="1bc18496096922b7a70b38d442ae8808fc6de7f3"

python src/test_model.py \
  --model "$SFT_ADAPTER_HUB" \
  --revision "$SFT_ADAPTER_REVISION" \
  --tokenizer_model "$MODEL_ID" \
  --prompt "Explain how spider silk achieves high toughness using graph-based reasoning." \
  --max_tokens 4096 \
  --temperature 0.2 \
  --chat_template_enable_thinking false
```

Merge that adapter revision into a full model and push it to Hub:

```bash
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

## Merge SFT Adapter for GRPO

```bash
export SFT_MERGED_OUT="./gemma4-e4b-sft-graph-10k-merged"
export SFT_MERGED_HUB="$HF_NAMESPACE/gemma4-e4b-sft-graph-10k-merged"

python - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

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

# This baseline does not add special tokens, so keep the clean base tokenizer.
# Loading from the adapter can preserve stale Gemma tokenizer metadata.
processor = AutoProcessor.from_pretrained(base, trust_remote_code=True, extra_special_tokens={})
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, extra_special_tokens={})

print("Saving merged SFT:", out)
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
processor.save_pretrained(out)
tok.save_pretrained(out)

print("Pushing merged SFT:", hub)
model.push_to_hub(hub, private=True, token=token)
processor.push_to_hub(hub, private=True, token=token)
tok.push_to_hub(hub, private=True, token=token)
PY
```

```bash
python src/test_model.py \
  --model "$SFT_MERGED_HUB" \
  --prompt "Explain how spider silk achieves high toughness." \
  --max_tokens 4096 \
  --temperature 0.2 \
  --chat_template_enable_thinking false
```

## ORPO variant

```bash
tmux new -s gemma4-grpo

export ORPO_OUT="./gemma4-e4b-orpo-graph-10k"
export ORPO_HUB="$HF_NAMESPACE/gemma4-e4b-orpo-graph-10k"
export WANDB_NAME="gemma4-e4b-orpo-graph_reasoning_10K"
export WANDB_TAGS="orpo,gemma4-e4b,graph_reasoning_10K"
export WANDB_RUN_GROUP="gemma4-e4b-graph-10k"

python src/run_orpo_graph.py \
  --base_model "$MODEL_ID" \
  --dataset "$DATASET_SFT" \
  --output_dir "$ORPO_OUT" \
  --mode orpo \
  --lora_target_modules language-default \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lr 5e-6 \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 8 \
  --max_length 3072 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10 \
  --push_to_hub \
  --hub_model_id "$ORPO_HUB" \
  --hf_token "$HF_TOKEN"
```
And merging:

```bash
export ORPO_MERGED_OUT="./gemma4-e4b-orpo-graph-10k-merged"
export ORPO_MERGED_HUB="$HF_NAMESPACE/gemma4-e4b-orpo-graph-10k-merged"

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

print("Loading base processor/tokenizer:", base)
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
tmux new -s gemma4-grpo

export GRPO_OUT="./gemma4-e4b-grpo-graph-10k"
export GRPO_HUB="$HF_NAMESPACE/gemma4-e4b-grpo-graph-10k"
export WANDB_NAME="gemma4-e4b-grpo-graph_reasoning_10K"
export WANDB_TAGS="grpo,gemma4-e4b,graph_reasoning_10K,vllm"

python src/run_grpo_graph.py \
  --base_model_dir "$SFT_MERGED_HUB" \
  --tokenizer_model "$MODEL_ID" \
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
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_generations 8 \
  --learning_rate 5e-6 \
  --epochs 1 --save_steps 50 \
  --max_completion_length 8000 \
  --temperature 0.4 \
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

Without vLLM (comp issues - vLLM requires an older `transformers` library, which in turn does not support Gemma-4):

```bash
export GRPO_OUT="./gemma4-e4b-grpo-graph-10k"
export GRPO_HUB="$HF_NAMESPACE/gemma4-e4b-grpo-graph-10k"
export WANDB_NAME="gemma4-e4b-grpo-graph_reasoning_10K"
export WANDB_TAGS="grpo,gemma4-e4b,graph_reasoning_10K,vllm"

python src/run_grpo_graph.py \
  --base_model_dir "$SFT_MERGED_HUB" \
  --tokenizer_model "$MODEL_ID" \
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
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --learning_rate 5e-6 \
  --epochs 1 --save_steps 50 \
  --max_completion_length 4000 \
  --temperature 0.4 \
  --scale_rewards batch \
  --loss_type dapo \
  --lora_target_modules language-default \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --save_steps 100 \
  --logging_steps 10 \
  --chat_template_enable_thinking false \
  --push_to_hub \
  --hub_model_id "$GRPO_HUB" \
  --hf_token "$HF_TOKEN" \
  --debug_rewards
```
With Grok model as judge

```bash
export GRPO_OUT="./gemma4-e4b-grpo-graph-10k"
export GRPO_HUB="$HF_NAMESPACE/gemma4-e4b-grpo-graph-10k"
export WANDB_NAME="gemma4-e4b-grpo-graph_reasoning_10K"
export WANDB_TAGS="grpo,gemma4-e4b,graph_reasoning_10K,vllm"

python src/run_grpo_graph.py \
  --base_model_dir "$SFT_MERGED_HUB" \
  --tokenizer_model "$MODEL_ID" \
  --dataset "$DATASET_GRPO" \
  --output_dir "$GRPO_OUT" \
  --judge_model grok-4-1-fast-non-reasoning --judge_api_key "$GROK_API_KEY" \
  --judge_base_url https://api.x.ai/v1 \
  --weight_correctness 0.30 \
  --weight_format 0.15 \
  --weight_graph_utility 0.25 \
  --weight_graph_networkx 0.10 \
  --weight_graph_diversity 0.10 \
  --weight_graph_structure 0.10 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --learning_rate 5e-6 \
  --epochs 1 --save_steps 50 \
  --max_completion_length 4000 \
  --temperature 0.4 \
  --scale_rewards batch \
  --loss_type dapo \
  --lora_target_modules language-default \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --save_steps 100 \
  --logging_steps 10 \
  --chat_template_enable_thinking false \
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

## Merge Final GRPO Adapter

Use the newest `checkpoint-*` directory after training. If there are no checkpoint directories, this falls back to `$GRPO_OUT`.

```bash
export GRPO_FINAL_MERGED_OUT="./gemma4-e4b-grpo-graph-10k-merged"
export GRPO_FINAL_MERGED_HUB="$HF_NAMESPACE/gemma4-e4b-grpo-graph-10k-merged"

python - <<'PY'
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

base = os.environ["SFT_MERGED_HUB"]
grpo_out = Path(os.environ["GRPO_OUT"])
out = os.environ["GRPO_FINAL_MERGED_OUT"]
hub = os.environ["GRPO_FINAL_MERGED_HUB"]
token = os.environ["HF_TOKEN"]
processor_source = os.environ["MODEL_ID"]

checkpoints = sorted(
    [p for p in grpo_out.glob("checkpoint-*") if p.is_dir()],
    key=lambda p: int(p.name.split("-")[-1]),
)
adapter = str(checkpoints[-1] if checkpoints else grpo_out)

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

print("Loading SFT merged base:", base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading final GRPO adapter:", adapter)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()

processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True, extra_special_tokens={})
tok = AutoTokenizer.from_pretrained(processor_source, trust_remote_code=True, extra_special_tokens={})

print("Saving merged GRPO:", out)
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
processor.save_pretrained(out)
tok.save_pretrained(out)

print("Pushing merged GRPO:", hub)
model.push_to_hub(hub, private=True, token=token)
processor.push_to_hub(hub, private=True, token=token)
tok.push_to_hub(hub, private=True, token=token)
PY
```

## Final Test

```bash
python src/test_model.py \
  --model "$GRPO_FINAL_MERGED_HUB" \
  --tokenizer_model "$MODEL_ID" \
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
