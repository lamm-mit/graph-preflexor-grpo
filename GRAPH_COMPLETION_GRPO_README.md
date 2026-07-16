# Gemma 4 Direct Graph-Completion GRPO

This is an additive workflow. It does not replace or modify the repository's
Graph-PRefLexOR SFT, ORPO, GRPO, advanced GRPO, inference, or merge workflows.

The task is:

```text
natural-language condition + incomplete/corrupted graph canvas
    -> native model reasoning
    -> <answer>{complete graph JSON}</answer>
```

Gold graphs are deterministic reward references. They are not teacher-forced
final-answer targets in this workflow. No external judge, OpenAI key, sentence
transformer, EditFlow, or diffusion implementation is used.

## Added files

- `src/run_grpo_graph_completion.py`: standalone GRPO trainer.
- `src/validate_graph_completion.py`: dataset audit and saved-completion scorer.
- `src/sample_graph_completion.py`: mode-balanced vLLM sampling with raw-rollout
  and scored-reference inspection views.
- `src/graph_completion_data.py`: official-split loading, pair auditing, grouped
  validation manifests, caps, and balanced sampling.
- `src/graph_completion_prompting.py`: complete-graph prompt and native-thinking
  chat-template application.
- `src/graph_completion_parsing.py`: graph canvas/JSON canonicalization, strict
  final-answer extraction, schema checks, and pair contracts.
- `src/graph_completion_metrics.py`: exact local graph metrics.
- `src/graph_completion_rewards.py`: format, shaped, and exact reward stages.
- `src/graph_completion_modeling.py`: narrow full-model/PEFT/LoRA loading helpers.
- `tests/test_graph_completion_*.py`: focused unit tests.

No existing repository file is changed by this workflow.

## Access and environment

Accept the model and dataset access conditions on Hugging Face for:

- `google/gemma-4-E4B-it`
- `lamm-mit/graph-canvas-inpainting-121k`

Then authenticate:

```bash
huggingface-cli login
```

The scripts require the same Transformers, TRL, PEFT, Datasets, PyTorch, and
Hugging Face dependencies as the existing GRPO workflow. Check the installed
interfaces without loading a model:

```bash
python -c "import transformers,trl,peft,datasets; print(transformers.__version__, trl.__version__, peft.__version__, datasets.__version__)"
python src/run_grpo_graph_completion.py --help
python src/validate_graph_completion.py --help
python src/sample_graph_completion.py --help
```

Transformers continuous batching requires Transformers 5.8 or newer. Hub
kernels may download compatible kernel implementations from Hugging Face when
the model is loaded, so the training host needs Hub access in addition to the
gated model and dataset access.

The locally inspected Transformers 5.13 installation requires the kernels
package in the `>=0.15.2,<0.16.0` range. Install it on a matching training
environment and confirm availability:

```bash
python -m pip install 'kernels>=0.15.2,<0.16.0'
python -c "from transformers.utils import is_kernels_available; assert is_kernels_available()"
```

The trainer fails early when Hub kernels are enabled but that package is
missing, instead of downloading the full model and failing afterward.

## Weights & Biases project

When `--report_to wandb` is active, the trainer sets the W&B project before
initializing `Trainer`. The graph-completion-specific default is:

```text
--wandb_project graph-completion-grpo --wandb_entity lamm-mit
```

Therefore no environment variable is needed for the normal training command.
These defaults target `https://wandb.ai/lamm-mit/graph-completion-grpo`. Override
the project or W&B entity directly through the trainer:

```text
--wandb_project my-graph-project --wandb_entity lamm-mit
```

`--run_name` continues to name the individual run within that project.

## Rollout performance defaults and Gemma 4 fallback

The trainer requests the initial performance profile for compatible decoder
models:

```text
Transformers continuous batching: enabled
Attention implementation:          paged|sdpa
FlashAttention:                     not forced
Hugging Face Hub kernels:           enabled (use_kernels=True)
Model and training dtype:           BF16
Continuous-batching CUDA graphs:    disabled
Continuous-batching compile level:  1
Paged KV-cache memory budget:       0.45 of free GPU memory after model load
```

Transformers 5.13 continuous batching is not compatible with Gemma 4's
composite configuration and mixed local/global attention head dimensions. Its
paged-cache implementation currently expects a single top-level decoder
geometry. The trainer detects Gemma 4 and safely falls back to ordinary
Transformers generation with standard `sdpa`; it does not copy nested config
attributes into an incorrectly shaped paged cache.

The effective Gemma 4 defaults are therefore:

```text
Generation backend:                 Transformers generate
Attention implementation:          sdpa
Hugging Face Hub kernels:           enabled
Model and training dtype:           BF16
CUDA graphs:                        disabled
Continuous-batching compile/cache:  not applicable
```

The corresponding explicit flags are:

```text
--use_transformers_continuous_batching
--attn_implementation 'paged|sdpa'
--use_hub_kernels
--dtype bfloat16
--transformers_compile_level 1
--transformers_kv_cache_memory_percent 0.45
```

CUDA graphs remain off unless `--transformers_cuda_graphs` is supplied. The
default `--continuous_batching_unsupported_policy fallback` makes the Gemma 4
fallback automatic; use `error` to fail instead. The trainer rejects
simultaneous vLLM and Transformers continuous batching so the requested rollout
backend is unambiguous.

## Dataset and pair validation

The loader requires the official `train` and `test` splits and raises if their
`source_index` sets overlap. Validation is selected only from whole source
groups in `train`; its source IDs are persisted and reused exactly.

The three data roles are deliberately separate:

- **Training:** official `train` rows whose `source_index` is not in the
  validation manifest.
- **Validation:** complete source groups withheld from the official `train`
  split and recorded in `--validation_manifest`. Use this split to select a
  checkpoint and tune inference settings.
- **Test:** the dataset's untouched official `test` split. Reserve this for the
  final, one-time result after checkpoint selection.

Consequently, the checkpoint-selection benchmark below uses `--split
validation`; it does **not** use the official test split. Always reuse the exact
manifest and seed from training so the validation identities remain fixed.

By default, internally impossible pairs are excluded with exact counts and
examples printed by mode. Use `--invalid_pair_policy error` to fail instead.

Audit a small representative subset:

```bash
python -u src/validate_graph_completion.py \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split test \
  --max_rows 60 \
  --invalid_pair_policy filter \
  --validation_manifest outputs/graph_completion/validation_source_indices.json
```

Run the complete schema, split, and pair audit by omitting `--max_rows`:

```bash
python -u src/validate_graph_completion.py \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split test \
  --invalid_pair_policy error \
  --validation_manifest outputs/graph_completion/validation_source_indices.json
```

`datasets.load_from_disk` directories are also supported by passing the local
directory to `--dataset`. It must contain a `DatasetDict` with official train
and test splits.

## Output contract

The model may use its native reasoning channel. Only the last complete block is
scored:

```text
<answer>
{
  "nodes": [...],
  "edges": [...]
}
</answer>
```

Strict reward mode rejects missing closing tags, trailing non-whitespace text,
malformed JSON, invalid schemas, duplicates, and dangling edges. JSON written
inside reasoning is never substituted for the final answer. Optional node and
edge payload fields are preserved and included in exact-object comparisons.

## Reward

The default `shaped` positive weights sum to one:

```text
0.10 format and parsing
0.10 schema and structural validity
0.15 exact fixed-object preservation
0.10 node F1
0.15 edge F1
0.15 mode-specific primary score
0.10 normalized improvement over unchanged x0
0.15 exact canonical match
```

The mode-specific score emphasizes:

- `prior_empty`: node F1, edge F1, and relation accuracy.
- `fixed_nodes_only`: missing-node completion and edge completion.
- `missing_edges`: edge addition recall, recall, and precision.
- `partial_subgraph`: node/edge additions and final edge F1.
- `wrong_relations`: relation repair and wrong-relation removal.
- `extra_edges`: spurious-edge removal, precision, and final edge F1.

Additional penalties cover forbidden fixed changes, duplicates, dangling
edges, spurious objects, unchanged non-target output, truncation, and excessive
length. Any fixed-payload mutation hard-caps reward. Invalid JSON receives no
semantic reward.

Available curricula are:

```text
--reward_stage format
--reward_stage shaped   # default
--reward_stage exact
```

Every positive weight and penalty has a corresponding CLI option shown by
`--help`. If positive shaped weights are changed, they must still sum to 1.0.

## One-step smoke test

This starts directly from untouched Gemma 4 IT and uses standard Transformers
generation. vLLM is not required.

```bash
python -u src/run_grpo_graph_completion.py \
  --base_model google/gemma-4-E4B-it \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --output_dir models/Gemma4-E4B-graph-completion-grpo-smoke \
  --validation_manifest outputs/graph_completion/smoke-validation-sources.json \
  --max_source_graphs 4 \
  --max_train_rows 24 \
  --max_eval_rows 12 \
  --max_steps 1 \
  --num_generations 4 \
  --reward_stage shaped \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --use_hub_kernels \
  --no_transformers_continuous_batching \
  --lora_target_modules language-default \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --chat_template_enable_thinking true \
  --report_to none
```

## One-source overfit/debug run

```bash
python -u src/run_grpo_graph_completion.py \
  --base_model google/gemma-4-E4B-it \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --output_dir models/Gemma4-E4B-graph-completion-one-source \
  --validation_manifest outputs/graph_completion/one-source-validation-sources.json \
  --max_source_graphs 1 \
  --max_steps 20 \
  --num_generations 4 \
  --learning_rate 1e-5 \
  --reward_stage shaped \
  --report_to wandb
```

## Small mode-balanced run

Rows are selected by deterministic round-robin across the requested modes.

```bash
python -u src/run_grpo_graph_completion.py \
  --base_model google/gemma-4-E4B-it \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --output_dir models/Gemma4-E4B-graph-completion-grpo-small \
  --validation_manifest outputs/graph_completion/small-validation-sources.json \
  --max_source_graphs 256 \
  --max_train_rows 1536 \
  --max_eval_rows 192 \
  --num_generations 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --reward_stage shaped \
  --scale_rewards batch \
  --loss_type dapo \
  --report_to wandb
```

## Recommended memory-safe colocated-vLLM LoRA run

This is the canonical conditioned graph-completion run for the DGX Spark/GB10
setup. It incorporates the settings validated during debugging: eight
rollouts, LoRA rank 64, token-level truncated vLLM importance sampling,
evaluation disabled, no truncated-completion masking, eager vLLM execution,
and a conservative colocated KV-cache budget.

```bash
python -u src/run_grpo_graph_completion.py \
  --base_model google/gemma-4-E4B-it \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --output_dir models/Gemma4-E4B-graph-completion-conditioned-grpo-token-tis \
  --validation_manifest outputs/graph_completion/full-validation-sources.json \
  --validation_source_count 512 \
  --invalid_pair_policy filter \
  --num_generations 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --max_steps 2000 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --temperature 0.8 \
  --top_p 0.95 \
  --beta 0.0 \
  --reward_stage shaped \
  --scale_rewards batch \
  --loss_type dapo \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --use_hub_kernels \
  --no_transformers_continuous_batching \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.30 \
  --vllm_enforce_eager \
  --vllm_importance_sampling_correction \
  --vllm_importance_sampling_mode token_truncate \
  --vllm_importance_sampling_clip_max 3.0 \
  --lora_target_modules language-default \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.00 \
  --chat_template_enable_thinking true \
  --save_steps 50 \
  --save_total_limit 3 \
  --eval_steps 0 \
  --logging_steps 1 \
  --report_to wandb \
  --wandb_project graph-completion-grpo \
  --wandb_entity lamm-mit \
  --run_name gemma4-e4b-graph-completion-token-tis-vllm \
  --push_to_hub \
  --hub_model_id lamm-mit/Gemma4-E4B-graph-completion-conditioned-grpo-token-tis
```

The Hub repository is private because `--hub_public` is absent. `--max_steps`
overrides the epoch default; 2,000 steps is still a multi-day run at the
observed eager-vLLM throughput.

### Memory and OOM rationale

- Keep `--per_device_train_batch_size 1`. Eight completions are already
  generated for every prompt, and the LoRA training model and colocated vLLM
  engine share the same device memory.
- Keep `--vllm_gpu_memory_utilization 0.30` for training. The 0.45 value used by
  the standalone sampler is not a safe default for colocated training because
  the sampler does not also hold optimizer state, gradients, activations, and
  trainable LoRA weights.
- `--vllm_enforce_eager` disables vLLM compilation and CUDA-graph capture,
  avoiding their additional startup/capture memory until explicitly
  benchmarked. Do not switch to `--no_vllm_enforce_eager` during an established
  run.
- Standard Transformers continuous batching remains off because Gemma 4's
  mixed local/global cache geometry is unsupported, and it must not be enabled
  simultaneously with vLLM.
- BF16 and SDPA apply to the training-side Transformers model. vLLM selects its
  compatible paged attention backend independently.
- If an OOM still occurs, first reduce `--vllm_gpu_memory_utilization` from 0.30
  to 0.25. Next reduce `--max_completion_length`; reduce `--num_generations`
  only as a final tradeoff because it changes the GRPO group size.

### Evaluation and truncation safeguards

`--eval_steps 0` intentionally disables periodic validation. The fixed
validation manifest is still prepared, but no validation generation occurs.
The full 5,629-row validation set previously projected to roughly 172 hours per
evaluation and increased memory pressure. If periodic evaluation is later
enabled, cap it first (for example `--max_eval_rows 48`) and ensure the global
evaluation batch is divisible by the evaluation generation count.

Do not add `--mask_truncated_completions`. vLLM rollouts frequently omit an EOS
token from the completion IDs even when the decoded response ends with a valid
`</answer>`, causing TRL to report `completions/clipped_ratio=1`. Masking those
rows previously produced zero loss and zero gradients.

During the first 5-10 steps, verify that
`sampling/importance_sampling_ratio/mean` is around order one rather than
`1e-8` or smaller, and that gradient norms do not collapse toward `1e-14`.
`token_truncate` is required because sequence-level correction collapsed across
the thousands of tokens in these graph completions.

Use `--lora_target_modules language-all-linear`, `all-linear`, or a
comma-separated explicit list for alternative LoRA policies. Use `--no_lora`
for full-model training. Arbitrary compatible causal-language-model paths or
Hub IDs can be passed through `--base_model`.

## vLLM

Standard Transformers generation is the default. Enable colocated vLLM only
when the installed vLLM supports the chosen model:

```bash
python -u src/run_grpo_graph_completion.py \
  --output_dir models/Gemma4-E4B-graph-completion-grpo-vllm \
  --validation_manifest outputs/graph_completion/vllm-validation-sources.json \
  --no_transformers_continuous_batching \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.30 \
  --vllm_enforce_eager
```

`--vllm_enforce_eager` is enabled by default and passes `enforce_eager=True`
to colocated vLLM, disabling its torch compilation and CUDA graphs. The
installed TRL interface does not expose this vLLM constructor setting directly,
so the graph-completion trainer applies a narrowly scoped initialization hook.
Use `--no_vllm_enforce_eager` only for an explicit optimized-backend benchmark.

The trainer also overrides the installed TRL `sequence_mask` importance-
sampling default. Multiplying small vLLM/Transformers log-probability
differences across graph completions thousands of tokens long can collapse the
sequence ratio toward zero and extinguish gradients. The graph-completion
defaults are therefore:

```text
--vllm_importance_sampling_correction
--vllm_importance_sampling_mode token_truncate
--vllm_importance_sampling_clip_max 3.0
```

This retains mismatch correction but applies and truncates it per token. Do not
use `sequence_mask` for these long rollouts without first demonstrating healthy
importance ratios and gradient norms in a bounded benchmark.

For an external server, use `--vllm_mode server` and set
`--vllm_server_host`/`--vllm_server_port`.

## Benchmark compile level 1 against 0 on compatible models

Keep the dataset manifest, row/source caps, seed, generation parameters, and
hardware identical. Change only compile level and use distinct output/run names:

```bash
python -u src/run_grpo_graph_completion.py \
  --output_dir models/Gemma4-E4B-graph-completion-compile1 \
  --validation_manifest outputs/graph_completion/compile-benchmark-sources.json \
  --max_source_graphs 4 \
  --max_train_rows 24 \
  --max_steps 10 \
  --seed 42 \
  --transformers_compile_level 1 \
  --run_name graph-completion-compile1

python -u src/run_grpo_graph_completion.py \
  --output_dir models/Gemma4-E4B-graph-completion-compile0 \
  --validation_manifest outputs/graph_completion/compile-benchmark-sources.json \
  --max_source_graphs 4 \
  --max_train_rows 24 \
  --max_steps 10 \
  --seed 42 \
  --transformers_compile_level 0 \
  --run_name graph-completion-compile0
```

Compare warmup time, total wall time, generated tokens/second, peak allocated
VRAM, and reward statistics. Compile level 1 is the recommended default only
after it wins or remains operationally preferable on the target DGX Spark
software stack. This benchmark is not currently applicable to Gemma 4 because
its Transformers continuous-batching path is disabled by the compatibility
guard.

## Resume paths

Exact Trainer crash recovery restores adapter weights, optimizer, scheduler,
and step state:

```bash
python -u src/run_grpo_graph_completion.py \
  --base_model google/gemma-4-E4B-it \
  --output_dir models/Gemma4-E4B-graph-completion-grpo \
  --validation_manifest outputs/graph_completion/full-validation-sources.json \
  --resume_from_checkpoint models/Gemma4-E4B-graph-completion-grpo/checkpoint-100 \
  --report_to wandb
```

Continue adapter weights with a fresh optimizer and schedule:

```bash
python -u src/run_grpo_graph_completion.py \
  --base_model google/gemma-4-E4B-it \
  --output_dir models/Gemma4-E4B-graph-completion-grpo-continued \
  --validation_manifest outputs/graph_completion/full-validation-sources.json \
  --resume_grpo_checkpoint models/Gemma4-E4B-graph-completion-grpo/checkpoint-100 \
  --learning_rate 2e-6 \
  --report_to wandb
```

## Score saved completions

Prediction JSONL rows must contain `source_index`, `variant_index`, and either
`raw_completion` or `completion`:

```json
{"source_index": 42, "variant_index": 3, "raw_completion": "...<answer>{...}</answer>"}
```

The scorer accepts `train`, `validation`, or `test`. The example below scores
against the official test references and should therefore be used only for the
final report, after choosing a checkpoint on validation:

```bash
python -u src/validate_graph_completion.py \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split test \
  --predictions outputs/graph_completion/predictions.jsonl \
  --output_file outputs/graph_completion/scored-predictions.jsonl \
  --validation_manifest outputs/graph_completion/full-validation-sources.json \
  --reward_stage shaped
```

The command writes row-level scores plus a `.summary.json` file with overall
and per-mode means.

## Inspect baseline model rollouts

The graph-completion sampler uses the same prompt builder, native-thinking
setting, prompt/completion limits, temperature, and top-p defaults as training.
It selects valid official-split rows deterministically and balances the sample
across the requested corruption modes. It uses vLLM with BF16, prefix caching,
a 0.45 KV-cache memory budget, and CUDA graphs disabled by default.

Print the exact effective prompt tokens and the untouched decoded model
continuation. No answer extraction, cleanup, reference display, or reward
calculation is performed in this view:

```bash
python -u src/sample_graph_completion.py \
  --model google/gemma-4-E4B-it \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split test \
  --modes missing_edges,wrong_relations,extra_edges \
  --num_tasks 3 \
  --num_generations 1 \
  --temperature 0.8 \
  --top_p 0.95 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view raw \
  --output_jsonl outputs/graph_completion/gemma4-baseline-raw-samples.jsonl
```

Everything between `RAW DECODED MODEL COMPLETION` and `END RAW COMPLETION` is
the rollout string retained in `raw_completion`, exactly as it is passed to the
graph-completion reward function. The JSONL also retains completion token IDs,
token count, finish reason, and stop reason. Increase `--num_generations` to 8
to inspect an entire GRPO-sized rollout group for each task.

For vLLM 0.23 compatibility, prompt truncation is passed through
`LLM.generate(..., tokenization_kwargs=...)`; it is intentionally not supplied
as `SamplingParams(truncate_prompt_tokens=...)`, which vLLM 0.23 rejects. The
sampler constructs `SamplingParams` before loading model weights so any future
sampling-interface incompatibility fails before the expensive Gemma load.

Use the scored view to print that same untouched rollout followed by the gold
complete graph and the deterministic training reward breakdown:

```bash
python -u src/sample_graph_completion.py \
  --model google/gemma-4-E4B-it \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split test \
  --modes missing_edges,wrong_relations,extra_edges \
  --num_tasks 3 \
  --num_generations 1 \
  --temperature 0.8 \
  --top_p 0.95 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view scored \
  --reward_stage shaped \
  --output_jsonl outputs/graph_completion/gemma4-baseline-scored-samples.jsonl
```

The default model is the untouched `google/gemma-4-E4B-it` baseline. A merged
trained checkpoint can be supplied through `--model` for a like-for-like visual
comparison. Hub kernels and Transformers attention flags do not apply to this
sampler because generation is performed directly by vLLM; vLLM manages its own
paged attention backend.

## Complete a user-provided partial graph

Manual inference uses the same graph-completion instruction, native-thinking
chat template, and raw vLLM output path as training. Supply a scientific
condition together with an inline partial graph:

```bash
python -u src/sample_graph_completion.py \
  --model google/gemma-4-E4B-it \
  --condition "Complete a mechanism graph explaining how humidity changes the stiffness of a silk fibroin film through water uptake and microstructure." \
  --partial_graph_json '{"nodes":[{"id":"Humidity"},{"id":"WaterUptake"},{"id":"BetaSheetContent"},{"id":"Stiffness"}],"edges":[{"source":"Humidity","relation":"increases","target":"WaterUptake"}]}' \
  --manual_mode partial_subgraph \
  --manual_fixed_policy all \
  --num_generations 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 11 \
  --max_prompt_length 4096 \
  --max_completion_length 1500 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view raw \
  --output_jsonl outputs/graph_completion/manual-humidity.jsonl \
  --output_text_file outputs/graph_completion/manual-humidity.txt
```

`--manual_fixed_policy all` is the safe completion default: every supplied
node and edge is marked `[FIXED]`, so the model may add missing content but must
not rewrite the user's graph. To submit a corrupted graph that may need edits,
use `--manual_fixed_policy none` and an appropriate `--manual_mode`, such as
`wrong_relations` or `extra_edges`.

Instead of shell-quoting a large graph, store the same JSON object in a file and
replace `--partial_graph_json ...` with:

```text
--partial_graph_file inputs/partial-graph.json
```

Manual inference has no gold graph, so it supports `--view raw` but not
`--view scored`. The JSONL preserves the exact decoded continuation, token IDs,
finish reason, condition, and rendered fixed graph. `--output_text_file` saves
the same human-readable prompt and raw response printed to the terminal.

The sampler accepts a full or merged model through `--model`. A PEFT-only LoRA
checkpoint must first be merged with the repository's existing merge command.

## Push and merge

Hub repositories are private by default. Add these flags to a training command:

```text
--push_to_hub --hub_model_id YOUR_ORG/Gemma4-E4B-graph-completion-grpo
```

Add `--hub_public` only when a public repository is intentional.

The resulting LoRA adapter remains compatible with the existing merge script:

```bash
python -u src/merge_lora_adapter.py \
  --adapter models/Gemma4-E4B-graph-completion-grpo \
  --base_model google/gemma-4-E4B-it \
  --output_dir models/Gemma4-E4B-graph-completion-grpo-merged \
  --no_push_to_hub
```

Consult `python src/merge_lora_adapter.py --help` for the exact installed merge
CLI and existing mistral.rs/Hub options.

## Recover, merge, and publish the selected checkpoint

For the conditioned token-TIS run documented above, training reward peaked
around steps 1,350-1,400 and completion length began a sustained runaway after
roughly step 1,500. Start checkpoint testing with **checkpoint 1,400** and use
checkpoint 1,350 as the fallback comparison. Do not select a checkpoint from
the collapsed 1,650+ region merely because it is newer.

First check whether the candidate checkpoints still exist locally:

```bash
ls -ld \
  models/Gemma4-E4B-graph-completion-conditioned-grpo-token-tis/checkpoint-{1350,1400,1450}
```

With `--save_total_limit 3`, these older local directories may have been
deleted. Because training used `--push_to_hub`, find the Hub revision that was
pushed at step 1,400:

```bash
python - <<'PY'
from huggingface_hub import HfApi

repo = "lamm-mit/Gemma4-E4B-graph-completion-conditioned-grpo-token-tis"
for commit in HfApi().list_repo_commits(repo):
    print(commit.created_at, commit.commit_id, commit.title)
PY
```

Find the entry corresponding to step 1,400, often titled `Training in progress,
step 1400`, and export its commit SHA:

```bash
export ADAPTER_REVISION="PASTE_STEP_1400_COMMIT_SHA"
```

Merge that Hub revision into the Gemma 4 base model and upload the resulting
standalone model:

```bash
python -u src/merge_lora_adapter.py \
  --adapter lamm-mit/Gemma4-E4B-graph-completion-conditioned-grpo-token-tis \
  --adapter_revision "${ADAPTER_REVISION}" \
  --base_model google/gemma-4-E4B-it \
  --tokenizer_model google/gemma-4-E4B-it \
  --processor_model google/gemma-4-E4B-it \
  --output_dir models/Gemma4-E4B-graph-completion-conditioned-grpo-token-tis-merged-step-1400 \
  --dtype bfloat16 \
  --device_map auto \
  --max_shard_size 4GB \
  --push_to_hub \
  --hub_model_id lamm-mit/Gemma4-E4B-graph-completion-conditioned-grpo-token-tis-merged-step-1400 \
  --commit_message "Merge graph-completion GRPO checkpoint 1400"
```

If the local checkpoint directory survived, replace the Hub adapter and
revision arguments above with:

```text
--adapter models/Gemma4-E4B-graph-completion-conditioned-grpo-token-tis \
--checkpoint 1400
```

The merge CLI pushes by default; `--push_to_hub` is included above to make that
side effect explicit. The destination repository is private because
`--hub_public` is absent. `--mistralrs_compat_save` is unnecessary for normal
Transformers or vLLM inference.

### Smoke-test raw and scored generations

Inspect a small, mode-balanced sample before starting a longer benchmark:

```bash
python -u src/sample_graph_completion.py \
  --model lamm-mit/Gemma4-E4B-graph-completion-conditioned-grpo-token-tis-merged-step-1400 \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split train \
  --modes missing_edges,missing_nodes,wrong_relations,extra_edges,partial_subgraph,prior_empty \
  --num_tasks 12 \
  --num_generations 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view scored \
  --reward_stage shaped \
  --output_jsonl outputs/graph_completion/step-1400-visual-samples.jsonl \
  --output_text_file outputs/graph_completion/step-1400-visual-samples.txt
```

This small training-split sample is only a qualitative format and termination
smoke test. Do not use its scores to choose between checkpoints. Verify that
responses contain a complete `<answer>...</answer>` block, terminate before
4,096 tokens, preserve fixed objects, and avoid large numbers of spurious
edges. If checkpoint 1,400 already shows runaway completions, repeat the
recovery and merge procedure for checkpoint 1,350.

### Generate the held-out validation predictions

`validate_graph_completion.py` is a deterministic scorer: it does not load a
model or generate predictions. The dataset sampler currently exposes official
`train` and `test` rows, so the following driver uses the existing preparation
and generation functions directly to select the exact withheld `validation`
rows from the training manifest. It generates one deterministic response for a
mode-balanced 512-row validation subset:

```bash
export MERGED_MODEL="lamm-mit/Gemma4-E4B-graph-completion-conditioned-grpo-token-tis-merged-step-1400"
export VAL_ROWS=512

python -u - <<'PY'
import os
import sys

sys.path.insert(0, "src")

from graph_completion_data import prepare_graph_completion_datasets
from sample_graph_completion import (
    generate_graph_completion_samples,
    write_graph_completion_samples_jsonl,
)

model = os.environ["MERGED_MODEL"]
num_rows = int(os.environ["VAL_ROWS"])
manifest = "outputs/graph_completion/full-validation-sources.json"
output = "outputs/graph_completion/step-1400-validation-512.jsonl"

prepared = prepare_graph_completion_datasets(
    "lamm-mit/graph-canvas-inpainting-121k",
    validation_manifest=manifest,
    validation_source_count=512,
    invalid_pair_policy="filter",
    seed=42,
    max_eval_rows=num_rows,
)

records = generate_graph_completion_samples(
    prepared.validation,
    model=model,
    tokenizer_model=model,
    dtype="bfloat16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.45,
    max_prompt_length=4096,
    max_completion_length=4096,
    num_generations=1,
    temperature=0.0,
    top_p=1.0,
    seed=42,
    enable_thinking=True,
    use_cuda_graphs=False,
    enable_prefix_caching=True,
)

write_graph_completion_samples_jsonl(records, output)
print(f"Wrote {len(records)} validation predictions to {output}")
PY
```

Score those predictions against the same 512 withheld validation rows:

```bash
python -u src/validate_graph_completion.py \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split validation \
  --validation_manifest outputs/graph_completion/full-validation-sources.json \
  --validation_source_count 512 \
  --invalid_pair_policy filter \
  --max_rows 512 \
  --seed 42 \
  --predictions outputs/graph_completion/step-1400-validation-512.jsonl \
  --output_file outputs/graph_completion/step-1400-validation-512-scored.jsonl \
  --max_completion_length 4096 \
  --reward_stage shaped
```

The aggregate overall and per-mode results are written to:

```text
outputs/graph_completion/step-1400-validation-512-scored.jsonl.summary.json
```

Use exactly the same manifest, seed, row cap, decoding settings, and reward
configuration when comparing checkpoint 1,350, checkpoint 1,400, and the base
model. After selecting one checkpoint on validation, run the equivalent
official `test` benchmark once for the final reported result.

### Run the final official test benchmark

Use the dataset's actual official `test` split only after the checkpoint and
decoding settings are frozen. For checkpoint 1,400, generate one deterministic
response for every valid official-test row:

```bash
python -u src/sample_graph_completion.py \
  --model lamm-mit/Gemma4-E4B-graph-completion-conditioned-grpo-token-tis-merged-step-1400 \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split test \
  --num_tasks 3637 \
  --num_generations 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_prompt_length 4096 \
  --max_completion_length 4096 \
  --dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.45 \
  --chat_template_enable_thinking true \
  --view raw \
  --output_jsonl outputs/graph_completion/step-1400-test-predictions.jsonl
```

The pair audit currently leaves 3,637 valid official-test rows after filtering
the four impossible `wrong_relations` pairs. If the dataset revision changes,
use the valid test-row count printed by the audit as `--num_tasks`.

Score the saved raw completions against the official test references:

```bash
python -u src/validate_graph_completion.py \
  --dataset lamm-mit/graph-canvas-inpainting-121k \
  --split test \
  --validation_manifest outputs/graph_completion/full-validation-sources.json \
  --validation_source_count 512 \
  --invalid_pair_policy filter \
  --seed 42 \
  --predictions outputs/graph_completion/step-1400-test-predictions.jsonl \
  --output_file outputs/graph_completion/step-1400-test-scored.jsonl \
  --max_completion_length 4096 \
  --reward_stage shaped
```

The final overall and per-mode test metrics are written to:

```text
outputs/graph_completion/step-1400-test-scored.jsonl.summary.json
```

Do not compare several checkpoints on this split and then report the best one;
that would turn the test set into another validation set. If checkpoint choice
is still open, return to the withheld validation workflow above.

## Tests

The focused tests do not download Gemma or the gated dataset:

```bash
pytest -q \
  tests/test_graph_completion_parsing.py \
  tests/test_graph_completion_metrics_rewards.py \
  tests/test_graph_completion_data_prompting.py \
  tests/test_graph_completion_sampling.py
```
