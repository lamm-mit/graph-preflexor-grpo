#!/usr/bin/env python
"""Direct natural-language + corrupted-canvas -> complete-graph GRPO training."""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from huggingface_hub import login
from trl import GRPOTrainer

from chat_template_utils import add_chat_template_args, parse_chat_template_enable_thinking
from graph_completion_data import DEFAULT_DATASET, prepare_graph_completion_datasets
from graph_completion_modeling import (
    build_compatible_grpo_config,
    configure_trainable_model,
    load_graph_completion_model_and_tokenizer,
)
from graph_completion_prompting import apply_graph_completion_chat_template
from graph_completion_rewards import (
    add_reward_arguments,
    make_grpo_reward_function,
    reward_config_from_args,
)
from lora_utils import add_lora_config_args


DEFAULT_MODEL = "google/gemma-4-E4B-it"


def _optional_positive(value: int) -> Optional[int]:
    return None if value <= 0 else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train direct complete-graph prediction with deterministic GRPO rewards."
    )
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--tokenizer_model", default=None)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model/training dtype. BF16 is the recommended and default path.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="paged|sdpa",
        help="Transformers attention backend (default: paged|sdpa; FlashAttention is not forced).",
    )
    parser.add_argument("--use_hub_kernels", dest="use_hub_kernels", action="store_true")
    parser.add_argument("--no_hub_kernels", dest="use_hub_kernels", action="store_false")
    parser.set_defaults(use_hub_kernels=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", default="models/Gemma4-E4B-graph-completion-grpo")
    parser.add_argument("--validation_manifest", default=None)
    parser.add_argument("--invalid_pair_policy", choices=["filter", "error"], default="filter")
    parser.add_argument("--validation_source_count", type=int, default=512)
    parser.add_argument("--validation_source_fraction", type=float, default=None)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_eval_rows", type=int, default=0)
    parser.add_argument("--max_source_graphs", type=int, default=0)
    parser.add_argument(
        "--modes",
        default=None,
        help="Optional comma-separated subset of the six graph corruption modes.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--scale_rewards", choices=["batch", "group", "none"], default="batch")
    parser.add_argument("--loss_type", choices=["grpo", "dapo", "dr_grpo", "rloo"], default="dapo")
    parser.add_argument("--mask_truncated_completions", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", default="wandb", help="Comma-separated integrations or 'none'.")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    add_lora_config_args(parser)
    parser.set_defaults(lora_target_modules="language-default")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument(
        "--resume_grpo_checkpoint",
        default=None,
        help="Load an existing LoRA adapter but start a fresh optimizer/schedule.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Exact Trainer recovery including optimizer, scheduler, and global step.",
    )

    add_chat_template_args(parser)
    parser.set_defaults(chat_template_enable_thinking="true")
    add_reward_arguments(parser)

    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", choices=["colocate", "server"], default="colocate")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.30)
    parser.add_argument("--vllm_server_host", default="0.0.0.0")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument(
        "--use_transformers_continuous_batching",
        dest="use_transformers_continuous_batching",
        action="store_true",
    )
    parser.add_argument(
        "--no_transformers_continuous_batching",
        dest="use_transformers_continuous_batching",
        action="store_false",
    )
    parser.set_defaults(use_transformers_continuous_batching=True)
    parser.add_argument(
        "--transformers_kv_cache_memory_percent",
        type=float,
        default=0.45,
        help="Fraction of post-load free GPU memory reserved for paged KV cache.",
    )
    parser.add_argument("--transformers_cache_block_size", type=int, default=256)
    parser.add_argument(
        "--transformers_compile_level",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Continuous-batching compile level; benchmark the default 1 against 0.",
    )
    parser.add_argument(
        "--transformers_cuda_graphs",
        action="store_true",
        help="Enable continuous-batching CUDA graphs (default: disabled initially).",
    )
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    parser.add_argument("--hub_public", action="store_true")
    parser.add_argument("--hf_token", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.resume_grpo_checkpoint and args.resume_from_checkpoint:
        raise ValueError(
            "--resume_grpo_checkpoint and --resume_from_checkpoint are mutually exclusive"
        )
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--hub_model_id is required with --push_to_hub")
    if args.use_vllm and args.use_transformers_continuous_batching:
        raise ValueError(
            "Choose one rollout backend: pass --no_transformers_continuous_batching when using --use_vllm"
        )
    if not 0.0 < args.transformers_kv_cache_memory_percent < 1.0:
        raise ValueError("--transformers_kv_cache_memory_percent must be between 0 and 1")
    if args.dtype == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 was requested but this CUDA device does not report BF16 support")
    print(
        "Rollout performance settings: "
        f"backend={'transformers-continuous-batching' if args.use_transformers_continuous_batching else ('vllm' if args.use_vllm else 'transformers-generate')}, "
        f"attention={args.attn_implementation}, hub_kernels={args.use_hub_kernels}, "
        f"dtype={args.dtype}, kv_cache_memory={args.transformers_kv_cache_memory_percent:.2f}, "
        f"compile_level={args.transformers_compile_level}, "
        f"cuda_graphs={args.transformers_cuda_graphs}"
    )
    if args.hf_token:
        login(token=args.hf_token, add_to_git_credential=False)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()] if args.modes else None
    manifest = args.validation_manifest or os.path.join(
        args.output_dir, "validation_source_indices.json"
    )
    prepared = prepare_graph_completion_datasets(
        args.dataset,
        validation_manifest=manifest,
        invalid_pair_policy=args.invalid_pair_policy,
        seed=args.seed,
        validation_source_count=args.validation_source_count,
        validation_source_fraction=args.validation_source_fraction,
        modes=modes,
        max_train_rows=_optional_positive(args.max_train_rows),
        max_eval_rows=_optional_positive(args.max_eval_rows),
        max_source_graphs=_optional_positive(args.max_source_graphs),
    )

    model, tokenizer = load_graph_completion_model_and_tokenizer(
        args.base_model,
        tokenizer_model=args.tokenizer_model,
        revision=args.model_revision,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        use_hub_kernels=args.use_hub_kernels,
    )
    thinking = parse_chat_template_enable_thinking(args.chat_template_enable_thinking)

    def add_prompt(row):
        return {
            "prompt": apply_graph_completion_chat_template(
                tokenizer,
                row["x0"],
                mode=str(row["mode"]),
                enable_thinking=thinking,
            )
        }

    train = prepared.train.map(add_prompt, desc="Applying graph-completion chat template")
    validation = prepared.validation.map(add_prompt, desc="Applying validation chat template")
    model = configure_trainable_model(
        model,
        no_lora=args.no_lora,
        resume_grpo_checkpoint=args.resume_grpo_checkpoint,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_modules_to_save=args.lora_modules_to_save,
    )

    use_bf16 = args.dtype == "bfloat16"
    use_fp16 = args.dtype == "float16"
    reports = [] if args.report_to.lower() == "none" else [
        item.strip() for item in args.report_to.split(",") if item.strip()
    ]
    config_kwargs = {
        "output_dir": args.output_dir,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "scale_rewards": args.scale_rewards,
        "loss_type": args.loss_type,
        "mask_truncated_completions": args.mask_truncated_completions,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "use_cpu": not torch.cuda.is_available(),
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "eval_strategy": "steps" if args.eval_steps > 0 else "no",
        "eval_steps": args.eval_steps if args.eval_steps > 0 else None,
        "remove_unused_columns": False,
        "report_to": reports,
        "run_name": args.run_name,
        "use_vllm": args.use_vllm,
        "vllm_mode": args.vllm_mode,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_server_host": args.vllm_server_host,
        "vllm_server_port": args.vllm_server_port,
        "vllm_max_model_length": args.max_prompt_length + args.max_completion_length,
        "use_transformers_continuous_batching": args.use_transformers_continuous_batching,
        "transformers_continuous_batching_config": {
            "block_size": args.transformers_cache_block_size,
            "max_memory_percent": args.transformers_kv_cache_memory_percent,
            "use_cuda_graph": args.transformers_cuda_graphs,
            "default_compile_level": args.transformers_compile_level,
            "seed": args.seed,
        }
        if args.use_transformers_continuous_batching
        else None,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "hub_private_repo": not args.hub_public,
        "hub_token": args.hf_token,
    }
    grpo_config = build_compatible_grpo_config(**config_kwargs)
    reward_config = reward_config_from_args(args)
    reward_function = make_grpo_reward_function(reward_config)
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=validation,
        reward_funcs=[reward_function],
        peft_config=None,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
