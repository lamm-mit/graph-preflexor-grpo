#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_orpo_graph.py

SFT or ORPO training for graph-native reasoning.

Modes:
- sft:  Supervised fine-tuning on chosen examples only (faster, simpler)
- orpo: Preference learning using chosen vs rejected (default)

Dataset schema (from make_graph_dataset.py):
- prompt:  user question
- chosen:  full assistant completion with structured reasoning:
    <think>
      <brainstorm>...</brainstorm>
      <graph>...</graph>
      <graph_json>...</graph_json>
      <patterns>...</patterns>
      <synthesis>...</synthesis>
    </think>
    Detailed final answer...
- rejected: weaker, shallow 1-3 sentence answer (only needed for ORPO)

Usage examples:

# ORPO (default)
python run_orpo_graph.py \
  --base_model Qwen/Qwen3-4B \
  --dataset lamm-mit/graph_reasoning_v3 \
  --output_dir ./orpo_model \
  --mode orpo

# SFT (faster, only uses chosen)
python run_orpo_graph.py \
  --base_model Qwen/Qwen3-4B \
  --dataset lamm-mit/graph_reasoning_v3 \
  --output_dir ./sft_model \
  --mode sft
"""

import argparse
import os
from typing import Any, Dict

import torch
from datasets import load_dataset
from huggingface_hub import HfFolder
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import ORPOConfig, ORPOTrainer, SFTConfig, SFTTrainer

# Sentinels (actual tags used in dataset)
SPECIAL_TOKENS = [
    "<think>", "</think>",
    "<brainstorm>", "</brainstorm>",
    "<graph>", "</graph>",
    "<graph_json>", "</graph_json>",
    "<patterns>", "</patterns>",
    "<synthesis>", "</synthesis>",
]


def main():
    parser = argparse.ArgumentParser(description="Run SFT or ORPO for graph-native reasoning.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./orpo_graph_model")
    parser.add_argument("--mode", type=str, default="orpo", choices=["sft", "orpo"],
                        help="Training mode: sft (supervised) or orpo (preference, default)")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=6144, help="Max total length (prompt + completion)")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--no_lora", action="store_true", help="Train full model instead of LoRA")
    parser.add_argument("--add_new_special_tokens", action="store_true",
                        help="Add custom special tokens and resize embeddings (default: False, uses existing tokenizer)")

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_public", action="store_true", help="Make Hub repo public (default: private)")
    parser.add_argument("--hf_token", type=str, default=None)

    args = parser.parse_args()

    if args.hf_token:
        HfFolder.save_token(args.hf_token)

    # Load dataset
    ds_full = load_dataset(args.dataset, split="train")

    print(f"Training mode: {args.mode.upper()}")

    # Validate required columns based on mode
    if args.mode == "orpo":
        required_cols = {"prompt", "chosen", "rejected"}
    else:  # sft
        required_cols = {"prompt", "chosen"}

    missing = required_cols - set(ds_full.column_names)
    if missing:
        raise ValueError(f"Dataset missing required columns for {args.mode}: {missing}")

    # Format data based on mode
    if args.mode == "orpo":
        # Convert to conversation format expected by TRL ORPOTrainer
        # prompt: list of messages, chosen/rejected: list with assistant message
        def format_for_orpo(example):
            return {
                "prompt": [{"role": "user", "content": example["prompt"]}],
                "chosen": [{"role": "assistant", "content": example["chosen"]}],
                "rejected": [{"role": "assistant", "content": example["rejected"]}],
            }
        ds_formatted = ds_full.map(format_for_orpo, remove_columns=ds_full.column_names)
    else:  # sft
        # Convert to messages format for SFTTrainer
        # messages: list of user + assistant messages
        def format_for_sft(example):
            return {
                "messages": [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["chosen"]},
                ],
            }
        ds_formatted = ds_full.map(format_for_sft, remove_columns=ds_full.column_names)

    # Train/eval split
    split = ds_formatted.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Dataset: {len(train_ds)} train, {len(eval_ds)} eval")

    # Load model + tokenizer
    print(f"Loading base model: {args.base_model}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Optionally add special tokens for graph reasoning structure
    if args.add_new_special_tokens:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            print(f"Added {num_added} special tokens, resized embeddings to {len(tokenizer)}")
    else:
        print("Using existing tokenizer vocabulary (no special tokens added)")

    # LoRA config (optional)
    if not args.no_lora:
        #target_modules = [  "q_proj", "k_proj", "v_proj", "o_proj",  "gate_proj", "up_proj", "down_proj", ]
        #target_modules=["embed_tokens", "q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj", "gate_proj"]
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("Training full model (no LoRA)")
        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Create config and trainer based on mode
    if args.mode == "orpo":
        training_args = ORPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            max_length=args.max_length,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            bf16=bool(dtype == torch.bfloat16),
            fp16=bool(dtype == torch.float16),
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id if args.push_to_hub else None,
            hub_private_repo=(not args.hub_public) if args.push_to_hub else None,
            remove_unused_columns=False,
            report_to=["wandb"],
        )

        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )
    else:  # sft
        training_args = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            max_seq_length=args.max_length,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            bf16=bool(dtype == torch.bfloat16),
            fp16=bool(dtype == torch.float16),
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id if args.push_to_hub else None,
            hub_private_repo=(not args.hub_public) if args.push_to_hub else None,
            report_to=["wandb"],
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing {args.mode.upper()} model to Hub: {args.hub_model_id}")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
