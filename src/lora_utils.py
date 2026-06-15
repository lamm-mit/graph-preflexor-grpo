"""Shared LoRA configuration helpers for training scripts."""

from __future__ import annotations

import argparse
from typing import List, Optional, Union


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LANGUAGE_DEFAULT_LORA_TARGET_MODULES = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
)

LANGUAGE_ALL_LINEAR_LORA_TARGET_MODULES = (
    r"model\.language_model\."
    r"(?:layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj)|"
    r"per_layer_input_gate|per_layer_projection)|per_layer_model_projection)"
)


def add_lora_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="default",
        help=(
            "LoRA target modules. Use 'default' for q/k/v/o/gate/up/down projections, "
            "'language-default' for Gemma 4 text-only attention/MLP projections, "
            "'language-all-linear' for Gemma 4 text-only linear modules, "
            "'all-linear' for PEFT's broad linear-layer targeting, or a comma-separated list."
        ),
    )
    parser.add_argument(
        "--lora_modules_to_save",
        type=str,
        default="auto",
        help=(
            "Comma-separated non-LoRA modules to save with the adapter. "
            "Use 'auto' to save lm_head,embed_tokens only when adding special tokens; "
            "use 'none' to disable."
        ),
    )


def parse_lora_target_modules(value: str) -> Union[str, List[str]]:
    normalized = value.strip()
    lower = normalized.lower()

    if lower in {"default", "decoder", "qwen"}:
        return list(DEFAULT_LORA_TARGET_MODULES)
    if lower in {"language-default", "text-default", "gemma4-language-default"}:
        return LANGUAGE_DEFAULT_LORA_TARGET_MODULES
    if lower in {"language-all-linear", "text-all-linear", "gemma4-language-all-linear"}:
        return LANGUAGE_ALL_LINEAR_LORA_TARGET_MODULES
    if lower == "all-linear":
        return "all-linear"

    modules = [module.strip() for module in normalized.split(",") if module.strip()]
    if not modules:
        raise ValueError(
            "--lora_target_modules must be 'default', 'language-default', "
            "'language-all-linear', 'all-linear', or a comma-separated list"
        )
    return modules


def parse_lora_modules_to_save(value: str, add_new_special_tokens: bool) -> Optional[List[str]]:
    normalized = value.strip()
    lower = normalized.lower()

    if lower == "none":
        return None
    if lower == "auto":
        return ["lm_head", "embed_tokens"] if add_new_special_tokens else None

    modules = [module.strip() for module in normalized.split(",") if module.strip()]
    if not modules:
        raise ValueError("--lora_modules_to_save must be 'auto', 'none', or a comma-separated list")
    return modules
