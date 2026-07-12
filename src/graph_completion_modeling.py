"""Narrow model/adapter helpers for the additive graph-completion workflow."""

from __future__ import annotations

import inspect
import json
import os
from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.utils import is_kernels_available
from trl import GRPOConfig

from lora_utils import parse_lora_modules_to_save, parse_lora_target_modules
from test_model import load_tokenizer_with_fallback


def build_compatible_grpo_config(**kwargs: Any) -> GRPOConfig:
    """Filter GRPOConfig arguments across installed TRL versions."""

    supported = inspect.signature(GRPOConfig).parameters
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    skipped = sorted(set(kwargs) - set(filtered))
    if skipped:
        print(f"Skipping unsupported GRPOConfig args: {', '.join(skipped)}")
    return GRPOConfig(**filtered)


def _adapter_config_path(model_path: str, revision: Optional[str] = None) -> Optional[str]:
    local = os.path.join(model_path, "adapter_config.json")
    if os.path.isfile(local):
        return local
    if os.path.isdir(model_path):
        return None
    try:
        return hf_hub_download(model_path, "adapter_config.json", revision=revision)
    except Exception:
        return None


def is_peft_adapter(model_path: str, revision: Optional[str] = None) -> bool:
    return _adapter_config_path(model_path, revision) is not None


def _adapter_base_model(model_path: str, revision: Optional[str] = None) -> str:
    config_path = _adapter_config_path(model_path, revision)
    if config_path is None:
        raise ValueError(f"no adapter_config.json found for {model_path}")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    base = config.get("base_model_name_or_path")
    if not base:
        raise ValueError(f"adapter {model_path} does not declare base_model_name_or_path")
    return str(base)


DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def training_dtype(name: str = "bfloat16") -> torch.dtype:
    try:
        return DTYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}; choose {sorted(DTYPES)}") from exc


def model_load_kwargs(
    *,
    dtype: str = "bfloat16",
    attn_implementation: str = "paged|sdpa",
    use_hub_kernels: bool = True,
    revision: Optional[str] = None,
) -> dict[str, Any]:
    """Build explicit Transformers load settings for the optimized rollout path."""

    return {
        "revision": revision,
        "dtype": training_dtype(dtype),
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
        "use_kernels": use_hub_kernels,
    }


def load_graph_completion_model_and_tokenizer(
    model_path: str,
    *,
    tokenizer_model: Optional[str] = None,
    revision: Optional[str] = None,
    dtype: str = "bfloat16",
    attn_implementation: str = "paged|sdpa",
    use_hub_kernels: bool = True,
) -> tuple[Any, Any]:
    """Load a full model or merge an input PEFT adapter before GRPO."""

    if use_hub_kernels and not is_kernels_available():
        raise ImportError(
            "Hugging Face Hub kernels are enabled but the compatible 'kernels' package is missing. "
            "Install the version required by Transformers, or pass --no_hub_kernels for a compatibility run."
        )
    adapter = is_peft_adapter(model_path, revision)
    source = _adapter_base_model(model_path, revision) if adapter else model_path
    tokenizer = load_tokenizer_with_fallback(
        model_path,
        tokenizer_model or (source if adapter else None),
        revision=revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    source_revision = None if os.path.isdir(source) or adapter else revision
    model = AutoModelForCausalLM.from_pretrained(
        source,
        **model_load_kwargs(
            dtype=dtype,
            attn_implementation=attn_implementation,
            use_hub_kernels=use_hub_kernels,
            revision=source_revision,
        ),
    )
    if adapter:
        print(f"Merging input adapter {model_path} into {source} before attaching GRPO LoRA")
        model = PeftModel.from_pretrained(model, model_path, revision=revision)
        model = model.merge_and_unload()
    model.config.use_cache = False
    return model, tokenizer


def configure_trainable_model(
    model: Any,
    *,
    no_lora: bool,
    resume_grpo_checkpoint: Optional[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    lora_modules_to_save: str,
) -> Any:
    if no_lora:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model
    if resume_grpo_checkpoint:
        print(f"Loading trainable GRPO adapter from {resume_grpo_checkpoint}")
        return PeftModel.from_pretrained(
            model, resume_grpo_checkpoint, is_trainable=True
        )
    target_modules = parse_lora_target_modules(lora_target_modules)
    modules_to_save = parse_lora_modules_to_save(
        lora_modules_to_save, add_new_special_tokens=False
    )
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
