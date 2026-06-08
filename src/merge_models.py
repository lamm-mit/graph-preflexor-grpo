#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
merge_models.py

Create interpolated checkpoints between a base model and a fine-tuned model
using various merging methods, and push them to the Hugging Face Hub.

Supported methods:
- linear:  Simple weighted average (1-α)·base + α·finetuned
- slerp:   Spherical linear interpolation (smooth interpolation on hypersphere)
- ties:    TIES-Merging (Trim, Elect Sign, Merge) - prunes small deltas, resolves sign conflicts
- dare:    DARE (Drop And REscale) - randomly drops delta weights, rescales remainder
- task_arithmetic: Task vectors with optional outlier trimming

Example CLI:

    python merge_models.py \
      --hf_token "$HF_TOKEN" \
      --hub_namespace lamm-mit \
      --method slerp \
      --fractions 0.0,0.25,0.5,0.75,1.0 \
      --base_model Qwen/Qwen3-1.7B \
      --grpo_model lamm-mit/Qwen3-1.7B-Bioinspired-Thinking-12-05-2025

References:
- TIES: https://arxiv.org/abs/2306.01708
- DARE: https://arxiv.org/abs/2311.03099
- MergeKit: https://github.com/arcee-ai/mergekit

"""

import argparse
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


SUPPORTED_METHODS = ["linear", "slerp", "ties", "dare", "task_arithmetic"]


def parse_fractions(s: str) -> List[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("No valid fractions parsed from --fractions")
    for v in vals:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Fraction {v} is out of [0,1] range")
    return vals


def slerp_tensor(t0: torch.Tensor, t1: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Spherical linear interpolation between tensors t0, t1 for scalar alpha in [0,1].

    If norms are too small or angle ~0, falls back to linear interpolation.
    """
    if not torch.is_floating_point(t0) or not torch.is_floating_point(t1):
        # Non-float tensors (e.g. int buffers) -> simple copy
        return ((1.0 - alpha) * t0 + alpha * t1) if t0.dtype == torch.float32 or t0.dtype == torch.float16 or t0.dtype == torch.bfloat16 else (t0 if alpha <= 0.5 else t1)

    v0 = t0.detach().float().flatten()
    v1 = t1.detach().float().flatten()

    v0_norm = torch.norm(v0)
    v1_norm = torch.norm(v1)

    if v0_norm == 0 or v1_norm == 0:
        return ((1.0 - alpha) * t0 + alpha * t1).to(t0.dtype)

    v0_unit = v0 / v0_norm
    v1_unit = v1 / v1_norm

    dot = torch.clamp(torch.dot(v0_unit, v1_unit), -1.0, 1.0)
    omega = torch.acos(dot)

    if omega.abs() < 1e-6:
        out = (1.0 - alpha) * v0 + alpha * v1
    else:
        so = torch.sin(omega)
        out = (torch.sin((1.0 - alpha) * omega) / so) * v0 + (torch.sin(alpha * omega) / so) * v1

    out = out.view_as(t0).to(t0.dtype)
    return out


def linear_tensor(t0: torch.Tensor, t1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Simple linear interpolation: (1-α)·t0 + α·t1"""
    if not torch.is_floating_point(t0) or not torch.is_floating_point(t1):
        return t0 if alpha <= 0.5 else t1
    return ((1.0 - alpha) * t0 + alpha * t1).to(t0.dtype)


def ties_merge(
    base_sd: Dict[str, torch.Tensor],
    finetuned_sd: Dict[str, torch.Tensor],
    alpha: float,
    density: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    TIES-Merging: Trim, Elect Sign, and Merge.

    1. Compute task vector (delta = finetuned - base)
    2. Trim: Keep only top-k% of weights by magnitude (density parameter)
    3. Elect sign: Resolve sign conflicts (not needed for single model merge)
    4. Merge: Add scaled task vector back to base

    Args:
        base_sd: Base model state dict
        finetuned_sd: Fine-tuned model state dict
        alpha: Scale factor for task vector
        density: Fraction of weights to keep (0.0-1.0), default 0.5
    """
    merged_sd = {}

    for key in base_sd.keys():
        t_base = base_sd[key]
        t_ft = finetuned_sd[key]

        if not torch.is_floating_point(t_base):
            merged_sd[key] = t_base if alpha <= 0.5 else t_ft
            continue

        # Compute task vector
        delta = (t_ft - t_base).float()

        # Trim: Keep only top density% by magnitude
        flat_delta = delta.flatten()
        k = max(1, int(density * flat_delta.numel()))
        threshold = torch.topk(flat_delta.abs(), k).values[-1]
        mask = flat_delta.abs() >= threshold
        trimmed_delta = torch.where(mask, flat_delta, torch.zeros_like(flat_delta))
        trimmed_delta = trimmed_delta.view_as(delta)

        # Merge: base + α * trimmed_delta
        merged = t_base.float() + alpha * trimmed_delta
        merged_sd[key] = merged.to(t_base.dtype)

    return merged_sd


def dare_merge(
    base_sd: Dict[str, torch.Tensor],
    finetuned_sd: Dict[str, torch.Tensor],
    alpha: float,
    drop_rate: float = 0.9,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    DARE: Drop And REscale.

    1. Compute task vector (delta = finetuned - base)
    2. Randomly drop (1-drop_rate) fraction of delta weights
    3. Rescale remaining weights by 1/(1-drop_rate) to preserve expected value
    4. Add rescaled delta to base with scale factor α

    Args:
        base_sd: Base model state dict
        finetuned_sd: Fine-tuned model state dict
        alpha: Scale factor for task vector
        drop_rate: Fraction of weights to drop (0.0-1.0), default 0.9
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)
    merged_sd = {}

    rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0

    for key in base_sd.keys():
        t_base = base_sd[key]
        t_ft = finetuned_sd[key]

        if not torch.is_floating_point(t_base):
            merged_sd[key] = t_base if alpha <= 0.5 else t_ft
            continue

        # Compute task vector
        delta = (t_ft - t_base).float()

        # Random drop mask (keep with probability 1-drop_rate)
        mask = torch.rand_like(delta) > drop_rate

        # Apply mask and rescale
        dropped_delta = torch.where(mask, delta * rescale, torch.zeros_like(delta))

        # Merge: base + α * dropped_delta
        merged = t_base.float() + alpha * dropped_delta
        merged_sd[key] = merged.to(t_base.dtype)

    return merged_sd


def task_arithmetic_merge(
    base_sd: Dict[str, torch.Tensor],
    finetuned_sd: Dict[str, torch.Tensor],
    alpha: float,
    trim_percentile: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Task Arithmetic: Simple task vector addition with optional outlier trimming.

    merged = base + α * task_vector

    Args:
        base_sd: Base model state dict
        finetuned_sd: Fine-tuned model state dict
        alpha: Scale factor for task vector
        trim_percentile: Percentile of extreme values to trim (0-50), default 0
    """
    merged_sd = {}

    for key in base_sd.keys():
        t_base = base_sd[key]
        t_ft = finetuned_sd[key]

        if not torch.is_floating_point(t_base):
            merged_sd[key] = t_base if alpha <= 0.5 else t_ft
            continue

        # Compute task vector
        delta = (t_ft - t_base).float()

        # Optional: trim outliers
        if trim_percentile > 0:
            flat_delta = delta.flatten()
            low = torch.quantile(flat_delta, trim_percentile / 100.0)
            high = torch.quantile(flat_delta, 1.0 - trim_percentile / 100.0)
            delta = torch.clamp(delta, low.item(), high.item())

        # Merge: base + α * delta
        merged = t_base.float() + alpha * delta
        merged_sd[key] = merged.to(t_base.dtype)

    return merged_sd


def merge_state_dicts(
    method: str,
    base_sd: Dict[str, torch.Tensor],
    finetuned_sd: Dict[str, torch.Tensor],
    alpha: float,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Dispatch to appropriate merge method.

    Args:
        method: One of 'linear', 'slerp', 'ties', 'dare', 'task_arithmetic'
        base_sd: Base model state dict
        finetuned_sd: Fine-tuned model state dict
        alpha: Interpolation weight (0=base, 1=finetuned)
        **kwargs: Method-specific arguments (density, drop_rate, etc.)
    """
    if method == "linear":
        return {k: linear_tensor(base_sd[k], finetuned_sd[k], alpha) for k in base_sd.keys()}

    elif method == "slerp":
        return {k: slerp_tensor(base_sd[k], finetuned_sd[k], alpha) for k in base_sd.keys()}

    elif method == "ties":
        density = kwargs.get("density", 0.5)
        return ties_merge(base_sd, finetuned_sd, alpha, density=density)

    elif method == "dare":
        drop_rate = kwargs.get("drop_rate", 0.9)
        seed = kwargs.get("seed", 42)
        return dare_merge(base_sd, finetuned_sd, alpha, drop_rate=drop_rate, seed=seed)

    elif method == "task_arithmetic":
        trim_percentile = kwargs.get("trim_percentile", 0.0)
        return task_arithmetic_merge(base_sd, finetuned_sd, alpha, trim_percentile=trim_percentile)

    else:
        raise ValueError(f"Unknown method: {method}. Supported: {SUPPORTED_METHODS}")


def main():
    parser = argparse.ArgumentParser(description="Create merged models using various interpolation methods.")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Hugging Face token (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Base model ID on Hugging Face Hub.",
    )
    parser.add_argument(
        "--grpo_model",
        type=str,
        default="lamm-mit/Qwen3-1.7B-Bioinspired-Thinking-12-05-2025",
        help="Fine-tuned model ID on Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_namespace",
        type=str,
        required=True,
        help="User or org namespace for the output models (e.g. 'lamm-mit').",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="slerp",
        choices=SUPPORTED_METHODS,
        help="Merge method: linear, slerp, ties, dare, task_arithmetic (default: slerp).",
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated list of fine-tuned weights α in [0,1].",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for loading & merging.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device map root ('cpu' or 'cuda').",
    )
    # Method-specific arguments
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="TIES: fraction of weights to keep by magnitude (default: 0.5).",
    )
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.9,
        help="DARE: fraction of weights to drop (default: 0.9).",
    )
    parser.add_argument(
        "--trim_percentile",
        type=float,
        default=0.0,
        help="Task Arithmetic: percentile of outliers to trim (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for DARE (default: 42).",
    )
    parser.add_argument(
        "--hub_public",
        action="store_true",
        help="Make Hub repo public (default: private).",
    )
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("Please provide --hf_token or set HF_TOKEN env var.")

    login(token=args.hf_token)

    fractions = parse_fractions(args.fractions)

    # Choose dtype
    if args.dtype == "float16":
        dt = torch.float16
    elif args.dtype == "bfloat16":
        dt = torch.bfloat16
    else:
        dt = torch.float32

    device_map = { "": args.device }

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dt,
        device_map=device_map,
        trust_remote_code=True,
    )
    base_sd = base_model.state_dict()

    print(f"Loading GRPO model: {args.grpo_model}")
    grpo_model = AutoModelForCausalLM.from_pretrained(
        args.grpo_model,
        torch_dtype=dt,
        device_map=device_map,
        trust_remote_code=True,
    )
    grpo_sd = grpo_model.state_dict()

    print(f"Loading tokenizer from GRPO model: {args.grpo_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.grpo_model, trust_remote_code=True)

    # Check keys match
    base_keys = set(base_sd.keys())
    grpo_keys = set(grpo_sd.keys())
    if base_keys != grpo_keys:
        diff1 = list(base_keys - grpo_keys)[:10]
        diff2 = list(grpo_keys - base_keys)[:10]
        raise ValueError(
            f"State dict keys differ.\n"
            f"  In base only:   {diff1}\n"
            f"  In grpo only:   {diff2}"
        )

    base_short = args.base_model.split("/")[-1]
    grpo_short = args.grpo_model.split("/")[-1]

    # Prepare method-specific kwargs
    merge_kwargs = {
        "density": args.density,
        "drop_rate": args.drop_rate,
        "trim_percentile": args.trim_percentile,
        "seed": args.seed,
    }

    print(f"\nMerge method: {args.method}")
    if args.method == "ties":
        print(f"  density={args.density}")
    elif args.method == "dare":
        print(f"  drop_rate={args.drop_rate}, seed={args.seed}")
    elif args.method == "task_arithmetic" and args.trim_percentile > 0:
        print(f"  trim_percentile={args.trim_percentile}")

    private = not args.hub_public

    for alpha in fractions:
        print(f"\n=== Building {args.method.upper()} model with alpha={alpha:.2f} ===")

        # Use the unified merge function
        merged_sd = merge_state_dicts(
            method=args.method,
            base_sd=base_sd,
            finetuned_sd=grpo_sd,
            alpha=alpha,
            **merge_kwargs,
        )

        base_model.load_state_dict(merged_sd, strict=True)

        alpha_str = f"{alpha:.2f}".replace(".", "p")
        local_dir = f"./{base_short}-{grpo_short}-{args.method}-{alpha_str}"
        hub_model_name = f"{base_short}-{grpo_short}-{args.method}-{alpha_str}"
        hub_repo_id = f"{args.hub_namespace}/{hub_model_name}"

        print(f"Saving merged model locally to: {local_dir}")
        os.makedirs(local_dir, exist_ok=True)
        base_model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)

        visibility = "public" if args.hub_public else "private"
        print(f"Pushing merged model to Hub: {hub_repo_id} ({visibility})")
        base_model.push_to_hub(hub_repo_id, private=private)
        tokenizer.push_to_hub(hub_repo_id, private=private)

    print(f"\nAll {args.method.upper()} models created and pushed successfully.")


if __name__ == "__main__":
    main()
