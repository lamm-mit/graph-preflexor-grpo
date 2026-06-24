#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Merge a PEFT/LoRA adapter into a base model and optionally upload it.

Examples:

    python src/merge_lora_adapter.py \
      --base_model lamm-mit/gemma4-e4b-sft-graph-10k-L_step_600 \
      --adapter lamm-mit/gemma4-e4b-grpo-graph-10k \
      --checkpoint 1300 \
      --tokenizer_model google/gemma-4-E4B-it \
      --processor_model google/gemma-4-E4B-it \
      --output_dir ./gemma4-e4b-grpo-graph-10k-merged_1300 \
      --hub_model_id lamm-mit/gemma4-e4b-grpo-graph-10k-merged_1300 \
      --hf_token "$HF_TOKEN"

    python src/merge_lora_adapter.py \
      --base_model google/gemma-4-E4B-it \
      --adapter lamm-mit/gemma4-e4b-sft-graph-10k-L \
      --adapter_commit 1bc18496096922b7a70b38d442ae8808fc6de7f3 \
      --output_dir ./gemma4-e4b-sft-graph-10k-L_step_600
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional


def resolve_token(explicit_token: Optional[str]) -> Optional[str]:
    return explicit_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def resolve_dtype(dtype_name: str):
    import torch

    normalized = dtype_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.float16
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"none", "default"}:
        return None
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def add_optional(kwargs: dict[str, Any], key: str, value: Optional[str]) -> None:
    if value:
        kwargs[key] = value


def normalize_checkpoint_subfolder(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip()
    if normalized.isdigit():
        return f"checkpoint-{normalized}"
    return normalized


def load_causal_lm(
    model_id: str,
    *,
    revision: Optional[str],
    device_map: Optional[str],
    dtype: Optional[torch.dtype],
    token: Optional[str],
    trust_remote_code: bool,
):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    add_optional(kwargs, "revision", revision)
    add_optional(kwargs, "token", token)
    if device_map and device_map.lower() != "none":
        kwargs["device_map"] = device_map
    if dtype is not None:
        kwargs["dtype"] = dtype

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc) or dtype is None:
            raise
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def try_load_tokenizer(
    source: str,
    *,
    revision: Optional[str],
    subfolder: Optional[str],
    token: Optional[str],
    trust_remote_code: bool,
):
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    add_optional(kwargs, "revision", revision)
    add_optional(kwargs, "subfolder", subfolder)
    add_optional(kwargs, "token", token)
    return AutoTokenizer.from_pretrained(source, **kwargs)


def load_tokenizer(args: argparse.Namespace, base_model: str, token: Optional[str]):
    if args.skip_tokenizer:
        return None

    candidates: list[tuple[str, Optional[str], Optional[str], str]] = []
    if args.tokenizer_model:
        candidates.append(
            (
                args.tokenizer_model,
                args.tokenizer_revision,
                args.tokenizer_subfolder,
                "explicit tokenizer",
            )
        )
    candidates.extend(
        [
            (args.adapter, args.adapter_revision, args.adapter_subfolder, "adapter with subfolder"),
            (args.adapter, args.adapter_revision, None, "adapter root"),
            (base_model, args.base_revision, None, "base model"),
        ]
    )

    errors = []
    seen: set[tuple[str, Optional[str], Optional[str]]] = set()
    for source, revision, subfolder, label in candidates:
        key = (source, revision, subfolder)
        if key in seen:
            continue
        seen.add(key)
        try:
            tokenizer = try_load_tokenizer(
                source,
                revision=revision,
                subfolder=subfolder,
                token=token,
                trust_remote_code=args.trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"Loaded tokenizer from {source} ({label})")
            return tokenizer
        except Exception as exc:  # noqa: BLE001 - keep trying candidates
            location = f"{source}"
            if revision:
                location += f" @ {revision}"
            if subfolder:
                location += f" / {subfolder}"
            errors.append(f"{location}: {type(exc).__name__}: {exc}")

    raise RuntimeError("Could not load tokenizer. Tried:\n" + "\n".join(errors))


def load_processor(args: argparse.Namespace, base_model: str, token: Optional[str]):
    from transformers import AutoProcessor

    if args.save_processor == "no":
        return None

    source = args.processor_model or args.tokenizer_model or base_model
    revision = args.processor_revision
    subfolder = args.processor_subfolder
    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    add_optional(kwargs, "revision", revision)
    add_optional(kwargs, "subfolder", subfolder)
    add_optional(kwargs, "token", token)

    try:
        processor = AutoProcessor.from_pretrained(source, **kwargs)
        print(f"Loaded processor from {source}")
        return processor
    except Exception as exc:  # noqa: BLE001 - optional in auto mode
        if args.save_processor == "yes":
            raise
        print(f"Processor not saved ({source} failed: {type(exc).__name__}: {exc})")
        return None


def load_peft_config(args: argparse.Namespace, token: Optional[str]) -> PeftConfig:
    from peft import PeftConfig

    kwargs: dict[str, Any] = {}
    add_optional(kwargs, "revision", args.adapter_revision)
    add_optional(kwargs, "subfolder", args.adapter_subfolder)
    add_optional(kwargs, "token", token)
    return PeftConfig.from_pretrained(args.adapter, **kwargs)


def load_adapter(model, args: argparse.Namespace, token: Optional[str]):
    from peft import PeftModel

    kwargs: dict[str, Any] = {}
    add_optional(kwargs, "revision", args.adapter_revision)
    add_optional(kwargs, "subfolder", args.adapter_subfolder)
    add_optional(kwargs, "token", token)
    return PeftModel.from_pretrained(model, args.adapter, **kwargs)


def resolve_safetensor_files(source: str, revision: Optional[str], token: Optional[str]) -> list[Path]:
    from huggingface_hub import snapshot_download

    source_path = Path(source)
    if source_path.exists():
        root = source_path
    else:
        root = Path(
            snapshot_download(
                repo_id=source,
                revision=revision,
                token=token,
                allow_patterns=["*.safetensors", "*.safetensors.index.json"],
            )
        )

    files = sorted(path for path in root.glob("*.safetensors") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in raw tensor source: {source}")
    return files


def add_missing_raw_tensors_from_source(
    state_dict: dict[str, Any],
    *,
    source: str,
    revision: Optional[str],
    token: Optional[str],
    max_missing_tensors: int,
) -> int:
    from safetensors import safe_open

    files = resolve_safetensor_files(source, revision, token)
    state_keys = set(state_dict)
    source_entries: list[tuple[Path, str]] = []

    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key not in state_keys:
                    source_entries.append((path, key))

    if not source_entries:
        print(f"No missing raw tensors to copy from {source}")
        return 0

    if len(source_entries) > max_missing_tensors:
        raise ValueError(
            f"Refusing to copy {len(source_entries)} missing tensors from {source}; "
            f"this is larger than --max_missing_raw_tensors={max_missing_tensors}. "
            "Check that --raw_tensor_source points to the matching original full model, "
            "or increase --max_missing_raw_tensors intentionally."
        )

    print(f"Copying {len(source_entries)} missing raw tensors from {source}")
    by_path: dict[Path, list[str]] = {}
    for path, key in source_entries:
        by_path.setdefault(path, []).append(key)

    for path, keys in by_path.items():
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in keys:
                state_dict[key] = handle.get_tensor(key).clone()

    return len(source_entries)


def add_missing_raw_tensors_from_candidates(
    state_dict: dict[str, Any],
    *,
    candidates: list[tuple[str, Optional[str], str]],
    token: Optional[str],
    max_missing_tensors: int,
) -> int:
    """Copy raw keys missing after HF merge from matching full-model checkpoints."""

    copied = 0
    seen: set[tuple[str, Optional[str]]] = set()
    for source, revision, label in candidates:
        if not source:
            continue
        key = (source, revision)
        if key in seen:
            continue
        seen.add(key)
        print(f"Checking missing raw tensors from {source} ({label})")
        copied += add_missing_raw_tensors_from_source(
            state_dict,
            source=source,
            revision=revision,
            token=token,
            max_missing_tensors=max_missing_tensors,
        )

    if copied == 0:
        print("No missing raw tensors were copied from candidate full-model sources.")
    return copied


def raw_tensor_candidates(args: argparse.Namespace, base_model: str) -> list[tuple[str, Optional[str], str]]:
    if args.raw_tensor_source:
        return [(args.raw_tensor_source, args.raw_tensor_revision, "explicit raw tensor source")]
    return [
        (base_model, args.base_revision, "base model"),
        (args.processor_model, args.processor_revision, "processor model"),
        (args.tokenizer_model, args.tokenizer_revision, "tokenizer model"),
    ]


def merge_adapter(args: argparse.Namespace):
    import torch
    from huggingface_hub import create_repo, upload_folder
    from transformers import AutoModelForCausalLM

    args.adapter_subfolder = normalize_checkpoint_subfolder(args.adapter_subfolder)
    token = resolve_token(args.hf_token)
    dtype = resolve_dtype(args.dtype)

    print("Inspecting adapter config:", args.adapter)
    peft_config = load_peft_config(args, token)
    inferred_base = getattr(peft_config, "base_model_name_or_path", None)
    base_model = args.base_model or inferred_base
    if not base_model:
        raise ValueError("--base_model is required because the adapter config does not define base_model_name_or_path")

    if args.base_model and inferred_base and args.base_model != inferred_base:
        print(f"Base override: {args.base_model}")
        print(f"Adapter config base_model_name_or_path: {inferred_base}")
    else:
        print(f"Base model: {base_model}")
    if args.adapter_revision:
        print(f"Adapter revision: {args.adapter_revision}")
    if args.adapter_subfolder:
        print(f"Adapter subfolder: {args.adapter_subfolder}")

    tokenizer = load_tokenizer(args, base_model, token)
    processor = load_processor(args, base_model, token)

    print("Loading base model...")
    model = load_causal_lm(
        base_model,
        revision=args.base_revision,
        device_map=args.device_map,
        dtype=dtype,
        token=token,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer is not None and args.resize_token_embeddings == "always":
        print(f"Resizing token embeddings before adapter load: {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    print("Loading adapter...")
    try:
        model = load_adapter(model, args, token)
    except RuntimeError as exc:
        if (
            args.resize_token_embeddings == "auto"
            and tokenizer is not None
            and "size mismatch" in str(exc)
        ):
            print(f"Adapter size mismatch; reloading base and resizing token embeddings to {len(tokenizer)}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = load_causal_lm(
                base_model,
                revision=args.base_revision,
                device_map=args.device_map,
                dtype=dtype,
                token=token,
                trust_remote_code=args.trust_remote_code,
            )
            model.resize_token_embeddings(len(tokenizer))
            model = load_adapter(model, args, token)
        else:
            raise

    print("Merging adapter into base model...")
    model = model.merge_and_unload()
    model.eval()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict[str, Any] = {
        "safe_serialization": args.safe_serialization,
        "max_shard_size": args.max_shard_size,
    }
    state_dict = None
    if args.mistralrs_compat_save:
        print("Materializing aliased/tied tensors with a cloned CPU state_dict before save.")
        print("This can use substantial host RAM and may produce larger safetensors files.")
        state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        add_missing_raw_tensors_from_candidates(
            state_dict,
            candidates=raw_tensor_candidates(args, base_model),
            token=token,
            max_missing_tensors=args.max_missing_raw_tensors,
        )
        save_kwargs["state_dict"] = state_dict

    print(f"Saving merged model to {out}")
    model.save_pretrained(out, **save_kwargs)
    del state_dict
    if tokenizer is not None:
        tokenizer.save_pretrained(out)
    if processor is not None:
        processor.save_pretrained(out)

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id is required with --push_to_hub")
        print(f"Pushing folder to Hub: {args.hub_model_id}")
        create_repo(
            args.hub_model_id,
            private=not args.hub_public,
            exist_ok=True,
            token=token,
        )
        upload_folder(
            repo_id=args.hub_model_id,
            folder_path=str(out),
            commit_message=args.commit_message,
            token=token,
        )

    if args.verify_load:
        print("Verifying saved model reload...")
        reloaded, info = AutoModelForCausalLM.from_pretrained(
            out,
            dtype=dtype,
            device_map=args.device_map if args.device_map.lower() != "none" else None,
            trust_remote_code=args.trust_remote_code,
            output_loading_info=True,
        )
        missing = list(info.get("missing_keys", []))
        unexpected = list(info.get("unexpected_keys", []))
        print(f"Reload missing_keys: {len(missing)}")
        print(f"Reload unexpected_keys: {len(unexpected)}")
        del reloaded

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a PEFT/LoRA adapter into a base model.")
    parser.add_argument("--adapter", required=True, help="Adapter path or Hub repo id.")
    parser.add_argument("--base_model", default=None, help="Base model path or Hub repo id. Overrides adapter config.")
    parser.add_argument("--output_dir", required=True, help="Local directory for the merged model.")
    parser.add_argument(
        "--adapter_revision",
        "--adapter_commit",
        dest="adapter_revision",
        default=None,
        help="Adapter Hub branch, tag, or commit SHA.",
    )
    parser.add_argument(
        "--adapter_subfolder",
        "--checkpoint",
        dest="adapter_subfolder",
        default=None,
        help="Adapter subfolder, e.g. checkpoint-1300. A bare number like 1300 becomes checkpoint-1300.",
    )
    parser.add_argument("--base_revision", default=None, help="Base model Hub branch, tag, or commit SHA.")
    parser.add_argument("--tokenizer_model", default=None, help="Tokenizer source. Defaults to adapter, then base.")
    parser.add_argument("--tokenizer_revision", default=None, help="Tokenizer Hub branch, tag, or commit SHA.")
    parser.add_argument("--tokenizer_subfolder", default=None, help="Tokenizer subfolder.")
    parser.add_argument("--processor_model", default=None, help="Processor source. Defaults to tokenizer source or base.")
    parser.add_argument("--processor_revision", default=None, help="Processor Hub branch, tag, or commit SHA.")
    parser.add_argument("--processor_subfolder", default=None, help="Processor subfolder.")
    parser.add_argument("--save_processor", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--skip_tokenizer", action="store_true", help="Save model only; do not load/save tokenizer.")
    parser.add_argument(
        "--resize_token_embeddings",
        choices=["auto", "always", "never"],
        default="auto",
        help="Resize base embeddings to tokenizer length before/retry adapter load.",
    )
    parser.add_argument("--dtype", default="auto", help="auto, bfloat16, float16, float32, or none.")
    parser.add_argument("--device_map", default="auto", help="Passed to from_pretrained. Use 'none' to omit.")
    parser.add_argument(
        "--max_shard_size",
        "--shard_size",
        dest="max_shard_size",
        default="4GB",
        help="Maximum shard size for save_pretrained, e.g. 4GB, 8GB, or 20GB. Default: 4GB.",
    )
    parser.add_argument(
        "--mistralrs_compat_save",
        "--materialize_tied_weights",
        dest="mistralrs_compat_save",
        action="store_true",
        help=(
            "Clone the merged model state_dict to CPU before save_pretrained and fill missing "
            "raw safetensors keys from matching full-model sources. Use for exports when a "
            "runtime such as mistral.rs expects materialized Gemma 4 tensor keys. By default "
            "the script tries base_model, then processor_model, then tokenizer_model. "
            "--raw_tensor_source can override that source list. Default: off."
        ),
    )
    parser.add_argument(
        "--raw_tensor_source",
        default=None,
        help=(
            "Optional full-model repo/path to copy missing raw safetensors keys from when "
            "--mistralrs_compat_save is enabled. If omitted, the script tries base_model, "
            "then processor_model, then tokenizer_model."
        ),
    )
    parser.add_argument("--raw_tensor_revision", default=None, help="Optional revision for --raw_tensor_source.")
    parser.add_argument(
        "--max_missing_raw_tensors",
        type=int,
        default=512,
        help="Safety cap for missing raw tensors copied during --mistralrs_compat_save. Default: 512.",
    )
    parser.add_argument("--no_safe_serialization", dest="safe_serialization", action="store_false")
    parser.set_defaults(safe_serialization=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument(
        "--push_to_hub",
        dest="push_to_hub",
        action="store_true",
        help="Upload merged output to Hub. This is the default.",
    )
    parser.add_argument(
        "--no_push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Only save locally; do not upload to Hub.",
    )
    parser.set_defaults(push_to_hub=True)
    parser.add_argument("--hub_model_id", default=None, help="Target Hub repo id for upload. Required by default.")
    parser.add_argument("--hub_public", action="store_true", help="Create/upload public repo. Default is private.")
    parser.add_argument("--hf_token", default=None, help="HF token. Defaults to HF_TOKEN env var.")
    parser.add_argument("--commit_message", default="Merge LoRA adapter")
    parser.add_argument("--verify_load", action="store_true", help="Reload saved model and report missing/unexpected keys.")
    return parser.parse_args()


def main() -> None:
    merge_adapter(parse_args())


if __name__ == "__main__":
    main()
