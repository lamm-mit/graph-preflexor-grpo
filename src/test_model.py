#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_model.py

Simple CLI to test model checkpoints with chat template + generation.

Usage:
    # Interactive mode (type your own prompts)
    python test_model.py --model ./orpo-graph_v1

    # Run built-in test prompts
    python test_model.py --model lamm-mit/orpo-graph_v1 --test

    # Single prompt from command line
    python test_model.py --model ./checkpoint --prompt "Explain how spider silk achieves high strength"

    # Adjust generation params
    python test_model.py --model ./checkpoint --test --max_tokens 2048 --temperature 0.7
"""

import argparse
import json
import os
from typing import Optional
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from chat_template_utils import add_chat_template_args, apply_chat_template, parse_chat_template_enable_thinking

# Built-in test prompts for graph reasoning
TEST_PROMPTS = [
    "What are the key mechanical properties of spider silk and how do they arise from its molecular structure?",
    "Explain the relationship between hierarchical structures and material toughness in biological materials.",
    "How do proteins fold and why is this important for their function?",
]


def get_config_vocab_size(config):
    """Return vocab size from standard or nested multimodal configs."""
    for attr in ("vocab_size",):
        value = getattr(config, attr, None)
        if value is not None:
            return value

    for nested_attr in ("text_config", "language_config", "llm_config"):
        nested = getattr(config, nested_attr, None)
        if nested is None:
            continue
        value = getattr(nested, "vocab_size", None)
        if value is not None:
            return value

    return None


def load_tokenizer_with_fallback(model_path: str, tokenizer_model: Optional[str] = None):
    """Load tokenizer, falling back when merged checkpoints contain stale tokenizer metadata."""
    candidates = [model_path]

    if tokenizer_model:
        candidates.append(tokenizer_model)
    else:
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config_source = getattr(cfg, "_name_or_path", None)
            if config_source and config_source not in candidates:
                candidates.append(config_source)
        except Exception as exc:
            print(f"Could not inspect config for tokenizer fallback: {exc}")

    errors = []
    for candidate in candidates:
        for label, kwargs in (
            ("default", {}),
            ("extra_special_tokens override", {"extra_special_tokens": {}}),
        ):
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True, **kwargs)
                if candidate != model_path or kwargs:
                    print(f"Loaded tokenizer from {candidate} ({label})")
                return tokenizer
            except Exception as exc:
                errors.append(f"{candidate} ({label}): {type(exc).__name__}: {exc}")
                print(f"Tokenizer load failed from {candidate} ({label}): {exc}")

    raise RuntimeError("Could not load tokenizer from any candidate:\n" + "\n".join(errors))


def load_model_and_tokenizer(model_path: str, device: str = "auto", tokenizer_model: Optional[str] = None):
    """Load model and tokenizer from local path or Hub (with PEFT adapter support)."""
    print(f"Loading model: {model_path}")

    # Determine device and dtype
    if device == "auto":
        if torch.cuda.is_available():
            device_map = "auto"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            device_map = {"": "mps"}
            dtype = torch.float16  # MPS works best with float16
            print("Using MPS (Apple Silicon)")
        else:
            device_map = {"": "cpu"}
            dtype = torch.float32
            print("Using CPU")
    elif device == "mps":
        device_map = {"": "mps"}
        dtype = torch.float16
    elif device == "cpu":
        device_map = {"": "cpu"}
        dtype = torch.float32
    else:
        device_map = device
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # Check if this is a PEFT adapter (has adapter_config.json)
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_peft_adapter = os.path.exists(adapter_config_path)

    # Also check on Hub if it's not a local path
    if not is_peft_adapter and not os.path.isdir(model_path):
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(model_path, "adapter_config.json")
            is_peft_adapter = True
        except Exception:
            pass

    if is_peft_adapter:
        # Load as PEFT adapter: load base model, load adapter, merge
        print("Detected PEFT adapter, loading base model + adapter...")
        try:
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
        except FileNotFoundError:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(model_path, "adapter_config.json")
            with open(config_path, "r") as f:
                adapter_config = json.load(f)

        base_model_name = adapter_config.get("base_model_name_or_path")
        print(f"  Base model: {base_model_name}")
        print(f"  Adapter: {model_path}")

        # Load tokenizer from adapter, falling back to base model if adapter metadata is stale.
        tokenizer = load_tokenizer_with_fallback(model_path, tokenizer_model or base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        base_vocab_size = get_config_vocab_size(model.config)
        adapter_tokenizer_size = len(tokenizer)
        print(f"  Base vocab: {base_vocab_size}, Adapter tokenizer: {adapter_tokenizer_size}")

        # Try loading adapter without resizing first (works if no special tokens were added)
        # If that fails, try resizing to match tokenizer
        try:
            model = PeftModel.from_pretrained(model, model_path)
            print("  Adapter loaded (no resize needed)")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"  Size mismatch, trying with resize to {adapter_tokenizer_size}...")
                # Reload base model and resize
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
                model.resize_token_embeddings(adapter_tokenizer_size)
                model = PeftModel.from_pretrained(model, model_path)
                print(f"  Adapter loaded after resize to {adapter_tokenizer_size}")
            else:
                raise

        model = model.merge_and_unload()
        print("  Adapter merged into base model")
    else:
        # Load as regular model
        tokenizer = load_tokenizer_with_fallback(model_path, tokenizer_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    model.eval()

    print(f"Model loaded: {model.config._name_or_path}")
    print(f"Device: {next(model.parameters()).device}, Dtype: {next(model.parameters()).dtype}")
    print(f"Vocab size: {len(tokenizer)}")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.9,
    do_sample: bool = True,
    chat_template_enable_thinking=None,
):
    """Generate a response using chat template."""

    # Format as chat
    messages = [{"role": "user", "content": prompt}]

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = apply_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=chat_template_enable_thinking,
        )
    else:
        # Fallback for tokenizers without chat template
        input_text = f"User: {prompt}\nAssistant:"

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)

    return response


def print_separator():
    print("\n" + "=" * 80 + "\n")


def run_test_prompts(model, tokenizer, max_new_tokens: int, temperature: float, chat_template_enable_thinking=None):
    """Run built-in test prompts."""
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[Test {i}/{len(TEST_PROMPTS)}]")
        print(f"PROMPT: {prompt}")
        print_separator()

        response = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            chat_template_enable_thinking=chat_template_enable_thinking,
        )

        print("RESPONSE:")
        print(response)
        print_separator()

        if i < len(TEST_PROMPTS):
            input("Press Enter for next test...")


def run_interactive(model, tokenizer, max_new_tokens: int, temperature: float, chat_template_enable_thinking=None):
    """Interactive mode - user types prompts."""
    print("\nInteractive mode. Type your prompt and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("YOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Exiting...")
            break

        print_separator()

        response = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            chat_template_enable_thinking=chat_template_enable_thinking,
        )

        print("MODEL:")
        print(response)
        print_separator()


def main():
    parser = argparse.ArgumentParser(description="Test model checkpoints with generation.")
    parser.add_argument("--model", type=str, required=True, help="Model path (local or Hub)")
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default=None,
        help="Optional tokenizer source. Useful when a merged checkpoint has stale tokenizer metadata.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run")
    parser.add_argument("--test", action="store_true", help="Run built-in test prompts")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max new tokens (default: 4096)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature (default: 0.3)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu (default: auto)")
    add_chat_template_args(parser)

    args = parser.parse_args()
    chat_template_enable_thinking = parse_chat_template_enable_thinking(args.chat_template_enable_thinking)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.tokenizer_model)
    print_separator()

    if args.prompt:
        # Single prompt mode
        print(f"PROMPT: {args.prompt}")
        print_separator()
        response = generate_response(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            chat_template_enable_thinking=chat_template_enable_thinking,
        )
        print("RESPONSE:")
        print(response)

    elif args.test:
        # Built-in test prompts
        run_test_prompts(model, tokenizer, args.max_tokens, args.temperature, chat_template_enable_thinking)

    else:
        # Interactive mode
        run_interactive(model, tokenizer, args.max_tokens, args.temperature, chat_template_enable_thinking)


if __name__ == "__main__":
    main()
