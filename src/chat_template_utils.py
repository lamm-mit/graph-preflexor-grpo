"""Chat template helpers shared by training and inference scripts."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional


def add_chat_template_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chat_template_enable_thinking",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "Pass enable_thinking to chat templates that support it. "
            "Use 'false' for Gemma 4 graph training so native thought channels do not replace graph tags."
        ),
    )


def parse_chat_template_enable_thinking(value: str) -> Optional[bool]:
    if value == "auto":
        return None
    return value == "true"


def apply_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    enable_thinking: Optional[bool] = None,
) -> str:
    kwargs: Dict[str, Any] = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)
