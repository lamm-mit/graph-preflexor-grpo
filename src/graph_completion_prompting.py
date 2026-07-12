"""Prompt construction for native-reasoning direct graph completion."""

from __future__ import annotations

from typing import Any, Optional

from chat_template_utils import apply_chat_template


GRAPH_COMPLETION_INSTRUCTION = """Infer the complete scientifically appropriate graph from the condition and incomplete or corrupted graph canvas below.

Rules:
- Preserve every [FIXED] node and edge exactly, including all payload fields.
- Make only changes permitted by the stated task and corruption mode.
- Add missing scientific content when needed.
- Remove spurious content when required.
- Correct wrong relations when required.
- Return the complete corrected graph, not a patch or a list of operations.
- Preserve every node and edge field present in the graph schema, including optional metadata.
- Put the final JSON object inside <answer>...</answer>.
- The JSON object must contain nodes and edges arrays.
- Emit nothing after </answer>.

You may reason using the model's native thinking channel. Only the final answer block is scored.
"""


def build_graph_completion_user_prompt(x0: str, *, mode: Optional[str] = None) -> str:
    mode_line = f"Corruption mode: {mode}\n\n" if mode else ""
    return f"{GRAPH_COMPLETION_INSTRUCTION}\n{mode_line}{str(x0).strip()}"


def apply_graph_completion_chat_template(
    tokenizer: Any,
    x0: str,
    *,
    mode: Optional[str] = None,
    enable_thinking: Optional[bool] = True,
) -> str:
    return apply_chat_template(
        tokenizer,
        [{"role": "user", "content": build_graph_completion_user_prompt(x0, mode=mode)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

