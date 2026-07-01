"""Shared config loader for the Graph-PRefLexOR plotting scripts.

Reads the wandb entity/project and per-model run IDs from a YAML file so the
run IDs live OUTSIDE the (publishable) plotting code. Point at a file with the
--config flag or the PLOT_CONFIG env var; default is ./plot_config.yaml.
"""
import os
import yaml

_LETTERS = "abcdefghijkl"


def load(path=None):
    path = path or os.environ.get("PLOT_CONFIG", "plot_config.yaml")
    if not os.path.exists(path):
        raise SystemExit(
            f"[plotcfg] config '{path}' not found.\n"
            f"  Copy plot_config.example.yaml -> plot_config.yaml and fill in your\n"
            f"  wandb run IDs, or set PLOT_CONFIG=/path/to/config.yaml."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def models(cfg, phase):
    """phase in {'orpo', 'grpo'}. Returns a list of dicts, one per model:
        {label, panel, ra, rb, color, continuation_mode, max_step_b}
    where 'panel' prepends an (a)/(b)/(c) tag. For GRPO, 'rb' is an optional
    continuation run; plotting scripts decide whether its steps are absolute or
    should be offset after 'ra'."""
    out = []
    for i, m in enumerate(cfg["models"]):
        if phase == "grpo":
            runs = list(m["grpo"])
            ra, rb = runs[0], (runs[1] if len(runs) > 1 else None)
        elif phase == "orpo":
            ra, rb = m["orpo"], None
        else:
            raise ValueError(f"unknown phase {phase!r}")
        out.append({
            "label": m["label"],
            "panel": f"({_LETTERS[i]}) {m['label']}",
            "ra": ra, "rb": rb, "color": m.get("color"),
            "continuation_mode": m.get(
                "continuation_mode",
                m.get("continuation", cfg.get("continuation_mode", cfg.get("continuation", "auto"))),
            ),
            "max_step_b": m.get("max_step_b", cfg.get("max_step_b")),
        })
    return out
