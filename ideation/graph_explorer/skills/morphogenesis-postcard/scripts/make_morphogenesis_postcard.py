#!/usr/bin/env python3
"""Generate a deterministic morphogenesis science-art postcard.

This script intentionally uses only the Python standard library so it can run
inside minimal skill containers. It simulates a compact Gray-Scott
reaction-diffusion field, renders it to a social-post PNG, and writes metadata
artifacts for the calling agent to report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import struct
import sys
import zlib
from pathlib import Path


PRESETS = {
    "labyrinth": {
        "feed": 0.0367,
        "kill": 0.0649,
        "du": 1.00,
        "dv": 0.50,
        "spots": 26,
        "radius": 7,
        "stripes": 10,
        "ridge": 0.24,
    },
    "spots": {
        "feed": 0.0250,
        "kill": 0.0600,
        "du": 1.00,
        "dv": 0.48,
        "spots": 42,
        "radius": 4,
        "stripes": 3,
        "ridge": 0.10,
    },
    "coral": {
        "feed": 0.0545,
        "kill": 0.0620,
        "du": 1.00,
        "dv": 0.50,
        "spots": 18,
        "radius": 8,
        "stripes": 8,
        "ridge": 0.19,
    },
    "veins": {
        "feed": 0.0305,
        "kill": 0.0575,
        "du": 1.00,
        "dv": 0.45,
        "spots": 22,
        "radius": 6,
        "stripes": 12,
        "ridge": 0.28,
    },
    "membranes": {
        "feed": 0.0430,
        "kill": 0.0610,
        "du": 1.00,
        "dv": 0.52,
        "spots": 14,
        "radius": 11,
        "stripes": 7,
        "ridge": 0.18,
    },
}


COMPOSITIONS = ("poster", "full-bleed", "specimen", "triptych", "field-guide")
MOTIFS = ("hybrid", "flow", "rings", "fibers", "fracture", "constellation", "none")
SYMMETRIES = ("none", "mirror-x", "mirror-y", "dihedral")
TITLE_MODES = ("panel", "caption", "none")


PROMPT_KEYWORDS = {
    "stress": {
        "stress",
        "strain",
        "fracture",
        "crack",
        "cracks",
        "rupture",
        "failure",
        "defect",
        "defects",
        "damage",
        "fault",
        "shear",
        "shock",
        "load",
    },
    "wave": {
        "wave",
        "waves",
        "interference",
        "phase",
        "oscillation",
        "oscillations",
        "resonance",
        "phonon",
        "vibration",
        "frequency",
        "ripple",
        "ripples",
    },
    "bio": {
        "bio",
        "biological",
        "morphogenesis",
        "cell",
        "cells",
        "cellular",
        "tissue",
        "growth",
        "bone",
        "collagen",
        "protein",
        "membrane",
        "enzyme",
        "organism",
    },
    "network": {
        "graph",
        "network",
        "node",
        "nodes",
        "edge",
        "edges",
        "bridge",
        "bridges",
        "module",
        "modules",
        "link",
        "links",
        "path",
        "paths",
        "community",
        "knowledge",
    },
    "quantum": {
        "quantum",
        "plasma",
        "neon",
        "electron",
        "electrons",
        "field",
        "fields",
        "charge",
        "electric",
        "magnetic",
        "glowing",
        "luminous",
    },
    "thermal": {
        "thermal",
        "heat",
        "fire",
        "magma",
        "flame",
        "temperature",
        "energy",
        "hot",
    },
    "quiet": {
        "quiet",
        "minimal",
        "clean",
        "museum",
        "specimen",
        "archive",
        "archival",
        "white",
        "ice",
        "clinical",
        "elegant",
    },
    "crystal": {
        "crystal",
        "crystals",
        "lattice",
        "periodic",
        "symmetry",
        "symmetric",
        "moire",
        "snow",
        "mineral",
        "ordered",
    },
    "branch": {
        "branch",
        "branches",
        "branching",
        "vascular",
        "tree",
        "dendrite",
        "dendritic",
        "root",
        "fiber",
        "fibers",
        "fibre",
        "fibres",
        "hierarchical",
        "hierarchy",
    },
    "art": {
        "postcard",
        "poster",
        "album",
        "cover",
        "social",
        "garden",
        "aesthetic",
        "art",
        "beautiful",
    },
}


PROMPT_STYLE_FLAGS = {
    "--seed": "seed",
    "--preset": "preset",
    "--palette": "palette",
    "--composition": "composition",
    "--motif": "motif",
    "--symmetry": "symmetry",
    "--title-mode": "title_mode",
    "--energy": "energy",
    "--contrast": "contrast",
    "--grain": "grain",
    "--ink-density": "ink_density",
    "--accent-count": "accent_count",
}


PALETTES = {
    "magma-cyan": {
        "bg0": (7, 10, 16),
        "bg1": (19, 15, 29),
        "stops": (
            (11, 15, 28),
            (71, 20, 62),
            (176, 47, 82),
            (250, 145, 84),
            (115, 240, 231),
        ),
        "ink": (235, 248, 255),
        "muted": (157, 190, 207),
        "accent": (125, 242, 232),
        "panel": (3, 7, 15),
    },
    "biofilm": {
        "bg0": (8, 17, 13),
        "bg1": (15, 32, 22),
        "stops": (
            (8, 25, 19),
            (21, 78, 49),
            (90, 134, 61),
            (197, 176, 87),
            (238, 226, 154),
        ),
        "ink": (243, 247, 221),
        "muted": (186, 204, 159),
        "accent": (215, 255, 120),
        "panel": (6, 14, 10),
    },
    "noir-neon": {
        "bg0": (4, 5, 10),
        "bg1": (17, 9, 30),
        "stops": (
            (3, 5, 12),
            (36, 16, 76),
            (105, 34, 148),
            (17, 178, 216),
            (242, 251, 255),
        ),
        "ink": (241, 245, 255),
        "muted": (183, 192, 216),
        "accent": (22, 217, 244),
        "panel": (5, 6, 14),
    },
    "ice": {
        "bg0": (225, 236, 241),
        "bg1": (177, 206, 219),
        "stops": (
            (238, 247, 249),
            (190, 222, 232),
            (113, 177, 202),
            (39, 93, 128),
            (8, 29, 54),
        ),
        "ink": (16, 32, 51),
        "muted": (56, 86, 109),
        "accent": (28, 111, 163),
        "panel": (239, 247, 249),
    },
    "graphite-fire": {
        "bg0": (11, 12, 13),
        "bg1": (31, 28, 24),
        "stops": (
            (15, 16, 18),
            (51, 50, 46),
            (118, 76, 47),
            (215, 104, 47),
            (255, 210, 123),
        ),
        "ink": (248, 239, 226),
        "muted": (202, 183, 160),
        "accent": (255, 156, 72),
        "panel": (12, 10, 9),
    },
}


FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    " ": ("000", "000", "000", "000", "000", "000", "000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    ":": ("000", "010", "010", "000", "010", "010", "000"),
    ".": ("000", "000", "000", "000", "000", "010", "010"),
    ",": ("000", "000", "000", "000", "010", "010", "100"),
    "'": ("010", "010", "100", "000", "000", "000", "000"),
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def prompt_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def prompt_choice(options: tuple[str, ...], prompt_hash: int, salt: str) -> str:
    return options[hash_string(f"{prompt_hash}:{salt}") % len(options)]


def prompt_weight(scores: dict[str, int], name: str) -> float:
    return clamp(scores.get(name, 0) / 3.0, 0.0, 1.0)


def explicit_style_args(argv: list[str]) -> set[str]:
    found: set[str] = set()
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in PROMPT_STYLE_FLAGS:
            found.add(PROMPT_STYLE_FLAGS[key])
    return found


def apply_prompt_style(args: argparse.Namespace) -> None:
    prompt = str(args.prompt or "").strip()
    prompt_source = prompt or f"{args.title} {args.subtitle}".strip()
    if not prompt_source:
        args.prompt_profile = {}
        return

    explicit = getattr(args, "_explicit_style_args", set())

    def set_auto(name: str, value: object) -> None:
        if name not in explicit:
            setattr(args, name, value)

    tokens = prompt_tokens(prompt_source)
    token_set = set(tokens)
    scores = {
        name: sum(1 for token in tokens if token in keywords)
        for name, keywords in PROMPT_KEYWORDS.items()
    }
    prompt_hash = hash_string(prompt_source)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    dominant = [name for name, score in ranked if score > 0][:4]
    if not dominant:
        fallback = ("wave", "bio", "network", "crystal", "branch", "thermal")
        dominant = [fallback[prompt_hash % len(fallback)]]

    weights = {name: prompt_weight(scores, name) for name in PROMPT_KEYWORDS}
    primary = dominant[0]
    secondary = dominant[1] if len(dominant) > 1 else ""

    if primary == "stress":
        set_auto("preset", "veins")
        set_auto("palette", "graphite-fire" if weights["thermal"] < 0.45 else "magma-cyan")
        set_auto("motif", "fracture")
        set_auto("composition", "poster")
    elif primary == "wave":
        set_auto("preset", "labyrinth")
        set_auto("palette", "noir-neon" if weights["quantum"] > 0.20 else "magma-cyan")
        set_auto("motif", prompt_choice(("flow", "rings", "hybrid"), prompt_hash, "wave-motif"))
        set_auto("composition", "full-bleed" if weights["art"] > 0.0 else "poster")
    elif primary == "bio":
        set_auto("preset", "coral" if weights["branch"] >= 0.25 else "membranes")
        set_auto("palette", "biofilm")
        set_auto("motif", "fibers" if weights["branch"] > 0.0 else "rings")
        set_auto("composition", "field-guide" if weights["quiet"] > 0.0 else "poster")
    elif primary == "network":
        set_auto("preset", "labyrinth")
        set_auto("palette", "magma-cyan" if weights["quantum"] < 0.35 else "noir-neon")
        set_auto("motif", "constellation")
        set_auto("composition", prompt_choice(("field-guide", "triptych", "poster"), prompt_hash, "network-composition"))
    elif primary == "quantum":
        set_auto("preset", "spots")
        set_auto("palette", "noir-neon")
        set_auto("motif", prompt_choice(("rings", "constellation", "flow"), prompt_hash, "quantum-motif"))
        set_auto("composition", "full-bleed")
    elif primary == "thermal":
        set_auto("preset", "veins")
        set_auto("palette", prompt_choice(("graphite-fire", "magma-cyan"), prompt_hash, "thermal-palette"))
        set_auto("motif", "hybrid")
        set_auto("composition", "poster")
    elif primary == "quiet":
        set_auto("preset", "membranes")
        set_auto("palette", "ice")
        set_auto("motif", "constellation" if weights["network"] > 0.0 else "rings")
        set_auto("composition", "specimen")
        set_auto("title_mode", "caption")
    elif primary == "crystal":
        set_auto("preset", "spots")
        set_auto("palette", "ice")
        set_auto("motif", "rings")
        set_auto("composition", "specimen")
        set_auto("symmetry", prompt_choice(("dihedral", "mirror-x", "mirror-y"), prompt_hash, "crystal-symmetry"))
    elif primary == "branch":
        set_auto("preset", "coral")
        set_auto("palette", "biofilm")
        set_auto("motif", "fibers")
        set_auto("composition", "poster")
    else:
        set_auto("preset", prompt_choice(tuple(sorted(PRESETS)), prompt_hash, "preset"))
        set_auto("palette", prompt_choice(tuple(sorted(PALETTES)), prompt_hash, "palette"))
        set_auto("motif", prompt_choice(MOTIFS[:-1], prompt_hash, "motif"))
        set_auto("composition", prompt_choice(COMPOSITIONS, prompt_hash, "composition"))

    if secondary == "network" and args.motif != "constellation":
        set_auto("motif", "hybrid")
    if (secondary == "crystal" or weights["crystal"] > 0.20) and args.symmetry == "none":
        set_auto("symmetry", prompt_choice(("mirror-x", "mirror-y", "dihedral"), prompt_hash, "secondary-symmetry"))
    if weights["quiet"] > 0.25:
        set_auto("composition", "specimen")
        set_auto("palette", "ice")
        set_auto("title_mode", "caption")
    if weights["art"] > 0.30 and args.composition == "specimen":
        set_auto("composition", "poster")
    if "triptych" in token_set or ({"three", "panel"} <= token_set) or ({"three", "panels"} <= token_set):
        set_auto("composition", "triptych")
    elif "field" in token_set and "guide" in token_set:
        set_auto("composition", "field-guide")
    elif "specimen" in token_set or "museum" in token_set:
        set_auto("composition", "specimen")
    elif "full" in token_set and "bleed" in token_set:
        set_auto("composition", "full-bleed")

    hash_float = (prompt_hash % 1000) / 999.0
    set_auto(
        "energy",
        clamp(
            0.54
            + 0.15 * weights["stress"]
            + 0.12 * weights["wave"]
            + 0.10 * weights["thermal"]
            + 0.08 * weights["quantum"]
            - 0.08 * weights["quiet"]
            + 0.10 * hash_float,
            0.42,
            0.98,
        ),
    )
    set_auto(
        "contrast",
        clamp(
            0.92
            + 0.30 * weights["stress"]
            + 0.22 * weights["quantum"]
            + 0.18 * weights["thermal"]
            + 0.12 * weights["crystal"]
            - 0.10 * weights["quiet"]
            + 0.18 * hash_float,
            0.72,
            1.78,
        ),
    )
    set_auto(
        "grain",
        clamp(
            0.035
            + 0.06 * weights["stress"]
            + 0.06 * weights["thermal"]
            + 0.04 * weights["art"]
            - 0.025 * weights["quiet"]
            + 0.055 * hash_float,
            0.01,
            0.22,
        ),
    )
    set_auto(
        "ink_density",
        clamp(
            0.46
            + 0.16 * weights["network"]
            + 0.14 * weights["branch"]
            + 0.12 * weights["wave"]
            + 0.09 * weights["bio"]
            + 0.08 * weights["stress"],
            0.38,
            0.94,
        ),
    )
    set_auto(
        "accent_count",
        int(
            6
            + 8 * weights["network"]
            + 7 * weights["branch"]
            + 6 * weights["wave"]
            + 5 * weights["stress"]
            + 5 * hash_float
        ),
    )
    if "seed" not in explicit and str(args.seed) == "42":
        args.seed = f"prompt-{prompt_hash:08x}"

    args.prompt_profile = {
        "prompt": prompt_source,
        "hash": f"{prompt_hash:08x}",
        "tokens": tokens[:80],
        "scores": scores,
        "dominant": dominant,
        "weights": weights,
        "style": {
            "preset": args.preset,
            "palette": args.palette,
            "composition": args.composition,
            "motif": args.motif,
            "symmetry": args.symmetry,
            "title_mode": args.title_mode,
            "energy": args.energy,
            "contrast": args.contrast,
            "grain": args.grain,
            "ink_density": args.ink_density,
            "accent_count": args.accent_count,
            "seed": args.seed,
        },
        "contains": {
            "prompt_tokens": len(tokens),
            "unique_prompt_tokens": len(token_set),
        },
    }


def load_rule_module(path: str | None):
    if not path:
        return None
    rule_path = Path(path)
    if not rule_path.exists():
        raise FileNotFoundError(f"Custom rule file not found: {rule_path}")
    spec = importlib.util.spec_from_file_location("morphogenesis_agent_rules", rule_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load custom rule file: {rule_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_rule_config(args: argparse.Namespace, module) -> None:
    if module is None or not hasattr(module, "configure"):
        return
    updates = module.configure(args)
    if updates is None:
        return
    if not isinstance(updates, dict):
        raise TypeError("configure(args) must return a dict or None")
    explicit = getattr(args, "_explicit_style_args", set())
    for key, value in updates.items():
        arg_name = str(key).replace("-", "_")
        if arg_name in explicit:
            continue
        setattr(args, arg_name, value)


def normalize_args(args: argparse.Namespace) -> None:
    if not hasattr(args, "prompt_profile"):
        args.prompt_profile = {}
    if args.preset not in PRESETS:
        raise ValueError(f"Invalid preset from custom rules: {args.preset}")
    if args.palette not in PALETTES:
        raise ValueError(f"Invalid palette from custom rules: {args.palette}")
    if args.composition not in COMPOSITIONS:
        raise ValueError(f"Invalid composition from custom rules: {args.composition}")
    if args.motif not in MOTIFS:
        raise ValueError(f"Invalid motif from custom rules: {args.motif}")
    if args.symmetry not in SYMMETRIES:
        raise ValueError(f"Invalid symmetry from custom rules: {args.symmetry}")
    if args.title_mode not in TITLE_MODES:
        raise ValueError(f"Invalid title mode from custom rules: {args.title_mode}")
    args.steps = max(60, min(520, int(args.steps)))
    args.field_size = max(96, min(220, int(args.field_size)))
    args.size = max(640, min(1600, int(args.size)))
    args.energy = clamp(float(args.energy), 0.0, 1.0)
    args.contrast = clamp(float(args.contrast), 0.35, 2.5)
    args.grain = clamp(float(args.grain), 0.0, 1.0)
    args.ink_density = clamp(float(args.ink_density), 0.0, 1.0)
    args.accent_count = max(0, min(80, int(args.accent_count)))
    args.prompt_strength = clamp(float(args.prompt_strength), 0.0, 1.0)
    if args.prompt_profile:
        args.prompt_profile["style"] = {
            "preset": args.preset,
            "palette": args.palette,
            "composition": args.composition,
            "motif": args.motif,
            "symmetry": args.symmetry,
            "title_mode": args.title_mode,
            "energy": args.energy,
            "contrast": args.contrast,
            "grain": args.grain,
            "ink_density": args.ink_density,
            "accent_count": args.accent_count,
            "seed": args.seed,
        }


def hash_string(value: object) -> int:
    h = 2166136261
    for ch in str(value):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h or 1


class Rng:
    def __init__(self, seed: object) -> None:
        self.state = hash_string(seed)

    def random(self) -> float:
        self.state ^= (self.state << 13) & 0xFFFFFFFF
        self.state ^= self.state >> 17
        self.state ^= (self.state << 5) & 0xFFFFFFFF
        self.state &= 0xFFFFFFFF
        return self.state / 4294967296.0


def mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def palette_color(stops: tuple[tuple[int, int, int], ...], t: float) -> tuple[int, int, int]:
    t = clamp(t, 0.0, 1.0)
    x = t * (len(stops) - 1)
    i = min(len(stops) - 2, int(x))
    return mix(stops[i], stops[i + 1], x - i)


def coerce_color(value: object, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return fallback
    if isinstance(value, str):
        cleaned = value.strip().lstrip("#")
        if len(cleaned) == 6:
            try:
                return (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))
            except ValueError:
                return fallback
        return fallback
    try:
        seq = list(value)  # type: ignore[arg-type]
    except TypeError:
        return fallback
    if len(seq) < 3:
        return fallback
    return (
        max(0, min(255, int(seq[0]))),
        max(0, min(255, int(seq[1]))),
        max(0, min(255, int(seq[2]))),
    )


def alpha_blend(
    image: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if x < 0 or y < 0 or x >= width or y >= height or alpha <= 0:
        return
    idx = (y * width + x) * 3
    inv = 1.0 - clamp(alpha, 0.0, 1.0)
    image[idx] = int(image[idx] * inv + color[0] * alpha)
    image[idx + 1] = int(image[idx + 1] * inv + color[1] * alpha)
    image[idx + 2] = int(image[idx + 2] * inv + color[2] * alpha)


def fill_rect(
    image: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x1)
    y1 = min(height, y1)
    for y in range(y0, y1):
        base = y * width * 3
        for x in range(x0, x1):
            idx = base + x * 3
            inv = 1.0 - alpha
            image[idx] = int(image[idx] * inv + color[0] * alpha)
            image[idx + 1] = int(image[idx + 1] * inv + color[1] * alpha)
            image[idx + 2] = int(image[idx + 2] * inv + color[2] * alpha)


def draw_circle(
    image: bytearray,
    width: int,
    height: int,
    cx: float,
    cy: float,
    radius: float,
    color: tuple[int, int, int],
    alpha: float,
    ring: bool = False,
) -> None:
    pad = max(2, int(radius * 0.18))
    x0 = max(0, int(cx - radius - pad))
    x1 = min(width - 1, int(cx + radius + pad))
    y0 = max(0, int(cy - radius - pad))
    y1 = min(height - 1, int(cy + radius + pad))
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            dist = math.hypot(x - cx, y - cy)
            if ring:
                edge = abs(dist - radius)
                if edge < pad:
                    alpha_blend(image, width, height, x, y, color, alpha * (1 - edge / pad))
            elif dist < radius:
                alpha_blend(image, width, height, x, y, color, alpha * (1 - dist / radius) ** 0.55)


def draw_line(
    image: bytearray,
    width: int,
    height: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    alpha: float,
    thickness: int = 1,
) -> None:
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1))
    for i in range(steps + 1):
        t = i / steps
        x = int(x0 + (x1 - x0) * t)
        y = int(y0 + (y1 - y0) * t)
        for yy in range(y - thickness, y + thickness + 1):
            for xx in range(x - thickness, x + thickness + 1):
                if (xx - x) * (xx - x) + (yy - y) * (yy - y) <= thickness * thickness:
                    alpha_blend(image, width, height, xx, yy, color, alpha)


def draw_frame(
    image: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    alpha: float,
    thickness: int = 2,
) -> None:
    fill_rect(image, width, height, x0, y0, x1, y0 + thickness, color, alpha)
    fill_rect(image, width, height, x0, y1 - thickness, x1, y1, color, alpha)
    fill_rect(image, width, height, x0, y0, x0 + thickness, y1, color, alpha)
    fill_rect(image, width, height, x1 - thickness, y0, x1, y1, color, alpha)


def text_width(text: str, scale: int) -> int:
    total = 0
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        total += (len(glyph[0]) + 1) * scale
    return max(0, total - scale)


def draw_text(
    image: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    scale: int,
    alpha: float = 1.0,
) -> int:
    cursor = x
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, cell in enumerate(row):
                if cell == "1":
                    fill_rect(
                        image,
                        width,
                        height,
                        cursor + gx * scale,
                        y + gy * scale,
                        cursor + (gx + 1) * scale,
                        y + (gy + 1) * scale,
                        color,
                        alpha,
                    )
        cursor += (len(glyph[0]) + 1) * scale
    return cursor


def fit_text(text: str, max_width: int, start_scale: int, min_scale: int) -> int:
    scale = start_scale
    while scale > min_scale and text_width(text, scale) > max_width:
        scale -= 1
    return scale


def trim_text_to_width(text: str, max_width: int, scale: int) -> str:
    if text_width(text, scale) <= max_width:
        return text
    suffix = "..."
    trimmed = text
    while trimmed and text_width(trimmed + suffix, scale) > max_width:
        trimmed = trimmed[:-1]
    return (trimmed.rstrip() + suffix) if trimmed else suffix


def lap(field: list[float], x: int, y: int, n: int) -> float:
    idx = y * n + x
    return (
        -field[idx]
        + 0.20
        * (
            field[y * n + x - 1]
            + field[y * n + x + 1]
            + field[(y - 1) * n + x]
            + field[(y + 1) * n + x]
        )
        + 0.05
        * (
            field[(y - 1) * n + x - 1]
            + field[(y - 1) * n + x + 1]
            + field[(y + 1) * n + x - 1]
            + field[(y + 1) * n + x + 1]
        )
    )


def seed_field(n: int, args: argparse.Namespace, preset: dict[str, float]) -> tuple[list[float], list[float]]:
    rng = Rng(f"{args.seed}:{args.preset}")
    total = n * n
    u = [1.0] * total
    v = [0.0] * total
    cx = n * 0.5
    cy = n * 0.5

    def stamp(sx: float, sy: float, radius: float, strength: float) -> None:
        r2 = radius * radius
        for yy in range(max(2, int(sy - radius)), min(n - 2, int(sy + radius) + 1)):
            for xx in range(max(2, int(sx - radius)), min(n - 2, int(sx + radius) + 1)):
                d2 = (xx - sx) * (xx - sx) + (yy - sy) * (yy - sy)
                if d2 <= r2:
                    w = 1.0 - d2 / r2
                    idx = yy * n + xx
                    u[idx] = min(u[idx], 0.40 + 0.18 * rng.random())
                    v[idx] = max(v[idx], strength * (0.45 + 0.55 * w))

    stamp(cx, cy, n * 0.075, 0.95)
    for _ in range(int(preset["spots"])):
        angle = rng.random() * math.tau
        radius = n * (0.10 + 0.40 * rng.random())
        sx = cx + math.cos(angle) * radius + (rng.random() - 0.5) * n * 0.12
        sy = cy + math.sin(angle) * radius + (rng.random() - 0.5) * n * 0.12
        stamp(sx, sy, preset["radius"] * (0.75 + rng.random() * 0.90), 0.78)

    for stripe in range(int(preset["stripes"])):
        angle = rng.random() * math.pi
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        phase = rng.random() * math.tau
        freq = 0.040 + rng.random() * 0.040
        bend = 0.004 + rng.random() * 0.010
        threshold = 0.08 if args.preset != "membranes" else 0.12
        for y in range(2, n - 2):
            dy = y - cy
            for x in range(2, n - 2):
                dx = x - cx
                projected = dx * cos_a + dy * sin_a
                across = -dx * sin_a + dy * cos_a
                wave = math.sin(projected * freq + math.sin(across * bend + phase) * 1.9 + phase)
                envelope = 1.0 - clamp(math.hypot(dx, dy) / (n * 0.66), 0.0, 1.0)
                if abs(wave) < threshold * (0.65 + envelope) and rng.random() < 0.32 + envelope * 0.30:
                    idx = y * n + x
                    u[idx] = min(u[idx], 0.46 + 0.10 * rng.random())
                    v[idx] = max(v[idx], 0.22 + 0.44 * envelope + 0.08 * rng.random())

    prompt_profile = getattr(args, "prompt_profile", {}) or {}
    if prompt_profile:
        weights = prompt_profile.get("weights", {})
        prompt_rng = Rng(f"prompt-field:{args.seed}:{prompt_profile.get('hash', '')}")
        prompt_strength = clamp(float(getattr(args, "prompt_strength", 0.72)), 0.0, 1.0)

        def w(name: str) -> float:
            return clamp(float(weights.get(name, 0.0)), 0.0, 1.0)

        wave_force = max(w("wave"), w("quantum") * 0.65)
        if wave_force > 0:
            angle = prompt_rng.random() * math.pi
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            phase = prompt_rng.random() * math.tau
            freq = 0.070 + 0.070 * prompt_rng.random()
            for y in range(2, n - 2):
                dy = y - cy
                for x in range(2, n - 2):
                    dx = x - cx
                    projected = dx * cos_a + dy * sin_a
                    across = -dx * sin_a + dy * cos_a
                    interference = math.sin(projected * freq + phase) + 0.55 * math.sin(across * freq * 1.7 - phase)
                    envelope = 1.0 - clamp(math.hypot(dx, dy) / (n * 0.72), 0.0, 1.0)
                    if abs(interference) < 0.18 * (0.65 + wave_force) and prompt_rng.random() < 0.16 * prompt_strength:
                        idx = y * n + x
                        u[idx] = min(u[idx], 0.42 + 0.10 * prompt_rng.random())
                        v[idx] = max(v[idx], 0.18 + 0.54 * envelope * wave_force)

        branch_force = max(w("branch"), w("bio") * 0.70)
        if branch_force > 0:
            branches = max(3, int(4 + branch_force * 10 + prompt_strength * 4))
            for _ in range(branches):
                x = cx + (prompt_rng.random() - 0.5) * n * 0.25
                y = cy + (prompt_rng.random() - 0.5) * n * 0.25
                angle = prompt_rng.random() * math.tau
                for step in range(12):
                    stamp(x, y, 2.0 + branch_force * 3.0, 0.42 + branch_force * 0.35)
                    angle += (prompt_rng.random() - 0.5) * (0.75 + 0.55 * branch_force)
                    length = n * (0.022 + 0.025 * prompt_rng.random()) * (0.7 + branch_force)
                    x += math.cos(angle) * length
                    y += math.sin(angle) * length
                    if not (4 < x < n - 4 and 4 < y < n - 4):
                        break
                    if prompt_rng.random() < 0.22 * branch_force:
                        side = angle + (prompt_rng.random() - 0.5) * 1.9
                        stamp(x + math.cos(side) * length, y + math.sin(side) * length, 2.0, 0.48)

        stress_force = w("stress")
        if stress_force > 0:
            cracks = max(2, int(3 + stress_force * 9))
            for _ in range(cracks):
                x = n * (0.10 + 0.80 * prompt_rng.random())
                y = n * (0.10 + 0.80 * prompt_rng.random())
                angle = -math.pi / 2 + (prompt_rng.random() - 0.5) * 1.8
                for _ in range(16):
                    stamp(x, y, 1.8 + 2.6 * stress_force, 0.55 + 0.34 * stress_force)
                    length = n * (0.018 + 0.030 * prompt_rng.random())
                    x += math.cos(angle) * length
                    y += math.sin(angle) * length
                    angle += (prompt_rng.random() - 0.5) * 0.95
                    if not (3 < x < n - 3 and 3 < y < n - 3):
                        break

        network_force = w("network")
        if network_force > 0:
            points = []
            count = max(5, int(6 + network_force * 12))
            for _ in range(count):
                px = n * (0.12 + 0.76 * prompt_rng.random())
                py = n * (0.12 + 0.76 * prompt_rng.random())
                points.append((px, py))
                stamp(px, py, 2.5 + 3.2 * network_force, 0.50 + 0.28 * network_force)
            for i, (x0, y0) in enumerate(points):
                x1, y1 = points[(i + 1 + int(prompt_rng.random() * max(1, count - 1))) % count]
                segments = 10
                for segment in range(segments + 1):
                    t = segment / segments
                    x = x0 + (x1 - x0) * t
                    y = y0 + (y1 - y0) * t
                    if prompt_rng.random() < 0.35 * network_force:
                        stamp(x, y, 1.2 + 1.4 * network_force, 0.26 + 0.18 * network_force)

        crystal_force = max(w("crystal"), w("quiet") * 0.60)
        if crystal_force > 0:
            spacing = max(8, int(n * (0.14 - 0.045 * crystal_force)))
            offset_x = int(prompt_rng.random() * spacing)
            offset_y = int(prompt_rng.random() * spacing)
            for y in range(4 + offset_y, n - 4, spacing):
                for x in range(4 + offset_x, n - 4, spacing):
                    jitter = spacing * 0.12 * (1.0 - crystal_force)
                    stamp(
                        x + (prompt_rng.random() - 0.5) * jitter,
                        y + (prompt_rng.random() - 0.5) * jitter,
                        2.5 + 3.0 * crystal_force,
                        0.38 + 0.38 * crystal_force,
                    )

    for i in range(total):
        noise = (rng.random() - 0.5) * 0.030
        u[i] = clamp(u[i] + noise, 0.0, 1.0)
        v[i] = clamp(v[i] - noise, 0.0, 1.0)

    module = getattr(args, "rule_module", None)
    if module is not None and hasattr(module, "seed_rule"):
        result = module.seed_rule(u, v, n, args)
        if result is not None:
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("seed_rule(u, v, n, args) must return (u, v) or None")
            u, v = result
        if len(u) != total or len(v) != total:
            raise ValueError("seed_rule produced fields with the wrong length")
        u = [clamp(float(value), 0.0, 1.0) for value in u]
        v = [clamp(float(value), 0.0, 1.0) for value in v]
    return u, v


def simulate(args: argparse.Namespace) -> tuple[int, list[float], list[float]]:
    preset = PRESETS[args.preset]
    n = args.field_size
    u, v = seed_field(n, args, preset)
    u2 = u[:]
    v2 = v[:]
    seed_phase = hash_string(args.seed) * 0.00001

    for step in range(args.steps):
        wave = math.sin(step / max(1, args.steps) * math.tau + seed_phase) * 0.0014
        for y in range(1, n - 1):
            for x in range(1, n - 1):
                idx = y * n + x
                uvv = u[idx] * v[idx] * v[idx]
                radial = math.hypot(x - n * 0.5, y - n * 0.5) / n
                feed = preset["feed"] + wave * math.sin(radial * 14.0 + step * 0.025)
                next_u = u[idx] + (preset["du"] * lap(u, x, y, n) - uvv + feed * (1.0 - u[idx]))
                next_v = v[idx] + (preset["dv"] * lap(v, x, y, n) + uvv - (preset["kill"] + feed) * v[idx])
                u2[idx] = clamp(next_u, 0.0, 1.0)
                v2[idx] = clamp(next_v, 0.0, 1.0)

        for x in range(n):
            top = x
            bottom = (n - 1) * n + x
            u2[top] = u2[n + x]
            v2[top] = v2[n + x]
            u2[bottom] = u2[(n - 2) * n + x]
            v2[bottom] = v2[(n - 2) * n + x]
        for y in range(n):
            left = y * n
            right = y * n + n - 1
            u2[left] = u2[left + 1]
            v2[left] = v2[left + 1]
            u2[right] = u2[right - 1]
            v2[right] = v2[right - 1]
        u, u2 = u2, u
        v, v2 = v2, v
    return n, u, v


def sample(field: list[float], n: int, x: float, y: float) -> float:
    x = clamp(x, 0, n - 1.001)
    y = clamp(y, 0, n - 1.001)
    x0 = int(x)
    y0 = int(y)
    x1 = min(n - 1, x0 + 1)
    y1 = min(n - 1, y0 + 1)
    tx = x - x0
    ty = y - y0
    a = field[y0 * n + x0] * (1 - tx) + field[y0 * n + x1] * tx
    b = field[y1 * n + x0] * (1 - tx) + field[y1 * n + x1] * tx
    return a * (1 - ty) + b * ty


def composition_bounds(args: argparse.Namespace, width: int, height: int) -> tuple[int, int, int, int]:
    if args.composition == "full-bleed":
        return 0, 0, width, height
    if args.composition == "specimen":
        margin = int(width * 0.095)
        top = int(height * 0.075)
        size = min(width - margin * 2, height - top - int(height * 0.22))
        return margin, top, size, size
    if args.composition == "field-guide":
        margin = int(width * 0.070)
        top = int(height * 0.065)
        field_h = height - top - int(height * 0.235)
        return margin, top, width - margin * 2, field_h
    if args.composition == "triptych":
        margin = int(width * 0.050)
        top = int(height * 0.060)
        field_h = height - top - int(height * 0.205)
        return margin, top, width - margin * 2, field_h
    margin = int(width * 0.045)
    top = int(height * 0.045)
    size = width - margin * 2
    return margin, top, size, size


def symmetric_uv(args: argparse.Namespace, uval: float, vval: float) -> tuple[float, float]:
    if args.symmetry == "mirror-x":
        uval = min(uval, 1.0 - uval) * 2.0
    elif args.symmetry == "mirror-y":
        vval = min(vval, 1.0 - vval) * 2.0
    elif args.symmetry == "dihedral":
        uval = min(uval, 1.0 - uval) * 2.0
        vval = min(vval, 1.0 - vval) * 2.0
    return clamp(uval, 0.0, 1.0), clamp(vval, 0.0, 1.0)


def field_coordinates(
    args: argparse.Namespace,
    x: int,
    y: int,
    texture_x: int,
    texture_y: int,
    texture_w: int,
    texture_h: int,
    n: int,
) -> tuple[float, float] | None:
    if args.composition != "triptych":
        uval = (x - texture_x) / max(1, texture_w - 1)
        vval = (y - texture_y) / max(1, texture_h - 1)
        uval, vval = symmetric_uv(args, uval, vval)
        return uval * (n - 1), vval * (n - 1)

    gap = max(8, int(texture_w * 0.018))
    panel_w = (texture_w - gap * 2) // 3
    rel_x = x - texture_x
    for panel in range(3):
        start = panel * (panel_w + gap)
        end = start + panel_w
        if start <= rel_x < end:
            uval = (rel_x - start) / max(1, panel_w - 1)
            vval = (y - texture_y) / max(1, texture_h - 1)
            uval, vval = symmetric_uv(args, uval, vval)
            fx = (uval * (n - 1) + panel * n * 0.19) % (n - 1)
            fy = (vval * (n - 1) + panel * n * 0.07) % (n - 1)
            return fx, fy
    return None


def draw_motifs(
    image: bytearray,
    width: int,
    height: int,
    args: argparse.Namespace,
    palette: dict[str, object],
    texture_x: int,
    texture_y: int,
    texture_w: int,
    texture_h: int,
) -> None:
    if args.motif == "none" or args.ink_density <= 0:
        return

    rng = Rng(f"motif:{args.seed}:{args.motif}:{args.composition}")
    energy = clamp(args.energy, 0.0, 1.0)
    density = clamp(args.ink_density, 0.0, 1.0)
    accent = palette["accent"]
    muted = palette["muted"]
    ink = palette["ink"]
    shadow = (0, 0, 0)
    active = {args.motif}
    if args.motif == "hybrid":
        active = {"rings", "flow", "fibers", "constellation"}

    if "rings" in active:
        count = max(1, int(args.accent_count * (0.45 + energy)))
        for _ in range(count):
            cx = texture_x + texture_w * (0.08 + 0.84 * rng.random())
            cy = texture_y + texture_h * (0.06 + 0.78 * rng.random())
            radius = min(texture_w, texture_h) * (0.08 + 0.22 * rng.random())
            draw_circle(image, width, height, cx, cy, radius, accent, 0.045 + 0.12 * density, ring=True)

    if "flow" in active:
        count = max(4, int(13 + 25 * density))
        for i in range(count):
            y = texture_y + texture_h * (0.05 + i / max(1, count - 1) * 0.88)
            last_x = texture_x
            last_y = y + math.sin(i * 0.41 + hash_string(args.seed) * 0.000003) * texture_h * 0.010
            step = max(10, texture_w // 46)
            for x in range(texture_x + step, texture_x + texture_w + 1, step):
                yy = y + math.sin(x * 0.012 + i * 0.55 + hash_string(args.seed) * 0.000002) * texture_h * 0.012
                draw_line(image, width, height, last_x, last_y, x, yy, muted, 0.045 + 0.12 * density, 1)
                last_x, last_y = x, yy

    if "fibers" in active:
        count = max(5, int(9 + 18 * density))
        for i in range(count):
            start_x = texture_x + texture_w * rng.random()
            start_y = texture_y + texture_h * rng.random()
            angle = rng.random() * math.tau
            length = texture_w * (0.35 + 0.42 * rng.random())
            segments = 18
            last_x, last_y = start_x, start_y
            for s in range(1, segments + 1):
                t = s / segments
                bend = math.sin(t * math.tau * (1.0 + rng.random()) + i) * texture_w * 0.025
                x = start_x + math.cos(angle) * length * (t - 0.5) + math.cos(angle + math.pi / 2) * bend
                y = start_y + math.sin(angle) * length * (t - 0.5) + math.sin(angle + math.pi / 2) * bend
                draw_line(image, width, height, last_x, last_y, x, y, ink, 0.035 + 0.08 * density, 1)
                last_x, last_y = x, y

    if "fracture" in active:
        count = max(2, int(3 + args.accent_count * 0.35))
        for _ in range(count):
            x = texture_x + texture_w * rng.random()
            y = texture_y + texture_h * rng.random()
            angle = -math.pi / 2 + (rng.random() - 0.5) * 1.7
            for branch in range(9):
                length = texture_w * (0.045 + 0.050 * rng.random()) * (1.0 + energy)
                nx = x + math.cos(angle) * length
                ny = y + math.sin(angle) * length
                draw_line(image, width, height, x, y, nx, ny, shadow, 0.35 * density, 2)
                draw_line(image, width, height, x, y, nx, ny, accent, 0.15 * density, 1)
                if rng.random() < 0.45:
                    side = angle + (rng.random() - 0.5) * 1.8
                    bx = x + math.cos(side) * length * 0.55
                    by = y + math.sin(side) * length * 0.55
                    draw_line(image, width, height, x, y, bx, by, accent, 0.11 * density, 1)
                x, y = nx, ny
                angle += (rng.random() - 0.5) * 0.9

    if "constellation" in active:
        count = max(5, int(args.accent_count * (0.8 + density)))
        points: list[tuple[float, float]] = []
        for _ in range(count):
            points.append(
                (
                    texture_x + texture_w * (0.07 + 0.86 * rng.random()),
                    texture_y + texture_h * (0.06 + 0.80 * rng.random()),
                )
            )
        for i, (x0, y0) in enumerate(points):
            x1, y1 = points[(i + 1 + int(rng.random() * max(1, count - 1))) % count]
            draw_line(image, width, height, x0, y0, x1, y1, muted, 0.06 * density, 1)
        for x, y in points:
            draw_circle(image, width, height, x, y, width * (0.006 + 0.006 * rng.random()), accent, 0.35 * density)


def resolve_palette_color(value: object, palette: dict[str, object]) -> tuple[int, int, int]:
    if isinstance(value, str) and value in palette and isinstance(palette[value], tuple):
        return palette[value]  # type: ignore[return-value]
    return coerce_color(value, palette["accent"])  # type: ignore[arg-type]


def rule_point_to_canvas(
    point: object,
    texture_x: int,
    texture_y: int,
    texture_w: int,
    texture_h: int,
) -> tuple[float, float]:
    seq = list(point)  # type: ignore[arg-type]
    if len(seq) < 2:
        raise ValueError("motif point must contain x and y")
    px = float(seq[0])
    py = float(seq[1])
    if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0:
        return texture_x + px * texture_w, texture_y + py * texture_h
    return px, py


def draw_rule_motifs(
    image: bytearray,
    width: int,
    height: int,
    args: argparse.Namespace,
    palette: dict[str, object],
    texture_x: int,
    texture_y: int,
    texture_w: int,
    texture_h: int,
    module,
) -> None:
    if module is None or not hasattr(module, "motif_paths"):
        return
    context = {
        "width": width,
        "height": height,
        "texture_x": texture_x,
        "texture_y": texture_y,
        "texture_w": texture_w,
        "texture_h": texture_h,
        "args": args,
        "palette": palette,
        "math": math,
        "clamp": clamp,
        "Rng": Rng,
    }
    commands = module.motif_paths(context)
    if not commands:
        return
    for command in commands:
        if not isinstance(command, dict):
            raise TypeError("motif_paths(context) must return a list of dictionaries")
        kind = command.get("type", "line")
        color = resolve_palette_color(command.get("color", "accent"), palette)
        alpha = clamp(float(command.get("alpha", 0.28)), 0.0, 1.0)
        thickness = max(1, int(command.get("thickness", 1)))
        if kind in {"line", "polyline"}:
            points = command.get("points")
            if not points:
                raise ValueError("line/polyline motif command needs points")
            canvas_points = [rule_point_to_canvas(point, texture_x, texture_y, texture_w, texture_h) for point in points]
            for (x0, y0), (x1, y1) in zip(canvas_points, canvas_points[1:]):
                draw_line(image, width, height, x0, y0, x1, y1, color, alpha, thickness)
        elif kind == "circle":
            x = float(command.get("x", 0.5))
            y = float(command.get("y", 0.5))
            cx, cy = rule_point_to_canvas((x, y), texture_x, texture_y, texture_w, texture_h)
            radius = float(command.get("radius", 0.05))
            if 0.0 < radius <= 1.0:
                radius *= min(texture_w, texture_h)
            draw_circle(image, width, height, cx, cy, radius, color, alpha, bool(command.get("ring", False)))
        elif kind == "frame":
            draw_frame(
                image,
                width,
                height,
                texture_x,
                texture_y,
                texture_x + texture_w,
                texture_y + texture_h,
                color,
                alpha,
                thickness,
            )
        else:
            raise ValueError(f"Unknown motif command type: {kind}")


def custom_helpers() -> dict[str, object]:
    return {
        "alpha_blend": alpha_blend,
        "fill_rect": fill_rect,
        "draw_line": draw_line,
        "draw_circle": draw_circle,
        "draw_frame": draw_frame,
        "draw_text": draw_text,
        "text_width": text_width,
        "fit_text": fit_text,
        "trim_text_to_width": trim_text_to_width,
        "clamp": clamp,
        "hash_string": hash_string,
        "Rng": Rng,
        "mix": mix,
        "coerce_color": coerce_color,
    }


def draw_labels(
    image: bytearray,
    width: int,
    height: int,
    args: argparse.Namespace,
    palette: dict[str, object],
    texture_x: int,
    texture_y: int,
    texture_w: int,
    texture_h: int,
) -> None:
    if args.title_mode == "none":
        return

    margin = max(20, int(width * 0.045))
    panel_h = int(height * 0.155)
    if args.title_mode == "caption":
        panel_y = height - panel_h - int(height * 0.020)
        panel_alpha = 0.38
    else:
        panel_y = min(height - panel_h - int(height * 0.030), texture_y + texture_h - panel_h - int(height * 0.004))
        panel_alpha = 0.82

    shadow = (0, 0, 0)
    panel_x0 = margin
    panel_x1 = width - margin
    if args.composition == "field-guide":
        panel_y = texture_y + texture_h + int(height * 0.025)
    fill_rect(image, width, height, panel_x0 + 10, panel_y + 10, panel_x1 + 10, panel_y + panel_h + 10, shadow, 0.28)
    fill_rect(image, width, height, panel_x0, panel_y, panel_x1, panel_y + panel_h, palette["panel"], panel_alpha)

    title = args.title[:64]
    subtitle = args.subtitle[:88]
    meta = f"{args.preset} / {args.palette} / {args.composition} / {args.motif} / seed {str(args.seed)[:16]}"
    max_text_width = panel_x1 - panel_x0 - margin
    title_scale = fit_text(title, max_text_width, max(5, width // 126), 2)
    sub_scale = fit_text(subtitle, max_text_width, max(3, width // 245), 2)
    meta_scale = fit_text(meta, max_text_width, max(2, width // 360), 2)
    title = trim_text_to_width(title, max_text_width, title_scale)
    subtitle = trim_text_to_width(subtitle, max_text_width, sub_scale)
    meta = trim_text_to_width(meta, max_text_width, meta_scale)

    text_x = panel_x0 + int(width * 0.028)
    text_y = panel_y + int(panel_h * 0.18)
    draw_text(image, width, height, text_x + 3, text_y + 4, title, shadow, title_scale, 0.45)
    draw_text(image, width, height, text_x, text_y, title, palette["ink"], title_scale, 0.98)
    draw_text(image, width, height, text_x, text_y + title_scale * 9, subtitle, palette["muted"], sub_scale, 0.95)
    draw_text(image, width, height, text_x, panel_y + panel_h - meta_scale * 9, meta, palette["accent"], meta_scale, 0.98)


def render(args: argparse.Namespace, n: int, u: list[float], v: list[float]) -> bytearray:
    width = args.size
    height = int(args.size * 1.25) if args.format == "portrait" else args.size
    palette = PALETTES[args.palette]
    preset = PRESETS[args.preset]
    image = bytearray(width * height * 3)
    texture_x, texture_y, texture_w, texture_h = composition_bounds(args, width, height)
    module = getattr(args, "rule_module", None)
    tone_rule = getattr(module, "tone_rule", None) if module is not None else None
    color_rule = getattr(module, "color_rule", None) if module is not None else None

    min_v = min(v)
    max_v = max(v)
    span_v = max(0.0001, max_v - min_v)
    seed_phase = hash_string(args.seed) * 0.000002
    contrast = clamp(args.contrast, 0.35, 2.50)
    energy = clamp(args.energy, 0.0, 1.0)
    grain = clamp(args.grain, 0.0, 1.0)
    prompt_profile = getattr(args, "prompt_profile", {}) or {}
    prompt_weights = prompt_profile.get("weights", {}) if prompt_profile else {}
    prompt_strength = clamp(float(getattr(args, "prompt_strength", 0.0)), 0.0, 1.0) if prompt_profile else 0.0

    def pw(name: str) -> float:
        return clamp(float(prompt_weights.get(name, 0.0)), 0.0, 1.0)

    for y in range(height):
        bg_t = y / max(1, height - 1)
        bg = mix(palette["bg0"], palette["bg1"], bg_t)
        for x in range(width):
            idx = (y * width + x) * 3
            image[idx] = bg[0]
            image[idx + 1] = bg[1]
            image[idx + 2] = bg[2]

    for y in range(max(0, texture_y), min(height, texture_y + texture_h)):
        for x in range(max(0, texture_x), min(width, texture_x + texture_w)):
            coords = field_coordinates(args, x, y, texture_x, texture_y, texture_w, texture_h, n)
            if coords is None:
                continue
            fx, fy = coords
            activator = (sample(v, n, fx, fy) - min_v) / span_v
            depletion = 1.0 - sample(u, n, fx, fy)
            left = sample(v, n, max(0.0, fx - 1.0), fy)
            right = sample(v, n, min(n - 1.0, fx + 1.0), fy)
            up = sample(v, n, fx, max(0.0, fy - 1.0))
            down = sample(v, n, fx, min(n - 1.0, fy + 1.0))
            edge = clamp(math.hypot(right - left, down - up) * (5.0 + 5.0 * energy), 0.0, 1.0)
            wave_a = math.sin(fx * 0.115 + math.sin(fy * 0.052 + seed_phase) * 2.1 + seed_phase * 3.0)
            wave_b = math.sin((fx * 0.052 - fy * 0.071) + seed_phase * 7.0)
            ridge_a = (1.0 - clamp(abs(wave_a) * 1.45, 0.0, 1.0)) ** 2.2
            ridge_b = (1.0 - clamp(abs(wave_b) * 1.70, 0.0, 1.0)) ** 2.6
            ridge = (ridge_a + ridge_b * 0.50) * preset["ridge"] * (0.55 + energy)
            tone = (activator * 0.70 + depletion * 0.58) ** 0.78 * (0.62 + 0.30 * energy)
            tone = clamp(tone + edge * 0.35 + ridge, 0.0, 1.0)
            if prompt_strength > 0:
                dx = fx - n * 0.5
                dy = fy - n * 0.5
                radial = math.hypot(dx, dy) / max(1.0, n)
                angle = math.atan2(dy, dx)
                prompt_signal = 0.0
                prompt_mass = 0.0
                wave_s = max(pw("wave"), pw("quantum") * 0.6)
                if wave_s > 0:
                    prompt_signal += wave_s * (0.5 + 0.5 * math.sin(radial * 52.0 + angle * 5.0 + seed_phase * 17.0))
                    prompt_mass += wave_s
                branch_s = max(pw("branch"), pw("bio") * 0.7)
                if branch_s > 0:
                    fibers = abs(math.sin(fx * 0.10 + math.sin(fy * 0.055 + seed_phase) * 2.5))
                    prompt_signal += branch_s * (1.0 - clamp(fibers, 0.0, 1.0))
                    prompt_mass += branch_s
                stress_s = pw("stress")
                if stress_s > 0:
                    prompt_signal += stress_s * clamp(edge * 1.8 + ridge * 0.7, 0.0, 1.0)
                    prompt_mass += stress_s
                network_s = pw("network")
                if network_s > 0:
                    lattice = abs(math.sin(fx * 0.115 + seed_phase) * math.sin(fy * 0.135 - seed_phase))
                    prompt_signal += network_s * (1.0 - clamp(lattice * 1.45, 0.0, 1.0))
                    prompt_mass += network_s
                crystal_s = max(pw("crystal"), pw("quiet") * 0.55)
                if crystal_s > 0:
                    crystalline = abs(math.cos(fx * 0.18) + math.cos(fy * 0.18)) * 0.5
                    prompt_signal += crystal_s * crystalline
                    prompt_mass += crystal_s
                if prompt_mass > 0:
                    prompt_tone = prompt_signal / prompt_mass
                    tone = clamp(tone * (1.0 - 0.22 * prompt_strength) + prompt_tone * 0.22 * prompt_strength, 0.0, 1.0)
            tone = clamp((tone - 0.5) * contrast + 0.5, 0.0, 1.0)
            if tone_rule is not None:
                custom = tone_rule(
                    fx=fx,
                    fy=fy,
                    x=x,
                    y=y,
                    activator=activator,
                    depletion=depletion,
                    edge=edge,
                    ridge=ridge,
                    tone=tone,
                    context={
                        "width": width,
                        "height": height,
                        "n": n,
                        "args": args,
                        "texture_x": texture_x,
                        "texture_y": texture_y,
                        "texture_w": texture_w,
                        "texture_h": texture_h,
                        "math": math,
                        "clamp": clamp,
                    },
                )
                if custom is not None:
                    if isinstance(custom, dict):
                        tone = clamp(float(custom.get("tone", tone)), 0.0, 1.0)
                    else:
                        tone = clamp(float(custom), 0.0, 1.0)
            color = palette_color(palette["stops"], tone)
            if color_rule is not None:
                custom_color = color_rule(
                    r=color[0],
                    g=color[1],
                    b=color[2],
                    tone=tone,
                    x=x,
                    y=y,
                    fx=fx,
                    fy=fy,
                    context={
                        "width": width,
                        "height": height,
                        "n": n,
                        "args": args,
                        "texture_x": texture_x,
                        "texture_y": texture_y,
                        "texture_w": texture_w,
                        "texture_h": texture_h,
                        "math": math,
                        "clamp": clamp,
                    },
                )
                color = coerce_color(custom_color, color)
            vignette = 1.0 - 0.30 * (math.hypot(x - width * 0.5, y - height * 0.45) / (width * 0.76)) ** 1.8
            color = tuple(max(0, min(255, int(c * clamp(vignette, 0.55, 1.0)))) for c in color)
            if grain > 0:
                noise = math.sin(x * 12.9898 + y * 78.233 + seed_phase * 997.0) * 43758.5453
                noise = noise - math.floor(noise)
                color = tuple(max(0, min(255, int(c + (noise - 0.5) * 72.0 * grain))) for c in color)
            idx = (y * width + x) * 3
            image[idx] = color[0]
            image[idx + 1] = color[1]
            image[idx + 2] = color[2]

    if args.composition in {"specimen", "field-guide", "triptych"}:
        draw_frame(
            image,
            width,
            height,
            texture_x,
            texture_y,
            texture_x + texture_w,
            texture_y + texture_h,
            palette["accent"],
            0.22,
            max(1, width // 360),
        )
    if args.composition == "triptych":
        gap = max(8, int(texture_w * 0.018))
        panel_w = (texture_w - gap * 2) // 3
        for panel in (1, 2):
            gx = texture_x + panel * panel_w + (panel - 1) * gap
            fill_rect(image, width, height, gx, texture_y, gx + gap, texture_y + texture_h, palette["bg0"], 0.92)

    draw_motifs(image, width, height, args, palette, texture_x, texture_y, texture_w, texture_h)
    draw_rule_motifs(image, width, height, args, palette, texture_x, texture_y, texture_w, texture_h, module)
    if module is not None and hasattr(module, "postprocess"):
        result = module.postprocess(
            image,
            width,
            height,
            args,
            palette,
            {
                **custom_helpers(),
                "texture_x": texture_x,
                "texture_y": texture_y,
                "texture_w": texture_w,
                "texture_h": texture_h,
            },
        )
        if result is not None:
            if not isinstance(result, bytearray) or len(result) != width * height * 3:
                raise TypeError("postprocess(...) must return a bytearray of RGB pixels or None")
            image = result
    draw_labels(image, width, height, args, palette, texture_x, texture_y, texture_w, texture_h)
    return image


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_png(path: Path, width: int, height: int, rgb: bytearray) -> None:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(rgb[y * stride : (y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            png_chunk(b"IHDR", ihdr),
            png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
            png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(data)


def write_metadata(args: argparse.Namespace, out_dir: Path, png_path: Path) -> None:
    params = {
        "title": args.title,
        "subtitle": args.subtitle,
        "prompt": args.prompt,
        "auto_style": bool(args.auto_style or args.prompt),
        "prompt_strength": args.prompt_strength,
        "prompt_profile": args.prompt_profile,
        "seed": args.seed,
        "preset": args.preset,
        "palette": args.palette,
        "composition": args.composition,
        "motif": args.motif,
        "symmetry": args.symmetry,
        "title_mode": args.title_mode,
        "energy": args.energy,
        "contrast": args.contrast,
        "grain": args.grain,
        "ink_density": args.ink_density,
        "accent_count": args.accent_count,
        "format": args.format,
        "steps": args.steps,
        "field_size": args.field_size,
        "size": args.size,
        "rule_code": args.rule_code,
        "model": "Gray-Scott reaction-diffusion",
        "outputs": {
            "poster_png": png_path.name,
            "parameters_json": "parameters.json",
            "caption_txt": "caption.txt",
            "readme": "README.md",
            "agent_rules": "agent_rules.py" if args.rule_code else None,
        },
    }
    (out_dir / "parameters.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    caption = (
        f"{args.title} - a deterministic Gray-Scott morphogenesis postcard "
        f"({args.preset}, {args.palette}, {args.composition}, {args.motif}, seed {args.seed}) showing how local "
        "diffusion and nonlinear reaction kinetics produce global visual order."
    )
    if args.prompt:
        caption += f" Prompt brief: {args.prompt}"
    (out_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")

    prompt_note = args.prompt or "none"
    prompt_dominant = ", ".join(args.prompt_profile.get("dominant", [])) if args.prompt_profile else "none"
    readme = f"""# Morphogenesis Postcard

Generated artifact: `{png_path.name}`

This image is produced by a deterministic Gray-Scott reaction-diffusion
simulation. The bright ridges and islands come from two coupled scalar fields:
an activator-like species `v` and a depleted substrate-like species `u`.

Parameters:

- title: {args.title}
- subtitle: {args.subtitle}
- prompt: {prompt_note}
- prompt-derived dominant themes: {prompt_dominant}
- auto style: {bool(args.auto_style or args.prompt)}
- prompt strength: {args.prompt_strength}
- seed: {args.seed}
- preset: {args.preset}
- palette: {args.palette}
- composition: {args.composition}
- motif: {args.motif}
- symmetry: {args.symmetry}
- title mode: {args.title_mode}
- energy: {args.energy}
- contrast: {args.contrast}
- grain: {args.grain}
- ink density: {args.ink_density}
- accent count: {args.accent_count}
- format: {args.format}
- steps: {args.steps}
- field size: {args.field_size} x {args.field_size}
- output size: {args.size}px wide
- agent rule code: {args.rule_code or "none"}

Use the same parameters to reproduce the image exactly.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="skill_output_morphogenesis-postcard", help="Output directory.")
    parser.add_argument("--title", default="Morphogenesis From Local Rules", help="Poster title.")
    parser.add_argument(
        "--subtitle",
        default="Local diffusion, nonlinear growth, emergent form",
        help="Poster subtitle.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Natural-language visual brief. When present, it drives auto-style and prompt-conditioned rules.",
    )
    parser.add_argument(
        "--auto-style",
        action="store_true",
        help="Derive preset, palette, motif, symmetry, and numeric controls from --prompt or title/subtitle.",
    )
    parser.add_argument(
        "--prompt-strength",
        type=float,
        default=0.78,
        help="0..1 strength of prompt-derived simulation and tone perturbations.",
    )
    parser.add_argument("--seed", default="42", help="Deterministic seed.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="labyrinth")
    parser.add_argument("--palette", choices=sorted(PALETTES), default="magma-cyan")
    parser.add_argument("--composition", choices=COMPOSITIONS, default="poster")
    parser.add_argument("--motif", choices=MOTIFS, default="hybrid")
    parser.add_argument("--symmetry", choices=SYMMETRIES, default="none")
    parser.add_argument("--title-mode", choices=TITLE_MODES, default="panel")
    parser.add_argument("--energy", type=float, default=0.72, help="0..1 field intensity and motif force.")
    parser.add_argument("--contrast", type=float, default=1.10, help="Field contrast multiplier.")
    parser.add_argument("--grain", type=float, default=0.08, help="0..1 deterministic print grain.")
    parser.add_argument("--ink-density", type=float, default=0.72, help="0..1 overlay density.")
    parser.add_argument("--accent-count", type=int, default=10, help="Number of accent motifs/nodes.")
    parser.add_argument("--format", choices=("square", "portrait"), default="square")
    parser.add_argument("--steps", type=int, default=240, help="Reaction-diffusion update steps.")
    parser.add_argument("--field-size", type=int, default=152, help="Internal simulation grid size.")
    parser.add_argument("--size", type=int, default=1080, help="PNG output width in pixels.")
    parser.add_argument("--rule-code", default=None, help="Optional Python file with agent-authored generative rules.")
    args = parser.parse_args()
    args._explicit_style_args = explicit_style_args(sys.argv[1:])
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rule_code:
        src = Path(args.rule_code)
        dst = out_dir / "agent_rules.py"
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        args.rule_code = str(dst)

    rule_module = load_rule_module(args.rule_code)
    if args.prompt or args.auto_style:
        apply_prompt_style(args)
    else:
        args.prompt_profile = {}
    apply_rule_config(args, rule_module)
    normalize_args(args)
    args.rule_module = rule_module

    width = args.size
    height = int(args.size * 1.25) if args.format == "portrait" else args.size

    n, u, v = simulate(args)
    image = render(args, n, u, v)

    safe = "".join(ch if ch.isalnum() else "_" for ch in args.title.lower()).strip("_")
    safe = safe[:48] or "morphogenesis_postcard"
    png_path = out_dir / f"{safe}_{args.preset}_{args.palette}_{args.seed}.png".replace("/", "_")
    write_png(png_path, width, height, image)
    write_metadata(args, out_dir, png_path)

    print(f"Wrote {png_path}")
    print(f"Wrote {out_dir / 'parameters.json'}")
    print(f"Wrote {out_dir / 'caption.txt'}")
    print(f"Wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
