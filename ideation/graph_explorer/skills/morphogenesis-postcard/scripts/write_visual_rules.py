#!/usr/bin/env python3
"""Write prompt-conditioned visual rules for morphogenesis-postcard.

The generated file is ordinary Python consumed by
make_morphogenesis_postcard.py --rule-code. It intentionally uses only the
standard library and keeps the rule API small so local models can rely on a
real code artifact without hand-writing fragile code on every run.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


KEYWORDS = {
    "stress": {"stress", "strain", "fracture", "crack", "cracks", "rupture", "failure", "defect", "defects", "damage", "shear"},
    "wave": {"wave", "waves", "interference", "phase", "oscillation", "resonance", "phonon", "vibration", "ripple", "frequency"},
    "bio": {"bio", "biological", "morphogenesis", "cell", "cells", "cellular", "tissue", "growth", "bone", "collagen", "protein", "membrane"},
    "network": {"graph", "network", "node", "nodes", "edge", "edges", "bridge", "bridges", "module", "modules", "link", "links", "path", "paths"},
    "quantum": {"quantum", "plasma", "neon", "electron", "field", "fields", "charge", "electric", "magnetic", "glowing", "luminous"},
    "thermal": {"thermal", "heat", "fire", "magma", "flame", "temperature", "energy", "hot"},
    "quiet": {"quiet", "minimal", "clean", "museum", "specimen", "archive", "white", "ice", "clinical", "elegant"},
    "crystal": {"crystal", "crystals", "lattice", "periodic", "symmetry", "symmetric", "moire", "snow", "mineral", "ordered"},
    "branch": {"branch", "branches", "branching", "vascular", "tree", "dendrite", "dendritic", "root", "fiber", "fibers", "hierarchy", "hierarchical"},
    "art": {"postcard", "poster", "album", "cover", "social", "garden", "aesthetic", "art", "beautiful"},
}


def hash_string(value: object) -> int:
    h = 2166136261
    for ch in str(value):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h or 1


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def weight(scores: dict[str, int], name: str) -> float:
    return max(0.0, min(1.0, scores.get(name, 0) / 3.0))


def choice(options: tuple[str, ...], prompt_hash: int, salt: str) -> str:
    return options[hash_string(f"{prompt_hash}:{salt}") % len(options)]


def derive_profile(prompt: str, title: str, subtitle: str) -> dict[str, object]:
    source = prompt.strip() or f"{title} {subtitle}".strip() or "morphogenesis reaction-diffusion science art"
    found = tokens(source)
    token_set = set(found)
    scores = {name: sum(1 for token in found if token in words) for name, words in KEYWORDS.items()}
    weights = {name: weight(scores, name) for name in KEYWORDS}
    prompt_hash = hash_string(source)
    ranked = [name for name, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if score > 0]
    if not ranked:
        ranked = [choice(("wave", "bio", "network", "branch", "crystal"), prompt_hash, "fallback")]
    primary = ranked[0]
    secondary = ranked[1] if len(ranked) > 1 else ""

    style = {
        "preset": "labyrinth",
        "palette": "magma-cyan",
        "composition": "poster",
        "motif": "hybrid",
        "symmetry": "none",
        "title_mode": "panel",
    }
    if primary == "stress":
        style.update({"preset": "veins", "palette": "graphite-fire", "motif": "fracture"})
    elif primary == "wave":
        style.update({"preset": "labyrinth", "palette": "noir-neon" if weights["quantum"] else "magma-cyan", "motif": "flow", "composition": "full-bleed"})
    elif primary == "bio":
        style.update({"preset": "coral" if weights["branch"] else "membranes", "palette": "biofilm", "motif": "fibers"})
    elif primary == "network":
        style.update({"preset": "labyrinth", "palette": "magma-cyan", "motif": "constellation", "composition": choice(("field-guide", "triptych", "poster"), prompt_hash, "network")})
    elif primary == "quantum":
        style.update({"preset": "spots", "palette": "noir-neon", "motif": choice(("rings", "flow", "constellation"), prompt_hash, "quantum"), "composition": "full-bleed"})
    elif primary == "thermal":
        style.update({"preset": "veins", "palette": "graphite-fire", "motif": "hybrid"})
    elif primary == "quiet":
        style.update({"preset": "membranes", "palette": "ice", "motif": "rings", "composition": "specimen", "title_mode": "caption"})
    elif primary == "crystal":
        style.update({"preset": "spots", "palette": "ice", "motif": "rings", "composition": "specimen", "symmetry": "dihedral"})
    elif primary == "branch":
        style.update({"preset": "coral", "palette": "biofilm", "motif": "fibers"})

    if secondary == "network" and style["motif"] != "constellation":
        style["motif"] = "hybrid"
    if weights["crystal"] > 0.20 and style["symmetry"] == "none":
        style["symmetry"] = choice(("mirror-x", "mirror-y", "dihedral"), prompt_hash, "symmetry")
    if weights["quiet"] > 0.25:
        style.update({"palette": "ice", "composition": "specimen", "title_mode": "caption"})
    if "triptych" in token_set or ({"three", "panel"} <= token_set) or ({"three", "panels"} <= token_set):
        style["composition"] = "triptych"
    elif "field" in token_set and "guide" in token_set:
        style["composition"] = "field-guide"
    elif "specimen" in token_set or "museum" in token_set:
        style["composition"] = "specimen"
    elif "full" in token_set and "bleed" in token_set:
        style["composition"] = "full-bleed"

    hash_float = (prompt_hash % 1000) / 999.0
    style["energy"] = max(0.42, min(0.98, 0.56 + 0.16 * weights["stress"] + 0.12 * weights["wave"] + 0.12 * weights["thermal"] + 0.10 * hash_float))
    style["contrast"] = max(0.72, min(1.82, 0.94 + 0.30 * weights["stress"] + 0.22 * weights["quantum"] + 0.16 * weights["crystal"] + 0.18 * hash_float))
    style["grain"] = max(0.01, min(0.22, 0.035 + 0.06 * weights["stress"] + 0.05 * weights["thermal"] + 0.05 * hash_float))
    style["ink_density"] = max(0.38, min(0.96, 0.48 + 0.18 * weights["network"] + 0.16 * weights["branch"] + 0.12 * weights["wave"] + 0.10 * weights["bio"]))
    style["accent_count"] = int(7 + 8 * weights["network"] + 7 * weights["branch"] + 6 * weights["wave"] + 5 * weights["stress"] + 4 * hash_float)

    return {
        "prompt": source,
        "hash": f"{prompt_hash:08x}",
        "tokens": found[:96],
        "scores": scores,
        "weights": weights,
        "dominant": ranked[:4],
        "style": style,
    }


def rule_code(profile: dict[str, object]) -> str:
    return f'''"""Prompt-written visual rules for morphogenesis-postcard.

This file was generated by write_visual_rules.py. It is intentionally ordinary
Python so the visual grammar for this run can be inspected, modified, or reused.
"""

import math


PROFILE = {profile!r}


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def hash_string(value):
    h = 2166136261
    for ch in str(value):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h or 1


class Rng:
    def __init__(self, seed):
        self.state = hash_string(seed)

    def random(self):
        self.state ^= (self.state << 13) & 0xFFFFFFFF
        self.state ^= self.state >> 17
        self.state ^= (self.state << 5) & 0xFFFFFFFF
        self.state &= 0xFFFFFFFF
        return self.state / 4294967296.0


def configure(args):
    style = dict(PROFILE["style"])
    style["seed"] = "rules-" + PROFILE["hash"]
    return style


def seed_rule(u, v, n, args):
    weights = PROFILE["weights"]
    rng = Rng("seed:" + PROFILE["hash"])
    cx = n * 0.5
    cy = n * 0.5

    def stamp(sx, sy, radius, strength):
        r2 = radius * radius
        for yy in range(max(2, int(sy - radius)), min(n - 2, int(sy + radius) + 1)):
            for xx in range(max(2, int(sx - radius)), min(n - 2, int(sx + radius) + 1)):
                d2 = (xx - sx) * (xx - sx) + (yy - sy) * (yy - sy)
                if d2 <= r2:
                    w = 1.0 - d2 / r2
                    idx = yy * n + xx
                    u[idx] = min(u[idx], 0.38 + 0.18 * rng.random())
                    v[idx] = max(v[idx], strength * (0.35 + 0.65 * w))

    branch = max(weights.get("branch", 0.0), weights.get("bio", 0.0) * 0.75)
    wave = max(weights.get("wave", 0.0), weights.get("quantum", 0.0) * 0.60)
    stress = weights.get("stress", 0.0)
    network = weights.get("network", 0.0)
    crystal = max(weights.get("crystal", 0.0), weights.get("quiet", 0.0) * 0.60)

    if wave:
        angle = rng.random() * math.pi
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        phase = rng.random() * math.tau
        for y in range(2, n - 2):
            dy = y - cy
            for x in range(2, n - 2):
                dx = x - cx
                projected = dx * cos_a + dy * sin_a
                across = -dx * sin_a + dy * cos_a
                signal = math.sin(projected * 0.10 + phase) + 0.55 * math.sin(across * 0.17 - phase)
                envelope = 1.0 - clamp(math.hypot(dx, dy) / (n * 0.72))
                if abs(signal) < 0.16 + wave * 0.06 and rng.random() < 0.12 + wave * 0.08:
                    idx = y * n + x
                    u[idx] = min(u[idx], 0.44)
                    v[idx] = max(v[idx], 0.18 + 0.56 * envelope * wave)

    if branch:
        for _ in range(max(4, int(5 + branch * 11))):
            x = cx + (rng.random() - 0.5) * n * 0.34
            y = cy + (rng.random() - 0.5) * n * 0.34
            angle = rng.random() * math.tau
            for _ in range(14):
                stamp(x, y, 2.2 + branch * 3.4, 0.45 + branch * 0.35)
                angle += (rng.random() - 0.5) * (0.65 + branch * 0.70)
                length = n * (0.020 + 0.032 * rng.random()) * (0.85 + branch)
                x += math.cos(angle) * length
                y += math.sin(angle) * length
                if not (4 < x < n - 4 and 4 < y < n - 4):
                    break

    if stress:
        for _ in range(max(2, int(3 + stress * 10))):
            x = n * (0.10 + 0.80 * rng.random())
            y = n * (0.10 + 0.80 * rng.random())
            angle = -math.pi / 2 + (rng.random() - 0.5) * 1.8
            for _ in range(18):
                stamp(x, y, 1.8 + 2.8 * stress, 0.58 + 0.32 * stress)
                length = n * (0.018 + 0.035 * rng.random())
                x += math.cos(angle) * length
                y += math.sin(angle) * length
                angle += (rng.random() - 0.5) * 0.95
                if not (3 < x < n - 3 and 3 < y < n - 3):
                    break

    if network:
        points = []
        count = max(6, int(7 + network * 14))
        for _ in range(count):
            px = n * (0.10 + 0.80 * rng.random())
            py = n * (0.10 + 0.80 * rng.random())
            points.append((px, py))
            stamp(px, py, 2.5 + 3.4 * network, 0.48 + 0.32 * network)
        for i, (x0, y0) in enumerate(points):
            x1, y1 = points[(i + 1 + int(rng.random() * max(1, count - 1))) % count]
            for segment in range(12):
                t = segment / 11
                if rng.random() < 0.38 * network:
                    stamp(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, 1.3 + network, 0.28 + 0.16 * network)

    if crystal:
        spacing = max(8, int(n * (0.145 - 0.050 * crystal)))
        for y in range(5 + int(rng.random() * spacing), n - 5, spacing):
            for x in range(5 + int(rng.random() * spacing), n - 5, spacing):
                stamp(x, y, 2.5 + 3.2 * crystal, 0.36 + 0.40 * crystal)
    return u, v


def tone_rule(fx, fy, x, y, activator, depletion, edge, ridge, tone, context):
    weights = PROFILE["weights"]
    n = context["n"]
    dx = fx - n * 0.5
    dy = fy - n * 0.5
    radial = math.hypot(dx, dy) / max(1.0, n)
    angle = math.atan2(dy, dx)
    signal = tone
    mass = 1.0
    wave = max(weights.get("wave", 0.0), weights.get("quantum", 0.0) * 0.6)
    if wave:
        signal += wave * (0.5 + 0.5 * math.sin(radial * 54.0 + angle * 6.0))
        mass += wave
    branch = max(weights.get("branch", 0.0), weights.get("bio", 0.0) * 0.7)
    if branch:
        fiber = 1.0 - clamp(abs(math.sin(fx * 0.11 + math.sin(fy * 0.055) * 2.8)))
        signal += branch * fiber
        mass += branch
    stress = weights.get("stress", 0.0)
    if stress:
        signal += stress * clamp(edge * 1.8 + ridge * 0.8)
        mass += stress
    network = weights.get("network", 0.0)
    if network:
        lattice = 1.0 - clamp(abs(math.sin(fx * 0.12) * math.sin(fy * 0.14)) * 1.45)
        signal += network * lattice
        mass += network
    crystal = max(weights.get("crystal", 0.0), weights.get("quiet", 0.0) * 0.55)
    if crystal:
        signal += crystal * (abs(math.cos(fx * 0.18) + math.cos(fy * 0.18)) * 0.5)
        mass += crystal
    return clamp(tone * 0.56 + (signal / mass) * 0.44)


def color_rule(r, g, b, tone, x, y, fx, fy, context):
    weights = PROFILE["weights"]
    warm = weights.get("thermal", 0.0) + weights.get("stress", 0.0) * 0.45
    neon = weights.get("quantum", 0.0) + weights.get("wave", 0.0) * 0.25
    bio = weights.get("bio", 0.0) + weights.get("branch", 0.0) * 0.45
    shimmer = 0.5 + 0.5 * math.sin(fx * 0.075 + fy * 0.12 + tone * 4.0)
    rr = r * (1.0 + 0.18 * warm)
    gg = g * (1.0 + 0.14 * bio + 0.06 * shimmer)
    bb = b * (1.0 + 0.20 * neon)
    return (min(255, int(rr)), min(255, int(gg)), min(255, int(bb)))


def motif_paths(context):
    args = context["args"]
    weights = PROFILE["weights"]
    rng = context["Rng"]("motifs:" + PROFILE["hash"])
    paths = []
    density = max(0.35, min(1.0, float(getattr(args, "ink_density", 0.7))))
    if weights.get("wave", 0.0) or weights.get("quantum", 0.0):
        for i in range(max(3, int(5 + 8 * density))):
            y = 0.12 + i * 0.10
            paths.append({{
                "type": "polyline",
                "points": [(0.04, y), (0.25, y + 0.05 * math.sin(i)), (0.52, y - 0.06 * math.cos(i * 0.7)), (0.78, y + 0.04 * math.sin(i * 1.3)), (0.96, y)],
                "color": "muted",
                "alpha": 0.12 + 0.08 * density,
                "thickness": 1,
            }})
    if weights.get("stress", 0.0):
        for _ in range(max(3, int(4 + weights["stress"] * 9))):
            x = 0.12 + 0.76 * rng.random()
            y = 0.10 + 0.76 * rng.random()
            pts = [(x, y)]
            angle = -math.pi / 2 + (rng.random() - 0.5) * 1.6
            for _ in range(5):
                x += math.cos(angle) * (0.05 + 0.05 * rng.random())
                y += math.sin(angle) * (0.04 + 0.06 * rng.random())
                pts.append((clamp(x, 0.02, 0.98), clamp(y, 0.02, 0.95)))
                angle += (rng.random() - 0.5) * 0.95
            paths.append({{"type": "polyline", "points": pts, "color": "accent", "alpha": 0.18, "thickness": 2}})
    if weights.get("network", 0.0):
        count = max(6, int(8 + weights["network"] * 12))
        pts = [(0.08 + 0.84 * rng.random(), 0.08 + 0.78 * rng.random()) for _ in range(count)]
        for i, point in enumerate(pts):
            paths.append({{"type": "circle", "x": point[0], "y": point[1], "radius": 0.012 + 0.006 * rng.random(), "color": "accent", "alpha": 0.24}})
            nxt = pts[(i + 1 + int(rng.random() * max(1, count - 1))) % count]
            paths.append({{"type": "polyline", "points": [point, nxt], "color": "muted", "alpha": 0.10, "thickness": 1}})
    if weights.get("crystal", 0.0) or weights.get("quiet", 0.0):
        paths.append({{"type": "frame", "color": "accent", "alpha": 0.20, "thickness": 2}})
        paths.append({{"type": "circle", "x": 0.50, "y": 0.50, "radius": 0.34, "ring": True, "color": "accent", "alpha": 0.14}})
    return paths


def postprocess(image, width, height, args, palette, helpers):
    weights = PROFILE["weights"]
    rng = helpers["Rng"]("post:" + PROFILE["hash"])
    draw_line = helpers["draw_line"]
    draw_circle = helpers["draw_circle"]
    accent = palette["accent"]
    muted = palette["muted"]
    count = int(4 + 10 * max(weights.get("network", 0.0), weights.get("branch", 0.0), weights.get("wave", 0.0)))
    for _ in range(count):
        x0 = width * rng.random()
        y0 = height * rng.random()
        x1 = width * rng.random()
        y1 = height * rng.random()
        draw_line(image, width, height, x0, y0, x1, y1, muted, 0.035 + 0.045 * rng.random(), 1)
        if rng.random() > 0.55:
            draw_circle(image, width, height, x1, y1, width * (0.006 + 0.010 * rng.random()), accent, 0.16)
    return image
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Path to write agent_rules.py.")
    parser.add_argument("--prompt", default="", help="Natural-language visual prompt.")
    parser.add_argument("--title", default="Morphogenesis From Local Rules")
    parser.add_argument("--subtitle", default="Prompt-written visual rules")
    parser.add_argument("--profile-json", default=None, help="Optional path to write the derived profile JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = derive_profile(args.prompt, args.title, args.subtitle)
    out_path.write_text(rule_code(profile), encoding="utf-8")
    if args.profile_json:
        profile_path = Path(args.profile_json)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {profile_path}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
