"""Example agent-authored rules for morphogenesis-postcard.

This file is intentionally small and dependency-free. A model can write a file
with this same API during a skill run, then pass it with --rule-code.
"""

import math


def configure(args):
    return {
        "composition": "full-bleed",
        "motif": "none",
        "title_mode": "caption",
        "energy": 0.88,
        "contrast": 1.32,
        "grain": 0.12,
        "ink_density": 0.68,
        "accent_count": 14,
    }


def seed_rule(u, v, n, args):
    cx = n * 0.50
    cy = n * 0.50
    for y in range(2, n - 2):
        for x in range(2, n - 2):
            dx = x - cx
            dy = y - cy
            radius = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)
            spiral = math.sin(radius * 0.23 + angle * 5.0)
            diagonal = math.sin((x + y) * 0.085)
            if abs(spiral) < 0.10 or diagonal > 0.93:
                idx = y * n + x
                strength = max(0.0, 1.0 - radius / (n * 0.72))
                u[idx] = min(u[idx], 0.42 + 0.12 * (1.0 - strength))
                v[idx] = max(v[idx], 0.26 + 0.54 * strength)
    return u, v


def tone_rule(fx, fy, x, y, activator, depletion, edge, ridge, tone, context):
    n = context["n"]
    cx = n * 0.5
    cy = n * 0.5
    dx = fx - cx
    dy = fy - cy
    radial = math.hypot(dx, dy) / max(1.0, n)
    wave = math.sin(radial * 34.0 + math.atan2(dy, dx) * 7.0)
    interference = math.sin(fx * 0.18) * math.cos(fy * 0.11)
    custom = tone * 0.62 + abs(wave) * 0.19 + abs(interference) * 0.15 + edge * 0.22
    return max(0.0, min(1.0, custom))


def color_rule(r, g, b, tone, x, y, fx, fy, context):
    shimmer = 0.5 + 0.5 * math.sin(fx * 0.07 + fy * 0.13)
    return (
        min(255, int(r * (0.86 + 0.18 * shimmer))),
        min(255, int(g * (0.82 + 0.30 * tone))),
        min(255, int(b * (0.96 + 0.22 * shimmer))),
    )


def motif_paths(context):
    rng = context["Rng"]("agent-rules:" + str(context["args"].seed))
    paths = []
    for i in range(7):
        y0 = 0.12 + i * 0.115
        paths.append(
            {
                "type": "polyline",
                "points": [
                    (0.04, y0),
                    (0.18 + 0.10 * rng.random(), y0 + 0.08 * math.sin(i)),
                    (0.43 + 0.12 * rng.random(), y0 - 0.10 * math.cos(i * 0.7)),
                    (0.70 + 0.08 * rng.random(), y0 + 0.07 * math.sin(i * 1.7)),
                    (0.96, y0 + 0.05 * math.cos(i)),
                ],
                "color": "muted",
                "alpha": 0.18,
                "thickness": 2,
            }
        )
    paths.append({"type": "circle", "x": 0.50, "y": 0.50, "radius": 0.36, "ring": True, "color": "accent", "alpha": 0.20, "thickness": 2})
    paths.append({"type": "circle", "x": 0.50, "y": 0.50, "radius": 0.18, "ring": True, "color": "ink", "alpha": 0.10, "thickness": 1})
    return paths


def postprocess(image, width, height, args, palette, helpers):
    rng = helpers["Rng"]("post:" + str(args.seed))
    draw_line = helpers["draw_line"]
    draw_circle = helpers["draw_circle"]
    accent = palette["accent"]
    muted = palette["muted"]
    for _ in range(10):
        x0 = width * rng.random()
        y0 = height * rng.random()
        x1 = width * rng.random()
        y1 = height * rng.random()
        draw_line(image, width, height, x0, y0, x1, y1, muted, 0.08, 1)
        if rng.random() > 0.55:
            draw_circle(image, width, height, x1, y1, width * (0.008 + 0.010 * rng.random()), accent, 0.22)
    return image
