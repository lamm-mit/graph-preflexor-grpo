#!/usr/bin/env python3
"""Create a social-post-ready Gray-Scott reaction-diffusion poster."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-reaction-diffusion")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


PRESETS = {
    "labyrinth": {"du": 0.16, "dv": 0.08, "feed": 0.037, "kill": 0.060, "noise": 0.020},
    "spots": {"du": 0.16, "dv": 0.08, "feed": 0.055, "kill": 0.062, "noise": 0.018},
    "coral": {"du": 0.18, "dv": 0.07, "feed": 0.026, "kill": 0.051, "noise": 0.025},
    "membranes": {"du": 0.14, "dv": 0.065, "feed": 0.030, "kill": 0.057, "noise": 0.020},
    "veins": {"du": 0.20, "dv": 0.070, "feed": 0.034, "kill": 0.056, "noise": 0.018},
    "turbulence": {"du": 0.12, "dv": 0.060, "feed": 0.044, "kill": 0.063, "noise": 0.030},
}

PALETTES = {
    "magma-cyan": ["#05060a", "#1b1233", "#68206d", "#c43c5c", "#ffb85c", "#c6fff2"],
    "biofilm": ["#06120d", "#123524", "#35724a", "#96b85a", "#f4dc77", "#fff6c4"],
    "noir-neon": ["#02030a", "#151231", "#38246b", "#006b91", "#00d4ff", "#fff9e6"],
    "graphite-fire": ["#070707", "#252525", "#51453c", "#a85f32", "#ff9b4a", "#ffe1a3"],
    "ice": ["#06131f", "#123550", "#3c7fa4", "#7dc8d8", "#d9fbff", "#ffffff"],
}

ACCENTS = {
    "magma-cyan": "#7fffee",
    "biofilm": "#f4dc77",
    "noir-neon": "#00d4ff",
    "graphite-fire": "#ffb35c",
    "ice": "#d9fbff",
}


def laplacian(field: np.ndarray) -> np.ndarray:
    return (
        -field
        + 0.20
        * (
            np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
        )
        + 0.05
        * (
            np.roll(np.roll(field, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(field, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(field, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(field, -1, axis=0), -1, axis=1)
        )
    )


def initialize(size: int, seed: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = np.ones((size, size), dtype=np.float32)
    v = np.zeros((size, size), dtype=np.float32)

    u += rng.normal(0, noise, size=(size, size)).astype(np.float32)
    v += rng.normal(0, noise, size=(size, size)).astype(np.float32)

    yy, xx = np.mgrid[0:size, 0:size]
    for _idx in range(18):
        cx = rng.uniform(0.16 * size, 0.84 * size)
        cy = rng.uniform(0.16 * size, 0.84 * size)
        radius = rng.uniform(0.025 * size, 0.080 * size)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
        u[mask] = rng.uniform(0.35, 0.55)
        v[mask] = rng.uniform(0.22, 0.36)

    # Add a soft radial seed so each image has a strong center of gravity.
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / size
    v += (0.09 * np.exp(-(r / 0.23) ** 2)).astype(np.float32)

    return np.clip(u, 0, 1), np.clip(v, 0, 1)


def simulate(size: int, steps: int, seed: int, preset: str) -> tuple[np.ndarray, np.ndarray, dict]:
    params = dict(PRESETS[preset])
    du = params["du"]
    dv = params["dv"]
    feed = params["feed"]
    kill = params["kill"]
    u, v = initialize(size, seed, params["noise"])

    for _step in range(steps):
        uvv = u * v * v
        u += du * laplacian(u) - uvv + feed * (1.0 - u)
        v += dv * laplacian(v) + uvv - (feed + kill) * v
        np.clip(u, 0, 1, out=u)
        np.clip(v, 0, 1, out=v)

    return u, v, params


def poster_field(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    field = v - 0.45 * u
    field = field - np.percentile(field, 1)
    field = field / max(1e-8, np.percentile(field, 99) - np.percentile(field, 1))
    return np.clip(field, 0, 1)


def make_colormap(name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, PALETTES[name], N=512)


def add_text_box(ax: plt.Axes, text: str, x: float, y: float, width: float) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#f7f4ec",
        fontsize=9.5,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.12",
            "facecolor": "#05060acc",
            "edgecolor": "#ffffff33",
            "linewidth": 0.8,
        },
        wrap=True,
    )


def render_posters(
    field: np.ndarray,
    *,
    out_dir: Path,
    seed: int,
    preset: str,
    palette: str,
    title: str,
    subtitle: str,
    params: dict,
    dpi: int,
) -> tuple[Path, Path]:
    cmap = make_colormap(palette)
    accent = ACCENTS[palette]
    levels = np.linspace(0.22, 0.86, 9)
    caption = (
        "Local activator-inhibitor chemistry is enough to create global order: "
        "spots, ridges, and labyrinths emerge from diffusion plus reaction."
    )

    poster_path = out_dir / "reaction_diffusion_poster.png"
    square_path = out_dir / "reaction_diffusion_poster_square.png"

    fig = plt.figure(figsize=(8, 10), dpi=dpi, facecolor="#05060a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(field, cmap=cmap, interpolation="bilinear", origin="lower")
    ax.contour(field, levels=levels, colors=accent, linewidths=0.22, alpha=0.38)

    ax.text(
        0.055,
        0.945,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#fffaf0",
        fontsize=28,
        fontweight="bold",
    )
    ax.text(
        0.058,
        0.902,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=accent,
        fontsize=13,
        fontweight="medium",
    )

    add_text_box(
        ax,
        "Dynamics\n"
        "Gray-Scott reaction-diffusion\n"
        f"preset: {preset}\n"
        f"F={params['feed']:.3f}, k={params['kill']:.3f}",
        0.060,
        0.205,
        0.30,
    )
    add_text_box(
        ax,
        "Interpretation\n"
        "morphology from local rules\n"
        "diffusion -> instability\n"
        "instability -> visible hierarchy",
        0.625,
        0.205,
        0.30,
    )
    ax.text(
        0.055,
        0.055,
        f"{caption}  seed={seed} | palette={palette}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#fffaf0cc",
        fontsize=8.5,
    )
    fig.savefig(poster_path, dpi=dpi, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8), dpi=dpi, facecolor="#05060a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(field, cmap=cmap, interpolation="bilinear", origin="lower")
    ax.contour(field, levels=levels, colors=accent, linewidths=0.22, alpha=0.36)
    ax.text(
        0.055,
        0.925,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#fffaf0",
        fontsize=25,
        fontweight="bold",
    )
    ax.text(
        0.058,
        0.875,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=accent,
        fontsize=12,
    )
    ax.text(
        0.055,
        0.055,
        f"Gray-Scott dynamics | {preset} | seed {seed}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#fffaf0cc",
        fontsize=8.5,
    )
    fig.savefig(square_path, dpi=dpi, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)

    return poster_path, square_path


def write_text_outputs(out_dir: Path, args: argparse.Namespace, params: dict) -> None:
    caption = (
        f"{args.title}\n\n"
        "A Gray-Scott reaction-diffusion system turns local activator-inhibitor "
        "rules into global morphology: spots, ridges, channels, and labyrinths. "
        f"Preset: {args.preset}. Seed: {args.seed}."
    )
    (out_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")

    readme = f"""# {args.title}

This poster was generated from a Gray-Scott reaction-diffusion simulation.
Two virtual chemicals diffuse at different rates and react locally. Depending
on feed and kill rates, tiny perturbations grow into Turing-like motifs such as
spots, membranes, vascular channels, and labyrinths.

## Parameters

- preset: `{args.preset}`
- palette: `{args.palette}`
- seed: `{args.seed}`
- grid size: `{args.size}`
- simulation steps: `{args.steps}`
- diffusion U: `{params['du']}`
- diffusion V: `{params['dv']}`
- feed: `{params['feed']}`
- kill: `{params['kill']}`

## Files

- `reaction_diffusion_poster.png`: 4:5 social-post image.
- `reaction_diffusion_poster_square.png`: square social-post image.
- `reaction_diffusion_data.npz`: compressed final U/V fields and rendered field.
- `parameters.json`: exact parameters used.
- `caption.txt`: short social caption.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="labyrinth")
    parser.add_argument("--palette", choices=sorted(PALETTES), default="magma-cyan")
    parser.add_argument("--title", default="Morphogenesis From Local Rules")
    parser.add_argument("--subtitle", default="Gray-Scott reaction-diffusion dynamics")
    parser.add_argument("--size", type=int, default=384, help="Simulation grid width/height.")
    parser.add_argument("--steps", type=int, default=2600)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    u, v, params = simulate(args.size, args.steps, args.seed, args.preset)
    field = poster_field(u, v)
    poster_path, square_path = render_posters(
        field,
        out_dir=args.out,
        seed=args.seed,
        preset=args.preset,
        palette=args.palette,
        title=args.title,
        subtitle=args.subtitle,
        params=params,
        dpi=args.dpi,
    )

    np.savez_compressed(args.out / "reaction_diffusion_data.npz", u=u, v=v, field=field)
    parameter_record = {
        "seed": args.seed,
        "preset": args.preset,
        "palette": args.palette,
        "title": args.title,
        "subtitle": args.subtitle,
        "size": args.size,
        "steps": args.steps,
        "dpi": args.dpi,
        "model": "Gray-Scott reaction-diffusion",
        "parameters": params,
        "outputs": {
            "poster": str(poster_path),
            "square": str(square_path),
            "data": str(args.out / "reaction_diffusion_data.npz"),
        },
    }
    (args.out / "parameters.json").write_text(json.dumps(parameter_record, indent=2), encoding="utf-8")
    write_text_outputs(args.out, args, params)

    for path in sorted(args.out.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
