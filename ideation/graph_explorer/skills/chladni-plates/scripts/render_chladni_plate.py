#!/usr/bin/env python3
"""Render Chladni plate resonance patterns and animations.

The model is intentionally lightweight: analytic plate mode superpositions are
used to create nodal-line patterns, high-resolution images, and GIF movies
without requiring a finite-element solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-chladni-plates")


try:
    from scipy.special import jv, jn_zeros
except Exception:  # pragma: no cover - fallback is used when scipy is absent.
    jv = None
    jn_zeros = None


PALETTES: dict[str, dict[str, tuple[float, float, float]]] = {
    "neon-noir": {
        "low": (0.015, 0.012, 0.030),
        "mid": (0.070, 0.095, 0.180),
        "high": (0.050, 0.740, 0.960),
        "node": (1.000, 0.820, 0.250),
        "accent": (0.960, 0.180, 0.780),
    },
    "copper-glass": {
        "low": (0.030, 0.026, 0.024),
        "mid": (0.210, 0.125, 0.070),
        "high": (0.850, 0.470, 0.220),
        "node": (0.980, 0.850, 0.580),
        "accent": (0.420, 0.820, 0.900),
    },
    "blue-sand": {
        "low": (0.018, 0.030, 0.060),
        "mid": (0.090, 0.200, 0.380),
        "high": (0.420, 0.820, 1.000),
        "node": (0.940, 0.880, 0.650),
        "accent": (0.980, 0.980, 1.000),
    },
    "emerald-gold": {
        "low": (0.010, 0.035, 0.028),
        "mid": (0.040, 0.260, 0.150),
        "high": (0.320, 0.900, 0.620),
        "node": (1.000, 0.820, 0.250),
        "accent": (0.860, 1.000, 0.830),
    },
    "ice-fire": {
        "low": (0.010, 0.018, 0.035),
        "mid": (0.080, 0.210, 0.430),
        "high": (0.950, 0.250, 0.080),
        "node": (0.920, 0.980, 1.000),
        "accent": (1.000, 0.760, 0.240),
    },
    "monochrome": {
        "low": (0.015, 0.015, 0.015),
        "mid": (0.230, 0.230, 0.230),
        "high": (0.760, 0.760, 0.760),
        "node": (0.980, 0.960, 0.900),
        "accent": (0.620, 0.620, 0.620),
    },
}


PRESETS: dict[str, dict[str, object]] = {
    "classic-square": {
        "shape": "square",
        "boundary": "free",
        "modes": "2,3:1.0:0;3,2:-1.0:0;5,5:0.28:0.7",
        "palette": "blue-sand",
        "node_width": 0.060,
        "description": "classic crossed nodal curves on a square plate",
    },
    "cathedral-window": {
        "shape": "square",
        "boundary": "free",
        "modes": "4,7:1.0:0;7,4:-0.90:0.4;3,10:0.34:1.1",
        "palette": "neon-noir",
        "node_width": 0.050,
        "description": "high-order square-plate interference with stained-glass symmetry",
    },
    "bronze-drum": {
        "shape": "circle",
        "boundary": "free",
        "modes": "6,2:1.0:0;11,1:0.32:0.8;3,4:0.28:1.5",
        "palette": "copper-glass",
        "node_width": 0.055,
        "description": "radial and angular disk modes resembling a bronze resonator",
    },
    "radial-bloom": {
        "shape": "circle",
        "boundary": "free",
        "modes": "8,2:1.0:0;4,4:0.52:0.9;12,1:0.25:1.4",
        "palette": "emerald-gold",
        "node_width": 0.050,
        "description": "flower-like disk pattern with radial nodal petals",
    },
    "long-bridge": {
        "shape": "rectangle",
        "aspect": 1.65,
        "boundary": "free",
        "modes": "2,5:1.0:0;6,3:0.72:0.5;9,2:0.30:1.1",
        "palette": "ice-fire",
        "node_width": 0.055,
        "description": "rectangular plate with long-span standing waves",
    },
    "quiet-glass": {
        "shape": "circle",
        "boundary": "free",
        "modes": "5,3:1.0:0;10,1:0.20:0.6;2,5:0.18:1.7",
        "palette": "monochrome",
        "node_width": 0.045,
        "description": "subtle glass-plate nodal lines for clean slides",
    },
}


def stable_seed(value: str | int) -> int:
    if isinstance(value, int):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**32)


def parse_modes(value: str) -> list[tuple[int, int, float, float]]:
    modes: list[tuple[int, int, float, float]] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        pair = parts[0].split(",")
        if len(pair) != 2:
            raise argparse.ArgumentTypeError(f"Invalid mode pair: {item}")
        try:
            a = int(pair[0])
            b = int(pair[1])
            amp = float(parts[1]) if len(parts) >= 2 and parts[1] else 1.0
            phase = float(parts[2]) if len(parts) >= 3 and parts[2] else 0.0
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid mode specification: {item}") from exc
        if a < 0 or b <= 0:
            raise argparse.ArgumentTypeError("Mode indices must be nonnegative for circular modes and positive otherwise")
        modes.append((a, b, amp, phase))
    if not modes:
        raise argparse.ArgumentTypeError("At least one mode is required")
    return modes


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "custom":
        return
    preset = PRESETS[args.preset]
    for key in ("shape", "boundary", "modes", "palette", "aspect", "node_width"):
        if getattr(args, key, None) in (None, "") and key in preset:
            setattr(args, key, preset[key])


def make_grid(shape: str, size: int, aspect: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if shape == "rectangle":
        width = max(4, int(round(size * aspect)))
        height = size
    else:
        width = height = size
        aspect = 1.0

    x = np.linspace(-aspect, aspect, width, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    if shape == "circle":
        rr = np.sqrt(xx * xx + yy * yy)
        mask = rr <= 0.985
    else:
        mask = np.ones_like(xx, dtype=bool)
    return xx, yy, mask


def rectangular_mode(xx: np.ndarray, yy: np.ndarray, m: int, n: int, boundary: str, aspect: float) -> np.ndarray:
    x01 = (xx / aspect + 1.0) * 0.5
    y01 = (yy + 1.0) * 0.5
    if boundary == "fixed":
        return np.sin(m * math.pi * x01) * np.sin(n * math.pi * y01)
    if boundary == "mixed":
        return np.cos(m * math.pi * x01) * np.sin(n * math.pi * y01)
    return np.cos(m * math.pi * x01) * np.cos(n * math.pi * y01)


def bessel_value(order: int, radial_index: int, r: np.ndarray) -> np.ndarray:
    if jv is not None and jn_zeros is not None:
        try:
            root = float(jn_zeros(order, radial_index)[-1])
            return jv(order, root * r)
        except Exception:
            pass
    root = (radial_index + order * 0.5 - 0.25) * math.pi
    radial = np.sin(root * r + order * math.pi * 0.25)
    return radial * np.power(np.clip(1.0 - r * 0.12, 0.0, 1.0), 0.7)


def circular_mode(xx: np.ndarray, yy: np.ndarray, order: int, radial: int, phase: float) -> np.ndarray:
    rr = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)
    return bessel_value(order, radial, rr) * np.cos(order * theta + phase)


def mode_frequency(shape: str, first: int, second: int, aspect: float) -> float:
    if shape == "circle":
        return max(0.35, 0.45 * first + 0.95 * second)
    return max(0.35, math.sqrt((first / max(aspect, 1.0e-6)) ** 2 + second**2))


def compute_field(
    xx: np.ndarray,
    yy: np.ndarray,
    modes: list[tuple[int, int, float, float]],
    *,
    shape: str,
    boundary: str,
    aspect: float,
    time_phase: float,
    drive: float,
) -> np.ndarray:
    field = np.zeros_like(xx, dtype=np.float64)
    for first, second, amp, phase in modes:
        if shape == "circle":
            spatial = circular_mode(xx, yy, first, second, phase)
        else:
            spatial = rectangular_mode(xx, yy, max(first, 1), max(second, 1), boundary, aspect)
        omega = mode_frequency(shape, max(first, 1), max(second, 1), aspect)
        field += amp * spatial * math.cos(drive * omega * time_phase + phase)
    return field


def normalize_field(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.abs(field[mask])
    scale = float(np.percentile(valid, 99.4)) if valid.size else 1.0
    if scale <= 1.0e-12:
        scale = 1.0
    return np.clip(field / scale, -1.25, 1.25)


def mix(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a * (1.0 - t[..., None]) + b * t[..., None]


def palette_image(
    field: np.ndarray,
    mask: np.ndarray,
    *,
    palette_name: str,
    node_width: float,
    grain: float,
    rng: np.random.Generator,
    plate_alpha: float,
) -> np.ndarray:
    palette = PALETTES[palette_name]
    low = np.asarray(palette["low"], dtype=np.float64)
    mid = np.asarray(palette["mid"], dtype=np.float64)
    high = np.asarray(palette["high"], dtype=np.float64)
    node_color = np.asarray(palette["node"], dtype=np.float64)
    accent = np.asarray(palette["accent"], dtype=np.float64)

    tone = np.clip((field + 1.0) * 0.5, 0.0, 1.0)
    lower = mix(low, mid, np.clip(tone * 2.0, 0.0, 1.0))
    upper = mix(mid, high, np.clip((tone - 0.5) * 2.0, 0.0, 1.0))
    base = np.where((tone[..., None] < 0.5), lower, upper)

    abs_field = np.abs(field)
    nodes = np.exp(-((abs_field / max(node_width, 1.0e-4)) ** 2.0))
    ridge = 0.5 + 0.5 * np.cos(34.0 * np.clip(field, -1.0, 1.0))
    fine = rng.random(field.shape)
    sand = np.clip(nodes**0.72 * (0.72 + 0.38 * fine + 0.10 * ridge), 0.0, 1.0)
    glow = np.clip(nodes**0.35 * 0.38, 0.0, 0.62)

    image = base * (0.68 + 0.30 * (1.0 - nodes[..., None]))
    image = mix(image, accent, glow * 0.42)
    image = mix(image, node_color, sand)

    plate = np.asarray(palette["mid"], dtype=np.float64) * 0.28 + np.asarray(palette["low"], dtype=np.float64) * 0.72
    image = mix(plate[None, None, :], image, plate_alpha * mask.astype(np.float64))
    image[~mask] = low * 0.18

    if grain > 0.0:
        noise = rng.normal(0.0, grain * 0.055, image.shape)
        image = np.clip(image + noise * mask[..., None], 0.0, 1.0)

    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def draw_outline(image: np.ndarray, mask: np.ndarray, palette_name: str) -> np.ndarray:
    palette = PALETTES[palette_name]
    accent = np.asarray(palette["accent"], dtype=np.float64)
    padded = np.pad(mask.astype(np.int16), 1, mode="constant")
    neighbors = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, :-2]
        + padded[:-2, 2:]
        + padded[2:, :-2]
        + padded[2:, 2:]
    )
    edge = mask & (neighbors < 8)
    out = image.copy()
    out[edge] = np.clip(0.70 * out[edge].astype(np.float64) + 0.30 * accent * 255.0, 0, 255).astype(np.uint8)
    return out


def save_png(path: Path, image: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(image).save(path)
        return
    except Exception:
        pass
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.imsave(path, image)


def write_gif(frame_paths: list[Path], gif_path: Path, fps: int) -> str:
    duration = 1.0 / max(1, fps)
    try:
        import imageio.v2 as imageio

        frames = [imageio.imread(path) for path in frame_paths]
        imageio.mimsave(gif_path, frames, duration=duration)
        return "imageio"
    except Exception:
        pass
    from PIL import Image

    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(duration * 1000),
        loop=0,
        optimize=True,
    )
    return "pillow"


def write_caption(args: argparse.Namespace, preset_description: str, out_dir: Path) -> None:
    caption = (
        f"Chladni plate resonance: {preset_description}. "
        f"Shape={args.shape}, boundary={args.boundary}, modes={args.modes}, palette={args.palette}. "
        "Bright sand-like ridges mark nodal lines where the oscillating plate is nearly still."
    )
    (out_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")


def write_readme(args: argparse.Namespace, out_dir: Path, artifacts: dict[str, object], preset_description: str) -> None:
    readme = f"""# Chladni Plate Render

This run visualizes standing-wave resonance on a thin plate using analytic mode
superposition. Bright ridges indicate nodal lines where sand would collect in a
classic Chladni demonstration.

## Configuration

- preset: `{args.preset}`
- description: {preset_description}
- shape: `{args.shape}`
- boundary: `{args.boundary}`
- modes: `{args.modes}`
- palette: `{args.palette}`
- seed: `{args.seed}`
- image size: `{args.size}`
- frames: `{args.frames}`
- fps: `{args.fps}`

## Outputs

- `chladni_pattern.png`: high-resolution static nodal pattern.
- `chladni_preview.png`: first animation frame.
- `chladni_animation.gif`: resonance animation.
- `frames/frame_*.png`: individual animation frames.
- `field_data.npz`: final normalized field and mask.
- `parameters.json`: exact parameters.
- `caption.txt`: short explanation.
"""
    if artifacts.get("gif") is None:
        readme = readme.replace("- `chladni_animation.gif`: resonance animation.\n", "")
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create high-resolution Chladni plate pattern images and GIF animations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--preset", choices=sorted(PRESETS) + ["custom"], default="classic-square")
    parser.add_argument("--shape", choices=["square", "rectangle", "circle"], default=None)
    parser.add_argument("--boundary", choices=["free", "fixed", "mixed"], default=None)
    parser.add_argument("--modes", default=None, help="Semicolon-separated modes: m,n:amplitude:phase.")
    parser.add_argument("--palette", choices=sorted(PALETTES), default=None)
    parser.add_argument("--aspect", type=float, default=None, help="Width/height for rectangular plates.")
    parser.add_argument("--seed", default="42", help="Deterministic seed string or integer.")
    parser.add_argument("--size", type=int, default=1080, help="Pixel height for image and movie.")
    parser.add_argument("--frames", type=int, default=48, help="Animation frames.")
    parser.add_argument("--fps", type=int, default=18, help="GIF frame rate.")
    parser.add_argument("--drive", type=float, default=1.0, help="Relative temporal drive strength.")
    parser.add_argument("--node-width", type=float, default=None, help="Width of bright nodal ridges.")
    parser.add_argument("--grain", type=float, default=0.20, help="Deterministic sand/film grain, 0-1.")
    parser.add_argument("--plate-alpha", type=float, default=1.0, help="Plate opacity against background, 0-1.")
    parser.add_argument("--no-gif", action="store_true", help="Write PNGs and data but skip GIF assembly.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_preset(args)

    if args.shape is None:
        args.shape = "square"
    if args.boundary is None:
        args.boundary = "free"
    if args.modes is None:
        args.modes = str(PRESETS["classic-square"]["modes"])
    if args.palette is None:
        args.palette = "blue-sand"
    if args.aspect is None:
        args.aspect = 1.0
    if args.node_width is None:
        args.node_width = 0.058
    if args.size < 128:
        parser.error("--size must be at least 128")
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if args.aspect <= 0:
        parser.error("--aspect must be positive")

    modes = parse_modes(args.modes)
    seed = stable_seed(str(args.seed))
    rng = np.random.default_rng(seed)
    out_dir = Path(args.out).expanduser().resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    preset_description = (
        str(PRESETS[args.preset]["description"]) if args.preset in PRESETS else "custom plate-mode superposition"
    )

    if not args.quiet:
        print(f"Rendering Chladni plate: {args.preset}")
        print(f"Output: {out_dir}")

    xx, yy, mask = make_grid(args.shape, args.size, args.aspect)
    static_field = compute_field(
        xx,
        yy,
        modes,
        shape=args.shape,
        boundary=args.boundary,
        aspect=args.aspect,
        time_phase=0.0,
        drive=args.drive,
    )
    static_field = normalize_field(static_field, mask)
    static_image = palette_image(
        static_field,
        mask,
        palette_name=args.palette,
        node_width=args.node_width,
        grain=args.grain,
        rng=rng,
        plate_alpha=args.plate_alpha,
    )
    static_image = draw_outline(static_image, mask, args.palette)
    static_path = out_dir / "chladni_pattern.png"
    save_png(static_path, static_image)

    frame_paths: list[Path] = []
    for frame_idx in range(args.frames):
        phase = 2.0 * math.pi * frame_idx / args.frames
        frame_field = compute_field(
            xx,
            yy,
            modes,
            shape=args.shape,
            boundary=args.boundary,
            aspect=args.aspect,
            time_phase=phase,
            drive=args.drive,
        )
        frame_field = normalize_field(frame_field, mask)
        frame_rng = np.random.default_rng(seed + frame_idx * 17)
        frame_image = palette_image(
            frame_field,
            mask,
            palette_name=args.palette,
            node_width=args.node_width,
            grain=args.grain,
            rng=frame_rng,
            plate_alpha=args.plate_alpha,
        )
        frame_image = draw_outline(frame_image, mask, args.palette)
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        save_png(frame_path, frame_image)
        frame_paths.append(frame_path)

    preview_path = out_dir / "chladni_preview.png"
    save_png(preview_path, static_image)

    gif_backend = None
    gif_path = out_dir / "chladni_animation.gif"
    if not args.no_gif:
        gif_backend = write_gif(frame_paths, gif_path, args.fps)

    data_path = out_dir / "field_data.npz"
    np.savez_compressed(
        data_path,
        x=xx,
        y=yy,
        mask=mask,
        field=static_field,
        modes=np.asarray(modes, dtype=np.float64),
    )

    artifacts = {
        "pattern_png": str(static_path),
        "preview_png": str(preview_path),
        "gif": str(gif_path) if gif_backend else None,
        "frames_dir": str(frames_dir),
        "field_data": str(data_path),
        "parameters": str(out_dir / "parameters.json"),
        "caption": str(out_dir / "caption.txt"),
        "readme": str(out_dir / "README.md"),
    }
    parameters = {
        "preset": args.preset,
        "preset_description": preset_description,
        "shape": args.shape,
        "boundary": args.boundary,
        "modes": args.modes,
        "parsed_modes": modes,
        "palette": args.palette,
        "aspect": args.aspect,
        "seed": args.seed,
        "resolved_seed": seed,
        "size": args.size,
        "frames": args.frames,
        "fps": args.fps,
        "drive": args.drive,
        "node_width": args.node_width,
        "grain": args.grain,
        "plate_alpha": args.plate_alpha,
        "gif_backend": gif_backend,
        "artifacts": artifacts,
    }
    (out_dir / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    write_caption(args, preset_description, out_dir)
    write_readme(args, out_dir, artifacts, preset_description)

    print("\nChladni render complete")
    print(f"Output directory: {out_dir}")
    print(f"Pattern PNG: {static_path}")
    print(f"Preview PNG: {preview_path}")
    if gif_backend:
        print(f"GIF: {gif_path}")
    print(f"Frames: {frames_dir}")
    print(f"Parameters: {out_dir / 'parameters.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
