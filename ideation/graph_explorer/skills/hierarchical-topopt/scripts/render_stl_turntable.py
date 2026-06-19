#!/usr/bin/env python3
"""Render an STL mesh as a polished turntable GIF.

This helper is intentionally standalone: it can render STLs produced by the
hierarchical-topopt skill, or any existing ASCII/binary STL file passed on the
command line. It uses Matplotlib's headless Agg backend plus imageio/Pillow when
available for GIF assembly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stl-turntable")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


MATERIALS: dict[str, dict[str, object]] = {
    "titanium": {
        "base": (0.70, 0.72, 0.72),
        "highlight": (1.00, 0.98, 0.90),
        "edge": (0.10, 0.12, 0.13),
        "ambient": 0.30,
        "diffuse": 0.78,
        "specular": 0.34,
        "shininess": 42.0,
    },
    "graphite": {
        "base": (0.16, 0.17, 0.18),
        "highlight": (0.82, 0.88, 0.90),
        "edge": (0.02, 0.02, 0.02),
        "ambient": 0.36,
        "diffuse": 0.70,
        "specular": 0.24,
        "shininess": 28.0,
    },
    "ceramic": {
        "base": (0.86, 0.84, 0.78),
        "highlight": (1.00, 0.98, 0.92),
        "edge": (0.35, 0.34, 0.30),
        "ambient": 0.34,
        "diffuse": 0.72,
        "specular": 0.18,
        "shininess": 18.0,
    },
    "polymer": {
        "base": (0.20, 0.52, 0.82),
        "highlight": (0.75, 0.92, 1.00),
        "edge": (0.04, 0.15, 0.22),
        "ambient": 0.32,
        "diffuse": 0.76,
        "specular": 0.20,
        "shininess": 24.0,
    },
    "gold": {
        "base": (0.94, 0.67, 0.23),
        "highlight": (1.00, 0.95, 0.70),
        "edge": (0.30, 0.18, 0.04),
        "ambient": 0.30,
        "diffuse": 0.76,
        "specular": 0.42,
        "shininess": 48.0,
    },
    "blue-steel": {
        "base": (0.34, 0.50, 0.68),
        "highlight": (0.82, 0.92, 1.00),
        "edge": (0.05, 0.08, 0.11),
        "ambient": 0.30,
        "diffuse": 0.76,
        "specular": 0.36,
        "shininess": 38.0,
    },
}

BACKGROUNDS: dict[str, dict[str, object]] = {
    "studio": {
        "figure": (0.93, 0.94, 0.94),
        "axes": (0.93, 0.94, 0.94),
        "text": (0.10, 0.11, 0.12),
        "shadow": (0.10, 0.11, 0.12, 0.16),
    },
    "dark": {
        "figure": (0.025, 0.030, 0.035),
        "axes": (0.025, 0.030, 0.035),
        "text": (0.90, 0.92, 0.92),
        "shadow": (0.00, 0.00, 0.00, 0.34),
    },
    "light": {
        "figure": (1.00, 1.00, 0.98),
        "axes": (1.00, 1.00, 0.98),
        "text": (0.12, 0.12, 0.12),
        "shadow": (0.12, 0.12, 0.12, 0.14),
    },
}


def parse_figsize(value: str) -> tuple[float, float]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use WIDTH,HEIGHT, for example 7,6")
    try:
        width = float(parts[0].strip())
        height = float(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Figure size must contain numbers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Figure size must be positive")
    return width, height


def read_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        tri_count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + tri_count * 50
        if expected == len(data):
            return read_binary_stl(data, tri_count)
    return read_ascii_stl(data, path)


def read_binary_stl(data: bytes, tri_count: int) -> np.ndarray:
    triangles = np.empty((tri_count, 3, 3), dtype=np.float64)
    offset = 84
    record = struct.Struct("<12fH")
    for idx in range(tri_count):
        values = record.unpack_from(data, offset)
        triangles[idx, 0, :] = values[3:6]
        triangles[idx, 1, :] = values[6:9]
        triangles[idx, 2, :] = values[9:12]
        offset += 50
    return clean_triangles(triangles)


def read_ascii_stl(data: bytes, path: Path) -> np.ndarray:
    try:
        text = data.decode("utf-8", errors="replace")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not a valid binary or ASCII STL") from exc

    vertices: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("vertex "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue

    if len(vertices) < 3 or len(vertices) % 3 != 0:
        raise ValueError(f"{path} does not contain a valid STL triangle list")
    return clean_triangles(np.asarray(vertices, dtype=np.float64).reshape((-1, 3, 3)))


def clean_triangles(triangles: np.ndarray) -> np.ndarray:
    if triangles.size == 0:
        raise ValueError("STL contains no triangles")
    edges_a = triangles[:, 1, :] - triangles[:, 0, :]
    edges_b = triangles[:, 2, :] - triangles[:, 0, :]
    areas = 0.5 * np.linalg.norm(np.cross(edges_a, edges_b), axis=1)
    triangles = triangles[areas > 1.0e-14]
    if triangles.size == 0:
        raise ValueError("STL contains only degenerate triangles")
    if not np.isfinite(triangles).all():
        raise ValueError("STL contains non-finite coordinates")
    return triangles


def normalize_mesh(triangles: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    points = triangles.reshape((-1, 3))
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    span = bounds_max - bounds_min
    scale = float(max(span.max(), 1.0e-12))
    normalized = (triangles - center) / scale
    metadata = {
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "span": span.tolist(),
        "center": center.tolist(),
        "normalization_scale": scale,
    }
    return normalized, metadata


def downsample_faces(triangles: np.ndarray, max_faces: int) -> tuple[np.ndarray, bool]:
    if max_faces <= 0 or triangles.shape[0] <= max_faces:
        return triangles, False
    indices = np.linspace(0, triangles.shape[0] - 1, max_faces).astype(int)
    return triangles[indices], True


def rotate_z(triangles: np.ndarray, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return triangles @ rotation.T


def face_normals(triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :])
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0.0] = 1.0
    return normals / lengths[:, None]


def normalized(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vector), dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return arr
    return arr / norm


def shaded_facecolors(triangles: np.ndarray, material_name: str) -> np.ndarray:
    material = MATERIALS[material_name]
    base = np.asarray(material["base"], dtype=np.float64)
    highlight = np.asarray(material["highlight"], dtype=np.float64)
    ambient = float(material["ambient"])
    diffuse_strength = float(material["diffuse"])
    specular_strength = float(material["specular"])
    shininess = float(material["shininess"])

    normals = face_normals(triangles)
    light_dir = normalized((0.30, -0.50, 0.82))
    fill_dir = normalized((-0.70, 0.30, 0.55))
    view_dir = normalized((0.20, -0.62, 0.76))
    half_vec = normalized(light_dir + view_dir)

    diffuse = np.clip(normals @ light_dir, 0.0, 1.0)
    fill = np.clip(normals @ fill_dir, 0.0, 1.0)
    specular = np.clip(normals @ half_vec, 0.0, 1.0) ** shininess
    shade = ambient + diffuse_strength * diffuse + 0.20 * fill
    colors = base[None, :] * shade[:, None] + highlight[None, :] * (specular_strength * specular[:, None])
    colors = np.clip(colors, 0.0, 1.0)
    return np.column_stack([colors, np.ones(colors.shape[0])])


def make_shadow(zmin: float, radius: float = 0.46, points: int = 80) -> list[tuple[float, float, float]]:
    angles = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
    return [(radius * math.cos(a), radius * 0.62 * math.sin(a), zmin) for a in angles]


def style_axis(ax, background: dict[str, object], limit: float, z_limit: float) -> None:
    ax.set_facecolor(background["axes"])
    ax.set_xlim((-limit, limit))
    ax.set_ylim((-limit, limit))
    ax.set_zlim((-z_limit, z_limit))
    ax.set_box_aspect((1, 1, max(0.32, z_limit / limit)))
    ax.grid(False)
    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_color((1, 1, 1, 0))


def render_frame(
    triangles: np.ndarray,
    angle_rad: float,
    output_path: Path,
    *,
    args: argparse.Namespace,
    limit: float,
    z_limit: float,
) -> None:
    background = BACKGROUNDS[args.background]
    material = MATERIALS[args.material]
    rotated = rotate_z(triangles, angle_rad)
    colors = shaded_facecolors(rotated, args.material)

    fig = plt.figure(figsize=args.figsize, dpi=args.dpi)
    fig.patch.set_facecolor(background["figure"])
    ax = fig.add_subplot(111, projection="3d")
    style_axis(ax, background, limit=limit, z_limit=z_limit)

    shadow = Poly3DCollection(
        [make_shadow(-z_limit * 0.96)],
        facecolors=[background["shadow"]],
        edgecolors=[(0, 0, 0, 0)],
        linewidths=0.0,
    )
    ax.add_collection3d(shadow)

    edge_rgb = tuple(float(v) for v in material["edge"])
    collection = Poly3DCollection(
        rotated,
        facecolors=colors,
        edgecolors=[(*edge_rgb, args.edge_alpha)],
        linewidths=args.edge_width,
        antialiaseds=True,
    )
    collection.set_sort_zpos(0.0)
    ax.add_collection3d(collection)
    ax.view_init(elev=args.elev, azim=args.azim)

    if args.title:
        fig.text(
            0.5,
            0.935,
            args.title,
            ha="center",
            va="center",
            color=background["text"],
            fontsize=args.title_size,
            fontweight="semibold",
        )

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_gif(frame_paths: list[Path], gif_path: Path, fps: int) -> str:
    duration = 1.0 / max(fps, 1)
    try:
        import imageio.v2 as imageio

        frames = [imageio.imread(path) for path in frame_paths]
        imageio.mimsave(gif_path, frames, duration=duration)
        return "imageio"
    except Exception:
        pass

    try:
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
    except Exception as exc:
        raise RuntimeError("GIF assembly requires imageio or Pillow") from exc


def write_readme(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: dict[str, object],
    gif_written: bool,
) -> None:
    gif_line = "- `turntable.gif`: rotating GIF animation.\n" if gif_written else ""
    readme = f"""# STL Turntable Render

Input STL: `{Path(args.stl).resolve()}`

This directory contains a headless 3D rendering of the STL mesh using fixed
camera limits, deterministic lighting, and the `{args.material}` material preset.

## Outputs

{gif_line}- `preview.png`: first-frame preview image.
- `frames/frame_*.png`: individual rendered frames.
- `render_manifest.json`: parameters, mesh statistics, and artifact paths.

## Render Settings

- material: `{args.material}`
- background: `{args.background}`
- frames: `{args.frames}`
- fps: `{args.fps}`
- elevation: `{args.elev}`
- azimuth: `{args.azim}`
- STL faces rendered: `{manifest["rendered_faces"]}`
- STL faces in source: `{manifest["source_faces"]}`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an ASCII or binary STL as a polished turntable GIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("stl", help="Path to an ASCII or binary STL file.")
    parser.add_argument("--out", default="stl_turntable_render", help="Output directory.")
    parser.add_argument("--title", default="", help="Optional title rendered above the model.")
    parser.add_argument("--frames", type=int, default=48, help="Number of animation frames.")
    parser.add_argument("--fps", type=int, default=18, help="GIF frames per second.")
    parser.add_argument("--dpi", type=int, default=140, help="Matplotlib figure DPI.")
    parser.add_argument("--figsize", type=parse_figsize, default=(7.0, 6.0), help="Figure size as WIDTH,HEIGHT.")
    parser.add_argument("--material", choices=sorted(MATERIALS), default="titanium", help="Material preset.")
    parser.add_argument("--background", choices=sorted(BACKGROUNDS), default="studio", help="Background preset.")
    parser.add_argument("--elev", type=float, default=26.0, help="Camera elevation angle in degrees.")
    parser.add_argument("--azim", type=float, default=-48.0, help="Camera azimuth angle in degrees.")
    parser.add_argument("--azim-start", type=float, default=0.0, help="Starting object rotation in degrees.")
    parser.add_argument("--turns", type=float, default=1.0, help="Number of full rotations across the GIF.")
    parser.add_argument("--max-faces", type=int, default=12000, help="Maximum triangles to render.")
    parser.add_argument("--edge-alpha", type=float, default=0.08, help="Triangle edge opacity.")
    parser.add_argument("--edge-width", type=float, default=0.04, help="Triangle edge line width.")
    parser.add_argument("--title-size", type=float, default=15.0, help="Title font size.")
    parser.add_argument("--no-gif", action="store_true", help="Render PNG frames only.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    stl_path = Path(args.stl).expanduser().resolve()
    if not stl_path.exists():
        parser.error(f"STL file does not exist: {stl_path}")

    out_dir = Path(args.out).expanduser().resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Reading STL: {stl_path}")
    source_triangles = read_stl(stl_path)
    normalized_triangles, geometry_metadata = normalize_mesh(source_triangles)
    render_triangles, was_downsampled = downsample_faces(normalized_triangles, args.max_faces)

    points = render_triangles.reshape((-1, 3))
    xy_extent = float(max(abs(points[:, 0]).max(), abs(points[:, 1]).max(), 0.55))
    z_extent = float(max(abs(points[:, 2]).max(), 0.18))
    limit = xy_extent * 1.35
    z_limit = max(z_extent * 1.55, limit * 0.34)

    frame_paths: list[Path] = []
    for frame_idx in range(args.frames):
        phase = frame_idx / args.frames
        angle = math.radians(args.azim_start) + phase * args.turns * 2.0 * math.pi
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        if not args.quiet:
            print(f"Rendering frame {frame_idx + 1}/{args.frames}: {frame_path}")
        render_frame(
            render_triangles,
            angle,
            frame_path,
            args=args,
            limit=limit,
            z_limit=z_limit,
        )
        frame_paths.append(frame_path)

    preview_path = out_dir / "preview.png"
    shutil.copyfile(frame_paths[0], preview_path)

    gif_path = out_dir / "turntable.gif"
    gif_backend = None
    if not args.no_gif:
        if not args.quiet:
            print(f"Writing GIF: {gif_path}")
        gif_backend = write_gif(frame_paths, gif_path, args.fps)

    manifest = {
        "input_stl": str(stl_path),
        "output_directory": str(out_dir),
        "source_faces": int(source_triangles.shape[0]),
        "rendered_faces": int(render_triangles.shape[0]),
        "downsampled": bool(was_downsampled),
        "geometry": geometry_metadata,
        "settings": {
            "frames": args.frames,
            "fps": args.fps,
            "dpi": args.dpi,
            "figsize": list(args.figsize),
            "material": args.material,
            "background": args.background,
            "elev": args.elev,
            "azim": args.azim,
            "azim_start": args.azim_start,
            "turns": args.turns,
            "max_faces": args.max_faces,
            "edge_alpha": args.edge_alpha,
            "edge_width": args.edge_width,
        },
        "artifacts": {
            "preview_png": str(preview_path),
            "frames_dir": str(frames_dir),
            "gif": str(gif_path) if gif_backend else None,
            "readme": str(out_dir / "README.md"),
            "manifest": str(out_dir / "render_manifest.json"),
        },
        "gif_backend": gif_backend,
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(out_dir, args=args, manifest=manifest, gif_written=bool(gif_backend))

    print("\nSTL render complete")
    print(f"Output directory: {out_dir}")
    print(f"Preview: {preview_path}")
    if gif_backend:
        print(f"GIF: {gif_path}")
    print(f"Frames: {frames_dir}")
    print(f"Manifest: {out_dir / 'render_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
