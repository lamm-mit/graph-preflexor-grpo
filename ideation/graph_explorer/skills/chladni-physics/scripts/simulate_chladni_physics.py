#!/usr/bin/env python3
"""Physics-based Chladni plate simulation.

This script solves a driven, damped Kirchhoff-Love thin rectangular plate using
a simply supported modal basis, then simulates sand particles drifting toward
low-vibration nodal regions. It is designed to be robust in a skill/container
environment, with no FEM package required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-chladni-physics")


MATERIALS: dict[str, dict[str, float]] = {
    "steel": {"E": 200.0e9, "rho": 7850.0, "nu": 0.30},
    "aluminum": {"E": 69.0e9, "rho": 2700.0, "nu": 0.33},
    "brass": {"E": 100.0e9, "rho": 8500.0, "nu": 0.34},
    "glass": {"E": 70.0e9, "rho": 2500.0, "nu": 0.23},
    "acrylic": {"E": 3.2e9, "rho": 1180.0, "nu": 0.35},
}

PRESETS: dict[str, dict[str, object]] = {
    "steel-square-43": {
        "material": "steel",
        "length": 0.30,
        "width": 0.30,
        "thickness": 0.0012,
        "target_mode": "4,3",
        "drive_ratio": 1.002,
        "drive_x": 0.37,
        "drive_y": 0.29,
        "damping": 0.006,
        "palette": "blue-sand",
    },
    "glass-square-65": {
        "material": "glass",
        "length": 0.24,
        "width": 0.24,
        "thickness": 0.0010,
        "target_mode": "6,5",
        "drive_ratio": 0.998,
        "drive_x": 0.31,
        "drive_y": 0.42,
        "damping": 0.004,
        "palette": "monochrome",
    },
    "aluminum-rectangle-27": {
        "material": "aluminum",
        "length": 0.42,
        "width": 0.24,
        "thickness": 0.0015,
        "target_mode": "2,7",
        "drive_ratio": 1.001,
        "drive_x": 0.28,
        "drive_y": 0.33,
        "damping": 0.008,
        "palette": "ice-fire",
    },
    "brass-square-54": {
        "material": "brass",
        "length": 0.28,
        "width": 0.28,
        "thickness": 0.0011,
        "target_mode": "5,4",
        "drive_ratio": 1.000,
        "drive_x": 0.41,
        "drive_y": 0.36,
        "damping": 0.005,
        "palette": "copper",
    },
}

PALETTES: dict[str, dict[str, tuple[float, float, float]]] = {
    "blue-sand": {
        "low": (0.010, 0.022, 0.045),
        "mid": (0.060, 0.180, 0.360),
        "high": (0.240, 0.740, 1.000),
        "sand": (0.980, 0.900, 0.600),
        "node": (1.000, 0.820, 0.300),
    },
    "copper": {
        "low": (0.030, 0.022, 0.018),
        "mid": (0.220, 0.105, 0.052),
        "high": (0.900, 0.440, 0.180),
        "sand": (0.980, 0.820, 0.500),
        "node": (0.990, 0.720, 0.350),
    },
    "ice-fire": {
        "low": (0.010, 0.018, 0.040),
        "mid": (0.060, 0.210, 0.460),
        "high": (1.000, 0.280, 0.080),
        "sand": (0.950, 0.970, 1.000),
        "node": (1.000, 0.720, 0.220),
    },
    "monochrome": {
        "low": (0.018, 0.018, 0.018),
        "mid": (0.220, 0.220, 0.220),
        "high": (0.800, 0.800, 0.800),
        "sand": (0.980, 0.960, 0.880),
        "node": (0.960, 0.960, 0.930),
    },
    "neon": {
        "low": (0.010, 0.006, 0.030),
        "mid": (0.090, 0.040, 0.190),
        "high": (0.040, 0.820, 1.000),
        "sand": (1.000, 0.780, 0.210),
        "node": (1.000, 0.150, 0.750),
    },
}


def stable_seed(seed: str) -> int:
    try:
        return int(seed) % (2**32)
    except ValueError:
        import hashlib

        return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12], 16) % (2**32)


def parse_mode(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Mode must be m,n, for example 4,3")
    try:
        m = int(parts[0])
        n = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Mode indices must be integers") from exc
    if m < 1 or n < 1:
        raise argparse.ArgumentTypeError("Mode indices must be positive")
    return m, n


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "custom":
        return
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def material_properties(args: argparse.Namespace) -> dict[str, float]:
    props = dict(MATERIALS[args.material])
    if args.youngs_modulus is not None:
        props["E"] = args.youngs_modulus
    if args.density is not None:
        props["rho"] = args.density
    if args.poisson is not None:
        props["nu"] = args.poisson
    return props


def plate_rigidity(E: float, nu: float, h: float) -> float:
    return E * h**3 / (12.0 * (1.0 - nu**2))


def mode_angular_frequency(m: int, n: int, args: argparse.Namespace, props: dict[str, float]) -> float:
    D = plate_rigidity(props["E"], props["nu"], args.thickness)
    mass_area = props["rho"] * args.thickness
    k2 = (m / args.length) ** 2 + (n / args.width) ** 2
    return math.pi**2 * math.sqrt(D / mass_area) * k2


def resolve_drive_frequency(args: argparse.Namespace, props: dict[str, float]) -> tuple[float, tuple[int, int]]:
    target = parse_mode(args.target_mode)
    target_frequency = mode_angular_frequency(target[0], target[1], args, props) / (2.0 * math.pi)
    if args.drive_frequency is None:
        args.drive_frequency = target_frequency * args.drive_ratio
    return target_frequency, target


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return np.asarray(Image.fromarray(image).resize((width, height), resampling))


def save_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(image).save(path)


def write_gif(frame_paths: list[Path], gif_path: Path, fps: int) -> str:
    from PIL import Image

    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=max(1, int(1000 / fps)),
        loop=0,
        optimize=True,
    )
    return "pillow"


def mix(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a * (1.0 - t[..., None]) + b * t[..., None]


def colorize_scalar(field: np.ndarray, palette_name: str, *, center: bool = False) -> np.ndarray:
    palette = PALETTES[palette_name]
    low = np.asarray(palette["low"], dtype=np.float64)
    mid = np.asarray(palette["mid"], dtype=np.float64)
    high = np.asarray(palette["high"], dtype=np.float64)
    if center:
        tone = np.clip((field + 1.0) * 0.5, 0.0, 1.0)
    else:
        tone = np.clip(field, 0.0, 1.0)
    lower = mix(low, mid, np.clip(tone * 2.0, 0.0, 1.0))
    upper = mix(mid, high, np.clip((tone - 0.5) * 2.0, 0.0, 1.0))
    return np.where(tone[..., None] < 0.5, lower, upper)


def blur2d(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    out = arr.astype(np.float64)
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        out = (
            padded[1:-1, 1:-1] * 4.0
            + padded[:-2, 1:-1] * 2.0
            + padded[2:, 1:-1] * 2.0
            + padded[1:-1, :-2] * 2.0
            + padded[1:-1, 2:] * 2.0
            + padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        ) / 16.0
    return out


def build_modal_response(
    args: argparse.Namespace,
    props: dict[str, float],
    *,
    nx: int,
    ny: int,
) -> dict[str, object]:
    x = np.linspace(0.0, args.length, nx, dtype=np.float64)
    y = np.linspace(0.0, args.width, ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    omega_drive = 2.0 * math.pi * args.drive_frequency
    modal_mass = props["rho"] * args.thickness * args.length * args.width / 4.0
    response = np.zeros((ny, nx), dtype=np.complex128)
    modal_rows: list[dict[str, float | int]] = []

    xd = args.drive_x * args.length
    yd = args.drive_y * args.width
    for m in range(1, args.max_m + 1):
        sin_x = np.sin(m * math.pi * xx / args.length)
        sin_x_drive = math.sin(m * math.pi * xd / args.length)
        for n in range(1, args.max_n + 1):
            omega_mn = mode_angular_frequency(m, n, args, props)
            phi_drive = sin_x_drive * math.sin(n * math.pi * yd / args.width)
            if abs(phi_drive) < 1.0e-9:
                continue
            phi = sin_x * np.sin(n * math.pi * yy / args.width)
            denominator = (omega_mn**2 - omega_drive**2) + 1j * (2.0 * args.damping * omega_mn * omega_drive)
            q = (args.force * phi_drive / modal_mass) / denominator
            response += q * phi
            modal_rows.append(
                {
                    "m": m,
                    "n": n,
                    "frequency_hz": omega_mn / (2.0 * math.pi),
                    "drive_coupling": phi_drive,
                    "complex_amplitude_abs_m": abs(q),
                }
            )

    amplitude = np.abs(response)
    if float(amplitude.max()) <= 1.0e-18:
        raise RuntimeError("Modal response is near zero; move the drive point or change target mode")
    acceleration = (omega_drive**2) * amplitude
    energy = acceleration**2
    scale = float(np.percentile(energy, 99.5))
    if scale <= 1.0e-30:
        scale = float(energy.max())
    energy_norm = np.clip(energy / max(scale, 1.0e-30), 0.0, 1.0)
    phase0 = np.real(response)
    phase0 /= max(float(np.percentile(np.abs(phase0), 99.5)), 1.0e-30)
    phase0 = np.clip(phase0, -1.0, 1.0)

    return {
        "x": x,
        "y": y,
        "response": response,
        "amplitude_m": amplitude,
        "acceleration_m_s2": acceleration,
        "energy_norm": energy_norm,
        "phase0_norm": phase0,
        "modal_rows": modal_rows,
    }


def normalized_gradient(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(energy)
    mag = np.sqrt(gx * gx + gy * gy)
    scale = float(np.percentile(mag, 99.0))
    if scale <= 1.0e-12:
        scale = float(mag.max()) if float(mag.max()) > 0 else 1.0
    return gx / scale, gy / scale


def sample_grid(arr: np.ndarray, positions: np.ndarray) -> np.ndarray:
    ny, nx = arr.shape
    x = np.clip(positions[:, 0] * (nx - 1), 0.0, nx - 1.000001)
    y = np.clip(positions[:, 1] * (ny - 1), 0.0, ny - 1.000001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    tx = x - x0
    ty = y - y0
    return (
        arr[y0, x0] * (1.0 - tx) * (1.0 - ty)
        + arr[y0, x1] * tx * (1.0 - ty)
        + arr[y1, x0] * (1.0 - tx) * ty
        + arr[y1, x1] * tx * ty
    )


def reflect_positions(positions: np.ndarray, velocities: np.ndarray) -> None:
    for axis in (0, 1):
        low = positions[:, axis] < 0.0
        high = positions[:, axis] > 1.0
        positions[low, axis] = -positions[low, axis]
        velocities[low, axis] *= -0.45
        positions[high, axis] = 2.0 - positions[high, axis]
        velocities[high, axis] *= -0.45
    np.clip(positions, 0.0, 1.0, out=positions)


def particle_histogram(positions: np.ndarray, nx: int, ny: int) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        positions[:, 1],
        positions[:, 0],
        bins=(ny, nx),
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    hist = blur2d(hist, passes=2)
    scale = float(np.percentile(hist, 99.4))
    if scale <= 0.0:
        scale = float(hist.max()) if float(hist.max()) > 0 else 1.0
    return np.clip(hist / scale, 0.0, 1.0)


def render_sand_frame(energy: np.ndarray, particles: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    ny, nx = energy.shape
    palette = PALETTES[args.palette]
    node_color = np.asarray(palette["node"], dtype=np.float64)
    sand_color = np.asarray(palette["sand"], dtype=np.float64)

    base = colorize_scalar(np.sqrt(np.clip(energy, 0.0, 1.0)), args.palette)
    nodes = np.exp(-energy / max(args.node_width, 1.0e-5))
    sand = particle_histogram(particles, nx, ny)
    image = base * (0.70 + 0.25 * (1.0 - nodes[..., None]))
    image = mix(image, node_color, nodes * 0.25)
    image = mix(image, sand_color, np.clip(sand**0.62, 0.0, 1.0) * 0.92)
    image = np.clip(image, 0.0, 1.0)
    img = (image * 255.0).astype(np.uint8)
    return resize_image(img, args.output_width, args.output_height)


def render_vibration_frame(response: np.ndarray, phase: float, energy: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    field = np.real(response * np.exp(1j * phase))
    scale = float(np.percentile(np.abs(field), 99.5))
    field = np.clip(field / max(scale, 1.0e-30), -1.0, 1.0)
    palette = PALETTES[args.palette]
    node_color = np.asarray(palette["node"], dtype=np.float64)
    image = colorize_scalar(field, args.palette, center=True)
    nodes = np.exp(-energy / max(args.node_width, 1.0e-5))
    image = mix(image, node_color, nodes * 0.20)
    img = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return resize_image(img, args.output_width, args.output_height)


def simulate_particles(
    energy: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    out_dir: Path,
) -> tuple[np.ndarray, list[Path]]:
    gx, gy = normalized_gradient(energy)
    particles = rng.random((args.particles, 2), dtype=np.float64)
    velocities = np.zeros_like(particles)
    sand_frames = out_dir / "frames" / "sand"
    sand_frames.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    steps_per_frame = max(1, args.sand_steps // max(args.frames, 1))

    for frame_idx in range(args.frames):
        for _ in range(steps_per_frame):
            grad = np.column_stack([sample_grid(gx, particles), sample_grid(gy, particles)])
            random_walk = rng.normal(0.0, args.particle_noise, particles.shape)
            velocities = args.particle_drag * velocities - args.particle_mobility * grad + random_walk
            particles += velocities
            reflect_positions(particles, velocities)
        frame = render_sand_frame(energy, particles, args)
        frame_path = sand_frames / f"frame_{frame_idx:04d}.png"
        save_png(frame_path, frame)
        frame_paths.append(frame_path)
    return particles, frame_paths


def write_csv(path: Path, rows: Iterable[dict[str, float | int]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def nearest_modal_rows(rows: list[dict[str, float | int]], drive_frequency: float, limit: int = 8) -> list[dict[str, float | int]]:
    ordered = sorted(rows, key=lambda row: abs(float(row["frequency_hz"]) - drive_frequency))
    return ordered[:limit]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate a driven damped rectangular Chladni plate and sand-particle settling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--preset", choices=sorted(PRESETS) + ["custom"], default="steel-square-43")
    parser.add_argument("--material", choices=sorted(MATERIALS), default=None)
    parser.add_argument("--youngs-modulus", type=float, default=None, help="Override Young's modulus in Pa.")
    parser.add_argument("--density", type=float, default=None, help="Override density in kg/m^3.")
    parser.add_argument("--poisson", type=float, default=None, help="Override Poisson ratio.")
    parser.add_argument("--length", type=float, default=None, help="Plate length in meters.")
    parser.add_argument("--width", type=float, default=None, help="Plate width in meters.")
    parser.add_argument("--thickness", type=float, default=None, help="Plate thickness in meters.")
    parser.add_argument("--target-mode", default=None, help="Mode used to choose drive frequency when --drive-frequency is omitted.")
    parser.add_argument("--drive-frequency", type=float, default=None, help="Drive frequency in Hz.")
    parser.add_argument("--drive-ratio", type=float, default=None, help="Frequency ratio relative to target mode.")
    parser.add_argument("--drive-x", type=float, default=None, help="Drive x position as fraction of length.")
    parser.add_argument("--drive-y", type=float, default=None, help="Drive y position as fraction of width.")
    parser.add_argument("--force", type=float, default=0.30, help="Harmonic drive force amplitude in N.")
    parser.add_argument("--damping", type=float, default=None, help="Modal damping ratio.")
    parser.add_argument("--max-m", type=int, default=12, help="Highest x mode included.")
    parser.add_argument("--max-n", type=int, default=12, help="Highest y mode included.")
    parser.add_argument("--grid", type=int, default=220, help="Physics grid height.")
    parser.add_argument("--size", type=int, default=1080, help="Output image height in pixels.")
    parser.add_argument("--frames", type=int, default=48, help="GIF frames.")
    parser.add_argument("--fps", type=int, default=18, help="GIF frame rate.")
    parser.add_argument("--particles", type=int, default=9000, help="Number of sand particles.")
    parser.add_argument("--sand-steps", type=int, default=1800, help="Particle transport steps.")
    parser.add_argument("--particle-mobility", type=float, default=0.010, help="Drift strength down vibration-energy gradients.")
    parser.add_argument("--particle-drag", type=float, default=0.56, help="Velocity damping for sand transport.")
    parser.add_argument("--particle-noise", type=float, default=0.0025, help="Random agitation in sand transport.")
    parser.add_argument("--node-width", type=float, default=0.035, help="Visual width of nodal highlight.")
    parser.add_argument("--palette", choices=sorted(PALETTES), default=None)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--skip-vibration-gif", action="store_true")
    parser.add_argument("--skip-sand-gif", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("length", "width", "thickness", "drive_x", "drive_y", "damping", "drive_ratio", "palette", "target_mode"):
        if getattr(args, name) is None:
            raise SystemExit(f"Missing required parameter after preset resolution: --{name.replace('_', '-')}")
    if args.length <= 0 or args.width <= 0 or args.thickness <= 0:
        raise SystemExit("Plate dimensions must be positive")
    if not (0.0 < args.drive_x < 1.0 and 0.0 < args.drive_y < 1.0):
        raise SystemExit("--drive-x and --drive-y must be fractions between 0 and 1")
    if args.damping <= 0.0:
        raise SystemExit("--damping must be positive")
    if args.max_m < 1 or args.max_n < 1:
        raise SystemExit("--max-m and --max-n must be positive")
    if args.grid < 64:
        raise SystemExit("--grid must be at least 64")
    if args.frames < 1 or args.fps < 1:
        raise SystemExit("--frames and --fps must be positive")
    if args.particles < 100:
        raise SystemExit("--particles must be at least 100")
    if args.sand_steps < args.frames:
        raise SystemExit("--sand-steps must be at least --frames")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_preset(args)
    validate_args(args)
    props = material_properties(args)
    target_frequency, target_mode = resolve_drive_frequency(args, props)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(stable_seed(args.seed))

    aspect = args.length / args.width
    args.output_height = int(args.size)
    args.output_width = max(1, int(round(args.size * aspect)))
    nx = max(64, int(round(args.grid * aspect)))
    ny = int(args.grid)

    if not args.quiet:
        print("Running Chladni physics simulation")
        print(f"Output: {out_dir}")
        print(f"Material: {args.material}, target mode: {target_mode}, drive frequency: {args.drive_frequency:.3f} Hz")

    result = build_modal_response(args, props, nx=nx, ny=ny)
    response = result["response"]
    energy = result["energy_norm"]

    vibration_dir = out_dir / "frames" / "vibration"
    vibration_dir.mkdir(parents=True, exist_ok=True)
    vibration_frame_paths: list[Path] = []
    for frame_idx in range(args.frames):
        phase = 2.0 * math.pi * frame_idx / args.frames
        frame = render_vibration_frame(response, phase, energy, args)
        frame_path = vibration_dir / f"frame_{frame_idx:04d}.png"
        save_png(frame_path, frame)
        vibration_frame_paths.append(frame_path)

    particles, sand_frame_paths = simulate_particles(energy, args, rng, out_dir)
    final_sand = render_sand_frame(energy, particles, args)
    amplitude_image = resize_image(
        (colorize_scalar(np.sqrt(energy), args.palette) * 255.0).astype(np.uint8),
        args.output_width,
        args.output_height,
    )
    snapshot_image = render_vibration_frame(response, 0.0, energy, args)

    save_png(out_dir / "chladni_sand_pattern.png", final_sand)
    save_png(out_dir / "vibration_amplitude.png", amplitude_image)
    save_png(out_dir / "displacement_snapshot.png", snapshot_image)

    sand_gif = None
    vibration_gif = None
    if not args.skip_sand_gif:
        sand_gif = out_dir / "sand_settling.gif"
        write_gif(sand_frame_paths, sand_gif, args.fps)
    if not args.skip_vibration_gif:
        vibration_gif = out_dir / "plate_vibration.gif"
        write_gif(vibration_frame_paths, vibration_gif, args.fps)

    modal_rows = result["modal_rows"]
    write_csv(out_dir / "modal_frequencies.csv", modal_rows)
    near_modes = nearest_modal_rows(modal_rows, args.drive_frequency)

    particle_energy = sample_grid(energy, particles)
    node_fraction = float(np.mean(particle_energy < 0.08))
    max_displacement = float(np.max(result["amplitude_m"]))
    max_acceleration = float(np.max(result["acceleration_m_s2"]))

    np.savez_compressed(
        out_dir / "physics_data.npz",
        x=result["x"],
        y=result["y"],
        response=response,
        amplitude_m=result["amplitude_m"],
        acceleration_m_s2=result["acceleration_m_s2"],
        energy_norm=energy,
        final_particles=particles,
    )

    artifacts = {
        "chladni_sand_pattern": str(out_dir / "chladni_sand_pattern.png"),
        "vibration_amplitude": str(out_dir / "vibration_amplitude.png"),
        "displacement_snapshot": str(out_dir / "displacement_snapshot.png"),
        "sand_settling_gif": str(sand_gif) if sand_gif else None,
        "plate_vibration_gif": str(vibration_gif) if vibration_gif else None,
        "sand_frames_dir": str(out_dir / "frames" / "sand"),
        "vibration_frames_dir": str(vibration_dir),
        "physics_data": str(out_dir / "physics_data.npz"),
        "modal_frequencies": str(out_dir / "modal_frequencies.csv"),
        "parameters": str(out_dir / "parameters.json"),
        "summary": str(out_dir / "summary.json"),
        "caption": str(out_dir / "caption.txt"),
        "readme": str(out_dir / "README.md"),
    }
    parameters = {
        "model": "driven damped Kirchhoff-Love rectangular plate, simply supported modal basis",
        "sand_model": "particles drift down gradient of time-averaged plate acceleration energy with drag and noise",
        "preset": args.preset,
        "material": args.material,
        "material_properties": props,
        "length_m": args.length,
        "width_m": args.width,
        "thickness_m": args.thickness,
        "target_mode": list(target_mode),
        "target_frequency_hz": target_frequency,
        "drive_frequency_hz": args.drive_frequency,
        "drive_ratio": args.drive_ratio,
        "drive_position_fraction": [args.drive_x, args.drive_y],
        "force_N": args.force,
        "damping_ratio": args.damping,
        "max_m": args.max_m,
        "max_n": args.max_n,
        "grid": [ny, nx],
        "size_px": [args.output_height, args.output_width],
        "frames": args.frames,
        "fps": args.fps,
        "particles": args.particles,
        "sand_steps": args.sand_steps,
        "palette": args.palette,
        "seed": args.seed,
        "artifacts": artifacts,
    }
    summary = {
        "drive_frequency_hz": args.drive_frequency,
        "target_frequency_hz": target_frequency,
        "target_mode": list(target_mode),
        "max_displacement_m": max_displacement,
        "max_acceleration_m_s2": max_acceleration,
        "final_particle_node_fraction_energy_below_0p08": node_fraction,
        "nearest_modes": near_modes,
        "artifacts": artifacts,
    }
    (out_dir / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "caption.txt").write_text(
        (
            f"Driven {args.material} plate, {args.length:.3f} m by {args.width:.3f} m by "
            f"{args.thickness * 1e3:.2f} mm, excited near mode {target_mode[0]},{target_mode[1]} "
            f"at {args.drive_frequency:.1f} Hz. Sand particles migrate toward low-acceleration nodal lines."
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        f"""# Chladni Physics Simulation

This run solves a driven damped Kirchhoff-Love thin rectangular plate using a
simply supported modal basis. Sand is modeled as particles transported down the
gradient of the time-averaged acceleration energy, with drag and small random
agitation.

## What You Are Seeing

- `chladni_sand_pattern.png` is the final simulated sand pattern. Bright grains
  concentrate along nodal regions: places where the driven plate has low
  vibration energy and sand can remain.
- `sand_settling.gif` shows the particle transport process. Particles begin
  spread across the plate, are pushed away from high-acceleration antinodes, and
  progressively accumulate near low-acceleration nodal lines.
- `plate_vibration.gif` shows the driven plate displacement field over one
  oscillation cycle. Bright/dark regions are opposite displacement phases; the
  near-stationary lines between them are the nodal structure that organizes the
  sand.
- `vibration_amplitude.png` shows the time-averaged vibration strength. Darker
  valleys in this map correspond to regions where sand preferentially collects.
- `displacement_snapshot.png` is a single phase snapshot of the oscillating
  plate, useful for seeing the active modal shape.

## How The Simulation Was Done

1. The plate was represented as a thin rectangular Kirchhoff-Love plate with
   dimensions `{args.length:.6g} m x {args.width:.6g} m x {args.thickness:.6g} m`.
   The material was `{args.material}` with Young's modulus `{props["E"]:.6g} Pa`,
   density `{props["rho"]:.6g} kg/m^3`, and Poisson ratio `{props["nu"]:.4g}`.
2. The bending rigidity was computed as
   `D = E h^3 / (12 * (1 - nu^2))`.
3. The displacement field was expanded in simply supported plate modes
   `phi_mn(x,y) = sin(m*pi*x/L) * sin(n*pi*y/W)` for
   `m = 1..{args.max_m}` and `n = 1..{args.max_n}`.
4. Each modal frequency was computed from
   `omega_mn = pi^2 * sqrt(D / (rho*h)) * ((m/L)^2 + (n/W)^2)`.
5. A harmonic point drive of `{args.force:.6g} N` was applied at fractional
   position `({args.drive_x:.4g}, {args.drive_y:.4g})`. The drive frequency was
   `{args.drive_frequency:.6g} Hz`, selected near target mode
   `{target_mode[0]},{target_mode[1]}`.
6. The complex modal response used the damped oscillator denominator
   `(omega_mn^2 - omega_drive^2) + i*(2*zeta*omega_mn*omega_drive)`, with
   damping ratio `{args.damping:.6g}`. Modes with stronger drive coupling and
   frequencies close to the drive dominate the pattern.
7. The time-averaged acceleration energy was approximated from
   `energy ~ (omega_drive^2 * |w(x,y)|)^2`, where `w(x,y)` is the complex
   steady-state displacement amplitude.
8. Sand particles were simulated as mobile points. At each step they drifted
   down the gradient of the acceleration-energy field, with velocity drag and a
   small random agitation term. Boundary collisions were reflected back into the
   plate.
9. The final sand image is a smoothed particle-density histogram overlaid on
   the vibration-energy field. This is why bright material appears on nodal
   curves rather than on the high-amplitude antinodes.

## Model Scope

This is a reduced physics model, not a full finite-element/contact simulation.
It captures the main mechanism of a Chladni demonstration: a driven resonant
plate creates nodal lines, and granular material migrates away from strongly
accelerating regions toward those lines. The simplifications are: simply
supported rectangular modes, linear plate response, point harmonic forcing,
phenomenological particle drift, no grain-grain collisions, and no detailed
frictional contact mechanics.

## Key Results

- drive frequency: `{args.drive_frequency:.6g} Hz`
- target mode: `{target_mode[0]},{target_mode[1]}`
- target-mode frequency: `{target_frequency:.6g} Hz`
- max displacement amplitude: `{max_displacement:.6e} m`
- max acceleration amplitude: `{max_acceleration:.6e} m/s^2`
- final particle node fraction: `{node_fraction:.4f}`

## Outputs

- `chladni_sand_pattern.png`: final simulated sand accumulation.
- `sand_settling.gif`: particle migration movie.
- `plate_vibration.gif`: driven plate displacement movie.
- `vibration_amplitude.png`: time-averaged vibration amplitude.
- `displacement_snapshot.png`: displacement snapshot at phase zero.
- `modal_frequencies.csv`: modal frequencies and drive coupling.
- `physics_data.npz`: fields and final particle positions.
- `parameters.json`, `summary.json`, `caption.txt`: metadata and interpretation.

## How To Read The Data Files

- `summary.json` contains the headline physical values and artifact paths.
- `parameters.json` records every input parameter and material property.
- `modal_frequencies.csv` lists every included mode, its frequency, drive
  coupling, and response amplitude. Sort or inspect this file to see which modes
  contributed most strongly to the final pattern.
- `physics_data.npz` stores the grid, complex modal response, displacement
  amplitude, acceleration amplitude, normalized energy field, and final particle
  positions for downstream analysis.
""",
        encoding="utf-8",
    )

    print("\nChladni physics simulation complete")
    print(f"Output directory: {out_dir}")
    print(f"Sand pattern: {out_dir / 'chladni_sand_pattern.png'}")
    if sand_gif:
        print(f"Sand settling GIF: {sand_gif}")
    if vibration_gif:
        print(f"Plate vibration GIF: {vibration_gif}")
    print(f"Summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
