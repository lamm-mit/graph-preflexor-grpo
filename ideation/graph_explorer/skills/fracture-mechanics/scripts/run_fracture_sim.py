#!/usr/bin/env python3
"""Run a parameterized 2D lattice fracture simulation.

This is an offline version of the "Fracture and Mechanics" notebook demo. It
simulates a pre-cracked triangular lattice under Mode I tension or Mode II shear
with velocity Verlet dynamics, pair potentials, grip loading, and virial stress
tracking. It writes a movie, stress-strain plot, CSV data, and run parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fracture-mechanics")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RCUT = 2.5


def smooth_cutoff(r: np.ndarray | float, rc: float = RCUT) -> np.ndarray | float:
    arr = np.asarray(r, dtype=float)
    out = np.where(arr < rc, 0.5 * (np.cos(np.pi * arr / rc) + 1.0), 0.0)
    return float(out) if np.ndim(r) == 0 else out


def smooth_cutoff_deriv(r: np.ndarray | float, rc: float = RCUT) -> np.ndarray | float:
    arr = np.asarray(r, dtype=float)
    out = np.where(arr < rc, -0.5 * (np.pi / rc) * np.sin(np.pi * arr / rc), 0.0)
    return float(out) if np.ndim(r) == 0 else out


def morse_pair(r: float, *, De: float = 1.0, a: float = 5.0, re: float = 1.0) -> tuple[float, float]:
    e = math.exp(-a * (r - re))
    v = De * (1.0 - e) ** 2 - De
    dv = 2.0 * a * De * (1.0 - e) * e
    fc = smooth_cutoff(r)
    dfc = smooth_cutoff_deriv(r)
    return v * fc, dv * fc + v * dfc


def lj_pair(r: float, *, eps: float = 1.0, sigma: float = 0.9) -> tuple[float, float]:
    s6 = (sigma / r) ** 6
    s12 = s6 * s6
    v = 4.0 * eps * (s12 - s6)
    dv = 4.0 * eps * (-12.0 * s12 + 6.0 * s6) / r
    fc = smooth_cutoff(r)
    dfc = smooth_cutoff_deriv(r)
    return v * fc, dv * fc + v * dfc


DEFAULT_MLIP = {
    "w": [18.0, 15.5, 13.0, 10.5, 8.0, 6.0, 4.0, 2.5, -2.5, -4.0, -6.0, -8.0],
    "b_in": [-17.0, -15.0, -13.0, -11.0, -8.8, -6.6, -4.8, -3.1, 2.8, 4.6, 6.4, 8.2],
    "c": [-1.10, -0.72, -0.42, -0.18, 0.05, 0.14, 0.18, 0.12, -0.09, -0.13, -0.10, -0.04],
    "b_out": -0.18,
}


def mlip_pair(r: float, *, weights: dict | None = None) -> tuple[float, float]:
    """Evaluate a compact neural pair potential with tanh features.

    The default is a small Morse-like neural surrogate for demos. Users can
    pass a JSON file with keys w, b_in, c, and b_out from the notebook to use a
    custom learned pair potential.
    """
    w = weights or DEFAULT_MLIP
    raw = float(w.get("b_out", 0.0))
    draw = 0.0
    for a, b, c in zip(w["w"], w["b_in"], w["c"]):
        h = math.tanh(float(a) * r + float(b))
        raw += float(c) * h
        draw += float(c) * (1.0 - h * h) * float(a)
    fc = smooth_cutoff(r)
    dfc = smooth_cutoff_deriv(r)
    return raw * fc, draw * fc + raw * dfc


def morse_pair_array(r: np.ndarray, *, De: float = 1.0, a: float = 5.0, re: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    e = np.exp(-a * (r - re))
    v = De * (1.0 - e) ** 2 - De
    dv = 2.0 * a * De * (1.0 - e) * e
    fc = smooth_cutoff(r)
    dfc = smooth_cutoff_deriv(r)
    return v * fc, dv * fc + v * dfc


def lj_pair_array(r: np.ndarray, *, eps: float = 1.0, sigma: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    s6 = (sigma / r) ** 6
    s12 = s6 * s6
    v = 4.0 * eps * (s12 - s6)
    dv = 4.0 * eps * (-12.0 * s12 + 6.0 * s6) / r
    fc = smooth_cutoff(r)
    dfc = smooth_cutoff_deriv(r)
    return v * fc, dv * fc + v * dfc


def mlip_pair_array(r: np.ndarray, *, weights: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    w = weights or DEFAULT_MLIP
    raw = np.full_like(r, float(w.get("b_out", 0.0)), dtype=float)
    draw = np.zeros_like(r, dtype=float)
    for a, b, c in zip(w["w"], w["b_in"], w["c"]):
        h = np.tanh(float(a) * r + float(b))
        raw += float(c) * h
        draw += float(c) * (1.0 - h * h) * float(a)
    fc = smooth_cutoff(r)
    dfc = smooth_cutoff_deriv(r)
    return raw * fc, draw * fc + raw * dfc


def make_potential(args: argparse.Namespace) -> Callable[[float], tuple[float, float]]:
    mlip_weights = None
    if args.mlip_weights:
        mlip_weights = json.loads(Path(args.mlip_weights).read_text(encoding="utf-8"))

    def potential(r: float) -> tuple[float, float]:
        if args.potential == "morse":
            return morse_pair(r, De=args.morse_De, a=args.morse_a, re=args.morse_re)
        if args.potential == "lj":
            return lj_pair(r, eps=args.lj_eps, sigma=args.lj_sigma)
        return mlip_pair(r, weights=mlip_weights)

    return potential


def make_vector_potential(args: argparse.Namespace) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    mlip_weights = None
    if args.mlip_weights:
        mlip_weights = json.loads(Path(args.mlip_weights).read_text(encoding="utf-8"))

    def potential(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if args.potential == "morse":
            return morse_pair_array(r, De=args.morse_De, a=args.morse_a, re=args.morse_re)
        if args.potential == "lj":
            return lj_pair_array(r, eps=args.lj_eps, sigma=args.lj_sigma)
        return mlip_pair_array(r, weights=mlip_weights)

    return potential


def make_triangular(nx: int, ny: int, a0: float, orientation: str) -> np.ndarray:
    dy = a0 * math.sqrt(3.0) / 2.0
    pos: list[tuple[float, float]] = []
    if orientation == "90":
        for i in range(nx):
            for j in range(ny):
                pos.append((i * dy, j * a0 + (0.5 * a0 if i % 2 else 0.0)))
    else:
        for j in range(ny):
            xoff = 0.5 * a0 if j % 2 else 0.0
            for i in range(nx):
                pos.append((i * a0 + xoff, j * dy))
    return np.array(pos, dtype=float)


def equilibrium_a0(potential: Callable[[float], tuple[float, float]]) -> float:
    best_a0 = 1.0
    best_e = float("inf")
    for a0 in np.arange(0.84, 1.16, 0.01):
        pos = make_triangular(6, 6, float(a0), "0")
        e = 0.0
        for i in range(len(pos) - 1):
            for j in range(i + 1, len(pos)):
                r = float(np.linalg.norm(pos[j] - pos[i]))
                if r < RCUT:
                    e += potential(r)[0]
        e /= len(pos)
        if e < best_e:
            best_a0 = float(a0)
            best_e = e
    return best_a0


def crosses_crack(pos: np.ndarray, i: int, j: int, yc: float, xtip: float) -> bool:
    xi, yi = pos[i]
    xj, yj = pos[j]
    if (yi - yc) * (yj - yc) >= 0:
        return False
    xc = xi + (xj - xi) * (yc - yi) / (yj - yi)
    return xc < xtip


def initial_broken_pairs(pos: np.ndarray, yc: float, xtip: float) -> set[int]:
    n = len(pos)
    broken: set[int] = set()
    candidates = [i for i, (_, y) in enumerate(pos) if abs(y - yc) < RCUT and pos[i, 0] < xtip + RCUT]
    for a, i in enumerate(candidates[:-1]):
        for j in candidates[a + 1 :]:
            if np.linalg.norm(pos[j] - pos[i]) < RCUT and crosses_crack(pos, i, j, yc, xtip):
                broken.add(min(i, j) * n + max(i, j))
    return broken


def build_cells(pos: np.ndarray) -> dict[tuple[int, int], list[int]]:
    low = pos.min(axis=0) - 1e-9
    cells: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(pos):
        key = (int((x - low[0]) // RCUT), int((y - low[1]) // RCUT))
        cells.setdefault(key, []).append(i)
    return cells


def initial_bond_pairs(pos: np.ndarray, broken: set[int], cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(pos)
    pairs_i: list[int] = []
    pairs_j: list[int] = []
    cells = build_cells(pos)
    cutoff2 = cutoff * cutoff
    visited: set[int] = set()
    for cell, indices in cells.items():
        cx, cy = cell
        for i in indices:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in cells.get((cx + dx, cy + dy), []):
                        if j <= i:
                            continue
                        pair_key = i * n + j
                        if pair_key in visited or pair_key in broken:
                            continue
                        visited.add(pair_key)
                        delta = pos[j] - pos[i]
                        if float(delta[0] * delta[0] + delta[1] * delta[1]) <= cutoff2:
                            pairs_i.append(i)
                            pairs_j.append(j)
    return np.array(pairs_i, dtype=np.int64), np.array(pairs_j, dtype=np.int64)


def forces_from_bonds(
    pos: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_alive: np.ndarray,
    potential: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    mode: str,
    coord_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(pos)
    force = np.zeros_like(pos)
    pe = np.zeros(n)
    virial = np.zeros(n)
    coord = np.zeros(n)
    distances = np.full(len(pair_i), np.inf, dtype=float)
    if not len(pair_i) or not pair_alive.any():
        return force, pe, virial, coord, distances

    alive_idx = np.flatnonzero(pair_alive)
    i_all = pair_i[alive_idx]
    j_all = pair_j[alive_idx]
    delta_all = pos[j_all] - pos[i_all]
    r2_all = np.einsum("ij,ij->i", delta_all, delta_all)
    active_local = (r2_all < RCUT * RCUT) & (r2_all > 1e-14)
    distances[alive_idx] = np.sqrt(np.maximum(r2_all, 0.0))
    if not active_local.any():
        return force, pe, virial, coord, distances

    active_idx = alive_idx[active_local]
    i = pair_i[active_idx]
    j = pair_j[active_idx]
    delta = pos[j] - pos[i]
    r = distances[active_idx]
    v, dv = potential(r)
    fmag = dv / r
    fij = delta * fmag[:, None]
    np.add.at(force, i, fij)
    np.add.at(force, j, -fij)

    half_v = 0.5 * v
    np.add.at(pe, i, half_v)
    np.add.at(pe, j, half_v)
    if mode == "I":
        scalar = delta[:, 1] * delta[:, 1] * dv / r
    else:
        scalar = delta[:, 0] * delta[:, 1] * dv / r
    half_scalar = 0.5 * scalar
    np.add.at(virial, i, half_scalar)
    np.add.at(virial, j, half_scalar)

    coordinated = r < coord_cutoff
    if coordinated.any():
        np.add.at(coord, i[coordinated], 1.0)
        np.add.at(coord, j[coordinated], 1.0)
    return force, pe, virial, coord, distances


def forces(
    pos: np.ndarray,
    potential: Callable[[float], tuple[float, float]],
    broken: set[int],
    mode: str,
    coord_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(pos)
    force = np.zeros_like(pos)
    pe = np.zeros(n)
    virial = np.zeros(n)
    coord = np.zeros(n)
    cells = build_cells(pos)
    visited: set[tuple[int, int]] = set()
    for cell, indices in cells.items():
        cx, cy = cell
        for i in indices:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in cells.get((cx + dx, cy + dy), []):
                        if j <= i:
                            continue
                        pair_key = i * n + j
                        if pair_key in visited or pair_key in broken:
                            continue
                        visited.add(pair_key)
                        delta = pos[j] - pos[i]
                        r2 = float(delta[0] * delta[0] + delta[1] * delta[1])
                        if r2 >= RCUT * RCUT or r2 < 1e-14:
                            continue
                        r = math.sqrt(r2)
                        v, dv = potential(r)
                        fmag = dv / r
                        fij = fmag * delta
                        force[i] += fij
                        force[j] -= fij
                        pe[i] += 0.5 * v
                        pe[j] += 0.5 * v
                        if mode == "I":
                            scalar = delta[1] * delta[1] * dv / r
                        else:
                            scalar = delta[0] * delta[1] * dv / r
                        virial[i] += 0.5 * scalar
                        virial[j] += 0.5 * scalar
                        if r < coord_cutoff:
                            coord[i] += 1.0
                            coord[j] += 1.0
    return force, pe, virial, coord


def color_values(
    color_by: str,
    vel: np.ndarray,
    pe: np.ndarray,
    virial: np.ndarray,
    coord: np.ndarray,
) -> tuple[np.ndarray, str]:
    if color_by == "pe":
        return pe, "potential energy"
    if color_by == "stress":
        return virial, "virial stress"
    if color_by == "coordination":
        return coord, "coordination"
    if color_by == "ke":
        return 0.5 * np.sum(vel * vel, axis=1), "kinetic energy"
    return np.linalg.norm(vel, axis=1), "speed"


def render_frame(
    path: Path,
    pos: np.ndarray,
    vel: np.ndarray,
    pe: np.ndarray,
    virial: np.ndarray,
    coord: np.ndarray,
    grip: np.ndarray,
    history: list[tuple[int, float, float]],
    args: argparse.Namespace,
    bounds: tuple[float, float, float, float],
    step: int,
    strain: float,
    stress: float,
    plot_limits: tuple[float, float, float, float],
    color_limits: tuple[float, float],
) -> None:
    xmin, xmax, ymin, ymax = bounds
    values, label = color_values(args.color_by, vel, pe, virial, coord)
    vmin, vmax = color_limits
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.4),
        dpi=args.dpi,
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )
    fig.patch.set_facecolor("#f7f7f4")
    ax.set_facecolor("#ffffff")
    size = max(4, min(22, 3600 / max(1, len(pos))))
    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=values,
        s=size,
        cmap="magma" if args.color_by in {"pe", "ke", "speed"} else "coolwarm",
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )
    if grip.any():
        ax.scatter(pos[grip, 0], pos[grip, 1], s=size * 0.75, facecolors="none", edgecolors="#1f1f1f", linewidths=0.35)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{args.title}\nstep {step} | strain {strain:.4f} | stress {stress:.4f}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(label, fontsize=8)

    if history:
        x = [row[1] for row in history]
        y = [row[2] for row in history]
        ax2.plot(x, y, color="#b3362d", lw=1.8)
        peak = max(y)
        ax2.axhline(peak, color="#999999", lw=0.8, ls="--")
        ax2.text(0.04, 0.94, f"peak stress {peak:.4f}", transform=ax2.transAxes, fontsize=8, va="top")
    ax2.set_xlim(plot_limits[0], plot_limits[1])
    ax2.set_ylim(plot_limits[2], plot_limits[3])
    ax2.set_title("Stress-strain response", fontsize=10)
    ax2.set_xlabel("engineering strain")
    ax2.set_ylabel("baseline-corrected virial stress")
    ax2.grid(alpha=0.25)
    fig.suptitle(f"{args.potential.upper()} | Mode {args.mode} | T={args.temperature:g} | rate={args.strain_rate:g}", fontsize=11)
    fig.subplots_adjust(left=0.055, right=0.955, bottom=0.12, top=0.84, wspace=0.30)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def frame_plot_limits(history: list[tuple[int, float, float]]) -> tuple[float, float, float, float]:
    if not history:
        return 0.0, 1.0, 0.0, 1.0
    strain = np.array([row[1] for row in history], dtype=float)
    stress = np.array([row[2] for row in history], dtype=float)
    xmax = float(max(strain.max(), 1e-9))
    ymax = float(max(stress.max(), 1e-9))
    return 0.0, xmax * 1.03, 0.0, ymax * 1.18


def frame_color_limits(
    snapshots: list[dict[str, object]],
    color_by: str,
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for snap in snapshots:
        vals, _ = color_values(
            color_by,
            snap["vel"],  # type: ignore[arg-type]
            snap["pe"],  # type: ignore[arg-type]
            snap["virial"],  # type: ignore[arg-type]
            snap["coord"],  # type: ignore[arg-type]
        )
        values.append(np.asarray(vals, dtype=float).ravel())
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return 0.0, 1.0
    if color_by == "stress":
        bound = float(np.max(np.abs(merged)))
        bound = bound if bound > 1e-12 else 1.0
        return -bound, bound
    vmin = float(np.min(merged))
    vmax = float(np.max(merged))
    if math.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1.0)
        return vmin - pad, vmax + pad
    pad = 0.03 * (vmax - vmin)
    return vmin - pad, vmax + pad


def save_gif(frame_paths: list[Path], out_path: Path, fps: int) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False
    if not frame_paths:
        return False
    frames = [Image.open(frame).convert("P", palette=Image.Palette.ADAPTIVE, colors=96) for frame in frame_paths]
    duration = int(1000 / max(1, fps))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    for frame in frames:
        frame.close()
    return True


def write_html_movie(frame_paths: list[Path], out_path: Path, fps: int) -> None:
    names = [f"frames/{path.name}" for path in frame_paths]
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Fracture Movie</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee;margin:24px}}img{{max-width:100%;height:auto;border:1px solid #444}}</style></head>
<body><h1>Fracture movie</h1><p>Fallback HTML playback at {fps} fps.</p><img id="frame" src="{names[0] if names else ''}">
<script>
const frames = {json.dumps(names)};
let i = 0; const img = document.getElementById('frame');
setInterval(() => {{ if (frames.length) {{ i = (i + 1) % frames.length; img.src = frames[i]; }} }}, {int(1000 / max(1, fps))});
</script></body></html>
"""
    out_path.write_text(html, encoding="utf-8")


def stress_plot(path: Path, history: list[tuple[int, float, float]], args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=args.dpi)
    fig.patch.set_facecolor("#ffffff")
    if history:
        strain = [row[1] for row in history]
        stress = [row[2] for row in history]
        ax.plot(strain, stress, color="#b3362d", lw=2)
        peak_idx = int(np.argmax(stress))
        ax.scatter([strain[peak_idx]], [stress[peak_idx]], color="#111111", s=24, zorder=3)
        ax.annotate(
            f"peak\n{stress[peak_idx]:.4f}",
            (strain[peak_idx], stress[peak_idx]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
        )
        ymax = max(stress)
        ax.set_ylim(top=ymax * 1.18 if ymax > 0 else 1.0)
    ax.set_title(f"Stress-strain | {args.potential.upper()} Mode {args.mode}")
    ax.set_xlabel("engineering strain")
    ax.set_ylabel("baseline-corrected virial stress")
    ax.grid(alpha=0.28)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    potential = make_potential(args)
    vector_potential = make_vector_potential(args)
    a0 = equilibrium_a0(potential) if args.a0 == "auto" else float(args.a0)
    pos = make_triangular(args.nx, args.ny, a0, args.orientation)
    n = len(pos)
    if n > args.max_atoms:
        scale = math.sqrt(args.max_atoms / n)
        nx = max(6, int(args.nx * scale))
        ny = max(6, int(args.ny * scale))
        pos = make_triangular(nx, ny, a0, args.orientation)
        n = len(pos)
    else:
        nx, ny = args.nx, args.ny

    x0 = pos[:, 0].copy()
    y0 = pos[:, 1].copy()
    lx0 = float(pos[:, 0].max() - pos[:, 0].min() + a0)
    ly0 = float(pos[:, 1].max() - pos[:, 1].min() + a0)
    yc = float(pos[:, 1].mean())
    xtip = float(pos[:, 0].min() + args.crack_length * lx0)
    broken = initial_broken_pairs(pos, yc, xtip)
    pair_i, pair_j = initial_bond_pairs(pos, broken, args.bond_cutoff * a0)
    pair_alive = np.ones(len(pair_i), dtype=bool)

    grip = np.zeros(n, dtype=bool)
    gsign = np.zeros(n, dtype=float)
    unique_y = np.unique(np.round(pos[:, 1], 8))
    bottom_cut = unique_y[min(args.grip_rows - 1, len(unique_y) - 1)] + 1e-8
    top_cut = unique_y[max(0, len(unique_y) - args.grip_rows)] - 1e-8
    grip[pos[:, 1] <= bottom_cut] = True
    grip[pos[:, 1] >= top_cut] = True
    gsign[pos[:, 1] <= bottom_cut] = -1.0
    gsign[pos[:, 1] >= top_cut] = 1.0

    vel = rng.normal(0.0, 1.0, size=pos.shape)
    vel[grip] = 0.0
    vel[~grip] -= vel[~grip].mean(axis=0)
    ke = 0.5 * float(np.sum(vel[~grip] * vel[~grip]))
    target_ke = max(args.temperature, 0.0) * max(1, (~grip).sum())
    if ke > 1e-12:
        vel[~grip] *= math.sqrt(target_ke / ke)
    for i in np.where(~grip)[0]:
        offset = pos[i, 1] - yc
        if args.mode == "I":
            vel[i, 1] += args.strain_rate * offset
        else:
            vel[i, 0] += args.strain_rate * offset

    force, pe, virial, coord, pair_distances = forces_from_bonds(
        pos,
        pair_i,
        pair_j,
        pair_alive,
        vector_potential,
        args.mode,
        args.coord_cutoff * a0,
    )
    history: list[tuple[int, float, float]] = []
    frame_steps = set(np.linspace(0, args.steps, args.frames, dtype=int).tolist())
    snapshots: list[dict[str, object]] = []
    disp = 0.0
    bounds_pad = max(lx0, ly0) * 0.10
    bounds = (
        float(pos[:, 0].min() - bounds_pad),
        float(pos[:, 0].max() + bounds_pad),
        float(pos[:, 1].min() - bounds_pad - args.strain_rate * args.steps * args.dt * ly0),
        float(pos[:, 1].max() + bounds_pad + args.strain_rate * args.steps * args.dt * ly0),
    )

    stress0 = float(-np.sum(virial) / (lx0 * ly0))

    def record(step: int) -> tuple[float, float]:
        strain = 2.0 * disp / ly0
        # Use the positive-loading convention for the reported stress-strain
        # curve and subtract the initial residual virial of the pre-cracked
        # lattice so the plotted response starts at zero.
        raw_stress = float(-np.sum(virial) / (lx0 * ly0 * max(0.25, 1.0 + strain)))
        stress = abs(raw_stress - stress0)
        history.append((step, strain, stress))
        return strain, stress

    def capture_snapshot(step: int, strain: float, stress: float) -> None:
        snapshots.append(
            {
                "step": step,
                "strain": strain,
                "stress": stress,
                "history_len": len(history),
                "pos": pos.copy(),
                "vel": vel.copy(),
                "pe": pe.copy(),
                "virial": virial.copy(),
                "coord": coord.copy(),
            }
        )

    strain, stress = record(0)
    if 0 in frame_steps:
        capture_snapshot(0, strain, stress)

    for step in range(1, args.steps + 1):
        free = ~grip
        vel[free] += 0.5 * args.dt * force[free]
        pos[free] += args.dt * vel[free]
        disp += args.strain_rate * ly0 * 0.5 * args.dt
        if args.mode == "I":
            pos[grip, 0] = x0[grip]
            pos[grip, 1] = y0[grip] + gsign[grip] * disp
            vel[grip, 0] = 0.0
            vel[grip, 1] = gsign[grip] * args.strain_rate * ly0 * 0.5
        else:
            pos[grip, 1] = y0[grip]
            pos[grip, 0] = x0[grip] + gsign[grip] * disp
            vel[grip, 0] = gsign[grip] * args.strain_rate * ly0 * 0.5
            vel[grip, 1] = 0.0
        force, pe, virial, coord, pair_distances = forces_from_bonds(
            pos,
            pair_i,
            pair_j,
            pair_alive,
            vector_potential,
            args.mode,
            args.coord_cutoff * a0,
        )
        if args.break_stretch > 0 and pair_alive.any():
            stretched = pair_alive & (pair_distances > args.break_stretch * a0)
            if stretched.any():
                pair_alive[stretched] = False
                force, pe, virial, coord, pair_distances = forces_from_bonds(
                    pos,
                    pair_i,
                    pair_j,
                    pair_alive,
                    vector_potential,
                    args.mode,
                    args.coord_cutoff * a0,
                )
        vel[free] = (vel[free] + 0.5 * args.dt * force[free]) * math.exp(-args.damping * args.dt)
        strain, stress = record(step)
        if step in frame_steps:
            capture_snapshot(step, strain, stress)

    stress_png = out_dir / "stress_strain.png"
    stress_plot(stress_png, history, args)

    plot_limits = frame_plot_limits(history)
    color_limits = frame_color_limits(snapshots, args.color_by)
    frame_paths: list[Path] = []
    for idx, snap in enumerate(snapshots):
        path = frame_dir / f"frame_{idx:04d}.png"
        history_len = int(snap["history_len"])
        render_frame(
            path,
            snap["pos"],  # type: ignore[arg-type]
            snap["vel"],  # type: ignore[arg-type]
            snap["pe"],  # type: ignore[arg-type]
            snap["virial"],  # type: ignore[arg-type]
            snap["coord"],  # type: ignore[arg-type]
            grip,
            history[:history_len],
            args,
            bounds,
            int(snap["step"]),
            float(snap["strain"]),
            float(snap["stress"]),
            plot_limits,
            color_limits,
        )
        frame_paths.append(path)

    final_png = out_dir / "final_lattice.png"
    render_frame(final_png, pos, vel, pe, virial, coord, grip, history, args, bounds, args.steps, strain, stress, plot_limits, color_limits)

    csv_path = out_dir / "stress_strain.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "strain", "baseline_corrected_stress"])
        writer.writerows(history)

    gif_path = out_dir / "fracture_movie.gif"
    gif_ok = save_gif(frame_paths, gif_path, args.movie_fps)
    html_path = out_dir / "fracture_movie.html"
    write_html_movie(frame_paths, html_path, args.movie_fps)

    stress_values = np.array([row[2] for row in history], dtype=float)
    strain_values = np.array([row[1] for row in history], dtype=float)
    peak_idx = int(np.argmax(stress_values)) if len(stress_values) else 0
    summary = {
        "title": args.title,
        "potential": args.potential,
        "mode": args.mode,
        "orientation": args.orientation,
        "nx": nx,
        "ny": ny,
        "atoms": n,
        "requested_atoms": args.nx * args.ny,
        "a0": a0,
        "steps": args.steps,
        "frames": len(frame_paths),
        "temperature": args.temperature,
        "strain_rate": args.strain_rate,
        "dt": args.dt,
        "damping": args.damping,
        "crack_length_fraction": args.crack_length,
        "bond_cutoff_a0": args.bond_cutoff,
        "break_stretch_a0": args.break_stretch,
        "initial_broken_pairs": len(broken),
        "initial_bonds": int(len(pair_i)),
        "final_bonds": int(pair_alive.sum()),
        "dynamically_broken_bonds": int(len(pair_i) - pair_alive.sum()),
        "final_strain": float(strain_values[-1]),
        "final_stress": float(stress_values[-1]),
        "peak_stress": float(stress_values[peak_idx]),
        "peak_strain": float(strain_values[peak_idx]),
        "gif_written": gif_ok,
        "outputs": {
            "movie_gif": gif_path.name if gif_ok else None,
            "movie_html": html_path.name,
            "stress_strain_png": stress_png.name,
            "final_lattice_png": final_png.name,
            "stress_strain_csv": csv_path.name,
            "parameters_json": "parameters.json",
            "summary_json": "summary.json",
            "readme": "README.md",
            "frames_dir": "frames",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    params = vars(args).copy()
    params["a0_resolved"] = a0
    params["outputs"] = summary["outputs"]
    (out_dir / "parameters.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Fracture Mechanics Run

Generated from a parameterized 2D triangular-lattice fracture simulation.

- potential: {args.potential}
- loading mode: {args.mode}
- orientation: {args.orientation} degrees
- lattice: {nx} x {ny} atoms ({n} atoms)
- steps: {args.steps}
- initial temperature: {args.temperature}
- strain rate: {args.strain_rate}
- damping: {args.damping}
- crack length fraction: {args.crack_length}
- bond cutoff: {args.bond_cutoff} x a0
- break stretch: {args.break_stretch} x a0
- initial intact bonds: {len(pair_i)}
- dynamically broken bonds: {int(len(pair_i) - pair_alive.sum())}
- peak stress: {summary['peak_stress']:.6f} at strain {summary['peak_strain']:.6f}
- final stress: {summary['final_stress']:.6f} at strain {summary['final_strain']:.6f}

Artifacts:

- `{gif_path.name}`: animated movie GIF{'' if gif_ok else ' (not written; use HTML fallback)'}
- `{html_path.name}`: HTML frame playback
- `{stress_png.name}`: stress-strain plot
- `{final_png.name}`: final lattice state
- `{csv_path.name}`: stress-strain data
- `summary.json`: key run metrics
- `parameters.json`: exact input parameters
- `frames/`: individual movie frames

This is a mesoscale demonstration model. It is useful for exploring trends and
visual mechanisms, not for quantitative prediction of a specific material.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="skill_output_fracture-mechanics", help="Output directory.")
    parser.add_argument("--title", default="2D Lattice Fracture", help="Title used in figures.")
    parser.add_argument("--potential", choices=("morse", "lj", "mlip"), default="morse")
    parser.add_argument("--mode", choices=("I", "II"), default="I", help="Mode I tension or Mode II shear.")
    parser.add_argument("--nx", type=int, default=96, help="Lattice cells in x.")
    parser.add_argument("--ny", type=int, default=36, help="Lattice cells in y.")
    parser.add_argument("--max-atoms", type=int, default=5000, help="Clamp requested lattice if nx*ny is larger.")
    parser.add_argument("--orientation", choices=("0", "90"), default="90")
    parser.add_argument("--crack-length", type=float, default=0.30, help="Initial edge crack length as fraction of specimen width.")
    parser.add_argument("--grip-rows", type=int, default=2, help="Rows clamped into top/bottom grips.")
    parser.add_argument("--steps", type=int, default=10000, help="Velocity-Verlet steps; use 10000 for fracture-producing runs.")
    parser.add_argument("--frames", type=int, default=48, help="Movie frames to render.")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.004, help="Initial kinetic temperature scale.")
    parser.add_argument("--strain-rate", type=float, default=0.003)
    parser.add_argument("--damping", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--a0", default="auto", help="'auto' or explicit triangular lattice spacing.")
    parser.add_argument("--morse-De", type=float, default=1.0)
    parser.add_argument("--morse-a", type=float, default=5.0, help="Higher is more brittle; lower is tougher.")
    parser.add_argument("--morse-re", type=float, default=1.0)
    parser.add_argument("--lj-eps", type=float, default=1.0)
    parser.add_argument("--lj-sigma", type=float, default=0.90)
    parser.add_argument("--mlip-weights", default=None, help="Optional JSON weights with w, b_in, c, b_out.")
    parser.add_argument("--bond-cutoff", type=float, default=1.35, help="Initial bonded-neighbor cutoff in units of a0.")
    parser.add_argument("--break-stretch", type=float, default=1.75, help="Permanently break bonds stretched beyond this multiple of a0; use 0 to disable.")
    parser.add_argument("--coord-cutoff", type=float, default=1.22, help="Coordination cutoff in units of a0.")
    parser.add_argument("--color-by", choices=("pe", "stress", "coordination", "ke", "speed"), default="stress")
    parser.add_argument("--movie-fps", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    args.nx = max(6, min(2048, args.nx))
    args.ny = max(6, min(2048, args.ny))
    args.max_atoms = max(64, min(40000, args.max_atoms))
    args.crack_length = max(0.0, min(0.80, args.crack_length))
    args.grip_rows = max(1, min(8, args.grip_rows))
    args.steps = max(1, min(20000, args.steps))
    args.frames = max(2, min(240, args.frames))
    args.dt = max(0.0005, min(0.05, args.dt))
    args.temperature = max(0.0, min(1.0, args.temperature))
    args.strain_rate = max(0.0, min(0.10, args.strain_rate))
    args.damping = max(0.0, min(20.0, args.damping))
    args.bond_cutoff = max(1.01, min(2.50, args.bond_cutoff))
    args.break_stretch = max(0.0, min(3.0, args.break_stretch))
    args.coord_cutoff = max(0.5, min(2.5, args.coord_cutoff))
    args.movie_fps = max(1, min(30, args.movie_fps))
    args.dpi = max(72, min(240, args.dpi))
    return args


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(f"Wrote {Path(args.out) / 'stress_strain.png'}")
    if summary["gif_written"]:
        print(f"Wrote {Path(args.out) / 'fracture_movie.gif'}")
    print(f"Wrote {Path(args.out) / 'fracture_movie.html'}")
    print(f"Wrote {Path(args.out) / 'summary.json'}")
    print(f"Wrote {Path(args.out) / 'parameters.json'}")
    print(f"Wrote {Path(args.out) / 'README.md'}")


if __name__ == "__main__":
    main()
