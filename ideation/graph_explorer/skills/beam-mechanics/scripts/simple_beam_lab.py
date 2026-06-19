#!/usr/bin/env python3
"""Simple dimensionless beam mechanics lab.

This is intentionally small and agent-friendly: no units, no YAML, no schema.
It solves a 1D Euler-Bernoulli beam with vertical displacement and rotation at
each node, then writes plots, CSV/JSON summaries, and an optional GIF.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


def configure_matplotlib_cache() -> None:
    for env_name, leaf in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "cache")):
        if os.environ.get(env_name):
            continue
        for candidate in (
            Path.cwd() / ".simple_beam_cache" / leaf,
            Path.home() / ".cache" / "simple-beam-lab" / leaf,
            Path("/tmp") / "simple-beam-lab" / leaf,
        ):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / ".write_test"
                probe.write_text("", encoding="utf-8")
                probe.unlink()
                os.environ[env_name] = str(candidate)
                break
            except OSError:
                continue


configure_matplotlib_cache()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - GIF is optional.
    imageio = None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_pair(text: str, name: str) -> tuple[float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"{name} expects X,VALUE")
    return float(parts[0]), float(parts[1])


def parse_triple(text: str, name: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"{name} expects START,END,VALUE")
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_support(text: str) -> tuple[float, str]:
    parts = [part.strip().lower() for part in text.split(",")]
    if len(parts) != 2:
        raise SystemExit("--support expects X,TYPE where TYPE is pin, roller, fixed, clamped, or free")
    return float(parts[0]), parts[1]


def parse_spring(text: str) -> tuple[float, float]:
    return parse_pair(text, "--spring")


def equivalent_point_force(xi: float, length: float, force: float) -> np.ndarray:
    n1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    n2 = length * (xi - 2.0 * xi**2 + xi**3)
    n3 = 3.0 * xi**2 - 2.0 * xi**3
    n4 = length * (-xi**2 + xi**3)
    return force * np.array([n1, n2, n3, n4], dtype=float)


def equivalent_point_moment(xi: float, length: float, moment: float) -> np.ndarray:
    dn1 = (-6.0 * xi + 6.0 * xi**2) / length
    dn2 = 1.0 - 4.0 * xi + 3.0 * xi**2
    dn3 = (6.0 * xi - 6.0 * xi**2) / length
    dn4 = -2.0 * xi + 3.0 * xi**2
    return moment * np.array([dn1, dn2, dn3, dn4], dtype=float)


def shape_values(xi: float, length: float) -> np.ndarray:
    return np.array(
        [
            1.0 - 3.0 * xi**2 + 2.0 * xi**3,
            length * (xi - 2.0 * xi**2 + xi**3),
            3.0 * xi**2 - 2.0 * xi**3,
            length * (-xi**2 + xi**3),
        ],
        dtype=float,
    )


def shape_d1(xi: float, length: float) -> np.ndarray:
    return np.array(
        [
            (-6.0 * xi + 6.0 * xi**2) / length,
            1.0 - 4.0 * xi + 3.0 * xi**2,
            (6.0 * xi - 6.0 * xi**2) / length,
            -2.0 * xi + 3.0 * xi**2,
        ],
        dtype=float,
    )


def shape_d2(xi: float, length: float) -> np.ndarray:
    return np.array(
        [
            (-6.0 + 12.0 * xi) / length**2,
            (-4.0 + 6.0 * xi) / length,
            (6.0 - 12.0 * xi) / length**2,
            (-2.0 + 6.0 * xi) / length,
        ],
        dtype=float,
    )


def shape_d3(_xi: float, length: float) -> np.ndarray:
    return np.array([12.0 / length**3, 6.0 / length**2, -12.0 / length**3, 6.0 / length**2], dtype=float)


def element_stiffness(ei: float, length: float) -> np.ndarray:
    le = length
    return ei / le**3 * np.array(
        [
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, 4.0 * le**2, -6.0 * le, 2.0 * le**2],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, 2.0 * le**2, -6.0 * le, 4.0 * le**2],
        ],
        dtype=float,
    )


def build_positions(length: float, elements: int, special: list[float]) -> np.ndarray:
    values = [float(x) for x in np.linspace(0.0, length, elements + 1)]
    values.extend(clamp(float(x), 0.0, length) for x in special)
    values = sorted(values)
    cleaned: list[float] = []
    for value in values:
        if not cleaned or abs(cleaned[-1] - value) > 1e-9:
            cleaned.append(value)
    return np.array(cleaned, dtype=float)


def nearest_node(x: float, positions: np.ndarray) -> int:
    return int(np.argmin(np.abs(positions - x)))


def element_index_for_x(x: float, positions: np.ndarray) -> int:
    if x <= positions[0]:
        return 0
    if x >= positions[-1]:
        return len(positions) - 2
    idx = int(np.searchsorted(positions, x, side="right") - 1)
    return max(0, min(idx, len(positions) - 2))


def support_constraints(kind: str) -> tuple[bool, bool]:
    clean = kind.lower().replace("-", "_")
    if clean in {"fixed", "clamped", "clamp"}:
        return True, True
    if clean in {"pin", "pinned", "roller", "simple", "simple_support"}:
        return True, False
    if clean in {"free", "none"}:
        return False, False
    raise SystemExit(f"Unsupported support type '{kind}'. Use pin, roller, fixed, or free.")


def preset_defaults(args: argparse.Namespace) -> None:
    preset = args.preset.lower().replace("_", "-")
    if preset == "simply-supported":
        args.left = "pin"
        args.right = "roller"
        if not args.point_down and not args.point and not args.udl and not args.udl_down:
            args.point_down.append(f"{0.5 * args.length},1.0")
    elif preset == "cantilever":
        args.left = "fixed"
        args.right = "free"
        if not args.point_down and not args.point and not args.udl and not args.udl_down:
            args.point_down.append(f"{args.length},1.0")
    elif preset == "fixed-fixed":
        args.left = "fixed"
        args.right = "fixed"
        if not args.udl and not args.udl_down and not args.point and not args.point_down:
            args.udl_down.append(f"0,{args.length},1.0")
    elif preset == "overhang":
        args.left = "pin"
        args.right = "free"
        if not any(abs(x - 0.7 * args.length) < 1e-9 for x, _kind in args.support):
            args.support.append((0.7 * args.length, "roller"))
        if not args.point_down and not args.point:
            args.point_down.append(f"{args.length},1.0")
    elif preset == "custom":
        pass
    else:
        raise SystemExit("--preset must be simply-supported, cantilever, fixed-fixed, overhang, or custom")


def solve_beam(args: argparse.Namespace) -> dict[str, Any]:
    preset_defaults(args)

    points = [parse_pair(text, "--point") for text in args.point]
    points += [(x, -abs(value)) for x, value in [parse_pair(text, "--point-down") for text in args.point_down]]
    moments = [parse_pair(text, "--moment") for text in args.moment]
    moments += [(x, -abs(value)) for x, value in [parse_pair(text, "--moment-clockwise") for text in args.moment_clockwise]]
    udls = [parse_triple(text, "--udl") for text in args.udl]
    udls += [(a, b, -abs(value)) for a, b, value in [parse_triple(text, "--udl-down") for text in args.udl_down]]
    springs = args.spring

    special = [0.0, args.length]
    special.extend(x for x, _value in points)
    special.extend(x for x, _value in moments)
    special.extend(x for x, _kind in args.support)
    special.extend(x for x, _k in springs)
    for a, b, _q in udls:
        special.extend([a, b])
    positions = build_positions(args.length, args.elements, special)
    n_nodes = len(positions)
    n_dof = 2 * n_nodes
    k_global = np.zeros((n_dof, n_dof), dtype=float)
    f_global = np.zeros(n_dof, dtype=float)

    element_equiv_loads: list[np.ndarray] = []
    for e in range(n_nodes - 1):
        le = positions[e + 1] - positions[e]
        if le <= 0:
            raise SystemExit("Mesh contains a zero-length element.")
        dofs = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1])
        k_global[np.ix_(dofs, dofs)] += element_stiffness(args.ei, le)
        fe = np.zeros(4, dtype=float)
        # Distributed loads are integrated over the portion overlapping this element.
        for a, b, q in udls:
            lo = max(min(a, b), positions[e])
            hi = min(max(a, b), positions[e + 1])
            if hi <= lo:
                continue
            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            for gauss in (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)):
                x = center + half * gauss
                xi = (x - positions[e]) / le
                fe += q * shape_values(xi, le) * half
        for x, p in points:
            if e == element_index_for_x(x, positions):
                xi = clamp((x - positions[e]) / le, 0.0, 1.0)
                fe += equivalent_point_force(xi, le, p)
        for x, m in moments:
            if e == element_index_for_x(x, positions):
                xi = clamp((x - positions[e]) / le, 0.0, 1.0)
                fe += equivalent_point_moment(xi, le, m)
        f_global[dofs] += fe
        element_equiv_loads.append(fe)

    supports = [(0.0, args.left), (args.length, args.right)] + args.support
    fixed_dofs: list[int] = []
    support_records: list[dict[str, Any]] = []
    for x, kind in supports:
        node = nearest_node(x, positions)
        fix_w, fix_theta = support_constraints(kind)
        record = {"x_requested": x, "x_node": float(positions[node]), "type": kind, "node": node, "fixed": []}
        if fix_w:
            fixed_dofs.append(2 * node)
            record["fixed"].append("w")
        if fix_theta:
            fixed_dofs.append(2 * node + 1)
            record["fixed"].append("theta")
        support_records.append(record)

    spring_records: list[dict[str, Any]] = []
    for x, stiffness in springs:
        node = nearest_node(x, positions)
        k_global[2 * node, 2 * node] += stiffness
        spring_records.append({"x_requested": x, "x_node": float(positions[node]), "node": node, "stiffness": stiffness})

    fixed_dofs = sorted(set(fixed_dofs))
    free_dofs = [idx for idx in range(n_dof) if idx not in fixed_dofs]
    if not free_dofs:
        raise SystemExit("All DOFs are fixed.")

    k_ff = k_global[np.ix_(free_dofs, free_dofs)]
    f_f = f_global[free_dofs]
    try:
        d_f = np.linalg.solve(k_ff, f_f)
    except np.linalg.LinAlgError as exc:
        raise SystemExit("The beam model is unstable or singular. Add a support or remove a mechanism.") from exc
    displacement = np.zeros(n_dof, dtype=float)
    displacement[free_dofs] = d_f
    residual = np.zeros(n_dof, dtype=float)
    for dof in fixed_dofs:
        residual[dof] = float(np.dot(k_global[dof, :], displacement) - f_global[dof])

    rows: list[dict[str, float]] = []
    for e in range(n_nodes - 1):
        le = positions[e + 1] - positions[e]
        dofs = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1])
        de = displacement[dofs]
        for xi in np.linspace(0.0, 1.0, args.samples_per_element):
            x = positions[e] + xi * le
            w = float(shape_values(float(xi), le) @ de)
            theta = float(shape_d1(float(xi), le) @ de)
            moment = float(args.ei * (shape_d2(float(xi), le) @ de))
            shear = float(args.ei * (shape_d3(float(xi), le) @ de))
            rows.append({"x": float(x), "w": w, "theta": theta, "moment": moment, "shear": shear})

    reactions = [
        {
            "type": "support",
            "dof": "w" if dof % 2 == 0 else "theta",
            "node": dof // 2,
            "x": float(positions[dof // 2]),
            "value": float(residual[dof]),
        }
        for dof in fixed_dofs
    ]
    for spring in spring_records:
        node = int(spring["node"])
        w_node = float(displacement[2 * node])
        reactions.append(
            {
                "type": "spring",
                "dof": "w",
                "node": node,
                "x": float(positions[node]),
                "value": float(-spring["stiffness"] * w_node),
                "stiffness": spring["stiffness"],
            }
        )
    vertical_reaction_sum = float(sum(item["value"] for item in reactions if item["dof"] == "w"))
    vertical_load_sum = float(sum(p for _x, p in points) + sum((max(a, b) - min(a, b)) * q for a, b, q in udls))
    moment_load_sum = float(sum(m for _x, m in moments))

    def extreme(field: str) -> dict[str, float]:
        item = max(rows, key=lambda row: abs(row[field]))
        return {"x": item["x"], "value": item[field]}

    summary = {
        "status": "ok",
        "preset": args.preset,
        "length": args.length,
        "ei": args.ei,
        "node_count": n_nodes,
        "element_count": n_nodes - 1,
        "max_abs_deflection": extreme("w"),
        "max_abs_rotation": extreme("theta"),
        "max_abs_moment": extreme("moment"),
        "max_abs_shear": extreme("shear"),
        "vertical_load_sum": vertical_load_sum,
        "vertical_reaction_sum": vertical_reaction_sum,
        "vertical_equilibrium_residual": vertical_reaction_sum + vertical_load_sum,
        "applied_moment_sum": moment_load_sum,
        "supports": support_records,
        "springs": spring_records,
        "loads": {
            "point": [{"x": x, "value": p} for x, p in points],
            "moment": [{"x": x, "value": m} for x, m in moments],
            "udl": [{"start": a, "end": b, "value": q} for a, b, q in udls],
        },
        "sign_convention": "vertical displacement/load: positive up, negative down; moment sign follows beam rotation convention",
    }

    return {
        "positions": positions,
        "displacement": displacement,
        "rows": rows,
        "reactions": reactions,
        "summary": summary,
        "supports": support_records,
        "springs": spring_records,
        "points": points,
        "moments": moments,
        "udls": udls,
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "w", "theta", "moment", "shear"])
        writer.writeheader()
        writer.writerows(rows)


def plot_loads_and_supports(ax: plt.Axes, result: dict[str, Any], y0: float, span: float) -> None:
    for support in result["supports"]:
        x = support["x_node"]
        label = support["type"]
        ax.plot([x], [y0], marker="^", markersize=8, color="#333333")
        ax.text(x, y0 - 0.07 * span, label, ha="center", va="top", fontsize=8)
    for spring in result["springs"]:
        x = spring["x_node"]
        ax.text(x, y0 + 0.05 * span, "spring", ha="center", va="bottom", fontsize=8, color="#775500")
    for x, p in result["points"]:
        if p < 0:
            start_y = y0 + 0.18 * span
            dy = -0.13 * span
            text_y = start_y + 0.03 * span
            text_va = "bottom"
        else:
            start_y = y0 - 0.18 * span
            dy = 0.13 * span
            text_y = start_y - 0.03 * span
            text_va = "top"
        ax.arrow(x, start_y, 0, dy, head_width=0.025 * span, color="#b00020")
        ax.text(x, text_y, f"P={p:g}", ha="center", va=text_va, fontsize=8, color="#b00020")
    for a, b, q in result["udls"]:
        xs = np.linspace(min(a, b), max(a, b), 8)
        if q < 0:
            start_y = y0 + 0.14 * span
            dy = -0.09 * span
            text_y = start_y + 0.03 * span
            text_va = "bottom"
        else:
            start_y = y0 - 0.14 * span
            dy = 0.09 * span
            text_y = start_y - 0.03 * span
            text_va = "top"
        for x in xs:
            ax.arrow(x, start_y, 0, dy, head_width=0.015 * span, color="#7b1fa2", alpha=0.65)
        ax.text(float(np.mean(xs)), text_y, f"q={q:g}", ha="center", va=text_va, fontsize=8, color="#7b1fa2")
    for x, m in result["moments"]:
        ax.text(x, y0 + 0.11 * span, f"M={m:g}", ha="center", fontsize=8, color="#00695c")


def save_plots(result: dict[str, Any], out: Path, title: str) -> list[str]:
    positions = result["positions"]
    rows = result["rows"]
    x = np.array([row["x"] for row in rows])
    w = np.array([row["w"] for row in rows])
    moment = np.array([row["moment"] for row in rows])
    shear = np.array([row["shear"] for row in rows])
    span = max(float(positions[-1] - positions[0]), 1.0)
    max_w = max(float(np.max(np.abs(w))), 1e-12)
    scale = 0.18 * span / max_w
    files: list[str] = []

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(positions, np.zeros_like(positions), "-o", color="#222222", linewidth=2, markersize=3)
    plot_loads_and_supports(ax, result, 0.0, span)
    ax.set_title(f"{title}: beam, supports, loads")
    ax.set_xlabel("x")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    path = out / "structure.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(x, np.zeros_like(x), "--", color="#777777", linewidth=1)
    ax.plot(x, scale * w, color="#1565c0", linewidth=2)
    plot_loads_and_supports(ax, result, 0.0, span)
    ax.set_title(f"{title}: deformed shape (scale {scale:.3g})")
    ax.set_xlabel("x")
    ax.set_ylabel("scaled w")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out / "deformed_shape.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    plot_specs = [
        ("deflection.png", "deflection w", w, "#1565c0"),
        ("moment.png", "bending moment", moment, "#b00020"),
        ("shear.png", "shear force", shear, "#2e7d32"),
    ]
    for filename, label, values, color in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(x, values, color=color, linewidth=2)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(f"{title}: {label}")
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = out / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    axes[0].plot(x, scale * w, color="#1565c0", linewidth=2)
    axes[0].set_title("scaled deformed shape")
    axes[1].plot(x, w, color="#1565c0", linewidth=2)
    axes[1].set_title("deflection")
    axes[2].plot(x, moment, color="#b00020", linewidth=2)
    axes[2].set_title("moment")
    axes[3].plot(x, shear, color="#2e7d32", linewidth=2)
    axes[3].set_title("shear")
    for ax in axes:
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x")
    fig.suptitle(title)
    fig.tight_layout()
    path = out / "dashboard.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))
    return files


def save_gif(result: dict[str, Any], out: Path, title: str, frames: int, fps: int) -> str | None:
    if imageio is None or frames <= 0:
        return None
    rows = result["rows"]
    x = np.array([row["x"] for row in rows])
    w = np.array([row["w"] for row in rows])
    span = max(float(x[-1] - x[0]), 1.0)
    max_w = max(float(np.max(np.abs(w))), 1e-12)
    scale = 0.18 * span / max_w
    y = scale * w
    ylim = max(0.25 * span, 1.25 * float(np.max(np.abs(y))))
    frame_dir = out / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for idx, alpha in enumerate(np.linspace(0.0, 1.0, frames)):
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(x, np.zeros_like(x), "--", color="#777777", linewidth=1)
        ax.plot(x, alpha * y, color="#1565c0", linewidth=2)
        plot_loads_and_supports(ax, result, 0.0, span)
        ax.set_xlim(float(x[0]) - 0.04 * span, float(x[-1]) + 0.04 * span)
        ax.set_ylim(-ylim, ylim)
        ax.set_title(f"{title}: deformation movie frame {idx + 1}/{frames}")
        ax.set_xlabel("x")
        ax.set_ylabel("scaled w")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = frame_dir / f"frame_{idx:03d}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        frame_paths.append(path)
    gif_path = out / "deformation.gif"
    imageio.mimsave(gif_path, [imageio.imread(path) for path in frame_paths], duration=1.0 / max(fps, 1))
    return str(gif_path)


def write_readme(path: Path, result: dict[str, Any], title: str) -> None:
    summary = result["summary"]
    text = f"""# {title}

This run uses a dimensionless Euler-Bernoulli beam model with vertical
displacement and rotation at each node.

Sign convention: positive vertical values point upward; negative values point
downward. The reported plots and CSV use the same convention.

## Key Results

- Max absolute deflection: {summary['max_abs_deflection']['value']:.6g} at x={summary['max_abs_deflection']['x']:.6g}
- Max absolute moment: {summary['max_abs_moment']['value']:.6g} at x={summary['max_abs_moment']['x']:.6g}
- Max absolute shear: {summary['max_abs_shear']['value']:.6g} at x={summary['max_abs_shear']['x']:.6g}
- Vertical equilibrium residual: {summary['vertical_equilibrium_residual']:.6g}

## Files

- `structure.png`
- `deformed_shape.png`
- `deflection.png`
- `moment.png`
- `shear.png`
- `dashboard.png`
- `deformation.gif` when GIF output is enabled
- `field_data.csv`
- `summary.json`
- `reactions.json`
- `manifest.json`
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", default="skill_output_beam-mechanics")
    parser.add_argument("--title", default="Simple Beam Mechanics")
    parser.add_argument("--preset", default="simply-supported", choices=["simply-supported", "cantilever", "fixed-fixed", "overhang", "custom"])
    parser.add_argument("--length", type=float, default=10.0)
    parser.add_argument("--elements", type=int, default=80)
    parser.add_argument("--ei", type=float, default=1.0, help="Dimensionless bending stiffness.")
    parser.add_argument("--left", default="pin", help="Left support: pin, roller, fixed, clamped, free.")
    parser.add_argument("--right", default="roller", help="Right support: pin, roller, fixed, clamped, free.")
    parser.add_argument("--support", action="append", type=parse_support, default=[], help="Internal support as X,TYPE. Repeatable.")
    parser.add_argument("--spring", action="append", type=parse_spring, default=[], help="Vertical spring as X,STIFFNESS. Repeatable.")
    parser.add_argument("--point", action="append", default=[], help="Signed point force as X,VALUE. Negative is downward.")
    parser.add_argument("--point-down", action="append", default=[], help="Positive downward point force as X,MAGNITUDE.")
    parser.add_argument("--moment", action="append", default=[], help="Signed point moment as X,VALUE.")
    parser.add_argument("--moment-clockwise", action="append", default=[], help="Positive clockwise point moment as X,MAGNITUDE.")
    parser.add_argument("--udl", action="append", default=[], help="Signed uniform load as START,END,VALUE. Negative is downward.")
    parser.add_argument("--udl-down", action="append", default=[], help="Positive downward uniform load as START,END,MAGNITUDE.")
    parser.add_argument("--samples-per-element", type=int, default=7)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use fewer elements and frames for fast smoke tests.")
    parser.add_argument("--quiet", action="store_true", help="Print only manifest path.")
    args = parser.parse_args()
    if args.length <= 0:
        raise SystemExit("--length must be positive")
    if args.elements < 2:
        raise SystemExit("--elements must be at least 2")
    if args.ei <= 0:
        raise SystemExit("--ei must be positive")
    if args.quick:
        args.elements = min(args.elements, 30)
        args.frames = min(args.frames, 8)
    return args


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = solve_beam(args)
    files: dict[str, str] = {}

    field_csv = out / "field_data.csv"
    write_csv(field_csv, result["rows"])
    files["field_data_csv"] = str(field_csv)

    summary_json = out / "summary.json"
    summary_json.write_text(json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8")
    files["summary_json"] = str(summary_json)

    reactions_json = out / "reactions.json"
    reactions_json.write_text(json.dumps(result["reactions"], indent=2) + "\n", encoding="utf-8")
    files["reactions_json"] = str(reactions_json)

    for path in save_plots(result, out, args.title):
        files[Path(path).stem + "_png"] = path
    if not args.no_gif:
        gif = save_gif(result, out, args.title, args.frames, args.fps)
        if gif:
            files["deformation_gif"] = gif

    readme = out / "README.md"
    write_readme(readme, result, args.title)
    files["readme_md"] = str(readme)

    manifest = {
        "status": "ok",
        "title": args.title,
        "summary": result["summary"],
        "files": files,
        "command_interface": "simple_beam_lab.py flags only; no units, no YAML",
    }
    manifest_json = out / "manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files["manifest_json"] = str(manifest_json)

    if args.quiet:
        print(str(manifest_json))
    else:
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
