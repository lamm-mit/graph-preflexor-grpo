#!/usr/bin/env python3
"""Run 2D SIMP topology optimization with flexible supports and loads.

This script is a headless, skill-friendly version of the topology optimization
notebook. It solves a minimum-compliance 2D problem with the optimality criteria
method, writes audit artifacts for boundary conditions, and exports simple STL
meshes from the optimized density field. It intentionally avoids voxel
microstructures, marching cubes, PyVista, and boolean mesh operations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hierarchical-topopt")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import warnings


BC_PRESETS: dict[str, dict[str, Any]] = {
    "cantilever-tip-down": {
        "supports": [{"selector": "left_edge", "ux": True, "uy": True, "source": "left edge clamp"}],
        "loads": [{"selector": "right_bottom", "fx": 0.0, "fy": -1.0, "distribution": "total", "source": "downward lower-right tip load"}],
    },
    "cantilever-mid-down": {
        "supports": [{"selector": "left_edge", "ux": True, "uy": True, "source": "left edge clamp"}],
        "loads": [{"selector": "right_mid", "fx": 0.0, "fy": -1.0, "distribution": "total", "source": "downward right-mid load"}],
    },
    "left-fixed-top-mid-20-down": {
        "supports": [{"selector": "left_edge", "ux": True, "uy": True, "source": "left edge clamp"}],
        "loads": [
            {
                "selector": {"type": "edge_fraction", "edge": "top", "start": 0.4, "end": 0.6},
                "fx": 0.0,
                "fy": -1.0,
                "distribution": "total",
                "source": "downward distributed load over middle 20 percent of top edge",
            }
        ],
    },
    "bridge-center-load": {
        "supports": [
            {"selector": "left_bottom", "ux": True, "uy": True, "source": "lower-left pin"},
            {"selector": "right_bottom", "ux": False, "uy": True, "source": "lower-right roller"},
        ],
        "loads": [{"selector": "top_mid", "fx": 0.0, "fy": -1.0, "distribution": "total", "source": "downward top-center load"}],
    },
    "bridge-top-mid-20-down": {
        "supports": [
            {"selector": "left_bottom", "ux": True, "uy": True, "source": "lower-left pin"},
            {"selector": "right_bottom", "ux": False, "uy": True, "source": "lower-right roller"},
        ],
        "loads": [
            {
                "selector": {"type": "edge_fraction", "edge": "top", "start": 0.4, "end": 0.6},
                "fx": 0.0,
                "fy": -1.0,
                "distribution": "total",
                "source": "downward distributed load over middle 20 percent of top edge",
            }
        ],
    },
    "simply-supported-center-load": {
        "supports": [
            {"selector": "left_bottom", "ux": True, "uy": True, "source": "lower-left pin"},
            {"selector": "right_bottom", "ux": False, "uy": True, "source": "lower-right roller"},
        ],
        "loads": [{"selector": "top_mid", "fx": 0.0, "fy": -1.0, "distribution": "total", "source": "downward top-center load"}],
    },
    "fixed-fixed-center-load": {
        "supports": [
            {"selector": "left_edge", "ux": True, "uy": True, "source": "left edge clamp"},
            {"selector": "right_edge", "ux": True, "uy": True, "source": "right edge clamp"},
        ],
        "loads": [{"selector": "top_mid", "fx": 0.0, "fy": -1.0, "distribution": "total", "source": "downward top-center load"}],
    },
    "fixed-fixed-top-mid-20-down": {
        "supports": [
            {"selector": "left_edge", "ux": True, "uy": True, "source": "left edge clamp"},
            {"selector": "right_edge", "ux": True, "uy": True, "source": "right edge clamp"},
        ],
        "loads": [
            {
                "selector": {"type": "edge_fraction", "edge": "top", "start": 0.4, "end": 0.6},
                "fx": 0.0,
                "fy": -1.0,
                "distribution": "total",
                "source": "downward distributed load over middle 20 percent of top edge",
            }
        ],
    },
    "tension-strip": {
        "supports": [{"selector": "left_edge", "ux": True, "uy": True, "source": "left edge clamp"}],
        "loads": [{"selector": "right_edge", "fx": 1.0, "fy": 0.0, "distribution": "total", "source": "right edge tension"}],
    },
    "shear-strip": {
        "supports": [{"selector": "bottom_edge", "ux": True, "uy": True, "source": "bottom edge clamp"}],
        "loads": [{"selector": "top_edge", "fx": 1.0, "fy": 0.0, "distribution": "total", "source": "top edge shear"}],
    },
}


def node_id(i: int, j: int, nely: int) -> int:
    return i * (nely + 1) + j


def node_ij(node: int, nely: int) -> tuple[int, int]:
    return node // (nely + 1), node % (nely + 1)


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def fraction_to_i(x: float, nelx: int) -> int:
    return clamp_int(round(float(x) * nelx), 0, nelx)


def fraction_to_j(y: float, nely: int) -> int:
    return clamp_int(round(float(y) * nely), 0, nely)


def fraction_range(start: float, end: float, count: int) -> range:
    lo = max(0.0, min(1.0, float(start)))
    hi = max(0.0, min(1.0, float(end)))
    if lo > hi:
        lo, hi = hi, lo
    a = clamp_int(math.ceil(lo * count - 1e-12), 0, count)
    b = clamp_int(math.floor(hi * count + 1e-12), 0, count)
    if b < a:
        b = a
    return range(a, b + 1)


def alias_selector(selector: str) -> dict[str, Any]:
    aliases: dict[str, dict[str, Any]] = {
        "left_edge": {"type": "edge", "edge": "left"},
        "right_edge": {"type": "edge", "edge": "right"},
        "top_edge": {"type": "edge", "edge": "top"},
        "bottom_edge": {"type": "edge", "edge": "bottom"},
        "left_mid": {"type": "node_fraction", "x": 0.0, "y": 0.5},
        "right_mid": {"type": "node_fraction", "x": 1.0, "y": 0.5},
        "top_mid": {"type": "node_fraction", "x": 0.5, "y": 1.0},
        "bottom_mid": {"type": "node_fraction", "x": 0.5, "y": 0.0},
        "center": {"type": "node_fraction", "x": 0.5, "y": 0.5},
        "left_bottom": {"type": "node_fraction", "x": 0.0, "y": 0.0},
        "bottom_left": {"type": "node_fraction", "x": 0.0, "y": 0.0},
        "lower_left": {"type": "node_fraction", "x": 0.0, "y": 0.0},
        "left_top": {"type": "node_fraction", "x": 0.0, "y": 1.0},
        "top_left": {"type": "node_fraction", "x": 0.0, "y": 1.0},
        "right_bottom": {"type": "node_fraction", "x": 1.0, "y": 0.0},
        "bottom_right": {"type": "node_fraction", "x": 1.0, "y": 0.0},
        "lower_right": {"type": "node_fraction", "x": 1.0, "y": 0.0},
        "right_top": {"type": "node_fraction", "x": 1.0, "y": 1.0},
        "top_right": {"type": "node_fraction", "x": 1.0, "y": 1.0},
    }
    if selector in aliases:
        return aliases[selector]
    if selector.startswith("node:"):
        parts = selector.split(":", 1)[1].split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid node selector {selector!r}; use node:i,j")
        return {"type": "node", "i": int(parts[0]), "j": int(parts[1])}
    raise ValueError(f"Unknown selector alias {selector!r}")


def resolve_selector(selector: str | dict[str, Any], nelx: int, nely: int) -> list[int]:
    spec = alias_selector(selector) if isinstance(selector, str) else dict(selector)
    typ = spec.get("type")
    nodes: list[int]
    if typ == "node":
        nodes = [node_id(clamp_int(spec["i"], 0, nelx), clamp_int(spec["j"], 0, nely), nely)]
    elif typ == "node_fraction":
        nodes = [node_id(fraction_to_i(spec["x"], nelx), fraction_to_j(spec["y"], nely), nely)]
    elif typ == "edge":
        edge = spec["edge"]
        if edge == "left":
            nodes = [node_id(0, j, nely) for j in range(nely + 1)]
        elif edge == "right":
            nodes = [node_id(nelx, j, nely) for j in range(nely + 1)]
        elif edge == "top":
            nodes = [node_id(i, nely, nely) for i in range(nelx + 1)]
        elif edge == "bottom":
            nodes = [node_id(i, 0, nely) for i in range(nelx + 1)]
        else:
            raise ValueError(f"Unknown edge {edge!r}")
    elif typ == "edge_fraction":
        edge = spec["edge"]
        start = spec.get("start", 0.0)
        end = spec.get("end", 1.0)
        if edge == "top":
            nodes = [node_id(i, nely, nely) for i in fraction_range(start, end, nelx)]
        elif edge == "bottom":
            nodes = [node_id(i, 0, nely) for i in fraction_range(start, end, nelx)]
        elif edge == "left":
            nodes = [node_id(0, j, nely) for j in fraction_range(start, end, nely)]
        elif edge == "right":
            nodes = [node_id(nelx, j, nely) for j in fraction_range(start, end, nely)]
        else:
            raise ValueError(f"Unknown edge {edge!r}")
    elif typ == "x_range_fraction":
        xs = fraction_range(spec.get("x0", 0.0), spec.get("x1", 1.0), nelx)
        nodes = [node_id(i, j, nely) for i in xs for j in range(nely + 1)]
    elif typ == "y_range_fraction":
        ys = fraction_range(spec.get("y0", 0.0), spec.get("y1", 1.0), nely)
        nodes = [node_id(i, j, nely) for i in range(nelx + 1) for j in ys]
    elif typ == "box_fraction":
        xs = fraction_range(spec.get("x0", 0.0), spec.get("x1", 1.0), nelx)
        ys = fraction_range(spec.get("y0", 0.0), spec.get("y1", 1.0), nely)
        nodes = [node_id(i, j, nely) for i in xs for j in ys]
    elif typ == "line_fraction":
        x0, y0 = float(spec["x0"]), float(spec["y0"])
        x1, y1 = float(spec["x1"]), float(spec["y1"])
        tol = float(spec.get("tol", 0.02))
        dx, dy = x1 - x0, y1 - y0
        denom = math.hypot(dx, dy)
        if denom < 1e-12:
            raise ValueError("line_fraction selector needs distinct endpoints")
        nodes = []
        for i in range(nelx + 1):
            for j in range(nely + 1):
                x, y = i / max(1, nelx), j / max(1, nely)
                t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (denom * denom)))
                px, py = x0 + t * dx, y0 + t * dy
                if math.hypot(x - px, y - py) <= tol:
                    nodes.append(node_id(i, j, nely))
    elif typ == "circle_fraction":
        cx, cy, r = float(spec["x"]), float(spec["y"]), float(spec["r"])
        nodes = []
        for i in range(nelx + 1):
            for j in range(nely + 1):
                x, y = i / max(1, nelx), j / max(1, nely)
                if math.hypot(x - cx, y - cy) <= r:
                    nodes.append(node_id(i, j, nely))
    else:
        raise ValueError(f"Unsupported selector type {typ!r}")
    unique = sorted(set(nodes))
    if not unique:
        raise ValueError(f"Selector {selector!r} matched no nodes")
    return unique


def load_bc_spec(args: argparse.Namespace) -> dict[str, Any]:
    if args.bc_json and args.bc_json_string:
        raise ValueError("Use only one of --bc-json or --bc-json-string")
    if args.bc_json:
        spec = json.loads(Path(args.bc_json).read_text(encoding="utf-8"))
    elif args.bc_json_string:
        spec = json.loads(args.bc_json_string)
    else:
        if args.bc_preset == "custom":
            raise ValueError("--bc-preset custom requires --bc-json or --bc-json-string")
        if args.bc_preset not in BC_PRESETS:
            raise ValueError(f"Unknown --bc-preset {args.bc_preset!r}")
        spec = json.loads(json.dumps(BC_PRESETS[args.bc_preset]))
        spec["preset"] = args.bc_preset
    spec.setdefault("supports", [])
    spec.setdefault("loads", [])
    spec.setdefault("metadata", {})
    return spec


def resolve_boundary_conditions(
    spec: dict[str, Any],
    nelx: int,
    nely: int,
    load_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ndof = 2 * (nelx + 1) * (nely + 1)
    fixed: set[int] = set()
    f = np.zeros((ndof, 1), dtype=float)
    resolved_supports: list[dict[str, Any]] = []
    resolved_loads: list[dict[str, Any]] = []

    for idx, support in enumerate(spec.get("supports", [])):
        selector = support["selector"]
        nodes = resolve_selector(selector, nelx, nely)
        dofs: list[int] = []
        if bool(support.get("ux", False)):
            dofs.extend(2 * node for node in nodes)
        if bool(support.get("uy", False)):
            dofs.extend(2 * node + 1 for node in nodes)
        for dof in dofs:
            fixed.add(int(dof))
        resolved_supports.append(
            {
                "index": idx,
                "source": support.get("source", ""),
                "selector": selector,
                "ux": bool(support.get("ux", False)),
                "uy": bool(support.get("uy", False)),
                "node_count": len(nodes),
                "nodes": nodes,
                "dofs": sorted(set(int(d) for d in dofs)),
            }
        )

    for idx, load in enumerate(spec.get("loads", [])):
        selector = load["selector"]
        nodes = resolve_selector(selector, nelx, nely)
        fx = float(load.get("fx", 0.0))
        fy = float(load.get("fy", 0.0))
        distribution = str(load.get("distribution", "total"))
        if distribution == "normalized":
            magnitude = float(load.get("magnitude", 1.0)) * load_scale
            norm = math.hypot(fx, fy)
            if norm < 1e-12:
                raise ValueError("normalized load needs a nonzero fx/fy direction")
            total_fx, total_fy = magnitude * fx / norm, magnitude * fy / norm
            per_fx, per_fy = total_fx / len(nodes), total_fy / len(nodes)
        elif distribution == "per_node":
            per_fx, per_fy = fx * load_scale, fy * load_scale
            total_fx, total_fy = per_fx * len(nodes), per_fy * len(nodes)
        elif distribution == "total":
            total_fx, total_fy = fx * load_scale, fy * load_scale
            per_fx, per_fy = total_fx / len(nodes), total_fy / len(nodes)
        else:
            raise ValueError(f"Unsupported load distribution {distribution!r}")
        for node in nodes:
            f[2 * node, 0] += per_fx
            f[2 * node + 1, 0] += per_fy
        resolved_loads.append(
            {
                "index": idx,
                "source": load.get("source", ""),
                "selector": selector,
                "distribution": distribution,
                "node_count": len(nodes),
                "nodes": nodes,
                "total_fx": total_fx,
                "total_fy": total_fy,
                "per_node_fx": per_fx,
                "per_node_fy": per_fy,
                "dofs": sorted([2 * node for node in nodes] + [2 * node + 1 for node in nodes]),
            }
        )

    fixed_arr = np.array(sorted(fixed), dtype=int)
    loaded_dofs = np.flatnonzero(np.abs(f[:, 0]) > 1e-14)
    if fixed_arr.size == 0:
        raise ValueError("No fixed DOFs were defined")
    if loaded_dofs.size == 0:
        raise ValueError("No nonzero load DOFs were defined")
    if fixed_arr.size >= ndof:
        raise ValueError("All DOFs are fixed")
    resolved = {
        "node_grid": {"nelx": nelx, "nely": nely, "nodes_x": nelx + 1, "nodes_y": nely + 1},
        "coordinate_convention": {
            "i": "0..nelx, left to right",
            "j": "0..nely, bottom to top",
            "node_id": "i * (nely + 1) + j",
            "ux_dof": "2 * node_id",
            "uy_dof": "2 * node_id + 1",
        },
        "supports": resolved_supports,
        "loads": resolved_loads,
        "fixed_dof_count": int(fixed_arr.size),
        "loaded_dof_count": int(loaded_dofs.size),
        "total_force": {"fx": float(f[0::2, 0].sum()), "fy": float(f[1::2, 0].sum())},
    }
    return fixed_arr, f, resolved


def lk() -> np.ndarray:
    e = 1.0
    nu = 0.3
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ],
        dtype=float,
    )
    return e / (1 - nu**2) * np.array(
        [
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ],
        dtype=float,
    )


def oc_update(x: np.ndarray, volfrac: float, dc: np.ndarray, dv: np.ndarray, move: float = 0.2) -> np.ndarray:
    l1, l2 = 0.0, 1e9
    xnew = x.copy()
    while (l2 - l1) / max(l1 + l2, 1e-12) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        ratio = np.maximum(1e-12, -dc / np.maximum(1e-12, dv) / lmid)
        candidate = x * np.sqrt(ratio)
        xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, candidate))))
        if xnew.mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


def build_filter(nelx: int, nely: int, rmin: float) -> tuple[Any, np.ndarray]:
    nfilter = int(nelx * nely * ((2 * (math.ceil(rmin) - 1) + 1) ** 2))
    iH = np.zeros(nfilter, dtype=int)
    jH = np.zeros(nfilter, dtype=int)
    sH = np.zeros(nfilter, dtype=float)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(max(i - (math.ceil(rmin) - 1), 0))
            kk2 = int(min(i + math.ceil(rmin), nelx))
            ll1 = int(max(j - (math.ceil(rmin) - 1), 0))
            ll2 = int(min(j + math.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - math.sqrt((i - k) ** 2 + (j - l) ** 2)
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = max(0.0, fac)
                    cc += 1
    h = coo_matrix((sH[:cc], (iH[:cc], jH[:cc])), shape=(nelx * nely, nelx * nely)).tocsc()
    hs = np.asarray(h.sum(1)).ravel()
    return h, hs


def optimize(
    nelx: int,
    nely: int,
    volfrac: float,
    penal: float,
    rmin: float,
    filter_kind: str,
    maxiter: int,
    change_tol: float,
    fixed: np.ndarray,
    f: np.ndarray,
    log_every: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    emin, emax = 1e-9, 1.0
    ndof = 2 * (nelx + 1) * (nely + 1)
    x = volfrac * np.ones(nely * nelx, dtype=float)
    xphys = x.copy()
    ke = lk()

    edof = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof[el, :] = np.array([2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])
    ik = np.kron(edof, np.ones((8, 1), dtype=int)).ravel()
    jk = np.kron(edof, np.ones((1, 8), dtype=int)).ravel()
    h, hs = build_filter(nelx, nely, rmin)
    dofs = np.arange(ndof)
    free = np.setdiff1d(dofs, fixed)
    u = np.zeros((ndof, 1), dtype=float)

    history: list[dict[str, float]] = []
    filter_id = 0 if filter_kind == "sensitivity" else 1
    change = 1.0
    for loop in range(1, maxiter + 1):
        sk = ((ke.ravel()[np.newaxis]).T * (emin + xphys**penal * (emax - emin))).ravel(order="F")
        kglobal = coo_matrix((sk, (ik, jk)), shape=(ndof, ndof)).tocsc()
        kred = kglobal[free, :][:, free]
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=MatrixRankWarning)
            u[free, 0] = spsolve(kred, f[free, 0])
        ue = u[edof].reshape(nelx * nely, 8)
        ce = (np.dot(ue, ke) * ue).sum(1)
        obj = float(((emin + xphys**penal * (emax - emin)) * ce).sum())
        dc = -penal * xphys ** (penal - 1) * (emax - emin) * ce
        dv = np.ones(nely * nelx, dtype=float)

        if filter_id == 0:
            dc = np.asarray((h @ (x * dc)) / hs) / np.maximum(0.001, x)
        else:
            dc = np.asarray(h @ (dc / hs)).ravel()
            dv = np.asarray(h @ (dv / hs)).ravel()

        xold = x.copy()
        x = oc_update(x, volfrac, dc, dv)
        if filter_id == 0:
            xphys = x.copy()
        else:
            xphys = np.asarray(h @ x / hs).ravel()
        change = float(np.linalg.norm(x - xold, ord=np.inf))
        row = {"iteration": float(loop), "compliance": obj, "volume_fraction": float(xphys.mean()), "change": change}
        history.append(row)
        if log_every > 0 and (loop == 1 or loop % log_every == 0 or loop == maxiter or change <= change_tol):
            print(f"iter={loop:4d} compliance={obj:12.5f} vol={xphys.mean():.4f} change={change:.5f}")
        if change <= change_tol:
            break
    return xphys, history


def density_image(xphys: np.ndarray, nelx: int, nely: int) -> np.ndarray:
    return xphys.reshape((nelx, nely)).T


def resize_density(density: np.ndarray, scale: float) -> np.ndarray:
    if math.isclose(scale, 1.0):
        return density.copy()
    scale = max(0.1, min(4.0, scale))
    return np.clip(zoom(density, zoom=scale, order=3), 0.0, 1.0)


def save_density_png(path: Path, density: np.ndarray, title: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=dpi)
    im = ax.imshow(density, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("element x")
    ax.set_ylabel("element y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="density")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_convergence_png(path: Path, history: list[dict[str, float]], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=dpi)
    if history:
        iterations = [h["iteration"] for h in history]
        compliance = [h["compliance"] for h in history]
        ax.plot(iterations, compliance, color="#b3362d", lw=2)
    ax.set_title("Topology optimization convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("compliance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_bc_preview(path: Path, nelx: int, nely: int, resolved: dict[str, Any], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=dpi)
    ax.set_xlim(-0.05 * nelx, 1.05 * nelx)
    ax.set_ylim(-0.12 * max(nely, 1), 1.15 * max(nely, 1))
    ax.add_patch(plt.Rectangle((0, 0), nelx, nely, fill=False, lw=1.5, color="#222222"))

    for support in resolved["supports"]:
        xs, ys = [], []
        for node in support["nodes"]:
            i, j = node_ij(node, nely)
            xs.append(i)
            ys.append(j)
        marker = "s" if support["ux"] and support["uy"] else "^"
        ax.scatter(xs, ys, s=18, marker=marker, color="#2f6fba", alpha=0.75, label="support")

    max_load = 0.0
    load_vectors: list[tuple[float, float, float, float]] = []
    for load in resolved["loads"]:
        fx = float(load["per_node_fx"])
        fy = float(load["per_node_fy"])
        max_load = max(max_load, math.hypot(fx, fy))
        for node in load["nodes"]:
            i, j = node_ij(node, nely)
            load_vectors.append((float(i), float(j), fx, fy))
    scale = 0.12 * max(nelx, nely) / max(max_load, 1e-12)
    for x, y, fx, fy in load_vectors:
        ax.arrow(x, y, fx * scale, fy * scale, color="#c83b32", width=0.02 * max(nely, 1), head_width=0.8, length_includes_head=True, alpha=0.8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Boundary condition preview")
    ax.set_xlabel("node i")
    ax.set_ylabel("node j")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def triangle_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def write_ascii_stl(path: Path, triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]], name: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"solid {name}\n")
        for a, b, c in triangles:
            n = triangle_normal(a, b, c)
            handle.write(f"  facet normal {n[0]:.8e} {n[1]:.8e} {n[2]:.8e}\n")
            handle.write("    outer loop\n")
            for p in (a, b, c):
                handle.write(f"      vertex {p[0]:.8e} {p[1]:.8e} {p[2]:.8e}\n")
            handle.write("    endloop\n")
            handle.write("  endfacet\n")
        handle.write(f"endsolid {name}\n")


def box_triangles(x: float, y: float, z: float, sx: float, sy: float, sz: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x0, x1 = x, x + sx
    y0, y1 = y, y + sy
    z0, z1 = z, z + sz
    p = [
        np.array([x0, y0, z0], dtype=float),
        np.array([x1, y0, z0], dtype=float),
        np.array([x1, y1, z0], dtype=float),
        np.array([x0, y1, z0], dtype=float),
        np.array([x0, y0, z1], dtype=float),
        np.array([x1, y0, z1], dtype=float),
        np.array([x1, y1, z1], dtype=float),
        np.array([x0, y1, z1], dtype=float),
    ]
    idx = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    return [(p[a], p[b], p[c]) for a, b, c in idx]


def voxel_stl_from_density(
    density: np.ndarray,
    path: Path,
    name: str,
    max_height: float,
    dens_cut_min: float,
    dens_cut_max: float,
    profile: bool,
    profile_origin: str = "bottom",
) -> dict[str, Any]:
    d = np.asarray(density, dtype=float)
    max_density = max(float(d.max()), 1e-12)
    triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    active = 0
    nrows, ncols = d.shape
    for y in range(nrows):
        for x in range(ncols):
            rho = float(np.clip(d[y, x] / max_density, 0.0, 1.0))
            if rho <= dens_cut_min or rho > dens_cut_max:
                continue
            height = max_height * rho if profile else max_height
            if height <= 1e-12:
                continue
            active += 1
            z0 = -0.5 * height if profile and profile_origin == "center" else 0.0
            triangles.extend(box_triangles(float(x), float(y), z0, 1.0, 1.0, height))
    write_ascii_stl(path, triangles, name)
    return {
        "path": path.name,
        "active_cells": active,
        "triangles": len(triangles),
        "profile_origin": profile_origin if profile else "bottom",
    }


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "compliance", "volume_fraction", "change"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def parse_material_cuts(text: str) -> list[float]:
    cuts = sorted({float(part.strip()) for part in text.split(",") if part.strip()})
    cuts = [max(0.0, min(1.0, cut)) for cut in cuts]
    if len(cuts) < 2:
        raise ValueError("--material-cuts needs at least two values")
    if cuts[-1] < 1.0:
        cuts.append(1.0)
    return cuts


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bc_spec = load_bc_spec(args)
    fixed, force, resolved = resolve_boundary_conditions(bc_spec, args.nelx, args.nely, args.load_scale)
    (out / "boundary_conditions.json").write_text(json.dumps(bc_spec, indent=2) + "\n", encoding="utf-8")
    (out / "boundary_conditions_resolved.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    save_bc_preview(out / "bc_preview.png", args.nelx, args.nely, resolved, args.dpi)

    xphys, history = optimize(
        args.nelx,
        args.nely,
        args.volfrac,
        args.penal,
        args.rmin,
        args.filter,
        args.maxiter,
        args.change_tol,
        fixed,
        force,
        args.log_every,
    )
    density = density_image(xphys, args.nelx, args.nely)
    resized = resize_density(density, args.resize_scale)
    np.save(out / "density.npy", density)
    np.save(out / "density_resized.npy", resized)
    np.savetxt(out / "density.csv", density, delimiter=",")
    save_density_png(out / "density.png", density, "Optimized density", args.dpi)
    save_density_png(out / "density_resized.png", resized, "Optimized density for mesh export", args.dpi)
    save_convergence_png(out / "convergence.png", history, args.dpi)
    write_history_csv(out / "optimization_history.csv", history)

    mesh_outputs: list[dict[str, Any]] = []
    mode = args.mesh_mode
    if mode in {"profile-stl", "all-basic"}:
        mesh_outputs.append(
            voxel_stl_from_density(
                resized,
                out / "result_profile.stl",
                "result_profile",
                args.max_height,
                args.dens_cut,
                1.0,
                True,
                args.profile_origin,
            )
        )
    if mode in {"flat-stl", "all-basic"}:
        mesh_outputs.append(voxel_stl_from_density(resized, out / "result_flat.stl", "result_flat", args.max_height, args.dens_cut, 1.0, False))
    if mode in {"multimaterial-stl", "all-basic"}:
        cuts = parse_material_cuts(args.material_cuts)
        band_index = 1
        for low, high in reversed(list(zip(cuts[:-1], cuts[1:]))):
            if high <= args.dens_cut:
                continue
            low = max(low, args.dens_cut)
            mesh_outputs.append(voxel_stl_from_density(resized, out / f"MAT{band_index}.stl", f"MAT{band_index}", args.max_height, low, high, False))
            band_index += 1

    final = history[-1] if history else {"iteration": 0.0, "compliance": float("nan"), "volume_fraction": float("nan"), "change": float("nan")}
    parameters = vars(args).copy()
    parameters["boundary_conditions"] = bc_spec
    outputs = {
        "density_png": "density.png",
        "density_resized_png": "density_resized.png",
        "density_npy": "density.npy",
        "density_resized_npy": "density_resized.npy",
        "density_csv": "density.csv",
        "optimization_history_csv": "optimization_history.csv",
        "boundary_conditions_json": "boundary_conditions.json",
        "boundary_conditions_resolved_json": "boundary_conditions_resolved.json",
        "bc_preview_png": "bc_preview.png",
        "convergence_png": "convergence.png",
        "summary_json": "summary.json",
        "parameters_json": "parameters.json",
        "readme": "README.md",
        "meshes": [item["path"] for item in mesh_outputs],
    }
    summary = {
        "title": args.title,
        "nelx": args.nelx,
        "nely": args.nely,
        "elements": args.nelx * args.nely,
        "volfrac_target": args.volfrac,
        "final_volume_fraction": float(final["volume_fraction"]),
        "final_compliance": float(final["compliance"]),
        "iterations": int(final["iteration"]),
        "change": float(final["change"]),
        "filter": args.filter,
        "penal": args.penal,
        "rmin": args.rmin,
        "fixed_dof_count": resolved["fixed_dof_count"],
        "loaded_dof_count": resolved["loaded_dof_count"],
        "total_force": resolved["total_force"],
        "mesh_mode": mode,
        "profile_origin": args.profile_origin,
        "mesh_outputs": mesh_outputs,
        "outputs": outputs,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Hierarchical Topology Optimization Run

Generated by `hierarchical-topopt`.

- title: {args.title}
- design: {args.nelx} x {args.nely} elements
- target volume fraction: {args.volfrac}
- final volume fraction: {summary['final_volume_fraction']:.6f}
- final compliance: {summary['final_compliance']:.6f}
- iterations: {summary['iterations']}
- filter: {args.filter}
- penalization: {args.penal}
- filter radius: {args.rmin}
- mesh mode: {args.mesh_mode}
- profile origin: {args.profile_origin}
- fixed DOFs: {resolved['fixed_dof_count']}
- loaded DOFs: {resolved['loaded_dof_count']}
- total force: fx={resolved['total_force']['fx']:.6f}, fy={resolved['total_force']['fy']:.6f}

Artifacts:

- `density.png`: optimized density field
- `density_resized.png`: mesh-export density field
- `density.npy`, `density_resized.npy`, `density.csv`: numerical density data
- `optimization_history.csv`: compliance, volume fraction, and change by iteration
- `boundary_conditions.json`: requested supports and loads
- `boundary_conditions_resolved.json`: exact node IDs, DOFs, and load distribution
- `bc_preview.png`: visual support/load preview
- `convergence.png`: compliance convergence plot
- `summary.json`: run metrics
- `parameters.json`: exact run parameters
"""
    if mesh_outputs:
        readme += "\nMeshes:\n\n" + "\n".join(f"- `{item['path']}`: {item['active_cells']} active cells, {item['triangles']} triangles" for item in mesh_outputs) + "\n"
    readme += "\nThis is a fast educational/research design generator, not a substitute for validated engineering analysis.\n"
    (out / "README.md").write_text(readme, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="skill_output_hierarchical-topopt")
    parser.add_argument("--title", default="Hierarchical Topology Optimization")
    parser.add_argument("--nelx", type=int, default=90)
    parser.add_argument("--nely", type=int, default=30)
    parser.add_argument("--volfrac", type=float, default=0.55)
    parser.add_argument("--penal", type=float, default=4.0)
    parser.add_argument("--rmin", type=float, default=4.5)
    parser.add_argument("--filter", choices=("density", "sensitivity"), default="density")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--change-tol", type=float, default=0.01)
    parser.add_argument("--bc-preset", default="cantilever-mid-down", choices=sorted(list(BC_PRESETS) + ["custom"]))
    parser.add_argument("--bc-json", default=None)
    parser.add_argument("--bc-json-string", default=None)
    parser.add_argument("--load-scale", type=float, default=1.0)
    parser.add_argument("--mesh-mode", choices=("density-only", "profile-stl", "flat-stl", "multimaterial-stl", "all-basic"), default="profile-stl")
    parser.add_argument("--resize-scale", type=float, default=1.0)
    parser.add_argument("--max-height", type=float, default=10.0)
    parser.add_argument("--profile-origin", choices=("bottom", "center"), default="bottom")
    parser.add_argument("--dens-cut", type=float, default=0.30)
    parser.add_argument("--material-cuts", default="0.2,0.8,1.0")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    args.nelx = max(4, min(400, args.nelx))
    args.nely = max(2, min(200, args.nely))
    args.volfrac = max(0.02, min(0.98, args.volfrac))
    args.penal = max(1.0, min(8.0, args.penal))
    args.rmin = max(1.0, min(50.0, args.rmin))
    args.maxiter = max(1, min(2000, args.maxiter))
    args.change_tol = max(1e-5, min(0.5, args.change_tol))
    args.resize_scale = max(0.1, min(4.0, args.resize_scale))
    args.max_height = max(0.01, min(1000.0, args.max_height))
    args.dens_cut = max(0.0, min(0.99, args.dens_cut))
    args.dpi = max(72, min(300, args.dpi))
    return args


def main() -> None:
    args = parse_args()
    summary = run(args)
    out = Path(args.out)
    print(f"Completed {summary['iterations']} iterations")
    print(f"Final compliance: {summary['final_compliance']:.6f}")
    print(f"Final volume fraction: {summary['final_volume_fraction']:.6f}")
    print(f"Fixed DOFs: {summary['fixed_dof_count']} | Loaded DOFs: {summary['loaded_dof_count']}")
    print(f"Wrote {out / 'density.png'}")
    print(f"Wrote {out / 'bc_preview.png'}")
    print(f"Wrote {out / 'boundary_conditions_resolved.json'}")
    print(f"Wrote {out / 'summary.json'}")
    print(f"Wrote {out / 'parameters.json'}")
    for mesh in summary["outputs"].get("meshes", []):
        print(f"Wrote {out / mesh}")


if __name__ == "__main__":
    main()
