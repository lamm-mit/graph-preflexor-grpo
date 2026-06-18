---
name: hierarchical-topopt
description: Run fast 2D SIMP topology optimization with flexible boundary conditions and lightweight STL exports. Use when asked to optimize cantilevers, beams, bridges, plates, tension/shear strips, custom support/load layouts, density fields, compliance minimization, profile/flat/multimaterial STL meshes, or topology optimization artifacts from text-defined boundary conditions. Produces density plots/data, boundary-condition previews and resolved node/DOF JSON, optimization history, summary files, and optional STL meshes by running a bundled Python script.
license: 3-clause BSD license
metadata: {"version": "1.0", "skill-author": "Markus J. Buehler, Massachusetts Institute of Technology (MIT), Laboratory for Atomistic and Molecular Mechanics (LAMM)"}
---

# Hierarchical TopOpt

Run a headless 2D minimum-compliance topology optimization workflow with flexible
supports and loads. Always use the bundled script; do not rewrite the optimizer
from scratch.

This skill intentionally does not generate gyroid, hole, voxel microstructure,
marching-cubes, or PyVista boolean meshes. Keep it fast and robust for local
agent execution.

## Required Workflow

1. Create or use an output directory, usually `skill_output_<timestamp>_hierarchical-topopt`.
2. Select a boundary condition preset or write `boundary_conditions.json`.
3. Run `scripts/run_hierarchical_topopt.py`.
4. Verify artifacts with `find`.
5. Final answer must list exact artifact paths and report compliance, final volume fraction, fixed DOFs, loaded DOFs, and total force from `summary.json`.

Default command:

```bash
python3 skills/hierarchical-topopt/scripts/run_hierarchical_topopt.py \
  --out skill_output_<timestamp>_hierarchical-topopt \
  --title "Hierarchical Topology Optimization" \
  --nelx 90 \
  --nely 30 \
  --volfrac 0.55 \
  --penal 4.0 \
  --rmin 4.5 \
  --filter density \
  --maxiter 200 \
  --bc-preset cantilever-mid-down \
  --mesh-mode profile-stl \
  --dens-cut 0.30 \
  --max-height 10
```

## Core Parameters

- `--nelx`, `--nely`: number of finite elements in x/y.
- `--volfrac`: target volume fraction.
- `--penal`: SIMP penalization.
- `--rmin`: density/sensitivity filter radius.
- `--filter`: `density` or `sensitivity`.
- `--maxiter`: maximum optimizer iterations.
- `--bc-preset`: named support/load preset.
- `--bc-json`: file containing custom boundary conditions.
- `--bc-json-string`: inline custom boundary-condition JSON.
- `--mesh-mode`: `density-only`, `profile-stl`, `flat-stl`, `multimaterial-stl`, or `all-basic`.
- `--resize-scale`: scale density field before STL export.
- `--dens-cut`: densities at or below this value are void for STL export.
- `--max-height`: STL z-height.
- `--material-cuts`: comma-separated density bands for multimaterial export, e.g. `0.2,0.55,0.8,1.0`.

## Boundary Condition Model

Use normalized geometry selectors whenever the user describes locations in words.
The script resolves them to exact node IDs and DOFs and writes
`boundary_conditions_resolved.json`.

Coordinate convention:

```text
elements: nelx by nely
nodes: (nelx + 1) by (nely + 1)
i: 0..nelx, left to right
j: 0..nely, bottom to top
node_id(i,j) = i * (nely + 1) + j
ux_dof = 2 * node_id
uy_dof = 2 * node_id + 1
```

Boundary condition JSON:

```json
{
  "supports": [
    {"selector": "left_edge", "ux": true, "uy": true}
  ],
  "loads": [
    {"selector": "right_mid", "fx": 0.0, "fy": -1.0, "distribution": "total"}
  ]
}
```

Always prefer `distribution: "total"` for loads unless the user explicitly says
force per node. A total load is split across selected nodes.

## Selector Types

String aliases:

- `left_edge`, `right_edge`, `top_edge`, `bottom_edge`
- `left_mid`, `right_mid`, `top_mid`, `bottom_mid`, `center`
- `left_bottom`, `right_bottom`, `left_top`, `right_top`
- `bottom_left`, `bottom_right`, `top_left`, `top_right`
- `node:i,j`

Structured selectors:

```json
{"type": "node", "i": 90, "j": 15}
{"type": "node_fraction", "x": 1.0, "y": 0.5}
{"type": "edge", "edge": "left"}
{"type": "edge_fraction", "edge": "top", "start": 0.4, "end": 0.6}
{"type": "x_range_fraction", "x0": 0.7, "x1": 1.0}
{"type": "y_range_fraction", "y0": 0.0, "y1": 0.1}
{"type": "box_fraction", "x0": 0.4, "x1": 0.6, "y0": 0.95, "y1": 1.0}
{"type": "line_fraction", "x0": 0.0, "y0": 0.5, "x1": 1.0, "y1": 0.5, "tol": 0.02}
{"type": "circle_fraction", "x": 0.5, "y": 0.5, "r": 0.05}
```

## Translation Examples

Use these examples to convert user text into `boundary_conditions.json`.

Fix the left edge:

```json
{"supports": [{"selector": "left_edge", "ux": true, "uy": true}], "loads": []}
```

Pin lower left and roller lower right:

```json
{
  "supports": [
    {"selector": "left_bottom", "ux": true, "uy": true},
    {"selector": "right_bottom", "ux": false, "uy": true}
  ],
  "loads": []
}
```

Fix all nodes 70 percent to the right:

```json
{
  "supports": [
    {"selector": {"type": "x_range_fraction", "x0": 0.7, "x1": 1.0}, "ux": true, "uy": true}
  ],
  "loads": []
}
```

Fix the bottom 10 percent strip:

```json
{
  "supports": [
    {"selector": {"type": "y_range_fraction", "y0": 0.0, "y1": 0.1}, "ux": true, "uy": true}
  ],
  "loads": []
}
```

Load the middle 20 percent of the top edge downward:

```json
{
  "supports": [{"selector": "left_edge", "ux": true, "uy": true}],
  "loads": [
    {"selector": {"type": "edge_fraction", "edge": "top", "start": 0.4, "end": 0.6}, "fx": 0.0, "fy": -1.0, "distribution": "total"}
  ]
}
```

Bridge with pin/roller supports and a distributed top load:

```json
{
  "supports": [
    {"selector": {"type": "node_fraction", "x": 0.0, "y": 0.0}, "ux": true, "uy": true},
    {"selector": {"type": "node_fraction", "x": 1.0, "y": 0.0}, "ux": false, "uy": true}
  ],
  "loads": [
    {"selector": {"type": "edge_fraction", "edge": "top", "start": 0.4, "end": 0.6}, "fx": 0.0, "fy": -1.0, "distribution": "total"}
  ]
}
```

Shear the right edge:

```json
{
  "supports": [{"selector": "left_edge", "ux": true, "uy": true}],
  "loads": [{"selector": "right_edge", "fx": 1.0, "fy": 0.0, "distribution": "total"}]
}
```

Push a square patch near the upper-right corner:

```json
{
  "supports": [{"selector": "left_edge", "ux": true, "uy": true}],
  "loads": [
    {"selector": {"type": "box_fraction", "x0": 0.8, "x1": 1.0, "y0": 0.8, "y1": 1.0}, "fx": 1.0, "fy": 0.0, "distribution": "total"}
  ]
}
```

Load the center point downward:

```json
{
  "supports": [{"selector": "bottom_edge", "ux": true, "uy": true}],
  "loads": [{"selector": {"type": "node_fraction", "x": 0.5, "y": 0.5}, "fx": 0.0, "fy": -1.0, "distribution": "total"}]
}
```

## Presets

Use presets for common cases:

- `cantilever-tip-down`
- `cantilever-mid-down`
- `bridge-center-load`
- `simply-supported-center-load`
- `fixed-fixed-center-load`
- `tension-strip`
- `shear-strip`

Use `--bc-json boundary_conditions.json` when the user describes a custom setup.
When using custom boundary conditions, write the JSON file first, then pass it to
the script.

## Outputs

Always written:

- `density.png`
- `density_resized.png`
- `density.npy`
- `density_resized.npy`
- `density.csv`
- `optimization_history.csv`
- `boundary_conditions.json`
- `boundary_conditions_resolved.json`
- `bc_preview.png`
- `convergence.png`
- `summary.json`
- `parameters.json`
- `README.md`

Mesh outputs depend on `--mesh-mode`:

- `result_profile.stl`
- `result_flat.stl`
- `MAT1.stl`, `MAT2.stl`, ...

Do not claim success unless `find` lists the density image, BC preview, resolved
BC JSON, optimization history, summary, parameters, README, and requested mesh
files.
