#!/usr/bin/env python3
"""Relax a structure and run a static sanity check with TorchSim."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from ase.io import read, write
import torch_sim as ts
try:
    from run_md import as_torch_dtype, build_model, choose_device, enum_value
except Exception:
    from templates.run_md import as_torch_dtype, build_model, choose_device, enum_value

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--input"); ap.add_argument("--output", default="relaxed.extxyz"); args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(cfg.get("output", {}).get("directory", "runs/relax")); outdir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.get("run", {}).get("device", "auto")); dtype = as_torch_dtype(cfg.get("run", {}).get("dtype", "float64"))
    model = build_model(cfg.get("model", {}), device, dtype)
    structure_file = args.input or cfg.get("system", {}).get("file")
    if not structure_file: raise ValueError("Provide --input or system.file")
    atoms = read(structure_file, format=cfg.get("system", {}).get("format"))
    if cfg.get("system", {}).get("pbc") is not None: atoms.pbc = cfg["system"]["pbc"]
    relax = cfg.get("relax", {}); optimizer = enum_value(ts.Optimizer, relax.get("optimizer", "fire"))
    init_kwargs = {}
    if relax.get("relax_cell", False) and hasattr(ts, "CellFilter"):
        try: init_kwargs["cell_filter"] = ts.CellFilter.frechet
        except Exception: pass
    kwargs = dict(system=atoms, model=model, optimizer=optimizer, max_steps=int(relax.get("max_steps", 500)), autobatcher=(device.type != "cpu"))
    if init_kwargs: kwargs["init_kwargs"] = init_kwargs
    final_state = ts.optimize(**kwargs)
    final_atoms = final_state.to_atoms(); final_atoms = final_atoms[0] if isinstance(final_atoms, list) else final_atoms
    outpath = outdir / args.output; write(outpath, final_atoms)
    summary = {"relaxed_structure": str(outpath), "n_atoms": len(final_atoms)}
    try:
        static_state = ts.static(system=final_atoms, model=model)
        if hasattr(static_state, "energy"): summary["energy"] = static_state.energy.detach().cpu().numpy().tolist()
    except Exception as exc: summary["static_warning"] = repr(exc)
    (outdir / "relax_summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
