#!/usr/bin/env python3
"""Batched TorchSim sweep over temperatures and/or input structures."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from ase.io import read
import torch_sim as ts
try:
    from run_md import as_torch_dtype, build_model, choose_device, enum_value
except Exception:
    from templates.run_md import as_torch_dtype, build_model, choose_device, enum_value

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--structures", nargs="+", required=True); ap.add_argument("--temperatures", nargs="+", type=float, required=True); ap.add_argument("--outdir", default="runs/batch_sweep"); args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()); outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.get("run", {}).get("device", "auto")); dtype = as_torch_dtype(cfg.get("run", {}).get("dtype", "float32")); model = build_model(cfg.get("model", {}), device, dtype)
    labels, systems = [], []
    for path in args.structures:
        atoms = read(path)
        for T in args.temperatures: systems.append(atoms.copy()); labels.append(f"{Path(path).stem}_T{int(T):04d}")
    md = cfg.get("md", {}); integrator = enum_value(ts.Integrator, md.get("ensemble", "nvt_langevin")); filenames = [str(outdir / f"{label}.h5") for label in labels]
    results = []
    for T in args.temperatures:
        group_indices = [i for i, label in enumerate(labels) if label.endswith(f"T{int(T):04d}")]
        group_systems = [systems[i] for i in group_indices]; group_files = [filenames[i] for i in group_indices]
        ts.integrate(system=group_systems, model=model, integrator=integrator, n_steps=int(md.get("n_steps",1000)), timestep=float(md.get("timestep_ps",0.001)), temperature=float(T), trajectory_reporter={"filenames": group_files, "state_frequency": int(md.get("state_frequency",100))}, autobatcher=(bool(md.get("autobatcher", True)) and device.type != "cpu"))
        results.append({"temperature_K": T, "n_systems": len(group_systems), "files": group_files})
    summary = {"labels": labels, "results": results, "device": str(device), "dtype": str(dtype)}; (outdir/"batch_summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
