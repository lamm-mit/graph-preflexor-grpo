#!/usr/bin/env python3
"""Run a TorchSim MD simulation from a YAML config."""
from __future__ import annotations
import argparse, json, platform, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml
from ase.build import bulk
from ase.io import read, write
import torch_sim as ts

def as_torch_dtype(name: str) -> torch.dtype:
    table = {"float32": torch.float32, "float64": torch.float64}
    if name not in table:
        raise ValueError(f"Unsupported dtype {name!r}; use {sorted(table)}")
    return table[name]

def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_system(cfg: dict[str, Any]):
    src = cfg.get("source", "file")
    if src == "file":
        path = cfg.get("file")
        if not path: raise ValueError("system.file required when system.source=file")
        atoms = read(path, format=cfg.get("format"))
    elif src == "ase_bulk":
        b = cfg.get("ase_bulk", {})
        atoms = bulk(b.get("symbol", "Cu"), crystalstructure=b.get("crystalstructure", "fcc"), a=b.get("a"), cubic=bool(b.get("cubic", True)))
        atoms = atoms.repeat(tuple(b.get("repeat", [1, 1, 1])))
    else:
        raise ValueError(f"Unsupported system.source={src!r}")
    if cfg.get("pbc") is not None: atoms.pbc = cfg["pbc"]
    return atoms

def build_model(cfg: dict[str, Any], device: torch.device, dtype: torch.dtype):
    kind = cfg.get("kind", "lennard_jones")
    if kind == "lennard_jones":
        from torch_sim.models.lennard_jones import LennardJonesModel
        p = cfg.get("lennard_jones", {})
        return LennardJonesModel(sigma=float(p.get("sigma", 2.0)), epsilon=float(p.get("epsilon", 0.1)), device=device, dtype=dtype)
    if kind == "morse":
        from torch_sim.models.morse import MorseModel
        p = cfg.get("morse", {})
        return MorseModel(sigma=float(p.get("sigma", 2.866)), epsilon=float(p.get("epsilon", 0.3429)), alpha=float(p.get("alpha", 1.3588)), device=device, dtype=dtype)
    if kind == "mace":
        from mace.calculators.foundations_models import mace_mp
        from torch_sim.models.mace import MaceModel
        p = cfg.get("mace", {})
        raw = mace_mp(model=p.get("foundation_model", "small"), return_raw_model=True)
        return MaceModel(model=raw, device=device, dtype=dtype, compute_forces=True)
    raise ValueError(f"Unsupported model.kind={kind!r}; extend build_model for other MLIPs")

def enum_value(enum_cls, requested: str):
    if hasattr(enum_cls, requested): return getattr(enum_cls, requested)
    names = [x for x in dir(enum_cls) if not x.startswith("_")]
    lowered = {x.lower(): x for x in names}
    if requested.lower() in lowered: return getattr(enum_cls, lowered[requested.lower()])
    raise ValueError(f"Cannot find {requested!r}. Available: {names}")

def make_reporter(filename: Path, scalar_frequency: int, state_frequency: int):
    def kinetic_energy(state):
        if not hasattr(state, "momenta"): return torch.full_like(state.energy, torch.nan)
        return ts.calc_kinetic_energy(momenta=state.momenta, masses=state.masses)
    def temperature(state):
        if not hasattr(state, "momenta"): return torch.full_like(state.energy, torch.nan)
        return ts.calc_temperature(momenta=state.momenta, masses=state.masses)
    return ts.TrajectoryReporter(
        filenames=str(filename),
        state_frequency=int(state_frequency),
        prop_calculators={int(scalar_frequency): {"potential_energy": lambda state: state.energy, "kinetic_energy": kinetic_energy, "temperature": temperature}},
    )

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("config"); args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(int(cfg.get("run", {}).get("seed", 12345)))
    outdir = Path(cfg.get("output", {}).get("directory", "runs/torchsim_run")); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.used.yaml").write_text(Path(args.config).read_text())
    device = choose_device(cfg.get("run", {}).get("device", "auto")); dtype = as_torch_dtype(cfg.get("run", {}).get("dtype", "float32"))
    atoms = build_system(cfg.get("system", {})); model = build_model(cfg.get("model", {}), device, dtype)
    md = cfg.get("md", {}); traj_file = outdir / cfg.get("output", {}).get("trajectory", "trajectory.h5")
    reporter = make_reporter(traj_file, int(md.get("scalar_frequency", 10)), int(md.get("state_frequency", 100)))
    autobatcher = bool(md.get("autobatcher", True))
    if device.type == "cpu" and autobatcher:
        # TorchSim memory-estimation autobatching is GPU-oriented; CPU pilot runs should disable it.
        autobatcher = False
    kwargs: dict[str, Any] = dict(system=atoms, model=model, integrator=enum_value(ts.Integrator, md.get("ensemble", "nvt_langevin")), n_steps=int(md.get("n_steps", 1000)), timestep=float(md.get("timestep_ps", 0.001)), trajectory_reporter=reporter, autobatcher=autobatcher)
    if md.get("temperature_K") is not None: kwargs["temperature"] = float(md["temperature_K"])
    if md.get("pressure_bar") is not None: kwargs["pressure"] = float(md["pressure_bar"])
    final_state = ts.integrate(**kwargs)
    final_atoms = final_state.to_atoms(); final_atoms = final_atoms[0] if isinstance(final_atoms, list) else final_atoms
    final_path = outdir / cfg.get("output", {}).get("final_structure", "final.extxyz"); write(final_path, final_atoms)
    summary = {"run_name": cfg.get("run", {}).get("name"), "trajectory": str(traj_file), "final_structure": str(final_path), "n_atoms": len(atoms), "device": str(device), "dtype": str(dtype), "python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (outdir / cfg.get("output", {}).get("summary", "summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
