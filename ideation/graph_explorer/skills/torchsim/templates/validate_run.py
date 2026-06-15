#!/usr/bin/env python3
"""Basic validation checks for TorchSim trajectory outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch_sim as ts

def arr(traj, name):
    try: return np.asarray(traj.get_array(name))
    except Exception: return None

def finite_stats(a):
    if a is None: return None
    x = np.asarray(a, dtype=float).reshape(-1)
    return {"finite": bool(np.isfinite(x).all()), "min": float(np.nanmin(x)) if x.size else None, "max": float(np.nanmax(x)) if x.size else None, "mean": float(np.nanmean(x)) if x.size else None, "std": float(np.nanstd(x)) if x.size else None, "n": int(x.size)}

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("trajectory"); ap.add_argument("--target-temperature", type=float); ap.add_argument("--summary", default="validation.json"); ap.add_argument("--temperature-tol-frac", type=float, default=0.20); args = ap.parse_args()
    checks = {"trajectory": args.trajectory, "warnings": []}
    with ts.TorchSimTrajectory(args.trajectory, mode="r") as traj:
        positions, pe, ke, temp, cell = (arr(traj, n) for n in ["positions", "potential_energy", "kinetic_energy", "temperature", "cell"])
        checks["positions_shape"] = list(positions.shape) if positions is not None else None
        for name, a in [("potential_energy", pe), ("kinetic_energy", ke), ("temperature", temp), ("cell", cell)]:
            checks[name] = finite_stats(a)
            if checks[name] is not None and not checks[name]["finite"]: checks["warnings"].append(f"{name} contains NaN/Inf")
        if pe is not None and len(np.asarray(pe).reshape(-1)) > 2:
            e = np.asarray(pe, dtype=float).reshape(-1); drift = e[-1] - e[0]; checks["potential_energy_drift"] = float(drift); checks["potential_energy_drift_per_frame"] = float(drift/max(len(e)-1, 1))
        if args.target_temperature is not None and temp is not None:
            t = np.asarray(temp, dtype=float).reshape(-1); mean_t = float(np.nanmean(t[int(0.2*len(t)):])) if len(t) else float("nan"); checks["mean_temperature_after_20pct"] = mean_t
            if np.isfinite(mean_t):
                rel = abs(mean_t - args.target_temperature)/max(args.target_temperature, 1.0); checks["temperature_relative_error"] = float(rel)
                if rel > args.temperature_tol_frac: checks["warnings"].append(f"Mean temperature {mean_t:.3g} K differs from target")
        try:
            final_atoms = traj.get_atoms(-1); d = final_atoms.get_all_distances(mic=bool(np.any(final_atoms.pbc))); d[d == 0] = np.nan; checks["final_min_pair_distance_A"] = float(np.nanmin(d))
            if checks["final_min_pair_distance_A"] < 0.5: checks["warnings"].append("Very small final nearest-neighbor distance")
        except Exception as exc: checks["warnings"].append(f"Could not compute final pair distances: {exc!r}")
    Path(args.summary).write_text(json.dumps(checks, indent=2)); print(json.dumps(checks, indent=2))
if __name__ == "__main__": main()
