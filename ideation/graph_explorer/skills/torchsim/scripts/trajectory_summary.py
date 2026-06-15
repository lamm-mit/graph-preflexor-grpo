#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
import numpy as np
import torch_sim as ts

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("trajectory"); args = ap.parse_args()
    names = ["positions","atomic_numbers","cell","masses","momenta","velocities","forces","potential_energy","kinetic_energy","temperature","total_energy","pressure","stress","volume"]
    info = {"trajectory": args.trajectory, "arrays": {}}
    with ts.TorchSimTrajectory(args.trajectory, mode="r") as traj:
        try: info["repr"] = str(traj)
        except Exception: pass
        for name in names:
            try:
                a = np.asarray(traj.get_array(name)); info["arrays"][name] = {"shape": list(a.shape), "dtype": str(a.dtype)}
            except Exception: continue
        try:
            atoms = traj.get_atoms(-1); info["final_atoms"] = {"n_atoms": len(atoms), "symbols": sorted(set(atoms.get_chemical_symbols()))}
        except Exception as exc: info["final_atoms_error"] = repr(exc)
    print(json.dumps(info, indent=2))
if __name__ == "__main__": main()
