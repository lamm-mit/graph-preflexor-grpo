#!/usr/bin/env python3
"""Analyze a TorchSim HDF5 trajectory: scalars, RDF, MSD, final/export structure."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
import torch_sim as ts

def safe_array(traj, name: str):
    try: return np.asarray(traj.get_array(name))
    except Exception: return None

def list_arrays(traj) -> list[str]:
    candidates = ["positions","atomic_numbers","cell","masses","forces","momenta","velocities","potential_energy","kinetic_energy","temperature","total_energy","pressure","stress","volume"]
    return [name for name in candidates if safe_array(traj, name) is not None]

def write_scalar_csv(out: Path, arrays: dict[str, np.ndarray]) -> None:
    scalar = {}; n = None
    for k, v in arrays.items():
        if v is None: continue
        a = np.asarray(v)
        if a.ndim == 1 or (a.ndim == 2 and 1 in a.shape):
            arr = a.reshape(-1); scalar[k] = arr; n = len(arr) if n is None else min(n, len(arr))
    if not scalar: return
    keys = sorted(scalar)
    with out.open("w") as f:
        f.write("frame," + ",".join(keys) + "\n")
        for i in range(n or 0):
            f.write(str(i) + ''.join(f",{float(scalar[k][i]):.12g}" for k in keys) + "\n")

def plot_scalar(outdir: Path, name: str, arr: np.ndarray) -> None:
    try: import matplotlib.pyplot as plt
    except Exception: return
    a = np.asarray(arr).reshape(-1); fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(np.arange(len(a)), a); ax.set_xlabel("reported frame"); ax.set_ylabel(name); ax.set_title(name); fig.tight_layout(); fig.savefig(outdir / f"{name}.png", dpi=200); plt.close(fig)

def compute_rdf_from_atoms(frames: list[Any], r_max: float, bins: int) -> tuple[np.ndarray, np.ndarray]:
    hist = np.zeros(bins); edges = np.linspace(0.0, r_max, bins + 1); n_frames = 0; rho_acc = 0.0; n_acc = 0
    for atoms in frames:
        n = len(atoms)
        if n < 2: continue
        d = atoms.get_all_distances(mic=True); iu = np.triu_indices(n, k=1); hist += np.histogram(d[iu], bins=edges)[0]
        vol = atoms.get_volume() if all(atoms.pbc) else np.nan
        if np.isfinite(vol) and vol > 0: rho_acc += n / vol; n_acc += n; n_frames += 1
    r = 0.5 * (edges[:-1] + edges[1:]); dr = edges[1] - edges[0]; shell = 4*np.pi*r**2*dr
    if n_frames == 0 or n_acc == 0: return r, hist
    rho = rho_acc / n_frames; n_mean = n_acc / n_frames; ideal = n_frames * n_mean * rho * shell / 2.0
    g = np.divide(hist, ideal, out=np.zeros_like(hist), where=ideal > 0); return r, g

def compute_msd(positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions)
    if pos.ndim == 4: pos = pos[:, 0]
    disp = pos - pos[0]
    return np.mean(np.sum(disp * disp, axis=-1), axis=-1)

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("trajectory"); ap.add_argument("--outdir", default="analysis"); ap.add_argument("--discard-fraction", type=float, default=0.0); ap.add_argument("--rdf", action="store_true"); ap.add_argument("--rdf-rmax", type=float, default=8.0); ap.add_argument("--rdf-bins", type=int, default=200); ap.add_argument("--msd", action="store_true"); ap.add_argument("--xyz", default=None); ap.add_argument("--stride", type=int, default=1); args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    with ts.TorchSimTrajectory(args.trajectory, mode="r") as traj:
        arrays = {name: safe_array(traj, name) for name in list_arrays(traj)}; write_scalar_csv(outdir / "scalars.csv", arrays)
        for name, arr in arrays.items():
            if arr is not None and (arr.ndim == 1 or (arr.ndim == 2 and 1 in arr.shape)): plot_scalar(outdir, name, arr)
        positions = arrays.get("positions"); summary = {"trajectory": args.trajectory, "arrays_found": sorted(arrays), "positions_shape": list(positions.shape) if positions is not None else None}
        if args.xyz:
            try: traj.write_ase_trajectory(args.xyz); summary["xyz_export"] = args.xyz
            except Exception as exc: summary["xyz_export_warning"] = repr(exc)
        if args.rdf and positions is not None:
            frames = []
            n_frames = positions.shape[0]; start = int(n_frames * args.discard_fraction)
            for i in range(start, n_frames, max(1, args.stride)):
                try: frames.append(traj.get_atoms(i))
                except Exception: break
            if frames:
                r, g = compute_rdf_from_atoms(frames, args.rdf_rmax, args.rdf_bins); np.savetxt(outdir / "rdf.csv", np.c_[r, g], delimiter=",", header="r_A,g_r", comments="")
                try:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(r, g); ax.set_xlabel("r (Angstrom)"); ax.set_ylabel("g(r)"); ax.set_title("Radial distribution function"); fig.tight_layout(); fig.savefig(outdir / "rdf.png", dpi=200); plt.close(fig)
                except Exception: pass
                summary["rdf"] = "rdf.csv"
        if args.msd and positions is not None:
            msd = compute_msd(positions); np.savetxt(outdir / "msd.csv", np.c_[np.arange(len(msd)), msd], delimiter=",", header="frame,msd_A2", comments=""); plot_scalar(outdir, "msd_A2", msd); summary["msd"] = "msd.csv"
        try:
            from ase.io import write
            write(outdir / "final.extxyz", traj.get_atoms(-1)); summary["final_structure"] = "final.extxyz"
        except Exception as exc: summary["final_structure_warning"] = repr(exc)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
