#!/usr/bin/env python3
"""Convert a TorchSim HDF5 trajectory to XYZ and/or a quick MP4/GIF movie."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ase.io import write
import torch_sim as ts

def get_frames(traj, stride: int, max_frames: int | None):
    positions = traj.get_array("positions"); n = positions.shape[0]
    indices = list(range(0, n, max(1, stride)))
    if max_frames is not None: indices = indices[:max_frames]
    return [traj.get_atoms(i) for i in indices], indices

def axis_limits(frames):
    xyz = np.concatenate([a.get_positions() for a in frames], axis=0); lo = xyz.min(axis=0); hi = xyz.max(axis=0); center = 0.5*(lo+hi); span = max(float((hi-lo).max()), 1.0); pad = 0.1*span; return center-span/2-pad, center+span/2+pad

def render_movie(frames, output: str, fps: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    lo, hi = axis_limits(frames); fig = plt.figure(figsize=(7,7)); ax = fig.add_subplot(111, projection="3d")
    def update(i):
        ax.clear(); atoms = frames[i]; pos = atoms.get_positions(); z = atoms.get_atomic_numbers(); sizes = 20 + 2*(z/max(z.max(),1)); ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=sizes, c=z, cmap="viridis"); ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2]); ax.set_xlabel("x (Angstrom)"); ax.set_ylabel("y (Angstrom)"); ax.set_zlabel("z (Angstrom)"); ax.set_title(f"Frame {i+1}/{len(frames)}")
        try:
            cell = atoms.get_cell().array; origin = np.zeros(3); corners = [origin, cell[0], cell[1], cell[2], cell[0]+cell[1], cell[0]+cell[2], cell[1]+cell[2], cell[0]+cell[1]+cell[2]]; edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
            for a,b in edges:
                p,q = corners[a], corners[b]; ax.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]], linewidth=0.8)
        except Exception: pass
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps); out = Path(output)
    if out.suffix.lower() == ".gif": anim.save(output, writer=PillowWriter(fps=fps))
    else: anim.save(output, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("trajectory"); ap.add_argument("--xyz"); ap.add_argument("--mp4"); ap.add_argument("--gif"); ap.add_argument("--stride", type=int, default=1); ap.add_argument("--fps", type=int, default=30); ap.add_argument("--max-frames", type=int); args = ap.parse_args()
    with ts.TorchSimTrajectory(args.trajectory, mode="r") as traj: frames, _ = get_frames(traj, args.stride, args.max_frames)
    if args.xyz: write(args.xyz, frames); print(f"Wrote {args.xyz} with {len(frames)} frames")
    if args.mp4: render_movie(frames, args.mp4, args.fps); print(f"Wrote {args.mp4}")
    if args.gif: render_movie(frames, args.gif, args.fps); print(f"Wrote {args.gif}")
if __name__ == "__main__": main()
