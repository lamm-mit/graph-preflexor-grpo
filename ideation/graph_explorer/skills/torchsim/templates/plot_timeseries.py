#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--outdir", default="plots"); args = ap.parse_args()
    import matplotlib.pyplot as plt
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True); data = np.genfromtxt(args.csv, delimiter=",", names=True); names = data.dtype.names or []; x = data["frame"] if "frame" in names else np.arange(len(data))
    for name in names:
        if name == "frame": continue
        fig = plt.figure(); ax = fig.add_subplot(111); ax.plot(x, data[name]); ax.set_xlabel("frame"); ax.set_ylabel(name); ax.set_title(name); fig.tight_layout(); fig.savefig(outdir/f"{name}.png", dpi=200); plt.close(fig)
if __name__ == "__main__": main()
