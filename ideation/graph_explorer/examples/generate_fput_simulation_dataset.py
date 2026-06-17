#!/usr/bin/env python3
"""Generate a compact FPUT nonlinear oscillator-chain dataset.

The Fermi-Pasta-Ulam-Tsingou model is a standard nonlinear lattice simulation:
point masses are coupled by springs with a weak nonlinear force term. This
script uses velocity-Verlet integration and writes a JSON dataset intended for
the d3-viz skill.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


OUT = Path(__file__).with_name("fput_nonlinear_chain_simulation.json")


def forces(x: list[float], alpha: float) -> list[float]:
    """Return forces for fixed-end alpha-FPUT chain."""
    n = len(x)
    padded = [0.0] + x + [0.0]
    f = []
    for i in range(1, n + 1):
        left = padded[i] - padded[i - 1]
        right = padded[i + 1] - padded[i]
        force = right - left + alpha * (right * right - left * left)
        f.append(force)
    return f


def potential_energy(x: list[float], alpha: float) -> float:
    padded = [0.0] + x + [0.0]
    energy = 0.0
    for i in range(len(padded) - 1):
        delta = padded[i + 1] - padded[i]
        energy += 0.5 * delta * delta + (alpha / 3.0) * delta**3
    return energy


def kinetic_energy(v: list[float]) -> float:
    return 0.5 * sum(value * value for value in v)


def mode_amplitudes(x: list[float], v: list[float]) -> list[dict[str, float]]:
    """Estimate linear normal-mode energies for fixed-end chain."""
    n = len(x)
    modes = []
    norm = math.sqrt(2.0 / (n + 1))
    for k in range(1, n + 1):
        q = 0.0
        p = 0.0
        for j in range(1, n + 1):
            basis = norm * math.sin(math.pi * k * j / (n + 1))
            q += x[j - 1] * basis
            p += v[j - 1] * basis
        omega = 2.0 * math.sin(math.pi * k / (2.0 * (n + 1)))
        energy = 0.5 * (p * p + omega * omega * q * q)
        modes.append({"mode": k, "amplitude": q, "velocity": p, "energy": energy})
    return modes


def simulate() -> dict:
    n = 12
    alpha = 0.25
    dt = 0.05
    steps = 2400
    sample_every = 20

    # Excite mostly the first normal mode with a small localized perturbation.
    x = [
        0.16 * math.sin(math.pi * (i + 1) / (n + 1))
        + (0.012 if i == n // 2 else 0.0)
        for i in range(n)
    ]
    v = [0.0 for _ in range(n)]
    f = forces(x, alpha)
    e0 = kinetic_energy(v) + potential_energy(x, alpha)

    snapshots = []
    energy_trace = []
    mode_trace = []

    for step in range(steps + 1):
        if step % sample_every == 0:
            time = round(step * dt, 6)
            ke = kinetic_energy(v)
            pe = potential_energy(x, alpha)
            total = ke + pe
            modes = mode_amplitudes(x, v)
            snapshots.append(
                {
                    "step": step,
                    "time": time,
                    "positions": [round(value, 8) for value in x],
                    "velocities": [round(value, 8) for value in v],
                }
            )
            energy_trace.append(
                {
                    "step": step,
                    "time": time,
                    "kinetic": round(ke, 10),
                    "potential": round(pe, 10),
                    "total": round(total, 10),
                    "relative_drift": round((total - e0) / e0, 12),
                }
            )
            mode_trace.append(
                {
                    "step": step,
                    "time": time,
                    "modes": [
                        {
                            "mode": item["mode"],
                            "energy": round(item["energy"], 10),
                        }
                        for item in modes[:8]
                    ],
                }
            )

        # Velocity-Verlet update.
        for i in range(n):
            x[i] += v[i] * dt + 0.5 * f[i] * dt * dt
        f_new = forces(x, alpha)
        for i in range(n):
            v[i] += 0.5 * (f[i] + f_new[i]) * dt
        f = f_new

    nodes = [
        {
            "id": f"mass-{i + 1}",
            "label": f"Mass {i + 1}",
            "index": i + 1,
            "type": "oscillator",
            "initial_position": round(snapshots[0]["positions"][i], 8),
            "max_abs_displacement": round(
                max(abs(snapshot["positions"][i]) for snapshot in snapshots), 8
            ),
        }
        for i in range(n)
    ]
    links = [
        {
            "source": f"mass-{i + 1}",
            "target": f"mass-{i + 2}",
            "type": "nonlinear_spring",
            "linear_k": 1.0,
            "alpha": alpha,
        }
        for i in range(n - 1)
    ]

    return {
        "title": "FPUT Nonlinear Chain Simulation",
        "description": (
            "Velocity-Verlet simulation of a fixed-end alpha-FPUT nonlinear "
            "oscillator chain. Useful for D3 visualizations of wave propagation, "
            "mode-energy transfer, phase-space traces, and energy conservation."
        ),
        "model": {
            "name": "alpha-FPUT chain",
            "equation": "x_i'' = (x_{i+1} - 2 x_i + x_{i-1}) + alpha[(x_{i+1}-x_i)^2 - (x_i-x_{i-1})^2]",
            "boundary_conditions": "fixed ends, x_0 = x_{N+1} = 0",
            "integrator": "velocity-Verlet",
            "n_masses": n,
            "alpha": alpha,
            "dt": dt,
            "steps": steps,
            "sample_every": sample_every,
            "initial_total_energy": round(e0, 10),
        },
        "nodes": nodes,
        "links": links,
        "snapshots": snapshots,
        "energy_trace": energy_trace,
        "mode_energy_trace": mode_trace,
        "visualization_suggestions": [
            "animated chain displacement",
            "time-position displacement heatmap",
            "normal-mode energy transfer stacked area plot",
            "energy conservation line chart",
            "phase-space plot for selected masses",
        ],
    }


def main() -> None:
    data = simulate()
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
