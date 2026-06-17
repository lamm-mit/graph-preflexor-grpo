#!/usr/bin/env python3
"""Generate a compact driven damped harmonic oscillator dataset.

The model is the standard mass-spring-damper equation

    m x'' + c x' + k x = F0 cos(omega_d t)

integrated with fourth-order Runge-Kutta. The output is intentionally small so
it is a quick test case for the d3-viz skill.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


OUT = Path(__file__).with_name("damped_driven_oscillator.json")


def acceleration(
    x: float,
    v: float,
    t: float,
    *,
    mass: float,
    damping: float,
    spring_constant: float,
    drive_amplitude: float,
    drive_omega: float,
) -> float:
    force_drive = drive_amplitude * math.cos(drive_omega * t)
    force_spring = -spring_constant * x
    force_damping = -damping * v
    return (force_drive + force_spring + force_damping) / mass


def rk4_step(x: float, v: float, t: float, dt: float, params: dict[str, float]) -> tuple[float, float]:
    def deriv(state_x: float, state_v: float, state_t: float) -> tuple[float, float]:
        return state_v, acceleration(state_x, state_v, state_t, **params)

    k1x, k1v = deriv(x, v, t)
    k2x, k2v = deriv(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, t + 0.5 * dt)
    k3x, k3v = deriv(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, t + 0.5 * dt)
    k4x, k4v = deriv(x + dt * k3x, v + dt * k3v, t + dt)

    next_x = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    next_v = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return next_x, next_v


def simulate() -> dict:
    params = {
        "mass": 1.0,
        "damping": 0.12,
        "spring_constant": 1.0,
        "drive_amplitude": 0.18,
        "drive_omega": 0.82,
    }
    dt = 0.02
    steps = 3600
    sample_every = 30
    x = 0.42
    v = 0.0

    natural_omega = math.sqrt(params["spring_constant"] / params["mass"])
    damping_ratio = params["damping"] / (
        2.0 * math.sqrt(params["spring_constant"] * params["mass"])
    )
    damped_omega = natural_omega * math.sqrt(max(0.0, 1.0 - damping_ratio**2))

    samples = []
    for step in range(steps + 1):
        t = step * dt
        if step % sample_every == 0:
            drive_force = params["drive_amplitude"] * math.cos(params["drive_omega"] * t)
            kinetic = 0.5 * params["mass"] * v * v
            spring = 0.5 * params["spring_constant"] * x * x
            samples.append(
                {
                    "step": step,
                    "time": round(t, 6),
                    "position": round(x, 8),
                    "velocity": round(v, 8),
                    "drive_force": round(drive_force, 8),
                    "kinetic_energy": round(kinetic, 10),
                    "spring_energy": round(spring, 10),
                    "mechanical_energy": round(kinetic + spring, 10),
                }
            )
        x, v = rk4_step(x, v, t, dt, params)

    zero_crossings = []
    for prev, cur in zip(samples, samples[1:]):
        if prev["position"] <= 0.0 < cur["position"] or cur["position"] <= 0.0 < prev["position"]:
            zero_crossings.append(round(0.5 * (prev["time"] + cur["time"]), 6))

    return {
        "title": "Driven Damped Harmonic Oscillator",
        "description": (
            "Compact real-physics simulation of a single mass on a spring with "
            "linear damping and sinusoidal forcing. The system transitions from "
            "free decay to a driven steady state near resonance."
        ),
        "model": {
            "name": "mass-spring-damper with sinusoidal drive",
            "equation": "m*x'' + c*x' + k*x = F0*cos(omega_d*t)",
            "integrator": "fourth-order Runge-Kutta",
            "dt": dt,
            "steps": steps,
            "sample_every": sample_every,
            "initial_position": 0.42,
            "initial_velocity": 0.0,
            "natural_omega": round(natural_omega, 8),
            "damped_omega": round(damped_omega, 8),
            "damping_ratio": round(damping_ratio, 8),
            **params,
        },
        "summary": {
            "duration": samples[-1]["time"],
            "n_samples": len(samples),
            "max_abs_position": round(max(abs(item["position"]) for item in samples), 8),
            "max_abs_velocity": round(max(abs(item["velocity"]) for item in samples), 8),
            "initial_mechanical_energy": samples[0]["mechanical_energy"],
            "final_mechanical_energy": samples[-1]["mechanical_energy"],
            "first_zero_crossings": zero_crossings[:8],
        },
        "time_series": samples,
        "phase_space": [
            {
                "time": item["time"],
                "position": item["position"],
                "velocity": item["velocity"],
            }
            for item in samples
        ],
        "visualization_suggestions": [
            "animated mass-spring displacement",
            "position, velocity, and drive force time-series panel",
            "phase-space trajectory colored by time",
            "kinetic, spring, and total mechanical energy chart",
            "parameter callout for natural frequency, drive frequency, and damping ratio",
        ],
    }


def main() -> None:
    data = simulate()
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
