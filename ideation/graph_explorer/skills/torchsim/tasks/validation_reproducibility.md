# Validation and reproducibility

For every run check: NaN/Inf, target temperature, NVE energy drift, NPT volume stability, atom overlaps, final structure, trajectory frame count.

```bash
python /home/oai/skills/torchsim/templates/validate_run.py trajectory.h5 --summary validation.json
```

Serious ensemble validation: Maxwell-Boltzmann kinetic energy distribution, ensemble consistency across T/P, NVE drift scaling with timestep, repeated seeds.

Capture Python/Torch/TorchSim/CUDA versions, package freeze, model checkpoint, config YAML, scripts, random seed, and input structures.

Determinism caveat: GPU MD, stochastic thermostats, and parallel reductions can vary run-to-run. Reproducibility usually means statistically reproducible observables, not identical trajectories.
