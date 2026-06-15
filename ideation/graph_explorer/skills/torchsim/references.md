# TorchSim references captured for this skill

- Source repo: https://github.com/torchsim/torch-sim
- Docs: https://torchsim.github.io/torch-sim/
- Package: `torch-sim-atomistic`
- Import: `torch_sim`
- Core docs pages checked while preparing this skill:
  - User guide / core concepts
  - High-level tutorial
  - Reporting tutorial
  - API reference for models, integrators, quantities, trajectories, elastic module
  - Pyproject dependency list

Core API facts to verify if updating this skill:

```python
import torch_sim as ts

ts.integrate(...)
ts.optimize(...)
ts.static(...)
ts.TrajectoryReporter(...)
ts.TorchSimTrajectory(...)
ts.Integrator.nve
ts.Integrator.nvt_langevin
ts.Integrator.npt_langevin_isotropic  # exact enum variants may evolve; inspect dir(ts.Integrator)
ts.Optimizer.fire
```

When in doubt, inspect the installed package rather than guessing:

```python
import torch_sim as ts
print(ts.__version__ if hasattr(ts, '__version__') else 'no __version__')
print([x for x in dir(ts.Integrator) if not x.startswith('_')])
print([x for x in dir(ts.Optimizer) if not x.startswith('_')])
```
