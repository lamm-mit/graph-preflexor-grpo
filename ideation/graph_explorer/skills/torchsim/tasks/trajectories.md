# Trajectory reporting and export

TorchSim trajectories use HDF5 through `TorchSimTrajectory`. `TrajectoryReporter` handles regular state/property reporting.

```python
import torch_sim as ts
reporter = ts.TrajectoryReporter(
    filenames='trajectory.h5',
    state_frequency=100,
    prop_calculators={10: {'potential_energy': lambda state: state.energy}},
)
```

Pass `trajectory_reporter=reporter` into `ts.integrate(...)` or `ts.optimize(...)`.

For batched systems, use one filename per system:

```python
filenames = [f'traj_{i}.h5' for i in range(len(systems))]
reporter = ts.TrajectoryReporter(filenames, state_frequency=100)
```

Reading:

```python
with ts.TorchSimTrajectory('trajectory.h5', mode='r') as traj:
    positions = traj.get_array('positions')
    atoms = traj.get_atoms(-1)
    state = traj.get_state(-1)
    traj.write_ase_trajectory('trajectory.traj')
```

Recommended outputs: `trajectory.h5`, `trajectory.extxyz`, `analysis.csv`, `summary.json`, optional `movie.mp4`.
