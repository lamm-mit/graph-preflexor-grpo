# Batching, sweeps, and HPC/GPU usage

Batch when running many independent systems, composition sweeps, T/P sweeps, defect configurations, replicas, or active-learning candidate evaluations.

```python
systems = [atoms0, atoms1, atoms2]
filenames = [f'run_{i}.h5' for i in range(len(systems))]
final_state = ts.integrate(
    system=systems,
    model=model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=1000,
    timestep=0.001,
    temperature=300,
    trajectory_reporter={'filenames': filenames, 'state_frequency': 100},
    autobatcher=True,
)
```

Start with small batches, scale up, save one trajectory per system, monitor memory and physical stability.


Note: Disable autobatcher on CPU pilot runs; TorchSim memory-estimation autobatching is intended for GPU memory management.
