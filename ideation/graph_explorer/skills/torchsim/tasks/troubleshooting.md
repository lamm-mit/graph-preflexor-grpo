# Troubleshooting

Correct import:

```python
import torch_sim as ts
```

Install:

```bash
pip install torch-sim-atomistic
```

Common fixes:

- Python version error: use Python >= 3.12.
- Optional dependency conflict: one env per MLIP family.
- CUDA false: install correct CUDA-enabled PyTorch and verify driver.
- Missing trajectory arrays: add property calculators to `TrajectoryReporter`.
- MD blows up: reduce timestep, relax first, lower temperature, check overlaps, check model validity, check timestep units in ps.
- NPT cell collapse: NVT first, reduce timestep/barostat strength, check pressure convention, avoid isotropic NPT for slabs.


Note: Disable autobatcher on CPU pilot runs; TorchSim memory-estimation autobatching is intended for GPU memory management.
