# Relaxation and static evaluation

Use `ts.optimize(...)` for geometry optimization and `ts.static(...)` for one-shot evaluation.

Rules:

- Relax positions before serious MD.
- Relax cell for bulk crystals/dense phases when stress/cell forces are valid.
- Do not freely relax vacuum direction for slabs unless intended.
- Monitor max force, energy, and final structure sanity.

Inspect optimizer names:

```python
import torch_sim as ts
print([x for x in dir(ts.Optimizer) if not x.startswith('_')])
```

Static precheck: finite energy, finite forces, reasonable max force, nonzero cell volume, correct atomic numbers/PBC.
