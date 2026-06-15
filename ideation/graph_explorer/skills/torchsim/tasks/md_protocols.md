# MD protocols

## Ensemble choice

- **NVE**: energy conservation tests and microcanonical production.
- **NVT**: thermal equilibration and canonical sampling.
- **NPT**: density/lattice/volume equilibration; use cautiously for slabs/vacuum.

Inspect installed enum names:

```python
import torch_sim as ts
print([x for x in dir(ts.Integrator) if not x.startswith('_')])
```

## Timestep guidance

- 0.001 ps = 1 fs is a conservative starting point for many inorganic MLIP simulations.
- 0.00025-0.0005 ps for H-rich, stiff, high-temperature, shock, or pilot runs.
- 0.002 ps only after stability and energy-drift checks.

## Standard solid workflow

Static check -> relax -> NVT -> optional NPT -> NVE/NVT production -> energy/RDF/MSD/stress/defect analysis.

## Standard liquid/glass workflow

Build dense structure -> remove overlaps -> NVT/NPT equilibrate -> production -> RDF/MSD/diffusion/structure factor.

## Reporting cadence

Save scalar properties more frequently than full coordinates. Save coordinates at a cadence matched to the physical dynamics and movie/analysis needs.
