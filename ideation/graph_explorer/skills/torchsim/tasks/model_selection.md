# Model selection

| Material/task | Default candidate | Notes |
|---|---|---|
| Quick code test | Lennard-Jones / Morse / soft-sphere | Not physically predictive unless parametrized. |
| Inorganic crystals | MACE-MP, MatterSim, ORB, SevenNet, FairChem | Validate elements, oxidation states, phase, stress, high-T behavior. |
| Surfaces/interfaces | MLIP with surface/interface coverage | Check undercoordinated atoms. |
| Defects/dislocations/fracture | MLIP with defect/high-strain data | Pilot runs mandatory. |
| Liquids/melts/glasses | MLIP with liquid/high-T coverage | Check density, RDF, diffusion vs reference. |
| Polymers/biomaterials | Chemistry-specific MLIP or classical FF outside TorchSim | Generic inorganic MLIPs may be invalid. |
| Ionic/charged/polar systems | Model with electrostatics/charge physics | Long-range treatment matters. |
| Thermal conductivity | Model with stable energy conservation and heat flux support | Validate NVE drift and flux definitions. |
| Elastic constants | Model with reliable stress/forces | Relax cell first; check symmetry. |

Practical rules:

- Use `float32` for long MD if the model recommends it and stability is acceptable.
- Use `float64` for relaxation/static/elastic calculations when supported.
- Record model name, version, checkpoint, cutoff, dtype, device, and source.
- For high-stakes predictions, compare a subset against DFT, experiment, or a trusted potential.

Minimal LJ:

```python
import torch
from torch_sim.models.lennard_jones import LennardJonesModel
model = LennardJonesModel(sigma=2.0, epsilon=0.1, device=torch.device('cpu'), dtype=torch.float64)
```

MACE:

```python
import torch
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel
raw = mace_mp(model='small', return_raw_model=True)
model = MaceModel(model=raw, device=torch.device('cuda'), dtype=torch.float32, compute_forces=True)
```
