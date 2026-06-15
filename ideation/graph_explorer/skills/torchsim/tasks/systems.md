# Building and importing systems

Prefer ASE-compatible formats: CIF, POSCAR/CONTCAR, XYZ/extxyz, PDB, trajectory frames, or generated ASE objects.

```python
from ase.io import read, write
atoms = read('structure.cif')
atoms.pbc = True
write('initial.extxyz', atoms)
```

## Bulk crystals

Build conventional or primitive cells with ASE/pymatgen, repeat to a sufficiently large supercell, relax positions and usually cell before production.

```python
from ase.build import bulk
atoms = bulk('Cu', 'fcc', a=3.615, cubic=True).repeat((5, 5, 5))
```

## Surfaces/slabs

Use vacuum in surface normal, lateral PBC, and avoid relaxing vacuum dimension unless intentional. Consider fixing bottom layers for substrates.

## Interfaces

Minimize lattice mismatch, use enough thickness, relax carefully, and analyze mixing, adhesion, RDF, stress, displacement fields.

## Defects

Use large supercells, relax first, track local coordination and defect position over time.

## Liquids/glasses

Start from reasonable density, equilibrate with NVT/NPT, analyze RDF, coordination, MSD/diffusion.

## Polymers/biomaterials

TorchSim can carry atomistic states, but validated force-field MD in OpenMM/GROMACS may be better unless a suitable MLIP exists. Analyze radius of gyration, end-to-end distance, orientation, contacts, bond/angle distributions.
