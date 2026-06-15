# Environment setup

## Minimal install

Use Python >= 3.12.

```bash
python3.12 -m venv .venv-torchsim
source .venv-torchsim/bin/activate
python -m pip install -U pip wheel setuptools
pip install 'torch-sim-atomistic[io,vesin]'
```

`io` gives ASE/pymatgen/phonopy interoperability. `vesin` is useful for neighbor lists.

## MLIP-specific environments

Create one environment per MLIP family to avoid dependency conflicts:

```bash
# MACE
pip install 'torch-sim-atomistic[io,mace]'

# MatterSim
pip install 'torch-sim-atomistic[io,mattersim]'

# ORB
pip install 'torch-sim-atomistic[io,orb]'

# SevenNet
pip install 'torch-sim-atomistic[io,sevenn]'

# FairChem / UMA / Equiformer-style stacks
pip install 'torch-sim-atomistic[io,fairchem]'

# NequIP / NequIX
pip install 'torch-sim-atomistic[io,nequip]'
pip install 'torch-sim-atomistic[io,nequix]'
```

Do not blindly install all extras together. The upstream project declares conflicts among several MLIP extras.

## GPU sanity check

Run:

```bash
python /home/oai/skills/torchsim/scripts/torchsim_doctor.py
```

Check Python >= 3.12, `torch`, `torch_sim`, CUDA, ASE/pymatgen, and optional MLIP dependency imports.

## Reproducible project skeleton

```text
project/
  configs/
  structures/
  runs/
  analysis/
  movies/
  logs/
  README.md
```

Save a copy of the exact config and script into each `runs/<run_id>/` directory.


Note: Disable autobatcher on CPU pilot runs; TorchSim memory-estimation autobatching is intended for GPU memory management.
