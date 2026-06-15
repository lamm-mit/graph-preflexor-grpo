# TorchSim MD Skill Directory

This directory is a reusable skill scaffold for creating, running, analyzing, and visualizing TorchSim atomistic simulations for materials.

The skill assumes the active TorchSim project at `https://github.com/torchsim/torch-sim`:

```bash
pip install torch-sim-atomistic
python -c "import torch_sim as ts; print(ts)"
```

For realistic workflows install only the extras you need. Examples:

```bash
pip install 'torch-sim-atomistic[io,vesin]'
pip install 'torch-sim-atomistic[io,mace]'
pip install 'torch-sim-atomistic[io,mattersim]'
pip install 'torch-sim-atomistic[io,orb]'
pip install 'torch-sim-atomistic[io,sevenn]'
pip install 'torch-sim-atomistic[io,fairchem]'
```

Start with `SKILL.md`, then route to `tasks/` and `templates/`.

## Typical deliverables for a user task

- `config.yaml`: protocol and model parameters.
- `run_md.py` or custom script.
- `trajectory.h5` / `trajectory.h5md`: TorchSim trajectory.
- `trajectory.extxyz` or `.xyz`: visualization/export trajectory.
- `analysis.csv`, `summary.json`: scalar results.
- `energy.png`, `temperature.png`, `rdf.png`, `msd.png`: figures.
- `movie.mp4` or `movie.gif`: visualization.
- `README.md`: exact run commands.
