---
name: torchsim
description: Comprehensive TorchSim / torch-sim atomistic simulation skill for molecular dynamics, relaxation, high-throughput MLIP workflows, trajectory generation, trajectory analysis, validation, and movie production for materials systems.
---

# TorchSim Atomistic Simulation Skill

Use this skill when the user asks for atomistic molecular dynamics (MD), relaxation, static property evaluation, trajectory analysis, or visualization using **TorchSim / `torch-sim`**.

This skill targets the active atomistic simulation engine at `https://github.com/torchsim/torch-sim`, installed as `torch-sim-atomistic` and imported as `torch_sim`. Do **not** confuse it with the unrelated PyPI package `torchsim` for MRI simulation or the older GoodAI TorchSim project.

## Core facts to preserve

- Package name: `torch-sim-atomistic`.
- Python import: `import torch_sim as ts`.
- Python requirement: Python >= 3.12.
- Core runners:
  - `ts.integrate(...)` for MD.
  - `ts.optimize(...)` for relaxation.
  - `ts.static(...)` for one-shot energy / force / stress-style evaluations.
- Core state object: `ts.SimState`, with tensor-native batched state.
- Trajectory writing/reading:
  - `ts.TrajectoryReporter(...)` for regular state/property reporting.
  - `ts.TorchSimTrajectory(...)` for HDF5 trajectory IO and analysis.
- Standard unit convention follows TorchSim's metal-style units: energy eV, length Angstrom, time ps, temperature K, pressure bar-style quantities unless explicitly transformed.
- Supported workflows include batched simulation, automatic GPU memory management/autobatching, ASE/pymatgen/phonopy interoperability, classical potentials, MLIPs, structural relaxation, elastic properties, MC/swap workflows, and differentiable simulations.

## Skill contract

When using this skill for a user task, produce a complete, reproducible workflow. Do not only give prose. Depending on the request, create or provide:

1. A clear simulation protocol: model, system, ensemble, timestep, equilibration, production length, reporting cadence, and validation checks.
2. Runnable Python scripts or notebooks when appropriate.
3. Trajectory outputs in TorchSim HDF5 plus optional ASE `.traj`, `.xyz`, `.extxyz`, or `.pdb` exports.
4. Analysis outputs: CSV/JSON summaries, figures, and physical validation checks.
5. Movie/visualization outputs when requested: MP4/GIF plus raw frames or an OVITO/VMD/ASE-compatible trajectory.
6. A README explaining exact run commands and expected outputs.

## Default workflow for MD tasks

Follow this order unless the user has a reason not to:

1. **System construction/import**
   - Read existing structures from CIF/POSCAR/XYZ/PDB/extxyz/traj if provided.
   - Otherwise construct with ASE/pymatgen: bulk crystals, surfaces/slabs, defects, interfaces, nanoparticles, liquids, polymers, or amorphous cells.
   - Make PBC explicit. Avoid accidental nonperiodic boxes.

2. **Model selection**
   - For realistic materials simulations, prefer a validated MLIP for the chemistry/conditions.
   - For demos/tests, use classical potentials such as Lennard-Jones, Morse, or soft-sphere.
   - Make model validity explicit: supported elements, pressure/temperature range, phase, charge/polarity, surfaces/defects/out-of-domain risks.

3. **Relaxation / static precheck**
   - Relax atomic positions before MD for solids and interfaces.
   - Relax cell only when physically appropriate and model supports stress/cell forces.
   - Run static energy/force sanity checks before a long MD.

4. **Equilibration**
   - Use NVT for thermal equilibration when volume is known.
   - Use NPT when density/lattice parameters should relax.
   - Choose timestep conservatively: typical MLIP solid/liquid MD often uses 0.5-2 fs = 0.0005-0.002 ps; reduce for H-rich systems, high T, reactive events, or stiff bonds.

5. **Production**
   - Save full state at a cadence that balances temporal resolution and file size.
   - Save scalar properties more frequently than full coordinates when needed.
   - Use separate trajectory files for batched systems.

6. **Validation**
   - Check energy drift for NVE.
   - Check target temperature in NVT.
   - Check pressure/volume stability in NPT.
   - Check force maxima, atom overlap, box collapse, model warnings, NaNs, and file completeness.

7. **Analysis**
   - Always include basic thermodynamics: potential energy, kinetic energy, total energy, temperature, pressure/volume if available.
   - Add material-specific analysis: RDF, MSD/diffusion, VACF, elastic tensor, stress-strain, heat flux/thermal conductivity, coordination, defects, surface/interface metrics, cluster statistics, polymer conformations, bond/angle/dihedral distributions.

8. **Visualization**
   - Export trajectories to standard formats for OVITO/VMD/ASE.
   - Make movies with cell axes/PBC wrapping awareness.
   - Include frame stride and atom rendering choices in the README.

## Routing

Consult the task files for deeper workflows:

- `tasks/setup_environment.md`: installation, dependency isolation, extras, GPU checks.
- `tasks/model_selection.md`: MLIP and classical potential choice.
- `tasks/systems.md`: bulk, liquids, surfaces, defects, interfaces, polymers, biomaterials.
- `tasks/md_protocols.md`: NVE/NVT/NPT protocols and ensembles.
- `tasks/relaxation_static.md`: geometry relaxation and static evaluation.
- `tasks/trajectories.md`: HDF5 trajectory reporting and export.
- `tasks/analysis.md`: trajectory and materials-property analysis.
- `tasks/movies_visualization.md`: MP4/GIF/XYZ/extxyz/OVITO/VMD visualization.
- `tasks/validation_reproducibility.md`: physical validation and reproducibility.
- `tasks/batching_hpc.md`: autobatching, large sweeps, GPU memory.
- `tasks/troubleshooting.md`: common errors.

## Ready templates

Use files in `templates/` as starting points:

- `templates/config.yaml`: canonical configuration schema.
- `templates/run_md.py`: production MD runner with trajectory reporting.
- `templates/relax_static.py`: relaxation and static precheck runner.
- `templates/batch_sweep.py`: high-throughput batched MD sweeps.
- `templates/analyze_trajectory.py`: energy/RDF/MSD/export analysis.
- `templates/make_movie.py`: HDF5 trajectory to XYZ/MP4/GIF.
- `templates/validate_run.py`: basic run-quality checks.

Utility scripts:

- `scripts/torchsim_doctor.py`: environment and GPU sanity check.
- `scripts/trajectory_summary.py`: inspect TorchSim HDF5 trajectory contents.

## Important implementation cautions

- Many MLIP extras have incompatible dependency stacks, especially around `e3nn`. Use one clean environment per MLIP family when possible.
- Do not install every optional extra in one environment unless the user explicitly wants a compatibility matrix.
- MACE/fairchem/sevenn/mattersim/orb/nequip/nequix may require model downloads, CUDA-specific builds, or license/model-card review. State these assumptions.
- For publication-grade MD, do not treat a generic foundation model as validated for arbitrary chemistry. Include an explicit model-domain warning.
- For biological polymers/proteins, TorchSim atomistic MLIPs may not replace force-field MD in explicit solvent unless a suitable model and long-range electrostatics treatment are validated.
- For charged/ionic/polar systems, check whether the chosen model includes electrostatics/charge physics appropriate to the task.
- For high-temperature, fracture, chemical reaction, defect migration, surfaces, or interfaces, run small pilot simulations first and inspect structural stability before long production.


Note: Disable autobatcher on CPU pilot runs; TorchSim memory-estimation autobatching is intended for GPU memory management.
