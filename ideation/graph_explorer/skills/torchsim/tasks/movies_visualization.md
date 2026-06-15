# Movies and visualization

Use `templates/make_movie.py` to convert TorchSim HDF5 trajectories into MP4/GIF and XYZ/extxyz outputs.

```bash
python /home/oai/skills/torchsim/templates/make_movie.py trajectory.h5 --xyz trajectory.xyz --stride 10
python /home/oai/skills/torchsim/templates/make_movie.py trajectory.h5 --mp4 movie.mp4 --stride 10 --fps 30
```

For publication-grade movies, load `.xyz`/`.extxyz` into OVITO, VMD, Blender, or ASE GUI.

Movie rules:

- Use wrapped coordinates for periodic visualization unless showing diffusion.
- Include simulation cell for crystals/slabs/interfaces.
- Record material, model, ensemble, temperature, timestep, frame stride, and real time represented.
- For fracture/deformation movies, color by displacement, velocity, strain, coordination, or force if available.
