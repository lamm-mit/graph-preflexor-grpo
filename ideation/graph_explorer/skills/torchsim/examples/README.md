# Example commands

```bash
python /home/oai/skills/torchsim/scripts/torchsim_doctor.py
mkdir -p work
cp /home/oai/skills/torchsim/templates/config.yaml work/config.yaml
python /home/oai/skills/torchsim/templates/run_md.py work/config.yaml
python /home/oai/skills/torchsim/templates/analyze_trajectory.py runs/demo_cu_nvt/trajectory.h5 --outdir runs/demo_cu_nvt/analysis --rdf --msd --xyz runs/demo_cu_nvt/trajectory.xyz
python /home/oai/skills/torchsim/templates/make_movie.py runs/demo_cu_nvt/trajectory.h5 --mp4 runs/demo_cu_nvt/movie.mp4 --xyz runs/demo_cu_nvt/trajectory.xyz --stride 5 --fps 30
```
