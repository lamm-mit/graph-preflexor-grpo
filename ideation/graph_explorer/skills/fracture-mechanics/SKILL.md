---
name: fracture-mechanics
description: Run parameterized 2D lattice fracture simulations from the Fracture and Mechanics notebook as offline artifacts. Use when asked to simulate a pre-cracked triangular lattice, Mode I tension, Mode II shear, Morse/Lennard-Jones/MLIP pair potentials, temperature effects, strain-rate effects, crack propagation, stress-strain curves, or fracture movies. Produces an animated movie, stress-strain plot, final lattice image, CSV data, JSON run metrics, and README by running a bundled Python script.
license: 3-clause BSD license
metadata: {"version": "1.1", "skill-author": "Markus J. Buehler, Massachusetts Institute of Technology (MIT), Laboratory for Atomistic and Molecular Mechanics (LAMM)"}
---

# Fracture Mechanics

Run a 2D triangular-lattice fracture simulation with a sharp edge pre-crack,
moving grip boundaries, and a vectorized bond network with permanent
stretch-based bond breaking. Always use the bundled script; do not rewrite the
simulation from scratch.

## Required Workflow

1. Create or use an output directory, usually `skill_output_<timestamp>_fracture-mechanics`.
2. Run `scripts/run_fracture_sim.py` with either the defaults or the user's selected parameters.
   Use `--steps 10000` for final fracture-producing runs unless the user explicitly asks
   for a quick smoke test.
3. Use the default high-quality movie settings, `--frames 48 --dpi 120 --movie-fps 16`,
   unless a server-side timeout requires the fallback `--frames 16 --dpi 90`.
   The renderer uses fixed slab bounds, fixed stress-strain axes, and a fixed color scale
   across frames to prevent movie flicker.
4. Run the simulation as its own shell command. Do not chain `find` or
   `cat summary.json` after the long simulation command in the same shell call.
5. Verify the artifacts with `find` in a separate shell call after the simulation succeeds.
6. Final answer must list exact artifact paths and mention the peak stress/strain from `summary.json`.

Default command:

```bash
python3 skills/fracture-mechanics/scripts/run_fracture_sim.py \
  --out skill_output_<timestamp>_fracture-mechanics \
  --title "2D Lattice Fracture" \
  --potential morse \
  --mode I \
  --orientation 90 \
  --nx 96 \
  --ny 36 \
  --max-atoms 5000 \
  --crack-length 0.30 \
  --temperature 0.004 \
  --strain-rate 0.003 \
  --damping 0.30 \
  --dt 0.005 \
  --steps 10000 \
  --frames 48 \
  --dpi 120 \
  --movie-fps 16 \
  --bond-cutoff 1.35 \
  --break-stretch 1.75 \
  --color-by stress
```

Verify:

```bash
find skill_output_<timestamp>_fracture-mechanics -maxdepth 2 -type f -print
```

## User Parameters

Use these options when the user asks for specific settings:

- `--potential`: `morse`, `lj`, or `mlip`.
- `--mode`: `I` for tension/opening, `II` for shear.
- `--nx`, `--ny`: lattice dimensions. Defaults are 96 x 36. Larger systems are clamped by `--max-atoms`.
- `--max-atoms`: safety clamp for local runs. Default 5000, maximum 40000.
- `--orientation`: `0` or `90`; `90` is default.
- `--crack-length`: edge crack length as a fraction of specimen width, 0.0-0.80.
- `--temperature`: initial kinetic temperature scale.
- `--strain-rate`: grip loading rate.
- `--damping`: viscous damping. Use `0` for near-NVE after loading; use 0.1-0.5 for stable visual demos.
- `--dt`: timestep.
- `--steps`: number of velocity-Verlet steps. Use 10000 for final fracture-producing runs.
- `--frames`: number of movie frames. Default 48 for a smoother GIF; use 16 only as a timeout fallback.
- `--seed`: deterministic random seed.
- `--a0`: `auto` or an explicit lattice spacing.
- `--morse-a`: Morse stiffness/brittleness. Lower values are tougher; higher values are more brittle.
- `--morse-De`, `--morse-re`: Morse energy and equilibrium distance.
- `--lj-eps`, `--lj-sigma`: Lennard-Jones parameters.
- `--mlip-weights`: optional JSON file with `w`, `b_in`, `c`, and `b_out` for a neural pair potential.
- `--bond-cutoff`: initial bonded-neighbor cutoff in units of `a0`; default 1.35.
- `--break-stretch`: permanently break bonds stretched past this multiple of `a0`; default 1.75. Use lower values for more brittle movies and higher values for tougher response.
- `--color-by`: `stress`, `pe`, `coordination`, `ke`, or `speed`.
- `--movie-fps`: GIF/HTML playback speed. Default 16.
- `--dpi`: figure and movie-frame resolution. Default 120; use 90 only as a timeout fallback.

## Local Shell Timeout Policy

The local mistral.rs shell executor may stop a single shell call after about 30
seconds. The CLI `--response-timeout` controls the HTTP response wait, not the
per-shell-call runtime. Avoid sending a long simulation plus verification
commands in one shell call, because the timeout can prevent `find` and
`summary.json` inspection from running.

Local-safe final artifact profile:

```text
nx <= 96
ny <= 36
max-atoms <= 5000
steps = 10000
frames = 48
dpi = 120
```

Timeout fallback profile:

```text
nx <= 96
ny <= 36
max-atoms <= 5000
steps = 10000
frames = 16
dpi = 90
```

If the user requests a larger exact run, such as `steps 30000`, `nx 128`,
`ny 48`, or `max-atoms 50000`, first decide whether exact parameters are
required:

- If exact parameters are required, do not claim the agent shell completed the
  run. Explain that the exact run exceeds the per-shell-call limit and provide
  the direct command for the user to run in a normal terminal.
- If the user mainly wants artifacts, run the timeout fallback profile while
  preserving the requested physics choices where possible: potential, mode,
  orientation, crack length, temperature, strain rate, damping, `dt`,
  `bond-cutoff`, `break-stretch`, `color-by`, and potential parameters. State
  explicitly which size/rendering/step parameters were reduced.

If a command times out, do not end with failure if a fallback artifact can still
be produced. Rerun with the timeout fallback profile, verify files with `find`,
read `summary.json`, and report the fallback results clearly.

## Recommended Recipes

### Brittle Mode I Fracture

```bash
python3 skills/fracture-mechanics/scripts/run_fracture_sim.py \
  --out skill_output_<timestamp>_fracture-mechanics \
  --title "Brittle Mode I Fracture" \
  --potential morse \
  --mode I \
  --orientation 90 \
  --morse-a 7.0 \
  --temperature 0.002 \
  --strain-rate 0.0035 \
  --crack-length 0.34 \
  --steps 10000 \
  --frames 48 \
  --dpi 120 \
  --movie-fps 16 \
  --break-stretch 1.65 \
  --color-by stress
```

### Tougher Morse Lattice

```bash
python3 skills/fracture-mechanics/scripts/run_fracture_sim.py \
  --out skill_output_<timestamp>_fracture-mechanics \
  --title "Tougher Morse Lattice" \
  --potential morse \
  --mode I \
  --orientation 90 \
  --morse-a 3.5 \
  --temperature 0.006 \
  --strain-rate 0.0025 \
  --crack-length 0.25 \
  --steps 10000 \
  --frames 48 \
  --dpi 120 \
  --movie-fps 16 \
  --break-stretch 2.05 \
  --color-by coordination
```

### Mode II Shear

```bash
python3 skills/fracture-mechanics/scripts/run_fracture_sim.py \
  --out skill_output_<timestamp>_fracture-mechanics \
  --title "Mode II Shear Crack" \
  --potential morse \
  --mode II \
  --orientation 90 \
  --morse-a 5.0 \
  --temperature 0.004 \
  --strain-rate 0.003 \
  --crack-length 0.30 \
  --steps 10000 \
  --frames 48 \
  --dpi 120 \
  --movie-fps 16 \
  --break-stretch 1.80 \
  --color-by speed
```

### Lennard-Jones Comparison

```bash
python3 skills/fracture-mechanics/scripts/run_fracture_sim.py \
  --out skill_output_<timestamp>_fracture-mechanics \
  --title "Lennard-Jones Fracture Response" \
  --potential lj \
  --mode I \
  --orientation 90 \
  --lj-eps 1.0 \
  --lj-sigma 0.90 \
  --temperature 0.004 \
  --strain-rate 0.003 \
  --steps 10000 \
  --frames 48 \
  --dpi 120 \
  --movie-fps 16 \
  --break-stretch 1.85 \
  --color-by pe
```

## Outputs

The script writes:

- `fracture_movie.gif`: animated fracture movie when Pillow GIF writing is available.
- `fracture_movie.html`: HTML fallback playback of rendered frames.
- `frames/frame_*.png`: individual movie frames.
- `stress_strain.png`: stress-strain plot.
- `final_lattice.png`: final atomistic lattice state.
- `stress_strain.csv`: step, strain, stress data.
- `summary.json`: peak stress, peak strain, final stress, final strain, atom count, and artifact names.
- `parameters.json`: exact input parameters and resolved lattice spacing.
- `README.md`: explanation and run summary.

If `fracture_movie.gif` is missing, report `fracture_movie.html` and the `frames/`
directory as the movie output. Do not claim success unless `find` lists the
movie or fallback, stress-strain plot, CSV, JSON run metrics, and README.

## Notes

This is a simple fracture model for qualitative mechanism exploration and
artifact generation, not a calibrated predictor of a specific material. Runs
with fewer than 10000 steps are useful only for code smoke tests; do not present
them as final fracture results because the crack may not propagate visibly.
If the shell tool times out after partial frames, rerun the same physics with
`--frames 16 --dpi 90`; do not reduce `--steps` below 10000 for the final run.
