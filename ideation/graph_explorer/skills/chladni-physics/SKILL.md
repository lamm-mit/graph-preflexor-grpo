---
name: chladni-physics
description: Run a physics-based Chladni plate simulation for driven damped rectangular or square thin plates with material properties, plate dimensions, drive frequency, drive location, damping, modal response, and sand-particle settling. Use when asked for more realistic Chladni movies, physical resonance simulations, sand migration to nodal lines, material/thickness/frequency-controlled Chladni figures, or scientifically defensible Chladni artifacts beyond stylized analytic visuals. Produces text-free PNGs, sand-settling GIFs, plate-vibration GIFs, modal frequency CSV, physics NPZ data, parameters JSON, summary JSON, caption, and README by running a bundled Python simulator.
---

# Chladni Physics

Run a physics-based Chladni workflow. Always use the bundled script; do not
write a new simulator from scratch.

## Tool Call Contract

Call the shell tool directly. Do not print JSON command blocks, markdown plans,
or a "Step 1" explanation before the first shell call.

First shell call for the default steel run:

```bash
python3 skills/chladni-physics/scripts/simulate_chladni_physics.py --out skill_output_<timestamp>_chladni-physics --preset steel-square-43 --palette blue-sand --size 1080 --frames 48 --fps 18 --particles 9000 --sand-steps 1800 --max-m 12 --max-n 12 --quiet
```

Second shell call:

```bash
find skill_output_<timestamp>_chladni-physics -maxdepth 3 -type f -print
```

Third shell call:

```bash
python3 - <<'PY'
import json
with open("skill_output_<timestamp>_chladni-physics/summary.json") as f:
    print(json.dumps(json.load(f), indent=2))
PY
```

The script solves a driven damped Kirchhoff-Love thin rectangular plate using a
simply supported modal basis, then simulates sand particles drifting down the
gradient of time-averaged acceleration energy. This is more physical than the
fast `chladni-plates` visual skill, but it is intentionally limited to
rectangular/square plates for reliability.

## Required Workflow

1. Create or use an output directory, usually `skill_output_<timestamp>_chladni-physics`.
2. Choose a preset or explicit physical parameters.
3. Run `scripts/simulate_chladni_physics.py`.
4. Verify artifacts with `find`.
5. Read `summary.json` and report exact artifact paths, drive frequency, target
   mode, max displacement, max acceleration, and final particle node fraction.

Default command:

```bash
python3 skills/chladni-physics/scripts/simulate_chladni_physics.py \
  --out skill_output_<timestamp>_chladni-physics \
  --preset steel-square-43 \
  --palette blue-sand \
  --size 1080 \
  --frames 48 \
  --fps 18 \
  --particles 9000 \
  --sand-steps 1800 \
  --max-m 12 \
  --max-n 12 \
  --quiet
```

## Presets

Use one of these values for `--preset`:

- `steel-square-43`: 30 cm steel square plate, excited near mode 4,3.
- `glass-square-65`: 24 cm glass square plate, excited near mode 6,5.
- `aluminum-rectangle-27`: 42 cm by 24 cm aluminum plate, excited near mode 2,7.
- `brass-square-54`: 28 cm brass square plate, excited near mode 5,4.
- `custom`: require explicit material, dimensions, target mode, drive point, and damping.

## Main Parameters

Physical controls:

- `--material`: `steel`, `aluminum`, `brass`, `glass`, or `acrylic`.
- `--youngs-modulus`, `--density`, `--poisson`: optional material overrides.
- `--length`, `--width`, `--thickness`: plate dimensions in meters.
- `--target-mode`: mode used to choose the drive frequency when `--drive-frequency` is omitted.
- `--drive-frequency`: explicit drive frequency in Hz.
- `--drive-ratio`: drive frequency divided by target-mode frequency.
- `--drive-x`, `--drive-y`: forcing location as fractions of plate length and width.
- `--force`: harmonic force amplitude in Newtons.
- `--damping`: modal damping ratio.
- `--max-m`, `--max-n`: highest modal indices included.

Sand and rendering controls:

- `--particles`: number of sand particles.
- `--sand-steps`: particle transport steps.
- `--particle-mobility`, `--particle-drag`, `--particle-noise`: sand transport model.
- `--grid`: physics grid height.
- `--size`: output image height in pixels.
- `--frames`, `--fps`: GIF frame count and frame rate.
- `--palette`: `blue-sand`, `copper`, `ice-fire`, `monochrome`, or `neon`.
- `--node-width`: visual nodal highlight width.
- `--skip-vibration-gif`, `--skip-sand-gif`: omit expensive GIFs when needed.

Images and GIFs intentionally contain no rendered text. Titles and explanations
belong in `caption.txt`, `parameters.json`, `summary.json`, and `README.md`.

## Recipes

Steel square plate, mode 4,3:

```bash
python3 skills/chladni-physics/scripts/simulate_chladni_physics.py \
  --out skill_output_<timestamp>_chladni-physics \
  --preset steel-square-43 \
  --palette blue-sand \
  --size 1080 \
  --frames 48 \
  --fps 18 \
  --particles 9000 \
  --sand-steps 1800
```

Glass plate with cleaner presentation styling:

```bash
python3 skills/chladni-physics/scripts/simulate_chladni_physics.py \
  --out skill_output_<timestamp>_chladni-physics \
  --preset glass-square-65 \
  --palette monochrome \
  --size 1200 \
  --frames 48 \
  --fps 16 \
  --particles 10000 \
  --sand-steps 2200
```

Long aluminum plate:

```bash
python3 skills/chladni-physics/scripts/simulate_chladni_physics.py \
  --out skill_output_<timestamp>_chladni-physics \
  --preset aluminum-rectangle-27 \
  --palette ice-fire \
  --size 900 \
  --frames 48 \
  --fps 18 \
  --particles 8000 \
  --sand-steps 1800
```

Custom steel plate:

```bash
python3 skills/chladni-physics/scripts/simulate_chladni_physics.py \
  --out skill_output_<timestamp>_chladni-physics \
  --preset custom \
  --material steel \
  --length 0.32 \
  --width 0.24 \
  --thickness 0.0011 \
  --target-mode 5,4 \
  --drive-ratio 1.002 \
  --drive-x 0.36 \
  --drive-y 0.31 \
  --damping 0.006 \
  --palette neon \
  --size 960 \
  --frames 60 \
  --fps 20 \
  --particles 10000 \
  --sand-steps 2400
```

## Outputs

The script writes:

- `chladni_sand_pattern.png`: final simulated sand accumulation.
- `sand_settling.gif`: sand migration movie.
- `plate_vibration.gif`: driven plate displacement movie.
- `vibration_amplitude.png`: time-averaged vibration amplitude.
- `displacement_snapshot.png`: displacement at phase zero.
- `frames/sand/frame_*.png`: sand settling frames.
- `frames/vibration/frame_*.png`: plate vibration frames.
- `modal_frequencies.csv`: modal frequencies, drive coupling, amplitudes.
- `physics_data.npz`: response fields and final particle coordinates.
- `parameters.json`: exact input parameters and artifact paths.
- `summary.json`: key physical results.
- `caption.txt`: short explanation.
- `README.md`: run summary.

Do not claim success unless `find` lists the requested PNGs, GIFs, frame
directories, `modal_frequencies.csv`, `physics_data.npz`, `parameters.json`,
`summary.json`, `caption.txt`, and `README.md`.
