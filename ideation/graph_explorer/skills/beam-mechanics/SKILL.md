---
name: beam-mechanics
description: Run simple dimensionless 1D Euler-Bernoulli beam mechanics with selectable boundary conditions, point forces, distributed loads, point moments, springs, plots, CSV/JSON results, and deformation GIFs. Use when asked to analyze a simple beam, cantilever, fixed-fixed beam, overhang, support reactions, shear/moment/deflection plots, or a small educational beam mechanics movie/artifact.
license: 3-clause BSD license
metadata: {"version": "0.2.0", "skill-author": "Markus J. Buehler, Massachusetts Institute of Technology (MIT), Laboratory for Atomistic and Molecular Mechanics (LAMM)"}
---

# Beam Mechanics

Run a simple dimensionless beam mechanics calculation. This skill is
intentionally low freedom: always run the bundled script instead of writing a
new solver or a custom input schema.

The model is a 1D Euler-Bernoulli beam with vertical displacement `w` and
rotation `theta` at each node. Inputs are dimensionless. Sign convention:
positive vertical force/displacement is upward, negative is downward. Prefer the
positive-down helper flags when the user says a load acts downward.

Do not use units. Do not write YAML. Do not write JSON input files unless the
user explicitly asks for a separate record. Translate the user's text directly
into command-line flags.

## Required Workflow

1. Create or use the requested output directory, usually
   `skill_output_<timestamp>_beam-mechanics`.
2. Select a preset or translate the user's boundary conditions and loads into
   flags.
3. Run `scripts/simple_beam_lab.py`.
4. Verify artifacts with `find <output_dir> -maxdepth 2 -type f -print`.
   If files are missing, fix the command and rerun.
4. Read `<output_dir>/manifest.json` and `<output_dir>/summary.json`.
5. Final answer must list exact artifact paths and report reactions, maximum
   absolute deflection, maximum absolute moment, maximum absolute shear, and
   vertical equilibrium residual from `summary.json` and `reactions.json`.

## Default Command

From the session root where the skill is mounted as `skills/beam-mechanics`:

```bash
python3 skills/beam-mechanics/scripts/simple_beam_lab.py \
  --out skill_output_<timestamp>_beam-mechanics \
  --title "Simple Beam Mechanics" \
  --preset simply-supported \
  --length 10 \
  --point-down 5,1 \
  --quiet
```

Verify:

```bash
find skill_output_<timestamp>_beam-mechanics -maxdepth 2 -type f -print
```

## Boundary Conditions

Use `--preset` for common setups:

- `simply-supported`: left pin, right roller.
- `cantilever`: left fixed, right free.
- `fixed-fixed`: both ends fixed.
- `overhang`: left pin, internal roller at `0.7 * length`, right end free.
- `custom`: use `--left`, `--right`, `--support`, and `--spring`.

Support flags:

```bash
--left fixed
--right roller
--support 6,pin
--support 8,fixed
--spring 5,20
```

Support types are `pin`, `roller`, `fixed`, `clamped`, and `free`. For this
simple 1D beam, `pin` and `roller` both constrain vertical displacement only.
`fixed` and `clamped` constrain vertical displacement and rotation.

The script automatically inserts mesh nodes at support, load, moment, UDL end,
and spring positions, so locations such as `x=3.25` are valid.

## Loads

Prefer these positive-down helpers:

```bash
--point-down X,MAGNITUDE
--udl-down START,END,MAGNITUDE
--moment-clockwise X,MAGNITUDE
```

Signed alternatives are available:

```bash
--point X,VALUE
--udl START,END,VALUE
--moment X,VALUE
```

Use negative signed vertical values for downward loads.

Repeat load and support flags for multiple loads/supports.

## Text-To-Flag Translation

Use these examples to convert user language into flags.

Simply supported beam:

```bash
--preset simply-supported
```

Cantilever fixed on the left, free on the right:

```bash
--preset cantilever
```

Custom left pin and right roller:

```bash
--preset custom --left pin --right roller
```

Fixed support at `x=3`:

```bash
--support 3,fixed
```

Roller support at `70 percent of a length-10 beam`:

```bash
--support 7,roller
```

Vertical spring at midspan with stiffness `20` on a length-10 beam:

```bash
--spring 5,20
```

Downward point load of magnitude `1` at midspan of a length-10 beam:

```bash
--point-down 5,1
```

Two downward point loads:

```bash
--point-down 4,1 --point-down 9,0.6
```

Downward uniform load of magnitude `0.2` across the middle 20 percent of a
length-10 beam:

```bash
--udl-down 4,6,0.2
```

Downward uniform load across the full length:

```bash
--udl-down 0,10,0.2
```

Clockwise point moment of magnitude `0.5` at `x=6`:

```bash
--moment-clockwise 6,0.5
```

## Outputs

The script writes:

- `manifest.json`: file map and summary.
- `summary.json`: key values and equilibrium check.
- `reactions.json`: support reactions.
- `field_data.csv`: sampled `x,w,theta,moment,shear`.
- `structure.png`: supports and loads.
- `deformed_shape.png`: scaled deformation.
- `deflection.png`: deflection curve.
- `moment.png`: bending moment curve.
- `shear.png`: shear curve.
- `dashboard.png`: combined figure.
- `deformation.gif`: deformation movie when GIF support is available.
- `frames/frame_*.png`: movie frames when GIF support is available.
- `README.md`: short run explanation.

Do not claim success unless `find` lists these files or `manifest.json` exists
and names them.

If `deformation.gif` is missing because GIF writing is unavailable, report the
`frames/` directory when it exists and still report the static plots, CSV, JSON,
and README.

## Runtime Guidance

Use default `--elements 80 --frames 32 --fps 12` for final artifacts. Add
`--quick` only for smoke tests or timeout recovery. Do not present `--quick`
outputs as final high-quality artifacts unless the user asked for a fast test.

## Copyable Recipes

Simply supported beam, center point load:

```bash
python3 skills/beam-mechanics/scripts/simple_beam_lab.py \
  --out skill_output_<timestamp>_beam-mechanics \
  --title "Simply Supported Center Load" \
  --preset simply-supported \
  --length 10 \
  --point-down 5,1 \
  --quiet
```

Cantilever, tip load plus uniform load:

```bash
python3 skills/beam-mechanics/scripts/simple_beam_lab.py \
  --out skill_output_<timestamp>_beam-mechanics \
  --title "Cantilever With Tip Load And UDL" \
  --preset cantilever \
  --length 8 \
  --point-down 8,1 \
  --udl-down 0,8,0.2 \
  --quiet
```

Fixed-fixed beam with a uniform load:

```bash
python3 skills/beam-mechanics/scripts/simple_beam_lab.py \
  --out skill_output_<timestamp>_beam-mechanics \
  --title "Fixed Fixed Uniform Load" \
  --preset fixed-fixed \
  --length 10 \
  --udl-down 0,10,0.3 \
  --quiet
```

Custom beam with an internal spring and two point loads:

```bash
python3 skills/beam-mechanics/scripts/simple_beam_lab.py \
  --out skill_output_<timestamp>_beam-mechanics \
  --title "Beam With Internal Spring" \
  --preset custom \
  --length 12 \
  --left pin \
  --right roller \
  --spring 6,20 \
  --point-down 4,1 \
  --point-down 9,0.6 \
  --quiet
```

Overhanging beam with an end load:

```bash
python3 skills/beam-mechanics/scripts/simple_beam_lab.py \
  --out skill_output_<timestamp>_beam-mechanics \
  --title "Overhanging Beam End Load" \
  --preset overhang \
  --length 10 \
  --point-down 10,1 \
  --udl-down 0,10,0.1 \
  --quiet
```

For fast tests, add `--quick`. For final visible artifacts, omit `--quick`.
