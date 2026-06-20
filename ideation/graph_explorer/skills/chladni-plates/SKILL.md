---
name: chladni-plates
description: Create high-resolution Chladni plate resonance visuals and animations from analytic vibrating-plate mode superpositions. Use when asked for Chladni figures, nodal-line sand patterns, sound-wave or resonance visualization, vibrating square/rectangular/circular plates, cymatics-inspired scientific art, physics demonstration images, or GIF movies of plate dynamics. Produces text-free PNG images, GIF animations, frame sequences, field data, parameters JSON, captions, and README by running a bundled Python renderer.
---

# Chladni Plates

Create text-free visual artifacts showing standing-wave nodal patterns on thin
plates. Always run the bundled script; do not write a new simulator unless the
user explicitly asks to modify the skill.

## Required Workflow

1. Create or use an output directory, usually `skill_output_<timestamp>_chladni-plates`.
2. Choose a preset, palette, shape, and optional custom modes.
3. Run `scripts/render_chladni_plate.py`.
4. Verify artifacts with `find`.
5. Final answer must list exact artifact paths and report preset, shape, modes,
   palette, frames, and FPS from `parameters.json`.

Default command:

```bash
python3 skills/chladni-plates/scripts/render_chladni_plate.py \
  --out skill_output_<timestamp>_chladni-plates \
  --preset classic-square \
  --palette blue-sand \
  --size 1080 \
  --frames 48 \
  --fps 18
```

## Presets

Use one of these values for `--preset`:

- `classic-square`: crossed square-plate nodal curves; best default.
- `cathedral-window`: high-order square interference with bright symmetry.
- `bronze-drum`: circular plate with radial and angular disk modes.
- `radial-bloom`: flower-like circular pattern for social visuals.
- `long-bridge`: rectangular plate with long-span standing waves.
- `quiet-glass`: subtle circular glass-like pattern for clean slides.
- `custom`: use explicit `--shape`, `--modes`, and other flags.

## Palettes

Use one of these values for `--palette`:

- `blue-sand`: dark blue plate with pale sand-like nodal ridges.
- `neon-noir`: high-contrast cyan/magenta resonance field.
- `copper-glass`: bronze and glass tones for classic demonstrations.
- `emerald-gold`: green-gold science-art palette.
- `ice-fire`: blue/orange contrast for strong motion.
- `monochrome`: clean grayscale for slides.

## Parameters

- `--shape`: `square`, `rectangle`, or `circle`.
- `--boundary`: `free`, `fixed`, or `mixed`. Use `free` unless the user specifies clamped/fixed edges.
- `--modes`: semicolon-separated mode list `m,n:amplitude:phase`.
  - Square/rectangle: `m,n` are x/y mode numbers.
  - Circle: `m,n` are angular order and radial index.
  - Example: `--modes "4,7:1.0:0;7,4:-0.9:0.4;3,10:0.34:1.1"`.
- `--aspect`: width/height for `rectangle`.
- `--size`: output pixel height. Use `1080` for normal runs, `720` for faster previews, `1400` or higher for final stills.
- `--frames`: GIF frame count. Use `36-72` for normal animations.
- `--fps`: GIF frame rate. Use `16-24`.
- `--node-width`: nodal ridge thickness. Smaller values create sharper sand lines.
- `--grain`: deterministic sand/film grain intensity.
- `--drive`: temporal drive strength for animation.
- `--no-gif`: skip GIF assembly when only a still image is needed.

Images and GIFs intentionally contain no rendered text. Titles and explanations
belong in `caption.txt`, `parameters.json`, and `README.md`.

## Recipes

Classic square plate:

```bash
python3 skills/chladni-plates/scripts/render_chladni_plate.py \
  --out skill_output_<timestamp>_chladni-plates \
  --preset classic-square \
  --palette blue-sand \
  --size 1080 \
  --frames 48 \
  --fps 18
```

Circular bronze resonance:

```bash
python3 skills/chladni-plates/scripts/render_chladni_plate.py \
  --out skill_output_<timestamp>_chladni-plates \
  --preset bronze-drum \
  --palette copper-glass \
  --size 1080 \
  --frames 60 \
  --fps 20
```

High-order square pattern:

```bash
python3 skills/chladni-plates/scripts/render_chladni_plate.py \
  --out skill_output_<timestamp>_chladni-plates \
  --preset cathedral-window \
  --palette neon-noir \
  --size 1200 \
  --frames 60 \
  --fps 20 \
  --node-width 0.045
```

Custom rectangular plate:

```bash
python3 skills/chladni-plates/scripts/render_chladni_plate.py \
  --out skill_output_<timestamp>_chladni-plates \
  --preset custom \
  --shape rectangle \
  --aspect 1.7 \
  --boundary free \
  --modes "2,5:1.0:0;6,3:0.72:0.5;9,2:0.30:1.1" \
  --palette ice-fire \
  --size 900 \
  --frames 48 \
  --fps 18
```

## Outputs

The script writes:

- `chladni_pattern.png`: high-resolution text-free static nodal pattern.
- `chladni_preview.png`: first-frame preview PNG.
- `chladni_animation.gif`: animated vibrating-plate GIF unless `--no-gif` is used.
- `frames/frame_*.png`: individual animation frames.
- `field_data.npz`: final normalized field, mask, grid, and modes.
- `parameters.json`: exact run parameters and artifact paths.
- `caption.txt`: short plain-language explanation.
- `README.md`: run summary.

Do not claim success unless `find` lists the requested artifacts. For a normal
GIF run, verify `chladni_pattern.png`, `chladni_preview.png`,
`chladni_animation.gif`, `frames/`, `field_data.npz`, `parameters.json`,
`caption.txt`, and `README.md`.
