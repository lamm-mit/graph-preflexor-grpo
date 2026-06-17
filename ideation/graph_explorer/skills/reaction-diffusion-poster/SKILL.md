---
name: reaction-diffusion-poster
description: Create deterministic science-art visuals from Gray-Scott reaction-diffusion dynamics. Use when asked for Turing patterns, morphogenesis, dynamics art, reaction-diffusion simulations, or social-post-ready scientific images. Produces PNG posters, simulation data, captions, and a README by running a bundled Python renderer.
metadata: {"version": "1.0", "skill-author": "Graph Explorer"}
---

# Reaction-Diffusion Poster

Create social-post-ready science art from a real dynamical system. This skill is intentionally low freedom: run the bundled script instead of writing plotting code from scratch.

## Required Workflow

1. Create or use the requested output directory, usually `skill_output_<timestamp>_reaction-diffusion-poster`.
2. Run:

```bash
python3 skills/reaction-diffusion-poster/scripts/make_reaction_diffusion_poster.py \
  --out skill_output_<timestamp>_reaction-diffusion-poster \
  --seed 42 \
  --preset labyrinth \
  --palette magma-cyan \
  --title "Morphogenesis From Local Rules" \
  --subtitle "Gray-Scott reaction-diffusion dynamics"
```

3. Verify artifacts exist before answering:

```bash
find skill_output_<timestamp>_reaction-diffusion-poster -maxdepth 2 -type f -print
```

4. Final answer must list exact artifact paths.

## Presets

Use one of these values for `--preset`:

- `labyrinth`: interconnected Turing stripes; best default.
- `spots`: cellular dots and islands.
- `coral`: branching biomorphic growth.
- `membranes`: folded soft interfaces.
- `veins`: vascular channels and ridges.
- `turbulence`: high-contrast chaotic texture.

## Palettes

Use one of these values for `--palette`:

- `magma-cyan`: dark scientific poster with cyan contour accents.
- `biofilm`: green-gold biological texture.
- `noir-neon`: black, violet, and electric blue.
- `graphite-fire`: graphite background with copper highlights.
- `ice`: pale blue morphology for clean slides.

## Outputs

The script writes:

- `reaction_diffusion_poster.png`: 4:5 social post image.
- `reaction_diffusion_poster_square.png`: square crop for social feeds.
- `reaction_diffusion_data.npz`: compressed final fields and parameters.
- `parameters.json`: exact run parameters.
- `caption.txt`: short social caption.
- `README.md`: plain-language explanation.

If the user does not specify a seed, preset, title, or palette, choose:

```text
seed=42
preset=labyrinth
palette=magma-cyan
title=Morphogenesis From Local Rules
subtitle=Gray-Scott reaction-diffusion dynamics
```

Do not claim success unless `find` lists the generated files.
