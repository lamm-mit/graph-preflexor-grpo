---
name: morphogenesis-postcard
description: Generate deterministic science-art postcard PNGs inside a shell/container skill environment. Use when asked for morphogenesis visuals, reaction-diffusion art, Turing patterns, dynamics-inspired scientific social posts, prompt-written visual rules/code, or reliable 4B-model-friendly visual artifacts. Always writes an inspectable Python visual-rule file, then renders PNG, parameters JSON, caption text, and README with the bundled renderer.
---

# Morphogenesis Postcard

Create social-post-ready science art from a deterministic Gray-Scott reaction-diffusion simulation. The normal workflow must always write a prompt-specific Python visual rule file first, then pass that file to the renderer with `--rule-code`. The rule file is part of the artifact: it exposes the visual grammar used for the run.

## Required Workflow

1. Create or use the requested output directory, usually `skill_output_<timestamp>_morphogenesis-postcard`.
2. Write `agent_rules.py` from the user's creative request. Keep the prompt intact when possible:

```bash
python3 skills/morphogenesis-postcard/scripts/write_visual_rules.py \
  --out skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py \
  --profile-json skill_output_<timestamp>_morphogenesis-postcard/visual_rule_profile.json \
  --prompt "stress-wave interference in hierarchical materials with branching cracks and glowing defects" \
  --title "Stress-Wave Morphogenesis" \
  --subtitle "Prompt-written reaction-diffusion rules"
```

3. Render with that rule file. Do not omit `--rule-code` in a final skill run:

```bash
python3 skills/morphogenesis-postcard/scripts/make_morphogenesis_postcard.py \
  --out skill_output_<timestamp>_morphogenesis-postcard \
  --rule-code skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py \
  --prompt "stress-wave interference in hierarchical materials with branching cracks and glowing defects" \
  --title "Stress-Wave Morphogenesis" \
  --subtitle "Prompt-written reaction-diffusion rules" \
  --auto-style \
  --format square \
  --steps 220
```

4. Verify artifacts exist before answering:

```bash
find skill_output_<timestamp>_morphogenesis-postcard -maxdepth 2 -type f -print
```

5. Final answer must list exact artifact paths, including `agent_rules.py`. Do not claim success unless `find` lists the files.

## Parameters

Use these command-line options for `write_visual_rules.py`:

- `--prompt`: Natural-language visual brief. This is the main creative input.
- `--title`: Short headline rendered on the poster.
- `--subtitle`: Short caption rendered under the title.
- `--out`: Path to the required `agent_rules.py` visual-rule file.
- `--profile-json`: Optional JSON summary of derived prompt themes and style.

Use these command-line options for `make_morphogenesis_postcard.py`:

- `--rule-code`: Required for final outputs. Point to the generated `agent_rules.py`.
- `--prompt`: Same visual brief passed to `write_visual_rules.py`.
- `--auto-style`: Derives any unpinned style option from the prompt.
- `--prompt-strength`: 0-1 strength for prompt-derived simulation/tone perturbations. Default is `0.78`.
- `--title`: Short headline rendered on the poster.
- `--subtitle`: Short caption rendered under the title.
- `--seed`: String or number for deterministic variation.
- `--preset`: One of `labyrinth`, `spots`, `coral`, `veins`, `membranes`.
- `--palette`: One of `magma-cyan`, `biofilm`, `noir-neon`, `ice`, `graphite-fire`.
- `--composition`: One of `poster`, `full-bleed`, `specimen`, `triptych`, `field-guide`.
- `--motif`: One of `hybrid`, `flow`, `rings`, `fibers`, `fracture`, `constellation`, `none`.
- `--symmetry`: One of `none`, `mirror-x`, `mirror-y`, `dihedral`.
- `--title-mode`: One of `panel`, `caption`, `none`.
- `--energy`: 0-1 field intensity and motif force.
- `--contrast`: Field contrast multiplier. Use 0.7-1.8.
- `--grain`: 0-1 deterministic print grain.
- `--ink-density`: 0-1 density of lines, rings, fibers, and nodes.
- `--accent-count`: Number of accent motifs/nodes. Use 4-24 for most posts.
- `--format`: `square` or `portrait`.
- `--steps`: Use 180-280 for normal runs, 80-120 for fast previews.
- `--size`: Optional PNG width in pixels. Default is 1080; use 720 for fast tests.
- `--field-size`: Optional internal simulation grid. Default is 152; use 124 for fast tests.

The generated visual rule file defines `configure`, `seed_rule`, `tone_rule`, `color_rule`, `motif_paths`, and `postprocess`. It recognizes scientific and aesthetic language including stress, fracture, waves, interference, phase, morphogenesis, cells, membranes, fibers, hierarchy, graphs, bridges, nodes, crystals, lattices, thermal energy, plasma, and quiet/specimen-style prompts. The prompt hash changes the deterministic rule seed, so different prompts produce different code and images.

If the user asks to pin a specific style option, pass that explicit flag too; prompt mode fills unpinned options and respects explicit `--preset`, `--palette`, `--composition`, `--motif`, `--symmetry`, `--title-mode`, `--seed`, `--energy`, `--contrast`, `--grain`, `--ink-density`, and `--accent-count`.

## Optional Manual Rule Editing

If the user explicitly asks for hand-authored rules, write or edit `agent_rules.py` yourself. Keep the same API:

```bash
cat > skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py <<'PY'
import math

def configure(args):
    return {
        "composition": "full-bleed",
        "motif": "none",
        "title_mode": "none",
        "energy": 0.92,
        "contrast": 1.35,
        "grain": 0.14,
    }

def tone_rule(fx, fy, x, y, activator, depletion, edge, ridge, tone, context):
    n = context["n"]
    wave = math.sin(fx * 0.21 + math.sin(fy * 0.08) * 2.5)
    cellular = math.sin((fx - n * 0.5) * 0.12) * math.sin((fy - n * 0.5) * 0.12)
    return max(0.0, min(1.0, tone * 0.72 + abs(wave) * 0.18 + abs(cellular) * 0.16 + edge * 0.18))

def motif_paths(context):
    return [
        {"type": "circle", "x": 0.50, "y": 0.50, "radius": 0.42, "ring": True, "color": "accent", "alpha": 0.18, "thickness": 2},
        {"type": "polyline", "points": [(0.08, 0.18), (0.25, 0.34), (0.52, 0.30), (0.80, 0.64), (0.94, 0.82)], "color": "muted", "alpha": 0.22, "thickness": 2},
    ]
PY
```

Then run:

```bash
python3 skills/morphogenesis-postcard/scripts/make_morphogenesis_postcard.py \
  --out skill_output_<timestamp>_morphogenesis-postcard \
  --rule-code skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py \
  --title "Agent-Written Morphogenesis" \
  --subtitle "Custom tone rules and motifs" \
  --seed custom-01 \
  --preset coral \
  --palette noir-neon \
  --format square \
  --steps 180
```

Supported functions in `agent_rules.py`:

- `configure(args) -> dict | None`: Override command-line options before rendering.
- `seed_rule(u, v, n, args) -> (u, v) | None`: Modify initial fields in place or return new fields.
- `tone_rule(fx, fy, x, y, activator, depletion, edge, ridge, tone, context) -> float | None`: Override per-pixel tone.
- `color_rule(r, g, b, tone, x, y, fx, fy, context) -> (r, g, b) | None`: Override per-pixel RGB.
- `motif_paths(context) -> list[dict]`: Add line/polyline/circle/frame motifs using relative coordinates.
- `postprocess(image, width, height, args, palette, helpers) -> bytearray | None`: Draw directly on the RGB bytearray using helper functions.

Keep custom code short and deterministic. Import only Python standard-library modules. If a rule file fails, fix it and rerun the renderer; do not claim success.

## Defaults

If the user does not specify parameters, use:

```text
title=Morphogenesis From Local Rules
subtitle=Prompt-shaped reaction-diffusion field
prompt=<copy the user's visual/art/science request>
auto-style=true
rule-code=skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py
prompt-strength=0.78
seed=42
preset/palette/composition/motif/symmetry/title-mode/energy/contrast/grain/ink-density/accent-count=derived from prompt
format=square
steps=220
```

## Recipes

### Prompt-Driven Materials Interface

```bash
python3 skills/morphogenesis-postcard/scripts/write_visual_rules.py \
  --out skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py \
  --profile-json skill_output_<timestamp>_morphogenesis-postcard/visual_rule_profile.json \
  --prompt "stress-wave interference in a growing hierarchical material interface with branching cracks, glowing defects, and nonlinear energy flow" \
  --title "Stress Waves in a Growing Interface" \
  --subtitle "Prompt-written reaction-diffusion rules"

python3 skills/morphogenesis-postcard/scripts/make_morphogenesis_postcard.py \
  --out skill_output_<timestamp>_morphogenesis-postcard \
  --rule-code skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py \
  --prompt "stress-wave interference in a growing hierarchical material interface with branching cracks, glowing defects, and nonlinear energy flow" \
  --title "Stress Waves in a Growing Interface" \
  --subtitle "Prompt-written reaction-diffusion rules" \
  --auto-style \
  --format square \
  --steps 220
```

### Prompt-Driven Museum Specimen

```bash
python3 skills/morphogenesis-postcard/scripts/write_visual_rules.py \
  --out skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py \
  --profile-json skill_output_<timestamp>_morphogenesis-postcard/visual_rule_profile.json \
  --prompt "quiet museum specimen of cellular morphogenesis in ice-blue symmetry, like an archived biological crystal" \
  --title "Specimen of Emergent Order" \
  --subtitle "Cellular fields constrained into symmetry"

python3 skills/morphogenesis-postcard/scripts/make_morphogenesis_postcard.py \
  --out skill_output_<timestamp>_morphogenesis-postcard \
  --rule-code skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py \
  --prompt "quiet museum specimen of cellular morphogenesis in ice-blue symmetry, like an archived biological crystal" \
  --title "Specimen of Emergent Order" \
  --subtitle "Cellular fields constrained into symmetry" \
  --auto-style \
  --format square \
  --steps 200
```

### Prompt-Driven Research Graph Artwork

```bash
python3 skills/morphogenesis-postcard/scripts/write_visual_rules.py \
  --out skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py \
  --profile-json skill_output_<timestamp>_morphogenesis-postcard/visual_rule_profile.json \
  --prompt "luminous graph of research ideas bridging distant modules, with nodes, links, hidden paths, and morphogenesis-like diffusion" \
  --title "Bridge Field" \
  --subtitle "A graph of ideas becomes a reaction field"

python3 skills/morphogenesis-postcard/scripts/make_morphogenesis_postcard.py \
  --out skill_output_<timestamp>_morphogenesis-postcard \
  --rule-code skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py \
  --prompt "luminous graph of research ideas bridging distant modules, with nodes, links, hidden paths, and morphogenesis-like diffusion" \
  --title "Bridge Field" \
  --subtitle "A graph of ideas becomes a reaction field" \
  --auto-style \
  --format square \
  --steps 240
```

### Prompt-Driven Scientific Triptych

```bash
python3 skills/morphogenesis-postcard/scripts/write_visual_rules.py \
  --out skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py \
  --profile-json skill_output_<timestamp>_morphogenesis-postcard/visual_rule_profile.json \
  --prompt "three-panel scientific triptych of branching biofilm fibers, cellular growth fronts, and nonlinear wave coupling" \
  --title "Three Views of a Reaction Field" \
  --subtitle "One seed, shifted sampling windows"

python3 skills/morphogenesis-postcard/scripts/make_morphogenesis_postcard.py \
  --out skill_output_<timestamp>_morphogenesis-postcard \
  --rule-code skill_output_<timestamp>_morphogenesis-postcard/agent_rules.py \
  --prompt "three-panel scientific triptych of branching biofilm fibers, cellular growth fronts, and nonlinear wave coupling" \
  --title "Three Views of a Reaction Field" \
  --subtitle "One seed, shifted sampling windows" \
  --auto-style \
  --composition triptych \
  --format portrait \
  --steps 220
```

## Outputs

The script writes:

- `<title>_<preset>_<palette>_<seed>.png`: social-post PNG.
- `parameters.json`: exact reproduction parameters.
- `caption.txt`: short social caption.
- `README.md`: plain-language explanation.
- `agent_rules.py`: required prompt-written visual rule code.
- `visual_rule_profile.json`: prompt theme/style profile, when written by `write_visual_rules.py`.

Do not use `run_js`. This skill is for the shell/container environment used by `mistralrs_skill_cli.py`.
