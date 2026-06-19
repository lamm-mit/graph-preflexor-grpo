# mistral.rs Skill CLI Examples

This directory contains examples for running local OpenAI-compatible Skills
through `mistralrs_skill_cli.py`.

The CLI is intentionally simple:

- it zips a skill directory,
- uploads the skill to `POST /v1/skills`,
- uploads each `--file` to `POST /v1/files` with `purpose=user_data`,
- sends a Responses request with `input_file.file_id`,
- enables shell, code execution, and web search by default,
- writes a local results package for every run.

## Start mistral.rs

The Skills runtime requires the shell executor. The easiest local mode is
`--agent`, which also enables code execution and web search.

From this examples directory, a minimal single-model server is:

```bash
cd /Users/mbuehler/LOCALCODES/graph-preflexor-grpo/ideation/graph_explorer/examples
mistralrs from-config -f models.toml
```

From the main ideation directory, use the multi-model server:

```bash
cd /Users/mbuehler/LOCALCODES/graph-preflexor-grpo/ideation
mistralrs from-config -f models.toml
```

For direct serving without TOML:

```bash
mkdir -p /tmp/mistralrs-shell-workdir /tmp/mistralrs-skills

mistralrs serve --agent \
  --host 0.0.0.0 \
  --port 1234 \
  --shell-workdir /tmp/mistralrs-shell-workdir \
  --skills-dir /tmp/mistralrs-skills \
  -m google/gemma-4-E4B-it
```

`--shell-workdir` is optional but useful. Without it, mistral.rs uses a
per-session temp directory for shell outputs. With it, generated files are also
visible directly on disk under the chosen path.

Check the server:

```bash
curl http://localhost:1234/v1/models
```

## Run The CLI

From this examples directory:

```bash
cd /Users/mbuehler/LOCALCODES/graph-preflexor-grpo/ideation/graph_explorer/examples

./mistralrs_skill_cli.py ../skills/graphml-deep-analysis \
  --file iter_0002.graphml \
  --query "Which concepts bridge distant thematic modules, and what three testable hypotheses follow?"
```

By default, each `--file` is uploaded first:

```json
{"type": "input_file", "file_id": "file-..."}
```

Use inline base64 only when you want a single self-contained request body:

```bash
./mistralrs_skill_cli.py ../skills/graphml-deep-analysis \
  --file iter_0002.graphml \
  --file-mode inline \
  --query "Summarize the graph modules and bridge concepts."
```

Useful flags:

```bash
--model google/gemma-4-E4B-it
--base-url http://localhost:1234/v1
--max-tool-rounds 12
--response-timeout 1800
--no-search
--no-code
--artifact-bundle
--require-tool
```

## Python API Examples

These examples use the OpenAI Python client against a local mistral.rs server.
They assume you already have a skill ZIP such as `d3-viz.zip` or
`latex-document-skill.zip`. The CLI can create equivalent ZIPs automatically,
but raw API clients upload ZIP files directly to `/v1/skills`.

### D3 Skill With Uploaded Data

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-used")
model = "google/gemma-4-E4B-it"

with Path("d3-viz.zip").open("rb") as f:
    skill = client.post(
        "/skills",
        cast_to=dict,
        files={"file": ("d3-viz.zip", f, "application/zip")},
    )

with Path("research_opportunity_bridges.json").open("rb") as f:
    uploaded = client.files.create(file=f, purpose="user_data")

response = client.responses.create(
    model=model,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": uploaded.id},
                {
                    "type": "input_text",
                    "text": (
                        "Use the d3-viz skill to create a self-contained HTML "
                        "force-directed network from research_opportunity_bridges.json. "
                        "Color nodes by module, size nodes by bridge_score, scale links "
                        "by strength, and add tooltips for hypotheses and experiments. "
                        "Save the HTML and transformed JSON as artifacts."
                    ),
                },
            ],
        }
    ],
    tools=[
        {
            "type": "shell",
            "environment": {
                "type": "container_auto",
                "skills": [
                    {
                        "type": "skill_reference",
                        "skill_id": skill["id"],
                        "version": "latest",
                    }
                ],
            },
        }
    ],
    max_tool_rounds=8,
)

print(response.output_text)
```

### Web Search Then D3 Visualization

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-used")
model = "google/gemma-4-E4B-it"

with Path("d3-viz.zip").open("rb") as f:
    skill = client.post(
        "/skills",
        cast_to=dict,
        files={"file": ("d3-viz.zip", f, "application/zip")},
    )

response = client.responses.create(
    model=model,
    input=(
        "Search the web for recent developments in self-healing materials, "
        "mechanophores, graph neural networks for materials, and bioinspired "
        "vascular repair. Build a compact source-cited dataset with categories, "
        "evidence strength, and cross-field links. Then use d3-viz to create a "
        "self-contained interactive D3 network or matrix. Save both the dataset "
        "as JSON and the visualization as HTML artifacts."
    ),
    tools=[
        {"type": "web_search", "search_context_size": "medium"},
        {
            "type": "shell",
            "environment": {
                "type": "container_auto",
                "skills": [
                    {
                        "type": "skill_reference",
                        "skill_id": skill["id"],
                        "version": "latest",
                    }
                ],
            },
        },
    ],
    max_tool_rounds=12,
)

print(response.output_text)
```

### Web Search Then LaTeX Document

This asks mistral.rs to search first, synthesize a compact evidence base, then
use `latex-document-skill` to create a compiled report artifact.

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-used")
model = "google/gemma-4-E4B-it"

with Path("latex-document-skill.zip").open("rb") as f:
    skill = client.post(
        "/skills",
        cast_to=dict,
        files={"file": ("latex-document-skill.zip", f, "application/zip")},
    )

response = client.responses.create(
    model=model,
    input=(
        "Search the web for recent work on autonomous laboratories, active "
        "learning for materials discovery, and self-healing polymer systems. "
        "Create a source-cited mini review as a polished LaTeX report with an "
        "abstract, introduction, evidence table, opportunity map, three proposed "
        "experiments, limitations, and bibliography. Use latex-document-skill to "
        "write the .tex file, compile a PDF, generate preview images if possible, "
        "and save all outputs as artifacts."
    ),
    tools=[
        {"type": "web_search", "search_context_size": "medium"},
        {
            "type": "shell",
            "environment": {
                "type": "container_auto",
                "skills": [
                    {
                        "type": "skill_reference",
                        "skill_id": skill["id"],
                        "version": "latest",
                    }
                ],
            },
        },
    ],
    max_tool_rounds=14,
)

print(response.output_text)
```

## Where Results Go

Every run creates a local folder:

```text
skill_cli_runs/<timestamp>_<skill>/
```

The important files are:

```text
RESULTS.md                  human-readable run index
run_manifest.json           machine-readable run index
answer.md                   final model answer
request.json                exact Responses request
response.json               raw Responses response
skill_upload.json           uploaded skill metadata
input_file_uploads.json     uploaded user file ids
artifact_summary.json       downloaded/extracted artifact summary
downloaded_files/           files downloaded from /v1/files/{id}/content
                            or /v1/containers/{container_id}/files/{file_id}/content
downloaded_files/shell_artifacts/
                            copied skill_output_* shell directories, when found
extracted_artifacts/        extracted ZIP artifacts
```

The CLI asks the skill to write shell artifacts into a directory named like
`skill_output_<timestamp>_<skill>` and to report exact artifact paths. If
mistral.rs surfaces generated files as response file objects, the CLI downloads
them. It also understands OpenAI-style `container_file_citation` annotations
with `container_id` and `file_id`, using the container file content endpoint.
If the server only leaves files in the shell workdir, the CLI scans for the
requested `skill_output_*` directory in local mistral.rs shell/temp workdirs and
copies it into `downloaded_files/shell_artifacts/`. If you pass
`--artifact-bundle`, the prompt also asks the skill to ZIP small artifacts and
print a base64 marker that the CLI decodes into `downloaded_files/` and extracts
into `extracted_artifacts/`. Keep this off for large D3/LaTeX/HTML/PDF artifacts
unless you really want the ZIP carried through the text response.

For durable server-side shell outputs, start mistral.rs with `--shell-workdir`.

## Skill Inventory

The CLI expects the skill directory name to match `SKILL.md` frontmatter
`name:`.

| Directory | Skill name | Best use |
|---|---|---|
| `graphml-deep-analysis` | `graphml-deep-analysis` | Deep Graph Explorer / Graph-PRefLexOR GraphML interpretation. |
| `networkx` | `networkx` | Standard graph algorithms, centrality, communities, paths, and graph plots. |
| `scientific-brainstorming` | `scientific-brainstorming` | High-variance research ideation and opportunity generation. |
| `scientific-critical-thinking` | `scientific-critical-thinking` | Stress-test claims, hypotheses, evidence, and experimental design. |
| `literature-review` | `literature-review` | Web-search-backed literature synthesis and citation-oriented reports. |
| `scientific-visualization` | `scientific-visualization` | Publication-style multi-panel figures and analysis plots. |
| `scientific-writing` | `scientific-writing` | Manuscript-style prose, concept notes, abstracts, and structured reports. |
| `seaborn` | `seaborn` | Fast statistical visual exploration with pandas/seaborn. |
| `xlsx` | `xlsx` | Spreadsheet deliverables, ranking tables, and Excel workbooks. |
| `docx` | `docx` | Word document deliverables. |
| `pptx` | `pptx` | PowerPoint files and deck manipulation. |
| `scientific-slides` | `scientific-slides` | Research talk structure and slide generation workflows. |
| `pptx-posters` | `pptx-posters` | HTML/CSS research posters exportable to PDF/PPTX. |
| `latex-document-skill` | `latex-document-skill` | LaTeX document creation, compilation, conversion, PDF previews, and bibliographies. |
| `pdf` | `pdf` | PDF reading, creation, extraction, merging, forms, OCR, and PDF output workflows. |
| `rdkit` | `rdkit` | Cheminformatics, SMILES/SDF, descriptors, similarity, and reactions. |
| `pymatgen` | `pymatgen` | Materials structures, CIF/POSCAR, phase diagrams, and materials analysis. |
| `torchsim` | `torchsim` | Atomistic simulation and MLIP-style workflows. |
| `scientific-schematics` | `scientific-schematics` | Scientific diagrams and schematics. |
| `d3-viz` | `d3-viz` | Bespoke D3.js charts, graph visualizations, and interactive SVG/network views. |
| `algorithmic-art` | `algorithmic-art` | Seeded p5.js generative art and interactive visual sketches. |
| `reaction-diffusion-poster` | `reaction-diffusion-poster` | Deterministic Gray-Scott dynamics posters for social-ready science art. |
| `morphogenesis-postcard` | `morphogenesis-postcard` | Container-rendered morphogenesis PNG postcards from reaction-diffusion dynamics. |
| `fracture-mechanics` | `fracture-mechanics` | Parameterized 2D pre-cracked lattice fracture movies and stress-strain plots. |
| `beam-mechanics` | `beam-mechanics` | Simple dimensionless 1D beam mechanics with selectable supports, loads, plots, and deformation GIFs. |
| `hierarchical-topopt` | `hierarchical-topopt` | Fast 2D SIMP topology optimization with flexible supports, loads, density plots, and STL exports. |
| `slack-gif-creator` | `slack-gif-creator` | Animated GIFs optimized for Slack constraints. |
| `skill-creator` | `skill-creator` | Creating, editing, benchmarking, and optimizing skills. |

## Impactful Examples

### 1. Deep GraphML Bridge Analysis

Best first example for Graph Explorer snapshots.

```bash
./mistralrs_skill_cli.py ../skills/graphml-deep-analysis \
  --file iter_0002.graphml \
  --max-tool-rounds 10 \
  --query "Which concepts are acting as the most surprising bridges between distant thematic modules? For each bridge, cite the graph evidence, explain why it is non-obvious, and propose one testable research hypothesis."
```

Expected outputs:

- ranked bridge concepts,
- module/theme names,
- path evidence,
- hypotheses tied to graph structure,
- analysis artifacts under `downloaded_files/` or `extracted_artifacts/`.

### 2. NetworkX Structural Audit

Use this when you want neutral graph metrics before interpretation.

```bash
./mistralrs_skill_cli.py ../skills/networkx \
  --file iter_0002.graphml \
  --max-tool-rounds 10 \
  --query "Read iter_0002.graphml with NetworkX. Produce a structural audit with node and edge counts, component structure, degree distribution, PageRank, betweenness, community detection, articulation/bridge-like nodes if applicable, and a compact CSV ranking the top 25 structurally important nodes."
```

Expected outputs:

- graph metric summary,
- top-node rankings,
- optional CSV/JSON/PNG artifacts if the model creates them.

### 3. Graph-To-Hypothesis Ideation Sprint

Use the graph as an idea map, not just a network.

```bash
./mistralrs_skill_cli.py ../skills/scientific-brainstorming \
  --file iter_0002.graphml \
  --max-tool-rounds 8 \
  --query "Treat this GraphML file as a concept map from an ideation run. Generate 12 high-risk, high-upside research directions that connect distant graph modules. For each direction, include the bridge concept, the scientific mechanism to test, a minimum viable experiment, and the strongest reason it might fail."
```

Expected outputs:

- idea portfolio,
- explicit failure modes,
- experiment sketches.

### 4. Critical Review Of Graph-Derived Hypotheses

Use this after a graph analysis or brainstorming run.

```bash
./mistralrs_skill_cli.py ../skills/scientific-critical-thinking \
  --file iter_0002.graphml \
  --max-tool-rounds 8 \
  --query "Use the attached graph as generated evidence, not ground truth. Identify the five most tempting but weakest inferences someone might draw from this graph. For each, explain the confounder, missing evidence, and what experiment or data would make the claim credible."
```

Expected outputs:

- claim risk analysis,
- bias/confounder list,
- evidence requirements.

### 5. Literature-Backed Opportunity Scan

This uses web search by default. Disable with `--no-search` if you want offline
reasoning only.

```bash
./mistralrs_skill_cli.py ../skills/literature-review \
  --file iter_0002.graphml \
  --max-tool-rounds 12 \
  --query "Use the graph to identify three cross-module research opportunities, then perform a compact literature scan for each. Return a table with opportunity, likely keywords, representative recent papers or search targets, what is already known, what appears underexplored, and a concrete next experiment."
```

Expected outputs:

- search-backed opportunity table,
- citations or search targets,
- gap analysis.

### 6. Publication-Ready Graph Figure

Use this to turn the graph into a figure package.

```bash
./mistralrs_skill_cli.py ../skills/scientific-visualization \
  --file iter_0002.graphml \
  --max-tool-rounds 12 \
  --query "Create a publication-ready multi-panel figure from the GraphML file. Panel A: community-level network overview. Panel B: top bridge nodes by betweenness. Panel C: relation-type distribution. Panel D: shortest bridge paths among the three most distinct modules. Save PNG, PDF/SVG if possible, and a CSV used to build the figure."
```

Expected outputs:

- figure files in `downloaded_files/` or extracted artifact bundle,
- source CSV/JSON,
- caption text in `answer.md`.

### 7. Interactive D3 Research Opportunity Map

Use this to exercise the `d3-viz` skill with a purpose-built JSON dataset.

```bash
./mistralrs_skill_cli.py ../skills/d3-viz \
  --file research_opportunity_bridges.json \
  --max-tool-rounds 10 \
  --query "Use d3-viz to create an interactive force-directed network from research_opportunity_bridges.json. Encode modules by color, bridge_score by node radius, relation strength by link width, and show tooltips with hypothesis and experiment text. Save a self-contained HTML artifact plus any extracted data files."
```

Expected outputs:

- self-contained HTML visualization,
- copied or transformed JSON data,
- notes on encodings and interactions.

### 8. Interactive D3 Physics Simulation

Use this to visualize a real simulated physics dataset. The included
`fput_nonlinear_chain_simulation.json` file is generated by
`generate_fput_simulation_dataset.py`, which runs a velocity-Verlet simulation
of a fixed-end alpha-FPUT nonlinear oscillator chain.

```bash
./mistralrs_skill_cli.py ../skills/d3-viz \
  --file fput_nonlinear_chain_simulation.json \
  --max-tool-rounds 10 \
  --query "Use d3-viz to create a self-contained interactive HTML visualization of fput_nonlinear_chain_simulation.json. Include an animated chain displacement view, a time-position heatmap, a normal-mode energy transfer chart, and an energy conservation plot. Add concise annotations explaining the FPUT model, the nonlinear spring coupling, and the observed energy drift."
```

Expected outputs:

- self-contained HTML physics visualization,
- transformed JSON or CSV data used by the visualization,
- explanation of the oscillator chain, mode transfer, and energy conservation.

Regenerate the dataset:

```bash
python generate_fput_simulation_dataset.py
```

### 9. Simpler D3 Driven Oscillator

Use this when you want a faster real-physics D3 test than the FPUT chain. The
included `damped_driven_oscillator.json` file is generated by
`generate_damped_oscillator_dataset.py`, which integrates a single
mass-spring-damper oscillator with sinusoidal forcing.

```bash
./mistralrs_skill_cli.py ../skills/d3-viz \
  --file damped_driven_oscillator.json \
  --max-tool-rounds 8 \
  --query "Use d3-viz to create a self-contained interactive HTML visualization of damped_driven_oscillator.json. Include an animated mass-spring oscillator, synchronized position/velocity/drive-force time series, a phase-space plot colored by time, and an energy panel. Add concise annotations explaining damping ratio, natural frequency, drive frequency, and the transition toward driven steady-state motion."
```

Expected outputs:

- self-contained HTML oscillator visualization,
- transformed JSON or CSV data used by the visualization,
- explanation of the damped driven oscillator parameters and energy behavior.

Regenerate the dataset:

```bash
python generate_damped_oscillator_dataset.py
```

### 10. Web Search To D3 Visualization

This example asks the model to search first, structure a small dataset, then use
`d3-viz` to build an interactive visual artifact. It uses web search by default.

```bash
./mistralrs_skill_cli.py ../skills/d3-viz \
  --max-tool-rounds 12 \
  --query "Search the web for recent developments in self-healing materials, mechanophores, graph neural networks for materials, and bioinspired vascular repair. Build a small source-cited dataset with categories, evidence strength, and cross-field links. Then use d3-viz to create an interactive D3 network or matrix that visualizes the opportunity landscape. Save the dataset as JSON and the visualization as self-contained HTML."
```

Expected outputs:

- source-cited JSON dataset,
- D3 HTML visualization,
- summary of search sources and design choices.

### 11. Web Search To LaTeX Mini Review

This example asks the model to search first, then use `latex-document-skill` to
produce a compiled report artifact.

```bash
./mistralrs_skill_cli.py ../skills/latex-document-skill \
  --max-tool-rounds 14 \
  --query "Search the web for recent work on autonomous laboratories, active learning for materials discovery, and self-healing polymer systems. Create a source-cited mini review as a polished LaTeX report with an abstract, introduction, evidence table, opportunity map, three proposed experiments, limitations, and bibliography. Use latex-document-skill to write the .tex file, compile a PDF, generate preview images if possible, and save all outputs as artifacts."
```

Expected outputs:

- `.tex` source document,
- compiled `.pdf`,
- preview PNGs if compilation support is available,
- source/citation notes and bibliography artifacts.

### 12. Excel Workbook For Graph Explorer Review

Use this when you want a spreadsheet deliverable for manual review.

```bash
./mistralrs_skill_cli.py ../skills/xlsx \
  --file iter_0002.graphml \
  --max-tool-rounds 12 \
  --query "Create an Excel workbook summarizing this GraphML snapshot. Include sheets for graph summary, top bridge nodes, module themes, relation types, candidate hypotheses, and follow-up experiments. Use formulas or conditional formatting where useful. Save the workbook as an artifact."
```

Expected outputs:

- `.xlsx` workbook,
- possibly CSV/JSON intermediates.

### 13. Manuscript-Style Concept Note

Use this after a graph run when you want a written research artifact.

```bash
./mistralrs_skill_cli.py ../skills/scientific-writing \
  --file iter_0002.graphml \
  --max-tool-rounds 10 \
  --query "Write a concise scientific concept note from this GraphML ideation snapshot. Include an abstract, rationale, graph-derived evidence, three hypotheses, proposed experiments, expected outcomes, risks, and a short significance section. Keep it grounded in the attached graph and mark speculative claims clearly."
```

Expected outputs:

- polished concept note in `answer.md`,
- optional markdown/doc artifacts if produced.

### 14. Materials-Oriented Follow-Up With pymatgen

Use this when a graph opportunity points toward crystal/materials structure
reasoning.

```bash
./mistralrs_skill_cli.py ../skills/pymatgen \
  --max-tool-rounds 10 \
  --query "Propose a computational materials workflow for testing whether graph-derived bridge concepts around self-healing, lattice defects, and transport can be mapped to candidate inorganic or hybrid materials. Include required input files, pymatgen analyses, filters, and output tables to generate."
```

Expected outputs:

- workflow design,
- candidate analysis plan,
- artifact checklist.

### 15. Molecular Candidate Screen With RDKit

Use this when the graph suggests molecular motifs or polymer chemistry.

```bash
./mistralrs_skill_cli.py ../skills/rdkit \
  --max-tool-rounds 10 \
  --query "Design a small RDKit screening workflow for self-healing polymer motifs. Include example SMILES for 8 candidate motifs, compute descriptors that matter for dynamic bonding and processability, cluster by fingerprint similarity, and save a CSV plus a short interpretation."
```

Expected outputs:

- candidate descriptor table,
- clustering/similarity summary,
- CSV artifact if generated.

### 16. Reaction-Diffusion Science-Art Poster

Use this as a reliable 4B-model-friendly visual skill. The model only needs to
run a bundled renderer, so it avoids fragile HTML, JavaScript, or custom plotting
code.

```bash
./mistralrs_skill_cli.py ../skills/reaction-diffusion-poster \
  --max-tool-rounds 6 \
  --response-timeout 900 \
  --require-tool \
  --query "Use reaction-diffusion-poster to create a square science-art social post titled 'Morphogenesis From Local Rules'. Use preset labyrinth, palette magma-cyan, seed 42. Save PNG posters, data, README, parameters, and caption. Verify files with find before answering."
```

Expected outputs:

- `reaction_diffusion_poster.png` 4:5 social image,
- `reaction_diffusion_poster_square.png` square social image,
- `reaction_diffusion_data.npz`,
- `parameters.json`,
- `caption.txt`,
- `README.md`.

### 17. One-Slide PPTX Visual Summary

Use this when you want to test the `pptx` skill directly with a narrow,
well-defined deliverable. It avoids web search and asks for exactly one slide.

```bash
./mistralrs_skill_cli.py ../skills/pptx \
  --max-tool-rounds 10 \
  --response-timeout 1200 \
  --require-tool \
  --query "Use the pptx skill to create exactly one PowerPoint slide named materiomics_one_slide_summary.pptx. The slide title is 'Materiomics: Hierarchy to Function'. Make a clean 16:9 slide with four labeled stages: nanoscale motif, fiber/interface, architecture, function/property. Use simple editable shapes, arrows, and short text only. Save the .pptx in the requested skill output directory. If possible, also export a PNG preview. Before answering, run find on the output directory and only report success if the .pptx file exists. Final answer must list exact artifact paths."
```

Expected outputs:

- `materiomics_one_slide_summary.pptx`,
- optional PNG preview,
- exact paths in `answer.md` and `RESULTS.md`.

### 18. Morphogenesis Science-Art Postcard

Use this as a reliable container/shell visual skill for smaller local models.
The model only needs to run a bundled standard-library Python renderer, so it
does not depend on browser execution, JavaScript, Pillow, matplotlib, or npm.

```bash
./mistralrs_skill_cli.py ../skills/morphogenesis-postcard \
  --max-tool-rounds 6 \
  --response-timeout 900 \
  --require-tool \
  --query "Use morphogenesis-postcard to create a square science-art social post titled 'Stress Waves in a Growing Interface'. First write prompt-specific visual rule code as agent_rules.py in the requested skill output directory. Use this exact visual brief for the rule code and renderer: 'stress-wave interference in a growing hierarchical material interface with branching cracks, glowing defects, and nonlinear energy flow'. Then render with --rule-code pointing to agent_rules.py. Save the PNG, agent_rules.py, visual_rule_profile.json, parameters JSON, caption, and README. Before answering, run find on the output directory and list exact artifact paths."
```

Expected outputs:

- social-post PNG,
- `agent_rules.py` visual rule code,
- `visual_rule_profile.json`,
- `parameters.json`,
- `caption.txt`,
- `README.md`,
- prompt profile recorded in `parameters.json`,
- exact paths in `answer.md` and `RESULTS.md`.

Alternative prompt-driven graph-art example:

```bash
./mistralrs_skill_cli.py ../skills/morphogenesis-postcard \
  --max-tool-rounds 6 \
  --response-timeout 900 \
  --require-tool \
  --query "Use morphogenesis-postcard to create a square science-art PNG titled 'Bridge Field'. First write prompt-specific visual rule code as agent_rules.py in the requested skill output directory. Use this exact visual brief for the rule code and renderer: 'luminous graph of research ideas bridging distant modules, with nodes, links, hidden paths, and morphogenesis-like diffusion'. Then render with --rule-code pointing to agent_rules.py. Save PNG, agent_rules.py, visual_rule_profile.json, parameters JSON, caption, and README artifacts. Before answering, run find on the output directory and list exact artifact paths."
```

Direct local script smoke test:

```bash
python ../skills/morphogenesis-postcard/scripts/write_visual_rules.py \
  --out /tmp/morphogenesis-postcard-demo/agent_rules.py \
  --profile-json /tmp/morphogenesis-postcard-demo/visual_rule_profile.json \
  --prompt "stress-wave interference in a growing hierarchical material interface with branching cracks, glowing defects, and nonlinear energy flow" \
  --title "Stress Waves in a Growing Interface" \
  --subtitle "Prompt-written reaction-diffusion rules"

python ../skills/morphogenesis-postcard/scripts/make_morphogenesis_postcard.py \
  --out /tmp/morphogenesis-postcard-demo \
  --rule-code /tmp/morphogenesis-postcard-demo/agent_rules.py \
  --prompt "stress-wave interference in a growing hierarchical material interface with branching cracks, glowing defects, and nonlinear energy flow" \
  --title "Stress Waves in a Growing Interface" \
  --subtitle "Prompt-written reaction-diffusion rules" \
  --auto-style \
  --format square \
  --steps 120 \
  --size 720
```

Advanced direct local script smoke test with a prewritten custom rule file:

```bash
python ../skills/morphogenesis-postcard/scripts/make_morphogenesis_postcard.py \
  --out /tmp/morphogenesis-agent-rules-demo \
  --rule-code morphogenesis_agent_rules_example.py \
  --title "Agent-Written Morphogenesis" \
  --subtitle "Custom seed fields, tone rules, and overlay motifs" \
  --seed agent-rules-01 \
  --preset coral \
  --palette noir-neon \
  --format square \
  --steps 120 \
  --size 720
```

Other expressive recipes:

```bash
python ../skills/morphogenesis-postcard/scripts/write_visual_rules.py \
  --out /tmp/morphogenesis-specimen-demo/agent_rules.py \
  --profile-json /tmp/morphogenesis-specimen-demo/visual_rule_profile.json \
  --prompt "quiet museum specimen of cellular morphogenesis in ice-blue symmetry, like an archived biological crystal" \
  --title "Specimen of Emergent Order" \
  --subtitle "Symmetric Turing field, constrained growth"

python ../skills/morphogenesis-postcard/scripts/make_morphogenesis_postcard.py \
  --out /tmp/morphogenesis-specimen-demo \
  --rule-code /tmp/morphogenesis-specimen-demo/agent_rules.py \
  --prompt "quiet museum specimen of cellular morphogenesis in ice-blue symmetry, like an archived biological crystal" \
  --title "Specimen of Emergent Order" \
  --subtitle "Symmetric Turing field, constrained growth" \
  --auto-style \
  --format square \
  --steps 120 \
  --size 720
```

```bash
python ../skills/morphogenesis-postcard/scripts/write_visual_rules.py \
  --out /tmp/morphogenesis-triptych-demo/agent_rules.py \
  --profile-json /tmp/morphogenesis-triptych-demo/visual_rule_profile.json \
  --prompt "three-panel scientific triptych of branching biofilm fibers, cellular growth fronts, and nonlinear wave coupling" \
  --title "Three Views of a Reaction Field" \
  --subtitle "One seed, shifted sampling windows"

python ../skills/morphogenesis-postcard/scripts/make_morphogenesis_postcard.py \
  --out /tmp/morphogenesis-triptych-demo \
  --rule-code /tmp/morphogenesis-triptych-demo/agent_rules.py \
  --prompt "three-panel scientific triptych of branching biofilm fibers, cellular growth fronts, and nonlinear wave coupling" \
  --title "Three Views of a Reaction Field" \
  --subtitle "One seed, shifted sampling windows" \
  --auto-style \
  --format portrait \
  --steps 120 \
  --size 720
```

### 19. Fracture Mechanics Movie And Stress-Strain Curve

Use this when you want a serious simulation artifact that a smaller local model
can produce reliably by running a bundled script. It simulates a pre-cracked 2D
triangular lattice under Mode I or Mode II loading and saves a movie plus
stress-strain data.

```bash
./mistralrs_skill_cli.py ../skills/fracture-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 2400 \
  --require-tool \
  --query "Use fracture-mechanics to simulate a pre-cracked 2D lattice titled 'Brittle Mode I Fracture'. Use potential morse, mode I, orientation 90, nx 96, ny 36, max-atoms 5000, crack-length 0.34, temperature 0.003, strain-rate 0.0035, damping 0.30, dt 0.005, steps 10000, frames 48, dpi 120, movie-fps 16, bond-cutoff 1.35, break-stretch 1.65, color-by stress, and morse-a 7.0. Use fixed slab axes, fixed stress-strain axes, and fixed color scale across frames to avoid movie flicker. Save fracture_movie.gif, fracture_movie.html, stress_strain.png, final_lattice.png, stress_strain.csv, summary.json, parameters.json, and README. Verify files with find and report peak stress, peak strain, and dynamically broken bonds from summary.json."
```

Expected outputs:

- `fracture_movie.gif`,
- `fracture_movie.html`,
- `frames/frame_*.png`,
- `stress_strain.png`,
- `final_lattice.png`,
- `stress_strain.csv`,
- `summary.json`,
- `parameters.json`,
- `README.md`.

```bash
./mistralrs_skill_cli.py ../skills/fracture-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 3600 \
  --require-tool \
  --query "Use fracture-mechanics to simulate a pre-cracked 2D lattice titled 'Brittle Mode I Fracture'. Use potential morse, mode I, orientation 90, nx 200, ny 100, max-atoms 50000, crack-length 0.34, temperature 0.003, strain-rate 0.0005, damping 0.30, dt 0.005, steps 30000, frames 48, dpi 120, movie-fps 16, bond-cutoff 1.35, break-stretch 1.65, color-by stress, and morse-a 7.0. Use fixed slab axes, fixed stress-strain axes, and fixed color scale across frames to avoid movie flicker. Save fracture_movie.gif, fracture_movie.html, stress_strain.png, final_lattice.png, stress_strain.csv, summary.json, parameters.json, and README. Verify files with find and report peak stress, peak strain, and dynamically broken bonds from summary.json."
```

Mode II:
```bash
./mistralrs_skill_cli.py ../skills/fracture-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 3600 \
  --require-tool \
  --query "Use fracture-mechanics to simulate a pre-cracked 2D lattice titled 'Brittle Mode I Fracture'. Use potential morse, mode II, orientation 90, nx 200, ny 100, max-atoms 50000, crack-length 0.34, temperature 0.003, strain-rate 0.0005, damping 0.30, dt 0.005, steps 80000, frames 48, dpi 120, movie-fps 16, bond-cutoff 1.35, break-stretch 1.65, color-by stress, and morse-a 7.0. Use fixed slab axes, fixed stress-strain axes, and fixed color scale across frames to avoid movie flicker. Save fracture_movie.gif, fracture_movie.html, stress_strain.png, final_lattice.png, stress_strain.csv, summary.json, parameters.json, and README. Verify files with find and report peak stress, peak strain, and dynamically broken bonds from summary.json."
```


Direct local full simulation:

```bash
python ../skills/fracture-mechanics/scripts/run_fracture_sim.py \
  --out /tmp/fracture-mechanics-demo \
  --title "Brittle Mode I Fracture" \
  --potential morse \
  --mode I \
  --orientation 90 \
  --nx 96 \
  --ny 36 \
  --max-atoms 5000 \
  --crack-length 0.34 \
  --temperature 0.003 \
  --strain-rate 0.0035 \
  --damping 0.30 \
  --dt 0.005 \
  --steps 10000 \
  --frames 48 \
  --dpi 120 \
  --movie-fps 16 \
  --bond-cutoff 1.35 \
  --break-stretch 1.65 \
  --morse-a 7.0 \
  --color-by stress
```

### 20. Beam Mechanics Plots And Deformation GIFs

Use this for a compact, reliable mechanics artifact. The skill is dimensionless:
no units, no YAML, no JSON input schema. For local 4B models, start without
`--require-tool`; add it only if your tool-calling setup is stable.

Baseline simply supported beam:

```bash
./mistralrs_skill_cli.py ../skills/beam-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 1200 \
  --query "Use beam-mechanics to analyze a dimensionless simply supported beam of length 10 with a downward point load of magnitude 1 at midspan. Use the bundled simple_beam_lab.py script. Save all plots, the deformation GIF, field data CSV, reactions JSON, summary JSON, manifest JSON, and README. Verify the output directory with find before answering. Final answer must report support reactions, maximum absolute deflection, maximum absolute bending moment, maximum absolute shear, vertical equilibrium residual, and exact artifact paths."
```

Overhanging beam with mixed loading:

```bash
./mistralrs_skill_cli.py ../skills/beam-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 1200 \
  --query "Use beam-mechanics to analyze a dimensionless overhanging beam of length 10. Use the overhang preset, apply a downward point load of magnitude 1 at the free right end x=10, and a downward uniform load of magnitude 0.1 across the full span from x=0 to x=10. Use the bundled simple_beam_lab.py script. Save structure, deformed shape, deflection, shear, moment, dashboard, deformation GIF, field_data.csv, reactions.json, summary.json, manifest.json, and README. Verify files with find and report reactions, max deflection, max moment, max shear, equilibrium residual, and exact artifact paths."
```

Fixed-fixed beam with a local distributed load:

```bash
./mistralrs_skill_cli.py ../skills/beam-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 1200 \
  --query "Use beam-mechanics to analyze a dimensionless fixed-fixed beam of length 12. Apply a downward uniform load of magnitude 0.2 only across the middle third from x=4 to x=8, plus a clockwise point moment of magnitude 0.5 at x=6. Use the bundled simple_beam_lab.py script. Save plots, deformation GIF, CSV data, reactions JSON, summary JSON, manifest JSON, and README. Verify the output directory with find before answering. Final answer must list exact artifact paths and summarize support reactions, maximum absolute deflection, maximum absolute moment, maximum absolute shear, and vertical equilibrium residual."
```

Custom beam with several supports and a spring:

```bash
./mistralrs_skill_cli.py ../skills/beam-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 1200 \
  --query "Use beam-mechanics to analyze a dimensionless custom beam of length 12. Use a pin at the left end, a roller at the right end, an additional roller support at x=8, and a vertical spring at x=6 with stiffness 20. Apply downward point loads of magnitude 1 at x=4 and 0.6 at x=9, plus a downward uniform load of magnitude 0.05 from x=2 to x=10. Use the bundled simple_beam_lab.py script. Save plots, deformation GIF, CSV data, reactions JSON, summary JSON, manifest JSON, and README. Verify the output directory with find before answering. Final answer must list exact artifact paths and summarize the reactions, max deflection, max moment, max shear, and equilibrium residual."
```

Visually clean point-load cases:

```bash
./mistralrs_skill_cli.py ../skills/beam-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 1200 \
  --query "Use beam-mechanics to analyze a dimensionless cantilever beam of length 9 titled 'Cantilever Tip Load'. Use a fixed support at the left end and a free right end. Apply only one downward point load of magnitude 1 at the free end x=9. Use the bundled simple_beam_lab.py script. Save structure, deformed shape, deflection, shear, moment, dashboard, deformation GIF, field_data.csv, reactions.json, summary.json, manifest.json, and README. Verify files with find. Final answer must report the fixed-end reactions, maximum absolute deflection, maximum absolute moment, maximum absolute shear, vertical equilibrium residual, and exact artifact paths."
```

```bash
./mistralrs_skill_cli.py ../skills/beam-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 1200 \
  --query "Use beam-mechanics to analyze a dimensionless fixed-fixed beam of length 10 titled 'Locked Beam Center Load'. Use fixed supports at both ends. Apply only one downward point load of magnitude 1 at midspan x=5. Use the bundled simple_beam_lab.py script. Save all plots, deformation GIF, field_data.csv, reactions.json, summary.json, manifest.json, and README. Verify files with find. Final answer must report support reactions, maximum absolute deflection, maximum absolute bending moment, maximum absolute shear, vertical equilibrium residual, and exact artifact paths."
```

```bash
./mistralrs_skill_cli.py ../skills/beam-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 1200 \
  --query "Use beam-mechanics to analyze a dimensionless simply supported beam of length 12 titled 'Twin Point Loads'. Use a pin at x=0 and a roller at x=12. Apply only two downward point loads: magnitude 0.8 at x=4 and magnitude 0.8 at x=8. Use the bundled simple_beam_lab.py script. Save structure, deformed shape, deflection, shear, moment, dashboard, deformation GIF, field_data.csv, reactions.json, summary.json, manifest.json, and README. Verify files with find. Final answer must report reactions, maximum absolute deflection, maximum absolute moment, maximum absolute shear, equilibrium residual, and exact artifact paths."
```

```bash
./mistralrs_skill_cli.py ../skills/beam-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 1200 \
  --query "Use beam-mechanics to analyze a dimensionless custom beam of length 12 titled 'Elastic Midspan Prop'. Use a pin at x=0, a roller at x=12, and a vertical spring at x=6 with stiffness 12. Apply only one downward point load of magnitude 1.2 at x=6. Use the bundled simple_beam_lab.py script. Save all plots, deformation GIF, field_data.csv, reactions.json, summary.json, manifest.json, and README. Verify files with find. Final answer must report support and spring reactions, maximum absolute deflection, maximum absolute moment, maximum absolute shear, vertical equilibrium residual, and exact artifact paths."
```

Visually clean distributed-load cases:

```bash
./mistralrs_skill_cli.py ../skills/beam-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 1200 \
  --query "Use beam-mechanics to analyze a dimensionless simply supported beam of length 14 titled 'Central Patch Load'. Use a pin at x=0 and a roller at x=14. Apply only a downward uniform load of magnitude 0.18 over the central patch from x=5 to x=9. Use the bundled simple_beam_lab.py script. Save structure, deformed shape, deflection, shear, moment, dashboard, deformation GIF, field_data.csv, reactions.json, summary.json, manifest.json, and README. Verify files with find. Final answer must report reactions, maximum absolute deflection, maximum absolute moment, maximum absolute shear, equilibrium residual, and exact artifact paths."
```

```bash
./mistralrs_skill_cli.py ../skills/beam-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 1200 \
  --query "Use beam-mechanics to analyze a dimensionless cantilever beam of length 10 titled 'Wind Loaded Cantilever'. Use a fixed support at the left end and a free right end. Apply only a downward uniform load of magnitude 0.12 over the full span from x=0 to x=10. Use the bundled simple_beam_lab.py script. Save all plots, deformation GIF, field_data.csv, reactions.json, summary.json, manifest.json, and README. Verify files with find. Final answer must report fixed-end reactions, maximum absolute deflection, maximum absolute moment, maximum absolute shear, vertical equilibrium residual, and exact artifact paths."
```

```bash
./mistralrs_skill_cli.py ../skills/beam-mechanics \
  --max-tool-rounds 8 \
  --response-timeout 1200 \
  --query "Use beam-mechanics to analyze a dimensionless custom beam of length 16 titled 'Suspended Span Patch Load'. Use a pin at x=0, a roller at x=16, and an internal roller support at x=10. Apply only a downward uniform load of magnitude 0.10 from x=3 to x=7. Use the bundled simple_beam_lab.py script. Save structure, deformed shape, deflection, shear, moment, dashboard, deformation GIF, field_data.csv, reactions.json, summary.json, manifest.json, and README. Verify files with find. Final answer must report support reactions, maximum absolute deflection, maximum absolute moment, maximum absolute shear, vertical equilibrium residual, and exact artifact paths."
```

Expected outputs:

- `structure.png`,
- `deformed_shape.png`,
- `deflection.png`,
- `moment.png`,
- `shear.png`,
- `dashboard.png`,
- `deformation.gif`,
- `frames/frame_*.png`,
- `field_data.csv`,
- `summary.json`,
- `reactions.json`,
- `manifest.json`,
- `README.md`.

### 21. Hierarchical Topology Optimization Examples

Use this when you want a more design-like artifact: density fields, resolved
boundary conditions, convergence history, and optional STL geometry. The skill
is intentionally lightweight and avoids gyroids, hole lattices, marching cubes,
and heavy boolean geometry.

Cantilever with midpoint load, profile STL:

```bash
./mistralrs_skill_cli.py ../skills/hierarchical-topopt \
  --max-tool-rounds 10 \
  --response-timeout 1800 \
  --query "Use hierarchical-topopt to optimize a 2D cantilever. Use nelx 100, nely 35, volfrac 0.50, penal 4.0, rmin 4.0, density filter, maxiter 220, and bc-preset cantilever-mid-down. Use mesh-mode profile-stl, dens-cut 0.30, max-height 10, and profile-origin bottom. Save density.png, density_resized.png, density.npy, density.csv, optimization_history.csv, boundary_conditions.json, boundary_conditions_resolved.json, bc_preview.png, convergence.png, result_profile.stl, summary.json, parameters.json, and README. Verify files with find and report compliance, final volume fraction, fixed DOFs, loaded DOFs, total force, and exact artifact paths."
```

Bridge with pin/roller supports and middle-20-percent top load:

```bash
./mistralrs_skill_cli.py ../skills/hierarchical-topopt \
  --max-tool-rounds 12 \
  --response-timeout 2400 \
  --query "Use hierarchical-topopt to optimize a bridge-like 2D structure. Use nelx 120, nely 40, volfrac 0.45, penal 4.0, rmin 4.2, density filter, and 250 iterations. Define custom boundary conditions with a pin support at the lower-left corner, a roller support at the lower-right corner, and a downward distributed total load across the middle 20 percent of the top edge. Write boundary_conditions.json using edge_fraction start 0.4 end 0.6 for the top-edge load, preview it as bc_preview.png, then run the optimization with mesh-mode profile-stl, dens-cut 0.30, max-height 10, and profile-origin center. Save density images/data, optimization history, boundary_conditions.json, boundary_conditions_resolved.json, bc_preview.png, result_profile.stl, summary.json, parameters.json, and README. Verify files with find and report compliance, final volume fraction, fixed DOFs, loaded DOFs, total force, and exact artifact paths."
```

Shear strip:

```bash
./mistralrs_skill_cli.py ../skills/hierarchical-topopt \
  --max-tool-rounds 10 \
  --response-timeout 1800 \
  --query "Use hierarchical-topopt to optimize a compact shear strip. Use nelx 90, nely 30, volfrac 0.55, penal 3.5, rmin 3.5, sensitivity filter, maxiter 180, and bc-preset shear-strip. Use mesh-mode density-only for a fast run. Save density.png, density_resized.png, density.npy, density.csv, optimization_history.csv, boundary_conditions.json, boundary_conditions_resolved.json, bc_preview.png, convergence.png, summary.json, parameters.json, and README. Verify files with find and report compliance, final volume fraction, fixed DOFs, loaded DOFs, total force, and exact artifact paths."
```

Fixed-fixed beam with centered top patch load and symmetric profile STL:

```bash
./mistralrs_skill_cli.py ../skills/hierarchical-topopt \
  --max-tool-rounds 12 \
  --response-timeout 2400 \
  --query "Use hierarchical-topopt to optimize a fixed-fixed beam-like plate with a localized downward top load. Use nelx 110, nely 32, volfrac 0.48, penal 4.0, rmin 4.0, density filter, and maxiter 220. Use the fixed-fixed-top-mid-20-down preset if available; otherwise write boundary_conditions.json with fixed left and right edges and an edge_fraction top load from start 0.4 to end 0.6 with distribution total. Run with mesh-mode profile-stl, dens-cut 0.32, max-height 8, and profile-origin center. Save all density plots/data, boundary-condition files, convergence plot, STL, summary.json, parameters.json, and README. Verify files with find and report compliance, final volume fraction, fixed DOFs, loaded DOFs, total force, and exact artifact paths."
```

Expected outputs:

- `density.png`,
- `density_resized.png`,
- `density.npy`,
- `density_resized.npy`,
- `density.csv`,
- `optimization_history.csv`,
- `boundary_conditions.json`,
- `boundary_conditions_resolved.json`,
- `bc_preview.png`,
- `convergence.png`,
- `summary.json`,
- `parameters.json`,
- `README.md`,
- optional `result_profile.stl`, `result_flat.stl`, or multimaterial STL files depending on `--mesh-mode`.

## Troubleshooting

`POST /v1/skills` returns 404:

- the server was probably not started with `--agent` or `--enable-shell`,
- or the request is reaching a different server on port 1234.

`POST /v1/files` returns 404:

- the running mistral.rs build may not include the file-input API,
- use `--file-mode inline` as a fallback.

The model answers but no artifacts appear:

- inspect `answer.md` and `response.json`,
- rerun with a more explicit request to save files,
- start the server with `--shell-workdir` so shell outputs are visible on disk,
- increase `--max-tool-rounds`.

The CLI rejects a skill because names do not match:

- make the directory name match `SKILL.md` `name:`,
- or change the `name:` field to match the directory.
