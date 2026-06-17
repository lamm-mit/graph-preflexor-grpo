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
