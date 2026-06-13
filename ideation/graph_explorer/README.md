# Graph Explorer

Interactive browser explorer for Graph-PRefLexOR `graph.graphml` files.

It runs as a local Python server and a Three.js browser app. The app can:

- upload any GraphML file from a Graph-PRefLexOR run
- load an existing run directory such as `runs/exp_leap`
- start a new `ideate.py` run from the browser and load the resulting graph
- search concepts and inspect node provenance
- focus selected neighborhoods by hop depth
- surface shortest paths between selected concepts
- show top hubs and structural components
- ask an OpenAI-compatible or local Hugging Face model about the selected graph context

## Run

From `ideation/`:

```bash
python graph_explorer/server.py --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

The browser imports Three.js from `unpkg.com`; the page will warn you if that module cannot load.

Optionally preload a run:

```bash
python graph_explorer/server.py --run runs/exp_leap --port 8765
```

## LLM Backends

Use **OpenAI-compatible** for most cases:

- vLLM: `http://localhost:8000/v1`
- mistral.rs / llama.cpp OpenAI adapters
- commercial OpenAI-compatible APIs

Set the model field to the exact served model name, for example:

```text
meta-llama/Llama-3.2-3B-Instruct
```

Use **Local HF** only when this machine has `transformers`, `torch`, and model weights available.

The graph question panel sends only a focused packet: selected nodes, an optional search query,
their hop neighborhood, important edges, and short selected paths. It does not send the whole graph
unless the focused graph is small enough to fit the chosen context limits.
