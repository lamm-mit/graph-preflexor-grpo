# Graph Explorer

Interactive browser explorer for Graph-PRefLexOR `graph.graphml` files.

It runs as a local Python server and a React/Sigma.js browser app. The app can:

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
graph_explorer/launch.sh --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

The launcher installs the frontend dependencies the first time, builds the React app, then starts
the Python API/static server.

Optionally preload a run:

```bash
graph_explorer/launch.sh --run runs/exp_leap --port 8765
```

For React development with hot reload:

```bash
graph_explorer/launch.sh --dev --run runs/exp_leap --port 8765 --vite-port 5177
```

## LLM Backends

Use **OpenAI-compatible** for most cases:

- vLLM: `http://localhost:8000/v1`
- mistral.rs / llama.cpp OpenAI adapters
- commercial OpenAI-compatible APIs

Set the model field to the exact served model name, for example:

```text
google/gemma-4-E4B
```

Use **Local HF** only when this machine has `transformers`, `torch`, and model weights available.

The graph question panel sends only a focused packet: selected nodes, an optional search query,
their hop neighborhood, important edges, and short selected paths. It does not send the whole graph
unless the focused graph is small enough to fit the chosen context limits.

## Engineering Notes

Refactor backlog captured 2026-06-14 while hardening the explorer:

- `server.py` is currently too large for sustained feature work. Split it into graph I/O, metrics/search, embeddings/cache, model calls, run jobs, profile jobs, and HTTP routing.
- `frontend/src/main.tsx` is also too large. Split graph canvases, chat, focus tools, model settings, and workspace layout into separate feature modules.
- Move expensive embedding cache work fully outside the global server lock. The server should snapshot graph state under lock, release it for hashing/cache I/O, then reacquire only to publish status.
- Replace the long manual API `if/elif` router with small route handlers plus typed request coercion to avoid fragile `int(...)` parsing and improve error messages.
- Scope persisted chat state by graph/run/session instead of using one global browser storage key.
- Replace the hand-rolled Markdown report renderer with a maintained Markdown/GFM renderer before reports become a primary surface.
- Add focused tests for GraphML candidate selection, embedding cache round-trips, path resolution, and a React smoke test.
