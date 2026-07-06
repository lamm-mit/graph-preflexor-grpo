# Graph-to-Answer Circuit Tracing

Mechanism-level activation-patching analysis for Graph-PRefLexOR.

## Scripts (run in order)

| Script | Description |
|--------|-------------|
| `circuit_tracing_pipeline.py` | Exp 1–4: corruption recovery, activation patching, path patching, stage ablation |
| `circuit_exp2_redo.py` | Exp 2 redo: mechanism-F1 recovery heatmap with proper metric |
| `circuit_extra_experiments.py` | V1–V4: edge corruption, control patching, do-no-harm, extractor validation |

## Key Results

| Finding | Value |
|---------|-------|
| Directed edge F1 under wrong-graph corruption | 0.000 |
| Mechanism-F1 recovery at layer 36 (synthesis stage) | 0.70 |
| Target vs random-region patch: Δ | +0.365 [+0.245, +0.475] *** |
| Late vs control layers bootstrap (wrong-graph) | +0.290 [+0.233, +0.346] p<0.001 |
| Extractor precision / recall / F1 | 0.689 / 0.515 / 0.567 |

## Models
- Generator: `Qwen/Qwen3-8B`
- Embeddings: `BAAI/bge-base-en-v1.5`
- Platform: NCSA DeltaAI GH200 (40 examples)

## Plots
See `plots/` for all figures used in the ACM paper.
