# Experiment 2: Backtracking Analysis Scripts

Run in order:

| Step | Script | Description |
|------|--------|-------------|
| 0 | `step0_generate_graph_stages.py` | Run Graph-PRefLexOR pipeline on 100 questions → `graph_8b_data_eval_100.jsonl` |
| 1 | `step1_full_pipeline.py` | Generate Qwen thinking (2A) → text similarity (2B) → hidden states (2C) → probes (2G.1) → logit lens (2G.2) |
| 2 | `step2_activation_patching.py` | Representation-space activation patching on 50 questions |
| 3 | `step3_mentor_plots.py` | Generate mentor-requested plots (A1, A2, B, B2, C1, C2) |
| 4 | `step4_publication_plots.py` | Regenerate all plots in publication style (PDF + PNG) |

All scripts use `Qwen/Qwen3-8B` and `BAAI/bge-base-en-v1.5`.
Submit GPU jobs with the `.slurm` files in `exp1_stage_deletion/`.
