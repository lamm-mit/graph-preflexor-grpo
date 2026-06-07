# Graph-PRefLexOR: Backtracking Analysis

This repository contains code and results for analyzing how Qwen3-8B's final answers 
relate to its own thinking trace versus Graph-PRefLexOR's structured reasoning stages.

## Structure
## Key Results (n=100 questions)

- Qwen's answer backtracks to Graph-PRefLexOR's answer in **46/100** cases
- Qwen's answer backtracks to its own thinking in only **16/100** cases
- Thinking-answer divergence peaks at **layers 7–10** and **layer 36**
- Linear probe AUROC = 1.0 from layer 5 onward (thinking vs answer fully separable)

## Models
- Generator: `Qwen/Qwen3-8B`
- Embeddings: `BAAI/bge-base-en-v1.5`
- Platform: NCSA DeltaAI (GH200 GPUs)
