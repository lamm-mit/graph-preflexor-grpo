#!/usr/bin/env python
"""Graph-PRefLexOR ideation loop CLI.

Seed a topic, expand it into an accumulating knowledge graph via the chosen
strategy + context mode under a compute budget, then dump the graph, transcript,
and ideation/creativity metrics.

Example:
    python ideate.py --topic "self-healing biopolymer composites" \
        --strategy frontier --context-mode branched --budget-calls 40 --out runs/exp1
"""
import argparse
import json
import os

import networkx as nx
import yaml

from clients import Clients
from strategies import get_strategy
from loop import run as run_loop
from metrics import all_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--topic", required=True)
    p.add_argument("--strategy", help="frontier|node|answer|edge|novelty|mixed")
    p.add_argument("--context-mode", dest="context_mode", help="fresh|chained|branched")
    p.add_argument("--budget-calls", type=int)
    p.add_argument("--budget-tokens", type=int)
    p.add_argument("--max-iters", type=int)
    p.add_argument("--fanout", type=int)
    p.add_argument("--dedup-threshold", type=float)
    p.add_argument("--out", default="runs/run")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # CLI overrides
    for k in ("strategy", "context_mode", "fanout", "dedup_threshold"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.budget_calls:  cfg["budget"]["generator_calls"] = args.budget_calls
    if args.budget_tokens: cfg["budget"]["max_tokens"] = args.budget_tokens
    if args.max_iters:     cfg["budget"]["max_iters"] = args.max_iters

    os.makedirs(args.out, exist_ok=True)
    clients = Clients(cfg)
    strat = get_strategy(cfg["strategy"])
    print(f"[ideate] topic={args.topic!r}  strategy={cfg['strategy']}  "
          f"context={cfg['context_mode']}  budget={cfg['budget']}")

    # Incremental writers: transcript + growth grow on disk as the loop runs, so you can
    # tail them immediately (the graph + summary are written at the end).
    tf = open(os.path.join(args.out, "transcript.jsonl"), "w")
    gf = open(os.path.join(args.out, "growth.csv"), "w")
    gf.write("iter,depth,n_nodes,n_edges,new_nodes,tokens,cum_tokens,diversity\n"); gf.flush()
    print(f"[ideate] writing to {os.path.abspath(args.out)}/  (transcript.jsonl, growth.csv live)")

    graphml_path = os.path.join(args.out, "graph.graphml")

    def on_step(rec, store):
        tf.write(json.dumps(rec) + "\n"); tf.flush()
        gf.write(f"{rec['iter']},{rec['depth']},{rec['n_nodes']},{rec['n_edges']},"
                 f"{len(rec['new_nodes'])},{rec['tokens']},{rec['cum_tokens']},"
                 f"{rec['diversity']:.4f}\n"); gf.flush()
        nx.write_graphml(store.G, graphml_path)        # checkpoint each step (usable mid-run)
        print(f"  iter {rec['iter']:>3}  +{len(rec['new_nodes'])} nodes  "
              f"({rec['n_nodes']}n/{rec['n_edges']}e)  q={rec['question'][:70]}")

    try:
        store, transcript, stats = run_loop(clients, cfg, args.topic, strat, on_step=on_step)
    finally:
        tf.close(); gf.close()

    m = all_metrics(store, stats)
    nx.write_graphml(store.G, os.path.join(args.out, "graph.graphml"))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"topic": args.topic,
                   "config": {k: cfg.get(k) for k in ("strategy", "context_mode",
                                                      "dedup_threshold", "fanout")},
                   "stats": stats, "metrics": m}, f, indent=2)

    print(f"[ideate] done: {m['nodes']} ideas, {m['edges']} links | "
          f"diversity={m['mean_pairwise_distance']:.3f}  ideas/call={m['ideas_per_call']:.2f}")
    print(f"[ideate] wrote {args.out}/ (graph.graphml, transcript.jsonl, summary.json, growth.csv)")


if __name__ == "__main__":
    main()
