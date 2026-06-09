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

    def on_step(rec):
        print(f"  iter {rec['iter']:>3}  +{len(rec['new_nodes'])} nodes  "
              f"({rec['n_nodes']}n/{rec['n_edges']}e)  q={rec['question'][:70]}")

    store, transcript, stats = run_loop(clients, cfg, args.topic, strat, on_step=on_step)
    m = all_metrics(store, stats)

    nx.write_graphml(store.G, os.path.join(args.out, "graph.graphml"))
    with open(os.path.join(args.out, "transcript.jsonl"), "w") as f:
        for r in transcript:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "growth.csv"), "w") as f:
        f.write("iter,n_nodes,n_edges,new_nodes,tokens,cum_tokens,diversity\n")
        for r in transcript:
            f.write(f"{r['iter']},{r['n_nodes']},{r['n_edges']},{len(r['new_nodes'])},"
                    f"{r['tokens']},{r['cum_tokens']},{r['diversity']:.4f}\n")
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
