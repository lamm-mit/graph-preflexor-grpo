#!/usr/bin/env python
"""Per-task Graph-PRefLexOR benchmark.

This benchmark tests the expensive test-time-compute claim directly:

  A. Llama single-shot: answer each benchmark task directly, with no graph.
  C. Graph-PRefLexOR short run: for each task, build a small ideation graph with
     ideate.py, mine insights, synthesize an answer from that graph, then judge
     graph answer vs baseline answer pairwise.

The runner is resumable. Existing task graphs / insights / answers are reused
unless --force is passed.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import compare


DEFAULT_ANSWER_TASK = (
    "Answer the benchmark task directly. Give the strongest concrete hypothesis, design, "
    "or experiment requested by the task; explain the mechanism and include a falsifiable "
    "prediction or decisive test."
)


def _read_tasks(path, limit=None):
    tasks = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
    return tasks[:limit] if limit else tasks


def _slug(i, task):
    words = []
    for w in task.lower().replace("/", " ").replace("-", " ").split():
        w = "".join(ch for ch in w if ch.isalnum())
        if w:
            words.append(w)
        if len(words) >= 6:
            break
    return f"{i:03d}_" + ("_".join(words) or "task")


def _cmd_str(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _run(cmd, *, cwd, dry_run=False):
    print("+ " + _cmd_str(cmd), flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def _synthesize_cmd(args, *, topic=None, run=None, out, no_insights=False):
    cmd = [
        "python", "synthesize.py",
        "--backend", args.backend,
        "--model", args.model,
        "--out", out,
        "--task", args.answer_task,
        "--temperature", str(args.temperature),
        "--max-tokens", str(args.max_tokens),
    ]
    if run:
        cmd += ["--run", run, "--max-leads", str(args.max_leads)]
    if topic:
        cmd += ["--topic", topic]
    if no_insights:
        cmd.append("--no-insights")
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    if args.device:
        cmd += ["--device", args.device]
    if args.dtype:
        cmd += ["--dtype", args.dtype]
    return cmd


def _maybe_exists(path):
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def _write_task_file(path, tasks):
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t + "\n")


def run(args):
    here = Path(__file__).resolve().parent
    out = Path(args.out).resolve()
    runs_dir = out / "task_runs"
    baseline_dir = out / "answers" / "baseline"
    graph_dir = out / "answers" / "graph"
    bench_dir = out / "benchmark"
    for d in (runs_dir, baseline_dir, graph_dir, bench_dir):
        d.mkdir(parents=True, exist_ok=True)

    tasks = _read_tasks(args.tasks, args.limit)
    tasks_file = out / "tasks.txt"
    _write_task_file(tasks_file, tasks)

    manifest = {
        "mode": "task_graph_benchmark",
        "tasks_file": str(Path(args.tasks).resolve()),
        "out": str(out),
        "n_tasks": len(tasks),
        "ideate": {
            "strategy": args.strategy,
            "context_mode": args.context_mode,
            "budget_calls": args.budget_calls,
            "budget_tokens": args.budget_tokens,
            "max_iters": args.max_iters,
            "fanout": args.fanout,
            "dedup_threshold": args.dedup_threshold,
            "config": args.config,
        },
        "synthesize": {
            "backend": args.backend,
            "model": args.model,
            "base_url": args.base_url,
            "answer_task": args.answer_task,
            "max_leads": args.max_leads,
        },
        "tasks": [],
    }

    for i, task in enumerate(tasks):
        name = _slug(i, task)
        task_run = runs_dir / name
        base_answer = baseline_dir / f"{i:03d}.md"
        graph_answer = graph_dir / f"{i:03d}.md"
        print(f"\n[task_graph_benchmark] task {i + 1}/{len(tasks)}: {task}", flush=True)

        ideate_cmd = [
            "python", "ideate.py",
            "--topic", task,
            "--strategy", args.strategy,
            "--budget-calls", str(args.budget_calls),
            "--max-iters", str(args.max_iters),
            "--out", str(task_run),
        ]
        if args.config:
            ideate_cmd += ["--config", args.config]
        if args.context_mode:
            ideate_cmd += ["--context-mode", args.context_mode]
        if args.budget_tokens:
            ideate_cmd += ["--budget-tokens", str(args.budget_tokens)]
        if args.fanout is not None:
            ideate_cmd += ["--fanout", str(args.fanout)]
        if args.dedup_threshold is not None:
            ideate_cmd += ["--dedup-threshold", str(args.dedup_threshold)]

        graphml = task_run / "graph.graphml"
        if args.force or not _maybe_exists(graphml):
            _run(ideate_cmd, cwd=here, dry_run=args.dry_run)
        else:
            print(f"[task_graph_benchmark] reusing {graphml}", flush=True)

        insights_json = task_run / "insights.json"
        insights_cmd = ["python", "insights.py", "--run", str(task_run), "--top", str(args.insights_top)]
        if args.force or not _maybe_exists(insights_json):
            _run(insights_cmd, cwd=here, dry_run=args.dry_run)
        else:
            print(f"[task_graph_benchmark] reusing {insights_json}", flush=True)

        if args.make_run_figures:
            fig_dir = task_run / "figures"
            _run(["python", "plot_ideation.py", "--runs", str(task_run), "--labels", name,
                  "--out", str(fig_dir / "graph")], cwd=here, dry_run=args.dry_run)
            _run(["python", "novelty.py", "--run", str(task_run), "--out", str(fig_dir / "novelty")],
                 cwd=here, dry_run=args.dry_run)

        if args.force or not _maybe_exists(base_answer):
            _run(_synthesize_cmd(args, topic=task, out=str(base_answer), no_insights=True),
                 cwd=here, dry_run=args.dry_run)
        else:
            print(f"[task_graph_benchmark] reusing {base_answer}", flush=True)

        if args.force or not _maybe_exists(graph_answer):
            _run(_synthesize_cmd(args, run=str(task_run), out=str(graph_answer)),
                 cwd=here, dry_run=args.dry_run)
        else:
            print(f"[task_graph_benchmark] reusing {graph_answer}", flush=True)

        manifest["tasks"].append({
            "index": i,
            "task": task,
            "run": str(task_run),
            "baseline_answer": str(base_answer),
            "graph_answer": str(graph_answer),
            "commands": {
                "ideate": _cmd_str(ideate_cmd),
                "insights": _cmd_str(insights_cmd),
            },
        })
        with open(out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if args.no_judge or args.dry_run:
        print(f"\n[task_graph_benchmark] wrote answers under {out}", flush=True)
        print("[task_graph_benchmark] skipping pairwise judge", flush=True)
        return

    pairwise_out = str(bench_dir / "pairwise")
    print("\n[task_graph_benchmark] pairwise judging graph answers vs baseline...", flush=True)
    compare.run_pairwise(SimpleNamespace(
        tasks=str(tasks_file),
        system=str(graph_dir),
        baseline=str(baseline_dir),
        out=pairwise_out,
        seed=args.seed,
        jm=args.judge_model,
        jbu=args.judge_base_url,
        jak=args.judge_api_key,
        judge_effort=args.judge_effort,
    ))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", required=True, help="benchmark task file, one task per line")
    p.add_argument("--out", default="runs/task_graph_benchmark", help="benchmark output directory")
    p.add_argument("--limit", type=int, default=None, help="optional first-N task cap for smoke tests")
    p.add_argument("--force", action="store_true", help="rebuild existing task runs and answers")
    p.add_argument("--dry-run", action="store_true", help="print commands without executing them")

    # Ideation run per task.
    p.add_argument("--config", default=None, help="config.yaml passed to ideate.py")
    p.add_argument("--strategy", default="frontier",
                   help="ideate.py strategy: frontier|node|answer|edge|novelty|leap|converse|mixed")
    p.add_argument("--context-mode", dest="context_mode", default=None,
                   help="optional ideate.py context mode: fresh|chained|branched")
    p.add_argument("--budget-calls", type=int, default=50)
    p.add_argument("--budget-tokens", type=int, default=None)
    p.add_argument("--max-iters", dest="max_iters", type=int, default=50)
    p.add_argument("--fanout", type=int, default=None)
    p.add_argument("--dedup-threshold", dest="dedup_threshold", type=float, default=None)
    p.add_argument("--insights-top", dest="insights_top", type=int, default=12)
    p.add_argument("--make-run-figures", action="store_true",
                   help="also run plot_ideation.py and novelty.py for every task graph")

    # Synthesis model for both arms.
    p.add_argument("--backend", choices=["openai", "hf"], default="openai")
    p.add_argument("--model", required=True, help="synthesis model id")
    p.add_argument("--base-url", dest="base_url", default=None,
                   help="[openai] OpenAI-compatible generator endpoint")
    p.add_argument("--api-key", dest="api_key", default=None)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=1200)
    p.add_argument("--device", default=None, help="[hf] device_map")
    p.add_argument("--dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"],
                   help="[hf] torch dtype")
    p.add_argument("--max-leads", dest="max_leads", type=int, default=10,
                   help="max mined graph leads passed to synthesize.py for graph answers")
    p.add_argument("--answer-task", dest="answer_task", default=DEFAULT_ANSWER_TASK,
                   help="shared synthesis instruction appended to each benchmark task")

    # Judge.
    p.add_argument("--no-judge", action="store_true", help="generate answers but skip pairwise scoring")
    p.add_argument("--judge-model", dest="judge_model", default="gpt-5.5")
    p.add_argument("--judge-base-url", dest="judge_base_url", default=None)
    p.add_argument("--judge-api-key", dest="judge_api_key", default=None)
    p.add_argument("--judge-effort", dest="judge_effort", default="high",
                   choices=["minimal", "low", "medium", "high"])
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
