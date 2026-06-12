#!/usr/bin/env python
"""Blind, multi-dimensional LLM-judge comparison of two sets of answers.

Built for the headline evaluation: does the graph reasoning make a small model's answers better?
You generate, for each of N tasks, a SYSTEM answer (synthesize.py, with graph insights) and a
BASELINE answer (synthesize.py --no-insights, same model+task, no graph). This script presents each
pair to a judge LLM **blind and order-randomized**, scores both 1-5 on several dimensions, records
the preference, then aggregates across tasks -> mean ± std-error bars + win-rates + a figure.

Inputs: a tasks file (one task per line) and two directories of answer files (sorted by name,
matched by position). Answer files may be raw text or `synthesize.py` output (the `*Topic*/*Task*`
header before the first `---` is stripped so the judge can't tell which is which).

    python compare.py --tasks tasks.txt --system answers/sys --baseline answers/base \
        --judge-model gpt-5.5 --out figures/benchmark

Judge defaults to gpt-5.5 via the OpenAI SDK (any OpenAI-compatible endpoint with --judge-base-url).
Judging is evaluation, not generation, so a strong judge here does NOT contaminate the
system/baseline comparison (both answers come from the SAME small model).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

DIMENSIONS = {
    "novelty":     "how original and non-obvious is the core idea (vs textbook/obvious)?",
    "insight":     "does it reveal a non-obvious connection or mechanism, not just restate the question?",
    "mechanism":   "is the proposed mechanism concrete, specific, and physically plausible?",
    "feasibility": "could it realistically be implemented with plausible materials/methods?",
    "testability": "is there a clear, falsifiable prediction or a concrete experiment?",
}

JUDGE_SYSTEM = (
    "You are a meticulous, impartial reviewer of scientific research ideas. You score answers on a "
    "strict 1-5 rubric (1=poor, 3=adequate, 5=exceptional). You are NOT swayed by length, confident "
    "tone, or formatting -- only by the substance of the idea. Reserve 5 for genuinely exceptional.")


def _strip_header(text):
    """Drop a synthesize.py-style provenance header (everything up to & incl. the first '---')."""
    m = re.search(r"^\s*---\s*$", text, flags=re.MULTILINE)
    return text[m.end():].strip() if m else text.strip()


def _read_answers(d):
    files = sorted(glob.glob(os.path.join(d, "*.md")) + glob.glob(os.path.join(d, "*.txt")))
    return [(os.path.basename(f), _strip_header(open(f, encoding="utf-8").read())) for f in files]


def _judge_prompt(task, ans_a, ans_b, dims):
    dimdesc = "\n".join(f"- {d}: {DIMENSIONS[d]}" for d in dims)
    keys = ", ".join(f'"{d}": <int 1-5>' for d in dims)
    return (f"Two answers (A and B) respond to the SAME task. Score EACH answer 1-5 (integers) on "
            f"every dimension, then state which you prefer overall.\n\n"
            f"Dimensions:\n{dimdesc}\n\n"
            f"TASK:\n{task}\n\n--- ANSWER A ---\n{ans_a}\n\n--- ANSWER B ---\n{ans_b}\n\n"
            f"Return ONLY a JSON object, no prose:\n"
            f'{{"A": {{{keys}}}, "B": {{{keys}}}, "preferred": "A"|"B"|"tie", "reason": "<one sentence>"}}')


def _judge_call(system, prompt, backend, model, base_url, api_key, temperature):
    """One chat-completions call, tolerant of API generations. Newer models (gpt-5.x) require
    `max_completion_tokens` and reject a custom `temperature`; older ones use `max_tokens`. We start
    modern and adapt to whatever the endpoint rejects, retrying the SAME request (no wasted calls)."""
    if backend != "openai":
        raise SystemExit(f"judge backend '{backend}' not supported (use openai)")
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY") or "x"
    client = OpenAI(base_url=base_url or None, api_key=key)
    # generous token budget: for reasoning models this cap also covers reasoning tokens, so a small
    # value can be exhausted before the JSON is emitted.
    kwargs = {"model": model, "max_completion_tokens": 4000, "temperature": temperature,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    last = None
    for _ in range(5):
        try:
            r = client.chat.completions.create(**kwargs)
            return r.choices[0].message.content or ""
        except Exception as e:
            last, msg = e, str(e).lower()
            if "max_completion_tokens" in kwargs and "max_completion_tokens" in msg and \
                    any(k in msg for k in ("unsupported", "unexpected", "not supported")):
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")     # older model
            elif "max_tokens" in kwargs and "max_tokens" in msg and \
                    any(k in msg for k in ("unsupported", "max_completion_tokens", "not supported")):
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")     # newer model
            elif "temperature" in kwargs and "temperature" in msg and \
                    any(k in msg for k in ("unsupported", "does not support", "not supported")):
                kwargs.pop("temperature")                                       # fixed-temp model
            else:
                raise
    raise last


def _parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", required=True, help="tasks file (one task per line)")
    p.add_argument("--system", required=True, help="dir of SYSTEM answers (with graph insights)")
    p.add_argument("--baseline", required=True, help="dir of BASELINE answers (--no-insights)")
    p.add_argument("--out", default="figures/benchmark", help="output basename")
    p.add_argument("--dimensions", nargs="+", default=list(DIMENSIONS),
                   help=f"score dimensions (default: {list(DIMENSIONS)})")
    p.add_argument("--judge-backend", dest="jb", choices=["openai"], default="openai")
    p.add_argument("--judge-model", dest="jm", default="gpt-5.5", help="judge model (default gpt-5.5)")
    p.add_argument("--judge-base-url", dest="jbu", default=None, help="OpenAI-compatible endpoint")
    p.add_argument("--judge-api-key", dest="jak", default=None, help="judge api key (else $OPENAI_API_KEY)")
    p.add_argument("--judge-temperature", dest="jt", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0, help="order-randomization seed")
    p.add_argument("--system-label", default="graph-insights")
    p.add_argument("--baseline-label", default="baseline (single-shot)")
    args = p.parse_args()
    dims = [d for d in args.dimensions if d in DIMENSIONS]

    tasks = [ln.strip() for ln in open(args.tasks, encoding="utf-8") if ln.strip()]
    sys_ans = _read_answers(args.system)
    base_ans = _read_answers(args.baseline)
    n = min(len(tasks), len(sys_ans), len(base_ans))
    if n == 0:
        raise SystemExit("need matching tasks + system + baseline answers")
    if not (len(tasks) == len(sys_ans) == len(base_ans)):
        print(f"[compare] WARNING: counts differ (tasks={len(tasks)}, system={len(sys_ans)}, "
              f"baseline={len(base_ans)}) -- comparing the first {n}.")

    import random
    sys_scores = {d: [] for d in dims}      # per-task system score per dim
    base_scores = {d: [] for d in dims}
    prefs = []                              # 'system' | 'baseline' | 'tie'
    records = []
    for i in range(n):
        task = tasks[i]
        rng = random.Random(args.seed + i)
        sys_is_A = rng.random() < 0.5       # blind order randomization
        a_txt, b_txt = (sys_ans[i][1], base_ans[i][1]) if sys_is_A else (base_ans[i][1], sys_ans[i][1])
        print(f"[compare] judging task {i+1}/{n}  (system={'A' if sys_is_A else 'B'})...", flush=True)
        out = _judge_call(JUDGE_SYSTEM, _judge_prompt(task, a_txt, b_txt, dims),
                          args.jb, args.jm, args.jbu, args.jak, args.jt)
        v = _parse_json(out)
        if not v or "A" not in v or "B" not in v:
            print(f"  (skipped task {i+1}: unparseable judge output)")
            continue
        sysv = v["A"] if sys_is_A else v["B"]
        basev = v["B"] if sys_is_A else v["A"]
        for d in dims:
            try:
                sys_scores[d].append(float(sysv[d])); base_scores[d].append(float(basev[d]))
            except Exception:
                pass
        pref = v.get("preferred", "tie")
        pref = "system" if pref == ("A" if sys_is_A else "B") else \
               ("baseline" if pref in ("A", "B") else "tie")
        prefs.append(pref)
        records.append({"task": task, "system": sysv, "baseline": basev,
                        "preferred": pref, "reason": v.get("reason", "")})

    if not records:
        raise SystemExit("no tasks were judged successfully")
    nj = len(records)

    # ---- aggregate ----
    def ms(xs):
        a = np.asarray(xs, float)
        return (float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 1 \
            else ((float(a.mean()), 0.0) if len(a) else (float("nan"), 0.0))
    sys_mean = {d: ms(sys_scores[d]) for d in dims}
    base_mean = {d: ms(base_scores[d]) for d in dims}
    dim_winrate = {d: float(np.mean([s > b for s, b in zip(sys_scores[d], base_scores[d])]))
                   if sys_scores[d] else float("nan") for d in dims}
    overall = {"system": prefs.count("system") / nj, "baseline": prefs.count("baseline") / nj,
               "tie": prefs.count("tie") / nj}

    # ---- figure: grouped bars (system vs baseline), SEM error bars ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(1.6 * len(dims) + 2, 5.2))
    x = np.arange(len(dims)); w = 0.38
    sm = [sys_mean[d][0] for d in dims]; se = [sys_mean[d][1] for d in dims]
    bm = [base_mean[d][0] for d in dims]; be = [base_mean[d][1] for d in dims]
    ax.bar(x - w/2, sm, w, yerr=se, capsize=4, color="#d62728", label=args.system_label)
    ax.bar(x + w/2, bm, w, yerr=be, capsize=4, color="#1f77b4", label=args.baseline_label)
    for xi, d in zip(x, dims):                          # per-dim win-rate annotation
        ax.annotate(f"{100*dim_winrate[d]:.0f}% win", (xi, 5.05), ha="center", fontsize=7.5,
                    color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(dims, rotation=15)
    ax.set_ylabel("judge score (1-5, mean ± s.e.)"); ax.set_ylim(0, 5.4)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title(f"Graph-insight vs baseline answers  (n={nj} tasks, judge={args.jm})\n"
                 f"overall preference: {args.system_label} {100*overall['system']:.0f}%  ·  "
                 f"{args.baseline_label} {100*overall['baseline']:.0f}%  ·  tie {100*overall['tie']:.0f}%",
                 fontsize=10)
    ax.grid(True, axis="y", color="0.92", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}.png/.svg/.pdf")

    # ---- json + markdown report ----
    report = {"n_tasks": nj, "judge": args.jm, "dimensions": dims, "overall_preference": overall,
              "system_mean_sem": sys_mean, "baseline_mean_sem": base_mean,
              "dim_winrate_system": dim_winrate, "per_task": records}
    with open(f"{args.out}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}.json")
    lines = [f"# Benchmark: {args.system_label} vs {args.baseline_label}",
             f"\n*{nj} tasks · judge: {args.jm}*\n",
             f"**Overall preference:** {args.system_label} **{100*overall['system']:.0f}%**, "
             f"{args.baseline_label} {100*overall['baseline']:.0f}%, tie {100*overall['tie']:.0f}%\n",
             "| dimension | " + args.system_label + " | " + args.baseline_label + " | system win-rate |",
             "|---|---|---|---|"]
    for d in dims:
        lines.append(f"| {d} | {sys_mean[d][0]:.2f} ± {sys_mean[d][1]:.2f} | "
                     f"{base_mean[d][0]:.2f} ± {base_mean[d][1]:.2f} | {100*dim_winrate[d]:.0f}% |")
    with open(f"{args.out}.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}.md")
    print(f"[compare] {args.system_label} preferred on {100*overall['system']:.0f}% of {nj} tasks "
          f"(baseline {100*overall['baseline']:.0f}%, tie {100*overall['tie']:.0f}%)")


if __name__ == "__main__":
    main()
