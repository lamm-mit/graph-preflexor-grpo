#!/usr/bin/env python
"""Synthesize a complete, insight-enriched answer to the original ideation query.

`ideate.py` grows a knowledge graph; `insights.py` mines it for non-obvious
structural leads (conceptual bridges, latent links, relational analogies, feedback
loops, broker concepts, similar-but-unlinked pairs). This tool feeds the *original
question* plus those mined leads to a language model and asks it to write a single,
coherent, original answer that is genuinely enriched by what the graph revealed —
i.e. the loop's reasoning is distilled back into prose.

It is a thin layer ON TOP of insights.py: it will load a run's `insights.json` if
present, or mine it on the fly (`--mine`).

Backends (pick with --backend):
  responses OpenAI Responses API. Covers the real OpenAI API and compatible servers
           that expose /v1/responses.
  openai   OpenAI SDK chat-completions. Covers BOTH the real OpenAI API and ANY
           OpenAI-compatible server (vLLM, mistral.rs, llama.cpp, TGI, Together,
           Groq, ...) — just point --base-url at it. --api-key or $OPENAI_API_KEY.
  hf       A local Hugging Face model via `transformers` (downloaded by id), run
           on this machine. Uses the tokenizer's chat template when available.

Prompting is deliberately flexible: choose a built-in --style preset, override the
task with free-text --task, and/or replace the system prompt with --system. Nothing
about the wording is hard-coded to a domain.

Examples
--------
# Real OpenAI:
python synthesize.py --run runs/exp2 --backend openai --model gpt-4o \
    --api-key $OPENAI_API_KEY --out runs/exp2/answer.md

# Any OpenAI-compatible server (e.g. local vLLM on :8000):
python synthesize.py --run runs/exp2 --backend openai \
    --base-url http://localhost:8000/v1 --model meta-llama/Llama-3.1-70B-Instruct

# Local Hugging Face model, mining insights fresh, as a research proposal:
python synthesize.py --run runs/exp2 --backend hf \
    --model mistralai/Mistral-7B-Instruct-v0.3 --mine --style proposal

# Override exactly what you want written:
python synthesize.py --run runs/exp2 --backend openai --model gpt-4o \
    --task "Write a 1-page Nature-style abstract proposing the single most novel mechanism."
"""
import argparse
import json
import os
import re

import insights as I


# --------------------------------------------------------------------------- #
#  Prompt construction
# --------------------------------------------------------------------------- #
# Baseline (--no-insights) system prompt: clean role, no mention of a graph or leads.
BASELINE_SYSTEM = (
    "You are a rigorous, inventive research scientist and synthesist. Produce a single, "
    "coherent, original answer to the question below: reason about mechanisms, propose concrete "
    "and testable ideas, and be explicit about what is well-grounded versus speculative."
)

# Insights arm uses the EXACT SAME system prompt as the baseline. The only difference between the two
# arms is a short neutral 'Background' block of clean leads in the user message — no extra instruction,
# no graph vocabulary, nothing scored to cite — so the comparison stays neutral and isolates purely
# what the graph-derived context adds.
DEFAULT_SYSTEM = BASELINE_SYSTEM

STYLES = {
    "report": (
        "Write a comprehensive, well-structured answer to the original question. "
        "Integrate the most promising mined connections; foreground genuinely novel or "
        "cross-domain hypotheses (analogies, bridges, feedback loops); explain the "
        "mechanism behind each and why it matters; propose concrete, testable predictions "
        "or next experiments; and clearly separate well-supported reasoning from "
        "speculative leaps. Use sections and bullets where they aid clarity."
    ),
    "hypotheses": (
        "Produce a ranked list of the most novel, specific, and testable hypotheses implied "
        "by the mined connections. For each: state the hypothesis in one sentence, name the "
        "mined connection(s) it derives from, give a brief mechanistic rationale, and propose "
        "an experiment or observation that would falsify it. Order by novelty x plausibility."
    ),
    "proposal": (
        "Write a focused research proposal answering the question. Include: Motivation, a "
        "Central Hypothesis built from the most striking mined connection, Specific Aims "
        "(2-4, each tied to a mined lead), Approach with methods, and Expected outcomes / "
        "risks. Make the novelty explicit and trace it back to the graph structure."
    ),
    "brief": (
        "Write a tight executive brief (≤ 400 words) answering the question, surfacing only "
        "the 2-3 most non-obvious, high-value connections from the mined leads, each with a "
        "one-line mechanism and a one-line 'why it matters'."
    ),
    "review": (
        "Write a critical mini-review answering the question. Organize by theme, weave in the "
        "mined connections as the through-line, contrast competing mechanisms, flag open "
        "questions exposed by the graph's gaps (latent links, open triads), and end with the "
        "single most promising direction and why."
    ),
}


# Plain-language lead-in per miner kind (no metric jargon reaches the model).
_LEAD_IN = {
    "conceptual_bridge": "a possible multi-step connection",
    "latent_link": "a likely missing link",
    "open_triad": "an inferable relationship",
    "relational_analogy": "an analogy",
    "feedback_loop": "a feedback loop",
    "semantic_dissonance": "two related but unconnected ideas",
    "broker_idea": "a pivotal connecting concept",
}


def _humanize(title):
    """Turn a miner's symbol-laden title into a clean phrase: strip kind prefixes and replace the
    relation glyphs with words. No scores, hops, cosines, or AA counts ever reach the model."""
    t = re.sub(r"^\s*(infer:|tension:)\s*", "", str(title))
    t = re.sub(r"^\s*loop\(\d+\):\s*", "", t)
    for a, b in (("⇝", "→"), ("⟶?", "→"), ("⟷", "↔"), ("≈", "≈"), ("⊕⊖", "vs"), ("::", "is to")):
        t = t.replace(a, b)
    return " ".join(t.split())


def _rank_leads(results, n):
    """The n strongest leads ACROSS miners. Uses the cross-miner `actionability` score when present
    (insights.json carries it); otherwise round-robins the per-miner ranks so the set stays small
    and diverse instead of dumping every candidate from every miner."""
    flat = [x for _, ins in results for x in ins]
    if flat and all("actionability" in x for x in flat):
        return sorted(flat, key=lambda x: x.get("actionability", 0.0), reverse=True)[:n]
    cols = [ins for _, ins in results if ins]                     # round-robin fallback
    out, r = [], 0
    while len(out) < n and any(r < len(c) for c in cols):
        for c in cols:
            if r < len(c):
                out.append(c[r])
            if len(out) >= n:
                break
        r += 1
    return out[:n]


def format_insights(results, G, n_leads=8):
    """A short, clean, ranked list of the best leads — natural language only, no metric jargon."""
    leads = _rank_leads(results, n_leads)
    if not leads:
        return "(no structural leads were found)"
    return "\n".join(f"{i}. ({_LEAD_IN.get(x.get('kind', ''), 'a connection')}) "
                     f"{_humanize(x.get('title', ''))}" for i, x in enumerate(leads, 1))


def build_prompt(topic, results, G, style, task, n_leads, use_insights=True):
    """Build the user prompt. The two arms are deliberately PARALLEL — identical topic + task framing
    — and differ ONLY by a short block of the best graph leads, so the comparison isolates what the
    graph reasoning adds. Baseline (use_insights=False): topic + task, no leads."""
    instruction = task or STYLES.get(style, STYLES["report"])
    if not use_insights:
        return f"""# Question / topic
{topic or "(answer the question below)"}

# Your task
{instruction}

Be specific, mechanistic, and original."""
    insight_block = format_insights(results, G, n_leads=n_leads)
    # Neutral: the ONLY addition over the baseline prompt is this 'Background' block. No instruction
    # to use/integrate/cite it, no graph vocabulary, no scores — the model just has extra context.
    return f"""# Question / topic
{topic or "(answer the question below)"}

# Background: connections surfaced while exploring this question
(Exploratory leads, not verified facts — consider any that are relevant.)

{insight_block}

# Your task
{instruction}

Be specific, mechanistic, and original."""


# --------------------------------------------------------------------------- #
#  Backends
# --------------------------------------------------------------------------- #
def _is_official_openai_endpoint(base_url):
    url = (base_url or "").strip().lower()
    return not url or "api.openai.com" in url


def _send_temperature(base_url):
    return not _is_official_openai_endpoint(base_url)


def answer_openai(system, prompt, *, model, base_url, api_key, temperature, max_tokens):
    """OpenAI SDK chat-completions — works for real OpenAI and any compatible server."""
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY") or "x"
    client = OpenAI(base_url=base_url or None, api_key=key)
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None and _send_temperature(base_url):
        kwargs["temperature"] = temperature
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content or ""


def _response_output_text(response):
    text = getattr(response, "output_text", "") or ""
    if text:
        return text
    chunks = []
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            value = getattr(part, "text", None)
            if value:
                chunks.append(value)
    return "\n".join(chunks).strip()


def answer_responses(system, prompt, *, model, base_url, api_key, temperature, max_tokens):
    """OpenAI Responses API — works for OpenAI and compatible /v1/responses servers."""
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY") or "x"
    client = OpenAI(base_url=base_url or None, api_key=key)
    kwargs = {
        "model": model,
        "input": [
            {"role": "developer", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_output_tokens": max_tokens,
    }
    if temperature is not None and _send_temperature(base_url):
        kwargs["temperature"] = temperature
    try:
        r = client.responses.create(**kwargs)
    except Exception as exc:
        if "temperature" in kwargs and "temperature" in str(exc).lower():
            kwargs.pop("temperature", None)
            r = client.responses.create(**kwargs)
        else:
            raise
    return _response_output_text(r)


def answer_hf(system, prompt, *, model, temperature, max_tokens, device, dtype):
    """Local Hugging Face causal LM via transformers (chat template if available)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    torch_dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}.get(dtype, "auto")
    lm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch_dtype, device_map=(device or "auto"))
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    # always build a dict of tensors (input_ids [+ attention_mask]) so generate(**enc) works
    # regardless of whether apply_chat_template returns a tensor or a BatchEncoding.
    if getattr(tok, "chat_template", None):
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    else:                                              # no chat template: concatenate
        enc = tok(system + "\n\n" + prompt, return_tensors="pt")
    enc = {k: v.to(lm.device) for k, v in dict(enc).items()}
    input_len = enc["input_ids"].shape[-1]
    gen = lm.generate(**enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                      temperature=max(temperature, 1e-5),
                      pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(gen[0][input_len:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("input")
    src.add_argument("--run", help="run dir from ideate.py (uses its insights.json or mines it)")
    src.add_argument("--insights", help="path to an insights.json directly (overrides --run lookup)")
    src.add_argument("--mine", action="store_true",
                     help="(re)mine insights now instead of loading insights.json")
    src.add_argument("--top", type=int, default=10, help="candidates per miner when mining")
    src.add_argument("--topic", help="override / supply the original question")
    src.add_argument("--embed-model", dest="embed_model", default=None,
                     help="sentence-transformers id used when mining (--mine / no insights.json); "
                          "default: the run's recorded model, else embeddinggemma-300m")
    src.add_argument("--max-iter", dest="max_iter", type=int, default=None,
                     help="truncate the graph to iter <= this when mining (fair cross-run cutoff)")

    pr = p.add_argument_group("prompt")
    pr.add_argument("--style", default="report", choices=list(STYLES),
                    help="answer preset (default: report)")
    pr.add_argument("--task", help="free-text task instruction (overrides --style wording)")
    pr.add_argument("--system", help="override the system prompt entirely")
    pr.add_argument("--max-leads", dest="max_leads", type=int, default=8,
                    help="how many top leads (ranked across miners by actionability) to feed the "
                         "model. Kept deliberately small — a few clean leads beat a wall of them.")
    pr.add_argument("--no-insights", action="store_true",
                    help="BASELINE: answer the topic+task WITHOUT the mined graph leads "
                         "(single-shot control — same model/task, no graph reasoning)")
    pr.add_argument("--show-prompt", action="store_true", help="print the built prompt and exit")

    be = p.add_argument_group("backend")
    be.add_argument("--backend", choices=["responses", "openai", "hf"], default="responses")
    be.add_argument("--model", help="model id (e.g. gpt-4o, or a HF repo id)")
    be.add_argument("--base-url", help="[openai] OpenAI-compatible server base url (e.g. http://localhost:8000/v1)")
    be.add_argument("--api-key", help="[openai] api key (else $OPENAI_API_KEY)")
    be.add_argument("--temperature", type=float, default=0.7)
    be.add_argument("--max-tokens", type=int, default=8000)
    be.add_argument("--device", help="[hf] device_map (default auto)")
    be.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                    help="[hf] torch dtype")

    p.add_argument("--out", help="write the answer here (default: <run>/answer.md or ./answer.md)")
    args = p.parse_args()

    if args.max_iter is not None:                      # shared cap: insights.load_graph applies it
        I.MAX_ITER = args.max_iter

    # ---- load / mine insights (skipped entirely in baseline mode) ------------
    results = []
    if args.no_insights:                               # BASELINE: no graph leads at all
        rundir = (os.path.dirname(args.insights) or ".") if args.insights else args.run
        jtopic = ""                                    # match the insights arm's topic source exactly
        if args.insights and os.path.exists(args.insights):
            try:
                jtopic = json.load(open(args.insights)).get("topic", "")
            except Exception:
                jtopic = ""
        topic = args.topic or jtopic or (I.read_topic(rundir) if rundir else "")
        G = None
    elif args.insights:
        data = json.load(open(args.insights))
        rundir = os.path.dirname(args.insights) or "."
        miners = data.get("miners", {})
        results = [(k, miners.get(k, [])) for k in I.KIND_ORDER]
        G = I.load_graph(rundir) if os.path.exists(os.path.join(rundir, "graph.graphml")) else None
        topic = args.topic or data.get("topic") or I.read_topic(rundir, G)
    elif args.run:
        topic, G, results = I.load_insights_or_mine(args.run, top=args.top, want_mine=args.mine,
                                                    embed_model=args.embed_model)
        topic = args.topic or topic
    else:
        raise SystemExit("provide --run <dir> or --insights <file.json>")

    if G is None:                                      # format_insights only needs titles/details
        import networkx as nx
        G = nx.DiGraph()

    # ---- build prompt --------------------------------------------------------
    use_insights = not args.no_insights
    system = args.system or (DEFAULT_SYSTEM if use_insights else BASELINE_SYSTEM)
    prompt = build_prompt(topic, results, G, args.style, args.task, args.max_leads, use_insights)
    n_leads = sum(len(ins) for _, ins in results)
    print(f"[synthesize] topic={topic!r}  "
          f"{'leads=' + str(n_leads) if use_insights else 'BASELINE (no insights)'}  "
          f"style={args.style}  backend={args.backend}  model={args.model}")
    if args.show_prompt:
        print("\n" + "=" * 80 + "\n" + system + "\n" + "-" * 80 + "\n" + prompt)
        return

    if not args.model:
        raise SystemExit("--model is required (e.g. --model gpt-4o, or a HF repo id). "
                         "Use --show-prompt to preview the prompt without a model.")

    # ---- generate ------------------------------------------------------------
    if args.backend == "responses":
        answer = answer_responses(system, prompt, model=args.model, base_url=args.base_url,
                                  api_key=args.api_key, temperature=args.temperature,
                                  max_tokens=args.max_tokens)
    elif args.backend == "openai":
        answer = answer_openai(system, prompt, model=args.model, base_url=args.base_url,
                               api_key=args.api_key, temperature=args.temperature,
                               max_tokens=args.max_tokens)
    else:
        answer = answer_hf(system, prompt, model=args.model, temperature=args.temperature,
                           max_tokens=args.max_tokens, device=args.device, dtype=args.dtype)

    # ---- write ---------------------------------------------------------------
    out = args.out or (os.path.join(args.run.rstrip("/"), "answer.md") if args.run else "answer.md")
    ask = args.task or f"(style preset: {args.style})"      # the actual instruction the model answered
    # Clean, condition-neutral header: topic + task only. No title/backend/model/leads — anything
    # that reveals which arm produced the answer would bias a downstream judge (compare.py).
    header = (f"*Topic:* {topic}\n\n"
              f"*Task:* {ask}\n\n---\n\n")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(header + answer.strip() + "\n")
    print(f"\n{'='*80}\n{answer.strip()}\n{'='*80}")
    print(f"[synthesize] wrote {out}")


if __name__ == "__main__":
    main()
