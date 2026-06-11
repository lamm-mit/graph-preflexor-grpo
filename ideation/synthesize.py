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

import insights as I


# --------------------------------------------------------------------------- #
#  Prompt construction
# --------------------------------------------------------------------------- #
DEFAULT_SYSTEM = (
    "You are a rigorous, inventive research scientist and synthesist. You are given a "
    "question and a set of non-obvious connections that were mined from a knowledge graph "
    "an AI assembled while reasoning about that question. The connections are structural "
    "leads — paths between distant concepts, missing-but-implied links, recurring relational "
    "analogies, feedback loops, broker concepts, and similar-but-unlinked pairs — NOT "
    "established facts. Your job is to produce a single, coherent, original answer that is "
    "demonstrably enriched by these leads: integrate the promising ones, reason about "
    "mechanisms, and be explicit about what is well-grounded versus speculative."
)

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


def format_insights(results, G, max_per_kind=6):
    """Render the mined leads as a compact, model-readable block, grouped by miner."""
    blocks = []
    for kind, ins in results:
        if not ins:
            continue
        header = I.KIND_HEADER.get(kind, kind)
        lines = [f"## {header}"]
        for x in ins[:max_per_kind]:
            detail = " ".join(str(x.get("detail", "")).split())
            lines.append(f"- {x['title']}  (score {x.get('score', 0):.2f}). {detail}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(no structural leads were found)"


def build_prompt(topic, results, G, style, task, max_per_kind):
    insight_block = format_insights(results, G, max_per_kind=max_per_kind)
    instruction = task or STYLES.get(style, STYLES["report"])
    prompt = f"""# Original question / topic
{topic or "(topic not recorded — answer the question implied by the leads below)"}

# Mined structural leads from the reasoning graph
The following were extracted from the *structure* of a knowledge graph built while
reasoning about the question. Treat them as promising leads to investigate and weave
in — not as verified facts.

{insight_block}

# Your task
{instruction}

Ground every claim you can in the leads above, but you may add your own domain knowledge
to connect, explain, and extend them. Where you rely on a specific mined lead, it is fine
to reference it briefly (e.g. the bridge or analogy it came from). Be specific, mechanistic,
and original."""
    return prompt


# --------------------------------------------------------------------------- #
#  Backends
# --------------------------------------------------------------------------- #
def answer_openai(system, prompt, *, model, base_url, api_key, temperature, max_tokens):
    """OpenAI SDK chat-completions — works for real OpenAI and any compatible server."""
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY") or "x"
    client = OpenAI(base_url=base_url or None, api_key=key)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content or ""


def answer_hf(system, prompt, *, model, temperature, max_tokens, device, dtype):
    """Local Hugging Face causal LM via transformers (chat template if available)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    torch_dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}.get(dtype, "auto")
    lm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch_dtype, device_map=(device or "auto"))
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if getattr(tok, "chat_template", None):
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    else:                                              # no chat template: concatenate
        inputs = tok(system + "\n\n" + prompt, return_tensors="pt").input_ids
    inputs = inputs.to(lm.device)
    gen = lm.generate(inputs, max_new_tokens=max_tokens, do_sample=temperature > 0,
                      temperature=max(temperature, 1e-5),
                      pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(gen[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


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
    pr.add_argument("--max-per-kind", type=int, default=6,
                    help="max leads per miner to include in the prompt")
    pr.add_argument("--show-prompt", action="store_true", help="print the built prompt and exit")

    be = p.add_argument_group("backend")
    be.add_argument("--backend", choices=["openai", "hf"], default="openai")
    be.add_argument("--model", help="model id (e.g. gpt-4o, or a HF repo id)")
    be.add_argument("--base-url", help="[openai] OpenAI-compatible server base url (e.g. http://localhost:8000/v1)")
    be.add_argument("--api-key", help="[openai] api key (else $OPENAI_API_KEY)")
    be.add_argument("--temperature", type=float, default=0.7)
    be.add_argument("--max-tokens", type=int, default=2048)
    be.add_argument("--device", help="[hf] device_map (default auto)")
    be.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                    help="[hf] torch dtype")

    p.add_argument("--out", help="write the answer here (default: <run>/answer.md or ./answer.md)")
    args = p.parse_args()

    if args.max_iter is not None:                      # shared cap: insights.load_graph applies it
        I.MAX_ITER = args.max_iter

    # ---- load / mine insights ------------------------------------------------
    if args.insights:
        data = json.load(open(args.insights))
        topic = args.topic or data.get("topic", "")
        miners = data.get("miners", {})
        results = [(k, miners.get(k, [])) for k in I.KIND_ORDER]
        G = I.load_graph(os.path.dirname(args.insights) or ".") if os.path.exists(
            os.path.join(os.path.dirname(args.insights) or ".", "graph.graphml")) else None
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
    system = args.system or DEFAULT_SYSTEM
    prompt = build_prompt(topic, results, G, args.style, args.task, args.max_per_kind)
    n_leads = sum(len(ins) for _, ins in results)
    print(f"[synthesize] topic={topic!r}  leads={n_leads}  style={args.style}  "
          f"backend={args.backend}  model={args.model}")
    if args.show_prompt:
        print("\n" + "=" * 80 + "\n" + system + "\n" + "-" * 80 + "\n" + prompt)
        return

    if not args.model:
        raise SystemExit("--model is required (e.g. --model gpt-4o, or a HF repo id). "
                         "Use --show-prompt to preview the prompt without a model.")

    # ---- generate ------------------------------------------------------------
    if args.backend == "openai":
        answer = answer_openai(system, prompt, model=args.model, base_url=args.base_url,
                               api_key=args.api_key, temperature=args.temperature,
                               max_tokens=args.max_tokens)
    else:
        answer = answer_hf(system, prompt, model=args.model, temperature=args.temperature,
                           max_tokens=args.max_tokens, device=args.device, dtype=args.dtype)

    # ---- write ---------------------------------------------------------------
    out = args.out or (os.path.join(args.run.rstrip("/"), "answer.md") if args.run else "answer.md")
    header = (f"# Insight-enriched answer\n\n"
              f"*Question:* {topic}\n\n"
              f"*Backend:* {args.backend} · *Model:* {args.model} · *Style:* {args.style} · "
              f"*Leads used:* {n_leads}\n\n---\n\n")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(header + answer.strip() + "\n")
    print(f"\n{'='*80}\n{answer.strip()}\n{'='*80}")
    print(f"[synthesize] wrote {out}")


if __name__ == "__main__":
    main()
