#!/usr/bin/env python
"""Diagnose what the served model returns via the Responses API.

Run from the ideation/ dir (uses config.yaml):  python diag.py
Shows whether <graph_json> appears, and WHERE (output_text vs reasoning items),
for a few input shapes — so we know how to extract it.
"""
import json
import yaml
from openai import OpenAI

cfg = yaml.safe_load(open("config.yaml"))
s, g = cfg["server"], cfg["generator"]
c = OpenAI(base_url=s["base_url"], api_key=s.get("api_key", "x"))
Q = "What governs fracture toughness in self-healing biopolymer composites?"


def report(tag, r):
    ot = getattr(r, "output_text", "") or ""
    print(f"\n===== {tag} =====")
    print("output_text length:", len(ot))
    print("  <graph_json> in output_text:", "<graph_json>" in ot)
    print("  <think> in output_text     :", "<think>" in ot)
    print("output_text[:400]:", repr(ot[:400]))
    # walk structured output to see where the trace really is
    try:
        d = json.loads(r.model_dump_json())
        for i, item in enumerate(d.get("output", []) or []):
            txt = json.dumps(item)
            print(f"  output[{i}] type={item.get('type')!r} "
                  f"has_graph_json={'<graph_json>' in txt} len={len(txt)}")
    except Exception as e:
        print("  (could not dump structured output:", e, ")")


# 1) string input + reasoning effort (what the loop currently does)
try:
    report("string input, reasoning=high",
           c.responses.create(model=g["model"], input=Q, max_output_tokens=4000,
                              reasoning={"effort": "high"}))
except Exception as e:
    print("string+reasoning FAILED:", repr(e))

# 2) string input, no reasoning param
try:
    report("string input, no reasoning",
           c.responses.create(model=g["model"], input=Q, max_output_tokens=4000))
except Exception as e:
    print("string FAILED:", repr(e))

# 3) message-list input (forces chat-template treatment)
try:
    report("message-list input",
           c.responses.create(model=g["model"],
                              input=[{"role": "user", "content": Q}], max_output_tokens=4000))
except Exception as e:
    print("message-list FAILED:", repr(e))
