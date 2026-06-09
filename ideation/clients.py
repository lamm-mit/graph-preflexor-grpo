"""OpenAI-compatible clients (mistral.rs Responses API) for the ideation loop.

One HTTP endpoint serves both models; the `model` field selects which.
Generator = Graph-PRefLexOR; questioner = the follow-up asker. An optional
separate `baseline` client (its own base_url) is used for head-to-head runs.
"""
from openai import OpenAI


def _full_text(r):
    """Concatenate every textual part of a Responses object so we can find the
    <think>/<graph_json> sentinels regardless of where mistral.rs puts them."""
    parts = []
    ot = getattr(r, "output_text", None)
    if ot:
        parts.append(ot)
    rsn = getattr(r, "reasoning", None)
    if isinstance(rsn, str) and rsn:
        parts.append(rsn)
    for item in (getattr(r, "output", None) or []):
        for c in (getattr(item, "content", None) or []):
            t = getattr(c, "text", None)
            if isinstance(t, str) and t:
                parts.append(t)
    return "\n".join(parts)


def _usage(r):
    u = getattr(r, "usage", None)
    return getattr(u, "total_tokens", 0) if u else 0


class Clients:
    def __init__(self, cfg):
        s = cfg["server"]
        self._gen = OpenAI(base_url=s["base_url"], api_key=s.get("api_key", "x"))
        self.gen_cfg = cfg["generator"]
        self.ask_cfg = cfg["questioner"]
        # optional baseline on its own endpoint (frontier API or another local model)
        b = cfg.get("baseline")
        self._base = OpenAI(base_url=b["base_url"], api_key=b.get("api_key", "x")) if b else None
        self.base_cfg = b

    def _responses(self, client, mcfg, text, previous_id=None, effort="high"):
        kwargs = dict(model=mcfg["model"], input=text,
                      max_output_tokens=mcfg.get("max_tokens", 6000))
        if previous_id:
            kwargs["previous_response_id"] = previous_id
        if "temperature" in mcfg:
            kwargs["temperature"] = mcfg["temperature"]
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        return client.responses.create(**kwargs)

    def generate(self, question, previous_id=None):
        r = self._responses(self._gen, self.gen_cfg, question, previous_id, effort="high")
        return {"id": r.id, "answer": getattr(r, "output_text", "") or "",
                "full": _full_text(r), "usage": _usage(r)}

    def baseline(self, question, previous_id=None):
        if not self._base:
            raise RuntimeError("no baseline endpoint configured (cfg['baseline'])")
        r = self._responses(self._base, self.base_cfg, question, previous_id, effort="high")
        return {"id": r.id, "answer": getattr(r, "output_text", "") or "",
                "full": _full_text(r), "usage": _usage(r)}

    def ask(self, prompt):
        r = self._responses(self._gen, self.ask_cfg, prompt, None, effort="low")
        return (getattr(r, "output_text", "") or "").strip(), _usage(r)
