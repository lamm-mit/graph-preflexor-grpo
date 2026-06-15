"""OpenAI-compatible clients for the ideation loop.

Generator = Graph-PRefLexOR (uses the Responses API for the <think>/<graph_json> trace).
Questioner = the follow-up asker (`answer` / `converse` strategies) — uses plain
chat-completions, which every OpenAI-compatible server supports (vLLM, mistral.rs, OpenAI),
and may live on its OWN endpoint via `questioner.base_url` (e.g. a second vLLM serving a
Llama-instruct), falling back to the main server otherwise. An optional `baseline` client
(its own base_url) is used for head-to-head runs.
"""
import os

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


def _api_key(cfg, fallback="x"):
    cfg = cfg or {}
    env_name = str(cfg.get("api_key_env") or "").strip()
    env_value = os.environ.get(env_name, "") if env_name and env_name != "x" else ""
    return cfg.get("api_key") or env_value or fallback


def _client_from(role_cfg, fallback_cfg=None):
    """Create an OpenAI-compatible client from a role block.

    Older configs keep the endpoint in `server.base_url`; newer Explorer-edited
    configs may put `base_url` directly on a role. Empty base_url means the
    OpenAI SDK default endpoint.
    """
    role_cfg = role_cfg or {}
    fallback_cfg = fallback_cfg or {}
    base_url = role_cfg.get("base_url") or fallback_cfg.get("base_url") or None
    api_key = _api_key(role_cfg, _api_key(fallback_cfg, "x"))
    return OpenAI(base_url=base_url, api_key=api_key)


def _is_configured_role(role_cfg):
    return bool((role_cfg or {}).get("model"))


class Clients:
    def __init__(self, cfg):
        s = cfg.get("server") or {}
        self.gen_cfg = cfg["generator"]
        self._gen = _client_from(self.gen_cfg, s)
        self.ask_cfg = cfg.get("questioner") or {}
        # questioner may live on its own OpenAI-compatible endpoint (e.g. a second vLLM serving a
        # Llama-instruct). If questioner.base_url is set, use it; else reuse the main server.
        q = self.ask_cfg
        self._ask = _client_from(q, s) if q.get("base_url") else self._gen
        # optional baseline on its own endpoint (frontier API or another local model)
        b = cfg.get("baseline") or None
        self._base = _client_from(b) if _is_configured_role(b) else None
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
        """Questioner: plain chat-completions (universally supported, unlike the Responses API the
        generator needs), on its own endpoint if `questioner.base_url` is set, else the main server."""
        m = self.ask_cfg
        r = self._ask.chat.completions.create(
            model=m["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=m.get("temperature", 0.9),
            max_tokens=m.get("max_tokens", 512))
        text = (r.choices[0].message.content or "").strip()
        return text, _usage(r)
