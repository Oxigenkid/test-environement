"""Optional LLM analyst. Talks to any OpenAI-compatible chat endpoint (e.g. a local Ollama server)."""
from __future__ import annotations

import json
import os
import re
import time

import requests

from .base import Agent, Context

PROMPT = """You are a cautious crypto market analyst. Read these recent headlines and rate the likely
short-term (next 24h) price impact for each key on a scale from -1 (very bearish) to 1 (very bullish).
Use 0 when unclear. Reply with JSON only, e.g. {{"BTC": 0.1, "MARKET": -0.2}}.
Keys: {keys}

Headlines:
{headlines}
"""


def parse_scores(text: str, keys: list[str]) -> dict[str, float]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    data = json.loads(m.group(0))
    out = {}
    for k in keys:
        try:
            out[k] = max(-1.0, min(1.0, float(data.get(k, 0.0))))
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


class LLMAgent(Agent):
    name = "llm"

    def __init__(self, cfg, journal):
        super().__init__(cfg, journal)
        self.every_seconds = cfg.llm["refresh_minutes"] * 60

    def run(self, ctx: Context) -> str:
        rows = self.journal.query("SELECT title, source FROM news WHERE published > ? ORDER BY published DESC LIMIT 40",
                                  (time.time() - 24 * 3600,))
        if not rows:
            return "no headlines yet"
        keys = self.cfg.base_assets + ["MARKET"]
        prompt = PROMPT.format(keys=", ".join(keys), headlines="\n".join(f"- {r['title']} ({r['source']})" for r in rows))
        headers = {"Content-Type": "application/json"}
        if os.environ.get("TRADER_LLM_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['TRADER_LLM_API_KEY']}"
        resp = requests.post(
            self.cfg.llm["base_url"].rstrip("/") + "/chat/completions",
            headers=headers,
            json={"model": self.cfg.llm["model"], "temperature": 0, "messages": [{"role": "user", "content": prompt}]},
            timeout=120,
        )
        resp.raise_for_status()
        scores = parse_scores(resp.json()["choices"][0]["message"]["content"], keys)
        self.journal.set_state("llm_scores", {"ts": ctx.now, "scores": scores})
        return ", ".join(f"{k} {v:+.2f}" for k, v in scores.items())
