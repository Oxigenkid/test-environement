"""Intel fusion: turns news, LLM and mood signals into a bounded size multiplier or a veto per coin."""
from __future__ import annotations

import time

from .base import Agent, Context


def fuse(sym: str, news: dict, llm: dict | None, mood: dict, intel_cfg: dict) -> dict:
    base = sym.split("/")[0]
    coin = news.get(base, {}).get("score", 0.0)
    market = news.get("MARKET", {}).get("score", 0.0)
    if llm:
        coin = (coin + llm.get(base, coin)) / 2
        market = (market + llm.get("MARKET", market)) / 2
    has_coin_news = news.get(base, {}).get("count", 0) > 0
    combined = 0.6 * coin + 0.4 * market if has_coin_news else market

    mult = 1 + combined * intel_cfg["max_size_boost"] if combined >= 0 else max(0.0, 1 + combined)
    reasons = []
    fg = (mood.get("fear_greed") or {}).get("value")
    if fg is not None and fg >= 80:
        mult *= 0.8
        reasons.append("extreme greed: size x0.8")
    mcap = (mood.get("global") or {}).get("mcap_change_24h")
    if mcap is not None and mcap <= -8:
        mult *= 0.5
        reasons.append("market cap -8% in 24h: size x0.5")

    veto = None
    if has_coin_news and coin <= intel_cfg["veto_threshold"]:
        veto = f"{base} news strongly negative ({coin:+.2f})"
    elif market <= intel_cfg["market_veto_threshold"]:
        veto = f"market news strongly negative ({market:+.2f})"
    mult = max(0.0, min(1 + intel_cfg["max_size_boost"], mult))
    return {"coin": round(coin, 3), "market": round(market, 3), "combined": round(combined, 3),
            "mult": round(mult, 3), "veto": veto, "fear_greed": fg, "reasons": reasons,
            "contrarian_fear": fg is not None and fg <= 20}


class IntelAgent(Agent):
    name = "intel"

    def run(self, ctx: Context) -> str:
        news = self.journal.get_state("news_scores", {}) or {}
        mood = self.journal.get_state("mood", {}) or {}
        llm_state = self.journal.get_state("llm_scores") if self.cfg.llm["enabled"] else None
        llm = llm_state["scores"] if llm_state and time.time() - llm_state["ts"] < 3 * 3600 else None
        for sym in self.cfg.symbols:
            ctx.intel[sym] = fuse(sym, news, llm, mood, self.cfg.intel)
        self.journal.set_state("intel", ctx.intel)
        return ", ".join(f"{s.split('/')[0]} x{v['mult']:.2f}{' VETO' if v['veto'] else ''}" for s, v in ctx.intel.items())
