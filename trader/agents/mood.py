"""Market mood: Crypto Fear & Greed index (alternative.me) and global market data (CoinGecko)."""
from __future__ import annotations

import requests

from .base import Agent, Context

UA = {"User-Agent": "Mozilla/5.0 (compatible; autonomous-trader/0.1)"}


def fetch_fear_greed() -> dict:
    r = requests.get("https://api.alternative.me/fng/?limit=1", headers=UA, timeout=10)
    r.raise_for_status()
    d = r.json()["data"][0]
    return {"value": int(d["value"]), "label": d["value_classification"]}


def fetch_global() -> dict:
    r = requests.get("https://api.coingecko.com/api/v3/global", headers=UA, timeout=10)
    r.raise_for_status()
    d = r.json()["data"]
    return {
        "mcap_change_24h": float(d.get("market_cap_change_percentage_24h_usd") or 0.0),
        "btc_dominance": float(d.get("market_cap_percentage", {}).get("btc") or 0.0),
    }


class MoodAgent(Agent):
    name = "mood"
    every_seconds = 15 * 60

    def run(self, ctx: Context) -> str:
        mood = self.journal.get_state("mood", {}) or {}
        notes = []
        try:
            mood["fear_greed"] = fetch_fear_greed()
            notes.append(f"F&G {mood['fear_greed']['value']} ({mood['fear_greed']['label']})")
        except Exception as e:
            notes.append(f"F&G unavailable: {type(e).__name__}")
        try:
            mood["global"] = fetch_global()
            notes.append(f"mcap 24h {mood['global']['mcap_change_24h']:+.1f}%")
        except Exception as e:
            notes.append(f"CoinGecko unavailable: {type(e).__name__}")
        self.journal.set_state("mood", mood)
        return " | ".join(notes)
