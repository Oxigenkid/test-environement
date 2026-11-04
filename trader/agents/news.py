"""News scout: scrapes finance/crypto RSS feeds and scores headlines for sentiment per coin."""
from __future__ import annotations

import calendar
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .base import Agent, Context

USER_AGENT = "Mozilla/5.0 (compatible; autonomous-trader/0.1; personal use)"
MAX_AGE_HOURS = 48

CRYPTO_LEXICON = {
    "hack": -3.0, "hacked": -3.0, "hacker": -2.5, "exploit": -2.5, "exploited": -2.5, "drained": -2.5,
    "rug": -3.0, "scam": -3.0, "fraud": -3.0, "lawsuit": -2.0, "sues": -2.0, "sued": -2.0, "charges": -1.5,
    "ban": -2.5, "bans": -2.5, "banned": -2.5, "crackdown": -2.5, "liquidation": -1.5, "liquidations": -1.5,
    "liquidated": -2.0, "selloff": -2.0, "sell-off": -2.0, "plunge": -2.5, "plunges": -2.5, "tumbles": -2.0,
    "crash": -3.0, "crashes": -3.0, "dump": -2.0, "dumps": -2.0, "bearish": -2.0, "outflows": -1.5,
    "delist": -2.5, "delisting": -2.5, "insolvency": -3.0, "insolvent": -3.0, "bankrupt": -3.0,
    "bankruptcy": -3.0, "depeg": -3.0, "recession": -2.0, "slump": -2.0, "slides": -1.5, "falls": -1.2,
    "approval": 2.0, "approves": 2.0, "approved": 2.0, "inflows": 1.5, "rally": 2.0, "rallies": 2.0,
    "surge": 2.0, "surges": 2.0, "soar": 2.5, "soars": 2.5, "bullish": 2.0, "adoption": 1.5,
    "record": 1.0, "ath": 2.0, "upgrade": 1.5, "partnership": 1.2, "jumps": 1.8, "gains": 1.5,
    "rebound": 1.5, "rebounds": 1.5, "accumulation": 1.2, "breakout": 1.5, "climbs": 1.5,
}

COIN_PATTERNS = {
    "BTC": r"\b(bitcoin|btc|xbt)\b",
    "ETH": r"\b(ethereum|ether|eth)\b",
    "SOL": r"\b(solana|sol)\b",
    "XRP": r"\b(xrp|ripple)\b",
    "ADA": r"\b(cardano|ada)\b",
    "DOGE": r"\b(dogecoin|doge)\b",
    "AVAX": r"\b(avalanche|avax)\b",
    "LINK": r"\b(chainlink|link)\b",
    "DOT": r"\b(polkadot)\b",
    "LTC": r"\b(litecoin|ltc)\b",
}
MARKET_PATTERN = (r"\b(crypto|cryptocurrency|market|markets|fed|federal reserve|fomc|rate|rates|inflation|cpi|"
                  r"sec|etf|regulation|regulators?|stocks?|nasdaq|s&p|treasury|dollar|tariffs?|economy|jobs)\b")

_analyzer: SentimentIntensityAnalyzer | None = None


def analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(CRYPTO_LEXICON)
    return _analyzer


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def score_text(title: str, summary: str = "") -> float:
    """Headline counts double; result in [-1, 1]."""
    a = analyzer()
    t = a.polarity_scores(title)["compound"]
    if summary:
        s = a.polarity_scores(summary[:400])["compound"]
        return (2 * t + s) / 3
    return t


def tag_symbols(text: str, tracked: list[str]) -> list[str]:
    low = text.lower()
    tags = [c for c in tracked if c in COIN_PATTERNS and re.search(COIN_PATTERNS[c], low)]
    if re.search(MARKET_PATTERN, low) or not tags:
        tags.append("MARKET")
    return tags


def fetch_feed(url: str, timeout: float = 12.0) -> list[dict]:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    source = urlparse(url).netloc.replace("www.", "")
    items = []
    for e in parsed.entries:
        link = e.get("link") or e.get("id")
        title = clean(e.get("title", ""))
        if not link or not title:
            continue
        tp = e.get("published_parsed") or e.get("updated_parsed")
        published = calendar.timegm(tp) if tp else time.time()
        items.append({"link": link, "title": title, "summary": clean(e.get("summary", "")), "source": source,
                      "published": float(published)})
    return items


def aggregate(rows: list[dict], now: float, half_life_h: float, keys: list[str]) -> dict[str, dict]:
    """Time-decayed average sentiment per key. Confidence grows with the amount of fresh news."""
    out = {}
    for k in keys:
        num = den = 0.0
        n = 0
        for r in rows:
            if k not in r["symbols"]:
                continue
            age_h = max(0.0, (now - r["published"]) / 3600)
            w = 0.5 ** (age_h / half_life_h)
            num += w * r["score"]
            den += w
            n += 1
        avg = num / den if den else 0.0
        confidence = min(1.0, den / 5.0)
        out[k] = {"score": round(avg * confidence, 4), "raw": round(avg, 4), "count": n, "weight": round(den, 3)}
    return out


class NewsAgent(Agent):
    name = "news"

    def __init__(self, cfg, journal):
        super().__init__(cfg, journal)
        self.every_seconds = cfg.intel["news_refresh_minutes"] * 60

    def fetch_all(self) -> tuple[int, int]:
        feeds = self.cfg.intel["feeds"]
        new = failed = 0
        with ThreadPoolExecutor(max_workers=min(8, len(feeds) or 1)) as pool:
            results = list(pool.map(self._safe_fetch, feeds))
        cutoff = time.time() - MAX_AGE_HOURS * 3600
        tracked = self.cfg.base_assets
        for items in results:
            if items is None:
                failed += 1
                continue
            for it in items:
                if it["published"] < cutoff:
                    continue
                if self.journal.query("SELECT 1 FROM news WHERE link=?", (it["link"],)):
                    continue
                syms = tag_symbols(f"{it['title']} {it['summary']}", tracked)
                score = score_text(it["title"], it["summary"])
                self.journal.execute(
                    "INSERT OR IGNORE INTO news VALUES(?,?,?,?,?,?,?)",
                    (it["link"], it["title"], it["source"], it["published"], time.time(), score, json.dumps(syms)),
                )
                new += 1
        self.journal.execute("DELETE FROM news WHERE published < ?", (time.time() - 7 * 86400,))
        return new, failed

    def _safe_fetch(self, url: str):
        try:
            return fetch_feed(url)
        except Exception as e:
            self.journal.log(self.name, f"feed failed {url}: {type(e).__name__}", "warn")
            return None

    def scores(self, now: float) -> dict[str, dict]:
        rows = self.journal.query("SELECT published, score, symbols FROM news WHERE published > ?",
                                  (now - MAX_AGE_HOURS * 3600,))
        for r in rows:
            r["symbols"] = json.loads(r["symbols"])
        return aggregate(rows, now, self.cfg.intel["news_half_life_hours"], self.cfg.base_assets + ["MARKET"])

    def run(self, ctx: Context) -> str:
        new, failed = self.fetch_all()
        sc = self.scores(ctx.now)
        self.journal.set_state("news_scores", sc)
        summary = ", ".join(f"{k} {v['score']:+.2f}" for k, v in sc.items())
        return f"{new} new headlines, {failed} feeds failed | {summary}"
