"""Market data agent: live prices every heartbeat, closed candles + indicators when a new bar completes."""
from __future__ import annotations

import pandas as pd

from ..indicators import TIMEFRAME_MINUTES, add_htf, add_indicators, bars_per_day
from .base import Agent, Context

COLS = ["ts", "open", "high", "low", "close", "volume"]


def to_frame(ohlcv: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=COLS).drop_duplicates("ts").set_index("ts").sort_index()
    return df.astype(float)


class MarketDataAgent(Agent):
    name = "market"

    def __init__(self, cfg, journal, exchange):
        super().__init__(cfg, journal)
        self.ex = exchange
        self.tf_ms = TIMEFRAME_MINUTES[cfg.timeframe] * 60_000
        self.raw: dict[str, pd.DataFrame] = {}
        self.feat: dict[str, pd.DataFrame] = {}
        self.daily: dict[str, pd.DataFrame] = {}
        self.daily_day: dict[str, int] = {}

    def _refresh_daily(self, sym: str, now_ms: float) -> bool:
        days = int(self.cfg.htf_sma_days or 0)
        today = int(now_ms // 86_400_000)
        if not days or self.daily_day.get(sym) == today:
            return False
        d = to_frame(self.ex.fetch_ohlcv(sym, "1d", limit=min(720, days + 30)))
        self.daily[sym] = d[d.index + 86_400_000 <= now_ms]
        self.daily_day[sym] = today
        return True

    def _refresh_candles(self, sym: str, now_ms: float) -> bool:
        df = self.raw.get(sym)
        if df is not None and now_ms < df.index[-1] + 2 * self.tf_ms:
            return False  # the next candle has not closed yet
        since = None if df is None else int(df.index[-1])
        new = to_frame(self.ex.fetch_ohlcv(sym, self.cfg.timeframe, since=since, limit=720))
        new = new[new.index + self.tf_ms <= now_ms]  # drop the still-forming candle
        merged = new if df is None else pd.concat([df, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index().tail(1000)
        changed = df is None or merged.index[-1] != df.index[-1]
        self.raw[sym] = merged
        if changed:
            self._build(sym)
        return changed

    def _build(self, sym: str) -> None:
        feat = add_indicators(self.raw[sym], mom_bars=int(7 * bars_per_day(self.cfg.timeframe)))
        if self.cfg.htf_sma_days:
            feat = add_htf(feat, self.daily.get(sym), int(self.cfg.htf_sma_days))
        self.feat[sym] = feat

    def run(self, ctx: Context) -> str:
        tickers = self.ex.fetch_tickers(self.cfg.symbols)
        for sym in self.cfg.symbols:
            t = tickers.get(sym) or {}
            px = t.get("last") or t.get("close") or t.get("bid")
            if px:
                ctx.prices[sym] = float(px)
        now_ms = ctx.now * 1000
        for sym in self.cfg.symbols:
            daily_changed = self._refresh_daily(sym, now_ms)
            ctx.new_bar[sym] = self._refresh_candles(sym, now_ms)
            if daily_changed and not ctx.new_bar[sym] and sym in self.raw:
                self._build(sym)
            if sym in self.feat:
                ctx.candles[sym] = self.feat[sym]
        missing = [s for s in self.cfg.symbols if s not in ctx.prices or s not in ctx.candles]
        if missing:
            raise RuntimeError(f"no data for {missing}")
        return " ".join(f"{s.split('/')[0]}={ctx.prices[s]:,.2f}" for s in self.cfg.symbols)
