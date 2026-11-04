"""Vectorised indicators. Every column at row i only uses data up to and including row i."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(dn.notna())


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame, mom_bars: int = 168, rank_window: int = 500) -> pd.DataFrame:
    """df needs open/high/low/close/volume columns indexed by timestamp (ms)."""
    out = df.copy()
    c = out["close"]
    out["ema20"] = ema(c, 20)
    out["ema50"] = ema(c, 50)
    out["ema100"] = ema(c, 100)
    out["rsi14"] = rsi(c, 14)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / c
    out["adx14"] = adx(out, 14)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    out["bb_mid"] = mid
    out["bb_low"] = mid - 2 * sd
    out["bb_up"] = mid + 2 * sd
    out["dc_high20"] = out["high"].rolling(20).max().shift(1)
    out["ret_mom"] = c / c.shift(mom_bars) - 1
    win = min(rank_window, max(50, len(out) // 2))
    out["atr_rank"] = out["atr_pct"].rolling(win, min_periods=50).rank(pct=True)
    out["dd_long"] = c / c.rolling(win, min_periods=50).max() - 1
    out["prev_close"] = c.shift(1)
    out["prev_ema20"] = out["ema20"].shift(1)
    return out


def add_htf(df: pd.DataFrame, daily: pd.DataFrame | None, sma_days: int) -> pd.DataFrame:
    """Higher-timeframe risk filter: 1 when the last *completed* daily close is above its N-day SMA."""
    out = df.copy()
    if daily is None or len(daily) < sma_days:
        out["htf_ok"] = float("nan")
        return out
    c = daily["close"]
    sma = c.rolling(sma_days).mean()
    ok = (c > sma).astype(float).where(sma.notna())
    ok.index = ok.index + 86_400_000  # a daily bar is only known once it closes
    out["htf_ok"] = ok.reindex(ok.index.union(out.index)).ffill().reindex(out.index)
    return out


TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


def bars_per_day(timeframe: str) -> float:
    return 1440 / TIMEFRAME_MINUTES[timeframe]
