"""Bar-by-bar backtester that reuses the live regime, strategy, allocator, sizing and exit logic.

Historical news is not available, so intel multipliers are neutral (1.0) in backtests.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .config import Config
from .exchange import make_exchange, market_limits
from .indicators import TIMEFRAME_MINUTES, add_htf, add_indicators, bars_per_day
from .logic import check_exit, plan_entries, size_position, strategy_weights, trade_stats, update_trailing
from .strategies import commit_signals, generate_signals

CACHE = Path("data/cache")


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = f"{TIMEFRAME_MINUTES[timeframe]}min"
    d = df.copy()
    d.index = pd.to_datetime(d.index, unit="ms")
    out = d.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    out.index = (out.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)  # pandas 2 (ns) and 3 (ms) safe
    out.index.name = "ts"
    return out


def fetch_history(exchange_id: str, symbol: str, timeframe: str, days: int, verbose: bool = True) -> pd.DataFrame:
    ex = make_exchange(exchange_id)
    if ex.timeframes and timeframe not in ex.timeframes:
        base = "1h" if TIMEFRAME_MINUTES[timeframe] >= 60 else "1m"
        return resample(fetch_history(exchange_id, symbol, base, days, verbose), timeframe)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{exchange_id}_{symbol.replace('/', '-')}_{timeframe}_{days}d.csv"
    if path.exists() and time.time() - path.stat().st_mtime < 6 * 3600:
        return pd.read_csv(path, index_col="ts")
    tf_ms = TIMEFRAME_MINUTES[timeframe] * 60_000
    since = int((time.time() - days * 86400) * 1000)
    end = int(time.time() * 1000)
    rows: list[list] = []
    while since < end:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=300)
        if not batch:
            since += 300 * tf_ms
            continue
        rows.extend(batch)
        nxt = batch[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
    if verbose:
        print(f"  fetched {len(rows)} {timeframe} candles for {symbol} from {exchange_id}")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").set_index("ts").sort_index().astype(float)
    df = df[df.index + tf_ms <= end]
    df.to_csv(path)
    return df


def run_backtest(cfg: Config, data: dict[str, pd.DataFrame], start_balance: float | None = None,
                 limits: dict[str, tuple[float, float]] | None = None, stop_at_target: bool = False,
                 daily: dict[str, pd.DataFrame] | None = None) -> dict:
    r = dict(cfg.risk)
    boost = cfg.intel["max_size_boost"]
    tf = cfg.timeframe
    limits = limits or {}
    sma_days = int(cfg.htf_sma_days or 0)
    feats = {}
    for s, df in data.items():
        f = add_indicators(df, mom_bars=int(7 * bars_per_day(tf)))
        if sma_days:
            d = daily[s] if daily and s in daily else resample(df, "1d")
            if len(d) >= sma_days:
                f = add_htf(f, d, sma_days)
        feats[s] = f.to_dict("index")
    timeline = sorted(set().union(*[set(f.keys()) for f in feats.values()]))
    warmup = 120
    cash = float(start_balance if start_balance is not None else cfg.paper_starting_balance)
    start = cash
    open_t: dict[str, dict] = {}
    closed: list[dict] = []
    curve: list[tuple[float, float]] = []
    peak = cash
    day_key, day_start = None, cash
    halted_at = None
    target_at = None
    last_px: dict[str, float] = {}
    cooldowns: dict[tuple[str, str], float] = {}
    cd_ms = float(cfg.learning.get("cooldown_hours", 12)) * 3_600_000

    def close(sym, t, px, ts, reason):
        nonlocal cash
        px *= 1 - r["slippage"]
        fee = px * t["qty"] * r["fee_rate"]
        cash += px * t["qty"] - fee
        pnl = px * t["qty"] - fee - (t["entry_price"] * t["qty"] + t["entry_fee"])
        t.update(exit_ts=ts, exit_price=px, exit_reason=reason, exit_fee=fee, pnl=pnl)
        closed.append(t)
        if pnl < 0 and cfg.learning.get("enabled", True):
            cooldowns[(sym, t["strategy"])] = ts + cd_ms

    for i, ts in enumerate(timeline):
        rows = {s: f[ts] for s, f in feats.items() if ts in f}
        for s, row in rows.items():
            last_px[s] = row["close"]
        # exits
        for sym in list(open_t):
            row = rows.get(sym)
            if row is None:
                continue
            t = open_t[sym]
            t["bars"] += 1
            ex = check_exit(t, row["low"], row["high"], row["close"], row, t["bars"])
            if ex:
                reason, px = ex
                if reason == "stop":
                    px = min(t["stop"], row["open"])
                close(sym, t, px, ts, reason)
                del open_t[sym]
            else:
                t.update(update_trailing(t, row["high"]))
        equity = cash + sum(t["qty"] * last_px.get(s, t["entry_price"]) for s, t in open_t.items())
        curve.append((ts, equity))
        peak = max(peak, equity)
        dk = int(ts // 86_400_000)
        if dk != day_key:
            day_key, day_start = dk, equity
        if equity >= cfg.target_equity and target_at is None:
            target_at = ts
            if stop_at_target:
                break
        if 1 - equity / peak >= r["kill_switch_drawdown"]:
            for sym, t in list(open_t.items()):
                close(sym, t, last_px[sym], ts, "kill_switch")
            open_t.clear()
            halted_at = ts
            curve.append((ts, cash))
            break
        if i < warmup or 1 - equity / day_start >= r["daily_loss_limit"]:
            continue
        slots = r["max_open_positions"] - len(open_t)
        if slots <= 0:
            continue
        benched = {k for k, until in cooldowns.items() if until > ts}
        weights = strategy_weights(list(reversed(closed[-200:])))
        sigs = []
        for sym, row in rows.items():
            if sym in open_t:
                continue
            _, ss = generate_signals(sym, row, r, cfg.strategies)
            for s in ss:
                if (s.symbol, s.strategy) in benched:
                    continue
                s.score = s.strength * weights.get(s.strategy, 1.0)
                sigs.append(s)
        if not sigs and not open_t and r.get("commit_when_flat", True):
            fresh, cooled = [], []
            for sym, row in rows.items():
                for s in commit_signals(sym, row, r, cfg.strategies):
                    s.score = s.strength * weights.get(s.strategy, 1.0)
                    if (s.symbol, s.strategy) in benched:
                        s.score *= 0.5
                        cooled.append(s)
                    else:
                        fresh.append(s)
            pool = fresh or cooled
            if pool:
                pool.sort(key=lambda s: s.score, reverse=True)
                sigs = [pool[0]]
        sigs.sort(key=lambda s: s.score, reverse=True)
        plan = plan_entries(sigs, set(open_t), slots, benched if len(sigs) > 1 else None)
        for k, s in enumerate(plan):
            min_amt, min_cost = limits.get(s.symbol, (0.0, 1.0))
            d = size_position(s, equity, cash, r, boost, min_amt, min_cost, split=len(plan) - k)
            if d.reason != "ok" and len(plan) - k > 1:
                d = size_position(s, equity, cash, r, boost, min_amt, min_cost, split=1)
            if d.reason != "ok":
                continue
            px = s.price * (1 + r["slippage"])
            fee = px * d.qty * r["fee_rate"]
            if px * d.qty + fee > cash:
                continue
            cash -= px * d.qty + fee
            open_t[s.symbol] = {"symbol": s.symbol, "strategy": s.strategy, "regime": s.regime, "entry_ts": ts,
                                "entry_price": px, "qty": d.qty, "stop": s.stop, "target": s.target,
                                "trail_atr": s.trail_atr, "highest": px, "entry_fee": fee, "bars": 0}
            slots -= 1

    final = curve[-1][1] if curve else start
    pnls = [t["pnl"] for t in closed]
    stats = trade_stats(pnls, start)
    eq_series = pd.Series([e for _, e in curve])
    max_dd = float((1 - eq_series / eq_series.cummax()).max()) if len(eq_series) else 0.0
    span_days = (timeline[-1] - timeline[0]) / 86_400_000 if timeline else 0
    by_strat = {}
    for t in closed:
        b = by_strat.setdefault(t["strategy"], [])
        b.append(t["pnl"])
    return {
        "start": start, "final": final, "return": final / start - 1, "days": span_days,
        "trades": stats["trades"], "win_rate": stats["win_rate"], "profit_factor": stats["profit_factor"],
        "max_drawdown": max_dd, "fees": sum(t["entry_fee"] + t["exit_fee"] for t in closed),
        "days_to_target": ((target_at - timeline[0]) / 86_400_000) if target_at else None,
        "halted_days": ((halted_at - timeline[0]) / 86_400_000) if halted_at else None,
        "by_strategy": {k: trade_stats(v, start) for k, v in by_strat.items()},
        "curve": curve, "closed": closed,
    }


def load_limits(cfg: Config) -> dict[str, tuple[float, float]]:
    try:
        ex = make_exchange(cfg.exchange)
        ex.load_markets()
        return {s: market_limits(ex, s) for s in cfg.symbols}
    except Exception:
        return {}


def format_result(name: str, res: dict) -> str:
    pf = res["profit_factor"]
    lines = [
        f"== {name} ==",
        f"  period        {res['days']:.0f} days",
        f"  equity        ${res['start']:.2f} -> ${res['final']:.2f} ({res['return']:+.1%})",
        f"  trades        {res['trades']}  win rate {res['win_rate']:.0%}  profit factor {pf:.2f}",
        f"  max drawdown  {res['max_drawdown']:.1%}   fees paid ${res['fees']:.2f}",
        f"  target hit    {'after %.0f days' % res['days_to_target'] if res['days_to_target'] else 'no'}",
    ]
    if res["halted_days"]:
        lines.append(f"  KILL-SWITCH   tripped after {res['halted_days']:.0f} days")
    for k, v in res["by_strategy"].items():
        lines.append(f"    {k:9} {v['trades']:4} trades  pf {v['profit_factor']:.2f}  net ${v['net_pnl']:+.2f}")
    return "\n".join(lines)
