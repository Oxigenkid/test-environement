"""Self-improvement: learn from failures.

Three feedback loops, from fastest to slowest:

1. Post-mortem: every losing trade is diagnosed (wrong from the start, gave back gains, fee drag...) and
   written to the `lessons` table.
2. Cool-down: a symbol+strategy pair that loses `max_consecutive_losses` times in a row is benched for
   `cooldown_hours`, so the bot stops repeating the same mistake while conditions are against it.
3. Re-tune: after a losing streak, a kill-switch, or every N trades, the parameter grid is replayed on recent
   history (first 2/3 train, last 1/3 validation). New parameters are adopted only if they beat the current
   ones on the blended score by `min_improvement`, which keeps it from chasing noise.

Adopted parameters live in the journal state `adaptive_params` and override config.yaml at runtime.
"""
from __future__ import annotations

import itertools
import time

import pandas as pd

from .config import Config, _merge

ADAPTIVE_KEY = "adaptive_params"
COOLDOWN_KEY = "cooldowns"
TUNABLE_RISK = ("stop_atr_mult", "target_atr_mult", "max_open_positions")
TRAIN_FRACTION = 2 / 3
VALIDATION_WARMUP_BARS = 150  # the backtester skips entries for its first 120 bars


# ---------------------------------------------------------------- runtime parameters
def effective_config(cfg: Config, journal) -> Config:
    """config.yaml merged with whatever the learner has adopted."""
    p = journal.get_state(ADAPTIVE_KEY) or {}
    over = params_to_overrides(p) if p else {}
    return Config(_merge(cfg.raw, over)) if over else cfg


def current_params(cfg: Config) -> dict:
    r = cfg.risk
    return {"stop_atr_mult": float(r["stop_atr_mult"]), "target_atr_mult": float(r["target_atr_mult"]),
            "max_open_positions": int(r["max_open_positions"]), "strategies": list(cfg.strategies)}


def params_to_overrides(p: dict) -> dict:
    over: dict = {}
    risk = {k: p[k] for k in TUNABLE_RISK if k in p}
    if risk:
        over["risk"] = risk
    if p.get("strategies"):
        over["strategies"] = list(p["strategies"])
    return over


def active_cooldowns(journal, now: float) -> set[tuple[str, str]]:
    cds = journal.get_state(COOLDOWN_KEY) or {}
    return {tuple(k.split("|", 1)) for k, until in cds.items() if until > now}


# ---------------------------------------------------------------- post-mortems
def diagnose(trade: dict) -> dict:
    """Classify why a trade lost money and what to take from it."""
    entry, qty = trade["entry_price"], trade["qty"]
    cost = entry * qty or 1.0
    pnl = trade.get("pnl") or 0.0
    fees = (trade.get("entry_fee") or 0) + (trade.get("exit_fee") or 0)
    highest = trade.get("highest") or entry
    mfe = highest / entry - 1  # best unrealised gain while the trade was open
    planned = trade["target"] / entry - 1 if trade.get("target") else 0.0
    reason = trade.get("exit_reason") or "?"
    if pnl < 0 and -pnl <= fees * 1.2:
        kind, lesson = "fee_drag", "the move was too small to cover fees; demand a larger expected gain"
    elif planned > 0 and mfe >= 0.5 * planned:
        kind, lesson = "gave_back_gains", "price reached half the target then reversed; trail tighter / take profit sooner"
    elif mfe < 0.003:
        kind, lesson = "wrong_from_start", f"never went in our favour ({reason}); entry signal or regime was wrong"
    elif reason in ("kill_switch", "flatten"):
        kind, lesson = "forced_exit", "closed by the risk guardian during a drawdown"
    else:
        kind, lesson = "stopped_out", f"moved {mfe:+.2%} in our favour, then hit the {reason}; stop may be too tight"
    return {"kind": kind, "lesson": lesson, "loss_pct": pnl / cost, "mfe": mfe, "exit_reason": reason}


def losing_streak(closed: list[dict], symbol: str | None = None, strategy: str | None = None) -> int:
    """Consecutive losses, newest first, optionally for one symbol+strategy."""
    n = 0
    for t in closed:
        if symbol and (t["symbol"] != symbol or t["strategy"] != strategy):
            continue
        if (t.get("pnl") or 0) < 0:
            n += 1
        else:
            break
    return n


def update_cooldown(journal, closed: list[dict], trade: dict, now: float, lc: dict) -> float | None:
    """Bench a symbol+strategy after too many consecutive losses. Returns the cool-down end or None."""
    streak = losing_streak(closed, trade["symbol"], trade["strategy"])
    if streak < lc["max_consecutive_losses"]:
        return None
    until = now + lc["cooldown_hours"] * 3600
    cds = {k: v for k, v in (journal.get_state(COOLDOWN_KEY) or {}).items() if v > now}
    cds[f"{trade['symbol']}|{trade['strategy']}"] = until
    journal.set_state(COOLDOWN_KEY, cds)
    return until


def adapt_on_loss(cfg: Config, journal, diagnosis: dict) -> dict:
    """Immediately refine stop/target multipliers from a trade's post-mortem diagnosis."""
    eff = effective_config(cfg, journal)
    p = current_params(eff)
    kind = diagnosis["kind"]
    if kind == "gave_back_gains":
        p["target_atr_mult"] = max(2.5, round(p["target_atr_mult"] * 0.85, 2))
        p["stop_atr_mult"] = max(1.5, min(p["target_atr_mult"] - 0.5, round(p["stop_atr_mult"] * 0.9, 2)))
    elif kind == "stopped_out":
        p["stop_atr_mult"] = min(4.5, round(p["stop_atr_mult"] * 1.15, 2))
        p["target_atr_mult"] = max(p["target_atr_mult"], round(p["stop_atr_mult"] * 1.8, 2))
    elif kind == "fee_drag":
        p["target_atr_mult"] = min(9.0, round(p["target_atr_mult"] * 1.2, 2))
    elif kind == "wrong_from_start":
        p["stop_atr_mult"] = max(1.8, round(p["stop_atr_mult"] * 0.92, 2))
    journal.set_state(ADAPTIVE_KEY, p)
    return p


# ---------------------------------------------------------------- re-tuning
def candidate_params(grid: dict) -> list[dict]:
    keys = [k for k in ("stop_atr_mult", "target_atr_mult", "max_open_positions", "strategies") if k in grid]
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = dict(zip(keys, combo))
        if p.get("target_atr_mult", 1) <= p.get("stop_atr_mult", 0):
            continue  # reward must exceed risk
        out.append(p)
    return out


def score(res: dict) -> float:
    """Return, penalised for drawdown, kill-switches and doing (almost) nothing."""
    s = res["return"] - 0.5 * res["max_drawdown"]
    if res.get("halted_days"):
        s -= 0.25
    if res["trades"] < 3:
        s -= 0.02
    return s


def split_data(data: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    train, val = {}, {}
    for s, df in data.items():
        cut = int(len(df) * TRAIN_FRACTION)
        train[s] = df.iloc[:cut]
        val[s] = df.iloc[max(0, cut - VALIDATION_WARMUP_BARS):]
    return train, val


def evaluate(cfg: Config, params: dict, train: dict, val: dict, daily: dict | None, limits: dict | None) -> dict:
    from .backtest import run_backtest

    c = Config(_merge(cfg.raw, params_to_overrides(params)))
    rt = run_backtest(c, train, 100.0, limits, daily=daily)
    rv = run_backtest(c, val, 100.0, limits, daily=daily)
    st, sv = score(rt), score(rv)
    return {"params": params, "score": 0.4 * st + 0.6 * sv, "train": st, "validation": sv,
            "val_return": rv["return"], "val_drawdown": rv["max_drawdown"], "val_trades": rv["trades"],
            "val_pf": rv["profit_factor"] if rv["profit_factor"] != float("inf") else 99.0}


def tune(cfg: Config, data: dict[str, pd.DataFrame], daily: dict | None = None,
         limits: dict | None = None, current: dict | None = None) -> dict:
    """Search the grid; return the best candidate and whether it beats the current parameters."""
    lc = cfg.learning
    current = current or current_params(cfg)
    train, val = split_data(data)
    base = evaluate(cfg, current, train, val, daily, limits)
    results = [evaluate(cfg, p, train, val, daily, limits) for p in candidate_params(lc["grid"])]
    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0] if results else base
    adopted = best["score"] > base["score"] + lc["min_improvement"] and best["params"] != current
    return {"ts": time.time(), "adopted": adopted, "current": base, "best": best, "top": results[:5],
            "candidates": len(results)}


def load_tuning_data(cfg: Config) -> tuple[dict, dict | None]:
    from .backtest import fetch_history

    days = int(cfg.learning["tune_days"])
    sma = int(cfg.htf_sma_days or 0)
    data, daily = {}, {}
    for s in cfg.symbols:
        src = s if cfg.quote == "USD" else s.replace("/" + cfg.quote, "/USD")
        data[s] = fetch_history(cfg.backtest["exchange"], src, cfg.timeframe, days, verbose=False)
        if sma:
            daily[s] = fetch_history(cfg.backtest["exchange"], src, "1d", days + sma + 5, verbose=False)
    return data, daily or None


def format_tuning(res: dict) -> str:
    def fmt(r):
        p = r["params"]
        return (f"stop {p['stop_atr_mult']}ATR target {p['target_atr_mult']}ATR max_pos {p['max_open_positions']} "
                f"{'+'.join(p['strategies'])} | score {r['score']:+.3f} (train {r['train']:+.3f}, "
                f"val {r['validation']:+.3f}, val return {r['val_return']:+.1%}, dd {r['val_drawdown']:.1%}, "
                f"{r['val_trades']} trades)")
    lines = [f"current : {fmt(res['current'])}", f"best    : {fmt(res['best'])}",
             f"decision: {'ADOPT new parameters' if res['adopted'] else 'keep current parameters'} "
             f"({res['candidates']} candidates tested)"]
    return "\n".join(lines)
