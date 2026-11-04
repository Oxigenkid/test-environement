"""Pure trade-management and position-sizing logic shared by live trading and the backtester."""
from __future__ import annotations

from dataclasses import dataclass

from .strategies import Signal

MAX_BARS_IN_TRADE = 48


@dataclass
class SizeDecision:
    qty: float
    cost: float
    risk_amount: float
    reason: str  # "ok" or why it was rejected


def plan_entries(signals: list[Signal], open_syms: set[str], slots: int,
                 blocked: set[tuple[str, str]] | None = None) -> list[Signal]:
    """Best-scored signal per symbol, up to `slots`, skipping symbols already held and blocked (symbol, strategy)
    pairs (cool-downs from the learner). `signals` must already be sorted best first."""
    blocked = blocked or set()
    out, seen = [], set(open_syms)
    for s in signals:
        if len(out) >= slots:
            break
        if s.symbol in seen or (s.symbol, s.strategy) in blocked:
            continue
        out.append(s)
        seen.add(s.symbol)
    return out


def size_position(sig: Signal, equity: float, cash: float, risk: dict, max_boost: float,
                  min_amount: float = 0.0, min_cost: float = 0.0, split: int = 1) -> SizeDecision:
    """Position sizing.

    allocation=full: commit all available cash, divided over the `split` entries taken this beat.
    allocation=risk: volatility/stop-based sizing; intel boosts never push risk above max_risk_per_trade.
    """
    entry = sig.price * (1 + risk["slippage"])
    stop_dist = entry - sig.stop
    if stop_dist <= 0:
        return SizeDecision(0, 0, 0, "invalid stop")
    round_trip_fees = 2 * risk["fee_rate"]
    reward = (sig.target - entry) / entry
    if reward < risk["min_reward_to_fees"] * round_trip_fees:
        return SizeDecision(0, 0, 0, f"reward {reward:.2%} too small vs fees")
    spendable = cash / (1 + risk["fee_rate"]) * 0.995
    if risk.get("allocation", "full") == "full":
        budget = spendable / max(1, split)
        if budget <= 0 or budget < 0.02 * equity:
            return SizeDecision(0, 0, 0, "no cash available (already fully invested)")
        qty = budget / entry
        if qty < min_amount or budget < min_cost:
            return SizeDecision(0, 0, 0, f"exchange minimum exceeds available cash ${budget:.2f}")
        return SizeDecision(qty, qty * entry, qty * stop_dist, "ok")
    max_risk_amt = equity * risk["max_risk_per_trade"]
    risk_amt = max_risk_amt / (1 + max_boost) * sig.intel_mult
    if sig.regime == "high_vol":
        risk_amt *= 0.5
    risk_amt = min(risk_amt, max_risk_amt)
    qty = risk_amt / stop_dist
    cap = min(equity * risk["max_position_fraction"], spendable)
    qty = min(qty, cap / entry)
    if qty < min_amount or qty * entry < min_cost:
        qty = max(min_amount, min_cost / entry) * 1.01
        if qty * entry > cap:
            return SizeDecision(0, 0, 0, f"exchange minimum ${qty * entry:.2f} exceeds position cap ${cap:.2f}")
        if qty * stop_dist > max_risk_amt:
            return SizeDecision(0, 0, 0, "exchange minimum would exceed max risk per trade")
    if qty <= 0:
        return SizeDecision(0, 0, 0, "no cash available")
    return SizeDecision(qty, qty * entry, qty * stop_dist, "ok")


def update_trailing(trade: dict, high: float) -> dict:
    """Raise the stop behind new highs for trailing strategies. Returns the changed fields."""
    changes = {}
    highest = max(trade.get("highest") or trade["entry_price"], high)
    if highest != trade.get("highest"):
        changes["highest"] = highest
    if trade.get("trail_atr"):
        new_stop = highest - trade["trail_atr"]
        if new_stop > trade["stop"]:
            changes["stop"] = new_stop
    return changes


def check_exit(trade: dict, low: float, high: float, close: float, row: dict | None, bars_held: int | None):
    """Return (reason, exit_price) or None. If a bar touches both stop and target, assume the stop (conservative)."""
    if low <= trade["stop"]:
        return "stop", min(trade["stop"], high) if high >= trade["stop"] else high
    if high >= trade["target"] and not trade.get("trail_atr"):
        return "target", trade["target"]
    if row is not None:
        if (trade["strategy"] in ("trend", "momentum") and row.get("ema50")
                and close < row["ema50"] and trade["entry_price"] >= row["ema50"]):
            return "trend_broken", close
        if trade["strategy"] == "meanrev" and row.get("bb_mid") and close >= row["bb_mid"] and close > trade["entry_price"]:
            return "mean_reached", close
        if bars_held is not None and bars_held >= MAX_BARS_IN_TRADE and close < trade["entry_price"] * 1.005:
            return "time_stop", close
    return None


def strategy_weights(closed_trades: list[dict], lookback: int = 20) -> dict[str, float]:
    """Meta-allocator: tilt toward strategies with better recent profit factor (0.5x to 1.5x)."""
    by: dict[str, list[float]] = {}
    for t in closed_trades:  # newest first
        lst = by.setdefault(t["strategy"], [])
        if len(lst) < lookback and t.get("pnl") is not None:
            lst.append(t["pnl"])
    weights = {}
    for name, pnls in by.items():
        gains = sum(p for p in pnls if p > 0)
        losses = -sum(p for p in pnls if p < 0)
        if len(pnls) < 5:
            # Respond immediately to early wins/losses with a gentle prior
            wins = sum(1 for p in pnls if p > 0)
            fails = sum(1 for p in pnls if p < 0)
            weights[name] = max(0.5, min(1.5, 1.0 + 0.12 * (wins - fails)))
            continue
        pf = gains / losses if losses > 0 else 3.0
        weights[name] = max(0.5, min(1.5, 0.5 + pf / 2))
    return weights


def trade_stats(pnls: list[float], start_equity: float) -> dict:
    eq = start_equity
    peak = eq
    max_dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak if peak > 0 else 0)
    gains = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    return {
        "trades": len(pnls),
        "win_rate": (sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else 0.0,
        "profit_factor": (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0),
        "net_pnl": sum(pnls),
        "max_drawdown": max_dd,
    }
