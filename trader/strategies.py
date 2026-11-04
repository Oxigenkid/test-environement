"""Regime detection and strategy rules (pure functions shared by live trading and the backtester)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

REGIMES = ("trend_up", "trend_down", "range", "high_vol", "crisis", "bear", "unknown")

# Which strategies may open trades in which regime. bear / trend_down / crisis => stay in cash (spot, long-only).
ALLOWED = {
    "trend_up": ("trend", "momentum"),
    "range": ("meanrev",),
    "high_vol": ("trend",),
    "trend_down": (),
    "crisis": (),
    "bear": (),
    "unknown": (),
}


@dataclass
class Signal:
    symbol: str
    strategy: str
    regime: str
    price: float
    stop: float
    target: float
    strength: float
    trail_atr: float | None = None
    score: float = 0.0
    intel_mult: float = 1.0
    notes: dict = field(default_factory=dict)


def _ok(*vals) -> bool:
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals)


def detect_regime(row: dict) -> str:
    if "htf_ok" in row:  # daily long-term trend filter (absent only when disabled)
        if not _ok(row["htf_ok"]):
            return "unknown"
        if row["htf_ok"] < 0.5:
            return "bear"
    keys = ("close", "ema20", "ema50", "ema100", "adx14", "atr_rank", "dd_long")
    if not _ok(*(row.get(k) for k in keys)):
        return "unknown"
    if row["dd_long"] < -0.15 and row["atr_rank"] > 0.9 and row["close"] < row["ema100"]:
        return "crisis"
    if row["atr_rank"] > 0.9:
        return "high_vol"
    if row["adx14"] > 22 and row["ema20"] > row["ema50"] and row["close"] > row["ema100"]:
        return "trend_up"
    if row["adx14"] > 22 and row["ema20"] < row["ema50"]:
        return "trend_down"
    return "range"


def _min_target_dist(c: float, risk: dict) -> float:
    slip = risk.get("slippage", 0.0005)
    min_rew = risk.get("min_reward_to_fees", 3.0) * 2 * risk.get("fee_rate", 0.004)
    return c * ((1 + slip) * (1 + min_rew * 1.05) - 1)


def trend_signal(symbol: str, row: dict, regime: str, risk: dict) -> Signal | None:
    if not _ok(row.get("dc_high20"), row.get("atr14"), row.get("adx14")):
        return None
    c, a = row["close"], row["atr14"]
    if c > row["dc_high20"] and row["ema20"] > row["ema50"] and c > row["ema100"]:
        strength = max(0.2, min(1.0, (row["adx14"] - 15) / 25))
        tgt = c + max(risk["target_atr_mult"] * a, _min_target_dist(c, risk))
        return Signal(symbol, "trend", regime, c, c - risk["stop_atr_mult"] * a, tgt,
                      strength, trail_atr=risk["stop_atr_mult"] * a)
    return None


def meanrev_signal(symbol: str, row: dict, regime: str, risk: dict) -> Signal | None:
    if not _ok(row.get("rsi14"), row.get("bb_low"), row.get("bb_mid"), row.get("atr14")):
        return None
    c, a = row["close"], row["atr14"]
    if row["rsi14"] < 30 and c < row["bb_low"]:
        strength = max(0.2, min(1.0, (30 - row["rsi14"]) / 15 + 0.3))
        tgt = max(row["bb_mid"], c + 1.5 * a, c + _min_target_dist(c, risk))
        return Signal(symbol, "meanrev", regime, c, c - 1.5 * a, tgt, strength)
    return None


def momentum_signal(symbol: str, row: dict, regime: str, risk: dict) -> Signal | None:
    if not _ok(row.get("ret_mom"), row.get("prev_close"), row.get("prev_ema20"), row.get("rsi14")):
        return None
    c, a = row["close"], row["atr14"]
    pullback_reclaim = row["prev_close"] < row["prev_ema20"] and c > row["ema20"]
    if row["ret_mom"] > 0.05 and c > row["ema50"] and 50 < row["rsi14"] < 70 and pullback_reclaim:
        strength = max(0.2, min(1.0, row["ret_mom"] / 0.2))
        tgt = c + max(risk["target_atr_mult"] * a, _min_target_dist(c, risk))
        return Signal(symbol, "momentum", regime, c, c - risk["stop_atr_mult"] * a, tgt,
                      strength, trail_atr=risk["stop_atr_mult"] * a)
    return None


STRATEGIES = {"trend": trend_signal, "meanrev": meanrev_signal, "momentum": momentum_signal}


def generate_signals(symbol: str, row: dict, risk: dict,
                     enabled: list[str] | tuple[str, ...] | None = None) -> tuple[str, list[Signal]]:
    regime = detect_regime(row)
    out = []
    for name in ALLOWED.get(regime, ()):
        if enabled is not None and name not in enabled:
            continue
        sig = STRATEGIES[name](symbol, row, regime, risk)
        if sig and sig.stop < sig.price < sig.target:
            out.append(sig)
    return regime, out


def commit_signals(symbol: str, row: dict, risk: dict,
                   enabled: list[str] | tuple[str, ...] | None = None) -> list[Signal]:
    """Build ranked commitment signals across enabled strategies when the account is flat and no rare
    breakout/extreme-RSI trigger fired on the bar. Avoids sitting idle in cash."""
    regime = detect_regime(row)
    if regime == "unknown" and "htf_ok" in row:
        regime = detect_regime({k: v for k, v in row.items() if k != "htf_ok"})
    if regime in ("crisis", "unknown"):
        return []
    keys = ("close", "atr14", "ema20", "ema50", "rsi14", "adx14")
    if not _ok(*(row.get(k) for k in keys)) or row["close"] <= 0 or row["atr14"] <= 0:
        return []
    c, a = float(row["close"]), float(row["atr14"])
    ema20, ema50 = float(row["ema20"]), float(row["ema50"])
    rsi, adx = float(row["rsi14"]), float(row["adx14"])
    ret_mom = float(row["ret_mom"]) if _ok(row.get("ret_mom")) else 0.0
    bb_mid = float(row["bb_mid"]) if _ok(row.get("bb_mid")) else ema20
    min_dist = _min_target_dist(c, risk)
    stop_mult = float(risk["stop_atr_mult"])
    tgt_mult = float(risk["target_atr_mult"])
    names = list(enabled) if enabled is not None else ["trend", "momentum", "meanrev"]

    out: list[Signal] = []
    for name in names:
        if name == "trend":
            above = c >= ema50 and ema20 >= ema50
            base = 0.58 if above else 0.28
            strength = max(0.2, min(0.95, base + max(-0.2, min(0.25, (adx - 20) / 40))
                                    + max(-0.1, min(0.15, (c / ema20 - 1) * 8))))
            stop = c - stop_mult * a
            tgt = c + max(tgt_mult * a, min_dist)
            sig = Signal(symbol, "trend", regime, c, stop, tgt, strength, trail_atr=stop_mult * a)
        elif name == "momentum":
            above = c >= ema50
            base = 0.54 if above else 0.26
            rsi_bonus = 0.12 if 45 <= rsi <= 68 else -0.10
            strength = max(0.2, min(0.95, base + max(-0.2, min(0.3, ret_mom * 3.5)) + rsi_bonus))
            stop = c - stop_mult * a
            tgt = c + max(tgt_mult * a, min_dist)
            sig = Signal(symbol, "momentum", regime, c, stop, tgt, strength, trail_atr=stop_mult * a)
        elif name == "meanrev":
            dip = (52 - rsi) / 38 + (0.12 if c < bb_mid else -0.04)
            strength = max(0.2, min(0.95, 0.52 + max(-0.25, min(0.38, dip))))
            mr_stop = max(1.5, stop_mult * 0.75) * a
            stop = c - mr_stop
            tgt = max(bb_mid, c + max(1.5 * a, tgt_mult * 0.6 * a, min_dist))
            sig = Signal(symbol, "meanrev", regime, c, stop, tgt, strength, trail_atr=None)
        else:
            continue
        if sig.stop < sig.price < sig.target:
            out.append(sig)
    out.sort(key=lambda s: s.strength, reverse=True)
    return out

