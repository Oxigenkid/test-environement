"""Execution agent: manages exits every heartbeat and opens new positions from ranked signals."""
from __future__ import annotations

import json

from ..indicators import TIMEFRAME_MINUTES
from ..learning import active_cooldowns, effective_config
from ..logic import check_exit, plan_entries, size_position, update_trailing
from .base import Agent, Context


class ExecutionAgent(Agent):
    name = "execution"

    def __init__(self, cfg, journal, broker):
        super().__init__(cfg, journal)
        self.broker = broker
        self.mode = broker.mode
        self.tf_s = TIMEFRAME_MINUTES[cfg.timeframe] * 60

    def close_trade(self, t: dict, price: float, reason: str) -> None:
        fill = self.broker.sell(t["symbol"], t["qty"], price)
        pnl = fill.price * fill.qty - fill.fee - (t["entry_price"] * t["qty"] + (t["entry_fee"] or 0))
        self.journal.execute(
            "UPDATE trades SET exit_ts=?, exit_price=?, exit_fee=?, pnl=?, exit_reason=? WHERE id=?",
            (self.ctx_now, fill.price, fill.fee, pnl, reason, t["id"]),
        )
        self.journal.log(self.name, f"EXIT {t['symbol']} {t['strategy']} @ {fill.price:.4f} ({reason}) pnl {pnl:+.4f}")

    def manage_exits(self, ctx: Context) -> list[str]:
        out = []
        for t in self.journal.open_trades(self.mode):
            sym = t["symbol"]
            price = ctx.prices.get(sym)
            if price is None:
                continue
            if ctx.flatten:
                self.close_trade(t, price, "flatten")
                out.append(f"flattened {sym}")
                continue
            row = None
            bars_held = None
            if ctx.new_bar.get(sym) and sym in ctx.candles:
                row = ctx.candles[sym].iloc[-1].to_dict()
                bars_held = int((ctx.now - t["entry_ts"]) // self.tf_s)
            close = row["close"] if row else price
            ex = check_exit(t, price, price, close, row, bars_held)
            if ex:
                self.close_trade(t, price, ex[0])
                out.append(f"closed {sym} ({ex[0]})")
                continue
            changes = update_trailing(t, price)
            if changes:
                sets = ", ".join(f"{k}=?" for k in changes)
                self.journal.execute(f"UPDATE trades SET {sets} WHERE id=?", (*changes.values(), t["id"]))
        return out

    def open_positions(self, ctx: Context) -> list[str]:
        out = []
        if not ctx.allow_entries or not ctx.signals:
            return out
        eff = effective_config(self.cfg, self.journal)
        risk = eff.risk
        open_syms = {t["symbol"] for t in self.journal.open_trades(self.mode)}
        slots = risk["max_open_positions"] - len(open_syms)
        blocked = active_cooldowns(self.journal, ctx.now)
        remaining = list(ctx.signals)
        while slots > 0:
            plan = plan_entries(remaining, open_syms, slots, blocked)
            if not plan:
                break
            sig = plan[0]
            remaining.remove(sig)
            price = ctx.prices.get(sig.symbol, sig.price)
            if sig.price and price != sig.price:
                ratio = price / sig.price
                sig.stop *= ratio
                sig.target *= ratio
                if sig.trail_atr:
                    sig.trail_atr *= ratio
            sig.price = price
            equity, cash = self.broker.equity(ctx.prices)
            min_amt, min_cost = self.broker.limits(sig.symbol)
            d = size_position(sig, equity, cash, risk, self.cfg.intel["max_size_boost"], min_amt, min_cost,
                              split=len(plan))
            if d.reason != "ok" and len(plan) > 1:
                d = size_position(sig, equity, cash, risk, self.cfg.intel["max_size_boost"], min_amt, min_cost, split=1)
            if d.reason != "ok":
                self.journal.log(self.name, f"skip {sig.symbol} {sig.strategy}: {d.reason}")
                continue
            qty = self.broker.precise(sig.symbol, d.qty)
            if qty <= 0 or qty < min_amt:
                self.journal.log(self.name, f"skip {sig.symbol}: size rounds below exchange minimum")
                continue
            fill = self.broker.buy(sig.symbol, qty, price)
            self.journal.execute(
                "INSERT INTO trades(mode,symbol,strategy,regime,entry_ts,entry_price,qty,stop,target,trail_atr,highest,"
                "entry_fee,intel) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.mode, sig.symbol, sig.strategy, sig.regime, ctx.now, fill.price, fill.qty, sig.stop, sig.target,
                 sig.trail_atr, fill.price, fill.fee, json.dumps({"mult": sig.intel_mult, "score": sig.score,
                                                                  **{k: v for k, v in sig.notes.items() if k != 'intel'}})),
            )
            alloc = f"{fill.price * fill.qty / equity:.0%} of equity" if equity > 0 else ""
            self.journal.log(self.name, f"ENTRY {sig.symbol} {sig.strategy} qty {fill.qty} @ {fill.price:.4f} "
                                        f"({alloc}, {risk.get('allocation', 'full')} allocation) "
                                        f"stop {sig.stop:.4f} target {sig.target:.4f} risk ${d.risk_amount:.3f} "
                                        f"intel x{sig.intel_mult:.2f}")
            open_syms.add(sig.symbol)
            slots -= 1
            out.append(f"opened {sig.symbol} {sig.strategy}")
        return out

    def run(self, ctx: Context) -> str:
        self.ctx_now = ctx.now
        exits = [] if getattr(ctx, "exits_done", False) else self.manage_exits(ctx)
        ctx.exits_done = True
        acts = exits + self.open_positions(ctx)
        return "; ".join(acts) or f"{len(self.journal.open_trades(self.mode))} open"
