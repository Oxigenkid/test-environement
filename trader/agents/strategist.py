"""Regime detector + strategy ensemble + meta-allocator."""
from __future__ import annotations

from ..learning import active_cooldowns, effective_config
from ..logic import strategy_weights
from ..strategies import commit_signals, detect_regime, generate_signals
from .base import Agent, Context


class StrategistAgent(Agent):
    name = "strategist"

    def __init__(self, cfg, journal, mode: str):
        super().__init__(cfg, journal)
        self.mode = mode

    def _score_signal(self, s, intel: dict, weights: dict) -> None:
        mult = intel.get("mult", 1.0)
        if s.strategy == "meanrev" and intel.get("contrarian_fear"):
            mult *= 1.1
        s.intel_mult = min(1 + self.cfg.intel["max_size_boost"], mult)
        s.score = s.strength * weights.get(s.strategy, 1.0) * s.intel_mult
        s.notes = {"weight": weights.get(s.strategy, 1.0), "intel": intel}

    def run(self, ctx: Context) -> str:
        evaluated = self.journal.get_state("last_signal_bar", {}) or {}
        weights = strategy_weights(self.journal.closed_trades(self.mode, limit=200))
        eff = effective_config(self.cfg, self.journal)
        benched = active_cooldowns(self.journal, ctx.now)
        produced = []
        for sym, df in ctx.candles.items():
            row = df.iloc[-1].to_dict()
            ctx.regimes[sym] = detect_regime(row)
            bar_ts = str(int(df.index[-1]))
            if evaluated.get(sym) == bar_ts:
                continue  # each closed bar is evaluated once for standard breakout signals
            evaluated[sym] = bar_ts
            if sym in ctx.prices:
                row = {**row, "close": ctx.prices[sym]}
            _, sigs = generate_signals(sym, row, eff.risk, eff.strategies)
            intel = ctx.intel.get(sym, {"mult": 1.0, "veto": None})
            for s in sigs:
                if (s.symbol, s.strategy) in benched:
                    self.journal.log(self.name, f"{sym} {s.strategy} signal skipped: on cool-down after losses")
                    continue
                if intel.get("veto"):
                    self.journal.log(self.name, f"{sym} {s.strategy} signal vetoed: {intel['veto']}", "warn")
                    continue
                self._score_signal(s, intel, weights)
                produced.append(s)

        if (not produced and ctx.allow_entries and eff.risk.get("commit_when_flat", True)
                and not self.journal.open_trades(self.mode)):
            fresh, cooled = [], []
            for sym, df in ctx.candles.items():
                intel = ctx.intel.get(sym, {"mult": 1.0, "veto": None})
                if intel.get("veto"):
                    continue
                row = df.iloc[-1].to_dict()
                if sym in ctx.prices:
                    row = {**row, "close": ctx.prices[sym]}
                for s in commit_signals(sym, row, eff.risk, eff.strategies):
                    self._score_signal(s, intel, weights)
                    s.notes["committed"] = True
                    if (s.symbol, s.strategy) in benched:
                        s.score *= 0.5
                        cooled.append(s)
                    else:
                        fresh.append(s)
            pool = fresh or cooled
            if pool:
                pool.sort(key=lambda s: s.score, reverse=True)
                produced = [pool[0]]

        produced.sort(key=lambda s: s.score, reverse=True)
        ctx.signals = produced
        self.journal.set_state("last_signal_bar", evaluated)
        self.journal.set_state("regimes", ctx.regimes)
        self.journal.set_state("strategy_weights", weights)
        sig_txt = ", ".join(f"{s.symbol} {s.strategy} {s.score:.2f}" for s in produced) or "no signals"
        return f"regimes {ctx.regimes} | {sig_txt}"
