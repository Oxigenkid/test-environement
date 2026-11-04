"""Risk guardian: equity tracking, kill-switch, daily loss limit and the capital-ladder target."""
from __future__ import annotations

import time

from ..learning import ADAPTIVE_KEY, effective_config
from .base import Agent, Context


class RiskAgent(Agent):
    name = "risk"

    def __init__(self, cfg, journal, broker):
        super().__init__(cfg, journal)
        self.broker = broker
        self.mode = broker.mode

    def _try_auto_resume(self, ctx: Context, equity: float) -> bool:
        lc = self.cfg.learning
        if not (lc["enabled"] and lc["auto_resume"]) or (self.mode == "live" and not lc["auto_resume_live"]):
            return False
        resume_at = self.journal.get_state("halt_resume_at")
        if not resume_at or ctx.now < resume_at or self.journal.get_state("tuning_running"):
            return False
        if self.journal.open_trades(self.mode):
            return False  # flatten first
        self.journal.execute("DELETE FROM state WHERE key IN ('halted', 'halt_resume_at', 'halt_handled')")
        self.journal.set_state(f"{self.mode}_peak", equity)
        params = self.journal.get_state(ADAPTIVE_KEY) or "config defaults"
        self.journal.log(self.name, f"auto-resume after kill-switch: equity {equity:.2f} is the new peak; "
                                    f"trading with re-tuned params {params}", "warn")
        return True

    def run(self, ctx: Context) -> str:
        r = effective_config(self.cfg, self.journal).risk
        equity, cash = self.broker.equity(ctx.prices)
        ctx.equity, ctx.cash = equity, cash
        self.journal.record_equity(self.mode, equity, cash)

        peak = max(self.journal.get_state(f"{self.mode}_peak", equity) or equity, equity)
        self.journal.set_state(f"{self.mode}_peak", peak)
        day = time.strftime("%Y-%m-%d", time.gmtime(ctx.now))
        day_state = self.journal.get_state(f"{self.mode}_day") or {}
        if day_state.get("day") != day:
            day_state = {"day": day, "start_equity": equity}
            self.journal.set_state(f"{self.mode}_day", day_state)

        halted = self.journal.get_state("halted")
        if halted and self._try_auto_resume(ctx, equity):
            halted, peak = None, equity
            day_state = {"day": day, "start_equity": equity}
            self.journal.set_state(f"{self.mode}_day", day_state)
        if halted:
            ctx.allow_entries = False
            if self.journal.open_trades(self.mode):
                ctx.flatten = f"halted: {halted}"
            return f"HALTED ({halted}) equity {equity:.2f}"

        drawdown = 1 - equity / peak if peak > 0 else 0.0
        if drawdown >= r["kill_switch_drawdown"]:
            reason = f"kill-switch: drawdown {drawdown:.1%} from peak {peak:.2f}"
            self.journal.set_state("halted", reason)
            self.journal.log(self.name, reason + " - flattening and halting. Run `python -m trader resume` after review.", "error")
            ctx.allow_entries = False
            ctx.flatten = reason
            return reason

        if equity >= self.cfg.target_equity:
            msg = f"target reached: equity {equity:.2f} >= {self.cfg.target_equity:.2f}"
            if not self.journal.get_state("goal_reached"):
                self.journal.set_state("goal_reached", {"ts": ctx.now, "equity": equity})
                self.journal.log(self.name, msg, "info")
            if self.cfg.on_target == "stop":
                ctx.allow_entries = False
                ctx.flatten = msg
                ctx.stop_loop = True
                return msg

        day_loss = 1 - equity / day_state["start_equity"] if day_state["start_equity"] > 0 else 0.0
        if day_loss >= r["daily_loss_limit"]:
            ctx.allow_entries = False
            return f"daily loss limit hit ({day_loss:.1%}); no new entries until 00:00 UTC | equity {equity:.2f}"
        if len(self.journal.open_trades(self.mode)) >= r["max_open_positions"]:
            ctx.allow_entries = False
        return f"equity {equity:.2f} cash {cash:.2f} peak {peak:.2f} dd {drawdown:.1%} day {-day_loss:+.1%}"
