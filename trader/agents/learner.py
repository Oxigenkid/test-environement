"""Learner agent: runs after execution every heartbeat and turns failures into improvements.

- diagnoses every newly closed losing trade (post-mortem -> `lessons` table)
- benches symbol+strategy pairs on a losing streak (cool-down)
- re-tunes parameters in a background thread after a losing streak, a kill-switch, or every N trades
- after a kill-switch, schedules an automatic resume once re-tuning is done (paper by default)
"""
from __future__ import annotations

import json
import threading
import time

from ..learning import (ADAPTIVE_KEY, adapt_on_loss, current_params, diagnose, effective_config, format_tuning,
                        load_tuning_data, losing_streak, tune, update_cooldown)
from .base import Agent, Context


class LearnerAgent(Agent):
    name = "learner"

    def __init__(self, cfg, journal, mode: str, loader=None, background: bool = True):
        super().__init__(cfg, journal)
        self.mode = mode
        self.loader = loader or load_tuning_data
        self.background = background
        self._thread: threading.Thread | None = None
        journal.set_state("tuning_running", False)  # a crash mid-tune must not block resumes forever

    # ------------------------------------------------------------ helpers
    def can_auto_resume(self) -> bool:
        lc = self.cfg.learning
        return bool(lc["auto_resume"]) and (self.mode == "paper" or bool(lc["auto_resume_live"]))

    @property
    def tuning(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _post_mortems(self, ctx: Context) -> tuple[list[str], int]:
        last_id = self.journal.get_state(f"learner_last_trade_{self.mode}")
        if last_id is None:  # first run on an existing journal: don't re-judge old history
            rows = self.journal.query("SELECT MAX(id) AS m FROM trades WHERE mode=?", (self.mode,))
            self.journal.set_state(f"learner_last_trade_{self.mode}", rows[0]["m"] or 0)
            return [], 0
        new = self.journal.query("SELECT * FROM trades WHERE mode=? AND exit_ts IS NOT NULL AND id>? ORDER BY id",
                                 (self.mode, last_id))
        if not new:
            return [], 0
        closed = self.journal.closed_trades(self.mode, limit=200)
        notes, losses = [], 0
        lc = self.cfg.learning
        for t in new:
            if (t["pnl"] or 0) >= 0:
                continue
            losses += 1
            d = diagnose(t)
            self.journal.execute(
                "INSERT INTO lessons(ts,mode,trade_id,symbol,strategy,regime,kind,lesson,pnl,loss_pct,mfe) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (ctx.now, self.mode, t["id"], t["symbol"], t["strategy"], t["regime"], d["kind"], d["lesson"],
                 t["pnl"], d["loss_pct"], d["mfe"]))
            self.journal.log(self.name, f"post-mortem {t['symbol']} {t['strategy']} {t['pnl']:+.4f} "
                                        f"({d['loss_pct']:+.1%}): {d['kind']} - {d['lesson']}", "warn")
            adapted = adapt_on_loss(self.cfg, self.journal, d)
            self.journal.log(self.name, f"adapted parameters after {d['kind']}: stop {adapted['stop_atr_mult']}ATR "
                                        f"target {adapted['target_atr_mult']}ATR", "info")
            until = update_cooldown(self.journal, closed, t, ctx.now, lc)
            if until:
                self.journal.log(self.name, f"cool-down: {t['symbol']} {t['strategy']} benched until "
                                            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(until))} UTC after "
                                            f"{lc['max_consecutive_losses']}+ losses in a row", "warn")
            notes.append(f"{t['symbol']} {d['kind']}")
        self.journal.set_state(f"learner_last_trade_{self.mode}", max(t["id"] for t in new))
        return notes, losses

    def _retune_reason(self, ctx: Context, new_losses: int) -> tuple[str | None, bool]:
        """Return (reason, forced). Forced reasons ignore retune_min_hours."""
        lc = self.cfg.learning
        halted = self.journal.get_state("halted")
        if halted and self.journal.get_state("halt_handled") != halted:
            self.journal.set_state("halt_handled", halted)
            if self.can_auto_resume():
                resume_at = ctx.now + lc["resume_after_hours"] * 3600
                self.journal.set_state("halt_resume_at", resume_at)
                self.journal.log(self.name, f"kill-switch post-mortem: re-tuning now, auto-resume after "
                                            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(resume_at))} UTC", "warn")
            return f"kill-switch ({halted})", True
        closed = self.journal.closed_trades(self.mode, limit=200)
        streak = losing_streak(closed)
        if new_losses and streak >= lc["retune_after_losses"]:
            return f"losing streak of {streak}", False
        last = self.journal.get_state(f"last_tuning_{self.mode}") or {}
        n_closed = len(self.journal.closed_trades(self.mode))
        if n_closed - last.get("n_closed", 0) >= lc["retune_every_trades"]:
            return f"periodic review after {n_closed - last.get('n_closed', 0)} trades", False
        return None, False

    # ------------------------------------------------------------ re-tuning
    def retune(self, reason: str) -> dict:
        eff = effective_config(self.cfg, self.journal)
        old = current_params(eff)
        self.journal.set_state("tuning_running", True)
        prev = self.journal.get_state(f"last_tuning_{self.mode}") or {}
        self.journal.set_state(f"last_tuning_{self.mode}", {**prev, "ts": time.time(), "reason": reason})
        try:
            data, daily = self.loader(eff)
            res = tune(eff, data, daily, current=old)
            n_closed = len(self.journal.closed_trades(self.mode))
            self.journal.set_state(f"last_tuning_{self.mode}", {"ts": time.time(), "n_closed": n_closed,
                                                                 "reason": reason})
            new = res["best"]["params"] if res["adopted"] else old
            if res["adopted"]:
                self.journal.set_state(ADAPTIVE_KEY, new)
            self.journal.execute(
                "INSERT INTO tunings(ts,reason,adopted,current_score,best_score,old_params,new_params) "
                "VALUES(?,?,?,?,?,?,?)",
                (time.time(), reason, int(res["adopted"]), res["current"]["score"], res["best"]["score"],
                 json.dumps(old), json.dumps(new)))
            self.journal.set_state("last_tuning_result", {"reason": reason, "adopted": res["adopted"],
                                                          "current": res["current"], "best": res["best"]})
            verdict = "ADOPTED" if res["adopted"] else "kept current parameters"
            self.journal.log(self.name, f"re-tune ({reason}): {verdict}\n{format_tuning(res)}",
                             "warn" if res["adopted"] else "info")
            return res
        except Exception as e:
            self.journal.log(self.name, f"re-tune failed: {type(e).__name__}: {e}", "error")
            raise
        finally:
            self.journal.set_state("tuning_running", False)

    def _start_retune(self, reason: str) -> None:
        if not self.background:
            self.retune(reason)
            return

        def work():
            try:
                self.retune(reason)
            except Exception:
                pass  # already logged

        self._thread = threading.Thread(target=work, name="learner-retune", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ heartbeat
    def run(self, ctx: Context) -> str:
        lc = self.cfg.learning
        if not lc["enabled"]:
            return "disabled"
        notes, new_losses = self._post_mortems(ctx)
        reason, forced = self._retune_reason(ctx, new_losses)
        if reason and not self.tuning:
            last = self.journal.get_state(f"last_tuning_{self.mode}") or {}
            if forced or ctx.now - last.get("ts", 0) >= lc["retune_min_hours"] * 3600:
                self.journal.log(self.name, f"starting re-tune: {reason}")
                self._start_retune(reason)
                notes.append(f"re-tuning ({reason})")
        if self.tuning:
            notes.append("re-tune in progress")
        p = self.journal.get_state(ADAPTIVE_KEY)
        return "; ".join(notes) or (f"params {p}" if p else "watching (config params)")
