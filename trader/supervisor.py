"""Heartbeat supervisor: wakes every `heartbeat_seconds` and runs the agent pipeline in order."""
from __future__ import annotations

import signal
import threading
import time

from .agents.base import Context
from .agents.execution import ExecutionAgent
from .agents.intel import IntelAgent
from .agents.learner import LearnerAgent
from .agents.llm import LLMAgent
from .agents.market import MarketDataAgent
from .agents.mood import MoodAgent
from .agents.news import NewsAgent
from .agents.risk import RiskAgent
from .agents.strategist import StrategistAgent
from .config import Config
from .journal import Journal


class Supervisor:
    def __init__(self, cfg: Config, journal: Journal, broker, exchange):
        self.cfg, self.journal, self.broker = cfg, journal, broker
        self.mode = broker.mode
        self.market = MarketDataAgent(cfg, journal, exchange)
        self.intel_agents = [NewsAgent(cfg, journal), MoodAgent(cfg, journal)]
        if cfg.llm["enabled"]:
            self.intel_agents.append(LLMAgent(cfg, journal))
        self.intel_agents.append(IntelAgent(cfg, journal))
        self.risk = RiskAgent(cfg, journal, broker)
        self.strategist = StrategistAgent(cfg, journal, self.mode)
        self.execution = ExecutionAgent(cfg, journal, broker)
        self.learner = LearnerAgent(cfg, journal, self.mode)
        self._stop = threading.Event()
        self.beats = 0

    def beat(self) -> Context:
        ctx = Context(now=time.time())
        health = self.journal.get_state("agent_health", {}) or {}
        if self.market.tick(ctx, health):
            for a in self.intel_agents:
                a.tick(ctx, health)
            if self.risk.tick(ctx, health):
                self.execution.ctx_now = ctx.now
                exits = self.execution.manage_exits(ctx)
                ctx.exits_done = True
                self.learner.tick(ctx, health)
                self.strategist.tick(ctx, health)
                self.execution.tick(ctx, health)
                if exits and health.get("execution", {}).get("note"):
                    health["execution"]["note"] = "; ".join(exits + [health["execution"]["note"]])
            else:
                self.learner.tick(ctx, health)
        else:
            self.journal.log("supervisor", "market data unavailable - skipping trading this beat", "warn")
        self.beats += 1
        self.journal.set_state("agent_health", health)
        self.journal.set_state("heartbeat", {"ts": ctx.now, "beats": self.beats, "mode": self.mode,
                                             "interval": self.cfg.heartbeat_seconds, "equity": ctx.equity,
                                             "prices": ctx.prices})
        if self.beats == 1 or self.beats % 10 == 0 or any(ctx.new_bar.values()):
            self.journal.log("heartbeat", f"beat #{self.beats} equity {ctx.equity:.4f} "
                                          f"open {len(self.journal.open_trades(self.mode))} "
                                          f"regimes {ctx.regimes or '-'}")
        if ctx.stop_loop:
            self.journal.log("supervisor", "capital-ladder target reached - stopping. Congratulations.")
            self._stop.set()
        return ctx

    def stop(self, *_):
        self._stop.set()

    def run_forever(self, max_beats: int | None = None) -> None:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.stop)
            signal.signal(signal.SIGTERM, self.stop)
        hb = float(self.cfg.heartbeat_seconds)
        self.journal.log("supervisor", f"starting {self.mode} trading on {self.cfg.exchange} {self.cfg.symbols} "
                                       f"{self.cfg.timeframe}, heartbeat {hb:.0f}s, target ${self.cfg.target_equity}")
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.beat()
            except Exception as e:
                self.journal.log("supervisor", f"beat failed: {type(e).__name__}: {e}", "error")
            if max_beats and self.beats >= max_beats:
                break
            elapsed = time.time() - t0
            if elapsed > hb:
                self.journal.log("supervisor", f"beat took {elapsed:.1f}s (> heartbeat {hb:.0f}s)", "warn")
            self._stop.wait(max(1.0, hb - elapsed))
        self.journal.log("supervisor", "stopped")
