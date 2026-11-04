from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config import Config
from ..journal import Journal
from ..strategies import Signal


@dataclass
class Context:
    """Blackboard shared by the agents during one heartbeat."""

    now: float
    candles: dict[str, pd.DataFrame] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    new_bar: dict[str, bool] = field(default_factory=dict)
    regimes: dict[str, str] = field(default_factory=dict)
    intel: dict[str, dict[str, Any]] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    equity: float = 0.0
    cash: float = 0.0
    allow_entries: bool = True
    flatten: str | None = None  # reason to close every position this beat
    stop_loop: bool = False
    notes: list[str] = field(default_factory=list)


class Agent:
    name = "agent"
    every_seconds: float = 0  # 0 = run on every heartbeat

    def __init__(self, cfg: Config, journal: Journal):
        self.cfg = cfg
        self.journal = journal
        self._last_run = 0.0

    def due(self, now: float) -> bool:
        return self.every_seconds <= 0 or now - self._last_run >= self.every_seconds

    def run(self, ctx: Context) -> str | None:  # returns a short status note
        raise NotImplementedError

    def tick(self, ctx: Context, health: dict) -> bool:
        if not self.due(ctx.now):
            return True
        t0 = time.time()
        try:
            note = self.run(ctx)
            ok = True
        except Exception as e:  # an agent failure must never crash the heartbeat
            note = f"{type(e).__name__}: {e}"
            ok = False
            self.journal.log(self.name, note, "error")
        self._last_run = ctx.now
        health[self.name] = {"last_run": ctx.now, "ok": ok, "ms": int((time.time() - t0) * 1000), "note": note or ""}
        return ok
