"""Go-live gate from the plan: live trading unlocks only after the paper record proves an edge."""
from __future__ import annotations

from .config import Config
from .journal import Journal
from .logic import trade_stats


def check_gate(cfg: Config, journal: Journal) -> tuple[bool, dict, list[str]]:
    trades = sorted(journal.closed_trades("paper"), key=lambda t: t["exit_ts"])
    stats = trade_stats([t["pnl"] for t in trades], float(cfg.paper_starting_balance))
    g = cfg.go_live_gate
    problems = []
    if stats["trades"] < g["min_trades"]:
        problems.append(f"{stats['trades']} paper trades < {g['min_trades']} required")
    if stats["profit_factor"] < g["min_profit_factor"]:
        problems.append(f"profit factor {stats['profit_factor']:.2f} < {g['min_profit_factor']}")
    if stats["max_drawdown"] > g["max_drawdown"]:
        problems.append(f"max drawdown {stats['max_drawdown']:.1%} > {g['max_drawdown']:.0%}")
    return not problems, stats, problems
